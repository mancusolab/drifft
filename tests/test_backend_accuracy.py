"""Operator tolerance planning and independently measured row errors."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import binom

import drifft

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "backend_type", [drifft.Dense, drifft.FFT, drifft.NUFFT, drifft.Jacobi]
)
@pytest.mark.parametrize("atol", [0, -1, True, float("nan"), float("inf")])
def test_invalid_absolute_tolerances(backend_type, atol):
    """Reject meaningless accuracy requests before allocating geometry."""
    with pytest.raises(ValueError, match="atol"):
        backend_type(8, atol=atol)


@pytest.mark.parametrize(
    "backend_type", [drifft.Dense, drifft.FFT, drifft.NUFFT, drifft.Jacobi]
)
def test_unbounded_components_never_become_a_total_certificate(backend_type):
    """A roundoff allowance cannot be reported as an established bound."""
    report = backend_type(8, atol=1e-8).error_report
    assert report.atol == 1e-8
    assert report.bound is None
    assert report.grad_bound is None
    assert any(part.bound is None for part in report.components)


@pytest.mark.parametrize("rank", [None, 3])
def test_jacobi_sampled_row_certificate(rank):
    """Include quadrature and sampled tails, including a manual rank override."""
    backend = drifft.Jacobi(64, atol=1e-6, rank=rank)
    q = jnp.array([0, 1e-9, 1 / 128, 0.1, 0.37, 0.5, 0.91, 1 - 1e-9, 1])
    rows = jax.jit(jax.vmap(lambda v: backend.expect(q, v)))(jnp.eye(129)).T
    reference = binom.pmf(np.arange(129), 128, np.asarray(q)[:, None])
    bound = backend.spectral_bound + backend.quadrature_bound + backend.sampling_bound
    assert np.max(np.sum(np.abs(rows - reference), axis=1)) <= bound + 2e-11
    assert backend.quadrature_bound <= backend.atol / 4
    assert backend.sampling_bound <= backend.atol / 4
    if rank is None:
        assert backend.spectral_bound <= backend.atol / 4
    else:
        assert backend.rank == rank
        assert backend.spectral_bound > backend.atol


def test_fft_target_tightens_discarded_mass_without_claiming_interpolation_bound():
    """Expose the missing interpolation certificate rather than calling tails total."""
    backend = drifft.FFT(8, atol=1e-10, tail_tol=1e-5)
    assert backend.tail_tol <= backend.atol / 4
    assert backend.error_report.bound is None


@pytest.mark.parametrize("backend_type", [drifft.NUFFT, drifft.Jacobi, drifft.Dense])
@pytest.mark.parametrize(
    "options",
    [
        {"grad_atol": 0, "angular_sensitivity": 1},
        {"grad_atol": True, "angular_sensitivity": 1},
        {"grad_atol": 1e-6},
        {"angular_sensitivity": 1},
        {"grad_atol": 1e-6, "angular_sensitivity": float("inf")},
        {"grad_atol": 1e-6, "angular_sensitivity": -1},
    ],
)
def test_gradient_planning_requires_a_valid_parameter_domain(backend_type, options):
    """A scalar gradient target must not silently assume a derivative coordinate."""
    with pytest.raises(ValueError):
        backend_type(8, **options)


def test_fft_rejects_unsupported_gradient_planning():
    """Unavailable derivative certificates cannot silently become heuristics."""
    with pytest.raises(NotImplementedError, match="grad_atol"):
        drifft.FFT(8, grad_atol=1e-6, angular_sensitivity=1)


def test_jacobi_quadrature_is_planned_independently_of_manual_rank():
    """A low rank cannot fix coefficient aliasing by assuming rank+32 nodes."""
    backend = drifft.Jacobi(2048, rank=3, atol=1e-8)
    # Only half of the symmetric quadrature nodes are stored.
    assert backend._geometry.weights.size > (backend.rank + 33) // 2
    assert backend.quadrature_bound <= backend.atol / 4
    assert backend.sampling_bound <= backend.atol / 4
    assert backend.spectral_bound > backend.atol


@pytest.mark.parametrize("encoding", ["lobatto", "filtered"])
def test_joint_planning_exact_fallback_and_zero_sensitivity(encoding):
    """Exact degree removes analytic spectral error, not floating-point error."""
    backend = drifft.NUFFT(8, encoding=encoding, grad_atol=1e-12, angular_sensitivity=1)
    assert backend.degree == 16
    assert backend.spectral_grad_bound == 0
    assert backend.sampling_grad_bound == 0
    assert backend.error_report.grad_bound is None
    base = drifft.NUFFT(128, encoding=encoding, atol=1e-3)
    constant = drifft.NUFFT(
        128, encoding=encoding, atol=1e-3, grad_atol=1e-12, angular_sensitivity=0
    )
    assert constant.degree == base.degree
    assert constant.spectral_grad_bound == 0


@pytest.mark.parametrize("kind", ["lobatto", "filtered", "jacobi"])
def test_parameter_derivative_rows_against_binomial_identity(kind):
    """Bound full derivative row L1 errors, including absorbing endpoint rows."""
    size = 128
    options = dict(atol=1e-3, grad_atol=1e-5, angular_sensitivity=1.0)
    backend = (
        drifft.Jacobi(size // 2, atol=1e-3, grad_atol=1e-5, angular_sensitivity=1.0)
        if kind == "jacobi"
        else drifft.NUFFT(
            size // 2, encoding=kind, atol=1e-3, grad_atol=1e-5, angular_sensitivity=1.0
        )
    )
    p = jnp.array([0, 1e-8, 1 / size, 0.07, 0.3, 0.5, 0.85, 1 - 1e-8, 1])

    def probabilities(s):
        # Additive selection has angular sensitivity <= 1 for |s| <= 0.1.
        return (1 + s) * p / (1 + s * p)

    def rows(s):
        return jax.vmap(lambda v: backend.expect(probabilities(s), v))(
            jnp.eye(size + 1)
        ).T

    actual, derivative = jax.jit(lambda s: jax.jvp(rows, (s,), (jnp.ones_like(s),)))(
        jnp.asarray(0.04)
    )
    q = np.asarray(probabilities(0.04))
    counts = np.arange(size + 1)
    reference = binom.pmf(counts, size, q[:, None])
    dq = np.asarray(p * (1 - p) / (1 + 0.04 * p) ** 2)
    expected = (
        size
        * (
            binom.pmf(counts - 1, size - 1, q[:, None])
            - binom.pmf(counts, size - 1, q[:, None])
        )
        * dq[:, None]
    )
    report = backend.error_report
    value_bound = sum(c.bound for c in report.components if c.bound is not None)
    grad_bound = sum(
        c.grad_bound for c in report.components if c.grad_bound is not None
    )
    assert value_bound <= options["atol"]
    assert grad_bound <= options["grad_atol"]
    assert np.max(np.abs(actual - reference).sum(axis=1)) <= value_bound + 3e-11
    assert np.max(np.abs(derivative - expected).sum(axis=1)) <= grad_bound + 3e-10
    # A separate reverse-mode contraction checks the same derivative target.
    mass = jnp.linspace(-1, 1, p.size)
    observable = jnp.cos(jnp.arange(size + 1))
    reverse = jax.jit(
        jax.grad(lambda s: backend.advance(probabilities(s), mass) @ observable)
    )(0.04)
    np.testing.assert_allclose(
        reverse,
        mass @ expected @ observable,
        atol=grad_bound * np.abs(mass).sum() + 3e-10,
        rtol=0,
    )
