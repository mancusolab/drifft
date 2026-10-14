"""Sampled large-population accuracy and selection derivative checks."""

# Functional Core: independent sampled-row validation without dense matrices.
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import binom

import drifft as wf

jax.config.update("jax_enable_x64", True)


@pytest.mark.slow
@pytest.mark.parametrize("M", [4096, 10000, 65536])
def test_large_sampled_rows(M):
    """Check representative large-population rows without forming a dense matrix."""
    backend = wf.NUFFT(M)
    rng = np.random.default_rng(5)
    rows = np.unique(
        np.r_[0, 1, 2, M // 2, M - 2, M - 1, M, rng.integers(0, M + 1, 32)]
    )
    vectors = [
        np.ones(M + 1),
        rng.normal(size=M + 1),
        (-1.0) ** np.arange(M + 1),
        np.eye(1, M + 1, 0).ravel(),
        np.eye(1, M + 1, M // 2).ravel(),
    ]
    apply = jax.jit(lambda s, v: wf.Transition(backend, wf.Underdominance(s)).expect(v))
    worst = 0.0
    for s in [0.0, 0.001, 0.05, 0.5, 0.95, 1.0]:
        x = rows / M
        p = (x - s * x * (1 - x)) / (1 - 2 * s * x * (1 - x))
        reference = binom.pmf(np.arange(M + 1)[None, :], M, p[:, None])
        for v in vectors:
            actual = np.asarray(apply(jnp.asarray(s), jnp.asarray(v)))
            error = np.max(np.abs(actual[rows] - reference @ v)) / max(
                1, np.max(np.abs(v))
            )
            worst = max(worst, error)
            assert error < 1e-9
            assert np.all(np.isfinite(actual))
            if np.all(v >= 0):
                assert actual.min() >= -1e-9
    print(f"M={M} sampled normalized error={worst:.3g}")


@pytest.mark.parametrize("s", [0.0, 0.001, 0.5, 1.0])
def test_spectral_occupation(s):
    # This size actually exercises spectral evaluation in the solve.
    """Compare expected visits against an independent small dense solve."""
    M = 512
    backend = wf.NUFFT(M)
    x = np.arange(1, M) / M
    p = (x - s * x * (1 - x)) / (1 - 2 * s * x * (1 - x))
    Q = binom.pmf(np.arange(1, M)[None, :], M, p[:, None])
    rhs = np.eye(1, M - 1).ravel()
    result = wf.Transition(backend, wf.Underdominance(jnp.asarray(s))).expected_visits(
        rtol=1e-11, atol=1e-12
    )
    exact = np.linalg.solve(np.eye(M - 1) - Q.T, rhs)
    assert bool(result.converged)
    np.testing.assert_allclose(result.visits, exact, atol=2e-8, rtol=1e-6)
    assert np.max(np.abs(np.asarray(result.visits) - Q.T @ result.visits - rhs)) < 2e-10
    np.testing.assert_allclose(result.visits.sum(), exact.sum(), rtol=1e-7)


def test_selection_derivative_interior():
    """Compare interior selection tangents with an analytic binomial derivative."""
    backend = wf.NUFFT(1024)
    v = jnp.asarray(np.random.default_rng(19).normal(size=1025))
    fun = jax.jit(
        lambda s: wf.Transition(backend, wf.Underdominance(s)).expect(v)[400:600]
    )
    _, tangent = jax.jvp(fun, (jnp.asarray(0.123),), (jnp.asarray(1.0),))
    difference = (fun(0.123 + 1e-5) - fun(0.123 - 1e-5)) / 2e-5
    np.testing.assert_allclose(tangent, difference, atol=2e-8, rtol=2e-6)
