"""Epoch actions checked against independent dense binomial transitions."""

# pattern: Functional Core
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from test_demography import dense_transition

import drifft

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("kind", ["sequential", "krylov"])
@pytest.mark.parametrize("segregating", [False, True])
@pytest.mark.parametrize("generations", [0, 1, 7, 100])
def test_epoch_matches_dense_with_immigration(kind, segregating, generations):
    """Include immigration after every generation, including the final one."""
    backend = drifft.FFT(16) if kind == "sequential" else drifft.Jacobi(16)
    method = (
        drifft.Sequential()
        if kind == "sequential"
        else drifft.RationalKrylov(dimension=32)
    )
    matrix = dense_transition(8, 8, 0.07).T
    if segregating:
        matrix = matrix[1:-1, 1:-1]
    initial = jnp.arange(1, matrix.shape[0] + 1, dtype=jnp.float64)
    initial /= initial.sum()
    source = jnp.zeros_like(initial).at[0 if segregating else 1].set(0.001)
    expected = initial
    for _ in range(generations):
        expected = matrix @ expected + source
    result = method.advance(
        backend,
        initial,
        frequency_map=drifft.Underdominance(0.07),
        generations=generations,
        segregating=segregating,
        immigration=source,
    )
    assert result.converged
    assert jnp.allclose(result.distribution, expected, atol=2e-10, rtol=0)


def test_epoch_selection_derivative_and_batching():
    """Reuse geometry across traced strengths and compare derivatives to dense powers."""
    backend = drifft.Jacobi(20)
    method = drifft.RationalKrylov(dimension=32)
    initial = jnp.linspace(0.01, 0.1, 19)
    values = jnp.sin(jnp.arange(19))

    def objective(strength):
        """Evaluate a selected segregating epoch."""
        return (
            method.advance(
                backend,
                initial,
                frequency_map=drifft.Underdominance(strength),
                generations=19,
                segregating=True,
            ).distribution
            @ values
        )

    def reference(strength):
        """Use full binomial matrices without Jacobi reduction."""
        matrix = dense_transition(10, 10, strength)[1:-1, 1:-1]
        return initial @ jnp.linalg.matrix_power(matrix, 19) @ values

    strength = jnp.asarray(0.03)
    assert jnp.allclose(
        jax.grad(objective)(strength), jax.grad(reference)(strength), atol=2e-8
    )
    tangent = jax.jvp(objective, (strength,), (jnp.ones_like(strength),))[1]
    assert jnp.allclose(tangent, jax.grad(reference)(strength), atol=2e-8)
    batch = eqx.filter_jit(eqx.filter_vmap(objective))(jnp.array([0.02, 0.03]))
    assert jnp.allclose(
        batch, jnp.array([reference(0.02), reference(0.03)]), atol=2e-10
    )


def test_epoch_zero_and_absorbing_inputs():
    """Happy breakdown and zero seeds must not produce NaNs."""
    backend = drifft.Jacobi(16)
    method = drifft.RationalKrylov()
    for initial in [jnp.zeros(17), jnp.eye(17)[0], jnp.eye(17)[-1]]:
        result = method.advance(
            backend, initial, frequency_map=drifft.Neutral(), generations=100
        )
        assert result.converged
        assert jnp.allclose(result.distribution, initial, atol=1e-13)


def test_demography_uses_jacobi_method_with_singleton_input():
    """Respect source holds and destination-sized immigration on boundary steps."""
    model = drifft.DemographicTransition(
        drifft.Demography([8, 5, 9], [7, 3]),
        backend=drifft.Jacobi,
        boundary_backend=drifft.FFT,
        method=drifft.RationalKrylov(),
        frequency_map=drifft.Neutral(),
    )
    initial = jnp.linspace(0.1, 0.2, 15)
    expected = initial
    for source, destination, count in [(8, 5, 7), (5, 9, 3)]:
        for _ in range(count - 1):
            expected = expected @ dense_transition(source, source, 0)[1:-1, 1:-1]
            expected = expected.at[0].add(2 * source * 1e-5)
        expected = expected @ dense_transition(source, destination, 0)[1:-1, 1:-1]
        expected = expected.at[0].add(2 * destination * 1e-5)
    actual = model.advance_segregating(initial, mutation_rate=1e-5).distribution
    assert jnp.allclose(actual, expected, atol=2e-10, rtol=0)


def test_epoch_rejects_unsupported_maps():
    """Rational Jacobi assembly does not silently ignore recurrent mutation."""
    backend = drifft.Jacobi(16)
    method = drifft.RationalKrylov()
    with pytest.raises(TypeError, match="Neutral.*Underdominance"):
        method.advance(
            backend, jnp.ones(17), frequency_map=drifft.Mutation(1e-5), generations=2
        )


def test_truncated_krylov_matches_dense_long_epoch():
    """Exercise an actual reduced Krylov space rather than a complete basis."""
    backend = drifft.Jacobi(128)
    method = drifft.RationalKrylov(dimension=40)
    initial = jnp.eye(127)[0]
    source = initial * 1e-4
    result = method.advance(
        backend,
        initial,
        frequency_map=drifft.Underdominance(0.03),
        generations=250,
        segregating=True,
        immigration=source,
    )
    matrix = dense_transition(64, 64, 0.03)[1:-1, 1:-1].T
    expected = initial
    for _ in range(250):
        expected = matrix @ expected + source
    assert backend.rank + 1 > method.dimension
    assert result.converged
    assert jnp.sum(jnp.abs(result.distribution - expected)) < 1e-9


@pytest.mark.parametrize("generations", [100, 10000])
def test_krylov_reports_and_throws_on_underresolved_power(generations):
    """A tiny Krylov budget must expose its failed estimate to callers."""
    copies = 80 if generations == 100 else 800
    backend = drifft.Jacobi(copies)
    method = drifft.RationalKrylov(dimension=2)
    initial = jnp.zeros(copies - 1).at[0].set(1)
    result = method.advance(
        backend,
        initial,
        frequency_map=drifft.Neutral(),
        generations=generations,
        segregating=True,
        throw=False,
    )
    assert not result.converged
    assert result.error_estimate > method.atol + method.rtol * jnp.sum(
        jnp.abs(result.distribution)
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="epoch failed"):
        method.advance(
            backend,
            initial,
            frequency_map=drifft.Neutral(),
            generations=generations,
            segregating=True,
        ).distribution.block_until_ready()


@pytest.mark.parametrize(
    "options",
    [
        {"gene_copies": 3},
        {"gene_copies": True},
        {"rank": 0},
        {"rank": 16},
        {"dimension": 1},
        {"max_strength": 1},
        {"max_strength": float("nan")},
        {"one_step_tolerance": 0},
        {"rtol": -1},
        {"atol": float("inf")},
    ],
)
def test_krylov_rejects_invalid_configuration(options):
    """Reject invalid structural choices before preparing expensive geometry."""
    with pytest.raises(ValueError):
        if any(key in options for key in ("dimension", "rtol", "atol")):
            drifft.RationalKrylov(**options)
        else:
            drifft.Jacobi(**({"gene_copies": 16} | options))


@pytest.mark.parametrize("generations", [-1, 1.5, True])
def test_epoch_rejects_invalid_generation_counts(generations):
    """Reject nonintegral and negative durations before tracing the recurrence."""
    backend = drifft.Jacobi(16)
    method = drifft.RationalKrylov()
    with pytest.raises(ValueError, match="generations"):
        method.advance(
            backend,
            jnp.ones(17),
            frequency_map=drifft.Neutral(),
            generations=generations,
        )


def test_epoch_runtime_and_shape_validation():
    """Traced selection and static shape failures remain explicit."""
    backend = drifft.Jacobi(16)
    method = drifft.RationalKrylov()
    with pytest.raises(ValueError, match="distribution"):
        method.advance(
            backend, jnp.ones(16), frequency_map=drifft.Neutral(), generations=2
        )
    with pytest.raises(ValueError, match="immigration"):
        method.advance(
            backend,
            jnp.ones(17),
            frequency_map=drifft.Neutral(),
            generations=2,
            immigration=jnp.ones(16),
        )
    with pytest.raises(eqx.EquinoxRuntimeError, match="selection strength"):
        method.advance(
            backend,
            jnp.ones(17),
            frequency_map=drifft.Underdominance(0.2),
            generations=2,
        ).distribution.block_until_ready()
    result = method.advance(
        backend,
        jnp.full(17, jnp.nan),
        frequency_map=drifft.Neutral(),
        generations=2,
        throw=False,
    )
    assert not result.converged


def test_krylov_mass_and_immigration_gradients():
    """Differentiate through a truncated basis, not only its selection matrix."""
    backend = drifft.Jacobi(64)
    method = drifft.RationalKrylov(dimension=32)
    mass = jnp.linspace(0.01, 0.05, 63)
    immigration = jnp.linspace(0.0001, 0.0002, 63)
    values = jnp.cos(jnp.arange(63) / 10)
    matrix = dense_transition(32, 32, 0.02)[1:-1, 1:-1].T

    def objective(initial, source):
        """Contract the epoch output with fixed terminal values."""
        return (
            method.advance(
                backend,
                initial,
                frequency_map=drifft.Underdominance(0.02),
                generations=30,
                segregating=True,
                immigration=source,
            ).distribution
            @ values
        )

    def reference(initial, source):
        """Use a dense affine recurrence independent of the Krylov basis."""
        result = jax.lax.fori_loop(
            0, 30, lambda _, current: matrix @ current + source, initial
        )
        return result @ values

    actual = jax.grad(objective, argnums=(0, 1))(mass, immigration)
    expected = jax.grad(reference, argnums=(0, 1))(mass, immigration)
    for observed, target in zip(actual, expected, strict=True):
        assert jnp.allclose(observed, target, atol=2e-8, rtol=2e-8)


def test_demographic_full_state_krylov_and_expect_contract():
    """Forward adaptive histories agree with dense products and reject adjoints."""
    model = drifft.DemographicTransition(
        drifft.Demography([8, 5], [20]),
        backend=drifft.Jacobi,
        boundary_backend=drifft.FFT,
        method=drifft.RationalKrylov(),
        frequency_map=drifft.Underdominance(0.03),
    )
    initial = jnp.eye(17)[2]
    expected = (
        initial
        @ jnp.linalg.matrix_power(dense_transition(8, 8, 0.03), 19)
        @ dense_transition(8, 5, 0.03)
    )
    assert jnp.allclose(model.advance(initial).distribution, expected, atol=2e-10)
    with pytest.raises(ValueError, match="Sequential"):
        model.expect(jnp.ones(11))
