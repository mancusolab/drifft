"""Matrix-free rational Krylov with independently checked shifted solves."""

# pattern: Functional Core
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest
from test_demography import dense_transition

import drifft
from drifft._epochs import _PowerProblem

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("backend_type", [drifft.FFT, drifft.NUFFT])
@pytest.mark.parametrize("segregating", [False, True])
def test_gmres_rational_matches_dense(backend_type, segregating):
    """Match selected dense recurrence including immigration and signed inputs."""
    n = 11 if segregating else 13
    mass = jnp.linspace(-0.01, 0.03, n)
    source = jnp.zeros(n).at[1].set(1e-3)
    matrix = dense_transition(6, 6, 0.3)
    if segregating:
        matrix = matrix[1:-1, 1:-1]
    expected = mass
    for _ in range(25):
        expected = matrix.T @ expected + source
    result = drifft.RationalKrylov(dimension=20).advance(
        backend_type(12),
        mass,
        frequency_map=drifft.Underdominance(0.3),
        generations=25,
        segregating=segregating,
        immigration=source,
    )
    assert result.converged
    assert jnp.allclose(result.distribution, expected, atol=2e-10, rtol=2e-9)


@pytest.mark.parametrize("backend_type", [drifft.FFT, drifft.NUFFT])
@pytest.mark.parametrize("generations", [0, 1, 15])
def test_gmres_zero_mass_and_immigration(backend_type, generations):
    """Zero right-hand sides and source-only histories remain valid."""
    backend = backend_type(12)
    for source in (None, jnp.eye(11)[0] * 1e-3):
        kwargs: dict[str, Any] = dict(
            frequency_map=drifft.Neutral(),
            generations=generations,
            segregating=True,
            immigration=source,
        )
        mass = jnp.zeros(11)
        reference = drifft.Sequential().advance(backend, mass, **kwargs)
        result = drifft.RationalKrylov(dimension=16).advance(backend, mass, **kwargs)
        assert result.converged
        assert jnp.allclose(result.distribution, reference.distribution, atol=1e-11)


@pytest.mark.parametrize("backend_type", [drifft.FFT, drifft.NUFFT])
def test_gmres_composed_demography_gradients(backend_type):
    """Check selection, mass and source derivatives through implicit GMRES solves."""
    model = drifft.DemographicTransition(
        drifft.Demography([4, 3, 5], [4, 3]),
        backend=backend_type,
        method=drifft.RationalKrylov(dimension=16),
        frequency_map=drifft.Underdominance(0.03),
    )
    mass = jnp.linspace(0.01, 0.03, 7)
    weights = jnp.sin(jnp.arange(9))

    def objective(s, p, rate):
        updated = eqx.tree_at(lambda m: m.frequency_map.strength, model, s)
        return updated.advance_segregating(p, mutation_rate=rate).distribution @ weights

    def reference(s, p, rate):
        for a, b, g in ((4, 3, 4), (3, 5, 3)):
            for _ in range(g - 1):
                p = (p @ dense_transition(a, a, s)[1:-1, 1:-1]).at[0].add(2 * a * rate)
            p = (p @ dense_transition(a, b, s)[1:-1, 1:-1]).at[0].add(2 * b * rate)
        return p @ weights

    args = (jnp.asarray(0.03), mass, jnp.asarray(1e-5))
    actual = eqx.filter_jit(jax.grad(objective, argnums=(0, 1, 2)))(*args)
    expected = jax.grad(reference, argnums=(0, 1, 2))(*args)
    for a, b in zip(actual, expected, strict=True):
        assert jnp.allclose(a, b, rtol=2e-7, atol=2e-8)
    tangent = jax.tree.map(jnp.ones_like, args)
    assert jnp.allclose(
        jax.jvp(objective, args, tangent)[1],
        jax.jvp(reference, args, tangent)[1],
        atol=2e-8,
        rtol=2e-7,
    )
    batched = eqx.filter_jit(
        jax.vmap(lambda p: model.advance_segregating(p).distribution)
    )(jnp.stack((mass, 2 * mass)))
    assert jnp.allclose(batched[1], 2 * batched[0], atol=1e-10)


def test_failed_gmres_budget_and_residual():
    """A finite underresolved shifted solve must retain an explicit failure flag."""
    from drifft._krylov import _gmres_power_action

    diagonal = jnp.linspace(0.01, 0.999, 40)
    operator = lx.FunctionLinearOperator(
        lambda x: diagonal * x, jax.ShapeDtypeStruct((40,), jnp.float64)
    )
    problem = _PowerProblem(
        operator, jnp.ones(40), 100, lambda x: x, jnp.zeros(40), False, jnp.asarray(1.0)
    )
    action, stats = _gmres_power_action(problem, 8, max_steps=1, restart=2)
    assert not action[-1]
    assert stats["num_steps"] > 0
    assert jnp.isfinite(stats["max_residual_norm"])


def test_singular_matrix_free_shift_reports_failure():
    """Singular implicit shifted systems do not silently enter happy breakdown."""
    operator = lx.FunctionLinearOperator(
        lambda x: 9 * x, jax.ShapeDtypeStruct((3,), jnp.float64)
    )
    problem = _PowerProblem(
        operator, jnp.ones(3), 1, lambda x: x, jnp.zeros(3), False, jnp.asarray(1.0)
    )
    result = drifft.RationalKrylov(dimension=3).solve(problem, throw=False)
    assert not result.converged
    assert result.status == drifft.EvolutionStatus.linear_solve_failed


@pytest.mark.parametrize("backend_type", [drifft.FFT, drifft.NUFFT])
@pytest.mark.parametrize("copies", [128, 512])
def test_truncated_gmres_power_matches_sequential(backend_type, copies):
    """Exercise a truncated basis, real FFT blocks and general frequency maps."""
    backend = backend_type(copies)
    mass = jnp.zeros(copies + 1).at[copies // 3].set(1)
    frequency_map = drifft.Composition(
        drifft.Underdominance(0.2), drifft.Mutation(1e-3, 2e-3)
    )
    expected = drifft.Sequential().advance(
        backend, mass, frequency_map=frequency_map, generations=150
    )
    result = drifft.RationalKrylov(dimension=64).advance(
        backend, mass, frequency_map=frequency_map, generations=150
    )
    assert result.converged
    assert jnp.allclose(
        result.distribution, expected.distribution, atol=1e-9, rtol=1e-8
    )
