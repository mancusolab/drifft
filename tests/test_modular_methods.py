"""Composition contracts checked against independent population-space references."""

# pattern: Functional Core
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from _reference import dense_history_derivatives, dense_transition

import drifft

jax.config.update("jax_enable_x64", True)


def test_modular_selected_history_and_diagnostics():
    """Preserve selected chronology, immigration and per-interval diagnostics."""
    model = drifft.DemographicTransition(
        drifft.Demography([8, 5, 9], [7, 3]),
        backend=drifft.Jacobi,
        boundary_backend=drifft.FFT,
        method=drifft.RationalKrylov(dimension=32),
        frequency_map=drifft.Underdominance(0.04),
    )
    initial = jnp.linspace(0.01, 0.03, 15)
    expected = initial
    for source, destination, count in [(8, 5, 7), (5, 9, 3)]:
        for _ in range(count - 1):
            expected = expected @ dense_transition(source, source, 0.04)[1:-1, 1:-1]
            expected = expected.at[0].add(2 * source * 1e-5)
        expected = expected @ dense_transition(source, destination, 0.04)[1:-1, 1:-1]
        expected = expected.at[0].add(2 * destination * 1e-5)
    result = model.advance_segregating(initial, mutation_rate=1e-5)
    assert result.converged
    assert result.status == drifft.EvolutionStatus.successful
    assert len(result.diagnostics) == 2
    assert result.diagnostics[0].source_ne == 8
    assert jnp.allclose(result.distribution, expected, atol=2e-10, rtol=0)


def test_method_reuses_settings_across_population_sizes():
    """Reuse a geometry-free method with independently prepared populations."""
    method = drifft.RationalKrylov(dimension=24)
    for copies in (8, 16):
        result = method.advance(
            drifft.Jacobi(copies),
            jnp.eye(copies + 1)[2],
            frequency_map=drifft.Neutral(),
            generations=13,
        )
        expected = jnp.linalg.matrix_power(
            dense_transition(copies // 2, copies // 2, 0), 13
        )[2]
        assert jnp.allclose(result.distribution, expected, atol=2e-10)


@pytest.mark.parametrize(
    "history", [drifft.Demography([8], []), drifft.Demography([8, 5], [1])]
)
def test_jacobi_default_boundary_and_reduced_map_validation_without_holds(history):
    """Reject invalid combinations even when no hold would execute."""
    model = drifft.DemographicTransition(
        history,
        backend=drifft.Jacobi,
        method=drifft.RationalKrylov(),
        frequency_map=drifft.Neutral(),
    )
    initial = jnp.eye(17)[1]
    expected = initial if not history.generations else dense_transition(8, 5, 0)[1]
    assert jnp.allclose(model.advance(initial).distribution, expected, atol=2e-12)
    with pytest.raises(TypeError, match="diploid selection"):
        drifft.DemographicTransition(
            history,
            backend=drifft.Jacobi,
            boundary_backend=drifft.FFT,
            method=drifft.RationalKrylov(),
            frequency_map=drifft.FrequencyMap(lambda x: x),
        )


def test_failed_power_status_survives_demography_and_batching():
    """Retain unsuccessful hold status after a finite boundary transition."""
    model = drifft.DemographicTransition(
        drifft.Demography([40, 40], [100]),
        backend=drifft.Jacobi,
        boundary_backend=drifft.FFT,
        method=drifft.RationalKrylov(dimension=2),
        frequency_map=drifft.Neutral(),
    )
    initial = jnp.eye(79)[0]
    result = eqx.filter_jit(
        eqx.filter_vmap(lambda p: model.advance_segregating(p, throw=False))
    )(jnp.stack([initial, initial * 2]))
    assert not jnp.any(result.converged)
    assert jnp.all(result.status == drifft.EvolutionStatus.unconverged)
    assert jnp.all(~result.diagnostics[0].converged)


def test_identity_history_reports_nonfinite_mass():
    """Check identity inputs despite having no interval diagnostics."""
    model = drifft.DemographicTransition(
        drifft.Demography([8], []), backend=drifft.FFT, frequency_map=drifft.Neutral()
    )
    result = model.advance(jnp.full(17, jnp.nan), throw=False)
    assert not result.converged
    assert result.status == drifft.EvolutionStatus.nonfinite
    assert result.diagnostics == ()


def test_inner_shifted_solve_failure_is_reported():
    """A singular resolvent must not be mistaken for happy Arnoldi breakdown."""
    import lineax as lx

    from drifft._evolution.problem import _PowerProblem

    problem = _PowerProblem(
        lx.MatrixLinearOperator(jnp.diag(jnp.array([9.0, 1.0]))),
        jnp.ones(2),
        1,
        lambda value: value,
        jnp.zeros(2),
        False,
        jnp.asarray(1.0),
    )
    result = drifft.RationalKrylov(dimension=2).solve(problem, throw=False)
    assert not result.converged
    assert result.status == drifft.EvolutionStatus.linear_solve_failed


@pytest.mark.parametrize("generations", [[], [1]])
def test_jacobi_domain_validation_survives_no_hold_histories(generations):
    """Selection outside prepared geometry remains invalid without a hold action."""
    sizes = [8] if not generations else [8, 5]
    model = drifft.DemographicTransition(
        drifft.Demography(sizes, generations),
        backend=drifft.Jacobi,
        boundary_backend=drifft.FFT,
        method=drifft.RationalKrylov(),
        frequency_map=drifft.Underdominance(0.2),
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="selection strength"):
        model.advance(jnp.ones(17), throw=False).distribution.block_until_ready()


def test_method_configuration_has_no_geometry_arrays():
    """Method instances contain algorithm settings and can be reused across sizes."""
    for method in (drifft.Sequential(), drifft.RationalKrylov()):
        assert isinstance(method, drifft.AbstractEpochMethod)
        assert not any(eqx.is_array(leaf) for leaf in jax.tree.leaves(method))


def test_composed_jacobi_history_gradients_match_dense_recurrence():
    """Differentiate selection, initial mass and immigration through size changes."""
    model = drifft.DemographicTransition(
        drifft.Demography([4, 3, 5], [4, 3]),
        backend=drifft.Jacobi,
        boundary_backend=drifft.FFT,
        method=drifft.RationalKrylov(dimension=16),
        frequency_map=drifft.Underdominance(0.03),
    )
    initial = jnp.linspace(0.01, 0.02, 7)
    values = jnp.sin(jnp.arange(9))

    def objective(strength, mass, rate):
        updated = eqx.tree_at(lambda m: m.frequency_map.strength, model, strength)
        return (
            updated.advance_segregating(mass, mutation_rate=rate).distribution @ values
        )

    inputs = (jnp.asarray(0.03), initial, jnp.asarray(1e-5))
    actual = eqx.filter_jit(jax.grad(objective, argnums=(0, 1, 2)))(*inputs)
    direction = jax.tree.map(jnp.ones_like, inputs)
    expected, reference_tangent = dense_history_derivatives(
        inputs, direction, ((4, 3, 4), (3, 5, 3)), values
    )
    for observed, target in zip(actual, expected, strict=True):
        assert jnp.allclose(observed, target, rtol=2e-8, atol=2e-8)
    tangent = jax.jvp(objective, inputs, direction)[1]
    assert jnp.allclose(tangent, reference_tangent, rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize("history", [([4], []), ([4, 4], [1]), ([4, 4], [3])])
def test_custom_population_method_uses_same_representation_in_demography(history):
    """Honor method requirements for identity, boundary-only, and hold histories."""
    from typing import ClassVar, final

    from drifft._evolution.methods import _advance

    @final
    class PopulationMethod(drifft.AbstractEpochMethod):
        """Exercise the public extension contract without inheriting Sequential."""

        uses_reduced_representation: ClassVar[bool] = False

        def validate(self, backend, frequency_map):
            """Accept the population-coordinate backend/map combinations."""
            drifft.Sequential().validate(backend, frequency_map)

        def solve(self, problem, *, throw=True):
            """Evaluate the prepared problem by repeated operator applications."""
            return drifft.Sequential().solve(problem, throw=throw)

        def advance(self, backend, distribution, **kwargs):
            """Share preparation with the built-in methods."""
            return _advance(self, backend, distribution, **kwargs)

    frequency_map = drifft.FrequencyMap(lambda x: x**2)
    initial = jnp.eye(9)[1]
    backend = drifft.Jacobi(8)
    method = PopulationMethod()
    direct = method.advance(
        backend, initial, frequency_map=frequency_map, generations=sum(history[1])
    )
    model = drifft.DemographicTransition(
        drifft.Demography(*history),
        backend=drifft.Jacobi,
        method=method,
        frequency_map=frequency_map,
    )
    result = model.advance(initial)
    assert result.converged
    assert jnp.allclose(result.distribution, direct.distribution, atol=1e-12)
