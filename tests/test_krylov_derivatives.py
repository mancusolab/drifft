"""Krylov sensitivities at zero seeds and Arnoldi invariant subspaces."""

# pattern: Functional Core
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import drifft

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "backend_type,method_type",
    [
        (drifft.FFT, drifft.PolynomialKrylov),
        (drifft.FFT, drifft.RationalKrylov),
        (drifft.Jacobi, drifft.RationalKrylov),
    ],
)
@pytest.mark.parametrize("throw", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("zero_seed", [False, True])
def test_breakdown_derivatives_match_sequential(
    backend_type, method_type, throw, reverse, zero_seed
):
    """Represent mass and mutation sensitivities outside the primal Krylov space."""
    backend = backend_type(12)
    method = method_type(dimension=20)
    absent = jnp.eye(13)[0]

    def action(parameter):
        mass = parameter if zero_seed else absent
        rule = drifft.Neutral() if zero_seed else drifft.Mutation(parameter, 0.0)
        return method.advance(
            backend, mass, frequency_map=rule, generations=7, throw=throw
        )

    parameter = jnp.zeros(13) if zero_seed else jnp.asarray(0.0)
    result = action(parameter)
    assert result.converged
    assert jnp.allclose(
        result.distribution, jnp.zeros(13) if zero_seed else absent, atol=1e-12
    )

    def fn(x):
        return action(x).distribution

    if reverse:
        differentiated = eqx.filter_jit(jax.grad(lambda x: fn(x).sum()))
    else:
        direction = (
            absent.at[0].set(0.0).at[3].set(1.0) if zero_seed else jnp.asarray(1.0)
        )
        differentiated = eqx.filter_jit(lambda x: jax.jvp(fn, (x,), (direction,))[1])
    reference_backend = drifft.FFT(12)

    def reference(parameter):
        mass = parameter if zero_seed else absent
        rule = drifft.Neutral() if zero_seed else drifft.Mutation(parameter, 0.0)
        return (
            drifft.Sequential()
            .advance(reference_backend, mass, frequency_map=rule, generations=7)
            .distribution
        )

    actual = differentiated(parameter)
    if reverse:
        expected = jax.grad(lambda x: reference(x).sum())(parameter)
    else:
        expected = jax.jvp(reference, (parameter,), (direction,))[1]
    assert jnp.allclose(actual, expected, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize(
    "method_type", [drifft.PolynomialKrylov, drifft.RationalKrylov]
)
def test_demographic_breakdown_sensitivities(method_type):
    """Sensitivities survive hold and rectangular boundary composition."""
    model = drifft.DemographicTransition(
        drifft.Demography([6, 4, 5], [7, 3]),
        backend=drifft.Jacobi,
        boundary_backend=drifft.FFT,
        method=method_type(dimension=20),
        frequency_map=drifft.Mutation(0.0, 0.0),
    )
    mass = jnp.eye(13)[0]
    assert model.advance(mass).converged

    def objective(u):
        updated = eqx.tree_at(lambda m: m.frequency_map.forward_rate, model, u)
        return updated.advance(mass, throw=False).distribution.sum()

    reference = drifft.DemographicTransition(
        model.demography,
        backend=drifft.FFT,
        method=drifft.Sequential(),
        frequency_map=drifft.Mutation(0.0, 0.0),
    )

    def expected(u):
        updated = eqx.tree_at(lambda m: m.frequency_map.forward_rate, reference, u)
        return updated.advance(mass).distribution.sum()

    actual = eqx.filter_jit(jax.grad(objective))(jnp.asarray(0.0))
    assert jnp.allclose(actual, jax.grad(expected)(jnp.asarray(0.0)), atol=2e-9)


@pytest.mark.parametrize(
    "backend_type,generations",
    [(drifft.FFT, 0), (drifft.NUFFT, 0), (drifft.Jacobi, 0), (drifft.Jacobi, 1)],
)
@pytest.mark.parametrize(
    "method_type", [drifft.PolynomialKrylov, drifft.RationalKrylov]
)
def test_no_power_still_differentiates_zero_mass(
    backend_type, generations, method_type
):
    """Identity and single-step encoding do not construct a Krylov basis."""
    backend = backend_type(12)
    initial = jnp.zeros(13)
    direction = jnp.eye(13)[3]
    method = method_type()

    def action(mass):
        return method.advance(
            backend, mass, frequency_map=drifft.Neutral(), generations=generations
        ).distribution

    _, tangent = jax.jvp(action, (initial,), (direction,))
    assert jnp.allclose(tangent, action(direction), atol=1e-12)


def test_batched_derivatives_include_breakdown_members():
    """Zero and nonzero mutation rates share the same derivative contract."""
    backend = drifft.FFT(12)
    method = drifft.PolynomialKrylov(dimension=20)
    mass = jnp.eye(13)[0]

    def objective(u):
        return method.advance(
            backend,
            mass,
            frequency_map=drifft.Mutation(u, 0.0),
            generations=7,
            throw=False,
        ).distribution.sum()

    rates = jnp.array([0.001, 0.0])
    assert jnp.allclose(eqx.filter_jit(jax.vmap(objective))(rates), 1.0, atol=1e-12)
    actual = eqx.filter_jit(jax.vmap(jax.grad(objective)))(rates)
    assert jnp.allclose(actual, 0.0, atol=2e-9)


def test_sequential_mutation_boundary_sensitivity_matches_dense_reference():
    """The recommended alternative represents sensitivity outside an absorbing space."""
    import numpy as np
    from scipy.stats import binom

    copies, generations = 12, 7
    x = np.arange(copies + 1) / copies
    counts = np.arange(copies + 1)[:, None]
    transition = binom.pmf(counts, copies, x[None, :])
    derivative = (
        copies
        * (
            binom.pmf(counts - 1, copies - 1, x[None, :])
            - binom.pmf(counts, copies - 1, x[None, :])
        )
        * (1 - x)[None, :]
    )
    mass = np.eye(copies + 1)[0]
    expected = np.zeros_like(mass)
    for _ in range(generations):
        expected, mass = transition @ expected + derivative @ mass, transition @ mass
    backend = drifft.FFT(copies)

    def action(u):
        return (
            drifft.Sequential()
            .advance(
                backend,
                jnp.eye(copies + 1)[0],
                frequency_map=drifft.Mutation(u, 0.0),
                generations=generations,
            )
            .distribution
        )

    _, actual = jax.jit(lambda u: jax.jvp(action, (u,), (jnp.ones_like(u),)))(
        jnp.asarray(0.0)
    )
    assert jnp.allclose(actual, expected, atol=2e-12, rtol=2e-12)
    assert jnp.abs(actual.sum()) < 1e-12


@pytest.mark.parametrize(
    "backend_type,method_type",
    [
        (drifft.FFT, drifft.PolynomialKrylov),
        (drifft.FFT, drifft.RationalKrylov),
        (drifft.Jacobi, drifft.RationalKrylov),
    ],
)
def test_zero_immigration_sensitivity(backend_type, method_type):
    """The affine auxiliary coordinate must not hide a zero physical seed."""
    backend = backend_type(12)
    method = method_type(dimension=20)
    mass = jnp.zeros(11)

    def objective(rate):
        source = mass.at[0].set(rate)
        return method.advance(
            backend,
            mass,
            frequency_map=drifft.Neutral(),
            generations=7,
            segregating=True,
            immigration=source,
            throw=False,
        ).distribution.sum()

    assert objective(jnp.asarray(0.0)) == 0
    actual = eqx.filter_jit(jax.grad(objective))(jnp.asarray(0.0))
    reference_backend = drifft.FFT(12)

    def reference(rate):
        return (
            drifft.Sequential()
            .advance(
                reference_backend,
                mass,
                frequency_map=drifft.Neutral(),
                generations=7,
                segregating=True,
                immigration=mass.at[0].set(rate),
            )
            .distribution.sum()
        )

    assert jnp.allclose(
        actual, jax.grad(reference)(jnp.asarray(0.0)), atol=2e-9, rtol=2e-8
    )
