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
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("zero_seed", [False, True])
def test_breakdown_derivatives_match_sequential(
    backend_type, method_type, reverse, zero_seed
):
    """Represent mass and mutation sensitivities outside the primal Krylov space."""
    backend = backend_type(6)
    method = method_type(dimension=20)
    absent = jnp.eye(13)[0]
    reward = jnp.sin(jnp.arange(13) * 0.3)

    def action(parameter):
        mass = parameter if zero_seed else absent
        rule = drifft.Neutral() if zero_seed else drifft.Mutation(parameter, 0.0)
        return method.advance(
            backend, mass, frequency_map=rule, generations=7, throw=False
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
        differentiated = eqx.filter_jit(jax.grad(lambda x: fn(x) @ reward))
    else:
        direction = (
            absent.at[0].set(0.0).at[3].set(1.0) if zero_seed else jnp.asarray(1.0)
        )
        differentiated = eqx.filter_jit(lambda x: jax.jvp(fn, (x,), (direction,))[1])
    reference_backend = drifft.FFT(6)

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
        expected = jax.grad(lambda x: reference(x) @ reward)(parameter)
    else:
        expected = jax.jvp(reference, (parameter,), (direction,))[1]
    assert jnp.max(jnp.abs(expected)) > 1e-3
    assert jnp.allclose(actual, expected, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize(
    "method_type", [drifft.PolynomialKrylov, drifft.RationalKrylov]
)
def test_demographic_breakdown_sensitivities(method_type):
    """Sensitivities survive hold and rectangular boundary composition."""
    model = drifft.DemographicTransition(
        drifft.Demography([6, 4, 5], [7, 3]).prepare(
            backend_factory=drifft.Jacobi, boundary_backend_factory=drifft.FFT
        ),
        method=method_type(dimension=20),
        frequency_map=drifft.Mutation(0.0, 0.0),
    )
    mass = jnp.eye(13)[0]
    reward = jnp.sin(jnp.arange(11) * 0.3)
    assert model.advance(mass).converged

    def objective(u):
        updated = eqx.tree_at(lambda m: m.frequency_map.forward_rate, model, u)
        return updated.advance(mass, throw=False).distribution @ reward

    reference = drifft.DemographicTransition(
        model.demography.prepare(backend_factory=drifft.FFT),
        method=drifft.Sequential(),
        frequency_map=drifft.Mutation(0.0, 0.0),
    )

    def expected(u):
        updated = eqx.tree_at(lambda m: m.frequency_map.forward_rate, reference, u)
        return updated.advance(mass).distribution @ reward

    actual = eqx.filter_jit(jax.grad(objective))(jnp.asarray(0.0))
    expected_gradient = jax.grad(expected)(jnp.asarray(0.0))
    assert jnp.abs(expected_gradient) > 1e-3
    assert jnp.allclose(actual, expected_gradient, atol=2e-9)


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
    backend = backend_type(6)
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
    """Mutation sensitivities remain nonzero for both zero and nonzero rates."""
    backend = drifft.FFT(6)
    method = drifft.PolynomialKrylov(dimension=20)
    mass = jnp.eye(13)[0]
    frequencies = jnp.arange(13) / 12

    def objective(u):
        return (
            method.advance(
                backend,
                mass,
                frequency_map=drifft.Mutation(u, 0.0),
                generations=7,
                throw=False,
            ).distribution
            @ frequencies
        )

    # Mutation alone obeys E[x_t] = 1 - (1-u)**t, independently of drift.
    rates = jnp.array([0.001, 0.0])
    assert jnp.allclose(
        eqx.filter_jit(jax.vmap(objective))(rates), 1 - (1 - rates) ** 7, atol=1e-12
    )
    actual = eqx.filter_jit(jax.vmap(jax.grad(objective)))(rates)
    assert jnp.allclose(actual, 7 * (1 - rates) ** 6, atol=2e-9)


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
    backend = drifft.FFT(copies // 2)

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
def test_zero_source_mass_sensitivity(backend_type, method_type):
    """The affine auxiliary coordinate must not hide a zero physical seed."""
    backend = backend_type(6)
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
            source_mass=source,
            throw=False,
        ).distribution.sum()

    assert objective(jnp.asarray(0.0)) == 0
    actual = eqx.filter_jit(jax.grad(objective))(jnp.asarray(0.0))
    reference_backend = drifft.FFT(6)

    def reference(rate):
        return (
            drifft.Sequential()
            .advance(
                reference_backend,
                mass,
                frequency_map=drifft.Neutral(),
                generations=7,
                segregating=True,
                source_mass=mass.at[0].set(rate),
            )
            .distribution.sum()
        )

    assert jnp.allclose(
        actual, jax.grad(reference)(jnp.asarray(0.0)), atol=2e-9, rtol=2e-8
    )


@pytest.mark.parametrize(
    "method_type", [drifft.PolynomialKrylov, drifft.RationalKrylov]
)
def test_absorbing_mass_preserves_extreme_scales(method_type):
    """Representable absorbing mass survives seed normalization at either extreme."""
    backend = drifft.Dense(2)
    method = method_type(dimension=4, atol=0.0)
    mass = jnp.eye(5)[0]

    def advance(scale):
        result = method.advance(
            backend,
            mass * scale,
            frequency_map=drifft.Neutral(),
            generations=2,
            throw=False,
        )
        return result.distribution / scale, result.converged

    states, converged = jax.jit(jax.vmap(advance))(jnp.array([1e-180, 1e180]))
    assert jnp.all(converged)
    assert jnp.allclose(states, mass, atol=0.0)


@pytest.mark.parametrize(
    "method_type", [drifft.PolynomialKrylov, drifft.RationalKrylov]
)
def test_jacobi_ad_of_batched_parameters(method_type):
    """Differentiate mapped operators against the direct sequential recurrence."""
    backend = drifft.Jacobi(3)
    mass = jnp.eye(5)[2]
    rates = jnp.array([0.01, 0.04])
    direction = jnp.array([0.3, -0.2])

    def batched(method, rates, mass):
        def epoch(rate):
            return method.advance(
                backend,
                mass,
                frequency_map=drifft.Mutation(rate, 0.03),
                generations=4,
                segregating=True,
            ).distribution

        return jnp.cos(jax.vmap(epoch)(rates)).sum()

    def derivatives(method):
        def fn(p, x):
            return batched(method, p, x)

        grad = jax.jit(jax.grad(fn, argnums=(0, 1)))(rates, mass)
        jvp = jax.jit(lambda p, d: jax.jvp(lambda q: fn(q, mass), (p,), (d,))[1])(
            rates, direction
        )
        return *grad, jvp

    for actual, expected in zip(
        derivatives(method_type(dimension=8)),
        derivatives(drifft.Sequential()),
        strict=True,
    ):
        assert jnp.allclose(actual, expected, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize(
    "method_type", [drifft.PolynomialKrylov, drifft.RationalKrylov]
)
def test_strict_sensitivity_accepts_scaled_directions(method_type):
    """A relative-only tangent budget follows the scale of its direction."""
    backend = drifft.Dense(2)
    mass = jnp.eye(5)[2]

    def tangent(method, direction):
        def epoch(selection):
            return method.advance(
                backend,
                mass,
                frequency_map=drifft.AdditiveSelection(selection),
                generations=5,
            ).distribution

        return jax.jvp(epoch, (jnp.asarray(0.1),), (direction,))[1]

    method = method_type(dimension=12, rtol=1e-6, atol=0, check_sensitivity=True)
    for scale in (0.0, 1e-10, 1.0):
        actual = tangent(method, jnp.asarray(scale))
        expected = tangent(drifft.Sequential(), jnp.asarray(scale))
        assert jnp.allclose(actual, expected, atol=scale * 1e-8, rtol=1e-6)


def test_reverse_rejects_overflow_in_parameter_contraction():
    """Finite adjoint factors can still overflow when contracted into parameters."""
    backend = drifft.Dense(4)
    mass = jnp.eye(9)[4]
    method = drifft.PolynomialKrylov(dimension=9)

    def objective(t):
        mapping = drifft.FrequencyMap(
            lambda x: jnp.broadcast_to(jax.nn.sigmoid(1e308 * t), x.shape)
        )
        result = method.advance(backend, mass, frequency_map=mapping, generations=2)
        return jnp.arange(9) @ result.distribution

    assert jnp.isclose(objective(0.0), 4.0)
    with pytest.raises(Exception, match="nonfinite parameter cotangent"):
        jax.block_until_ready(jax.jit(jax.grad(objective))(0.0))
