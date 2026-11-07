"""Mathematical history actions and their joint parameter/vector derivatives."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.scipy.stats import binom

import drifft

jax.config.update("jax_enable_x64", True)


def reference(parameters, vector, *, backward, segregating):
    """Compose explicit binomial matrices and post-reproduction singleton sources."""
    s, u, v, rate = parameters
    forward = jnp.eye(5 if segregating else 7)
    source = jnp.zeros(forward.shape[0])
    for start, end in [(3, 3), (3, 3), (3, 2), (2, 2), (2, 4)]:
        x = jnp.arange(2 * start + 1) / (2 * start)
        q = (x * (1 - s / 2) - s * x**2 / 2) / (1 - s * x)
        q = q * (1 - v) + (1 - q) * u
        matrix = binom.pmf(jnp.arange(2 * end + 1)[None, :], 2 * end, q[:, None])
        if segregating:
            matrix = matrix[1:-1, 1:-1]
        forward = forward @ matrix
        source = matrix.T @ source
        if segregating:
            source = source.at[0].add(2 * end * rate)
    if backward:
        return jnp.concatenate((forward @ vector, jnp.atleast_1d(source @ vector)))
    return forward.T @ vector + source


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("segregating", [False, True])
@pytest.mark.parametrize(
    "backend,method",
    [
        (drifft.FFT, drifft.Sequential()),
        (drifft.FFT, drifft.PolynomialKrylov(dimension=16)),
        (drifft.Jacobi, drifft.RationalKrylov(dimension=16)),
    ],
)
def test_history_joint_derivatives_match_dense(backend, method, backward, segregating):
    """Check both orientations including selection, mutation, source, and vector inputs."""
    geometry = drifft.Demography([3, 2, 4], [3, 2]).prepare(backend_factory=backend)
    parameters = jnp.array([0.03, 0.01, 0.02, 0.001])
    size = 2 * (4 if backward else 3) + (-1 if segregating else 1)
    vector = jnp.linspace(0.01, 0.2, size)

    def evaluate(parameters, vector):
        s, u, v, rate = parameters
        model = drifft.DemographicTransition(
            geometry,
            drifft.Composition(drifft.AdditiveSelection(s), drifft.Mutation(u, v)),
            method=method,
        )
        source_rate = rate if segregating else None
        if backward:
            result = model.expect(
                vector, segregating=segregating, singleton_mutation_rate=source_rate
            )
            return jnp.concatenate((result.values, jnp.atleast_1d(result.offset)))
        return model.advance(
            vector, segregating=segregating, singleton_mutation_rate=source_rate
        ).distribution

    def oracle(parameters, vector):
        return reference(parameters, vector, backward=backward, segregating=segregating)

    inputs = (parameters, vector)
    direction = (jnp.array([0.2, -0.1, 0.3, 0.02]), jnp.sin(vector))
    value, tangent = jax.jit(lambda p, x: jax.jvp(evaluate, (p, x), direction))(*inputs)
    expected, expected_tangent = jax.jvp(oracle, inputs, direction)
    assert jnp.allclose(value, expected, atol=2e-10, rtol=2e-9)
    assert jnp.allclose(tangent, expected_tangent, atol=2e-9, rtol=2e-8)
    weight = jnp.cos(jnp.arange(value.size))
    actual_grad = jax.jit(
        jax.grad(lambda p, x: evaluate(p, x) @ weight, argnums=(0, 1))
    )(*inputs)
    reference_grad = jax.grad(lambda p, x: oracle(p, x) @ weight, argnums=(0, 1))(
        *inputs
    )
    for actual, target in zip(actual_grad, reference_grad, strict=True):
        assert jnp.allclose(actual, target, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize(
    "method", [drifft.Sequential(), drifft.PolynomialKrylov(dimension=12)]
)
def test_batched_vector_pullback_matches_expectation(method):
    """The vector adjoint is the backward action, including at zero input mass."""
    geometry = drifft.Demography([3, 2], [4]).prepare(backend_factory=drifft.FFT)
    model = drifft.DemographicTransition(
        geometry, drifft.AdditiveSelection(0.03), method=method
    )
    rewards = jnp.stack((jnp.arange(3.0), jnp.cos(jnp.arange(3.0))))
    mass = jnp.zeros(5)

    def reward_gradient(reward):
        return jax.grad(
            lambda x: (
                model.advance(
                    x, segregating=True, singleton_mutation_rate=0.001
                ).distribution
                @ reward
            )
        )(mass)

    actual = jax.jit(jax.vmap(reward_gradient))(rewards)
    expected = jax.jit(
        jax.vmap(lambda reward: model.expect(reward, segregating=True).values)
    )(rewards)
    assert jnp.allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize("backward", [False, True])
def test_sequential_second_derivative_matches_dense(backward):
    """Higher derivatives retain process dependence in both action orientations."""
    geometry = drifft.Demography([3, 2, 4], [3, 2]).prepare(backend_factory=drifft.FFT)
    vector = jnp.linspace(0.01, 0.2, 7 if backward else 5)

    def evaluate(s):
        model = drifft.DemographicTransition(
            geometry,
            drifft.Composition(
                drifft.AdditiveSelection(s), drifft.Mutation(0.01, 0.02)
            ),
        )
        if backward:
            result = model.expect(
                vector, segregating=True, singleton_mutation_rate=0.001
            )
            return result.values.sum() + result.offset
        return model.advance(
            vector, segregating=True, singleton_mutation_rate=0.001
        ).distribution.sum()

    def oracle(s):
        return reference(
            jnp.array([s, 0.01, 0.02, 0.001]),
            vector,
            backward=backward,
            segregating=True,
        ).sum()

    actual = jax.jit(jax.grad(jax.grad(evaluate)))(0.03)
    expected = jax.grad(jax.grad(oracle))(0.03)
    assert jnp.allclose(actual, expected, atol=1e-9)


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize(
    "backend,method",
    [
        (drifft.FFT, drifft.PolynomialKrylov(dimension=12)),
        (drifft.Jacobi, drifft.RationalKrylov(dimension=12)),
    ],
)
def test_krylov_higher_parameter_derivatives_raise(backward, backend, method):
    """The demographic boundary must not bypass the epoch sensitivity restriction."""
    geometry = drifft.Demography([3, 2], [4]).prepare(backend_factory=backend)

    def action(s):
        model = drifft.DemographicTransition(
            geometry,
            drifft.AdditiveSelection(s),
            method=method,
        )
        if backward:
            return model.expect(jnp.arange(5.0)).values.sum()
        return model.advance(jnp.eye(7)[2]).distribution @ jnp.arange(5.0)

    with pytest.raises(
        (NotImplementedError, RuntimeError), match="[Hh]igher-order Krylov"
    ):
        jax.jit(jax.grad(jax.grad(action)))(0.03)


@pytest.mark.parametrize("backward", [False, True])
def test_sequential_forward_over_forward(backward):
    """Nested JVPs match dense derivatives, including unused augmented coordinates."""
    geometry = drifft.Demography([3, 2, 4], [3, 2]).prepare(backend_factory=drifft.FFT)
    vector = jnp.linspace(0.01, 0.2, 9 if backward else 7)
    weights = jnp.cos(jnp.arange(8 if backward else 9))

    def action(s):
        model = drifft.DemographicTransition(
            geometry,
            drifft.Composition(
                drifft.AdditiveSelection(s), drifft.Mutation(0.01, 0.02)
            ),
        )
        if backward:
            result = model.expect(vector)
            return (
                jnp.concatenate((result.values, jnp.atleast_1d(result.offset)))
                @ weights
            )
        return model.advance(vector).distribution @ weights

    def oracle(s):
        return (
            reference(
                jnp.array([s, 0.01, 0.02, 0.0]),
                vector,
                backward=backward,
                segregating=False,
            )
            @ weights
        )

    def first(s):
        return jax.jvp(action, (s,), (jnp.ones_like(s),))[1]

    actual = jax.jit(lambda s: jax.jvp(first, (s,), (jnp.ones_like(s),))[1])(
        jnp.asarray(0.03)
    )
    expected = jax.grad(jax.grad(oracle))(0.03)
    assert jnp.abs(expected) > 1e-5
    assert jnp.allclose(actual, expected, atol=1e-10)


def test_demographic_failure_status_and_backward_error():
    """The cached primal retains diagnostics; backward failure cannot return silently."""
    geometry = drifft.Demography([40, 30], [101]).prepare(backend_factory=drifft.FFT)
    model = drifft.DemographicTransition(
        geometry, drifft.Neutral(), method=drifft.PolynomialKrylov(dimension=2)
    )
    result = model.advance(jnp.eye(79)[0], segregating=True, throw=False)
    assert not result.converged
    assert result.status == drifft.EvolutionStatus.unconverged
    with pytest.raises(eqx.EquinoxRuntimeError, match="epoch failed"):
        model.expect(jnp.eye(59)[0], segregating=True).values.block_until_ready()


@pytest.mark.parametrize("backward", [False, True])
def test_identity_history_derivative(backward):
    """An empty history has an identity vector derivative and zero parameter derivatives."""
    geometry = drifft.Demography([3], []).prepare(backend_factory=drifft.FFT)

    def objective(parameters, vector):
        model = drifft.DemographicTransition(
            geometry, drifft.AdditiveSelection(parameters[0])
        )
        if backward:
            result = model.expect(
                vector, segregating=True, singleton_mutation_rate=parameters[1]
            )
            return result.values.sum() + result.offset
        return model.advance(
            vector, segregating=True, singleton_mutation_rate=parameters[1]
        ).distribution.sum()

    params = jnp.array([0.03, 0.0])
    vector = jnp.arange(5.0)
    pgrad, vgrad = jax.jit(jax.grad(objective, argnums=(0, 1)))(params, vector)
    assert jnp.array_equal(pgrad, jnp.zeros_like(params))
    assert jnp.array_equal(vgrad, jnp.ones_like(vector))


@pytest.mark.parametrize(
    "method", [drifft.Sequential(), drifft.RationalKrylov(dimension=16)]
)
@pytest.mark.parametrize("rate", [0.0, 0.001])
def test_source_offset_pullback_matches_dense(method, rate):
    """Offset-only cotangents retain source and process sensitivities at zero rate."""
    geometry = drifft.Demography([3, 2, 4], [3, 2]).prepare(
        backend_factory=drifft.Jacobi
    )
    parameters = jnp.array([0.03, 0.01, 0.02, rate])
    values = jnp.cos(jnp.arange(7))

    def offset(parameters, values):
        s, u, v, rate = parameters
        model = drifft.DemographicTransition(
            geometry,
            drifft.Composition(drifft.AdditiveSelection(s), drifft.Mutation(u, v)),
            method=method,
        )
        # Discard values to exercise a symbolic zero cotangent for that output.
        return model.expect(
            values, segregating=True, singleton_mutation_rate=rate
        ).offset

    def oracle(parameters, values):
        return reference(parameters, values, backward=True, segregating=True)[-1]

    actual = jax.jit(jax.value_and_grad(offset, argnums=(0, 1)))(parameters, values)
    expected = jax.value_and_grad(oracle, argnums=(0, 1))(parameters, values)
    for observed, target in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        assert jnp.allclose(observed, target, atol=2e-9, rtol=2e-8)
    # A zero source has zero reward but still a nonzero rate derivative.
    assert jnp.abs(expected[1][0][-1]) > 1e-3


@pytest.mark.parametrize(
    "backend,method",
    [
        (drifft.FFT, drifft.Sequential()),
        (drifft.FFT, drifft.PolynomialKrylov(dimension=16)),
        (drifft.Jacobi, drifft.RationalKrylov(dimension=16)),
    ],
)
def test_batched_jvp_and_pullback_pairing(backend, method):
    """Tangent-only batching and parameter pullbacks obey the dense pairing."""
    geometry = drifft.Demography([3, 2, 4], [3, 2]).prepare(backend_factory=backend)
    parameters = jnp.array([0.03, 0.01, 0.02, 0.001])
    mass = jnp.linspace(0.01, 0.2, 5)
    directions = jnp.array([[0.2, -0.1, 0.3, 0.02], [-0.3, 0.4, 0.1, -0.01]])
    weights = jnp.cos(jnp.arange(7))

    def action(parameters):
        s, u, v, rate = parameters
        model = drifft.DemographicTransition(
            geometry,
            drifft.Composition(drifft.AdditiveSelection(s), drifft.Mutation(u, v)),
            method=method,
        )
        return model.advance(
            mass, segregating=True, singleton_mutation_rate=rate
        ).distribution

    def oracle(parameters):
        return reference(parameters, mass, backward=False, segregating=True)

    tangents = jax.jit(
        jax.vmap(lambda direction: jax.jvp(action, (parameters,), (direction,))[1])
    )(directions)
    expected = jax.vmap(
        lambda direction: jax.jvp(oracle, (parameters,), (direction,))[1]
    )(directions)
    gradient = jax.jit(jax.grad(lambda p: action(p) @ weights))(parameters)
    assert jnp.allclose(tangents, expected, atol=2e-9, rtol=2e-8)
    assert jnp.allclose(tangents @ weights, directions @ gradient, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("backend", [drifft.FFT, drifft.Jacobi])
def test_demographic_sensitivity_failure_raises_after_successful_primal(
    reverse, backend
):
    """An absorbing primal can converge while its mutation sensitivity is unresolved."""
    geometry = drifft.Demography([8, 5], [20]).prepare(backend_factory=backend)
    initial = jnp.eye(17)[0]

    def result(rate):
        model = drifft.DemographicTransition(
            geometry,
            drifft.Mutation(rate, 0.0),
            method=drifft.PolynomialKrylov(dimension=2, check_sensitivity=True),
        )
        return model.advance(initial, throw=False)

    def objective(rate):
        return result(rate).distribution @ jnp.cos(jnp.arange(11))

    assert result(0.0).converged
    differentiated = (
        jax.grad(objective)
        if reverse
        else lambda rate: jax.jvp(objective, (rate,), (jnp.ones_like(rate),))[1]
    )
    with pytest.raises(
        eqx.EquinoxRuntimeError,
        match="Krylov (sensitivity action|operator pullback) did not converge",
    ):
        eqx.filter_jit(differentiated)(jnp.asarray(0.0)).block_until_ready()


@pytest.mark.parametrize("backward", [False, True])
def test_geometry_is_not_a_differentiation_target(backward):
    """Reject geometry cotangents when differentiating an entire model."""
    geometry = drifft.Demography([3, 2], [3]).prepare(backend_factory=drifft.FFT)
    model = drifft.DemographicTransition(geometry, drifft.AdditiveSelection(0.03))

    def objective(model):
        if backward:
            return model.expect(jnp.arange(5.0)).values.sum()
        return model.advance(jnp.eye(7)[1]).distribution @ jnp.arange(5.0)

    with pytest.raises(TypeError, match="geometry.*nondifferentiable"):
        eqx.filter_grad(objective)(model)


@pytest.mark.parametrize("backward", [False, True])
def test_mixed_parameter_vector_derivatives_match_dense(backward):
    """Retained pullback intermediates preserve mixed process/vector derivatives."""
    geometry = drifft.Demography([3, 2, 4], [3, 2]).prepare(backend_factory=drifft.FFT)
    parameters = jnp.array([0.03, 0.01, 0.02, 0.001])
    vector = jnp.linspace(0.01, 0.2, 7 if backward else 5)
    weights = jnp.cos(jnp.arange(6 if backward else 7))

    def objective(parameters, vector):
        s, u, v, rate = parameters
        model = drifft.DemographicTransition(
            geometry,
            drifft.Composition(drifft.AdditiveSelection(s), drifft.Mutation(u, v)),
        )
        if backward:
            result = model.expect(
                vector, segregating=True, singleton_mutation_rate=rate
            )
            return (
                jnp.concatenate((result.values, jnp.atleast_1d(result.offset)))
                @ weights
            )
        return (
            model.advance(
                vector, segregating=True, singleton_mutation_rate=rate
            ).distribution
            @ weights
        )

    def oracle(parameters, vector):
        return (
            reference(parameters, vector, backward=backward, segregating=True) @ weights
        )

    actual = jax.jit(jax.jacfwd(jax.grad(objective, argnums=0), argnums=1))(
        parameters, vector
    )
    expected = jax.jacfwd(jax.grad(oracle, argnums=0), argnums=1)(parameters, vector)
    assert jnp.max(jnp.abs(expected)) > 1e-3
    assert jnp.allclose(actual, expected, atol=2e-9, rtol=2e-8)


def test_diagnostic_derivatives_are_zero_for_unconverged_primal():
    """Status inspection must not trigger a sensitivity solve for a failed primal."""
    geometry = drifft.Demography([40, 30], [101]).prepare(backend_factory=drifft.FFT)

    def diagnostics(s):
        model = drifft.DemographicTransition(
            geometry,
            drifft.AdditiveSelection(s),
            method=drifft.PolynomialKrylov(dimension=2),
        )
        result = model.advance(jnp.eye(79)[0], segregating=True, throw=False)
        return result.diagnostics[0].error_estimate

    _, tangent = jax.jit(lambda s: jax.jvp(diagnostics, (s,), (jnp.ones_like(s),)))(
        jnp.asarray(0.03)
    )
    gradient = eqx.filter_jit(jax.grad(diagnostics))(jnp.asarray(0.03))
    assert tangent == 0
    assert gradient == 0


def test_joint_expect_pullback_retains_higher_derivatives():
    """Joint parameter/value pullbacks keep all mixed derivatives of the source reward."""
    geometry = drifft.Demography([3, 2, 4], [3, 2]).prepare(backend_factory=drifft.FFT)
    parameters = jnp.array([0.03, 0.01, 0.02, 0.001])
    values = jnp.cos(jnp.arange(7))
    weights = jnp.sin(jnp.arange(6) + 0.3)

    def objective(parameters, values):
        s, u, v, rate = parameters
        model = drifft.DemographicTransition(
            geometry,
            drifft.Composition(drifft.AdditiveSelection(s), drifft.Mutation(u, v)),
        )
        result = model.expect(values, segregating=True, singleton_mutation_rate=rate)
        return jnp.concatenate((result.values, jnp.atleast_1d(result.offset))) @ weights

    def oracle(parameters, values):
        return reference(parameters, values, backward=True, segregating=True) @ weights

    direction = (jnp.array([0.2, -0.1, 0.3, 0.01]), jnp.sin(values))
    joint_gradient = jax.grad(objective, argnums=(0, 1))
    dense_gradient = jax.grad(oracle, argnums=(0, 1))
    actual = jax.jit(lambda p, f: jax.jvp(joint_gradient, (p, f), direction))(
        parameters, values
    )
    expected = jax.jvp(dense_gradient, (parameters, values), direction)
    for observed, target in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        assert jnp.allclose(observed, target, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_symbolic_zero_action_cotangent_skips_pullback_work(
    monkeypatch, backward, cached
):
    """An unused action needs neither a parameter sensitivity nor an opposite action."""
    import equinox.internal as eqxi
    from jax.experimental import hijax

    import drifft._demography.autodiff as actions

    geometry = drifft.Demography([3, 2], [3]).prepare(backend_factory=drifft.FFT)
    model = drifft.DemographicTransition(geometry, drifft.AdditiveSelection(0.03))
    state = (jnp.ones(3 if backward else 5), jnp.asarray(0.0 if backward else 1.0))
    dynamic, static = eqx.partition((model, state, jnp.asarray(0.001)), eqx.is_array)
    avals = jax.tree.map(jax.typeof, dynamic)
    primitive = actions._DemographicAction(
        avals,
        static,
        segregating=True,
        transpose=not backward,
        diagnostics=not backward,
        throw=True,
        cached=False,
    )
    if cached:
        # Cached primal outputs are extra dynamic inputs, but never receive
        # cotangents from the mathematical action, including in higher AD.
        cached_value = jax.tree.map(
            lambda a: jnp.zeros(a.shape, a.dtype), primitive.out_aval
        )
        cached_dynamic, cached_static = eqx.partition(cached_value, eqx.is_array)
        dynamic = (*dynamic, cached_dynamic)
        static = (*static, cached_static)
        primitive = actions._DemographicAction(
            jax.tree.map(jax.typeof, dynamic),
            static,
            segregating=True,
            transpose=not backward,
            diagnostics=not backward,
            throw=True,
            cached=True,
        )
    flags = jax.tree.map(lambda _: False, dynamic[0])
    flags = eqx.tree_at(lambda m: m.frequency_map.s, flags, True)
    nzs = (flags, (True, False), True)
    if cached:
        nzs = (*nzs, jax.tree.map(eqx.is_inexact_array, dynamic[3]))
    cotangent = jax.tree.map(
        lambda a: hijax.Zero(a.to_tangent_aval()), primitive.out_aval
    )
    if not backward:
        # Diagnostics are nondifferentiable even when their cotangent is nonzero.
        state_ct, (status_ct, intervals) = cotangent
        cotangent = (
            state_ct,
            (status_ct, tuple((jnp.asarray(1.0), status) for _, status in intervals)),
        )

    def unexpected_work(*args, **kwargs):
        raise AssertionError("a symbolic-zero action cotangent invoked a pullback")

    monkeypatch.setattr(actions, "_parameter_vjp", unexpected_work)
    monkeypatch.setattr(actions, "_bind_action", unexpected_work)
    gradients = primitive.vjp_bwd_retval(
        (dynamic, unexpected_work, eqxi.Static(nzs)), cotangent
    )
    assert jax.tree.structure(
        gradients, is_leaf=lambda x: isinstance(x, hijax.Zero)
    ) == jax.tree.structure(dynamic)
    for primal, gradient in zip(
        jax.tree.leaves(dynamic),
        jax.tree.leaves(gradients, is_leaf=lambda x: isinstance(x, hijax.Zero)),
        strict=True,
    ):
        assert isinstance(gradient, hijax.Zero)
        assert gradient.aval == jax.typeof(primal).to_tangent_aval()


@pytest.mark.parametrize("backward", [False, True])
def test_numerically_zero_cotangent_retains_curvature(backward):
    """A zero loss gradient at a stationary point must retain its nonzero Hessian."""
    geometry = drifft.Demography([3, 2, 4], [3, 2]).prepare(backend_factory=drifft.FFT)
    vector = jnp.linspace(0.01, 0.2, 7 if backward else 5)
    weights = jnp.cos(jnp.arange(6 if backward else 7))

    def reward(s):
        model = drifft.DemographicTransition(
            geometry,
            drifft.Composition(
                drifft.AdditiveSelection(s), drifft.Mutation(0.01, 0.02)
            ),
        )
        if backward:
            result = model.expect(
                vector, segregating=True, singleton_mutation_rate=0.001
            )
            return (
                jnp.concatenate((result.values, jnp.atleast_1d(result.offset)))
                @ weights
            )
        return (
            model.advance(
                vector, segregating=True, singleton_mutation_rate=0.001
            ).distribution
            @ weights
        )

    def oracle(s):
        return (
            reference(
                jnp.array([s, 0.01, 0.02, 0.001]),
                vector,
                backward=backward,
                segregating=True,
            )
            @ weights
        )

    center = jax.lax.stop_gradient(reward(0.03))

    def loss(s):
        return 0.5 * (reward(s) - center) ** 2

    gradient, curvature = jax.jit(
        lambda s: (jax.grad(loss)(s), jax.grad(jax.grad(loss))(s))
    )(0.03)
    expected = jax.grad(oracle)(0.03) ** 2
    assert jnp.abs(gradient) < 1e-12
    assert expected > 1e-5
    assert jnp.allclose(curvature, expected, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize(
    "backend,method",
    [
        (drifft.FFT, drifft.Sequential()),
        (drifft.FFT, drifft.PolynomialKrylov(dimension=16)),
        (drifft.Jacobi, drifft.RationalKrylov(dimension=16)),
    ],
)
def test_batched_expect_parameter_gradients_match_dense(backend, method):
    """Each batched backward linearization includes process and source dependence."""
    geometry = drifft.Demography([3, 2, 4], [3, 2]).prepare(backend_factory=backend)
    parameters = jnp.array([[0.03, 0.01, 0.02, 0.0], [0.04, 0.02, 0.01, 0.001]])
    values = jnp.cos(jnp.arange(7))
    weights = jnp.sin(jnp.arange(5) + 0.3)

    def objective(parameters):
        s, u, v, rate = parameters
        model = drifft.DemographicTransition(
            geometry,
            drifft.Composition(drifft.AdditiveSelection(s), drifft.Mutation(u, v)),
            method=method,
        )
        result = model.expect(values, segregating=True, singleton_mutation_rate=rate)
        # Weight both outputs; zero source rate still has a nonzero rate derivative.
        return result.values @ weights + 0.7 * result.offset

    def oracle(parameters):
        result = reference(parameters, values, backward=True, segregating=True)
        return result[:-1] @ weights + 0.7 * result[-1]

    actual = jax.jit(jax.vmap(jax.value_and_grad(objective)))(parameters)
    expected = jax.vmap(jax.value_and_grad(oracle))(parameters)
    for observed, target in zip(actual, expected, strict=True):
        assert jnp.allclose(observed, target, atol=2e-9, rtol=2e-8)
