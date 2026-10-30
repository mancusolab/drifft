"""Krylov approximations of power sensitivities and their low-rank adjoints."""

# pattern: Functional Core
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import drifft

jax.config.update("jax_enable_x64", True)


def dense_action(a, x, steps):
    """Use an independent repeated recurrence for derivative references."""
    return jax.lax.fori_loop(0, steps, lambda _, v: a @ v, x)


@pytest.mark.parametrize(
    "method_type", [drifft.PolynomialKrylov, drifft.RationalKrylov]
)
@pytest.mark.parametrize("steps", [0, 1, 7])
@pytest.mark.parametrize("implicit", [False, True])
def test_frechet_power_matches_joint_dense_jvp(method_type, steps, implicit):
    """Recover state and operator perturbations for a signed nonnormal matrix."""
    from drifft._krylov import _frechet_power

    a = jnp.array([[0.9, 0.14, -0.03], [0.0, 0.8, 0.09], [0.0, 0.0, 0.7]])
    e = jnp.array([[0.01, -0.02, 0.03], [0.03, 0.02, 0.01], [0.01, -0.03, 0.02]])
    x, h = jnp.array([0.3, -0.2, 0.4]), jnp.array([-0.1, 0.3, 0.2])
    spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
    op = (
        lx.FunctionLinearOperator(lambda v: a @ v, spec)
        if implicit
        else lx.MatrixLinearOperator(a)
    )
    perturbation = (
        lx.FunctionLinearOperator(lambda v: e @ v, spec)
        if implicit
        else lx.MatrixLinearOperator(e)
    )
    method = method_type(dimension=12, rtol=1e-10, atol=1e-12)
    actual = eqx.filter_jit(_frechet_power)(method, op, perturbation, x, h, steps)
    expected = jax.jvp(lambda a, x: dense_action(a, x, steps), (a, x), (e, h))[1]
    assert actual.converged
    assert jnp.allclose(actual.distribution, expected, atol=2e-11, rtol=2e-10)


@pytest.mark.parametrize(
    "method_type", [drifft.PolynomialKrylov, drifft.RationalKrylov]
)
@pytest.mark.parametrize("steps", [0, 1, 9])
def test_low_rank_frechet_adjoint_matches_dense_pullback(method_type, steps):
    """Check the projected right block transpose and Frobenius adjoint identity."""
    from drifft._krylov import _frechet_adjoint_factors

    a = jnp.array([[0.9, 0.14, -0.03], [0.0, 0.8, 0.09], [0.0, 0.0, 0.7]])
    x, c = jnp.array([0.3, -0.2, 0.4]), jnp.array([-0.1, 0.3, 0.2])
    method = method_type(dimension=8, rtol=1e-10, atol=1e-12)
    left, right, error, converged = eqx.filter_jit(_frechet_adjoint_factors)(
        method, lx.MatrixLinearOperator(a), x, c, steps
    )
    expected = jax.grad(lambda a: c @ dense_action(a, x, steps))(a)
    assert converged
    assert jnp.isfinite(error)
    assert jnp.allclose(left @ right.T, expected, atol=2e-11, rtol=2e-10)


@pytest.mark.parametrize(
    "method_type", [drifft.PolynomialKrylov, drifft.RationalKrylov]
)
@pytest.mark.parametrize("zero_state", [False, True])
def test_frechet_actions_cover_zero_and_invariant_seeds(method_type, zero_state):
    """Breakdown in a primal invariant space must not erase derivative directions."""
    from drifft._krylov import _frechet_adjoint_factors, _frechet_power

    a = jnp.array([[1.0, 0.2, 0.0], [0.0, 0.8, 0.1], [0.0, 0.0, 0.9]])
    e = jnp.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    x = jnp.zeros(3) if zero_state else jnp.array([1.0, 0.0, 0.0])
    h = jnp.array([0.0, 0.0, 1.0])
    c = jnp.array([0.2, -0.3, 0.7])
    method = method_type(dimension=12, rtol=1e-10, atol=1e-12)
    actual = _frechet_power(
        method, lx.MatrixLinearOperator(a), lx.MatrixLinearOperator(e), x, h, 7
    )
    expected = jax.jvp(lambda a, x: dense_action(a, x, 7), (a, x), (e, h))[1]
    left, right, _, converged = _frechet_adjoint_factors(
        method, lx.MatrixLinearOperator(a), x, c, 7
    )
    expected_g = jax.grad(lambda a: c @ dense_action(a, x, 7))(a)
    assert actual.converged & converged
    assert jnp.allclose(actual.distribution, expected, atol=2e-11)
    assert jnp.allclose(left @ right.T, expected_g, atol=2e-11)


@pytest.mark.parametrize(
    "method_type", [drifft.PolynomialKrylov, drifft.RationalKrylov]
)
def test_operator_pullback_preserves_filtered_parameter_tree(method_type):
    """Return callable-operator cotangents without differentiating fixed geometry."""
    from drifft._krylov import _frechet_operator_pullback

    n, steps = 6, 7
    geometry = jnp.linspace(0.7, 0.9, n)
    spec = jax.ShapeDtypeStruct((n,), jnp.float64)

    def make_operator(parameters):
        return lx.FunctionLinearOperator(
            lambda v: parameters[0] * geometry * v + parameters[1] * jnp.roll(v, 1),
            spec,
        )

    parameters = jnp.array([1.0, 0.01])
    operator, template = eqx.filter_jvp(
        make_operator, (parameters,), (jnp.ones_like(parameters),)
    )
    x, c = jnp.cos(jnp.arange(n)), jnp.sin(jnp.arange(n))
    method = method_type(dimension=12, rtol=1e-10, atol=1e-12)
    actual, error, converged = eqx.filter_jit(_frechet_operator_pullback)(
        method, operator, template, x, c, steps
    )
    _, pullback = eqx.filter_vjp(make_operator, parameters)
    # The low-level transpose leaves unperturbed captures symbolic; an ordinary
    # VJP expects explicit zeros for every floating output leaf.
    complete = jax.tree.map(
        lambda g, p: jnp.zeros_like(p) if g is None and eqx.is_inexact_array(p) else g,
        actual,
        operator,
        is_leaf=lambda x: x is None,
    )
    gradient = pullback(complete)[0]

    def ref(parameters):
        op = make_operator(parameters)
        return c @ jax.lax.fori_loop(0, steps, lambda _, v: op.mv(v), x)

    assert converged
    assert jnp.isfinite(error)
    assert jnp.allclose(gradient, jax.grad(ref)(parameters), atol=2e-11, rtol=2e-10)
    assert (
        sum(leaf.size for leaf in jax.tree.leaves(actual) if eqx.is_inexact_array(leaf))
        == parameters.size
    )


@pytest.mark.parametrize(
    "method_type", [drifft.PolynomialKrylov, drifft.RationalKrylov]
)
@pytest.mark.parametrize("dimension,accepted", [(2, False), (32, True)])
def test_truncated_frechet_actions_and_adjoint_pairing(
    method_type, dimension, accepted
):
    """Validate genuine truncation for a long, slowly decaying nonnormal power."""
    from drifft._krylov import (
        _frechet_adjoint_factors,
        _frechet_power,
        _working_problem,
    )

    size, steps = 48, 400
    a = jnp.diag(jnp.linspace(0.99, 0.9995, size)) + jnp.diag(
        jnp.full(size - 1, 0.0001), 1
    )
    e = jnp.sin(jnp.arange(size * size).reshape(size, size) * 0.17) * 0.0003
    x = jnp.cos(jnp.arange(size) * 0.3)
    h = jnp.sin(jnp.arange(size) * 0.2) * 0.01
    c = jnp.sin(jnp.arange(size) * 0.4)
    method = method_type(dimension=dimension, rtol=1e-8, atol=1e-11)
    actual = eqx.filter_jit(_frechet_power)(
        method, lx.MatrixLinearOperator(a), lx.MatrixLinearOperator(e), x, h, steps
    )
    left, right, _, converged = eqx.filter_jit(_frechet_adjoint_factors)(
        method, lx.MatrixLinearOperator(a), x, c, steps
    )
    assert bool(actual.converged) == accepted
    assert bool(converged) == accepted
    if accepted:
        expected = jax.jvp(lambda a, x: dense_action(a, x, steps), (a, x), (e, h))[1]
        expected_g = jax.grad(lambda a: c @ dense_action(a, x, steps))(a)
        state_cotangent = method.solve(
            _working_problem(lx.MatrixLinearOperator(a.T), c, steps)
        ).distribution
        assert jnp.allclose(actual.distribution, expected, atol=2e-9, rtol=2e-8)
        assert jnp.allclose(left @ right.T, expected_g, atol=2e-8, rtol=2e-8)
        assert jnp.allclose(
            c @ actual.distribution,
            jnp.sum((left @ right.T) * e) + state_cotangent @ h,
            atol=2e-9,
            rtol=2e-8,
        )


@pytest.mark.parametrize(
    "method_type", [drifft.PolynomialKrylov, drifft.RationalKrylov]
)
def test_public_operator_rules_validate_truncated_jvp_and_vjp(method_type):
    """The same operator-level AD contract covers matrix-free truncated powers."""
    from drifft._krylov import _working_problem

    size, steps = 48, 400
    a = jnp.diag(jnp.linspace(0.99, 0.9995, size)) + jnp.diag(
        jnp.full(size - 1, 0.0001), 1
    )
    e = jnp.sin(jnp.arange(size * size).reshape(size, size) * 0.17) * 0.0003
    x = jnp.cos(jnp.arange(size) * 0.3)
    h = jnp.sin(jnp.arange(size) * 0.2) * 0.01
    c = jnp.sin(jnp.arange(size) * 0.4)
    spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
    method = method_type(dimension=32, rtol=1e-8, atol=1e-11)

    def action(scale, initial):
        operator = lx.FunctionLinearOperator(lambda v: a @ v + scale * (e @ v), spec)
        return method.solve(_working_problem(operator, initial, steps)).distribution

    def ref(scale, initial):
        return dense_action(a + scale * e, initial, steps)

    args, direction = (jnp.asarray(0.0), x), (jnp.asarray(1.0), h)
    actual = eqx.filter_jit(lambda *args: jax.jvp(action, args, direction))(*args)
    expected = jax.jvp(ref, args, direction)
    for value, target in zip(actual, expected, strict=True):
        assert jnp.allclose(value, target, atol=2e-9, rtol=2e-8)
    g = eqx.filter_jit(
        jax.grad(lambda scale, initial: c @ action(scale, initial), argnums=(0, 1))
    )(*args)
    expected_g = jax.grad(
        lambda scale, initial: c @ ref(scale, initial), argnums=(0, 1)
    )(*args)
    for value, target in zip(g, expected_g, strict=True):
        assert jnp.allclose(value, target, atol=2e-8, rtol=2e-8)
    assert jnp.allclose(c @ actual[1], g[0] + g[1] @ h, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize(
    "method_type", [drifft.PolynomialKrylov, drifft.RationalKrylov]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_derivative_nonconvergence_raises_even_with_throw_false(method_type, reverse):
    """Primal status cannot suppress an unsuccessful sensitivity or pullback."""
    from drifft._krylov import _working_problem

    size, steps = 48, 400
    diagonal = jnp.linspace(0.99, 0.9995, size)
    spec = jax.ShapeDtypeStruct((size,), jnp.float64)
    method = method_type(dimension=2, rtol=1e-8, atol=1e-11)
    initial = jnp.cos(jnp.arange(size) * 0.3)
    weights = jnp.sin(jnp.arange(size) * 0.4)

    def objective(scale):
        operator = lx.FunctionLinearOperator(lambda v: scale * diagonal * v, spec)
        return (
            weights
            @ method.solve(
                _working_problem(operator, initial, steps), throw=False
            ).distribution
        )

    fn = (
        jax.grad(objective)
        if reverse
        else lambda x: jax.jvp(objective, (x,), (jnp.asarray(1.0),))[1]
    )
    with pytest.raises(
        Exception,
        match="Krylov (sensitivity action|operator pullback) did not converge",
    ):
        jax.block_until_ready(eqx.filter_jit(fn)(jnp.asarray(1.0)))


@pytest.mark.parametrize(
    "method_type", [drifft.PolynomialKrylov, drifft.RationalKrylov]
)
def test_higher_operator_derivatives_fail_explicitly(method_type):
    """Unsupported higher derivatives must not fall through to basis differentiation."""
    from drifft._krylov import _working_problem

    method = method_type(dimension=12)
    initial = jnp.array([0.3, 0.5, 0.2])

    def objective(scale):
        operator = lx.MatrixLinearOperator(jnp.diag(jnp.array([0.7, 0.8, 0.9])) * scale)
        return method.solve(_working_problem(operator, initial, 5)).distribution.sum()

    assert jnp.isfinite(jax.grad(objective)(jnp.asarray(1.0)))
    with pytest.raises(
        Exception, match="Higher-order Krylov sensitivities are not implemented"
    ):
        eqx.filter_jit(jax.grad(jax.grad(objective)))(jnp.asarray(1.0))
    with pytest.raises(
        Exception, match="Higher-order Krylov sensitivities are not implemented"
    ):
        eqx.filter_jit(
            lambda x: jax.jvp(jax.grad(objective), (x,), (jnp.asarray(1.0),))
        )(jnp.asarray(1.0))


@pytest.mark.parametrize(
    "method_type", [drifft.PolynomialKrylov, drifft.RationalKrylov]
)
@pytest.mark.parametrize("dimension,accepted", [(2, False), (8, True)])
def test_adjoint_estimate_detects_delayed_propagation(method_type, dimension, accepted):
    """Vanishing fine and coarse cores do not establish an accurate pullback."""
    from drifft._krylov import _frechet_adjoint_factors, _working_problem

    size, steps = 8, 6
    matrix = jnp.diag(jnp.ones(size - 1), -1).at[1, 0].set(0.0)
    initial = jnp.eye(size)[0]
    cotangent = jnp.eye(size)[6]
    perturbation = jnp.zeros_like(matrix).at[1, 0].set(1.0)
    method = method_type(dimension=dimension, rtol=1e-8, atol=1e-12)
    operator = lx.MatrixLinearOperator(matrix)
    primal = method.solve(_working_problem(operator, initial, steps), throw=False)
    assert primal.converged
    assert jnp.allclose(primal.distribution, 0.0)
    expected = jax.grad(
        lambda scale: (
            cotangent @ dense_action(matrix + scale * perturbation, initial, steps)
        )
    )(jnp.asarray(0.0))
    assert expected == 1.0
    left, right, error, converged = eqx.filter_jit(_frechet_adjoint_factors)(
        method, operator, initial, cotangent, steps
    )
    assert bool(converged) == accepted
    if accepted:
        assert jnp.allclose(
            jnp.sum((left @ right.T) * perturbation), expected, atol=1e-12
        )
    else:
        assert error > method.atol

    def objective(scale):
        op = lx.MatrixLinearOperator(matrix + scale * perturbation)
        return (
            cotangent
            @ method.solve(
                _working_problem(op, initial, steps), throw=False
            ).distribution
        )

    if accepted:
        assert jnp.allclose(
            eqx.filter_jit(jax.grad(objective))(jnp.asarray(0.0)), expected, atol=1e-12
        )
    else:
        with pytest.raises(
            Exception, match="Krylov operator pullback did not converge"
        ):
            eqx.filter_jit(jax.grad(objective))(jnp.asarray(0.0))


@pytest.mark.parametrize(
    "method_type", [drifft.PolynomialKrylov, drifft.RationalKrylov]
)
def test_truncated_jacobi_likelihood_gradient_matches_sequential_fft(method_type):
    """Retain encoding and assembly sensitivities with genuinely truncated geometry."""
    from scipy.stats import binom

    copies = 512
    jacobi, fft = drifft.Jacobi(copies), drifft.FFT(copies)
    assert jacobi.rank < copies - 1
    initial = jnp.zeros(copies + 1).at[3].set(1.0)
    sampling_weights = jnp.asarray(binom.pmf(5, 100, jnp.arange(copies + 1) / copies))
    parameters = jnp.array([0.001, 0.003, 1e-5, 2e-5])
    method = method_type(dimension=64, rtol=1e-8, atol=1e-11)

    def objective(parameters, backend, algorithm):
        frequency_map = drifft.Composition(
            drifft.DiploidSelection(parameters[0], parameters[1]),
            drifft.Mutation(parameters[2], parameters[3]),
        )
        return (
            sampling_weights
            @ algorithm.advance(
                backend,
                initial,
                frequency_map=frequency_map,
                generations=25,
            ).distribution
        )

    actual = eqx.filter_jit(jax.value_and_grad(lambda p: objective(p, jacobi, method)))(
        parameters
    )
    expected = eqx.filter_jit(
        jax.value_and_grad(lambda p: objective(p, fft, drifft.Sequential()))
    )(parameters)
    assert jnp.allclose(actual[0], expected[0], atol=2e-12, rtol=2e-10)
    assert jnp.allclose(actual[1], expected[1], atol=2e-9, rtol=2e-8)
