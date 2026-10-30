"""Operator-level power, tangent and transpose contracts."""

# pattern: Functional Core
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import drifft

jax.config.update("jax_enable_x64", True)


def reference(matrix, initial, steps):
    """Repeat matrix applications independently of the custom power boundary."""
    return jax.lax.fori_loop(0, steps, lambda _, x: matrix @ x, initial)


@pytest.mark.parametrize("steps", [0, 1, 7, 31])
@pytest.mark.parametrize("implicit", [False, True])
def test_power_action_operator_and_state_derivatives(steps, implicit):
    """Match joint JVPs, reverse derivatives and Hessians for nonnormal operators."""
    from drifft._power import _power_action

    matrix = jnp.array([[0.9, 0.12, 0.01], [0.0, 0.8, 0.08], [0.0, 0.0, 0.7]])
    initial = jnp.array([0.2, -0.1, 0.5])
    perturbation = jnp.array(
        [[0.01, -0.02, 0.03], [0.02, 0.01, 0.0], [0.0, -0.03, 0.01]]
    )
    direction = (perturbation, -initial)
    weights = jnp.array([0.3, -0.4, 0.8])

    def action(a, x):
        operator = (
            lx.FunctionLinearOperator(
                lambda v: a @ v, jax.ShapeDtypeStruct(x.shape, x.dtype)
            )
            if implicit
            else lx.MatrixLinearOperator(a)
        )
        return _power_action(drifft.Sequential(), operator, x, steps)

    def ref(a, x):
        return reference(a, x, steps)

    actual = eqx.filter_jit(lambda a, x: jax.jvp(action, (a, x), direction))(
        matrix, initial
    )
    expected = jax.jvp(ref, (matrix, initial), direction)
    for a, b in zip(actual, expected, strict=True):
        assert jnp.allclose(a, b, atol=1e-12, rtol=1e-12)

    def derivatives(fn):
        grad = eqx.filter_jit(
            jax.grad(lambda a, x: weights @ fn(a, x), argnums=(0, 1))
        )(matrix, initial)
        hessian = eqx.filter_jit(jax.hessian(lambda a: weights @ fn(a, initial)))(
            matrix
        )
        return grad, hessian

    for a, b in zip(
        jax.tree.leaves(derivatives(action)),
        jax.tree.leaves(derivatives(ref)),
        strict=True,
    ):
        assert jnp.allclose(a, b, atol=2e-12, rtol=2e-12)
    transposed = jax.linear_transpose(lambda x: action(matrix, x), initial)(weights)[0]
    assert jnp.allclose(transposed, reference(matrix.T, weights, steps), atol=1e-12)
    batch = jnp.stack((initial, -initial))
    assert jnp.allclose(
        eqx.filter_jit(jax.vmap(action, in_axes=(None, 0)))(matrix, batch),
        jax.vmap(ref, in_axes=(None, 0))(matrix, batch),
        atol=1e-12,
    )


def test_power_action_pytree_state_and_parameter_closure():
    """Preserve callable captures and nested state structure under filtered AD."""
    from drifft._power import _power_action

    initial = {"left": jnp.array([0.2, 0.3]), "right": jnp.asarray(-0.1)}

    def action(rate, initial):
        def mv(x):
            return {"left": rate * x["left"] + x["right"], "right": 0.8 * x["right"]}

        op = lx.FunctionLinearOperator(
            mv, jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), initial)
        )
        return _power_action(drifft.Sequential(), op, initial, 5)

    def ref(rate, initial):
        return jax.lax.fori_loop(
            0,
            5,
            lambda _, x: {
                "left": rate * x["left"] + x["right"],
                "right": 0.8 * x["right"],
            },
            initial,
        )

    def objective(fn, rate, x):
        result = fn(rate, x)
        return result["left"].sum() + result["right"]

    actual = eqx.filter_jit(
        jax.grad(lambda r, x: objective(action, r, x), argnums=(0, 1))
    )(jnp.asarray(0.9), initial)
    expected = jax.grad(lambda r, x: objective(ref, r, x), argnums=(0, 1))(
        jnp.asarray(0.9), initial
    )
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert jnp.allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("backend_type", [drifft.FFT, drifft.NUFFT, drifft.Jacobi])
@pytest.mark.parametrize("segregating", [False, True])
def test_prepared_epoch_derivatives_include_encoding_and_source(
    backend_type, segregating
):
    """Preparation and reconstruction remain differentiated around the power boundary."""
    from scipy.special import comb

    from drifft._epochs import _prepare_epoch

    copies, steps = 12, 5
    backend = backend_type(copies)
    size = copies - 1 if segregating else copies + 1
    initial = jnp.linspace(0.01, 0.03, size)
    source = jnp.linspace(0.0001, 0.0002, size)
    parameters = jnp.array([0.01, 0.03, 0.002, 0.003])
    weights = jnp.sin(jnp.arange(size))

    def frequency_map(parameters):
        return drifft.Composition(
            drifft.DiploidSelection(parameters[0], parameters[1]),
            drifft.Mutation(parameters[2], parameters[3]),
        )

    def action(parameters, initial, source):
        problem = _prepare_epoch(
            backend, initial, frequency_map(parameters), steps, segregating, source
        )
        return drifft.Sequential().solve(problem).distribution

    def ref(parameters, initial, source):
        q = frequency_map(parameters)(jnp.arange(copies + 1) / copies)
        counts = jnp.arange(copies + 1)
        matrix = (
            jnp.asarray(comb(copies, counts))
            * q[:, None] ** counts
            * (1 - q[:, None]) ** (copies - counts)
        )
        if segregating:
            matrix = matrix[1:-1, 1:-1]
        return jax.lax.fori_loop(0, steps, lambda _, x: matrix.T @ x + source, initial)

    args = (parameters, initial, source)
    direction = (jnp.array([0.1, -0.2, 0.03, -0.04]), initial[::-1], -source)
    for a, b in zip(
        eqx.filter_jit(lambda *args: jax.jvp(action, args, direction))(*args),
        jax.jvp(ref, args, direction),
        strict=True,
    ):
        assert jnp.allclose(a, b, atol=1e-11, rtol=2e-10)
    actual = eqx.filter_jit(
        jax.grad(lambda *args: weights @ action(*args), argnums=(0, 1, 2))
    )(*args)
    expected = jax.grad(lambda *args: weights @ ref(*args), argnums=(0, 1, 2))(*args)
    for a, b in zip(actual, expected, strict=True):
        assert jnp.allclose(a, b, atol=1e-10, rtol=2e-10)
