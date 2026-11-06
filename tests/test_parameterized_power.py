"""Parameter-space power semantics and prepared-operator compatibility."""

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import drifft
from drifft._evolution.parameterized import _JacobiOperator, _pack, _power

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("augmented", [False, True])
def test_compact_power_matches_repeated_dense_action(transposed, augmented):
    """Check shared-operator accumulation, source gradients, JVP and mapped AD."""
    backend = drifft.Jacobi(4)
    method = drifft.RationalKrylov(dimension=16)
    parameters = jnp.array([0.03, 0.01, 0.02])
    source = jnp.linspace(0.01, 0.03, backend.rank)
    initial = jnp.linspace(-0.2, 0.3, backend.rank + augmented)

    def operator(p, source):
        return _JacobiOperator(
            backend,
            drifft.Composition(
                drifft.AdditiveSelection(p[0]), drifft.Mutation(p[1], p[2])
            ),
            source if augmented else None,
            True,
            transposed,
        )

    def objective(p, source, x, *, native):
        op = operator(p, source)
        matrix = op.as_matrix()
        if native:
            op = _pack(op, jax.lax.stop_gradient(matrix))
            # Reusing the opaque value requires addition in parameter space.
            result = _power(op, x, method, 3) + _power(op, x * 0.7, method, 2)
        else:
            result = matrix @ matrix @ matrix @ x + matrix @ matrix @ (x * 0.7)
        return jnp.cos(jnp.arange(x.size)) @ result

    def actual(p, source, x):
        return objective(p, source, x, native=True)

    def reference(p, source, x):
        return objective(p, source, x, native=False)

    inputs = parameters, source, initial
    directions = parameters * 0.1, source * 0.2, initial * -0.3
    for transform in (
        lambda fn: jax.value_and_grad(fn, argnums=(0, 1, 2)),
        lambda fn: lambda *args: jax.jvp(fn, args, directions),
    ):
        expected = transform(reference)(*inputs)
        result = jax.jit(transform(actual))(*inputs)
        for a, b in zip(
            jax.tree.leaves(result), jax.tree.leaves(expected), strict=True
        ):
            assert jnp.allclose(a, b, rtol=2e-8, atol=2e-10)
    batch = jnp.stack((parameters, parameters * 0.8))
    result = jax.jit(jax.vmap(jax.grad(actual), in_axes=(0, None, None)))(
        batch, source, initial
    )
    expected = jax.vmap(jax.grad(reference), in_axes=(0, None, None))(
        batch, source, initial
    )
    assert jnp.allclose(result, expected, rtol=2e-8, atol=2e-10)
    result = jax.jit(
        jax.grad(
            lambda p: jnp.sum(
                jax.vmap(actual, in_axes=(None, None, 0))(
                    p, source, jnp.stack((initial, -initial * 0.5))
                )
            )
        )
    )(parameters)
    expected = jax.grad(lambda p: reference(p, source, initial * 0.5))(parameters)
    assert jnp.allclose(result, expected, rtol=2e-8, atol=2e-10)


def test_operator_cotangent_is_parameter_vector():
    """A matrix cache must not become a dense matrix cotangent."""
    backend = drifft.Jacobi(4)
    op = _JacobiOperator(backend, drifft.AdditiveSelection(0.02), None, False)
    opaque = _pack(op, op.as_matrix())
    initial = jnp.linspace(0.1, 0.4, backend.rank + 2)
    gradient = jax.jit(
        jax.grad(
            lambda a: jnp.sum(
                _power(a, initial, drifft.RationalKrylov(dimension=16), 3)
            )
        )
    )(opaque)
    assert gradient.shape == (1,)
    expected = jax.grad(
        lambda s: jnp.sum(
            jnp.linalg.matrix_power(
                _JacobiOperator(
                    backend, drifft.AdditiveSelection(s), None, False
                ).as_matrix(),
                3,
            )
            @ initial
        )
    )(jnp.asarray(0.02))
    assert jnp.allclose(gradient[0], expected, rtol=1e-9, atol=1e-11)


def test_prepared_operator_keeps_parameter_derivatives():
    """Custom solvers can differentiate the ordinary Lineax prepared operator."""
    backend = drifft.Jacobi(4)
    mass = jnp.linspace(0.01, 0.03, 9)
    method = drifft.RationalKrylov(dimension=16)

    def prepared(s):
        problem = drifft.prepare_epoch(
            method,
            backend,
            mass,
            frequency_map=drifft.AdditiveSelection(s),
            generations=4,
        )
        return problem.reconstruct(
            problem.operator.mv(
                problem.operator.mv(problem.operator.mv(problem.initial))
            )
        )

    def builtin(s):
        return method.advance(
            backend, mass, frequency_map=drifft.AdditiveSelection(s), generations=4
        ).distribution

    value = jnp.asarray(0.02)
    actual = eqx.filter_jit(jax.jacfwd(prepared))(value)
    expected = eqx.filter_jit(jax.jacfwd(builtin))(value)
    assert jnp.allclose(actual, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("transposed", [False, True])
def test_tangent_only_batch_and_linear_solve(transposed):
    """Share the primal across JVP directions and support Lineax custom solvers."""
    backend = drifft.Jacobi(4)
    method = drifft.RationalKrylov(dimension=16)
    parameters = jnp.array([0.02, 0.01])
    initial = jnp.linspace(0.1, 0.3, backend.rank)

    def operator(p):
        return _JacobiOperator(backend, drifft.Mutation(*p), None, True, transposed)

    def action(p):
        op = operator(p)
        return _power(
            _pack(op, jax.lax.stop_gradient(op.as_matrix())), initial, method, 3
        )

    directions = jnp.eye(2)
    result = jax.jit(jax.vmap(lambda d: jax.jvp(action, (parameters,), (d,))))(
        directions
    )
    expected = jax.vmap(
        lambda d: jax.jvp(
            lambda p: jnp.linalg.matrix_power(operator(p).as_matrix(), 3) @ initial,
            (parameters,),
            (d,),
        )
    )(directions)
    for a, b in zip(result, expected, strict=True):
        assert jnp.allclose(a, b, rtol=1e-8, atol=1e-10)

    def solve(p, *, dense):
        op = operator(p)
        shifted = lx.IdentityLinearOperator(op.in_structure()) * 2 - op
        result = (
            jnp.linalg.solve(shifted.as_matrix(), initial)
            if dense
            else lx.linear_solve(shifted, initial, lx.LU()).value
        )
        return jnp.sum(result)

    actual = jax.jit(jax.value_and_grad(lambda p: solve(p, dense=False)))(parameters)
    expected = jax.value_and_grad(lambda p: solve(p, dense=True))(parameters)
    for a, b in zip(actual, expected, strict=True):
        assert jnp.allclose(a, b, rtol=1e-10, atol=1e-12)


def test_source_only_operator_has_no_process_directions():
    """A neutral map still propagates encoded-source and state cotangents."""
    backend = drifft.Jacobi(4)
    source = jnp.linspace(0.001, 0.003, backend.rank)
    initial = jnp.linspace(0.1, 0.4, backend.rank + 1)
    method = drifft.RationalKrylov(dimension=16)

    def objective(source, *, native):
        op = _JacobiOperator(backend, drifft.Neutral(), source, True)
        matrix = op.as_matrix()
        result = (
            _power(_pack(op, jax.lax.stop_gradient(matrix)), initial, method, 3)
            if native
            else jnp.linalg.matrix_power(matrix, 3) @ initial
        )
        return jnp.cos(jnp.arange(initial.size)) @ result

    actual = jax.jit(jax.grad(lambda x: objective(x, native=True)))(source)
    expected = jax.grad(lambda x: objective(x, native=False))(source)
    assert jnp.allclose(actual, expected, rtol=1e-9, atol=1e-11)


def test_nested_parameter_batches():
    """Parameter grids preserve the prepared power under nested mapping."""
    backend = drifft.Jacobi(3)
    initial = jnp.linspace(0.1, 0.3, backend.rank)
    method = drifft.RationalKrylov(dimension=12)

    def action(rate, *, native):
        op = _JacobiOperator(backend, drifft.Mutation(rate, 0.02), None, True)
        matrix = op.as_matrix()
        return (
            _power(_pack(op, jax.lax.stop_gradient(matrix)), initial, method, 3)
            if native
            else jnp.linalg.matrix_power(matrix, 3) @ initial
        )

    parameters = jnp.array([[0.01, 0.02], [0.03, 0.04]])
    actual = jax.jit(jax.vmap(jax.vmap(lambda p: action(p, native=True))))(parameters)
    expected = jax.vmap(jax.vmap(lambda p: action(p, native=False)))(parameters)
    assert jnp.allclose(actual, expected, rtol=1e-9, atol=1e-11)
    actual_gradient = jax.jit(
        jax.vmap(jax.vmap(jax.grad(lambda p: jnp.sum(action(p, native=True)))))
    )(parameters)
    expected_gradient = jax.vmap(
        jax.vmap(jax.grad(lambda p: jnp.sum(action(p, native=False))))
    )(parameters)
    assert jnp.allclose(actual_gradient, expected_gradient, rtol=1e-9, atol=1e-11)
