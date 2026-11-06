"""Operator-coordinate derivatives independent of power algorithms."""

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import drifft
from drifft._evolution.autodiff import _cached_power_action
from drifft._evolution.operators import (
    _contract_operator_factors,
    _numerical_operator,
    _operator_pushforward,
)
from drifft._evolution.parameterized import (
    _contract_parameter_factors,
    _JacobiOperator,
    _pack,
    _parameter_pushforward,
    _unpack,
)

jax.config.update("jax_enable_x64", True)


def test_generic_operator_derivatives_preserve_matrix_free_structure(monkeypatch):
    """Generic derivatives use actions and return only active captured coordinates."""

    def forbid_materialization(self):
        raise AssertionError("matrix-free operator was materialized")

    monkeypatch.setattr(lx.FunctionLinearOperator, "as_matrix", forbid_materialization)
    geometry = jnp.array([0.7, 0.8, 0.9])
    parameters = jnp.array([0.8, 0.02])
    direction = jnp.array([0.1, -0.2])
    x = jnp.array([0.2, -0.4, 0.6])
    left = jnp.array([[0.1, 0.2], [0.4, -0.2], [-0.1, 0.3]])
    right = left + 0.2

    def make(p):
        # The same parameter contributes twice; geometry is a fixed capture.
        return lx.FunctionLinearOperator(
            lambda v: p[0] * geometry * v + p[1] * jnp.roll(v, 1) + p[0] * v,
            jax.ShapeDtypeStruct((3,), jnp.float64),
        )

    def matrix(p):
        return jnp.diag(p[0] * (geometry + 1)) + p[1] * jnp.roll(jnp.eye(3), 1, axis=0)

    op, template = eqx.filter_jvp(make, (parameters,), (direction,))
    assert _numerical_operator(op) is op
    actual = eqx.filter_jit(lambda a, d: _operator_pushforward(a, d).mv(x))(
        op, template
    )
    expected = jax.jvp(lambda p: matrix(p) @ x, (parameters,), (direction,))[1]
    assert jnp.allclose(actual, expected, atol=1e-12)
    ct = eqx.filter_jit(_contract_operator_factors)(op, template, left, right)
    _, pullback = eqx.filter_vjp(make, parameters)
    complete = jax.tree.map(
        lambda g, p: jnp.zeros_like(p) if g is None and eqx.is_inexact_array(p) else g,
        ct,
        op,
        is_leaf=lambda v: v is None,
    )
    expected = jax.grad(lambda p: jnp.sum(left * (matrix(p) @ right)))(parameters)
    assert jnp.allclose(pullback(complete)[0], expected, atol=1e-12)
    assert (
        sum(a.size for a in jax.tree.leaves(ct) if eqx.is_array(a)) == parameters.size
    )


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("augmented", [False, True])
def test_parameter_factors_match_direct_assembly_derivatives(transposed, augmented):
    """Source and active process contractions obey the same scalar pairing."""
    backend = drifft.Jacobi(3)
    parameters = jnp.array([0.02, 0.01, 0.03])
    source = jnp.linspace(0.01, 0.02, backend.rank)

    def make(p, source):
        return _JacobiOperator(
            backend,
            drifft.Composition(
                drifft.AdditiveSelection(p[0]), drifft.Mutation(p[1], p[2])
            ),
            source if augmented else None,
            True,
            transposed,
        )

    size = backend.rank + augmented
    left = jnp.sin(jnp.arange(size * 2).reshape(size, 2))
    right = jnp.cos(jnp.arange(size * 2).reshape(size, 2))
    direction = jnp.linspace(0.1, 0.3, 3 + (backend.rank if augmented else 0))

    def evaluate(p, source):
        op = make(p, source)
        bound = _unpack(_pack(op, op.as_matrix()))
        return (
            _parameter_pushforward(bound, direction).as_matrix(),
            _contract_parameter_factors(bound, left, right, (0, 2)),
        )

    perturbation, cotangent = jax.jit(evaluate)(parameters, source)
    ds = direction[3:] if augmented else jnp.zeros_like(source)
    expected = jax.jvp(
        lambda p, s: make(p, s).as_matrix(), (parameters, source), (direction[:3], ds)
    )[1]
    assert jnp.allclose(perturbation, expected, atol=1e-12)
    gp, gs = jax.grad(
        lambda p, s: jnp.sum(left * (make(p, s).as_matrix() @ right)), argnums=(0, 1)
    )(parameters, source)
    gp = gp.at[1].set(0)
    expected = jnp.concatenate((gp, gs)) if augmented else gp
    assert jnp.allclose(cotangent, expected, atol=1e-12)


def test_parameter_cached_binding_ignores_cache_tangents_and_preserves_aliases():
    """Cached value/matrix are numerical state; repeated process leaves sum once."""
    backend = drifft.Jacobi(3)
    method = drifft.PolynomialKrylov(8)
    initial = jnp.linspace(0.1, 0.3, backend.rank)
    rate = jnp.array(0.01)

    def make(rate):
        return _JacobiOperator(backend, drifft.Mutation(rate, rate), None, True)

    matrix = make(rate).as_matrix()
    value = matrix @ matrix @ initial

    def action(p, x, cached, matrix):
        return _cached_power_action(
            make(p), method, x, 2, cached, lx.MatrixLinearOperator(matrix)
        )

    args = rate, initial, value, matrix
    directions = (
        jnp.array(0.2),
        jnp.ones_like(initial),
        jnp.ones_like(value),
        jnp.ones_like(matrix),
    )
    actual_value, actual = jax.jit(lambda *args: jax.jvp(action, args, directions))(
        *args
    )
    expected = jax.jvp(
        lambda p, x: make(p).as_matrix() @ make(p).as_matrix() @ x,
        args[:2],
        directions[:2],
    )[1]
    assert jnp.array_equal(actual_value, value)
    assert jnp.allclose(actual, expected, atol=1e-11)
    gradient = jax.jit(jax.grad(lambda p: jnp.sum(action(p, initial, value, matrix))))(
        rate
    )
    expected_gradient = jax.grad(
        lambda p: jnp.sum(make(p).as_matrix() @ make(p).as_matrix() @ initial)
    )(rate)
    assert jnp.allclose(gradient, expected_gradient, atol=1e-11)
