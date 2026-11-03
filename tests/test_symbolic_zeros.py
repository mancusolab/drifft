"""Inactive derivative directions remain symbolic across expensive kernels."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import hijax

from drifft._backend._spectral import (
    _BernsteinSampling,
    _CosineEvaluation,
    _CosineTranspose,
)
from drifft._backend.fft import _BinomialWeights
from drifft._polynomial import _GegenbauerMoments

jax.config.update("jax_enable_x64", True)


def _case(name):
    """Pair each primitive with a small independent polynomial reference."""
    q = jnp.array([0.1, 0.3, 0.8])
    values = jnp.array([0.3, -0.2, 0.7])

    def cosine_matrix(q):
        z = 1 - 2 * q
        return jnp.stack((jnp.ones_like(z), z, 2 * z**2 - 1), axis=1)

    if name == "cosine":
        args = (values, q, jnp.array([0.8, -0.1]))
        primitive = _CosineEvaluation(*map(jax.typeof, args), tolerance=1e-12)
        return primitive, args, lambda c, q, e: cosine_matrix(q) @ c
    if name == "transpose":
        args = (q, values)
        primitive = _CosineTranspose(*map(jax.typeof, args), count=3, tolerance=1e-12)
        return primitive, args, lambda q, m: (cosine_matrix(q).T @ m, jnp.zeros(2))
    if name == "binomial":
        args = (jnp.arange(4), jnp.log(jnp.array([1.0, 3.0, 3.0, 1.0])), q)
        primitive = _BinomialWeights(*map(jax.typeof, args), size=3)
        return (
            primitive,
            args,
            lambda k, logs, q: (
                jnp.exp(logs) * q[:, None] ** k * (1 - q[:, None]) ** (3 - k)
            ),
        )
    args = (q, values)
    transposed = name == "evaluation"
    primitive = _GegenbauerMoments(
        *map(jax.typeof, args), rank=3, order=0, transposed=transposed
    )

    def reference(z, v):
        basis = jnp.stack(
            (jnp.ones_like(z), jnp.sqrt(5.0) * z, jnp.sqrt(7 / 8) * (5 * z**2 - 1)),
            axis=1,
        )
        return (basis if transposed else basis.T) @ v

    return primitive, args, reference


@pytest.mark.parametrize(
    "name", ["cosine", "transpose", "binomial", "moments", "evaluation"]
)
def test_symbolic_activity_matches_independent_derivative(name):
    """Check every activity mask, including fully inactive inputs."""
    primitive, args, reference = _case(name)
    differentiable = [
        i for i, x in enumerate(args) if jnp.issubdtype(x.dtype, jnp.inexact)
    ]
    for mask in itertools.product((False, True), repeat=len(differentiable)):
        active = dict(zip(differentiable, mask, strict=True))
        tangents = tuple(
            jnp.ones_like(x)
            if active.get(i, False)
            else hijax.Zero(jax.typeof(x).to_tangent_aval())
            for i, x in enumerate(args)
        )
        primal, tangent = primitive.jvp(args, tangents)
        expected, derivative = jax.jvp(
            reference, args, tuple(map(hijax.instantiate_zeros, tangents))
        )
        for actual, wanted in zip(
            jax.tree.leaves(primal), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_allclose(actual, wanted, atol=2e-11)
        leaves = jax.tree.leaves(tangent, is_leaf=lambda x: isinstance(x, hijax.Zero))
        if not any(mask):
            assert all(isinstance(x, hijax.Zero) for x in leaves)
        for actual, wanted in zip(leaves, jax.tree.leaves(derivative), strict=True):
            np.testing.assert_allclose(
                hijax.instantiate_zeros(actual), wanted, atol=2e-10
            )


@pytest.mark.parametrize("name", ["cosine", "transpose", "moments", "evaluation"])
def test_symbolic_zero_transpose_preserves_existing_accumulation(name):
    """An unused output contributes nothing and leaves accumulated gradients intact."""
    primitive, args, _ = _case(name)
    linear_indices = (0, 2) if name == "cosine" else (1,)
    operands = list(args)
    for i in linear_indices:
        operands[i] = hijax.ValAccum(jax.typeof(args[i]), jnp.ones_like(args[i]))
    cotangent = jax.tree.map(
        lambda a: hijax.Zero(a.to_tangent_aval()), primitive.out_aval
    )
    primitive.transpose(cotangent, *operands)
    for i in linear_indices:
        np.testing.assert_array_equal(operands[i].freeze(), jnp.ones_like(args[i]))


def test_sampling_symbolic_zero_and_fixed_geometry_contract():
    """Skipping inactive values must not accept active, numerically zero node tangents."""
    nodes, values = jnp.array([0.0, 0.5, 1.0]), jnp.arange(9.0)
    primitive = _BernsteinSampling(
        jax.typeof(nodes),
        jax.typeof(values),
        size=8,
        radius=8,
        batch_size=2,
        transposed=False,
    )
    zeros = tuple(hijax.Zero(jax.typeof(x)) for x in (nodes, values))
    _, tangent = primitive.jvp((nodes, values), zeros)
    assert isinstance(tangent, hijax.Zero)
    with pytest.raises(ValueError, match="nontrainable nodes"):
        primitive.jvp((nodes, values), (jnp.zeros_like(nodes), zeros[1]))
    accum = hijax.ValAccum(jax.typeof(values), jnp.ones_like(values))
    primitive.transpose(hijax.Zero(primitive.out_aval), nodes, accum)
    np.testing.assert_array_equal(accum.freeze(), jnp.ones_like(values))


def test_cosine_transpose_tangent_batch_keeps_shared_primal():
    """The two-output adjoint also preserves unmapped primals under tangent batching."""
    primitive, args, reference = _case("transpose")
    directions = jnp.stack((jnp.ones(3), -jnp.ones(3)), axis=1)
    actual = jax.jit(
        jax.vmap(
            lambda d: jax.jvp(lambda m: primitive(args[0], m), (args[1],), (d,)),
            in_axes=1,
            out_axes=(None, 1),
        )
    )(directions)
    expected = jax.vmap(
        lambda d: jax.jvp(lambda m: reference(args[0], m), (args[1],), (d,)),
        in_axes=1,
        out_axes=(None, 1),
    )(directions)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, atol=2e-11)


@pytest.mark.parametrize("output", [0, 1])
def test_cosine_transpose_unused_output_pullback(output):
    """Dropping one adjoint output preserves the other's endpoint contributions."""
    q = jnp.array([0.0, 0.3, 1.0])
    mass = jnp.array([0.2, -0.3, 0.7])
    primitive = _CosineTranspose(
        jax.typeof(q), jax.typeof(mass), count=3, tolerance=1e-12
    )
    gradient = jax.jit(jax.grad(lambda m: primitive(q, m)[output].sum()))(mass)
    expected = (
        jnp.array([0.0, 1 + 0.4 + 2 * 0.4**2 - 1, 0.0])
        if output == 0
        else jnp.array([1.0, 0.0, 1.0])
    )
    np.testing.assert_allclose(gradient, expected, atol=2e-11)
