"""Matched Bernstein sampling transpose and bounded-work accumulation."""

# Functional Core: signed linear-map and transformation contracts.
import re

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import binom

from drifft import NUFFT
from drifft.backends._spectral import sample_bernstein

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("size,batch_size", [(16, 3), (512, 17)])
def test_sampling_signed_adjoint_and_transforms(size, batch_size):
    plan = NUFFT(size, batch_size=batch_size)
    rng = np.random.default_rng(size)
    values = jnp.asarray(rng.normal(size=(2, size + 1)))
    mass = jnp.asarray(rng.normal(size=plan.degree + 1))

    def sample(v):
        return sample_bernstein(plan, v)

    def transpose(m):
        return jax.linear_transpose(sample, values[0])(m)[0]

    actual = jax.jit(transpose)(mass)
    matrix = binom.pmf(
        np.arange(size + 1)[None, :], size, np.asarray(plan.nodes)[:, None]
    )
    np.testing.assert_allclose(actual, matrix.T @ mass, atol=3e-12, rtol=0)
    np.testing.assert_allclose(mass @ sample(values[0]), actual @ values[0], atol=1e-12)
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda v: mass @ sample(v)))(values[0]), actual, atol=1e-12
    )
    np.testing.assert_allclose(
        jax.jit(jax.vmap(sample))(values), values @ matrix.T, atol=3e-12, rtol=0
    )
    np.testing.assert_allclose(
        jax.jit(jax.vmap(transpose))(jnp.stack((mass, -mass))),
        jnp.stack((actual, -actual)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda m: transpose(m) @ values[0]))(mass),
        sample(values[0]),
        atol=1e-12,
    )
    _, tangent = jax.jvp(sample, (values[0],), (values[1],))
    np.testing.assert_allclose(tangent, sample(values[1]), atol=1e-12)
    endpoint_mass = jnp.zeros_like(mass).at[0].set(-2).at[-1].set(3)
    expected = jnp.zeros(size + 1).at[0].set(-2).at[-1].set(3)
    np.testing.assert_array_equal(jax.jit(transpose)(endpoint_mass), expected)


@pytest.mark.parametrize("size,batch_size", [(512, 7), (2048, 19)])
def test_sampling_transpose_initializes_full_accumulator_only_once(size, batch_size):
    plan = NUFFT(size, batch_size=batch_size)
    transpose = jax.jit(
        lambda mass: jax.linear_transpose(
            lambda v: sample_bernstein(plan, v), jnp.zeros(size + 1)
        )(mass)[0]
    )
    hlo = transpose.lower(jnp.ones(plan.degree + 1)).as_text()
    # A full-vector broadcast in the loop body would reintroduce one M-sized
    # zero fill per node batch. Only the outer accumulator needs such a fill.
    full_vector_fills = re.findall(
        rf"stablehlo.broadcast_in_dim[^\n]*-> tensor<{size + 1}xf64>", hlo
    )
    assert len(full_vector_fills) == 1


def test_sampling_rejects_trainable_geometry():
    plan = NUFFT(16)

    def sample_nodes(nodes):
        updated = eqx.tree_at(lambda p: p.nodes, plan, nodes)
        return sample_bernstein(updated, jnp.arange(17.0)).sum()

    with pytest.raises(ValueError, match="fixed.*nodes"):
        jax.grad(sample_nodes)(plan.nodes)
