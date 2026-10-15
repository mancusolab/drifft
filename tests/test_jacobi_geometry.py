"""Compact quadrature storage and asymmetric distribution reconstruction."""

# pattern: Functional Core
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.scipy.special import gammaln, xlog1py, xlogy
from scipy.special import roots_jacobi

from drifft._jacobi import _JacobiGeometry, _polynomials

jax.config.update("jax_enable_x64", True)


def full_quadrature_decoder(copies, rank, segregating):
    """Build the full-node, full-binomial decoder independently of stored stencils."""
    nodes, weights = roots_jacobi(rank + 32, 1, 1)
    nodes, weights = jnp.asarray(nodes), 0.75 * jnp.asarray(weights)
    probability = (nodes[:, None] + 1) / 2
    count = jnp.arange(copies - 1, dtype=jnp.float64)[None, :]
    trials = copies - 2
    binomial = jnp.exp(
        gammaln(trials + 1)
        - gammaln(count + 1)
        - gammaln(trials - count + 1)
        + xlogy(count, probability)
        + xlog1py(trials - count, -probability)
    )
    coordinates = 2 * jnp.arange(1, copies) / copies - 1
    interior = (binomial.T @ (weights[:, None] * _polynomials(nodes, rank))) / (
        (1 - coordinates) * (1 + coordinates)
    )[:, None]
    if segregating:
        return interior
    result = jnp.zeros((copies + 1, rank + 2)).at[1:-1, 2:].set(interior)
    result = result.at[0, :2].set(jnp.array([0.5, -0.5]))
    result = result.at[-1, :2].set(jnp.array([0.5, 0.5]))
    result = result.at[0, 2:].set(-0.5 * (1 - coordinates) @ interior)
    return result.at[-1, 2:].set(-0.5 * (1 + coordinates) @ interior)


@pytest.mark.parametrize("copies,rank", [(16, 1), (17, 2), (32, 7), (33, 8)])
@pytest.mark.parametrize("segregating", [True, False])
def test_compact_decoder_preserves_asymmetric_inputs_and_derivatives(
    copies, rank, segregating
):
    """Even/odd nodes, the central node, and signed coefficients retain the same map."""
    geometry = _JacobiGeometry(copies, rank)
    reference = full_quadrature_decoder(copies, rank, segregating)
    coefficients = jnp.sin(jnp.arange(reference.shape[1]) + 0.3)
    decode = eqx.filter_jit(lambda values: geometry.decode(values, segregating))
    assert jnp.allclose(
        decode(coefficients), reference @ coefficients, atol=2e-13, rtol=2e-13
    )
    assert jnp.allclose(
        jax.jacfwd(decode)(coefficients), reference, atol=2e-13, rtol=2e-13
    )
    terminal = jnp.cos(jnp.arange(reference.shape[0]))
    assert jnp.allclose(
        jax.grad(lambda c: decode(c) @ terminal)(coefficients),
        reference.T @ terminal,
        atol=2e-13,
        rtol=2e-13,
    )
    batch = jnp.stack((coefficients, coefficients[::-1]))
    assert jnp.allclose(
        jax.vmap(decode)(batch), batch @ reference.T, atol=2e-13, rtol=2e-13
    )


def test_geometry_storage_uses_half_nodes_and_compact_indices():
    """Enforce the intended storage budget without relying on process RSS."""
    copies, rank = 1000, 96
    geometry = _JacobiGeometry(copies, rank)
    assert geometry.polynomials.shape == ((rank + 33) // 2, rank)
    integer_leaves = [
        leaf
        for leaf in jax.tree.leaves(geometry)
        if jnp.issubdtype(leaf.dtype, jnp.integer)
    ]
    assert integer_leaves
    assert all(leaf.dtype == jnp.int32 and leaf.ndim == 1 for leaf in integer_leaves)
    assert sum(leaf.size for leaf in integer_leaves) <= rank + 32


@pytest.mark.parametrize("copies,rank", [(1000, 96), (1001, 97)])
def test_compact_truncated_stencils_match_full_binomial_quadrature(copies, rank):
    """Reflection remains accurate when stencil tails are actually truncated."""
    geometry = _JacobiGeometry(copies, rank)
    assert geometry.stencil.shape[1] < copies - 1
    coefficients = jnp.sin(jnp.arange(rank) + 0.3) / (1 + jnp.arange(rank))
    actual = eqx.filter_jit(geometry.decode)(coefficients, True)
    reference = full_quadrature_decoder(copies, rank, True) @ coefficients
    assert jnp.allclose(actual, reference, atol=2e-11, rtol=2e-11)
    # Odd coefficient signs reflect the output, for arbitrary signed spectra.
    reflected = eqx.filter_jit(geometry.decode)(
        coefficients * (-1.0) ** jnp.arange(rank), True
    )
    assert jnp.allclose(reflected, actual[::-1], atol=2e-12, rtol=2e-12)
