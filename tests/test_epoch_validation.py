"""Epoch domain checks must survive compilation of identity actions."""

# pattern: Functional Core
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import drifft

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("throw", [False, True])
@pytest.mark.parametrize("strength", [-0.01, 0.2, float("nan"), float("inf")])
def test_zero_generation_epoch_rejects_invalid_selection(strength, throw):
    """Domain violations raise even when no transition consumes the strength."""
    backend = drifft.Jacobi(8, max_strength=0.1)
    method = drifft.RationalKrylov()
    initial = jnp.ones(9) / 9
    with pytest.raises(eqx.EquinoxRuntimeError, match="selection strength"):
        method.advance(
            backend,
            initial,
            frequency_map=drifft.Underdominance(strength),
            generations=0,
            throw=throw,
        ).distribution.block_until_ready()


@pytest.mark.parametrize("rate", [-0.01, float("nan"), float("inf")])
def test_identity_demography_rejects_invalid_mutation_rate(rate):
    """An empty history retains the mutation-rate domain check under JIT."""
    model = drifft.DemographicTransition(
        drifft.Demography([8], []),
        backend=drifft.FFT,
        frequency_map=drifft.Neutral(),
    )
    initial = jnp.ones(15) / 15
    with pytest.raises(eqx.EquinoxRuntimeError, match="mutation_rate"):
        model.advance_segregating(
            initial, mutation_rate=rate
        ).distribution.block_until_ready()
