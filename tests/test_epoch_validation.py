"""Epoch domain checks must survive compilation of identity actions."""

# pattern: Functional Core
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import drifft

jax.config.update("jax_enable_x64", True)


# All invalid domains must raise with throw=False; one representative also
# checks the default throw=True path.
@pytest.mark.parametrize(
    "strength,throw",
    [
        (-0.01, False),
        (0.2, False),
        (float("nan"), False),
        (float("inf"), False),
        (-0.01, True),
    ],
)
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
        backend_factory=drifft.FFT,
        frequency_map=drifft.Neutral(),
    )
    initial = jnp.ones(15) / 15
    with pytest.raises(eqx.EquinoxRuntimeError, match="mutation_rate"):
        model.advance(
            initial, segregating=True, singleton_mutation_rate=rate
        ).distribution.block_until_ready()


def test_singleton_input_requires_segregating_states():
    """Keep singleton-source input distinct from full-state recurrent mutation."""
    model = drifft.DemographicTransition(
        drifft.Demography([4], []),
        backend_factory=drifft.Dense,
        frequency_map=drifft.Neutral(),
    )
    with pytest.raises(ValueError, match="requires segregating=True"):
        model.advance(jnp.ones(9), singleton_mutation_rate=0.0)
    with pytest.raises(ValueError, match="must be scalar"):
        model.advance(
            jnp.ones(7), segregating=True, singleton_mutation_rate=jnp.ones(2)
        )
    with pytest.raises(ValueError, match="Python bool"):
        model.advance(jnp.ones(9), segregating=1)  # ty: ignore[invalid-argument-type] -- unsupported flag
