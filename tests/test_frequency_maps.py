"""Frequency-map composition, mutation, and differentiation contracts."""

# pattern: Functional Core
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import drifft

jax.config.update("jax_enable_x64", True)


def test_frequency_map_interface_and_composition_order():
    """Expose apply and callable shorthand with explicit left-to-right order."""
    frequencies = jnp.array([0.0, 0.2, 0.7, 1.0])
    selection = drifft.Underdominance(0.3)
    mutation = drifft.Mutation(forward_rate=0.1, backward_rate=0.2)
    custom = drifft.FrequencyMap(lambda x: x**2)
    for model in (drifft.Neutral(), selection, mutation, custom):
        assert isinstance(model, drifft.AbstractFrequencyMap)
        assert jnp.allclose(model.apply(frequencies), model(frequencies))
    assert jnp.array_equal(drifft.Neutral().apply(frequencies), frequencies)
    assert jnp.allclose(custom.apply(frequencies), frequencies**2)
    selected = selection.apply(frequencies)
    expected = selected * 0.8 + (1 - selected) * 0.1
    combined = drifft.Composition(selection, mutation)
    assert jnp.allclose(combined.apply(frequencies), expected)
    assert jnp.allclose(combined(frequencies), expected)
    assert not jnp.allclose(
        drifft.Composition(mutation, selection)(frequencies), expected
    )


@pytest.mark.parametrize("backend", [drifft.FFT, drifft.NUFFT])
def test_mutation_reintroduces_alleles_from_both_endpoints(backend):
    """Forward mutation escapes loss and back mutation escapes fixation."""
    transition = drifft.Transition(backend(16), frequency_map=drifft.Mutation(0.1, 0.2))
    initial = jnp.eye(17)[jnp.array([0, 16])]
    actual = eqx.filter_vmap(transition.advance)(initial)
    assert jnp.allclose(actual.sum(axis=1), 1, atol=1e-11)
    assert jnp.allclose(actual @ jnp.arange(17), jnp.array([1.6, 12.8]), atol=1e-10)
    assert jnp.all(actual[:, 1:-1].sum(axis=1) > 0)


@pytest.mark.parametrize("backend", [drifft.FFT, drifft.NUFFT])
def test_demographic_mutation_gradients(backend):
    """Keep both mutation rates dynamic through constant-size and changing-size steps."""
    model = drifft.DemographicTransition(
        drifft.Demography([8, 5], [3]),
        backend=backend,
        frequency_map=drifft.Composition(drifft.Neutral(), drifft.Mutation(0.1, 0.2)),
    )
    initial = jnp.eye(17)[4]
    frequencies = jnp.arange(11) / 10

    def objective(rates):
        """Reuse geometry while replacing the two mutation-rate leaves."""
        updated = eqx.tree_at(
            lambda m: (
                m.frequency_map.maps[1].forward_rate,
                m.frequency_map.maps[1].backward_rate,
            ),
            model,
            (rates[0], rates[1]),
        )
        return updated.advance(initial).distribution @ frequencies

    def reference(rates):
        """Propagate the exact expected frequency under neutral mutation."""
        frequency = 0.25
        for _ in range(3):
            frequency = frequency * (1 - rates[1]) + (1 - frequency) * rates[0]
        return frequency

    rates = jnp.array([0.1, 0.2])
    assert jnp.allclose(objective(rates), reference(rates), atol=1e-10)
    assert jnp.allclose(
        jax.jit(jax.grad(objective))(rates), jax.grad(reference)(rates), atol=1e-9
    )


def test_frequency_map_configuration_validation():
    """Reject malformed components and nonscalar rate parameters."""
    with pytest.raises(ValueError, match="scalar"):
        drifft.Mutation(jnp.array([0.1, 0.2]), 0.1)
    with pytest.raises(TypeError):
        drifft.FrequencyMap(None)  # ty: ignore[invalid-argument-type] -- invalid wrapper
    with pytest.raises(ValueError):
        drifft.Composition()
    with pytest.raises(TypeError):
        drifft.Composition(lambda x: x)  # ty: ignore[invalid-argument-type] -- unwrapped function
    backend = drifft.FFT(16)
    with pytest.raises(TypeError):
        drifft.Transition(backend, frequency_map=lambda x: x)  # ty: ignore[invalid-argument-type] -- unwrapped function
    with pytest.raises(TypeError):
        drifft.DemographicTransition(
            drifft.Demography([8, 8], [1]),
            backend=drifft.FFT,
            frequency_map=lambda x: x,  # ty: ignore[invalid-argument-type] -- unwrapped function
        )
