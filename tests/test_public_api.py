"""Public keyword signatures and result containers."""

# Functional Core: public API and JAX transformation contracts.
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import binom

import drifft

jax.config.update("jax_enable_x64", True)


def test_transition_api_keywords_under_jit():
    """Preserve keyword calls and segregating semantics under filtered JIT."""
    backend = drifft.NUFFT(gene_copies=512)
    assert isinstance(backend, drifft.NUFFT)
    assert backend.gene_copies == 512
    values = jnp.linspace(-1.0, 1.0, 513)
    model = drifft.Transition(
        backend, frequency_map=drifft.Underdominance(strength=0.1)
    )
    actual = model.expect(values=values)
    frequency = np.arange(513) / 512
    probabilities = (frequency - 0.1 * frequency * (1 - frequency)) / (
        1 - 0.2 * frequency * (1 - frequency)
    )
    transition = binom.pmf(np.arange(513)[None, :], 512, probabilities[:, None])
    np.testing.assert_allclose(actual, transition @ values, atol=1e-12)
    segregating = model.expect_segregating(values=values[1:-1])
    np.testing.assert_allclose(
        segregating, transition[1:-1, 1:-1] @ values[1:-1], atol=1e-12
    )


def test_expected_visits_result_fields_survive_pytree_roundtrip():
    """Retain occupation values and diagnostics through PyTree flattening."""
    backend = drifft.NUFFT(gene_copies=16)
    result = drifft.Transition(
        backend, drifft.Underdominance(jnp.asarray(0.1))
    ).expected_visits()
    assert isinstance(result, drifft.ExpectedVisitsResult)
    assert bool(result.converged)
    assert result.visits.shape == (15,)
    rebuilt = drifft.ExpectedVisitsResult(
        visits=result.visits,
        status=result.status,
        stats=result.stats,
        residual_norm=result.residual_norm,
        converged=result.converged,
    )
    leaves, structure = jax.tree.flatten(rebuilt)
    restored = jax.tree.unflatten(structure, leaves)
    np.testing.assert_array_equal(restored.visits, result.visits)


@pytest.mark.parametrize("backend_type", [drifft.FFT, drifft.NUFFT])
@pytest.mark.parametrize(
    "method_name", ["expect", "advance", "expect_segregating", "advance_segregating"]
)
def test_direct_methods_compile_once_for_dynamic_inputs(backend_type, method_name):
    """Cache direct method calls while keeping vector values dynamic."""
    traces = []

    def selection(frequencies):
        """Record Python tracing while returning a valid nonlinear selection map."""
        traces.append(1)
        return frequencies**2

    transition = drifft.Transition(backend_type(32), drifft.FrequencyMap(selection))
    method = getattr(transition, method_name)
    size = 31 if method_name.endswith("segregating") else 33
    vector = jnp.linspace(0.1, 0.9, size)
    first = method(vector).block_until_ready()
    second = method(2 * vector).block_until_ready()
    assert len(traces) == 1
    np.testing.assert_allclose(second, 2 * first, atol=1e-12)
