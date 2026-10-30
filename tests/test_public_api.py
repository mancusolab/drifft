"""Public keyword signatures and result containers."""

# Functional Core: public API and JAX transformation contracts.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import drifft

jax.config.update("jax_enable_x64", True)


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
