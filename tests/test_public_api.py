"""Public keyword signatures and result containers."""

# Functional Core: public API and JAX transformation contracts.
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import drifft

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("backend_type", [drifft.FFT, drifft.NUFFT])
@pytest.mark.parametrize("method_name", ["expect", "advance"])
@pytest.mark.parametrize("segregating", [False, True])
def test_direct_methods_compile_once_for_dynamic_inputs(
    backend_type, method_name, segregating
):
    """Cache direct method calls while keeping vector values dynamic."""
    traces = []

    def selection(frequencies):
        """Record Python tracing while returning a valid nonlinear selection map."""
        traces.append(1)
        return frequencies**2

    transition = drifft.Transition(backend_type(32), drifft.FrequencyMap(selection))
    method = getattr(transition, method_name)
    size = 31 if segregating else 33
    vector = jnp.linspace(0.1, 0.9, size)
    first = method(vector, segregating=segregating).block_until_ready()
    second = method(2 * vector, segregating=segregating).block_until_ready()
    assert len(traces) == 1
    np.testing.assert_allclose(second, 2 * first, atol=1e-12)


@pytest.mark.parametrize("operation", ["expect", "advance", "as_lineax"])
def test_transition_flags_reject_traced_booleans(operation):
    """Reject dynamic state-layout choices before Python shape branching."""
    transition = drifft.Transition(drifft.Dense(8), drifft.Neutral())
    vector = jnp.ones(7)

    @eqx.filter_jit
    def call(flag):
        """Supply a traced flag to each public transition boundary."""
        if operation == "as_lineax":
            return transition.as_lineax(segregating=flag).mv(vector)
        return getattr(transition, operation)(vector, segregating=flag)

    with pytest.raises(ValueError, match="segregating must be a Python bool"):
        call(jnp.asarray(True))
