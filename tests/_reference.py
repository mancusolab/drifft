"""Independent dense binomial references shared by evolution tests."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, xlog1py, xlogy


def dense_transition(source_ne, destination_ne, strength):
    """Build a binomial reference independently of production backends and maps."""
    frequency = jnp.arange(2 * source_ne + 1) / (2 * source_ne)
    # Absorbing rows are exact and independent of selection; differentiate only
    # interior probabilities to avoid log derivatives at zero and one.
    x = frequency[1:-1]
    probability = ((x - strength * x * (1 - x)) / (1 - 2 * strength * x * (1 - x)))[
        :, None
    ]
    copies = 2 * destination_ne
    count = jnp.arange(copies + 1, dtype=jnp.float64)[None, :]
    interior = jnp.exp(
        gammaln(copies + 1)
        - gammaln(count + 1)
        - gammaln(copies - count + 1)
        + xlogy(count, probability)
        + xlog1py(copies - count, -probability)
    )
    endpoints = jnp.eye(copies + 1)[jnp.array([0, copies])]
    return jnp.concatenate([endpoints[:1], interior, endpoints[1:]])


def _segregating_history_objective(strength, mass, rate, intervals, values):
    """Propagate a dense killed chain with post-generation singleton sources."""
    for source, destination, duration in intervals:
        # The hold transition is unchanged within an interval. Build it once;
        # the recurrence remains independent of production power algorithms.
        hold = dense_transition(source, source, strength)[1:-1, 1:-1]
        for _ in range(duration - 1):
            mass = (mass @ hold).at[0].add(2 * source * rate)
        boundary = dense_transition(source, destination, strength)[1:-1, 1:-1]
        mass = (mass @ boundary).at[0].add(2 * destination * rate)
    return mass @ values


@eqx.filter_jit
def dense_history_derivatives(inputs, direction, intervals, values):
    """Reuse one compiled dense JVP/VJP oracle across backend and method cases.

    Biological inputs, directions, and rewards remain dynamic arrays. Only
    the integer interval schedule and input shapes/dtypes determine specialization.
    """

    def objective(strength, mass, rate):
        """Bind the fixed schedule while exposing all differentiable inputs."""
        return _segregating_history_objective(strength, mass, rate, intervals, values)

    gradient = jax.grad(objective, argnums=(0, 1, 2))(*inputs)
    tangent = jax.jvp(objective, inputs, direction)[1]
    return gradient, tangent
