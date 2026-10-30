"""Independent dense binomial references shared by evolution tests."""

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
