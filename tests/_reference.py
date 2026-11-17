"""Independent binomial references shared by backend and evolution tests."""

from decimal import Context, Decimal, localcontext

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln, xlog1py, xlogy


def decimal_binomial_rows(M: int, probabilities) -> tuple[np.ndarray, np.ndarray]:
    """Return independent binomial rows and angular derivatives at 60 digits.

    Decimal avoids mistaking reference PMF/log-factorial error for a backend's
    accuracy floor on platforms where longdouble is only binary64. The angular
    derivative is evaluated in high precision before conversion to float64.
    Inputs are a nonnegative integer degree and a finite one-dimensional vector
    of probabilities in [0,1]; invalid inputs raise ValueError. Both outputs
    have shape (len(probabilities), M+1), including empty query vectors. The
    Decimal context is isolated from caller rounding and trap settings.
    """
    if isinstance(M, bool) or not isinstance(M, int) or M < 0:
        raise ValueError("reference degree must be a nonnegative Python integer")
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or not np.all(
        np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)
    ):
        raise ValueError("probabilities must be a finite vector in [0, 1]")
    rows = np.zeros((len(probabilities), M + 1))
    grads = np.zeros_like(rows)
    with localcontext(Context(prec=60)):
        for index, probability in enumerate(probabilities):
            q = Decimal(float(probability))
            if q == 0 or q == 1:
                rows[index, 0 if q == 0 else M] = 1
                continue
            odds = q / (1 - q)
            mode = int((M + 1) * q)
            weights = [Decimal(0)] * (M + 1)
            weights[mode] = Decimal(1)
            for j in range(mode, M):
                weights[j + 1] = weights[j] * (M - j) / ((j + 1) / odds)
            for j in range(mode, 0, -1):
                weights[j - 1] = weights[j] * j / ((M - j + 1) * odds)
            total = sum(weights)
            variance_root = (q * (1 - q)).sqrt()
            rows[index] = [float(w / total) for w in weights]
            grads[index] = [
                float(w / total * (j - M * q) / variance_root)
                for j, w in enumerate(weights)
            ]
    return rows, grads


def dense_transition(source_pop_size, dest_pop_size, strength):
    """Build a binomial reference independently of production backends and maps."""
    frequency = jnp.arange(2 * source_pop_size + 1) / (2 * source_pop_size)
    # Absorbing rows are exact and independent of selection; differentiate only
    # interior probabilities to avoid log derivatives at zero and one.
    x = frequency[1:-1]
    probability = ((x - strength * x * (1 - x)) / (1 - 2 * strength * x * (1 - x)))[
        :, None
    ]
    copies = 2 * dest_pop_size
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
