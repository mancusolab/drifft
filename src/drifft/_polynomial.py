"""Polynomial kernels independent of population geometry and transition models.

Gegenbauer values are orthonormal for weight 3(1-z²)/4 on [-1, 1], with
rank counting degrees 0 through rank-1. Streaming moments retain only adjacent
polynomial degrees; their derivative order is with respect to z.

Cosine coefficients represent sum_k c[k] T_k(1-2q). Differentiation includes
the -2 coordinate factor. Backend-specific evaluators own their acceleration
and endpoint substitutions; these coefficient operations do not depend on them.
"""

from typing import cast

import jax
import jax.core
import jax.numpy as jnp
from jax.experimental import hijax


def gegenbauer_beta(degree):
    """Coefficient of p[n+1] in z*p[n] for the normalized Gegenbauer basis."""
    return jnp.sqrt((degree + 1) * (degree + 3) / ((2 * degree + 3) * (2 * degree + 5)))


def gegenbauer_values(points: jax.Array, rank: int) -> jax.Array:
    """Evaluate normalized Gegenbauer polynomials for weight 3(1-z²)/4."""
    first = jnp.ones_like(points)
    if rank == 1:
        return first[..., None]
    second = jnp.sqrt(5.0) * points

    def step(carry, degree):
        """Advance the orthonormal three-term recurrence."""
        previous, current = carry
        lower = gegenbauer_beta(degree - 1)
        upper = gegenbauer_beta(degree)
        following = (points * current - lower * previous) / upper
        return (current, following), following

    _, rest = jax.lax.scan(
        step, (first, second), jnp.arange(1, rank - 1, dtype=points.dtype)
    )
    return jnp.moveaxis(jnp.concatenate((first[None], second[None], rest)), 0, -1)


def _moment_recurrence(points, values, *, rank, order, transposed):
    """Stream moments or evaluations at a fixed polynomial derivative order."""
    # Row k carries the kth derivative; only adjacent degrees are retained.
    first = jnp.zeros((order + 1, points.size), dtype=points.dtype).at[0].set(1)
    second = jnp.zeros_like(first).at[0].set(jnp.sqrt(5.0) * points)
    if order:
        second = second.at[1].set(jnp.sqrt(5.0))
    if transposed:
        total = values[0] * first[order]
        if rank > 1:
            total = total + values[1] * second[order]
    else:
        total = jnp.zeros_like(points)
    if rank == 1:
        return total if transposed else jnp.array([values @ first[order]])

    def step(carry, degree):
        previous, current, total = carry
        lower = gegenbauer_beta(degree - 1)
        upper = gegenbauer_beta(degree)
        product = points * current
        if order:
            # (z p)^(k) = z p^(k) + k p^(k-1).
            product = product.at[1:].add(
                jnp.arange(1, order + 1)[:, None] * current[:-1]
            )
        following = (product - lower * previous) / upper
        if transposed:
            total = total + values[degree + 1] * following[order]
            moment = None
        else:
            moment = values @ following[order]
        return (current, following, total), moment

    (_, _, total), rest = jax.lax.scan(
        step, (first, second, total), jnp.arange(1, rank - 1)
    )
    if transposed:
        return total
    return jnp.concatenate(
        (jnp.array([values @ first[order], values @ second[order]]), rest)
    )


class _GegenbauerMoments(hijax.HiPrim):
    """Streaming polynomial moments and their matched evaluation transpose."""

    def __init__(self, points, values, *, rank, order, transposed):
        """Declare array shapes and static recurrence configuration."""
        self.in_avals = (points, values)
        self.out_aval = jax.core.ShapedArray(
            points.shape if transposed else (rank,), values.dtype
        )
        self.params = dict(rank=rank, order=order, transposed=transposed)
        super().__init__()

    def expand(self, *args):
        """Evaluate moments or their transpose without storing a polynomial matrix."""
        return _moment_recurrence(*args, **self.params)

    def jvp(self, primals, tangents):
        """Differentiate points and values using polynomial derivative recurrences."""
        points, values = primals
        dpoints, dvalues = map(hijax.instantiate_zeros, tangents)
        primal = self(points, values)
        tangent = self(points, dvalues)
        config = self.params | {"order": cast(int, self.params["order"]) + 1}
        if self.params["transposed"]:
            tangent = tangent + dpoints * gegenbauer_moments(points, values, **config)
        else:
            tangent = tangent + gegenbauer_moments(points, values * dpoints, **config)
        return primal, tangent

    lin, linearized = hijax.linearize_from_jvp
    vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp

    def transpose(self, out_ct, *maybe_accums):
        """Transpose only the linear values argument at fixed polynomial points."""
        points, values = maybe_accums
        if isinstance(points, hijax.GradAccum):
            raise ValueError("Gegenbauer moments are linear only in values")
        if isinstance(values, hijax.GradAccum):
            values.accum(
                gegenbauer_moments(
                    points,
                    hijax.instantiate_zeros(out_ct),
                    **(self.params | {"transposed": not self.params["transposed"]}),
                )
            )

    def batch_dim_rule(self, axis_data, dims):
        """Keep the mapped axis first in moments and evaluations."""
        # An unmapped primal must stay unmapped when only its tangent is batched.
        return 0 if any(dim is not None for dim in dims) else None


def gegenbauer_moments(points, values, *, rank, order=0, transposed=False):
    """Bind the streaming recurrence with rules preserved under nested transforms."""
    return _GegenbauerMoments(
        jax.typeof(points),
        jax.typeof(values),
        rank=rank,
        order=order,
        transposed=transposed,
    )(points, values)


def compute_cosine_coefficients(samples: jax.Array) -> jax.Array:
    """DCT-I through an even real-FFT extension, with both endpoint modes halved."""
    degree = samples.size - 1
    extended = jnp.concatenate((samples, samples[-2:0:-1]))
    coefficients = jnp.fft.rfft(extended).real / degree
    return coefficients.at[0].multiply(0.5).at[-1].multiply(0.5)


def filter_cosine_coefficients(
    coefficients: jax.Array, width: int, endpoints: jax.Array
) -> jax.Array:
    """Taper DCT-I modes and restore endpoints by a global linear correction.

    Drop the terminal (zero-weight) mode. For x=1-2q, endpoint residuals
    e0,e1 add (e0+e1)/2 to c0 and (e0-e1)/2 to c1. Differentiation and
    transposition therefore see the same corrected polynomial everywhere.
    """
    modes = coefficients.size - 1
    taper = jnp.minimum(1.0, (modes - jnp.arange(modes)) / width)
    filtered = coefficients[:-1] * taper
    residual = endpoints - jnp.stack(
        (filtered.sum(), filtered @ ((-1.0) ** jnp.arange(modes)))
    )
    return (
        filtered.at[0]
        .add((residual[0] + residual[1]) / 2)
        .at[1]
        .add((residual[0] - residual[1]) / 2)
    )


def differentiate_cosine_coefficients(coefficients: jax.Array) -> jax.Array:
    """Differentiate with respect to q, where the Chebyshev coordinate is 1-2q."""
    degree = coefficients.size - 1
    if degree == 0:
        return jnp.zeros_like(coefficients)
    weighted = 2 * jnp.arange(1, degree + 1) * coefficients[1:]
    # The reverse recurrence separates into even and odd suffix sums. This
    # avoids updating an L-length carry L times, including in its transpose.
    derivative = jnp.zeros(degree, dtype=coefficients.dtype)
    derivative = derivative.at[::2].set(jnp.cumsum(weighted[::2][::-1])[::-1])
    derivative = derivative.at[1::2].set(jnp.cumsum(weighted[1::2][::-1])[::-1])
    return -2 * derivative.at[0].multiply(0.5)
