"""Batched Bernstein sampling and endpoint-safe cosine-series evaluation."""

# Functional Core: linear array maps, with no cached dense sampling matrix.
from functools import partial

import jax
import jax.core
import jax.numpy as jnp
from jax.extend import core
from jax.interpreters import ad, batching, mlir
from jax_finufft import nufft2

from ._plan import TransitionPlan


def _stencil(size: int, radius: int, probabilities: jax.Array):
    """Generate normalized mode-relative weights for one fixed-size node batch."""
    low = jnp.minimum(probabilities, 1 - probabilities)
    safe = jnp.where(low == 0, 0.5, low)[:, None]
    mode = jnp.floor((size + 1) * safe).astype(jnp.int32)
    distance = jnp.arange(radius)
    up = jnp.clip(mode + distance, 0, size - 1)
    down = jnp.clip(mode - distance, 1, size)
    forward = (size - up) * safe / ((up + 1) * (1 - safe))
    backward = down * (1 - safe) / ((size - down + 1) * safe)
    # Ratios start at the mode, so retained weights stay <= 1 without subtracting
    # large log-factorials or large prefix sums. Mask before normalization.
    right = jnp.exp(jnp.cumsum(jnp.log(forward), axis=-1))
    left = jnp.exp(jnp.cumsum(jnp.log(backward), axis=-1))
    weights = jnp.concatenate((left[:, ::-1], jnp.ones_like(safe), right), axis=-1)
    indices = mode + jnp.arange(-radius, radius + 1)
    weights = jnp.where((indices >= 0) & (indices <= size), weights, 0)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    indices = jnp.clip(indices, 0, size)
    indices = jnp.where(probabilities[:, None] > 0.5, size - indices, indices)
    # Exact endpoint samples; unused safe=.5 weights cannot contaminate them.
    indices = jnp.where(
        low[:, None] == 0, jnp.where(probabilities[:, None] == 0, 0, size), indices
    )
    weights = jnp.where(
        low[:, None] == 0, (jnp.arange(2 * radius + 1) == radius)[None, :], weights
    )
    return indices, weights


def _raw_sample(nodes, values, *, size, radius, batch_size, transpose):
    """Matched gather/scatter maps with one live batch of regenerated weights."""
    padding = (-nodes.size) % batch_size
    batches = jnp.pad(nodes, (0, padding), constant_values=0.5).reshape(-1, batch_size)
    if transpose:
        mass = jnp.pad(values, (0, padding)).reshape(-1, batch_size)

        def scatter(accumulator, batch):
            probabilities, weights_out = batch
            indices, weights = _stencil(size, radius, probabilities)
            updates = weights * weights_out[:, None]
            return accumulator.at[indices.reshape(-1)].add(updates.reshape(-1)), None

        # The only M-sized zero fill is outside the scan. Each batch scatters
        # directly into this carry, rather than adding a fresh dense cotangent.
        return jax.lax.scan(
            scatter, jnp.zeros(size + 1, dtype=values.dtype), (batches, mass)
        )[0]

    def gather(probabilities):
        indices, weights = _stencil(size, radius, probabilities)
        return jnp.sum(weights * values[indices], axis=-1)

    return jax.lax.map(gather, batches).reshape(-1)[: nodes.size]


_sample_p = core.Primitive("drifft_bernstein_sample")
_sample_p.def_impl(_raw_sample)
_sample_p.def_abstract_eval(
    lambda nodes, values, *, size, radius, batch_size, transpose: jax.core.ShapedArray(
        (size + 1,) if transpose else nodes.shape, values.dtype
    )
)


def _sample_lowering(ctx, *args, **config):
    return mlir.lower_fun(partial(_raw_sample, **config), multiple_results=False)(
        ctx, *args
    )


mlir.register_lowering(_sample_p, _sample_lowering)


def _sample_jvp(primals, tangents, **config):
    nodes, values = primals
    dnodes, dvalues = tangents
    if not isinstance(dnodes, ad.Zero):
        raise ValueError("Bernstein sampling requires fixed, nontrainable nodes")
    return (
        _sample_p.bind(nodes, values, **config),
        _sample_p.bind(nodes, ad.instantiate_zeros(dvalues), **config),
    )


def _sample_transpose(mass, nodes, values, *, transpose, **config):
    if ad.is_undefined_primal(nodes):
        raise ValueError("Bernstein sampling requires fixed, nontrainable nodes")
    if not ad.is_undefined_primal(values):
        return None, None
    return None, _sample_p.bind(
        nodes, ad.instantiate_zeros(mass), transpose=not transpose, **config
    )


def _sample_batch(arguments, axes, **config):
    count = next(
        arg.shape[axis]
        for arg, axis in zip(arguments, axes, strict=True)
        if axis is not None
    )
    arrays = tuple(
        jnp.broadcast_to(arg, (count,) + arg.shape)
        if axis is None
        else jnp.moveaxis(arg, axis, 0)
        for arg, axis in zip(arguments, axes, strict=True)
    )
    return jax.lax.map(lambda xs: _sample_p.bind(*xs, **config), arrays), 0


ad.primitive_jvps[_sample_p] = _sample_jvp
ad.primitive_transposes[_sample_p] = _sample_transpose
batching.primitive_batchers[_sample_p] = _sample_batch


def sample_bernstein(plan: TransitionPlan, values: jax.Array) -> jax.Array:
    """Sample fixed binomial nodes, with a matched streaming scatter transpose.

    The plan's nodes are nontrainable geometry; differentiating them is rejected.
    Input values support JIT, batching and differentiation in both directions.
    Neither direction retains a node-by-stencil matrix.
    """
    return _sample_p.bind(
        plan.nodes,
        jnp.asarray(values, dtype=plan.nodes.dtype),
        size=plan.gene_copies,
        radius=plan.radius,
        batch_size=plan.batch_size,
        transpose=False,
    )


def cosine_coefficients(samples: jax.Array) -> jax.Array:
    """DCT-I through an even real-FFT extension, with both endpoint modes halved."""
    degree = samples.size - 1
    extended = jnp.concatenate((samples, samples[-2:0:-1]))
    coefficients = jnp.fft.rfft(extended).real / degree
    return coefficients.at[0].multiply(0.5).at[-1].multiply(0.5)


def _derivative_coefficients(coefficients: jax.Array) -> jax.Array:
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


def _raw_evaluate(coefficients, probabilities, tolerance, endpoints):
    """NUFFT primal with exact endpoint values and a polynomial derivative rule."""
    if coefficients.size == 1:
        result = jnp.broadcast_to(coefficients[0], probabilities.shape)
    else:
        theta = 2 * jnp.arcsin(jnp.sqrt(probabilities))
        modes = jnp.concatenate(
            (coefficients[:0:-1] / 2, coefficients[:1], coefficients[1:] / 2)
        )
        result = nufft2(modes.astype(jnp.complex128), theta, eps=tolerance).real
    return jnp.where(
        probabilities == 0,
        endpoints[0],
        jnp.where(probabilities == 1, endpoints[1], result),
    )


@partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def _transpose_evaluate(probabilities, mass, count, tolerance):
    """Transpose the NUFFT and exact endpoint rows, with safe mixed derivatives."""
    coefficients = jnp.zeros(count, dtype=jnp.float64)
    endpoints = jnp.zeros(2, dtype=jnp.float64)
    return jax.linear_transpose(
        lambda a, e: _raw_evaluate(a, probabilities, tolerance, e),
        coefficients,
        endpoints,
    )(mass)


def _transpose_cosine(probabilities, mass, count, tolerance):
    """Include the polynomial's endpoint sums in its coefficient transpose."""
    coefficients, endpoints = _transpose_evaluate(probabilities, mass, count, tolerance)
    return coefficients + endpoints[0] + endpoints[1] * (-1.0) ** jnp.arange(count)


@_transpose_evaluate.defjvp
def _transpose_evaluate_jvp(count, tolerance, primals, tangents):
    probabilities, mass = primals
    dprobabilities, dmass = tangents
    primal = _transpose_evaluate(probabilities, mass, count, tolerance)
    dcoefficients, dendpoints = _transpose_evaluate(
        probabilities, dmass, count, tolerance
    )
    if count > 1:
        derivative_mass = _transpose_cosine(
            probabilities, mass * dprobabilities, count - 1, tolerance
        )
        dcoefficients = (
            dcoefficients
            + jax.linear_transpose(
                _derivative_coefficients, jnp.zeros(count, dtype=jnp.float64)
            )(derivative_mass)[0]
        )
    return primal, (dcoefficients, dendpoints)


# The primitive gives automatic transposition an explicit endpoint-safe rule.
# A custom_jvp alone is insufficient: linear_transpose can transpose its primal
# and differentiating that transpose would expose the singular angular map.
_evaluate_p = core.Primitive("drifft_cosine_evaluate")
_evaluate_p.def_impl(lambda a, q, e, *, tolerance: _raw_evaluate(a, q, tolerance, e))
_evaluate_p.def_abstract_eval(
    lambda a, q, e, *, tolerance: jax.core.ShapedArray(q.shape, a.dtype)
)
mlir.register_lowering(
    _evaluate_p,
    lambda ctx, *args, **params: mlir.lower_fun(
        lambda a, q, e: _raw_evaluate(a, q, params["tolerance"], e),
        multiple_results=False,
    )(ctx, *args),
)


def _evaluate(coefficients, probabilities, tolerance, endpoints):
    """Bind a linear cosine evaluation with polynomial first/higher derivatives."""
    return _evaluate_p.bind(coefficients, probabilities, endpoints, tolerance=tolerance)


def _evaluate_jvp(primals, tangents, *, tolerance):
    coefficients, probabilities, endpoints = primals
    dcoefficients, dprobabilities, dendpoints = map(ad.instantiate_zeros, tangents)
    primal = _evaluate(coefficients, probabilities, tolerance, endpoints)
    derivative = evaluate_cosine(
        _derivative_coefficients(coefficients), probabilities, tolerance
    )
    tangent = (
        _evaluate(dcoefficients, probabilities, tolerance, dendpoints)
        + dprobabilities * derivative
    )
    return primal, tangent


def _evaluate_transpose(mass, coefficients, probabilities, endpoints, *, tolerance):
    if ad.is_undefined_primal(probabilities):
        raise ValueError("Cosine evaluation is linear only in coefficients/endpoints")
    mass = ad.instantiate_zeros(mass)
    count = (
        coefficients.aval.shape[0]
        if ad.is_undefined_primal(coefficients)
        else coefficients.size
    )
    da, de = _transpose_evaluate(probabilities, mass, count, tolerance)
    return (
        da if ad.is_undefined_primal(coefficients) else None,
        None,
        de if ad.is_undefined_primal(endpoints) else None,
    )


def _evaluate_batch(arguments, axes, *, tolerance):
    size = next(
        arg.shape[axis]
        for arg, axis in zip(arguments, axes, strict=True)
        if axis is not None
    )
    arrays = tuple(
        jnp.broadcast_to(arg, (size,) + arg.shape)
        if axis is None
        else jnp.moveaxis(arg, axis, 0)
        for arg, axis in zip(arguments, axes, strict=True)
    )
    return jax.lax.map(lambda xs: _evaluate(xs[0], xs[1], tolerance, xs[2]), arrays), 0


ad.primitive_jvps[_evaluate_p] = _evaluate_jvp
ad.primitive_transposes[_evaluate_p] = _evaluate_transpose
batching.primitive_batchers[_evaluate_p] = _evaluate_batch


def evaluate_cosine(
    coefficients: jax.Array, probabilities: jax.Array, tolerance: float
) -> jax.Array:
    """Evaluate a cosine polynomial with finite first/higher endpoint derivatives."""
    signs = (-1.0) ** jnp.arange(coefficients.size)
    endpoints = jnp.stack((coefficients.sum(), coefficients @ signs))
    return _evaluate(coefficients, probabilities, tolerance, endpoints)


def expect_probabilities(
    plan: TransitionPlan, probabilities: jax.Array, values: jax.Array
) -> jax.Array:
    """Apply the fixed approximate matrix with exact absorbing endpoint rows."""
    if values.shape != (plan.gene_copies + 1,) or probabilities.ndim != 1:
        raise ValueError(
            "Expected values.shape == (M+1,) and one-dimensional probabilities"
        )
    coefficients = cosine_coefficients(sample_bernstein(plan, values))
    return _evaluate(
        coefficients,
        probabilities,
        plan.nufft_tol,
        values[jnp.array([0, plan.gene_copies])],
    )
