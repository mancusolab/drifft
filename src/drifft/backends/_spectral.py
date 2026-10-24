"""Batched Bernstein sampling and endpoint-safe cosine-series evaluation."""

# pattern: Functional Core
# Linear array maps, with no cached dense sampling matrix.
from typing import TYPE_CHECKING, cast

import jax
import jax.core
import jax.numpy as jnp
from jax.experimental import hijax
from jax_finufft import nufft2

if TYPE_CHECKING:
    from .nufft import NUFFT


def _build_stencil(size: int, radius: int, probabilities: jax.Array):
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


def _sample_batches(nodes, values, *, size, radius, batch_size, transpose):
    """Matched gather/scatter maps with one live batch of regenerated weights."""
    padding = (-nodes.size) % batch_size
    batches = jnp.pad(nodes, (0, padding), constant_values=0.5).reshape(-1, batch_size)
    if transpose:
        mass = jnp.pad(values, (0, padding)).reshape(-1, batch_size)

        def scatter(accumulator, batch):
            """Scatter one node batch into the shared destination accumulator."""
            probabilities, weights_out = batch
            indices, weights = _build_stencil(size, radius, probabilities)
            updates = weights * weights_out[:, None]
            return accumulator.at[indices.reshape(-1)].add(updates.reshape(-1)), None

        # The only M-sized zero fill is outside the scan. Each batch scatters
        # directly into this carry, rather than adding a fresh dense cotangent.
        return jax.lax.scan(
            scatter, jnp.zeros(size + 1, dtype=values.dtype), (batches, mass)
        )[0]

    def gather(probabilities):
        """Evaluate one node batch from its regenerated binomial stencil."""
        indices, weights = _build_stencil(size, radius, probabilities)
        return jnp.sum(weights * values[indices], axis=-1)

    return jax.lax.map(gather, batches).reshape(-1)[: nodes.size]


class _BernsteinSampling(hijax.HiPrim):
    """Linear sampling with an explicit streaming adjoint and fixed geometry."""

    def __init__(self, nodes, values, *, size, radius, batch_size, transposed):
        """Declare input/output avals and static sampling geometry for Hijax."""
        self.in_avals = (nodes, values)
        self.out_aval = jax.core.ShapedArray(
            (size + 1,) if transposed else nodes.shape, values.dtype
        )
        self.params = dict(
            size=size, radius=radius, batch_size=batch_size, transposed=transposed
        )
        super().__init__()

    def expand(self, *args):
        """Lower sampling to batched gathers or streaming scatters."""
        nodes, values = args
        return _sample_batches(
            nodes,
            values,
            size=self.params["size"],
            radius=self.params["radius"],
            batch_size=self.params["batch_size"],
            transpose=self.params["transposed"],
        )

    def jvp(self, primals, tangents):
        """Differentiate values while rejecting trainable sampling geometry."""
        nodes, values = primals
        dnodes, dvalues = tangents
        if not isinstance(dnodes, hijax.Zero):
            raise ValueError("Bernstein sampling requires fixed, nontrainable nodes")
        return self(nodes, values), self(nodes, hijax.instantiate_zeros(dvalues))

    lin, linearized = hijax.linearize_from_jvp
    vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp

    def transpose(self, out_ct, *maybe_accums):
        """Accumulate the opposite sampling map into the values cotangent."""
        mass = out_ct
        nodes, values = maybe_accums
        if isinstance(nodes, hijax.GradAccum):
            raise ValueError("Bernstein sampling requires fixed, nontrainable nodes")
        if isinstance(values, hijax.GradAccum):
            values.accum(
                _sample(
                    nodes,
                    hijax.instantiate_zeros(mass),
                    **(self.params | {"transposed": not self.params["transposed"]}),
                )
            )

    def batch_dim_rule(self, axis_data, dims):
        """Place the mapped axis first; Hijax supplies the mapped expansion."""
        return 0


def _sample(nodes, values, **config):
    """Bind array avals so JAX retains sampling rules through nested transforms."""
    return _BernsteinSampling(jax.typeof(nodes), jax.typeof(values), **config)(
        nodes, values
    )


def sample_bernstein(backend: "NUFFT", values: jax.Array) -> jax.Array:
    """Sample fixed binomial nodes, with a matched streaming scatter transpose.

    The backend's nodes are nontrainable geometry; differentiating them is rejected.
    Input values support JIT, batching and differentiation in both directions.
    Neither direction retains a node-by-stencil matrix.
    """
    return _sample(
        backend.nodes,
        jnp.asarray(values, dtype=backend.nodes.dtype),
        size=backend.gene_copies,
        radius=backend.radius,
        batch_size=backend.batch_size,
        transposed=False,
    )


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


def _differentiate_coefficients(coefficients: jax.Array) -> jax.Array:
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


def _evaluate_nufft(coefficients, probabilities, tolerance, endpoints):
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


class _CosineTranspose(hijax.HiPrim):
    """Cosine adjoint whose probability derivatives avoid singular angles."""

    def __init__(self, probabilities, mass, *, count, tolerance):
        """Declare the coefficient and endpoint cotangent shapes."""
        self.in_avals = (probabilities, mass)
        self.out_aval = (
            jax.core.ShapedArray((count,), mass.dtype),
            jax.core.ShapedArray((2,), mass.dtype),
        )
        self.params = dict(count=count, tolerance=tolerance)
        super().__init__()

    def expand(self, *args):
        """Transpose the linear NUFFT value map at fixed probabilities."""
        probabilities, mass = args
        coefficients = jnp.zeros(cast(int, self.params["count"]), dtype=mass.dtype)
        endpoints = jnp.zeros(2, dtype=mass.dtype)
        return jax.linear_transpose(
            lambda a, e: _evaluate_nufft(
                a, probabilities, cast(float, self.params["tolerance"]), e
            ),
            coefficients,
            endpoints,
        )(mass)

    def jvp(self, primals, tangents):
        """Differentiate mass and probabilities using coefficient recurrences."""
        probabilities, mass = primals
        dprobabilities, dmass = map(hijax.instantiate_zeros, tangents)
        count, tolerance = (
            cast(int, self.params["count"]),
            cast(float, self.params["tolerance"]),
        )
        primal = self(probabilities, mass)
        dcoefficients, dendpoints = self(probabilities, dmass)
        if count > 1:
            derivative_mass = _transpose_cosine(
                probabilities, mass * dprobabilities, count - 1, tolerance
            )
            dcoefficients = (
                dcoefficients
                + jax.linear_transpose(
                    _differentiate_coefficients, jnp.zeros(count, dtype=mass.dtype)
                )(derivative_mass)[0]
            )
        return primal, (dcoefficients, dendpoints)

    lin, linearized = hijax.linearize_from_jvp
    vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp

    def transpose(self, out_ct, *maybe_accums):
        """Accumulate cosine evaluation as the adjoint of coefficient scattering."""
        cotangents = out_ct
        probabilities, mass = maybe_accums
        if isinstance(probabilities, hijax.GradAccum):
            raise ValueError("Cosine adjoint is linear only in mass")
        if isinstance(mass, hijax.GradAccum):
            coefficients, endpoints = map(hijax.instantiate_zeros, cotangents)
            mass.accum(
                _evaluate(
                    coefficients,
                    probabilities,
                    cast(float, self.params["tolerance"]),
                    endpoints,
                )
            )

    def batch_dim_rule(self, axis_data, dims):
        """Place the mapped axis first in both cotangent outputs."""
        return (0, 0)


def _transpose_evaluate(probabilities, mass, count, tolerance):
    """Bind the endpoint-safe adjoint, retaining its rules in higher derivatives."""
    return _CosineTranspose(
        jax.typeof(probabilities), jax.typeof(mass), count=count, tolerance=tolerance
    )(probabilities, mass)


def _transpose_cosine(probabilities, mass, count, tolerance):
    """Include the polynomial's endpoint sums in its coefficient transpose."""
    coefficients, endpoints = _transpose_evaluate(probabilities, mass, count, tolerance)
    return coefficients + endpoints[0] + endpoints[1] * (-1.0) ** jnp.arange(count)


class _CosineEvaluation(hijax.HiPrim):
    """Cosine evaluation with polynomial JVPs and explicit linear transposition.

    Transposing the raw angular implementation and then differentiating would
    expose singular endpoint coordinates. Keep the safe adjoint as a primitive.
    """

    def __init__(self, coefficients, probabilities, endpoints, *, tolerance):
        """Declare evaluation avals and the static FINUFFT tolerance."""
        self.in_avals = (coefficients, probabilities, endpoints)
        self.out_aval = jax.core.ShapedArray(probabilities.shape, coefficients.dtype)
        self.params = dict(tolerance=tolerance)
        super().__init__()

    def expand(self, *args):
        """Lower the primal to FINUFFT with exact endpoint substitutions."""
        coefficients, probabilities, endpoints = args
        return _evaluate_nufft(
            coefficients,
            probabilities,
            cast(float, self.params["tolerance"]),
            endpoints,
        )

    def jvp(self, primals, tangents):
        """Use polynomial derivatives instead of differentiating singular angles."""
        coefficients, probabilities, endpoints = primals
        dcoefficients, dprobabilities, dendpoints = map(
            hijax.instantiate_zeros, tangents
        )
        primal = self(coefficients, probabilities, endpoints)
        derivative = evaluate_cosine(
            _differentiate_coefficients(coefficients),
            probabilities,
            cast(float, self.params["tolerance"]),
        )
        tangent = (
            self(dcoefficients, probabilities, dendpoints) + dprobabilities * derivative
        )
        return primal, tangent

    lin, linearized = hijax.linearize_from_jvp
    vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp

    def transpose(self, out_ct, *maybe_accums):
        """Accumulate coefficient and endpoint cotangents at fixed probabilities."""
        mass = out_ct
        coefficients, probabilities, endpoints = maybe_accums
        if isinstance(probabilities, hijax.GradAccum):
            raise ValueError(
                "Cosine evaluation is linear only in coefficients/endpoints"
            )
        da, de = _transpose_evaluate(
            probabilities,
            hijax.instantiate_zeros(mass),
            self.in_avals[0].shape[0],
            cast(float, self.params["tolerance"]),
        )
        if isinstance(coefficients, hijax.GradAccum):
            coefficients.accum(da)
        if isinstance(endpoints, hijax.GradAccum):
            endpoints.accum(de)

    def batch_dim_rule(self, axis_data, dims):
        """Place the mapped axis first; Hijax handles the mapped expansion."""
        return 0


def _evaluate(coefficients, probabilities, tolerance, endpoints):
    """Bind cosine evaluation with finite first and higher endpoint derivatives."""
    return _CosineEvaluation(
        jax.typeof(coefficients),
        jax.typeof(probabilities),
        jax.typeof(endpoints),
        tolerance=tolerance,
    )(coefficients, probabilities, endpoints)


def evaluate_cosine(
    coefficients: jax.Array, probabilities: jax.Array, tolerance: float
) -> jax.Array:
    """Evaluate a cosine polynomial with finite first/higher endpoint derivatives."""
    signs = (-1.0) ** jnp.arange(coefficients.size)
    endpoints = jnp.stack((coefficients.sum(), coefficients @ signs))
    return _evaluate(coefficients, probabilities, tolerance, endpoints)
