"""Selection-independent, host-side FFT geometry construction."""

# pattern: Functional Core
# Deterministic planning, no file or network I/O.
# NumPy/SciPy are intentional here: quantile-based windows and FFT buckets have
# variable host-side sizes. The constructor converts the prepared geometry to JAX
# arrays; this module's construction routines never run inside a matvec or solve.
import math
from typing import cast, final

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax.experimental import hijax
from jax.scipy.special import xlog1py, xlogy
from numpy.typing import DTypeLike
from scipy.special import expit, logit
from scipy.stats import binom

from ._base import AbstractTransitionBackend


@final
class _FFTBlock(eqx.Module):
    """One local factorization with inclusive input and output intervals.

    `b` scales input columns; `a` includes the modal probability and scales
    output nodes. `kernel_fft` represents the zero-padded convolution kernel.
    Shapes and transform lengths are static; numerical arrays are PyTree leaves.
    """

    b: jax.Array
    a: jax.Array
    kernel_fft: jax.Array
    nfft: int = eqx.field(static=True)
    length: int = eqx.field(static=True)
    rows: int = eqx.field(static=True)


def _stirling_correction(n: int) -> float:
    """Return the log-factorial correction beyond leading Stirling terms."""
    # n is a host geometry count; math computes constants before JAX application.
    if n < 16:
        return (
            math.lgamma(n + 1)
            - (n + 0.5) * math.log(n)
            + n
            - 0.5 * math.log(2 * math.pi)
        )
    # Below 16 the asymptotic series is less accurate; above it, the omitted
    # n**-11 term is below float64 precision for the modal normalization.
    z = 1.0 / n
    return z * (
        1 / 12
        + z * z * (-1 / 360 + z * z * (1 / 1260 + z * z * (-1 / 1680 + z * z / 1188)))
    )


def _build_block(
    M: int, j0: int, lo: int, hi: int, rlo: int, rhi: int, *, balance: bool = True
) -> _FFTBlock:
    """Construct exact DTD factors with recurrence and stable modal normalization."""
    if not (0 < j0 < M and 0 <= lo <= hi <= M and rlo <= rhi):
        raise ValueError("Invalid block intervals or reference count")
    # Build variable-sized blocks on the host, using NumPy's longdouble where
    # available for cancellation-sensitive recurrences; store float64 JAX arrays.
    p = np.longdouble(j0) / M
    eta = np.log(p) - np.log1p(-p)
    h = 1 / (M * p * (1 - p))
    r = np.arange(rlo, rhi + 1, dtype=np.longdouble)
    delta = h * r
    loga = delta * j0 - M * np.log1p(p * np.expm1(delta)) + h * r**2 / 2
    # Build the recurrence through zero even when the requested window excludes it.
    left, right = min(lo, j0), max(hi, j0)
    j = np.arange(left, right, dtype=np.longdouble)
    increments = np.log(M - j) - np.log(j + 1) + eta + h * (j - j0 + 0.5)
    cumulative = np.r_[np.longdouble(0), np.cumsum(increments)]
    logb = (cumulative - cumulative[j0 - left])[lo - left : hi - left + 1]
    logB = (
        -0.5 * np.log(2 * np.pi * M * p * (1 - p))
        + _stirling_correction(M)
        - _stirling_correction(j0)
        - _stirling_correction(M - j0)
    )
    # Opposite shifts in log(a) and log(b) preserve the exact product while
    # centering log(b)'s range, limiting large intermediate diagonal factors.
    shift = (np.max(logb) + np.min(logb)) / 2 if balance else 0.0
    a = np.exp(logB + loga + shift).astype(np.float64)
    b = np.exp(logb - shift).astype(np.float64)
    L, R = hi - lo + 1, rhi - rlo + 1
    nfft = 1 << (R + 2 * L - 3).bit_length()
    # The kernel has R+L-1 entries; convolution with L inputs needs R+2L-2
    # slots. Padding to at least that length prevents circular wraparound.
    d = np.arange(rlo - (hi - j0), rhi - (lo - j0) + 1, dtype=np.float64)
    kernel = np.exp(-float(h) * d * d / 2)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError(
            "Nonfinite local factors; use narrower blocks or a larger cutoff"
        )
    return _FFTBlock(
        jnp.asarray(b),
        jnp.asarray(a),
        jnp.asarray(np.fft.rfft(kernel, nfft)),
        nfft,
        L,
        R,
    )


@final
class _FFTBucket(eqx.Module):
    """Blocks sharing an FFT length, padded only within the bucket.

    `indices` and `b` have shape (blocks, max input length); `a`,
    `conv_indices` and `output_indices` have shape (blocks, max node count).
    Padded outputs have zero scale and scatter to the backend's sentinel node.
    """

    indices: jax.Array
    b: jax.Array
    a: jax.Array
    kernel_fft: jax.Array
    conv_indices: jax.Array
    output_indices: jax.Array
    nfft: int = eqx.field(static=True)


@final
class FFT(AbstractTransitionBackend):
    r"""Binomial expectations through local FFT convolutions.

    For destination values $v_j$ and gene-copy count $M$, approximate
    $B_M[v](q)=\sum_{j=0}^M\binom{M}{j}q^j(1-q)^{M-j}v_j$.
    On each local log-odds grid, the retained binomial kernel factors as

    $$y_r=a_r\sum_j e^{-h(r-(j-j_0))^2/2}\,b_jv_j.$$

    Here $j_0$ is the reference allele count, $h$ is the log-odds spacing,
    and $a_r,b_j$ are diagonal scaling weights. The sum is a convolution,
    evaluated by zero-padded FFTs; local polynomial interpolation supplies
    expectations at arbitrary $q$. Near zero and one, direct binomial sums
    replace log-odds interpolation.

    For offspring probabilities $q_i$, `Transition` uses
    $(Pv)_i=B_M[v](q_i)$ for expectations and the transpose $P^Tp$ for forward
    mass. The approximation comes from tail truncation and interpolation;
    the Gaussian-shaped convolution above is a factorization of the binomial
    kernel, not a normal approximation to it.
    """

    buckets: tuple[_FFTBucket, ...]
    edges: jax.Array
    eta0: jax.Array
    variance: jax.Array
    rmin: jax.Array
    offsets: jax.Array
    boundary_logchoose: jax.Array
    max_tail_mass: jax.Array
    gene_copies: int = eqx.field(static=True)
    tail_tol: float = eqx.field(static=True)
    interp_order: int = eqx.field(static=True)
    min_mode_count: int = eqx.field(static=True)
    block_spacing: float = eqx.field(static=True)
    balance: bool = eqx.field(static=True)
    total_nodes: int = eqx.field(static=True)
    boundary_width: int = eqx.field(static=True)

    def __init__(
        self,
        gene_copies: int,
        *,
        tail_tol: float = 1e-12,
        interp_order: int = 12,
        min_mode_count: int = 128,
        block_spacing: float = 1.0,
        balance: bool = True,
        dtype: DTypeLike = jnp.float64,
    ) -> None:
        r"""Plan local binomial FFTs over all success probabilities, independent of selection.

        **Arguments:**

        - `gene_copies`: Integer gene-copy count, at least two (diploid population size is `gene_copies/2`).
        - `tail_tol`: Discarded mass budget in `[1e-15, .1)`, default `1e-12`.
        - `interp_order`: Even polynomial stencil size from 2 through 16, default 12.
        - `min_mode_count`: Boundary cutoff in expected copies, at least 8, default 128.
        - `block_spacing`: Center separation in local standard deviations, `(0, 4]`.
        - `balance`: Whether to recenter the diagonal multipliers, default true.
        - `dtype`: Must be float64, with `jax_enable_x64` already enabled.

        **Returns:**

        `None`. Initializes host-computed quantile windows and cached kernel FFTs.
        Construction is not JIT-compatible. Core intervals partition log-odds;
        halos support interpolation. For `gene_copies <= 2*min_mode_count` all rows use the
        direct fallback. Tail tolerance does not bound interpolation or roundoff.

        **Raises:**

        `ValueError` for invalid configuration, disabled x64 or nonfinite factors;
        `TypeError` if `balance` is not boolean. Results are deterministic for a
        fixed environment but need not be bitwise identical across platforms.
        """
        if (
            not isinstance(gene_copies, int)
            or isinstance(gene_copies, bool)
            or gene_copies < 2
        ):
            raise ValueError("gene_copies must be an integer >= 2")
        # These are host configuration checks, not operations on traced values.
        # NumPy also handles dtype inspection before geometry becomes JAX arrays.
        if not (np.isfinite(tail_tol) and 1e-15 <= tail_tol < 0.1):
            raise ValueError("tail_tol must be finite and in [1e-15, 0.1)")
        if (
            not isinstance(interp_order, int)
            or interp_order < 2
            or interp_order > 16
            or interp_order % 2
        ):
            raise ValueError("interp_order must be an even integer from 2 to 16")
        if not isinstance(min_mode_count, int) or min_mode_count < 8:
            raise ValueError("min_mode_count must be an integer >= 8")
        if not (np.isfinite(block_spacing) and 0 < block_spacing <= 4):
            raise ValueError("block_spacing must be in (0, 4]")
        if not isinstance(balance, bool):
            raise TypeError("balance must be a bool")
        if np.dtype(dtype) != np.dtype("float64") or not jax.config.x64_enabled:
            raise ValueError("Enable jax_enable_x64 and use dtype=float64")
        cutoff = min_mode_count / gene_copies
        groups: dict[int, list[tuple[int, _FFTBlock, int]]] = {}
        offsets, rmins = [], []
        centers: list[int] = []
        if gene_copies > 2 * min_mode_count:
            centers = [min_mode_count]
            while centers[-1] < gene_copies - min_mode_count:
                j = centers[-1]
                centers.append(
                    min(
                        gene_copies - min_mode_count,
                        j
                        + max(
                            1, int(block_spacing * np.sqrt(j * (1 - j / gene_copies)))
                        ),
                    )
                )
        eta0 = logit(np.asarray(centers, dtype=float) / gene_copies)
        variance = np.asarray(centers, dtype=float) * (
            1 - np.asarray(centers, dtype=float) / gene_copies
        )
        edges = (eta0[:-1] + eta0[1:]) / 2
        core_edges = (
            np.r_[logit(cutoff), edges, logit(1 - cutoff)] if centers else np.array([])
        )
        total, max_tail = 0, 0.0
        for i, j0 in enumerate(centers):
            rlo = (
                int(np.floor((core_edges[i] - eta0[i]) * variance[i]))
                - interp_order // 2
            )
            rhi = (
                int(np.ceil((core_edges[i + 1] - eta0[i]) * variance[i]))
                + interp_order // 2
            )
            pmin, pmax = expit(eta0[i] + np.array([rlo, rhi]) / variance[i])
            lo = max(0, int(binom.ppf(tail_tol / 2, gene_copies, pmin)))
            hi = min(gene_copies, int(binom.isf(tail_tol / 2, gene_copies, pmax)))
            tail = max(
                float(
                    binom.cdf(lo - 1, gene_copies, pmin)
                    + binom.sf(hi, gene_copies, pmin)
                ),
                float(
                    binom.cdf(lo - 1, gene_copies, pmax)
                    + binom.sf(hi, gene_copies, pmax)
                ),
            )
            max_tail = max(max_tail, tail)
            block = _build_block(gene_copies, j0, lo, hi, rlo, rhi, balance=balance)
            groups.setdefault(block.nfft, []).append((lo, block, total))
            offsets.append(total)
            rmins.append(rlo)
            total += block.rows
        buckets: list[_FFTBucket] = []
        for nfft, blocks in sorted(groups.items()):
            block_count = len(blocks)
            input_length = max(block.length for _, block, _ in blocks)
            node_count = max(block.rows for _, block, _ in blocks)
            indices = np.zeros((block_count, input_length), dtype=np.int32)
            input_scale = np.zeros((block_count, input_length))
            output_scale = np.zeros((block_count, node_count))
            conv_indices = np.zeros((block_count, node_count), dtype=np.int32)
            # The final flat node is a sentinel: padded rows never alias real nodes.
            output_indices = np.full((block_count, node_count), total, dtype=np.int32)
            for k, (lo, block, offset) in enumerate(blocks):
                indices[k, : block.length] = np.arange(lo, lo + block.length)
                input_scale[k, : block.length] = block.b
                output_scale[k, : block.rows] = block.a
                conv_indices[k, : block.rows] = np.arange(
                    block.length - 1, block.length - 1 + block.rows
                )
                output_indices[k, : block.rows] = np.arange(offset, offset + block.rows)
            buckets.append(
                _FFTBucket(
                    indices=jnp.asarray(indices),
                    b=jnp.asarray(input_scale),
                    a=jnp.asarray(output_scale),
                    kernel_fft=jnp.stack([block.kernel_fft for _, block, _ in blocks]),
                    conv_indices=jnp.asarray(conv_indices),
                    output_indices=jnp.asarray(output_indices),
                    nfft=nfft,
                )
            )
        # Stochastic monotonicity makes the upper quantile at the cutoff sufficient
        # for all boundary probabilities. Reflect the upper boundary onto the lower.
        width = min(
            gene_copies + 1, int(binom.isf(tail_tol, gene_copies, min(cutoff, 0.5))) + 1
        )
        j = np.arange(width, dtype=float)
        # Recurrence is accurate for small boundary counts, even at very large gene_copies.
        logchoose = np.r_[0.0, np.cumsum(np.log(gene_copies - j[:-1]) - np.log(j[1:]))]
        if not centers:
            max_tail = float(binom.sf(width - 1, gene_copies, 0.5))
        else:
            max_tail = max(max_tail, float(binom.sf(width - 1, gene_copies, cutoff)))
        self.buckets = tuple(buckets)
        self.edges = jnp.asarray(edges)
        self.eta0 = jnp.asarray(eta0)
        self.variance = jnp.asarray(variance)
        self.rmin = jnp.asarray(rmins, dtype=jnp.int32)
        self.offsets = jnp.asarray(offsets, dtype=jnp.int32)
        self.boundary_logchoose = jnp.asarray(logchoose)
        self.max_tail_mass = jnp.asarray(max_tail)
        self.gene_copies = gene_copies
        self.tail_tol = tail_tol
        self.interp_order = interp_order
        self.min_mode_count = min_mode_count
        self.block_spacing = block_spacing
        self.balance = balance
        self.total_nodes = total
        self.boundary_width = width

    def __check_init__(self) -> None:
        """Check structural invariants without inspecting traced array values."""
        block_count = self.eta0.size
        if (
            self.eta0.ndim != 1
            or self.variance.shape != (block_count,)
            or self.rmin.shape != (block_count,)
            or self.offsets.shape != (block_count,)
            or self.edges.shape != (max(0, block_count - 1),)
        ):
            raise ValueError(
                "Backend block metadata must have matching one-dimensional shapes"
            )
        if (
            not 1 <= self.boundary_width <= self.gene_copies + 1
            or self.boundary_logchoose.shape != (self.boundary_width,)
        ):
            raise ValueError("boundary_width must match the boundary coefficient array")

    def expect(self, probabilities: jax.Array, values: jax.Array) -> jax.Array:
        r"""Compute binomial expectations using local FFTs.

        **Arguments:**

        - `probabilities`: Float64 vector of finite source probabilities in `[0, 1]`.
        - `values`: Finite float64 destination vector of length `gene_copies + 1`.

        **Returns:**

        A float64 vector with the same shape as `probabilities`, linear in `values`.

        **Raises:**

        `ValueError` for incompatible vector shapes, including during JIT tracing.

        Finiteness and probability ranges follow the unchecked preconditions of
        `AbstractTransitionBackend.expect`; signed values are supported.
        """
        values = jnp.asarray(values, dtype=jnp.float64)
        if values.shape != (self.gene_copies + 1,) or probabilities.ndim != 1:
            raise ValueError(
                "Expected values.shape == (M+1,) and one-dimensional probabilities"
            )
        probabilities = jnp.asarray(probabilities, dtype=jnp.float64)
        cutoff = min(self.min_mode_count / self.gene_copies, 0.5)
        is_boundary = (probabilities <= cutoff) | (probabilities >= 1 - cutoff)
        boundary_columns = jnp.arange(self.boundary_width)
        boundary_values = jnp.stack(
            (values[boundary_columns], values[self.gene_copies - boundary_columns])
        )

        # Batched map bounds per-batch forward temporary memory independently of M.
        # Transposition can retain O(M*boundary_width) residuals. Entire interior
        # batches skip the direct work; every branch remains linear in values.
        # Gather BEFORE the scan: automatic transposition then accumulates only
        # 2*boundary_width entries per batch rather than an M-length cotangent.
        def expect_boundary_batch(prob):
            """Skip the direct sum when the batch contains only interior rows."""
            boundary = (prob <= cutoff) | (prob >= 1 - cutoff)
            return jax.lax.cond(
                jnp.any(boundary),
                lambda: _expect_boundary(self, prob, boundary_values),
                lambda: jnp.zeros_like(prob),
            )

        batch_size = 64
        pad = (-probabilities.size) % batch_size
        padded = jnp.pad(probabilities, (0, pad), constant_values=0.5).reshape(
            -1, batch_size
        )
        direct = jax.lax.map(expect_boundary_batch, padded).reshape(-1)[
            : probabilities.size
        ]
        if not self.buckets:
            return direct
        # One extra sentinel receives zero-scaled padding from every FFT bucket.
        nodes = jnp.zeros(self.total_nodes + 1, dtype=values.dtype)
        for bucket in self.buckets:
            scaled_values = bucket.b * values[bucket.indices]
            convolved = jnp.fft.irfft(
                jnp.fft.rfft(scaled_values, n=bucket.nfft, axis=-1) * bucket.kernel_fft,
                n=bucket.nfft,
                axis=-1,
            )
            node_values = bucket.a * jnp.take_along_axis(
                convolved, bucket.conv_indices, axis=-1
            )
            nodes = nodes.at[bucket.output_indices].add(node_values)
        # Only unused interpolation coordinates are clipped: actual boundary rows
        # still come from the direct sum, and no output or coefficient is clipped.
        safe_probabilities = jnp.clip(probabilities, cutoff, 1 - cutoff)
        eta = jnp.log(safe_probabilities) - jnp.log1p(-safe_probabilities)
        block = jnp.searchsorted(self.edges, eta, side="right")
        r = (eta - self.eta0[block]) * self.variance[block]
        start = jnp.floor(r).astype(jnp.int32) - self.interp_order // 2 + 1
        t = r - start
        k = jnp.arange(self.interp_order)
        differences = k[:, None] - k[None, :]
        denominators = jnp.prod(
            jnp.where(jnp.eye(self.interp_order, dtype=bool), 1, differences), axis=-1
        )
        # Lagrange basis product, with its own node omitted by the diagonal mask.
        # Weights depend on probabilities and geometry only, preserving linearity.
        factors = t[:, None, None] - jnp.asarray(k)[None, None, :]
        factors = jnp.where(
            jnp.eye(self.interp_order, dtype=bool)[None, :, :], 1.0, factors
        )
        weights = jnp.prod(factors, axis=-1) / jnp.asarray(denominators)
        indices = (
            self.offsets[block, None]
            + start[:, None]
            - self.rmin[block, None]
            + jnp.asarray(k)
        )
        interpolated = jnp.sum(weights * nodes[indices], axis=-1)
        return jnp.where(is_boundary, direct, interpolated)


class _BinomialWeights(hijax.HiPrim):
    """Endpoint-safe binomial weights with polynomial probability derivatives.

    Counts are integer indices and size is static. Probability and log-weight
    derivatives retain their existing rules. The operation is nonlinear in
    both arguments, so reverse mode transposes the JVP's linearization rather
    than the weight evaluation itself.
    """

    def __init__(self, counts, logchoose, probabilities, *, size):
        """Declare the row-by-count output and static binomial degree."""
        self.in_avals = (counts, logchoose, probabilities)
        self.out_aval = jax.core.ShapedArray(
            (probabilities.shape[0], counts.shape[0]), probabilities.dtype
        )
        self.params = dict(size=size)
        super().__init__()

    def expand(self, *args):
        """Evaluate weights with exact endpoint values and zero out-of-range indices."""
        counts, logchoose, probabilities = args
        size = cast(int, self.params["size"])
        endpoint = (probabilities == 0) | (probabilities == 1)
        safe = jnp.where(endpoint, 0.5, probabilities)[:, None]
        weights = jnp.exp(
            logchoose + xlogy(counts, safe) + xlog1py(size - counts, -safe)
        )
        weights = jnp.where(
            probabilities[:, None] == 0,
            counts == 0,
            jnp.where(probabilities[:, None] == 1, counts == size, weights),
        )
        return jnp.where((counts >= 0) & (counts <= size), weights, 0.0)

    def jvp(self, primals, tangents):
        """Apply the lower-degree binomial identity, including at endpoints."""
        counts, logchoose, probabilities = primals
        _, dlogchoose, dprobabilities = tangents
        dlogchoose, dprobabilities = map(
            hijax.instantiate_zeros, (dlogchoose, dprobabilities)
        )
        size = cast(int, self.params["size"])
        weights = self(counts, logchoose, probabilities)
        if size == 0:
            derivative = jnp.zeros_like(weights)
        else:
            # d B[n,j]/dp = n*(B[n-1,j-1] - B[n-1,j]). Reusing this
            # rule recursively also gives finite, matched higher derivatives.
            # size is static, so math.log supplies a trace-time constant.
            left_logchoose = (
                logchoose + jnp.log(jnp.maximum(counts, 1)) - math.log(size)
            )
            right_logchoose = (
                logchoose + jnp.log(jnp.maximum(size - counts, 1)) - math.log(size)
            )
            derivative = size * (
                _binomial_weights(size - 1, counts - 1, left_logchoose, probabilities)
                - _binomial_weights(size - 1, counts, right_logchoose, probabilities)
            )
        return weights, weights * dlogchoose + derivative * dprobabilities[:, None]

    lin, linearized = hijax.linearize_from_jvp
    vjp_fwd, vjp_bwd_retval = hijax.vjp_from_jvp

    def batch_dim_rule(self, axis_data, dims):
        """Keep tangent-only batching from marking the primal output as mapped."""
        return 0 if any(dim is not None for dim in dims) else None


def _binomial_weights(size, counts, logchoose, probabilities):
    """Bind endpoint-safe weights with rules preserved through nested transforms."""
    return _BinomialWeights(
        jax.typeof(counts),
        jax.typeof(logchoose),
        jax.typeof(probabilities),
        size=size,
    )(counts, logchoose, probabilities)


def _expect_boundary(
    backend: FFT, p: jax.Array, boundary_values: jax.Array
) -> jax.Array:
    """Sum reflected boundary PMFs over pre-gathered lower/upper input windows."""
    # At .5 choose one reflection branch rather than averaging opposite slopes.
    low = jnp.where(p > 0.5, 1 - p, p)
    weights = _binomial_weights(
        backend.gene_copies,
        jnp.arange(backend.boundary_width),
        backend.boundary_logchoose,
        low,
    )
    values = boundary_values[(p > 0.5).astype(jnp.int32)]
    return jnp.sum(weights * values, axis=-1)
