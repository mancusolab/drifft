"""Selection-independent, host-side FFT geometry construction."""

# Functional Core: deterministic planning, no file or network I/O.
# NumPy/SciPy are intentional here: quantile-based windows and FFT buckets have
# variable host-side sizes. The factory converts the completed plan to JAX
# arrays; this module's construction routines never run inside a matvec or solve.
import math
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import DTypeLike
from scipy.special import expit, logit
from scipy.stats import binom


@final
class Block(eqx.Module):
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


def make_block(
    M: int, j0: int, lo: int, hi: int, rlo: int, rhi: int, *, balance: bool = True
) -> Block:
    """Construct exact DTD factors with recurrence and stable modal normalization."""
    if not (0 < j0 < M and 0 <= lo <= hi <= M and rlo <= rhi):
        raise ValueError("Invalid block intervals or reference count")
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
    # A constant balancing shift preserves the exact product.
    shift = (np.max(logb) + np.min(logb)) / 2 if balance else 0.0
    a = np.exp(logB + loga + shift).astype(np.float64)
    b = np.exp(logb - shift).astype(np.float64)
    L, R = hi - lo + 1, rhi - rlo + 1
    nfft = 1 << (R + 2 * L - 3).bit_length()
    d = np.arange(rlo - (hi - j0), rhi - (lo - j0) + 1, dtype=np.float64)
    kernel = np.exp(-float(h) * d * d / 2)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError(
            "Nonfinite local factors; use narrower blocks or a larger cutoff"
        )
    return Block(
        jnp.asarray(b),
        jnp.asarray(a),
        jnp.asarray(np.fft.rfft(kernel, nfft)),
        nfft,
        L,
        R,
    )


@final
class Bucket(eqx.Module):
    """Blocks sharing an FFT length, padded only within the bucket.

    `indices` and `b` have shape (blocks, max input length); `a`,
    `conv_indices` and `output_indices` have shape (blocks, max node count).
    Padded outputs have zero scale and scatter to the plan's sentinel node.
    """

    indices: jax.Array
    b: jax.Array
    a: jax.Array
    kernel_fft: jax.Array
    conv_indices: jax.Array
    output_indices: jax.Array
    nfft: int = eqx.field(static=True)


@final
class TransitionPlan(eqx.Module):
    r"""Immutable PyTree geometry constructed by `build_transition_plan`.

    **Arguments:**

    Use the factory rather than assembling this container manually. `edges`
    partitions block cores in log-odds; `eta0` and `variance` define each grid.
    `rmin` and `offsets` locate block nodes in a flat output array. `buckets`
    contains the batched factors and cached transforms. Remaining fields record
    boundary coefficients, configuration and the checked tail-mass diagnostic.

    **Returns:**

    A reusable selection-independent plan. Arrays are dynamic PyTree leaves;
    configuration and dimensions are static Equinox metadata. The static fields
    are retained for shape decisions and reproducible configuration reporting.

    **Raises:**

    `ValueError` if block metadata or boundary dimensions are inconsistent.
    The factory additionally validates configuration and requires float64.
    """

    buckets: tuple[Bucket, ...]
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
                "Plan block metadata must have matching one-dimensional shapes"
            )
        if (
            not 1 <= self.boundary_width <= self.gene_copies + 1
            or self.boundary_logchoose.shape != (self.boundary_width,)
        ):
            raise ValueError("boundary_width must match the boundary coefficient array")


def build_transition_plan(
    gene_copies: int,
    *,
    tail_tol: float = 1e-12,
    interp_order: int = 12,
    min_mode_count: int = 128,
    block_spacing: float = 1.0,
    balance: bool = True,
    dtype: DTypeLike = jnp.float64,
) -> TransitionPlan:
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

    A `TransitionPlan` with host-computed quantile windows and cached kernel FFTs.
    This factory is not JIT-compatible. Core intervals partition log-odds;
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
    groups: dict[int, list[tuple[int, Block, int]]] = {}
    offsets, rmins = [], []
    centers: list[int] = []
    if gene_copies > 2 * min_mode_count:
        centers = [min_mode_count]
        while centers[-1] < gene_copies - min_mode_count:
            j = centers[-1]
            centers.append(
                min(
                    gene_copies - min_mode_count,
                    j + max(1, int(block_spacing * np.sqrt(j * (1 - j / gene_copies)))),
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
        rlo = int(np.floor((core_edges[i] - eta0[i]) * variance[i])) - interp_order // 2
        rhi = (
            int(np.ceil((core_edges[i + 1] - eta0[i]) * variance[i]))
            + interp_order // 2
        )
        pmin, pmax = expit(eta0[i] + np.array([rlo, rhi]) / variance[i])
        lo = max(0, int(binom.ppf(tail_tol / 2, gene_copies, pmin)))
        hi = min(gene_copies, int(binom.isf(tail_tol / 2, gene_copies, pmax)))
        tail = max(
            float(
                binom.cdf(lo - 1, gene_copies, pmin) + binom.sf(hi, gene_copies, pmin)
            ),
            float(
                binom.cdf(lo - 1, gene_copies, pmax) + binom.sf(hi, gene_copies, pmax)
            ),
        )
        max_tail = max(max_tail, tail)
        block = make_block(gene_copies, j0, lo, hi, rlo, rhi, balance=balance)
        groups.setdefault(block.nfft, []).append((lo, block, total))
        offsets.append(total)
        rmins.append(rlo)
        total += block.rows
    buckets: list[Bucket] = []
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
            Bucket(
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
    return TransitionPlan(
        buckets=tuple(buckets),
        edges=jnp.asarray(edges),
        eta0=jnp.asarray(eta0),
        variance=jnp.asarray(variance),
        rmin=jnp.asarray(rmins, dtype=jnp.int32),
        offsets=jnp.asarray(offsets, dtype=jnp.int32),
        boundary_logchoose=jnp.asarray(logchoose),
        max_tail_mass=jnp.asarray(max_tail),
        gene_copies=gene_copies,
        tail_tol=tail_tol,
        interp_order=interp_order,
        min_mode_count=min_mode_count,
        block_spacing=block_spacing,
        balance=balance,
        total_nodes=total,
        boundary_width=width,
    )
