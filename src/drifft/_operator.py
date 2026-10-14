"""Pure JAX transition kernels."""

# Functional Core: array-only numerical kernels.
import math

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, xlog1py, xlogy

from ._plan import Block, TransitionPlan


def apply_block(block: Block, v: jax.Array) -> jax.Array:
    """Apply one block to its input window, returning its log-odds nodes."""
    transformed = jnp.fft.rfft(block.b * v, n=block.nfft)
    conv = jnp.fft.irfft(transformed * block.kernel_fft, n=block.nfft)
    return block.a * conv[block.length - 1 : block.length - 1 + block.rows]


def underdominance(x: jax.Array, s: jax.Array | float) -> jax.Array:
    r"""Compute $q_s(x)$ for symmetric underdominance.

    **Arguments:**

    - `x`: Allele frequencies in `[0, 1]`.
    - `s`: Selection coefficient in `[0, 1]`, broadcastable with `x`.

    **Returns:**

    Selected probabilities with the broadcast shape of `x` and `s`.

    **Failure Modes:**

    Value ranges are caller preconditions; no host validation or callbacks run.
    """
    return (x - s * x * (1 - x)) / (1 - 2 * s * x * (1 - x))


def _boundary(
    plan: TransitionPlan, p: jax.Array, boundary_values: jax.Array
) -> jax.Array:
    """Sum reflected boundary PMFs over pre-gathered lower/upper input windows."""
    low = jnp.minimum(p, 1 - p)
    j = jnp.arange(plan.boundary_width)
    # Keep unused endpoint derivatives finite; the exact absorbing value is
    # selected below, independently of these temporary probabilities.
    safe_low = jnp.where(low == 0, 0.5, low)
    counts = j.astype(jnp.float64)
    weights = jnp.exp(
        plan.boundary_logchoose
        + xlogy(counts, safe_low[:, None])
        + xlog1py(plan.gene_copies - counts, -safe_low[:, None])
    )
    values = boundary_values[(p > 0.5).astype(jnp.int32)]
    result = jnp.sum(weights * values, axis=-1)
    return jnp.where(low == 0, values[:, 0], result)


def apply_probabilities(plan: TransitionPlan, p: jax.Array, v: jax.Array) -> jax.Array:
    r"""Evaluate binomial expectations for an arbitrary probability map.

    **Arguments:**

    - `plan`: Selection-independent float64 plan.
    - `p`: One-dimensional success probabilities in `[0, 1]`.
    - `v`: Finite real input vector of shape `(M+1,)`, cast to float64.

    **Returns:**

    A float64 vector shaped like `p`. This is the extension point for other
    selection models. The map is linear in `v`; outputs are never clipped or
    renormalized. Wrap in `jax.jit` at the surrounding workflow boundary.

    **Raises:**

    `ValueError` for incompatible shapes, checked when tracing.

    **Failure Modes:**

    Finite inputs and valid probability ranges are caller preconditions. Tail
    tolerance excludes interpolation/roundoff error. Polynomial interpolation
    may yield small negative coefficients, and seams need not be differentiable.
    Results can differ at roundoff level across devices.
    """
    if v.shape != (plan.gene_copies + 1,) or p.ndim != 1:
        raise ValueError("Expected v.shape == (M+1,) and one-dimensional p")
    v, p = jnp.asarray(v, dtype=jnp.float64), jnp.asarray(p, dtype=jnp.float64)
    cutoff = min(plan.min_mode_count / plan.gene_copies, 0.5)
    is_boundary = (p <= cutoff) | (p >= 1 - cutoff)
    boundary_columns = jnp.arange(plan.boundary_width)
    boundary_values = jnp.stack(
        (v[boundary_columns], v[plan.gene_copies - boundary_columns])
    )

    # Batched map bounds per-batch forward temporary memory independently of M.
    # Transposition can retain O(M*boundary_width) residuals. Entire interior
    # batches skip the direct work; every branch remains linear in v.
    # Gather BEFORE the scan: automatic transposition then accumulates only
    # 2*boundary_width entries per batch rather than an M-length cotangent.
    def direct_batch(prob):
        boundary = (prob <= cutoff) | (prob >= 1 - cutoff)
        return jax.lax.cond(
            jnp.any(boundary),
            lambda: _boundary(plan, prob, boundary_values),
            lambda: jnp.zeros_like(prob),
        )

    batch_size = 64
    pad = (-p.size) % batch_size
    padded = jnp.pad(p, (0, pad), constant_values=0.5).reshape(-1, batch_size)
    direct = jax.lax.map(direct_batch, padded).reshape(-1)[: p.size]
    if not plan.buckets:
        return direct
    # One extra sentinel receives zero-scaled padding from every FFT bucket.
    nodes = jnp.zeros(plan.total_nodes + 1, dtype=v.dtype)
    for bucket in plan.buckets:
        u = bucket.b * v[bucket.indices]
        conv = jnp.fft.irfft(
            jnp.fft.rfft(u, n=bucket.nfft, axis=-1) * bucket.kernel_fft,
            n=bucket.nfft,
            axis=-1,
        )
        values = bucket.a * jnp.take_along_axis(conv, bucket.conv_indices, axis=-1)
        nodes = nodes.at[bucket.output_indices].add(values)
    # Only unused interpolation coordinates are clipped: actual boundary rows
    # still come from the direct sum, and no output or coefficient is clipped.
    safe_p = jnp.clip(p, cutoff, 1 - cutoff)
    eta = jnp.log(safe_p) - jnp.log1p(-safe_p)
    block = jnp.searchsorted(plan.edges, eta, side="right")
    r = (eta - plan.eta0[block]) * plan.variance[block]
    start = jnp.floor(r).astype(jnp.int32) - plan.interp_order // 2 + 1
    t = r - start
    k = jnp.arange(plan.interp_order)
    differences = k[:, None] - k[None, :]
    denominators = jnp.prod(
        jnp.where(jnp.eye(plan.interp_order, dtype=bool), 1, differences), axis=-1
    )
    # Lagrange basis product, with its own node omitted by the diagonal mask.
    # Weights depend on probabilities and geometry only, preserving linearity.
    factors = t[:, None, None] - jnp.asarray(k)[None, None, :]
    factors = jnp.where(
        jnp.eye(plan.interp_order, dtype=bool)[None, :, :], 1.0, factors
    )
    weights = jnp.prod(factors, axis=-1) / jnp.asarray(denominators)
    indices = (
        plan.offsets[block, None]
        + start[:, None]
        - plan.rmin[block, None]
        + jnp.asarray(k)
    )
    interpolated = jnp.sum(weights * nodes[indices], axis=-1)
    return jnp.where(is_boundary, direct, interpolated)


def expected_next_values(
    plan: TransitionPlan, selection_strength: jax.Array | float, values: jax.Array
) -> jax.Array:
    r"""Compute expected next-generation values for each current allele count.

    **Arguments:**

    - `plan`: Float64 FFT geometry for `M` gene copies.
    - `selection_strength`: Dynamic scalar selection coefficient in `[0, 1]`.
    - `values`: Finite real vector of length `M+1`, including absorbing boundaries.

    **Returns:**

    A float64 vector of length `M+1` with unchanged absorbing endpoint values.

    **Raises:**

    `ValueError` for an incompatible vector shape.

    **Failure Modes:**

    Inherits the probability-range and approximation contract of
    `apply_probabilities`. Keep x64 enabled when applying the plan.
    """
    x = jnp.arange(plan.gene_copies + 1, dtype=jnp.float64) / plan.gene_copies
    return apply_probabilities(plan, underdominance(x, selection_strength), values)


def expected_segregating_values(
    plan: TransitionPlan, selection_strength: jax.Array | float, values: jax.Array
) -> jax.Array:
    r"""Compute expected next-generation values with zero values at absorption.

    **Arguments:**

    - `plan`: Float64 FFT geometry for `M` gene copies.
    - `selection_strength`: Dynamic scalar selection coefficient in `[0, 1]`.
    - `values`: Finite real vector of length `M-1`; index zero represents state one.

    **Returns:**

    A float64 vector of length `M-1`, computed by zero-padding absorbing inputs
    and stripping absorbing output rows from the full operator.

    **Raises:**

    `ValueError` for an incompatible transient vector shape.

    **Failure Modes:**

    Inherits the input-range and approximation contract of `apply_probabilities`.
    The substochastic row sums intentionally retain probability lost to absorption.
    """
    if values.shape != (plan.gene_copies - 1,):
        raise ValueError("Expected transient values.shape == (M-1,)")
    return expected_next_values(plan, selection_strength, jnp.pad(values, (1, 1)))[1:-1]


def apply_p_banded(
    M: int, s: jax.Array | float, v: jax.Array, *, tail_tol: float = 1e-12
) -> jax.Array:
    r"""Apply an independent binomial reference with a Bernstein truncation band.

    **Arguments:**

    - `M`: Integer gene-copy count, at least two; static under JIT.
    - `s`: Dynamic scalar selection coefficient in `[0, 1]`.
    - `v`: Finite real vector of length `M+1`.
    - `tail_tol`: Discarded mass budget in `(0, 1)`; static under JIT.

    **Returns:**

    A vector of length `M+1`. Work scales as $M^{3/2}$; batches bound temporary
    memory. PMFs are recomputed on every application without cached band weights.

    **Raises:**

    `ValueError` for invalid dimensions or truncation tolerance.

    **Failure Modes:**

    Selection range and finite inputs are caller preconditions. Log-PMF rounding
    is additional to the truncation budget. Enable x64 for reference accuracy.
    """
    if v.shape != (M + 1,):
        raise ValueError("Expected v.shape == (M+1,)")
    if M < 2 or not 0 < tail_tol < 1:
        raise ValueError("M >= 2 and 0 < tail_tol < 1 are required")
    # These Python scalars determine array shapes and must remain static in JIT.
    leps = math.log(2 / tail_tol)
    radius = min(M, math.ceil(math.sqrt(M / 2 * leps) + 2 / 3 * leps) + 1)
    offsets = jnp.arange(-radius, radius + 1)
    p = underdominance(jnp.arange(M + 1, dtype=jnp.float64) / M, s)

    def batch(prob):
        indices = jnp.floor(M * prob[:, None]).astype(jnp.int32) + offsets
        j = jnp.clip(indices, 0, M)
        logs = (
            gammaln(jnp.asarray(M + 1.0, dtype=jnp.float64))
            - gammaln(j + 1.0)
            - gammaln(M - j + 1.0)
            + xlogy(j.astype(jnp.float64), prob[:, None])
            + xlog1py((M - j).astype(jnp.float64), -prob[:, None])
        )
        weights = jnp.where((indices >= 0) & (indices <= M), jnp.exp(logs), 0.0)
        return jnp.sum(weights * v[j], axis=-1)

    pad = (-(M + 1)) % 64
    return jax.lax.map(
        batch, jnp.pad(p, (0, pad), constant_values=0.5).reshape(-1, 64)
    ).reshape(-1)[: M + 1]
