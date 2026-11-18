"""Local binomial FFTs with host-side planning and differentiable JAX application."""

# pattern: Functional Core
# Deterministic planning, no file or network I/O.
# NumPy/SciPy are intentional here: quantile-based windows and FFT buckets have
# variable host-side sizes. The constructor converts the prepared geometry to JAX
# arrays; this module's construction routines never run inside a matvec or solve.
import math
from decimal import Context, Decimal, localcontext
from fractions import Fraction as F
from functools import lru_cache
from typing import NamedTuple, cast, final

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax.experimental import hijax
from jax.scipy.special import xlog1py, xlogy
from scipy.special import expit, logit
from scipy.stats import binom

from .._batching import BatchablePrimitive
from ._accuracy import (
    BackendErrorReport,
    ErrorComponent,
    _gradient_coordinate,
    _validate_atol,
    _validate_gradient,
)
from ._base import AbstractTransitionBackend


def _trim(p):
    """Remove trailing zeros from ascending power coefficients, retaining a constant."""
    while len(p) > 1 and p[-1] == 0:
        p = p[:-1]
    return p


def _add(a, b):
    """Add polynomials stored as ascending exact power coefficients."""
    return _trim(
        tuple(
            (a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
            for i in range(max(len(a), len(b)))
        )
    )


def _mul(a, b):
    """Multiply ascending power coefficients by exact discrete convolution."""
    out = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return _trim(tuple(out))


def _derivative(a):
    """Differentiate ascending power coefficients, including constant polynomials."""
    return tuple(i * a[i] for i in range(1, len(a))) or (F(0),)


@lru_cache(maxsize=128)
def _energy(M: int, n: int):
    """Return exact ascending coefficients of the binomial derivative energy H_n.

    With D=q(1-q)d/dq and binomial weights B_j, H_n=sum_j (D^n B_j)^2/B_j
    is a polynomial in v=q(1-q). Its square root bounds the row L1 derivative.
    """
    a = {1: (F(1),)}
    for order in range(1, n):
        nxt = {}
        for k in range(1, min(order + 1, M) + 1):
            p = a.get(k, (F(0),))
            if (order - k) % 2 == 0:
                term = _add(tuple(k * x for x in p), (0, *_derivative(p)))
            else:
                term = _add(_mul((k, -4 * k - 2), p), _mul((0, 1, -4), _derivative(p)))
            nxt[k] = _add(term, a.get(k - 1, (F(0),)))
        a = nxt
    result = (F(0),)
    falling = 1
    for k in range(1, min(n, M) + 1):
        falling *= M - k + 1
        p = _mul(a[k], a[k])
        if (n - k) % 2:
            p = _mul(p, (1, -4))
        result = _add(
            result, (F(0),) * k + tuple(math.factorial(k) * falling * x for x in p)
        )
    return result


def _bernstein(p, lo: F, hi: F):
    """Convert power coefficients to exact Bernstein coefficients on [lo,hi]."""
    n = len(p) - 1
    power = [
        sum(p[j] * math.comb(j, k) * lo ** (j - k) for j in range(k, n + 1))
        * (hi - lo) ** k
        for k in range(n + 1)
    ]
    return tuple(
        sum(power[k] * F(math.comb(j, k), math.comb(n, k)) for k in range(j + 1))
        for j in range(n + 1)
    )


def _split(b):
    """Bisect Bernstein coefficients with exact de Casteljau subdivision."""
    left, right = [b[0]], [b[-1]]
    while len(b) > 1:
        b = tuple((x + y) / 2 for x, y in zip(b[:-1], b[1:], strict=True))
        left.append(b[0])
        right.append(b[-1])
    return tuple(left), tuple(reversed(right))


def _range(p, lo: F, hi: F, depth: int = 3):
    """Enclose a polynomial on [lo,hi] using subdivided Bernstein coefficients."""
    pieces = [_bernstein(p, lo, hi)]
    for _ in range(depth):
        pieces = [child for b in pieces for child in _split(b)]
    return min(min(b) for b in pieces), max(max(b) for b in pieces)


@lru_cache(maxsize=128)
def _increasing_on_variance_domain(p):
    """Certify a nonnegative derivative on [0, 1/4] for a fixed polynomial."""
    return min(_bernstein(_derivative(p), F(0), F(1, 4))) >= 0


def _energy_upper(p, lo: F, hi: F):
    """Bound energy on a variance interval produced by _variance_interval."""
    if not _increasing_on_variance_domain(p):
        return _range(p, lo, hi)[1]
    # The maximum is attained at hi. Horner evaluation stays exact.
    value = p[-1]
    for coefficient in reversed(p[:-1]):
        value = value * hi + coefficient
    return value


def _logistic(x: F):
    """Enclose the halo endpoints with correctly rounded Decimal exponentials."""
    # Bound evaluation must not inherit the caller's rounding or trap settings.
    with localcontext(Context(prec=70)):
        d = Decimal(x.numerator) / Decimal(x.denominator)
        a = F(d.next_minus().exp().next_minus())
        b = F(d.next_plus().exp().next_plus())
    return a / (1 + a), b / (1 + b)


def _variance_interval(lo: F, hi: F):
    """Enclose q(1-q) on [lo,hi], rounding outwards to a dyadic grid."""
    lower = min(lo * (1 - lo), hi * (1 - hi))
    upper = F(1, 4) if lo <= F(1, 2) <= hi else max(lo * (1 - lo), hi * (1 - hi))
    # Outward dyadic rounding keeps subsequent exact polynomial arithmetic cheap.
    scale = 2**48
    return F(lower * scale // 1, scale), F(-(-upper * scale // 1), scale)


def _upper(x: F):
    """Round an exact rational upwards to binary64; allow infinity on overflow."""
    try:
        value = float(x)
    except OverflowError:
        return math.inf
    if math.isfinite(value) and F(value) < x:
        value = math.nextafter(value, math.inf)
    return value


def _sqrt_upper(x: F):
    """Round the square root of a nonnegative rational upwards to binary64."""
    value = math.sqrt(_upper(x))
    if math.isfinite(value) and F(value) ** 2 < x:
        value = math.nextafter(value, math.inf)
    return value


@lru_cache(maxsize=4096)
def _interpolation_bound(
    M: int,
    m: int,
    eta: float,
    inverse_step: float,
    rlo: int,
    rhi: int,
    gradient: bool = False,
) -> tuple[float, float, float]:
    """Bound value/gradient remainders and gradient amplification of node tails.

    The inclusive node halo has log-odds eta + r/inverse_step. Returned bounds
    use the angular coordinate before applying parameter sensitivity; the last
    two entries are zero when gradient planning is disabled.
    """
    V, e = F(inverse_step), F(eta)
    qlo = _logistic(e + rlo / V)[0]
    qhi = _logistic(e + rhi / V)[1]
    vlo, vhi = _variance_interval(qlo, qhi)
    energy = max(F(0), _energy_upper(_energy(M, m), vlo, vhi))
    derivative = F(_sqrt_upper(energy / V ** (2 * m)))
    alpha = F(math.comb(m, m // 2), 4**m)
    value = _upper(alpha * derivative)
    if not gradient:
        return value, 0.0, 0.0
    # Differentiated divided-difference bound (analysis eq. 4.7), not
    # differentiation of an unspecified scalar remainder point.
    next_energy = max(F(0), _energy_upper(_energy(M, m + 1), vlo, vhi))
    next_derivative = F(_sqrt_upper(next_energy / V ** (2 * (m + 1))))
    gamma = F(1, (m // 2) * math.comb(m, m // 2))
    # The halo contains the core. Its minimum variance gives a conservative
    # angular conversion over every selected branch, without query-time checks.
    min_variance = min(qlo * (1 - qlo), qhi * (1 - qhi))
    scale = F(_sqrt_upper(V * V / min_variance))
    grad = scale * (gamma * derivative + alpha * next_derivative / (m + 1))
    # sum |lambda'_k| <= 4 for every supported central stencil (tested exactly).
    return value, _upper(grad), _upper(4 * scale)


def _tail_bound(M: int, lo: int, hi: int, qlo: float, qhi: float) -> float:
    """Bound omitted mass over a probability interval, rejecting numerical failures."""
    lower = 0.0 if lo <= 0 else float(binom.cdf(lo - 1, M, qlo))
    upper = 0.0 if hi >= M else float(binom.sf(hi, M, qhi))
    # NaN comparisons otherwise let window loops stop and max(0, NaN) hide
    # a failed tail evaluation as an apparent zero error bound.
    if not (0 <= lower <= 1 and 0 <= upper <= 1):
        raise ValueError("invalid binomial tail probability during FFT planning")
    return lower + upper


def _direct_grad_bound(
    M: int, width: int, cutoff: float, sensitivity: float, tail: float
) -> float:
    """Uniform parameter-derivative tail error on a direct boundary branch."""
    if width > M or sensitivity == 0:
        return 0.0
    if width < M * cutoff:
        return sensitivity * math.sqrt(M * tail)
    # Omitted derivative entries have one sign, so their sum telescopes to
    # M*b_(M-1,W-1). Maximize its angular version analytically on [0,cutoff].
    q = min(cutoff, (width - 0.5) / M)
    log_bound = (
        math.log(sensitivity)
        + math.log(M)
        + 0.5 * (math.log(q) + math.log1p(-q))
        + float(binom.logpmf(width - 1, M - 1, q))
    )
    return (
        math.exp(log_bound) if log_bound < math.log(np.finfo(float).max) else math.inf
    )


def _fft_grid(
    M: int,
    cutoff_count: int,
    spacing: float,
    order: int,
    sensitivity: float = 0.0,
    *,
    atol: float = 1e-10,
    grad_atol: float | None = None,
):
    """Choose coarse interpolation grids over quarter-step convolution grids.

    H/h is a power of two, at most 64 before final-window curvature refinement.
    Each candidate includes its full stencil halo in both analytic bounds.
    The finest candidate can exceed the budgets; the caller then changes order
    or cutoff, or reports the explicit-order override.

    Returns (centers, log-odds centers, 1/H, core edges, coarse node intervals,
    strides) and the maximum value, parameter-gradient, and tail-amplification
    bounds. Direct-only geometry has empty arrays/lists and zero bounds.
    """
    centers = []
    if M > 2 * cutoff_count:
        centers = [cutoff_count]
        while centers[-1] < M - cutoff_count:
            j = centers[-1]
            centers.append(
                min(
                    M - cutoff_count,
                    j + max(1, int(spacing * np.sqrt(j * (1 - j / M)))),
                )
            )
    counts = np.asarray(centers, dtype=float)
    p = counts / M
    eta = logit(p)
    inverse_step = 4 * counts * (1 - p)
    edges = (eta[:-1] + eta[1:]) / 2
    core = (
        np.r_[logit(cutoff_count / M), edges, logit(1 - cutoff_count / M)]
        if centers
        else []
    )
    intervals, strides = [], []
    bounds = np.zeros(3)
    grad_budget = math.inf if grad_atol is None else grad_atol / 4
    for i in range(len(centers)):

        def candidate(stride, i=i):
            """Enclose one interpolation halo at the proposed dyadic spacing."""
            inverse = inverse_step[i] / stride
            lo = math.floor((core[i] - eta[i]) * inverse) - order // 2
            hi = math.ceil((core[i + 1] - eta[i]) * inverse) + order // 2
            bound = np.asarray(
                _interpolation_bound(
                    M, order, float(eta[i]), float(inverse), lo, hi, bool(sensitivity)
                )
            )
            bound[1:] *= sensitivity
            return lo, hi, bound

        stride = 1
        lo, hi, bound = candidate(stride)
        while stride < 64:
            next_lo, next_hi, next_bound = candidate(2 * stride)
            if next_bound[0] > atol / 4 or next_bound[1] > grad_budget:
                break
            stride *= 2
            lo, hi, bound = next_lo, next_hi, next_bound
        inverse_step[i] /= stride
        strides.append(stride)
        intervals.append((lo, hi))
        bounds = np.maximum(bounds, bound)
    return (centers, eta, inverse_step, edges, intervals, strides), tuple(
        map(float, bounds)
    )


@final
class _FFTBlock(NamedTuple):
    """One temporary host factorization with inclusive input and output intervals.

    `b` scales input columns; `a` includes the modal probability and scales
    output nodes. `kernel_fft` represents the zero-padded convolution kernel.
    Successive output nodes are `stride` convolution indices apart.
    Arrays remain in NumPy until blocks are packed into JAX buckets. This
    planning-only record is never retained in the runtime backend.
    """

    b: np.ndarray
    a: np.ndarray
    kernel_fft: np.ndarray
    nfft: int
    stride: int = 1

    @property
    def length(self) -> int:
        """Number of input columns in the block."""
        return self.b.shape[0]

    @property
    def rows(self) -> int:
        """Number of output nodes in the block."""
        return self.a.shape[0]


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


_EXPM1_REMAINDER = tuple(1 / math.factorial(k + 2) for k in range(26))
_LOG1P_REMAINDER = tuple((-1.0) ** (k + 1) / (k + 2) for k in range(64))


def _centered_cumulant(p: float, delta: np.ndarray) -> np.ndarray:
    """Evaluate log(1-p+p*exp(delta))-p*delta without first-order cancellation.

    For 0<p<1, reflect to p <= 1/2. Near zero, combine expm1(delta)-delta
    with log1p(t)-t, t=p*expm1(delta); both start quadratically. The Taylor
    tails on |delta|<=1 and |t|<=1/2 are below binary64 rounding. For |delta|>1
    use logaddexp directly: combining the exponential remainders would
    cancel large terms. Clip inactive series arguments to avoid overflow.
    """
    if p > 0.5:
        p, delta = 1 - p, -delta
    d = np.clip(delta, -1, 1)
    t = p * np.expm1(d)
    small_exp = d**2 * np.polynomial.polynomial.polyval(d, _EXPM1_REMAINDER)
    z = np.clip(t, -0.5, 0.5)
    small_log = z**2 * np.polynomial.polynomial.polyval(z, _LOG1P_REMAINDER)
    log_remainder = np.where(abs(t) <= 0.5, small_log, np.log1p(t) - t)
    small = p * small_exp + log_remainder
    large = np.logaddexp(np.log1p(-p), np.log(p) + delta) - p * delta
    return np.where(abs(delta) <= 1, small, large)


def _build_block(
    M: int,
    j0: int,
    lo: int,
    hi: int,
    rlo: int,
    rhi: int,
    *,
    step: float,
    stride: int = 1,
    balance: bool = True,
) -> _FFTBlock:
    """Build stable DTD factors on a fine grid, retaining coarse output nodes.

    Output indices rlo/rhi use interpolation spacing H=stride*step. Refine the
    convolution step if the final input window lacks a strict log-concavity
    margin, doubling the stride to preserve H and every interpolation node.
    This structural guard is not a bound on all floating-point errors.
    """
    if not (0 < j0 < M and 0 <= lo <= hi <= M and rlo <= rhi):
        raise ValueError("Invalid block intervals or reference count")
    # Build variable-sized blocks on the host in float64, matching stored factors.
    p = np.float64(j0) / M
    if not (
        math.isfinite(step) and step > 0 and isinstance(stride, int) and stride >= 1
    ):
        raise ValueError(
            "step must be positive and finite and stride a positive integer"
        )
    # kappa_j >= 1/(j+1)+1/(M-j+1); its minimum is nearest M/2.
    # Use the final tail window, including the center needed by the recurrence.
    left, right = min(lo, j0), max(hi, j0)
    middle = min(max(M // 2, max(1, left)), min(M - 1, right))
    curvature_lower = F(1, middle + 1) + F(1, M - middle + 1)
    h = step
    while F(h) >= curvature_lower:
        h /= 2
        stride *= 2
    r = stride * np.arange(rlo, rhi + 1, dtype=np.float64)
    delta = h * r
    loga = -M * _centered_cumulant(p, delta) + h * r**2 / 2
    # Include the reference count even when the requested window excludes it.
    j = np.arange(left, right, dtype=np.float64)
    x = j - j0
    increments = np.log1p(-x / (M - j0)) - np.log1p((x + 1) / j0) + h * (x + 0.5)
    center = j0 - left
    cumulative = np.r_[
        -np.cumsum(increments[:center][::-1])[::-1],
        0.0,
        np.cumsum(increments[center:]),
    ]
    logb = cumulative[lo - left : hi - left + 1]
    logB = (
        -0.5 * np.log(2 * np.pi * M * p * (1 - p))
        + _stirling_correction(M)
        - _stirling_correction(j0)
        - _stirling_correction(M - j0)
    )
    # Normalize the log-concave input factor to max(b)=1. Centering its entire
    # log range instead can overflow b when very small tails are retained.
    # Opposite shifts preserve the product and its norm amplification.
    shift = np.max(logb) if balance else 0.0
    a = np.exp(logB + loga + shift).astype(np.float64)
    b = np.exp(logb - shift).astype(np.float64)
    rlo, rhi = stride * rlo, stride * rhi
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
        b,
        a,
        np.fft.rfft(kernel, nfft),
        nfft,
        stride,
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
    r"""Evaluate binomial expectations by factoring local kernels into convolutions
    computed with the [Fast Fourier Transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) (FFT):

    $$
    y_t=a_t\sum_j e^{-h(kt-(j-j_0))^2/2}\,b_jv_j.
    $$

    Here $v_j$ are destination values at counts $j=0,\ldots,M$, with $M=2N$
    for diploid destination population size $N$ (`pop_size`). The intermediate
    values $y_t$ are expectations on a local grid, $j_0$ is a reference allele count,
    $h$ is the fine convolution spacing, $k$ is an integer output stride, and
    $a_t,b_j$ are diagonal scaling weights. Zero-padded FFTs evaluate the convolution;
    polynomial interpolation approximates $(Pv)_i$ at the supplied offspring
    allele probabilities $q_i$ for source rows $i$. Near zero and one, direct
    binomial sums replace interpolation.
    The Gaussian-shaped kernel is a factorization of the binomial kernel,
    not a normal approximation. Errors arise from tail truncation,
    interpolation, and floating-point arithmetic.

    !!! note "Theory"

        See [local FFT convolution](../methods/fft.md) for the factorization,
        computational cost, and accuracy controls.
    """

    buckets: tuple[_FFTBucket, ...]
    edges: jax.Array
    eta0: jax.Array
    inverse_step: jax.Array
    rmin: jax.Array
    offsets: jax.Array
    boundary_logchoose: jax.Array
    max_tail_mass: jax.Array
    pop_size: int = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    interpolation_bound: float = eqx.field(static=True)
    sampling_bound: float = eqx.field(static=True)
    grad_atol: float | None = eqx.field(static=True)
    angular_sensitivity: float | None = eqx.field(static=True)
    interpolation_grad_bound: float | None = eqx.field(static=True)
    sampling_grad_bound: float | None = eqx.field(static=True)
    tail_tol: float = eqx.field(static=True)
    interp_order: int = eqx.field(static=True)
    min_mode_count: int = eqx.field(static=True)
    block_spacing: float = eqx.field(static=True)
    balance: bool = eqx.field(static=True)
    total_nodes: int = eqx.field(static=True)
    boundary_width: int = eqx.field(static=True)

    def __init__(
        self,
        pop_size: int,
        *,
        atol: float = 1e-10,
        grad_atol: float | None = None,
        angular_sensitivity: float | None = None,
        tail_tol: float = 1e-12,
        interp_order: int | None = None,
        min_mode_count: int = 32,
        block_spacing: float = 1.0,
        balance: bool = True,
    ) -> None:
        r"""Plan local binomial FFTs over all success probabilities, independent of selection.

        Initializes host-computed quantile windows and cached kernel FFTs.
        Convolution starts at h=1/(4 M p(1-p)); interpolation uses H=k h, with
        dyadic k selected from the analytic value and gradient budgets.
        Construction is not JIT-compatible. Core intervals partition log-odds;
        halos support interpolation. For `pop_size <= min_mode_count` all rows use the
        direct fallback. Tail tolerance does not bound interpolation or roundoff.

        **Arguments:**

        - `pop_size`: Diploid effective population size, a Python integer at least one.
        - `atol`: Requested one-step maximum row L1 error, in `[4e-15, 1)`.
          Automatic preparation budgets one quarter each for interpolation and
          sampling. Floating-point error remains uncertified.
        - `grad_atol`, `angular_sensitivity`: Optional branchwise first-derivative
          error target and bound on abs(dq/ds) / sqrt(q(1-q)), as in NUFFT/Jacobi.
          Preparation jointly plans interpolation and tail bounds. The promise
          applies within selected branches, excludes value jumps and arithmetic
          error, and assumes parameter-independent absorbing endpoints.
        - `tail_tol`: Discarded mass budget in `[1e-15, .1)`, default `1e-12`;
          tightened to `atol / 8` to account for signed interpolation weights.
        - `interp_order`: None selects order 12, 14, or 16 from the analytic bound.
          An explicit even order from 2 through 16 overrides order selection;
          its reported value bound may exceed `atol`. An explicit order that
          cannot meet `grad_atol` raises `ValueError`.
        - `min_mode_count`: Initial boundary cutoff in expected copies, at least 8,
          default 32. Automatic planning can enlarge it, up to direct-only
          evaluation, if no candidate order meets both interpolation budgets.
        - `block_spacing`: Center separation in local standard deviations, `(0, 4]`.
        - `balance`: Normalize each input factor to maximum one, compensating in
          its output factor, default true. This preserves the exact operator.

        **Raises:**

        `ValueError` for invalid configuration, disabled x64, nonfinite factors,
        or a gradient target incompatible with explicit geometry or numerical planning;
        `TypeError` if `balance` is not boolean. Results are deterministic for a
        fixed environment but need not be bitwise identical across platforms.
        """
        _validate_atol(atol)
        _validate_gradient(grad_atol, angular_sensitivity)
        grad_budget = math.inf if grad_atol is None else grad_atol / 4
        if grad_budget == 0:
            raise ValueError("grad_atol is too small for floating-point planning")
        sensitivity = 0.0 if angular_sensitivity is None else float(angular_sensitivity)
        if atol / 4 < 1e-15:
            raise ValueError("FFT atol must be at least 4e-15")
        if isinstance(pop_size, bool) or not isinstance(pop_size, int) or pop_size < 1:
            raise ValueError("pop_size must be a Python integer >= 1")
        gene_copies = 2 * pop_size
        # These are host configuration checks, not operations on traced values.
        if not (np.isfinite(tail_tol) and 1e-15 <= tail_tol < 0.1):
            raise ValueError("tail_tol must be finite and in [1e-15, 0.1)")
        # Central Lagrange stencils of all supported orders have Lebesgue
        # constant <= 2 (verified with rational enclosures in test_fft_bounds).
        # Allocate half the sampling budget to each node's tail.
        tail_tol = min(tail_tol, atol / 8)
        if interp_order is not None and (
            isinstance(interp_order, bool)
            or not isinstance(interp_order, int)
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
        if not jax.config.x64_enabled:
            raise ValueError("FFT requires jax_enable_x64=True")
        # A fixed order is an explicit approximation override, like Jacobi.rank.
        # Automatic search is finite; direct-only evaluation is the final fallback.
        if interp_order is None:
            while True:
                for order in (12, 14, 16):
                    geometry, bounds = _fft_grid(
                        gene_copies,
                        min_mode_count,
                        block_spacing,
                        order,
                        sensitivity,
                        atol=atol,
                        grad_atol=grad_atol,
                    )
                    value_bound, grad_bound, _ = bounds
                    if value_bound <= atol / 4 and grad_bound <= grad_budget:
                        break
                if value_bound <= atol / 4 and grad_bound <= grad_budget:
                    interp_order = order
                    break
                min_mode_count = min(pop_size, 2 * min_mode_count)
        else:
            geometry, bounds = _fft_grid(
                gene_copies,
                min_mode_count,
                block_spacing,
                interp_order,
                sensitivity,
                atol=atol,
                grad_atol=grad_atol,
            )
            _, grad_bound, _ = bounds
            if grad_bound > grad_budget:
                raise ValueError(
                    "explicit interp_order cannot meet grad_atol; use interp_order=None"
                )
        centers, eta0, inverse_step, edges, intervals, strides = geometry
        interpolation_bound, interpolation_grad_bound, grad_amplification = bounds
        if grad_amplification:
            tail_tol = min(tail_tol, grad_budget / grad_amplification)
            if tail_tol == 0:
                raise ValueError(
                    "grad_atol is too small for floating-point tail planning"
                )
        # Quantile APIs lose resolution for sub-epsilon probabilities. Start
        # from a representable quantile and enlarge windows by checking tails.
        quantile_tail = max(tail_tol, 2 * np.finfo(float).eps)
        cutoff = min_mode_count / gene_copies
        groups: dict[int, list[tuple[int, _FFTBlock, int]]] = {}
        offsets, rmins = [], []
        total, max_tail = 0, 0.0
        for i, j0 in enumerate(centers):
            rlo, rhi = intervals[i]
            pmin, pmax = expit(eta0[i] + np.array([rlo, rhi]) / inverse_step[i])
            lo = max(0, int(binom.ppf(quantile_tail / 2, gene_copies, pmin)))
            hi = min(gene_copies, int(binom.isf(quantile_tail / 2, gene_copies, pmax)))
            # Monotonic tails give a uniform bound throughout the node halo.
            tail = _tail_bound(gene_copies, lo, hi, pmin, pmax)
            while tail > tail_tol:
                lo, hi = max(0, lo - 1), min(gene_copies, hi + 1)
                tail = _tail_bound(gene_copies, lo, hi, pmin, pmax)
            max_tail = max(max_tail, tail)
            block = _build_block(
                gene_copies,
                j0,
                lo,
                hi,
                rlo,
                rhi,
                # Derive both grids from the same stored reciprocal spacing.
                step=1 / (inverse_step[i] * strides[i]),
                stride=strides[i],
                balance=balance,
            )
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
                conv_indices[k, : block.rows] = (
                    block.length - 1 + block.stride * np.arange(block.rows)
                )
                output_indices[k, : block.rows] = np.arange(offset, offset + block.rows)
            buckets.append(
                _FFTBucket(
                    indices=jnp.asarray(indices),
                    b=jnp.asarray(input_scale),
                    a=jnp.asarray(output_scale),
                    kernel_fft=jnp.asarray(
                        np.stack([block.kernel_fft for _, block, _ in blocks])
                    ),
                    conv_indices=jnp.asarray(conv_indices),
                    output_indices=jnp.asarray(output_indices),
                    nfft=nfft,
                )
            )
        # Stochastic monotonicity makes the upper quantile at the cutoff sufficient
        # for all boundary probabilities. Reflect the upper boundary onto the lower.
        width = min(
            gene_copies + 1,
            int(binom.isf(quantile_tail, gene_copies, min(cutoff, 0.5))) + 1,
        )
        direct_tail = _tail_bound(gene_copies, 0, width - 1, 0.0, min(cutoff, 0.5))
        direct_grad = _direct_grad_bound(
            gene_copies, width, min(cutoff, 0.5), sensitivity, direct_tail
        )
        while direct_tail > tail_tol or direct_grad > grad_budget:
            width += 1
            direct_tail = _tail_bound(gene_copies, 0, width - 1, 0.0, min(cutoff, 0.5))
            direct_grad = _direct_grad_bound(
                gene_copies, width, min(cutoff, 0.5), sensitivity, direct_tail
            )
        sampling_bound = max(2 * max_tail, direct_tail)
        sampling_grad_bound = max(grad_amplification * max_tail, direct_grad)
        if grad_atol is not None and not (
            interpolation_grad_bound <= grad_budget
            and sampling_grad_bound <= grad_budget
        ):
            raise ValueError("cannot meet grad_atol in floating-point planning")
        j = np.arange(width, dtype=float)
        # Recurrence is accurate for small boundary counts, even at very large gene_copies.
        logchoose = np.r_[0.0, np.cumsum(np.log(gene_copies - j[:-1]) - np.log(j[1:]))]
        max_tail = max(max_tail, direct_tail)
        self.buckets = tuple(buckets)
        self.edges = jnp.asarray(edges)
        self.eta0 = jnp.asarray(eta0)
        self.inverse_step = jnp.asarray(inverse_step)
        self.rmin = jnp.asarray(rmins, dtype=jnp.int32)
        self.offsets = jnp.asarray(offsets, dtype=jnp.int32)
        self.boundary_logchoose = jnp.asarray(logchoose)
        self.max_tail_mass = jnp.asarray(max_tail)
        self.pop_size = pop_size
        self.atol = float(atol)
        self.interpolation_bound = interpolation_bound
        self.sampling_bound = sampling_bound
        self.grad_atol = None if grad_atol is None else float(grad_atol)
        self.angular_sensitivity = None if angular_sensitivity is None else sensitivity
        self.interpolation_grad_bound = (
            None if grad_atol is None else interpolation_grad_bound
        )
        self.sampling_grad_bound = None if grad_atol is None else sampling_grad_bound
        self.tail_tol = tail_tol
        self.interp_order = interp_order
        self.min_mode_count = min_mode_count
        self.block_spacing = block_spacing
        self.balance = balance
        self.total_nodes = total
        self.boundary_width = width

    @property
    def error_report(self) -> BackendErrorReport:
        """Read prepared analytic row bounds; arithmetic error remains unknown."""
        grad_budget = None if self.grad_atol is None else self.grad_atol / 4
        coordinate = _gradient_coordinate(self.angular_sensitivity)
        return BackendErrorReport(
            self.atol,
            self.grad_atol,
            (
                ErrorComponent(
                    "interpolation",
                    self.interpolation_bound,
                    self.interpolation_grad_bound,
                    self.atol / 4,
                    grad_budget,
                ),
                ErrorComponent(
                    "sampling",
                    self.sampling_bound,
                    self.sampling_grad_bound,
                    self.atol / 4,
                    grad_budget,
                ),
                ErrorComponent(
                    "roundoff",
                    None,
                    None,
                    self.atol / 2,
                    None if self.grad_atol is None else self.grad_atol / 2,
                ),
            ),
            None if coordinate is None else "branchwise; " + coordinate,
        )

    def __check_init__(self) -> None:
        """Check structural invariants without inspecting traced array values."""
        block_count = self.eta0.size
        if (
            self.eta0.ndim != 1
            or self.inverse_step.shape != (block_count,)
            or self.rmin.shape != (block_count,)
            or self.offsets.shape != (block_count,)
            or self.edges.shape != (max(0, block_count - 1),)
        ):
            raise ValueError(
                "Backend block metadata must have matching one-dimensional shapes"
            )
        if (
            not 1 <= self.boundary_width <= self._gene_copies + 1
            or self.boundary_logchoose.shape != (self.boundary_width,)
        ):
            raise ValueError("boundary_width must match the boundary coefficient array")

    def expect(self, probabilities: jax.Array, values: jax.Array) -> jax.Array:
        r"""Compute binomial expectations using local FFTs.

        **Arguments:**

        - `probabilities`: Vector of finite source probabilities in `[0, 1]`.
        - `values`: Finite real destination vector of length `2 * pop_size + 1`.

        Both inputs accept Python lists, NumPy arrays, or JAX arrays and are
        converted to float64 before shape validation.

        **Returns:**

        A float64 vector with the same shape as `probabilities`, linear in `values`.

        **Raises:**

        `ValueError` for incompatible vector shapes, including during JIT tracing.

        Finiteness and probability ranges follow the unchecked preconditions of
        [`AbstractTransitionBackend.expect`][drifft.AbstractTransitionBackend.expect]; signed values are supported.
        """
        probabilities = jnp.asarray(probabilities, dtype=jnp.float64)
        values = jnp.asarray(values, dtype=jnp.float64)
        if values.shape != (self._gene_copies + 1,) or probabilities.ndim != 1:
            raise ValueError(
                "Expected values.shape == (M+1,) and one-dimensional probabilities"
            )
        cutoff = min(self.min_mode_count / self._gene_copies, 0.5)
        is_boundary = (probabilities <= cutoff) | (probabilities >= 1 - cutoff)
        boundary_columns = jnp.arange(self.boundary_width)
        boundary_values = jnp.stack(
            (values[boundary_columns], values[self._gene_copies - boundary_columns])
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
        r = (eta - self.eta0[block]) * self.inverse_step[block]
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


class _BinomialWeights(BatchablePrimitive):
    r"""Evaluate binomial weights with endpoint-safe probability derivatives.

    For integer count $j$ and fixed degree $n$, the mathematical weight is
    $B_{n,j}(q)=\binom nj q^j(1-q)^{n-j}$. The primal uses log weights in the
    interior and exact endpoint values. Its probability derivative uses
    $B'_{n,j}=n(B_{n-1,j-1}-B_{n-1,j})$, with out-of-range counts contributing
    zero, rather than differentiating logarithms at $q=0$ or $q=1$.

    Counts are integer indices and the degree is static. The auxiliary log-weight
    input also carries multiplicative weight tangents when active. Probability
    rules recursively reuse lower-degree primitives, retaining finite polynomial
    endpoint derivatives under nested AD. The operation is nonlinear, so reverse
    mode transposes its JVP's linearization, not the weight evaluation itself.
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
        size = cast(int, self.params["size"])
        weights = self(counts, logchoose, probabilities)
        tangent = (
            hijax.Zero(self.out_aval.to_tangent_aval())
            if isinstance(dlogchoose, hijax.Zero)
            else weights * dlogchoose
        )
        if isinstance(dprobabilities, hijax.Zero) or size == 0:
            return weights, tangent
        # d B[n,j]/dp = n*(B[n-1,j-1] - B[n-1,j]). Reusing this
        # rule recursively also gives finite, matched higher derivatives.
        # size is static, so math.log supplies a trace-time constant.
        left_logchoose = logchoose + jnp.log(jnp.maximum(counts, 1)) - math.log(size)
        right_logchoose = (
            logchoose + jnp.log(jnp.maximum(size - counts, 1)) - math.log(size)
        )
        derivative = size * (
            _binomial_weights(size - 1, counts - 1, left_logchoose, probabilities)
            - _binomial_weights(size - 1, counts, right_logchoose, probabilities)
        )
        contribution = derivative * dprobabilities[:, None]
        tangent = (
            contribution if isinstance(tangent, hijax.Zero) else tangent + contribution
        )
        return weights, tangent

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
        backend._gene_copies,
        jnp.arange(backend.boundary_width),
        backend.boundary_logchoose,
        low,
    )
    values = boundary_values[(p > 0.5).astype(jnp.int32)]
    return jnp.sum(weights * values, axis=-1)
