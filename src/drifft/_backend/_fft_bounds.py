"""Host-side analytic bounds for the unchanged local FFT approximation.

Functional core: rational polynomial bounds and immutable diagnostics. These
bounds describe exact binomial nodes at the stored grid coordinates. Factor
construction, coordinate arithmetic and FFT rounding are deliberately excluded.
"""

import math
from decimal import Decimal, localcontext
from fractions import Fraction as F
from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from .fft import FFT


class FFTBlockDiagnostics(NamedTuple):
    """Whole-core row-L1 bounds, with derivatives in theta=2 asin(sqrt(q))."""

    interpolation_bound: float
    sampling_bound: float
    value_bound: float
    angular_derivative_bound: float
    stencil_derivative_jump_bound: float
    amplification_estimate: float


class FFTSwitchDiagnostics(NamedTuple):
    """An interface's value jump bound; no classical derivative is claimed there."""

    kind: str
    probability: float
    value_jump_bound: float


class FFTDiagnostics(NamedTuple):
    """Exact-arithmetic bounds and a separate, uncertified stability estimate.

    Angular derivative bounds hold on selected smooth branches only. They do
    not include distributional derivatives of value jumps. Multiply value
    errors by the observable infinity norm (or forward-input L1 norm).
    ``amplification_estimate`` is a conditioning diagnostic, not a roundoff
    bound. None of these fields certifies the floating-point implementation.
    """

    value_bound: float
    angular_derivative_bound: float
    direct_value_bound: float
    direct_angular_derivative_bound: float
    blocks: tuple[FFTBlockDiagnostics, ...]
    switches: tuple[FFTSwitchDiagnostics, ...]
    amplification_estimate: float


def _trim(p):
    while len(p) > 1 and p[-1] == 0:
        p = p[:-1]
    return p


def _add(a, b):
    return _trim(
        tuple(
            (a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
            for i in range(max(len(a), len(b)))
        )
    )


def _mul(a, b):
    out = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return _trim(tuple(out))


def _derivative(a):
    return tuple(i * a[i] for i in range(1, len(a))) or (F(0),)


@lru_cache(maxsize=128)
def _energy(M: int, n: int):
    """Exact coefficients of H_n as a polynomial in v=q(1-q)."""
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
    left, right = [b[0]], [b[-1]]
    while len(b) > 1:
        b = tuple((x + y) / 2 for x, y in zip(b[:-1], b[1:], strict=True))
        left.append(b[0])
        right.append(b[-1])
    return tuple(left), tuple(reversed(right))


def _range(p, lo: F, hi: F, depth: int = 3):
    pieces = [_bernstein(p, lo, hi)]
    for _ in range(depth):
        pieces = [child for b in pieces for child in _split(b)]
    return min(min(b) for b in pieces), max(max(b) for b in pieces)


@lru_cache(maxsize=8)
def _constants(m: int):
    """Rational bounds for Lambda, Lambda', and the Taylor/Peano constant."""
    weights = []
    nodes = range(1 - m // 2, 1 + m // 2)
    for k in nodes:
        p = (F(1),)
        for j in nodes:
            if j != k:
                p = _mul(p, (F(-j, k - j), F(1, k - j)))
        weights.append(p)
    derivatives = [_derivative(p) for p in weights]
    remainders = []
    for k, p in zip(nodes, derivatives, strict=True):
        for _ in range(m):
            p = _mul(p, (k, -1))
        remainders.append(p)

    def absolute_sum_bound(polys):
        # Common subdivisions retain the coupling in the sum of absolute values.
        pieces = [[_bernstein(p, F(0), F(1)) for p in polys]]
        for _ in range(4):
            children = []
            for group in pieces:
                pairs = [_split(b) for b in group]
                children.extend([[pair[side] for pair in pairs] for side in (0, 1)])
            pieces = children
        return max(sum(max(abs(x) for x in b) for b in group) for group in pieces)

    return (
        F(math.comb(m, m // 2), 4**m),
        absolute_sum_bound(weights),
        absolute_sum_bound(derivatives),
        absolute_sum_bound(remainders) / math.factorial(m),
    )


def _transcendental(x: F, operation: str):
    """Enclose exp/ln using correctly rounded Decimal operations at 70 digits."""
    with localcontext() as ctx:
        ctx.prec = 70
        d = Decimal(x.numerator) / Decimal(x.denominator)
        low, high = d.next_minus(), d.next_plus()
        # Both functions are increasing and correctly rounded (ROUND_HALF_EVEN).
        if operation == "exp":
            a, b = low.exp(), high.exp()
        else:
            a, b = low.ln(), high.ln()
        return F(a.next_minus()), F(b.next_plus())


def _logistic(x: F):
    a, b = _transcendental(x, "exp")
    return a / (1 + a), b / (1 + b)


def _variance_interval(lo: F, hi: F):
    lower = min(lo * (1 - lo), hi * (1 - hi))
    upper = F(1, 4) if lo <= F(1, 2) <= hi else max(lo * (1 - lo), hi * (1 - hi))
    # Outward dyadic rounding keeps subsequent exact polynomial arithmetic cheap.
    scale = 2**48
    return F(lower * scale // 1, scale), F(-(-upper * scale // 1), scale)


def _upper(x: F):
    try:
        value = float(x)
    except OverflowError:
        return math.inf
    if math.isfinite(value) and F(value) < x:
        value = math.nextafter(value, math.inf)
    return value


def _sqrt_upper(x: F):
    value = math.sqrt(_upper(x))
    if math.isfinite(value) and F(value) ** 2 < x:
        value = math.nextafter(value, math.inf)
    return value


def _tail(M: int, k: int, q: F, *, upper: bool):
    """Chernoff bound for P[J>=k] or P[J<=k], enclosing log/exp evaluation."""
    if (upper and k > M) or (not upper and k < 0):
        return F(0)
    if (upper and k <= M * q) or (not upper and k >= M * q):
        return F(1)
    x = F(k, M)
    kl = F(0)
    if k:
        kl += x * _transcendental(x / q, "ln")[0]
    if k < M:
        kl += (1 - x) * _transcendental((1 - x) / (1 - q), "ln")[0]
    return min(F(1), _transcendental(-M * kl, "exp")[1])


def _direct_gradient(M: int, width: int, cutoff: F, tail: F):
    if width > M:
        return 0.0
    bound = _sqrt_upper(M * tail)
    if width >= M * cutoff:
        # Above the mean, omitted-row derivatives have one sign. Their sum
        # telescopes to M binom(M-1,W-1) q^(W-1/2)(1-q)^(M-W+1/2).
        q = min(cutoff, F(2 * width - 1, 2 * M))
        log_bound = _transcendental(F(M * math.comb(M - 1, width - 1)), "ln")[1]
        log_bound += F(2 * width - 1, 2) * _transcendental(q, "ln")[1]
        log_bound += F(2 * (M - width) + 1, 2) * _transcendental(1 - q, "ln")[1]
        bound = min(bound, _upper(_transcendental(log_bound, "exp")[1]))
    return bound


def fft_diagnostics(backend: "FFT") -> FFTDiagnostics:
    """Analyze stored geometry on the host without modifying its application."""
    M = 2 * backend.pop_size
    cutoff = F(min(backend.min_mode_count / M, 0.5))
    direct = _tail(M, backend.boundary_width, cutoff, upper=True)
    direct_grad = _direct_gradient(M, backend.boundary_width, cutoff, direct)
    eta = np.asarray(backend.eta0)
    variance = np.asarray(backend.variance)
    rmin = np.asarray(backend.rmin)
    offsets = np.asarray(backend.offsets)
    edges = np.asarray(backend.edges)
    blocks = []
    alpha, lebesgue, dlebesgue, peano = _constants(backend.interp_order)
    for i in range(len(eta)):
        # Bucket padding is identified structurally, never by nonzero factors.
        offset = int(offsets[i])
        bucket, row = next(
            (bucket, int(rows[0]))
            for bucket in backend.buckets
            if len(
                rows := np.flatnonzero(
                    np.asarray(bucket.output_indices)[:, 0] == offset
                )
            )
        )
        length = int(np.asarray(bucket.conv_indices)[row, 0]) + 1
        count = int(
            (offsets[i + 1] if i + 1 < len(eta) else backend.total_nodes) - offset
        )
        lo = int(np.asarray(bucket.indices)[row, 0])
        hi = lo + length - 1
        V, e = F(float(variance[i])), F(float(eta[i]))
        qlo = _logistic(e + int(rmin[i]) / V)[0]
        qhi = _logistic(e + (int(rmin[i]) + count - 1) / V)[1]
        vl, vh = _variance_interval(qlo, qhi)
        m = backend.interp_order
        energy = max(F(0), _range(_energy(M, m), vl, vh)[1])
        dm = F(_sqrt_upper(energy / V ** (2 * m)))
        tail = min(
            F(1), _tail(M, lo - 1, qlo, upper=False) + _tail(M, hi + 1, qhi, upper=True)
        )
        interpolation, sampling = alpha * dm, lebesgue * tail
        corelo = cutoff if i == 0 else _logistic(F(float(edges[i - 1])))[0]
        corehi = 1 - cutoff if i == len(eta) - 1 else _logistic(F(float(edges[i])))[1]
        corev = min(corelo * (1 - corelo), corehi * (1 - corehi))
        scale = F(_sqrt_upper(V * V / corev))
        grad = scale * (peano * dm + dlebesgue * tail)
        # This is deliberately an estimate using the actual stored factors.
        a = np.asarray(bucket.a)[row, :count]
        b = np.asarray(bucket.b)[row, :length]
        kernel = np.fft.irfft(np.asarray(bucket.kernel_fft)[row], n=bucket.nfft)
        with np.errstate(over="ignore"):
            amplification = float(
                math.sqrt(bucket.nfft)
                * np.max(np.abs(a))
                * np.linalg.norm(kernel)
                * np.linalg.norm(b)
            )
        blocks.append(
            FFTBlockDiagnostics(
                _upper(interpolation),
                _upper(sampling),
                _upper(interpolation + sampling),
                _upper(grad),
                _upper(2 * grad),
                amplification,
            )
        )
    switches = []
    if blocks:
        switches.append(
            FFTSwitchDiagnostics(
                "direct/interior",
                float(cutoff),
                _upper(direct + F(blocks[0].value_bound)),
            )
        )
        for i, edge in enumerate(edges):
            switches.append(
                FFTSwitchDiagnostics(
                    "block",
                    float(_logistic(F(float(edge)))[0]),
                    _upper(F(blocks[i].value_bound) + F(blocks[i + 1].value_bound)),
                )
            )
        switches.append(
            FFTSwitchDiagnostics(
                "interior/direct",
                float(1 - cutoff),
                _upper(direct + F(blocks[-1].value_bound)),
            )
        )
    else:
        switches.append(FFTSwitchDiagnostics("reflection", 0.5, _upper(2 * direct)))
    return FFTDiagnostics(
        max([_upper(direct)] + [b.value_bound for b in blocks]),
        max([direct_grad] + [b.angular_derivative_bound for b in blocks]),
        _upper(direct),
        direct_grad,
        tuple(blocks),
        tuple(switches),
        max((b.amplification_estimate for b in blocks), default=0.0),
    )
