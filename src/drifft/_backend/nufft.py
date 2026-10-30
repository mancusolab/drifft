"""Selection-independent Chebyshev geometry and explicit error allocations."""

# pattern: Functional Core
# Deterministic host-side dimensions and JAX sampling nodes.
# math is intentional for constructor budgets and the degree/bound planners:
# counts and tolerances are host configuration that determines array shapes.
# Runtime probabilities, values, and coefficients use JAX operations instead.
import math
from typing import Literal, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from .._polynomial import compute_cosine_coefficients, filter_cosine_coefficients
from ._base import AbstractTransitionBackend
from ._spectral import _evaluate, sample_bernstein


@final
class NUFFT(AbstractTransitionBackend):
    r"""Binomial expectations through Chebyshev approximation and NUFFT evaluation.

    For destination values $v_j$ and gene-copy count $M$, approximate
    $B_M[v](q)=\sum_{j=0}^M\binom{M}{j}q^j(1-q)^{M-j}v_j$ by

    $$B_M[v](q)\approx\sum_{k=0}^L c_kT_k(1-2q)
      =\sum_{k=0}^L c_k\cos(k\theta),\qquad\theta=\arccos(1-2q).$$

    Here $T_k$ is the degree-$k$ Chebyshev polynomial and $L$ is the prepared
    approximation degree. The coefficients $c_k$ come from binomial sampling
    at fixed Chebyshev–Lobatto nodes followed by a DCT-I. A type-2 NUFFT
    evaluates the cosine series at arbitrary offspring probabilities.
    `encoding="lobatto"` interpolates the samples. `encoding="filtered"`
    tapers the modes and adds a linear endpoint correction before evaluation.
    Endpoint expectations are set directly to $v_0$ and $v_M$ to suppress
    evaluation roundoff. `degree` always means the output polynomial degree;
    compressed filtered encoding uses `degree + 2` sampling nodes.

    For offspring probabilities $q_i$, `Transition` uses
    $(Pv)_i=B_M[v](q_i)$ for expectations and the transpose $P^Tp$ for forward
    mass. Polynomial truncation, truncated binomial sampling, and NUFFT
    evaluation each contribute approximation error.
    """

    nodes: jax.Array
    gene_copies: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    encoding: Literal["lobatto", "filtered"] = eqx.field(static=True)
    taper_width: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    radius: int = eqx.field(static=True)
    accuracy: float = eqx.field(static=True)
    spectral_tol: float = eqx.field(static=True)
    spectral_bound: float = eqx.field(static=True)
    tail_tol: float = eqx.field(static=True)
    sampling_bound: float = eqx.field(static=True)
    nufft_tol: float = eqx.field(static=True)
    roundoff_budget: float = eqx.field(static=True)
    derivative_accuracy: bool = eqx.field(static=True)

    def __init__(
        self,
        gene_copies: int,
        *,
        accuracy: float = 1e-10,
        encoding: Literal["lobatto", "filtered"] = "lobatto",
        tail_tol: float | None = None,
        nufft_tol: float | None = None,
        batch_size: int = 32,
        derivative_accuracy: bool = False,
        dtype: DTypeLike = jnp.float64,
    ) -> None:
        r"""Prepare a fixed-degree Bernstein/Chebyshev/NUFFT transition.

        **Arguments:**

        - `gene_copies`: Destination gene copies, at least two.
        - `accuracy`: Nominal value-error budget, in `[1e-13, 0.1)`.
        - `encoding`: `"lobatto"` (default) interpolates the samples;
          `"filtered"` tapers them and corrects endpoints as a polynomial.
          Both use the exact degree without tapering when compression cannot
          meet the spectral budget. Neither guarantees positivity.
        - `tail_tol`: Optional stricter per-node discarded binomial mass tolerance.
        - `nufft_tol`: Optional stricter FINUFFT tolerance (at least `1e-15`).
        - `batch_size`: Number of sampling nodes processed together.
        - `derivative_accuracy`: Increase the fixed degree by 50 percent (capped at
          the exact degree). This is a conservative heuristic, not a derivative bound.
        - `dtype`: Only float64 is supported; enable JAX x64 before construction.

        **Returns:**

        `None`. Initializes selection-independent geometry. The value budget is
        split equally between spectral error, sampling error, requested NUFFT
        tolerance and roundoff.
        Roundoff and actual NUFFT error must still be measured independently.

        **Raises:**

        `ValueError` for invalid configuration, disabled x64 or unsupported dtype.
        FFT-specific settings belong to the FFT backend.
        """
        if (
            isinstance(gene_copies, bool)
            or not isinstance(gene_copies, int)
            or gene_copies < 2
        ):
            raise ValueError("gene_copies must be an integer >= 2")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(derivative_accuracy, bool):
            raise ValueError("derivative_accuracy must be bool")
        if encoding not in ("lobatto", "filtered"):
            raise ValueError('encoding must be "lobatto" or "filtered"')
        # Validate and plan with Python scalars before allocating JAX geometry;
        # these tolerances and dimensions are not differentiable model inputs.
        if not math.isfinite(accuracy) or not 1e-13 <= accuracy < 0.1:
            raise ValueError("accuracy must be finite and in [1e-13, 0.1)")
        if jnp.dtype(dtype) != jnp.dtype(jnp.float64) or not jax.config.read(
            "jax_enable_x64"
        ):
            raise ValueError("float64 and jax_enable_x64=True are required")
        # Bound planners act on unit-bounded input values; allocating the budget
        # does not certify FINUFFT or roundoff errors for a particular device.
        budget = accuracy / 4
        taper_width = 0
        if encoding == "filtered":
            modes = _choose_filtered_modes(gene_copies, budget)
            if derivative_accuracy:
                modes = min(gene_copies, math.ceil(1.5 * (modes - 1)) + 1)
            if modes < gene_copies:
                taper_width = math.ceil(1 / math.atanh(modes / gene_copies))
                degree = modes - 1
            else:
                degree = gene_copies
        else:
            degree = _choose_degree(gene_copies, budget)
            if derivative_accuracy:
                degree = min(gene_copies, math.ceil(1.5 * degree))
        sample_degree = degree + (taper_width > 0)
        # Lambda_L <= 1 + (2/pi) log(L+1) is deliberately loose at small L.
        lebesgue = 1 + 2 / math.pi * math.log(degree + 1)
        if taper_width:
            # Endpoint correction doubles the sample-error amplification.
            amplification = 2 * min(
                (2 * sample_degree - taper_width) / taper_width,
                5 + math.log(sample_degree / taper_width),
            )
            bound = 12 * math.exp(-_rate(gene_copies, sample_degree))
        else:
            amplification = lebesgue
            bound = _lobatto_bound(gene_copies, degree)
        allocated_tail = budget / (2 * amplification)
        if tail_tol is None:
            tail_tol = allocated_tail
        if not math.isfinite(tail_tol) or not 0 < tail_tol < 0.1:
            raise ValueError("tail_tol must be finite and in (0, 0.1)")
        tail_tol = min(tail_tol, allocated_tail)
        if nufft_tol is None:
            nufft_tol = min(1e-13, budget)
        if not math.isfinite(nufft_tol) or not 1e-15 <= nufft_tol <= budget:
            raise ValueError("nufft_tol must be in [1e-15, accuracy / 4]")
        # Bernstein: P(|J-Mp| >= t) <= 2 exp(-t²/(2(var+t/3))).
        # t=sqrt(2 var A)+2A/3 suffices. var<=M/4; mode differs
        # from Mp by <=1. The extra 1 covers centering and integer rounding.
        log_tail_bound = math.log(2) - math.log(tail_tol)
        radius = min(
            gene_copies,
            math.ceil(
                math.sqrt(gene_copies * log_tail_bound / 2) + 2 * log_tail_bound / 3
            )
            + 1,
        )
        index = jnp.arange(sample_degree + 1, dtype=jnp.float64)
        # Reflection avoids cancellation in tiny lower-tail nodes.
        low = (
            jnp.sin(
                jnp.pi * jnp.minimum(index, sample_degree - index) / (2 * sample_degree)
            )
            ** 2
        )
        nodes = jnp.where(index <= sample_degree / 2, low, 1 - low)
        self.nodes = nodes
        self.gene_copies = gene_copies
        self.degree = degree
        self.encoding = encoding
        self.taper_width = taper_width
        self.batch_size = min(batch_size, sample_degree + 1)
        self.radius = radius
        self.accuracy = accuracy
        self.spectral_tol = budget
        self.spectral_bound = bound
        self.tail_tol = tail_tol
        self.sampling_bound = 2 * tail_tol * amplification
        self.nufft_tol = nufft_tol
        self.roundoff_budget = budget
        self.derivative_accuracy = derivative_accuracy

    def __check_init__(self) -> None:
        """Reject inconsistent dimensions without inspecting array values."""
        if self.gene_copies < 2 or not 1 <= self.degree <= self.gene_copies:
            raise ValueError("degree must be between 1 and gene_copies")
        if self.encoding not in ("lobatto", "filtered"):
            raise ValueError('encoding must be "lobatto" or "filtered"')
        if self.taper_width and (
            self.encoding != "filtered"
            or not 1 <= self.taper_width <= self.degree + 1 < self.gene_copies
        ):
            raise ValueError("taper width and compressed degree are inconsistent")
        if self.nodes.shape != (self.degree + 1 + (self.taper_width > 0),):
            raise ValueError("nodes must match the encoding's sampling degree")
        if self.batch_size < 1 or not 1 <= self.radius <= self.gene_copies:
            raise ValueError("batch_size and radius must be positive")

    @property
    def stencil_width(self) -> int:
        r"""Padded sampling width `2 * radius + 1`, including masked off-grid entries."""
        return 2 * self.radius + 1

    def expect(self, probabilities: jax.Array, values: jax.Array) -> jax.Array:
        r"""Compute binomial expectations using Chebyshev sampling and NUFFT.

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
        if values.shape != (self.gene_copies + 1,) or probabilities.ndim != 1:
            raise ValueError(
                "Expected values.shape == (M+1,) and one-dimensional probabilities"
            )
        coefficients = compute_cosine_coefficients(sample_bernstein(self, values))
        if self.taper_width:
            coefficients = filter_cosine_coefficients(
                coefficients,
                self.taper_width,
                values[jnp.array([0, self.gene_copies])],
            )
        return _evaluate(
            coefficients,
            probabilities,
            self.nufft_tol,
            values[jnp.array([0, self.gene_copies])],
        )


def _choose_degree(size: int, tolerance: float) -> int:
    """Find the least degree certified by either Lobatto interpolation bound."""
    # This and the bound helpers below are host-only scalar planners. Keeping
    # math/Python arithmetic here yields concrete integer dimensions, not tracers.
    # The sharper certificate stops at M-2, so the combined bound can rise
    # at M-1. Search only its monotone prefix, handling the last two degrees
    # separately instead of allowing that rise to hide a valid smaller degree.
    if size == 2 or _lobatto_bound(size, size - 2) > tolerance:
        return size - 1 if _lobatto_bound(size, size - 1) <= tolerance else size
    lo, hi = 1, size - 2
    while lo < hi:
        mid = (lo + hi) // 2
        if _lobatto_bound(size, mid) <= tolerance:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _rate(size: int, modes: int) -> float:
    """Compute M*Psi(L/M), without cancellation for small L/M."""
    u = modes / size
    if u < 0.01:
        # Psi(u) = sum u^(2k)/((2k-1)*2k); omitted terms are < 3e-18
        # relative to the leading term at this branch's upper limit.
        squared = u * u
        return (
            size
            * squared
            * (0.5 + squared * (1 / 12 + squared * (1 / 30 + squared / 56)))
        )
    return size * ((1 + u) * math.log1p(u) + (1 - u) * math.log1p(-u)) / 2


def _lobatto_bound(size: int, degree: int) -> float:
    """Best of the old tail certificate and the sharper interpolation bound."""
    if degree == size:
        return 0.0
    old = 4 * size / degree * math.exp(-degree * degree / (2 * size))
    if math.sqrt(size) <= degree + 1 < size:
        lebesgue = 1 + 2 / math.pi * math.log(degree + 1)
        sharp = 4.2 * (1 + lebesgue) * math.exp(-_rate(size, degree + 1))
        return min(old, sharp)
    return old


def _choose_filtered_modes(size: int, tolerance: float) -> int:
    """Select retained modes with the factor-two endpoint correction included."""
    lo, hi = math.ceil(math.sqrt(size)), size
    target = math.log(12 / tolerance)
    while lo < hi:
        mid = (lo + hi) // 2
        if _rate(size, mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo
