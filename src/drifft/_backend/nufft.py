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

from .._polynomial import compute_cosine_coefficients, filter_cosine_coefficients
from ._accuracy import (
    BackendErrorReport,
    ErrorComponent,
    _gradient_coordinate,
    _validate_atol,
    _validate_gradient,
)
from ._base import AbstractTransitionBackend
from ._spectral import _evaluate, _load_nufft2, sample_bernstein


@final
class NUFFT(AbstractTransitionBackend):
    r"""Approximate binomial expectations by a Chebyshev series evaluated with a
    [Non-uniform Fast Fourier Transform](https://en.wikipedia.org/wiki/Non-uniform_discrete_Fourier_transform#Nonuniform_fast_Fourier_transform) (NUFFT):

    $$
    (Pv)_i\approx\sum_{k=0}^L c_kT_k(1-2q_i)
    =\sum_{k=0}^L c_k\cos(k\vartheta_i),\qquad\vartheta_i=\arccos(1-2q_i).
    $$

    Here $v_j$ are destination values at counts $j=0,\ldots,M$, with $M=2N$
    for diploid destination population size $N$ (`pop_size`). The offspring
    allele probability for source row $i$ is $q_i$, $T_k$ is a Chebyshev
    polynomial, and $L$ is the prepared approximation degree. Binomial sampling
    of $v$ at fixed Chebyshev–Lobatto nodes followed by a DCT-I supplies the
    coefficients $c_k$. A type-2 NUFFT evaluates the cosine series at $q_i$.
    Polynomial truncation, truncated binomial sampling, and NUFFT evaluation
    contribute separate approximation errors.

    !!! note "Theory"

        See [Chebyshev and NUFFT transforms](../methods/nufft.md) for the
        encodings, endpoint corrections, and accuracy controls.
    """

    nodes: jax.Array
    pop_size: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    encoding: Literal["lobatto", "filtered"] = eqx.field(static=True)
    taper_width: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    radius: int = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    spectral_tol: float = eqx.field(static=True)
    spectral_bound: float = eqx.field(static=True)
    tail_tol: float = eqx.field(static=True)
    sampling_bound: float = eqx.field(static=True)
    nufft_tol: float = eqx.field(static=True)
    roundoff_budget: float = eqx.field(static=True)
    grad_atol: float | None = eqx.field(static=True)
    angular_sensitivity: float | None = eqx.field(static=True)
    spectral_grad_bound: float | None = eqx.field(static=True)
    sampling_grad_bound: float | None = eqx.field(static=True)

    def __init__(
        self,
        pop_size: int,
        *,
        atol: float = 1e-10,
        encoding: Literal["lobatto", "filtered"] = "lobatto",
        tail_tol: float | None = None,
        nufft_tol: float | None = None,
        batch_size: int = 32,
        grad_atol: float | None = None,
        angular_sensitivity: float | None = None,
    ) -> None:
        r"""Prepare a fixed-degree Bernstein/Chebyshev/NUFFT transition.

        Initializes selection-independent geometry. The value budget is
        split equally between spectral error, sampling error, requested NUFFT
        tolerance and roundoff.
        Roundoff and actual NUFFT error must still be measured independently.

        **Arguments:**

        - `pop_size`: Diploid effective population size, a Python integer at least one.
        - `atol`: Requested one-step maximum row L1 error, in `[1e-13, 0.1)`.
          This bounds expectation errors for unit-bounded observables and forward
          L1 errors for unit-L1 mass when all contributions are controlled.
          `error_report` exposes the uncertified transform and arithmetic terms.
        - `encoding`: `"lobatto"` (default) interpolates the samples;
          `"filtered"` tapers them and corrects endpoints as a polynomial.
          Both use the exact degree without tapering when compression cannot
          meet the spectral budget. Neither guarantees positivity.
        - `tail_tol`: Optional stricter per-node discarded binomial mass tolerance.
        - `nufft_tol`: Optional stricter FINUFFT tolerance (at least `1e-15`).
        - `batch_size`: Target sampling batch size in nodes. Reflected nodes are processed
          in pairs, with at least one pair per batch.
        - `grad_atol`: Optional absolute first-derivative operator-error target,
          in `(0, 1)`. Requires `angular_sensitivity`; None plans only values.
        - `angular_sensitivity`: A finite nonnegative bound on
          $|\partial_s q|/\sqrt{q(1-q)}$ over the entire parameter domain.
          Here $s$ is the chosen parameter coordinate (for example, selection
          strength or its logarithm), or a unit direction in a specified parameter
          norm. Constant absorbing rows have zero sensitivity. The caller must
          establish this bound; support-changing endpoints are not covered.
          Geometry is fixed across this domain and is never replanned inside AD.

        **Raises:**

        `ValueError` for invalid configuration or disabled x64. Only float64 is
        supported; enable JAX x64 before construction.
        FFT-specific settings belong to the [`FFT`][drifft.FFT] backend.
        """
        _load_nufft2()
        if isinstance(pop_size, bool) or not isinstance(pop_size, int) or pop_size < 1:
            raise ValueError("pop_size must be a Python integer >= 1")
        gene_copies = 2 * pop_size
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        _validate_gradient(grad_atol, angular_sensitivity)
        if encoding not in ("lobatto", "filtered"):
            raise ValueError('encoding must be "lobatto" or "filtered"')
        # Validate and plan with Python scalars before allocating JAX geometry;
        # these tolerances and dimensions are not differentiable model inputs.
        _validate_atol(atol)
        if not math.isfinite(atol) or not 1e-13 <= atol < 0.1:
            raise ValueError("atol must be finite and in [1e-13, 0.1)")
        if not jax.config.x64_enabled:
            raise ValueError("NUFFT requires jax_enable_x64=True")
        # Bound planners act on unit-bounded input values; allocating the budget
        # does not certify FINUFFT or roundoff errors for a particular device.
        budget = atol / 4
        grad_budget = None if grad_atol is None else grad_atol / 4
        taper_width = 0
        if encoding == "filtered":
            modes = _choose_filtered_joint_modes(
                gene_copies, budget, grad_budget, angular_sensitivity
            )
            if modes < gene_copies:
                taper_width = math.ceil(1 / math.atanh(modes / gene_copies))
                degree = modes - 1
            else:
                degree = gene_copies
        else:
            spectral_target = budget
            if grad_budget is not None and angular_sensitivity:
                # The residual is a trigonometric polynomial of degree M.
                # Bernstein's inequality bounds its angular derivative by M
                # times its uniform norm, including interpolation aliasing.
                spectral_target = min(
                    spectral_target, grad_budget / angular_sensitivity / gene_copies
                )
            degree = _choose_degree(gene_copies, spectral_target)
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
            grad_bound = (
                None
                if angular_sensitivity is None
                else (
                    0.0
                    if angular_sensitivity == 0
                    else math.exp(
                        math.log(angular_sensitivity)
                        + math.log(9 * sample_degree + 6)
                        - _rate(gene_copies, sample_degree)
                    )
                )
            )
        else:
            amplification = lebesgue
            bound = _lobatto_bound(gene_copies, degree)
            grad_bound = (
                None
                if angular_sensitivity is None
                else angular_sensitivity * (gene_copies * bound)
            )
        allocated_tail = budget / (2 * amplification)
        if grad_budget is not None and angular_sensitivity:
            # Sample perturbations produce a degree-d polynomial. The same
            # fixed polynomial (including endpoint correction) is differentiated.
            allocated_tail = min(
                allocated_tail,
                grad_budget / angular_sensitivity / degree / (2 * amplification),
            )
        if allocated_tail == 0:
            raise ValueError("grad_atol is too small for floating-point planning")
        if tail_tol is None:
            tail_tol = allocated_tail
        if not math.isfinite(tail_tol) or not 0 < tail_tol < 0.1:
            raise ValueError("tail_tol must be finite and in (0, 0.1)")
        tail_tol = min(tail_tol, allocated_tail)
        if nufft_tol is None:
            # Conditional coefficient-L1 model: derivative coefficients in q
            # grow by at most 2d²; |dq/ds| <= angular_sensitivity/2.
            # FINUFFT's tolerance floor is not a proof of this model, so these
            # remain requests, not reported bounds.
            coefficient_norm = math.sqrt(2 * degree + 1) * (
                1 + bound + 2 * tail_tol * amplification
            )
            requested = min(1e-13, budget / coefficient_norm)
            if grad_budget is not None and angular_sensitivity:
                requested = min(
                    requested,
                    grad_budget / angular_sensitivity / degree**2 / coefficient_norm,
                )
            nufft_tol = max(1e-15, requested)
        if not math.isfinite(nufft_tol) or not 1e-15 <= nufft_tol <= budget:
            raise ValueError("nufft_tol must be in [1e-15, atol / 4]")
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
        self.pop_size = pop_size
        self.degree = degree
        self.encoding = encoding
        self.taper_width = taper_width
        self.batch_size = min(batch_size, sample_degree + 1)
        self.radius = radius
        self.atol = float(atol)
        self.spectral_tol = budget
        self.spectral_bound = bound
        self.tail_tol = tail_tol
        self.sampling_bound = 2 * tail_tol * amplification
        self.nufft_tol = nufft_tol
        self.roundoff_budget = budget
        self.grad_atol = None if grad_atol is None else float(grad_atol)
        self.angular_sensitivity = (
            None if angular_sensitivity is None else float(angular_sensitivity)
        )
        self.spectral_grad_bound = grad_bound
        if radius == gene_copies:
            self.sampling_bound = 0.0
        self.sampling_grad_bound = (
            None
            if angular_sensitivity is None
            else angular_sensitivity * (degree * self.sampling_bound)
        )

    @property
    def error_report(self) -> BackendErrorReport:
        """Report analytic bounds separately from numerical budget allocations."""
        grad_budget = None if self.grad_atol is None else self.grad_atol / 4
        budget = self.atol / 4
        return BackendErrorReport(
            self.atol,
            self.grad_atol,
            (
                ErrorComponent(
                    "spectral",
                    self.spectral_bound,
                    self.spectral_grad_bound,
                    budget,
                    grad_budget,
                ),
                ErrorComponent(
                    "sampling",
                    self.sampling_bound,
                    self.sampling_grad_bound,
                    budget,
                    grad_budget,
                ),
                ErrorComponent("transform", None, None, budget, grad_budget),
                ErrorComponent("roundoff", None, None, budget, grad_budget),
            ),
            _gradient_coordinate(self.angular_sensitivity),
        )

    def __check_init__(self) -> None:
        """Reject inconsistent dimensions without inspecting array values."""
        if self._gene_copies < 2 or not 1 <= self.degree <= self._gene_copies:
            raise ValueError("degree must be between 1 and 2 * pop_size")
        if self.encoding not in ("lobatto", "filtered"):
            raise ValueError('encoding must be "lobatto" or "filtered"')
        if self.taper_width and (
            self.encoding != "filtered"
            or not 1 <= self.taper_width <= self.degree + 1 < self._gene_copies
        ):
            raise ValueError("taper width and compressed degree are inconsistent")
        if self.nodes.shape != (self.degree + 1 + (self.taper_width > 0),):
            raise ValueError("nodes must match the encoding's sampling degree")
        if self.batch_size < 1 or not 1 <= self.radius <= self._gene_copies:
            raise ValueError("batch_size and radius must be positive")

    @property
    def stencil_width(self) -> int:
        r"""Padded sampling width `2 * radius + 1`, including masked off-grid entries."""
        return 2 * self.radius + 1

    def expect(self, probabilities: jax.Array, values: jax.Array) -> jax.Array:
        r"""Compute binomial expectations using Chebyshev sampling and NUFFT.

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
        coefficients = compute_cosine_coefficients(sample_bernstein(self, values))
        if self.taper_width:
            coefficients = filter_cosine_coefficients(
                coefficients,
                self.taper_width,
                values[jnp.array([0, self._gene_copies])],
            )
        return _evaluate(
            coefficients,
            probabilities,
            self.nufft_tol,
            values[jnp.array([0, self._gene_copies])],
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


def _choose_filtered_joint_modes(
    size: int, atol: float, grad_atol: float | None, sensitivity: float | None
) -> int:
    """Plan the same endpoint-corrected filter for values and first derivatives.

    The joint note gives raw bounds 6 exp(-rate) and 9L exp(-rate).
    Endpoint correction doubles the value bound and adds the raw value bound
    to the angular derivative bound. Both decrease for L >= sqrt(M).
    """
    lo, hi = _choose_filtered_modes(size, atol), size
    if grad_atol is None or not sensitivity:
        return lo
    while lo < hi:
        mid = (lo + hi) // 2
        log_bound = math.log(sensitivity) + math.log(9 * mid + 6) - _rate(size, mid)
        if log_bound <= math.log(grad_atol):
            hi = mid
        else:
            lo = mid + 1
    return lo
