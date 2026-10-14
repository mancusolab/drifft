"""Selection-independent Chebyshev geometry and explicit error allocations."""

# Functional Core: deterministic host-side dimensions and JAX sampling nodes.
import math
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from ._base import AbstractTransitionBackend
from ._spectral import expect_probabilities


@final
class NUFFT(AbstractTransitionBackend):
    r"""Compact geometry prepared by `NUFFT` outside JIT.

    **Returns:**

    An immutable PyTree with one dynamic node vector. Dimensions, tolerances and
    diagnostics are static. No node-by-stencil table is retained. `spectral_bound`
    bounds value interpolation in exact arithmetic, not derivatives or roundoff.
    `tail_tol` bounds omitted mass at each sampling node; `sampling_bound`
    includes normalization and the conservative Lebesgue amplification.

    **Failure Modes:**

    Construct outside JIT. Invalid structural metadata raises `ValueError`. FINUFFT's
    requested tolerance is not a rigorous operator-norm error certificate.
    """

    nodes: jax.Array
    gene_copies: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
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
        tail_tol: float | None = None,
        nufft_tol: float | None = None,
        batch_size: int = 32,
        derivative_accuracy: bool = False,
        dtype: DTypeLike = jnp.float64,
    ):
        r"""Prepare a fixed-degree Bernstein/Chebyshev/NUFFT transition.

        **Arguments:**

        - `gene_copies`: Destination gene copies, at least two.
        - `accuracy`: Nominal value-error budget, in `[1e-13, 0.1)`.
        - `tail_tol`: Optional stricter per-node discarded binomial mass tolerance.
        - `nufft_tol`: Optional stricter FINUFFT tolerance (at least `1e-15`).
        - `batch_size`: Number of sampling nodes processed together.
        - `derivative_accuracy`: Increase the fixed degree by 50 percent (capped at
          the exact degree). This is a conservative heuristic, not a derivative bound.
        - `dtype`: Only float64 is supported; enable JAX x64 before construction.

        **Returns:**

        Selection-independent geometry. The value budget is split equally between
        spectral error, sampling error, requested NUFFT tolerance and roundoff.
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
        if not math.isfinite(accuracy) or not 1e-13 <= accuracy < 0.1:
            raise ValueError("accuracy must be finite and in [1e-13, 0.1)")
        if jnp.dtype(dtype) != jnp.dtype(jnp.float64) or not jax.config.read(
            "jax_enable_x64"
        ):
            raise ValueError("float64 and jax_enable_x64=True are required")
        budget = accuracy / 4
        degree = _degree(gene_copies, budget)
        if derivative_accuracy:
            degree = min(gene_copies, math.ceil(1.5 * degree))
        # Lambda_L <= 1 + (2/pi) log(L+1) is deliberately loose at small L.
        lebesgue = 1 + 2 / math.pi * math.log(degree + 1)
        allocated_tail = budget / (2 * lebesgue)
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
        a = math.log(2) - math.log(tail_tol)
        radius = min(
            gene_copies, math.ceil(math.sqrt(gene_copies * a / 2) + 2 * a / 3) + 1
        )
        index = jnp.arange(degree + 1, dtype=jnp.float64)
        # Reflection avoids cancellation in tiny lower-tail nodes.
        low = jnp.sin(jnp.pi * jnp.minimum(index, degree - index) / (2 * degree)) ** 2
        nodes = jnp.where(index <= degree / 2, low, 1 - low)
        bound = (
            0.0
            if degree == gene_copies
            else 4 * gene_copies / degree * math.exp(-(degree**2) / (2 * gene_copies))
        )
        self.nodes = nodes
        self.gene_copies = gene_copies
        self.degree = degree
        self.batch_size = min(batch_size, degree + 1)
        self.radius = radius
        self.accuracy = accuracy
        self.spectral_tol = budget
        self.spectral_bound = bound
        self.tail_tol = tail_tol
        self.sampling_bound = 2 * tail_tol * lebesgue
        self.nufft_tol = nufft_tol
        self.roundoff_budget = budget
        self.derivative_accuracy = derivative_accuracy

    def __check_init__(self) -> None:
        """Reject inconsistent dimensions without inspecting array values."""
        if self.gene_copies < 2 or not 1 <= self.degree <= self.gene_copies:
            raise ValueError("degree must be between 1 and gene_copies")
        if self.nodes.shape != (self.degree + 1,):
            raise ValueError("nodes must have degree + 1 entries")
        if self.batch_size < 1 or not 1 <= self.radius <= self.gene_copies:
            raise ValueError("batch_size and radius must be positive")

    @property
    def stencil_width(self) -> int:
        """Maximum padded stencil width, including masked off-grid entries."""
        return 2 * self.radius + 1

    def expect(self, probabilities: jax.Array, values: jax.Array) -> jax.Array:
        """Apply this backend to destination values at the supplied probabilities."""
        return expect_probabilities(self, probabilities, values)


def _degree(size: int, tolerance: float) -> int:
    """Binary-search the log of the supplied Lobatto interpolation error bound."""
    lo, hi = 1, size
    while lo < hi:
        mid = (lo + hi) // 2
        bound = math.log(4 * size / mid) - mid * mid / (2 * size)
        if bound <= math.log(tolerance):
            hi = mid
        else:
            lo = mid + 1
    return lo
