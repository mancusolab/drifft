"""Pure JAX transition kernels."""

# Functional Core: array-only numerical kernels.
import math
from collections.abc import Callable
from typing import Literal, final

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax.scipy.special import gammaln, xlog1py, xlogy

from ._plan import Block, TransitionPlan
from ._solve import ExpectedVisitsResult, _expected_visits


def _apply_block(block: Block, v: jax.Array) -> jax.Array:
    """Apply one block to its input window, returning its log-odds nodes."""
    transformed = jnp.fft.rfft(block.b * v, n=block.nfft)
    conv = jnp.fft.irfft(transformed * block.kernel_fft, n=block.nfft)
    return block.a * conv[block.length - 1 : block.length - 1 + block.rows]


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


def _expect_probabilities(
    plan: TransitionPlan, p: jax.Array, v: jax.Array
) -> jax.Array:
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


def _expect_banded(
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
    p = Underdominance(s)(jnp.arange(M + 1, dtype=jnp.float64) / M)

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


@final
class Underdominance(eqx.Module):
    r"""Symmetric heterozygote disadvantage, callable on allele frequencies.

    **Arguments:**

    - `strength`: Scalar selection coefficient in `[0, 1]`, converted to a
      float64 JAX leaf. Changing its value does not change model structure.

    **Failure Modes:**

    Frequencies and strength must lie in `[0, 1]`; these are caller preconditions.
    Non-scalar strength raises `ValueError`. Keep x64 enabled.
    """

    strength: jax.Array = eqx.field(
        converter=lambda x: jnp.asarray(x, dtype=jnp.float64)
    )

    def __check_init__(self) -> None:
        """Require scalar selection without inspecting traced values."""
        if self.strength.ndim != 0:
            raise ValueError("strength must be scalar")

    def __call__(self, frequencies: jax.Array) -> jax.Array:
        """Return offspring probabilities with the same shape as frequencies."""
        disadvantage = self.strength * frequencies * (1 - frequencies)
        return (frequencies - disadvantage) / (1 - 2 * disadvantage)


@final
class Transition(eqx.Module):
    r"""A reusable plan combined with a selection model or explicit probabilities.

    **Arguments:**

    - `plan`: Geometry for the destination population, prepared outside JIT.
    - `selection`: A JAX-compatible callable mapping source frequencies to
      offspring probabilities. Equinox models keep numeric parameters dynamic.
    - `source_gene_copies`: Source population size; defaults to the plan's size.
    - `probabilities`: Alternative to selection; normally supplied through
      `Transition.from_probabilities`. Each entry defines one source row.

    **Returns:**

    An immutable PyTree. Construction binds existing geometry without rebuilding
    it. Selection probabilities are evaluated during application, so updates and
    differentiation through model parameters cannot use a stale cached vector.

    **Raises:**

    `ValueError` for ambiguous configuration, invalid source dimensions, or an
    empty/non-vector probability array; `TypeError` for non-callable selection.

    **Failure Modes:**

    Probabilities must be finite and in `[0, 1]`. No runtime callbacks enforce
    these value preconditions. Approximation errors are those of the plan; no
    outputs are clipped or renormalized. Methods are pure and return new arrays.
    `expect` and `advance` are undecorated for use inside an outer `eqx.filter_jit`.
    `expected_visits` supplies its own JIT boundary.
    """

    plan: TransitionPlan
    selection: Callable[[jax.Array], jax.Array] | None
    _probabilities: jax.Array | None
    source_gene_copies: int = eqx.field(static=True)

    def __init__(
        self,
        plan: TransitionPlan,
        selection: Callable[[jax.Array], jax.Array] | None = None,
        *,
        source_gene_copies: int | None = None,
        probabilities: jax.Array | None = None,
    ):
        self.plan = plan
        self.selection = selection
        self._probabilities = (
            None
            if probabilities is None
            else jnp.asarray(probabilities, dtype=jnp.float64)
        )
        inferred = (
            plan.gene_copies
            if self._probabilities is None
            else self._probabilities.size - 1
        )
        self.source_gene_copies = (
            inferred if source_gene_copies is None else source_gene_copies
        )

    def __check_init__(self) -> None:
        """Validate static dimensions and mutually exclusive construction modes."""
        if (self.selection is None) == (self._probabilities is None):
            raise ValueError("Provide exactly one of selection or probabilities")
        if not isinstance(self.source_gene_copies, int) or isinstance(
            self.source_gene_copies, bool
        ):
            raise ValueError("source_gene_copies must be an integer")
        if self.selection is not None:
            if not callable(self.selection):
                raise TypeError("selection must be callable")
            if self.source_gene_copies < 2:
                raise ValueError(
                    "source_gene_copies must be >= 2 for a selection model"
                )
        elif self._probabilities is not None:
            if self._probabilities.ndim != 1 or self._probabilities.size == 0:
                raise ValueError("probabilities must be a nonempty vector")
            if self._probabilities.size != self.source_gene_copies + 1:
                raise ValueError("source_gene_copies must match the probability vector")

    @classmethod
    def from_probabilities(
        cls, plan: TransitionPlan, probabilities: jax.Array
    ) -> "Transition":
        """Bind one probability per source row; destination size comes from plan.

        Arrays remain dynamic leaves. To differentiate through their construction,
        evaluate the probability map inside the differentiated function. Rows may
        represent sampled probabilities rather than a complete allele-count grid.
        """
        return cls(plan, probabilities=probabilities)

    @property
    def probabilities(self) -> jax.Array:
        """Evaluate the selection model on the source grid, or return supplied rows."""
        if self._probabilities is not None:
            return self._probabilities
        assert self.selection is not None
        frequencies = (
            jnp.arange(self.source_gene_copies + 1, dtype=jnp.float64)
            / self.source_gene_copies
        )
        probabilities = jnp.asarray(self.selection(frequencies), dtype=jnp.float64)
        if probabilities.shape != frequencies.shape:
            raise ValueError(
                "selection must return one probability per source frequency"
            )
        return probabilities

    def expect(self, values: jax.Array) -> jax.Array:
        r"""Compute $Pv$: destination values to expectations at each source state.

        Input length is `plan.gene_copies + 1`; output length is
        `source_gene_copies + 1`. Signed values are supported. Shape mismatches
        raise `ValueError`, including when traced under JIT.
        """
        return _expect_probabilities(
            self.plan, self.probabilities, jnp.asarray(values, dtype=jnp.float64)
        )

    def advance(self, distribution: jax.Array) -> jax.Array:
        r"""Compute $P^T\mu$: source mass to next-generation destination mass.

        Input length is `source_gene_copies + 1`; output length is
        `plan.gene_copies + 1`. This transposes the implemented approximation
        exactly, including routing. It does not mutate or normalize the input.
        """
        distribution = jnp.asarray(distribution, dtype=jnp.float64)
        if distribution.shape != (self.source_gene_copies + 1,):
            raise ValueError("distribution must have source_gene_copies + 1 entries")
        spec = jax.ShapeDtypeStruct((self.plan.gene_copies + 1,), jnp.float64)
        return jax.linear_transpose(self.expect, spec)(distribution)[0]

    def expect_segregating(self, values: jax.Array) -> jax.Array:
        """Expect interior-state values, treating both destination endpoints as zero.

        Input has `plan.gene_copies - 1` entries; output has
        `source_gene_copies - 1`. Source and destination rows must represent
        complete allele-count grids. Endpoint loss is not conditioned away.
        """
        values = jnp.asarray(values, dtype=jnp.float64)
        if self.source_gene_copies < 2 or values.shape != (self.plan.gene_copies - 1,):
            raise ValueError(
                "segregating values require interior source and destination states"
            )
        return self.expect(jnp.pad(values, (1, 1)))[1:-1]

    def advance_segregating(self, distribution: jax.Array) -> jax.Array:
        """Advance interior mass, removing trajectories that reach either endpoint.

        Input has `source_gene_copies - 1` entries; output has
        `plan.gene_copies - 1`. Total mass may decrease; no renormalization occurs.
        """
        distribution = jnp.asarray(distribution, dtype=jnp.float64)
        if self.source_gene_copies < 2 or distribution.shape != (
            self.source_gene_copies - 1,
        ):
            raise ValueError("segregating distribution must match the source interior")
        return self.advance(jnp.pad(distribution, (1, 1)))[1:-1]

    def as_lineax(self, *, segregating: bool = False) -> lx.FunctionLinearOperator:
        """Return an expectation operator; its transpose advances distributions.

        `segregating=True` excludes endpoint states. No symmetry or positive
        definiteness tags are attached. Rectangular operators are supported.
        """
        size = self.plan.gene_copies - 1 if segregating else self.plan.gene_copies + 1
        return lx.FunctionLinearOperator(
            self.expect_segregating if segregating else self.expect,
            jax.ShapeDtypeStruct((size,), jnp.float64),
        )

    @eqx.filter_jit
    def expected_visits(
        self,
        *,
        solver: Literal["gmres", "bicgstab"] = "gmres",
        rtol: float = 1e-9,
        atol: float = 1e-11,
        max_steps: int = 2000,
        restart: int = 50,
    ) -> ExpectedVisitsResult:
        """Solve expected interior visits from one copy, including generation zero.

        Requires a square, constant-size transition on a complete allele-count
        grid. Visits end at the first endpoint hit, even if a custom selection
        map would otherwise permit leaving endpoints. The chain must eventually
        hit an endpoint for finite visits to exist.

        Solver controls and failure diagnostics are preserved in the returned
        `ExpectedVisitsResult`. Always check `converged`: a small residual only
        certifies the approximate operator. Invalid static controls raise
        `ValueError`; numerical nonconvergence is returned without throwing.
        Global differentiability of solves is not certified.
        """
        if self.source_gene_copies != self.plan.gene_copies:
            raise ValueError("expected_visits requires a square transition")
        return _expected_visits(
            self.as_lineax(segregating=True),
            self.plan.gene_copies,
            solver=solver,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            restart=restart,
        )
