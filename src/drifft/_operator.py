"""Matrix-free transition operations driven by frequency maps."""

# pattern: Functional Core; pure operations over a prepared transition backend.
from typing import Literal, final

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from ._frequency import AbstractFrequencyMap
from ._solve import ExpectedVisitsResult, _solve_expected_visits
from .backends import AbstractTransitionBackend


@final
class Transition(eqx.Module):
    r"""A reusable backend combined with a frequency map or explicit probabilities.

    **Arguments:**

    - `backend`: Geometry for the destination population, prepared outside JIT.
    - `frequency_map`: An `AbstractFrequencyMap` mapping source frequencies to
      offspring probabilities. Wrap custom functions in `FrequencyMap`.
    - `source_gene_copies`: Source population size; defaults to the backend's size.
    - `probabilities`: Alternative to `frequency_map`; normally supplied through
      `Transition.from_probabilities`. Each entry defines one source row.

    **Returns:**

    An immutable PyTree. Construction binds existing geometry without rebuilding
    it. Offspring probabilities are evaluated during application, so updates and
    differentiation through model parameters cannot use a stale cached vector.

    **Raises:**

    `ValueError` for ambiguous configuration, invalid source dimensions, or an
    empty/non-vector probability array; `TypeError` for a map not implementing
    `AbstractFrequencyMap`.

    **Failure Modes:**

    Probabilities must be finite and in `[0, 1]`. No runtime callbacks enforce
    these value preconditions. Approximation errors are those of the backend; no
    outputs are clipped or renormalized. Methods are pure and return new arrays.
    Numerical methods compile automatically with `eqx.filter_jit`. Array leaves
    remain dynamic across calls; changing shapes or static configuration can
    trigger recompilation. Methods also compose inside an outer compiled workflow.
    """

    backend: AbstractTransitionBackend
    frequency_map: AbstractFrequencyMap | None
    _probabilities: jax.Array | None
    source_gene_copies: int = eqx.field(static=True)

    def __init__(
        self,
        backend: AbstractTransitionBackend,
        frequency_map: AbstractFrequencyMap | None = None,
        *,
        source_gene_copies: int | None = None,
        probabilities: jax.Array | None = None,
    ) -> None:
        """Bind destination geometry and either a frequency map or explicit rows."""
        self.backend = backend
        self.frequency_map = frequency_map
        self._probabilities = (
            None
            if probabilities is None
            else jnp.asarray(probabilities, dtype=jnp.float64)
        )
        inferred = (
            backend.gene_copies
            if self._probabilities is None
            else self._probabilities.size - 1
        )
        self.source_gene_copies = (
            inferred if source_gene_copies is None else source_gene_copies
        )

    def __check_init__(self) -> None:
        """Validate static dimensions and mutually exclusive construction modes."""
        if (self.frequency_map is None) == (self._probabilities is None):
            raise ValueError("Provide exactly one of frequency_map or probabilities")
        if not isinstance(self.source_gene_copies, int) or isinstance(
            self.source_gene_copies, bool
        ):
            raise ValueError("source_gene_copies must be an integer")
        if self.frequency_map is not None:
            if not isinstance(self.frequency_map, AbstractFrequencyMap):
                raise TypeError(
                    "frequency_map must be an AbstractFrequencyMap; wrap functions in FrequencyMap"
                )
            if self.source_gene_copies < 2:
                raise ValueError("source_gene_copies must be >= 2 for a frequency map")
        elif self._probabilities is not None:
            if self._probabilities.ndim != 1 or self._probabilities.size == 0:
                raise ValueError("probabilities must be a nonempty vector")
            if self._probabilities.size != self.source_gene_copies + 1:
                raise ValueError("source_gene_copies must match the probability vector")

    @classmethod
    def from_probabilities(
        cls, backend: AbstractTransitionBackend, probabilities: jax.Array
    ) -> "Transition":
        """Bind one probability per source row; destination size comes from backend.

        Arrays remain dynamic leaves. To differentiate through their construction,
        evaluate the probability map inside the differentiated function. Rows may
        represent sampled probabilities rather than a complete allele-count grid.
        """
        return cls(backend, probabilities=probabilities)

    @property
    def probabilities(self) -> jax.Array:
        """Evaluate the frequency map on the source grid, or return supplied rows."""
        if self._probabilities is not None:
            return self._probabilities
        assert self.frequency_map is not None
        frequencies = (
            jnp.arange(self.source_gene_copies + 1, dtype=jnp.float64)
            / self.source_gene_copies
        )
        probabilities = jnp.asarray(
            self.frequency_map.apply(frequencies), dtype=jnp.float64
        )
        if probabilities.shape != frequencies.shape:
            raise ValueError(
                "frequency_map must return one probability per source frequency"
            )
        return probabilities

    @eqx.filter_jit
    def expect(self, values: jax.Array) -> jax.Array:
        r"""Compute $Pv$: destination values to expectations at each source state.

        Input length is `backend.gene_copies + 1`; output length is
        `source_gene_copies + 1`. Signed values are supported. Shape mismatches
        raise `ValueError`, including when traced under JIT.
        """
        return self.backend.expect(
            self.probabilities, jnp.asarray(values, dtype=jnp.float64)
        )

    @eqx.filter_jit
    def advance(self, distribution: jax.Array) -> jax.Array:
        r"""Compute $P^T\mu$: source mass to next-generation destination mass.

        Input length is `source_gene_copies + 1`; output length is
        `backend.gene_copies + 1`. This transposes the implemented approximation
        exactly, including routing. It does not mutate or normalize the input.
        """
        distribution = jnp.asarray(distribution, dtype=jnp.float64)
        if distribution.shape != (self.source_gene_copies + 1,):
            raise ValueError("distribution must have source_gene_copies + 1 entries")
        spec = jax.ShapeDtypeStruct((self.backend.gene_copies + 1,), jnp.float64)
        return jax.linear_transpose(self.expect, spec)(distribution)[0]

    @eqx.filter_jit
    def expect_segregating(self, values: jax.Array) -> jax.Array:
        """Expect interior-state values, treating both destination endpoints as zero.

        Input has `backend.gene_copies - 1` entries; output has
        `source_gene_copies - 1`. Source and destination rows must represent
        complete allele-count grids. Endpoint loss is not conditioned away.
        """
        values = jnp.asarray(values, dtype=jnp.float64)
        if self.source_gene_copies < 2 or values.shape != (
            self.backend.gene_copies - 1,
        ):
            raise ValueError(
                "segregating values require interior source and destination states"
            )
        return self.expect(jnp.pad(values, (1, 1)))[1:-1]

    @eqx.filter_jit
    def advance_segregating(self, distribution: jax.Array) -> jax.Array:
        """Advance interior mass, removing trajectories that reach either endpoint.

        Input has `source_gene_copies - 1` entries; output has
        `backend.gene_copies - 1`. Total mass may decrease; no renormalization occurs.
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
        size = (
            self.backend.gene_copies - 1
            if segregating
            else self.backend.gene_copies + 1
        )
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
        grid. Visits end at the first endpoint hit, even if a custom frequency map
        would otherwise permit leaving endpoints. The chain must eventually
        hit an endpoint for finite visits to exist.

        Solver controls and failure diagnostics are preserved in the returned
        `ExpectedVisitsResult`. Always check `converged`: a small residual only
        certifies the approximate operator. Invalid static controls raise
        `ValueError`; numerical nonconvergence is returned without throwing.
        Global differentiability of solves is not certified.
        """
        if self.source_gene_copies != self.backend.gene_copies:
            raise ValueError("expected_visits requires a square transition")
        return _solve_expected_visits(
            self.as_lineax(segregating=True),
            self.backend.gene_copies,
            solver=solver,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            restart=restart,
        )
