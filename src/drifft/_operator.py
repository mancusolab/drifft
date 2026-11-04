"""Transition operations driven by frequency maps."""

# pattern: Functional Core; pure operations over a prepared transition backend.
from functools import partial
from typing import Literal, final

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from ._backend import AbstractTransitionBackend
from ._frequency import AbstractFrequencyMap
from ._solve import ExpectedVisitsResult, _solve_expected_visits


@final
class Transition(eqx.Module):
    r"""A reusable backend combined with a frequency map or explicit probabilities.

    Given source-row probability $q_i$ and $M=2N$ for diploid destination size
    $N$ (`backend.diploid_size`),

    $$P_{ij} = \binom{M}{j} q_i^j (1-q_i)^{M-j}.$$

    The backend controls how $P$ is represented and evaluated.

    Segregating operations restrict the kernel to interior allele counts and
    discard endpoint mass. They require complete, ordered allele-count grids;
    sampled probability rows are for full-state operations.
    """

    backend: AbstractTransitionBackend
    frequency_map: AbstractFrequencyMap | None
    _probabilities: jax.Array | None
    source_diploid_size: int | None = eqx.field(static=True)
    _source_gene_copies: int = eqx.field(static=True)

    def __init__(
        self,
        backend: AbstractTransitionBackend,
        frequency_map: AbstractFrequencyMap | None = None,
        *,
        source_diploid_size: int | None = None,
        probabilities: jax.Array | None = None,
    ) -> None:
        r"""Bind a frequency map or explicit probabilities to prepared geometry.

        For a frequency map $q$, source-row probabilities are $q_i=q(i/M_s)$,
        where $M_s$ is `2 * source_diploid_size`. Explicit probabilities supply
        the $q_i$ directly.

        **Arguments:**

        - `backend`: Prepared destination backend.
        - `frequency_map`: `AbstractFrequencyMap`; mutually exclusive with `probabilities`.
        - `source_diploid_size`: Positive integer diploid source size. Defaults to
          `backend.diploid_size` for a map. For explicit rows, inferred as
          `(len(probabilities) - 1) // 2` when the length is odd and at least three;
          otherwise remains `None`. If supplied, requires `2 * source_diploid_size + 1` rows.
        - `probabilities`: Optional nonempty vector of source probabilities in `[0, 1]`;
          converted to float64. Rows may be arbitrary sampled probabilities for
          full-state operations; segregating operations require ordered count grids.

        **Returns:**

        `None`. Reuses the backend without rebuilding geometry.

        **Raises:**

        `ValueError` for ambiguous configuration or invalid source dimensions or
        probability shape; `TypeError` for an invalid backend or frequency-map type.

        Finiteness and probability ranges are caller preconditions, not checked
        here. Enable JAX x64 before preparing the backend. For derivatives through
        explicit probabilities, compute them inside the differentiated function.
        """
        if not isinstance(backend, AbstractTransitionBackend):
            raise TypeError("backend must be an AbstractTransitionBackend")
        self.backend = backend
        self.frequency_map = frequency_map
        self._probabilities = (
            None
            if probabilities is None
            else jnp.asarray(probabilities, dtype=jnp.float64)
        )
        if source_diploid_size is not None and (
            isinstance(source_diploid_size, bool)
            or not isinstance(source_diploid_size, int)
            or source_diploid_size < 1
        ):
            raise ValueError("source_diploid_size must be a Python integer >= 1")
        if self._probabilities is None:
            self.source_diploid_size = (
                backend.diploid_size
                if source_diploid_size is None
                else source_diploid_size
            )
            self._source_gene_copies = 2 * self.source_diploid_size
        else:
            self._source_gene_copies = self._probabilities.size - 1
            inferred = (
                self._source_gene_copies // 2
                if self._source_gene_copies >= 2 and self._source_gene_copies % 2 == 0
                else None
            )
            self.source_diploid_size = (
                inferred if source_diploid_size is None else source_diploid_size
            )
            if (
                source_diploid_size is not None
                and 2 * source_diploid_size != self._source_gene_copies
            ):
                raise ValueError(
                    "source_diploid_size must match the probability vector"
                )

    def __check_init__(self) -> None:
        """Validate static dimensions and mutually exclusive construction modes."""
        if (self.frequency_map is None) == (self._probabilities is None):
            raise ValueError("Provide exactly one of frequency_map or probabilities")
        if self.frequency_map is not None:
            if not isinstance(self.frequency_map, AbstractFrequencyMap):
                raise TypeError(
                    "frequency_map must be an AbstractFrequencyMap; wrap functions in FrequencyMap"
                )
        elif self._probabilities is not None:
            if self._probabilities.ndim != 1 or self._probabilities.size == 0:
                raise ValueError("probabilities must be a nonempty vector")

    def dest_size(self) -> int:
        r"""Number of destination states (columns of the transition matrix $P$).

        Each column represents a destination allele count. For diploid
        destination size $N_d$, there are $2N_d+1$ columns, from zero copies
        (loss) to $2N_d$ copies (fixation).

        **Arguments:**

        None.

        **Returns:**

        The Python integer `2 * backend.diploid_size + 1`.
        """
        return self.backend._gene_copies + 1

    def source_size(self) -> int:
        r"""Number of source states (rows of the transition matrix $P$).

        Each row represents a source allele count. For diploid source size
        $N_s$, there are $2N_s+1$ rows, including loss and fixation. With
        explicit probabilities, each supplied probability defines one row.

        **Arguments:**

        None.

        **Returns:**

        The Python integer `2 * source_diploid_size + 1` for a frequency map,
        or the number of supplied rows for explicit probabilities.
        """
        return self._source_gene_copies + 1

    @property
    def probabilities(self) -> jax.Array:
        r"""Current float64 source probabilities, shape `(source_size(),)`.

        Evaluates the bound map on each access, or returns the explicit rows
        with their supplied length.
        Raises `ValueError` if the map returns an incompatible shape.
        """
        if self._probabilities is not None:
            return self._probabilities
        assert self.frequency_map is not None
        frequencies = (
            jnp.arange(self._source_gene_copies + 1, dtype=jnp.float64)
            / self._source_gene_copies
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
    def expect(self, values: jax.Array, *, segregating: bool = False) -> jax.Array:
        r"""Compute destination-value expectations at each source state.

        For destination values $v$, return

        $$ (Pv)_i = \sum_j P_{ij}v_j. $$

        With `segregating=True`, return $Qv$, where $Q$ restricts $P$ to
        interior source and destination counts. Endpoint values are zero;
        expectations are not conditioned on survival.

        **Arguments:**

        - `values`: Finite real vector of length `dest_size()`,
          or `dest_size() - 2` when segregating. Converted to float64;
          signed entries are supported.
        - `segregating`: Use interior states with endpoint values fixed to zero.

        **Returns:**

        A float64 vector of length `source_size()`, or `source_size() - 2` when
        segregating. No survival conditioning.

        !!! example "Example: Expected allele count"

            ```python
            # For each current count, find the expected allele count next generation.
            counts = jnp.arange(transition.dest_size())
            expected_counts = transition.expect(counts)
            ```

        **Raises:**

        `ValueError` for incompatible input/map-output shapes or a nonboolean flag.
        """
        if not isinstance(segregating, bool):
            raise ValueError("segregating must be a Python bool")
        values = jnp.asarray(values, dtype=jnp.float64)
        if segregating:
            if self.source_diploid_size is None or values.shape != (
                self.backend._gene_copies - 1,
            ):
                raise ValueError(
                    "segregating values must match the destination interior"
                )
            expectations = self.backend.expect(
                self.probabilities, jnp.pad(values, (1, 1))
            )
            return expectations[1:-1]
        return self.backend.expect(self.probabilities, values)

    @eqx.filter_jit
    def advance(
        self, distribution: jax.Array, *, segregating: bool = False
    ) -> jax.Array:
        r"""Propagate source mass by one generation.

        For source mass $p$, return

        $$ (P^T p)_j = \sum_i P_{ij}p_i. $$

        With `segregating=True`, return $Q^T p$, where $Q$ restricts $P$ to
        interior source and destination counts. Mass reaching an endpoint
        is discarded without renormalization.

        **Arguments:**

        - `distribution`: Finite real vector of length `source_size()`,
          or `source_size() - 2` when segregating. Converted to float64;
          signed entries are supported.
        - `segregating`: Discard endpoint hits after the transition.

        **Returns:**

        A float64 vector of length `dest_size()`, or `dest_size() - 2` when
        segregating. No renormalization occurs.

        !!! example "Example: Offspring from a singleton"

            ```python
            # Starting with one copy, find the probability of each count next generation.
            initial = jnp.zeros(transition.source_size()).at[1].set(1)
            next_distribution = transition.advance(initial)
            ```

        **Raises:**

        `ValueError` for incompatible input/map-output shapes or a nonboolean flag.
        """
        distribution = jnp.asarray(distribution, dtype=jnp.float64)
        if not isinstance(segregating, bool):
            raise ValueError("segregating must be a Python bool")
        offset = -1 if segregating else 1
        if segregating and self.source_diploid_size is None:
            raise ValueError("segregating distribution requires source interior states")
        if distribution.shape != (self._source_gene_copies + offset,):
            raise ValueError(
                f"distribution must have {self._source_gene_copies + offset} entries "
                f"for the {'interior' if segregating else 'full'} source grid"
            )
        spec = jax.ShapeDtypeStruct((self.backend._gene_copies + offset,), jnp.float64)
        # Transpose the actual approximate expectation map, rather than a
        # separately implemented forward kernel, to preserve adjoint pairing.
        return jax.linear_transpose(
            partial(self.expect, segregating=segregating), spec
        )(distribution)[0]

    def as_lineax(self, *, segregating: bool = False) -> lx.FunctionLinearOperator:
        r"""Expose the expectation operation as a Lineax operator.

        The operator acts as $v \mapsto Pv$, and its transpose acts as
        $p \mapsto P^T p$. With `segregating=True`, both actions use the
        interior kernel $Q$ in place of $P$.

        **Arguments:**

        - `segregating`: Exclude endpoint states when true; defaults to false.

        **Returns:**

        A rectangular `lineax.FunctionLinearOperator` from destination values to source
        expectations. Its transpose advances mass. No symmetry or definiteness tags
        are attached.

        **Raises:**

        `ValueError` for incompatible dimensions/map-output shape or a nonboolean flag.
        """
        if not isinstance(segregating, bool):
            raise ValueError("segregating must be a Python bool")
        size = (
            self.backend._gene_copies - 1
            if segregating
            else self.backend._gene_copies + 1
        )
        return lx.FunctionLinearOperator(
            partial(self.expect, segregating=segregating),
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
        throw: bool = True,
    ) -> ExpectedVisitsResult:
        r"""Compute interior visits from one copy, including the initial generation.

        For the square interior kernel $Q$, solve

        $$ (I-Q^T)u=e_1, \qquad u=\sum_{t=0}^{\infty}(Q^T)^t e_1. $$

        Here $e_1$ places unit mass at allele count one, and $u_i$ is the
        expected number of visits to interior count $i$. The $t=0$ term
        includes the initial generation. Finite visits require eventual
        endpoint hitting; mutation does not restore discarded trajectories.

        **Arguments:**

        - `solver`: `"gmres"` (default) or `"bicgstab"`.
        - `rtol`: Nonnegative relative tolerance; default `1e-9`.
        - `atol`: Nonnegative absolute tolerance; default `1e-11`.
          Both tolerances must be finite and at least one must be positive.
        - `max_steps`: Positive iteration limit; default 2000 (outer steps for GMRES).
        - `restart`: Positive GMRES subspace size; default 50, capped at the number
          of interior states. Validated but unused for BiCGStab.
        - `throw`: Raise on numerical failure by default. False returns the
          candidate and diagnostics; under vmap, true may fail the whole batch.

        **Returns:**

        An `ExpectedVisitsResult` with float64 `visits` of length
        `dest_size() - 2`, plus convergence diagnostics.

        !!! example "Example: Visits from a singleton"

            ```python
            result = transition.expected_visits()
            visits = result.visits
            ```

        **Raises:**

        `ValueError` for a nonsquare transition, invalid solver controls, or map shape.
        Runtime errors for failed solves, nonfinite output, or rejected residuals
        with `throw=True`.

        !!! warning "Numerical limitations"

            With `throw=False`, check `converged` before using visits; under vmap,
            check each result. Visits stop at the first endpoint
            hit, even with mutation. Long absorption times can amplify backend
            errors despite convergence; check [numerical accuracy](../guide/numerics.md).
            Global differentiability of solves is not certified.
        """
        if self._source_gene_copies != self.backend._gene_copies:
            raise ValueError("expected_visits requires a square transition")
        return _solve_expected_visits(
            self.as_lineax(segregating=True),
            self.backend._gene_copies,
            solver=solver,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            restart=restart,
            throw=throw,
        )
