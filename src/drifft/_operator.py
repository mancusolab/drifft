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

    Given source-row probability $q_i$ and destination gene-copy count $M$,

    $$P_{ij} = \binom{M}{j} q_i^j (1-q_i)^{M-j}.$$

    `expect` applies $P$ to destination values; `advance` applies $P^T$ to
    source mass. The backend controls how $P$ is represented and evaluated.
    For a frequency map, $q_i=q(i/M_s)$ where $M_s$ is `source_gene_copies`.

    Segregating operations restrict the kernel to interior allele counts and
    discard endpoint mass. They require complete, ordered allele-count grids;
    sampled probability rows are for full-state operations. For a square
    interior kernel $Q$, `expected_visits`
    solves $(I-Q^T)u=e_1$, including the initial generation. Finite visits require
    eventual endpoint hitting; mutation does not restore discarded trajectories.
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
        r"""Bind a frequency map or explicit probabilities to prepared geometry.

        **Arguments:**

        - `backend`: Prepared destination backend.
        - `frequency_map`: `AbstractFrequencyMap`; mutually exclusive with `probabilities`.
        - `source_gene_copies`: Source size, at least two for a map. Defaults to the
          backend size, or `len(probabilities) - 1` for explicit rows.
        - `probabilities`: Optional nonempty vector of source probabilities in `[0, 1]`;
          converted to float64. Rows may be arbitrary sampled probabilities for
          full-state operations; segregating operations require ordered count grids.

        **Returns:**

        `None`. Reuses the backend without rebuilding geometry.

        **Raises:**

        `ValueError` for ambiguous configuration or invalid source dimensions or
        probability shape; `TypeError` for an invalid backend or frequency-map type.

        Finiteness and probability ranges are caller preconditions, not checked
        here. Enable JAX x64 before preparing the backend.
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

    @property
    def probabilities(self) -> jax.Array:
        r"""Current float64 source probabilities, shape `(source_gene_copies + 1,)`.

        Evaluates the bound map on each access, or returns the explicit rows.
        Raises `ValueError` if the map returns an incompatible shape.
        """
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
    def expect(self, values: jax.Array, *, segregating: bool = False) -> jax.Array:
        r"""Compute destination-value expectations at each source state.

        **Arguments:**

        - `values`: Finite real vector of length `backend.gene_copies + 1`,
          or `backend.gene_copies - 1` when segregating. Converted to float64;
          signed entries are supported.
        - `segregating`: Use interior states with endpoint values fixed to zero.

        **Returns:**

        A float64 vector of length `source_gene_copies + 1`, or
        `source_gene_copies - 1` when segregating. No survival conditioning.

        **Raises:**

        `ValueError` for incompatible input/map-output shapes or a nonboolean flag.
        """
        if not isinstance(segregating, bool):
            raise ValueError("segregating must be a Python bool")
        values = jnp.asarray(values, dtype=jnp.float64)
        if segregating:
            if self.source_gene_copies < 2 or values.shape != (
                self.backend.gene_copies - 1,
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

        **Arguments:**

        - `distribution`: Finite real vector of length `source_gene_copies + 1`,
          or `source_gene_copies - 1` when segregating. Converted to float64;
          signed entries are supported.
        - `segregating`: Discard endpoint hits after the transition.

        **Returns:**

        A float64 vector of length `backend.gene_copies + 1`, or
        `backend.gene_copies - 1` when segregating. No renormalization occurs.

        **Raises:**

        `ValueError` for incompatible input/map-output shapes or a nonboolean flag.
        """
        distribution = jnp.asarray(distribution, dtype=jnp.float64)
        if not isinstance(segregating, bool):
            raise ValueError("segregating must be a Python bool")
        offset = -1 if segregating else 1
        if segregating and self.source_gene_copies < 2:
            raise ValueError("segregating distribution requires source interior states")
        if distribution.shape != (self.source_gene_copies + offset,):
            raise ValueError(
                f"distribution must have {self.source_gene_copies + offset} entries "
                f"for the {'interior' if segregating else 'full'} source grid"
            )
        spec = jax.ShapeDtypeStruct((self.backend.gene_copies + offset,), jnp.float64)
        # Transpose the actual approximate expectation map, rather than a
        # separately implemented forward kernel, to preserve adjoint pairing.
        return jax.linear_transpose(
            partial(self.expect, segregating=segregating), spec
        )(distribution)[0]

    def as_lineax(self, *, segregating: bool = False) -> lx.FunctionLinearOperator:
        r"""Expose the expectation operation as a Lineax operator.

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
            self.backend.gene_copies - 1
            if segregating
            else self.backend.gene_copies + 1
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
        `backend.gene_copies - 1`, plus convergence diagnostics.

        **Raises:**

        `ValueError` for a nonsquare transition, invalid solver controls, or map shape.
        Runtime errors for failed solves, nonfinite output, or rejected residuals
        with `throw=True`.

        !!! warning "Numerical limitations"

            With `throw=False`, check `converged` before using visits; under vmap,
            check each result. Visits stop at the first endpoint
            hit, even with mutation. Global differentiability of solves is not certified.
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
            throw=throw,
        )
