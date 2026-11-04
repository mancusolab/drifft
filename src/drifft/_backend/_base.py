"""Abstract contract for prepared, selection-independent transition backends."""

# Functional Core: numerical interface, with no concrete storage requirements.
from abc import abstractmethod
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp


class AbstractTransitionBackend(eqx.Module):
    r"""Interface for prepared geometry with a linear expectation operation.

    For diploid destination population size $N$ (`pop_size`), let $M=2N$.
    Given offspring allele probability $q_i$ for source row $i$, define

    $$P_{ij}=\binom{M}{j}q_i^j(1-q_i)^{M-j},\qquad j=0,\ldots,M.$$

    For destination values $v_j$, backends evaluate the expectations
    $(Pv)_i=\sum_{j=0}^{M}P_{ij}v_j$.

    Implementations provide `pop_size` and an `expect` operation linear in
    destination values, supporting transposition, JVP, VJP, and batching.
    Prepare geometry outside JIT; use `Transition` for compiled application.
    """

    pop_size: eqx.AbstractVar[int]

    @property
    def _gene_copies(self) -> int:
        """Number of allele-bearing copies in the diploid population."""
        return 2 * self.pop_size

    @abstractmethod
    def expect(self, probabilities: jax.Array, values: jax.Array) -> jax.Array:
        r"""Compute destination-value expectations at each source probability.

        For each supplied allele probability $q_i$, average destination values
        $v_j$ over the possible allele counts in the next generation. With
        $M = 2N$ for destination population size $N$ (`pop_size`), return

        $$ (Pv)_i = \sum_{j=0}^{M}\binom{M}{j}q_i^j(1-q_i)^{M-j}v_j. $$

        !!! example "Example: Expected allele count"

            ```python
            # For each source probability, find the expected count next generation.
            counts = jnp.arange(2 * backend.pop_size + 1)
            expected_counts = backend.expect(probabilities, counts)
            ```

        **Arguments:**

        - `probabilities`: Vector of finite source probabilities in `[0, 1]`.
        - `values`: Finite real destination vector of length `2 * pop_size + 1`.

        Both inputs accept Python lists, NumPy arrays, or JAX arrays and are
        converted to float64 before shape validation.

        **Returns:**

        A float64 vector with the same shape as `probabilities`, linear in `values`.

        **Raises:**

        `ValueError` for incompatible vector shapes, including during JIT tracing.

        !!! warning "Numerical preconditions"

            Finiteness and probability ranges are unchecked. Outputs are not
            clipped or normalized; approximation and derivative limitations
            depend on the concrete backend. Roundoff can vary across devices.
        """
        ...

    def advance(self, probabilities: jax.Array, distribution: jax.Array) -> jax.Array:
        r"""Propagate source mass to destination allele counts.

        Each source row distributes its mass over the possible destination
        counts according to its offspring allele probability. For source mass
        $p$, return

        $$ (P^\mathsf{T}p)_j = \sum_i P_{ij}p_i. $$

        !!! example "Example: Offspring from one source state"

            ```python
            # Find the offspring count distribution from the first source state.
            initial = jnp.zeros(len(probabilities)).at[0].set(1)
            offspring = backend.advance(probabilities, initial)
            ```

        !!! note "Forward propagation by transposition"

            With probabilities fixed, `expect` is linear in destination values.
            The default implementation uses `jax.linear_transpose` to obtain
            $P^\mathsf{T}p$, including the approximation used by `expect`.
            Backends may override `advance` with a specialized implementation
            that preserves this transpose relationship.

        **Arguments:**

        - `probabilities`: Vector of finite source probabilities in `[0, 1]`.
        - `distribution`: Finite real source mass vector with the same shape as
          `probabilities`. Signed entries are supported.

        Both inputs accept Python lists, NumPy arrays, or JAX arrays and are
        converted to float64 before shape validation.

        **Returns:**

        A float64 vector of length `2 * pop_size + 1`, linear in `distribution`.
        Outputs are not clipped or normalized. Numerical preconditions and
        approximation limitations are the same as for `expect`.

        **Raises:**

        `ValueError` for incompatible vector shapes, including during JIT tracing.
        """
        probabilities = jnp.asarray(probabilities, dtype=jnp.float64)
        distribution = jnp.asarray(distribution, dtype=jnp.float64)
        if probabilities.ndim != 1 or distribution.shape != probabilities.shape:
            raise ValueError(
                "advance requires probability and distribution vectors of the same shape"
            )
        spec = jax.ShapeDtypeStruct((self._gene_copies + 1,), jnp.float64)
        return jax.linear_transpose(partial(self.expect, probabilities), spec)(
            distribution
        )[0]
