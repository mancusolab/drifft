"""Abstract contract for prepared, selection-independent transition backends."""

# Functional Core: numerical interface, with no concrete storage requirements.
from abc import abstractmethod

import equinox as eqx
import jax


class AbstractTransitionBackend(eqx.Module):
    r"""Interface for prepared geometry with a linear expectation operation.

    For diploid destination size $N$ (`diploid_size`), let $M=2N$. For values
    $f_j$ at allele counts $j=0,\ldots,M$, backends evaluate the binomial expectation

    $$B_M[f](q)=\sum_{j=0}^{M}\binom{M}{j}q^j(1-q)^{M-j}f_j.$$

    Implementations provide `diploid_size` and an `expect` operation linear in
    destination values, supporting transposition, JVP, VJP, and batching.
    Prepare geometry outside JIT; use `Transition` for compiled application.
    """

    diploid_size: eqx.AbstractVar[int]

    @property
    def _gene_copies(self) -> int:
        """Number of allele-bearing copies in the diploid population."""
        return 2 * self.diploid_size

    @abstractmethod
    def expect(self, probabilities: jax.Array, values: jax.Array) -> jax.Array:
        r"""Compute binomial expectations at the supplied probabilities.

        **Arguments:**

        - `probabilities`: Vector of finite source probabilities in `[0, 1]`.
        - `values`: Finite real destination vector of length `2 * diploid_size + 1`.

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
