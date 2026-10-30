"""Abstract contract for prepared, selection-independent transition backends."""

# Functional Core: numerical interface, with no concrete storage requirements.
from abc import abstractmethod

import equinox as eqx
import jax


class AbstractTransitionBackend(eqx.Module):
    r"""Interface for prepared geometry with a linear expectation operation.

    For destination gene-copy count $M$ and values $v_j$, backends evaluate the
    binomial expectation

    $$B_M[v](q)=\sum_{j=0}^{M}\binom{M}{j}q^j(1-q)^{M-j}v_j.$$

    Implementations provide `gene_copies` and an `expect` operation linear in
    destination values, supporting transposition, JVP, VJP, and batching.
    Prepare geometry outside JIT; use `Transition` for compiled application.
    """

    gene_copies: eqx.AbstractVar[int]

    @abstractmethod
    def expect(self, probabilities: jax.Array, values: jax.Array) -> jax.Array:
        r"""Compute binomial expectations at the supplied probabilities.

        **Arguments:**

        - `probabilities`: Float64 vector of finite source probabilities in `[0, 1]`.
        - `values`: Finite float64 destination vector of length `gene_copies + 1`.

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
