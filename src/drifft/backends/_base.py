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

    **Arguments:**

    Concrete constructors define their own geometry controls. Implementations
    provide `gene_copies`, the integer destination gene-copy count, and `expect`.

    **Returns:**

    A backend implementing expectations linear in destination values, with
    transposition, JVP, VJP, and batching support. Construct concrete geometry
    outside JIT; arrays remain dynamic leaves and dimensions remain static.

    **Failure Modes:**

    Finite inputs and probabilities in `[0, 1]` are caller preconditions.
    Concrete implementations document approximation and derivative limitations.
    No clipping or normalization is allowed; roundoff can vary across devices.
    Backend methods are not JIT-decorated; `Transition` provides compiled calls.
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
        """
        ...
