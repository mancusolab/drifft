"""Abstract contract for prepared, selection-independent transition backends."""

# Functional Core: numerical interface, with no concrete storage requirements.
from abc import abstractmethod

import equinox as eqx
import jax


class AbstractTransitionBackend(eqx.Module):
    """Prepared destination geometry with a linear expectation operation.

    Construct concrete backends outside JIT. Array fields remain dynamic leaves;
    shapes and algorithm choices are static. Probabilities must be finite and in
    [0, 1]. Implementations must preserve linearity in values and support its
    transpose, JVPs, VJPs, and batching. No clipping or normalization is allowed.
    """

    gene_copies: eqx.AbstractVar[int]

    @abstractmethod
    def expect(self, probabilities: jax.Array, values: jax.Array) -> jax.Array:
        r"""Map destination values to expectations at arbitrary source probabilities.

        **Arguments:**

        - `probabilities`: A vector of finite probabilities in `[0, 1]`.
        - `values`: A float64 destination vector of length `gene_copies + 1`.

        **Returns:**

        A vector shaped like `probabilities`, linear in `values` and compatible
        with JAX transposition, differentiation, and batching.

        **Raises:**

        Implementations reject incompatible shapes with `ValueError`. Numerical
        probability ranges remain caller preconditions inside traced execution.
        """
        ...
