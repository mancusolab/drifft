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
        """Map destination values to expectations at arbitrary source probabilities."""
        ...
