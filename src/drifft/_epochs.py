"""Common contracts for constant-size forward evolution."""

# pattern: Functional Core
from abc import abstractmethod
from typing import NamedTuple, final

import equinox as eqx
import jax
import jax.numpy as jnp

from ._frequency import AbstractFrequencyMap
from ._operator import Transition
from .backends import AbstractTransitionBackend


class EpochResult(NamedTuple):
    r"""Epoch output and an a posteriori power-action error estimate.

    `distribution` is the evolved mass; `error_estimate` is an absolute L1
    estimate for the power approximation only. `converged` indicates a finite
    output satisfying the backend's stopping criterion. It does not certify
    the underlying one-generation approximation.
    """

    distribution: jax.Array
    error_estimate: jax.Array
    converged: jax.Array


class AbstractEpochBackend(eqx.Module):
    r"""Interface for forward evolution at a fixed population size.

    An epoch evaluates $p_{t+1}=P^T p_t+b$ for a fixed frequency map and
    immigration vector $b$. In segregating mode, $P$ is restricted to interior
    counts and boundary mass is discarded each generation. Approximate epoch
    actions may depend on the input vector; no linear-transpose contract is
    imposed on this interface.
    """

    gene_copies: eqx.AbstractVar[int]

    @abstractmethod
    def advance(
        self,
        distribution: jax.Array,
        *,
        frequency_map: AbstractFrequencyMap,
        generations: int,
        segregating: bool = False,
        immigration: jax.Array | None = None,
        throw: bool = True,
    ) -> EpochResult:
        r"""Advance through a constant-size epoch.

        **Arguments:**

        - `distribution`: Finite real mass vector, converted to float64; length
          `gene_copies + 1`, or `gene_copies - 1` in segregating mode.
        - `frequency_map`: Process shared by every generation.
        - `generations`: Nonnegative Python integer; static under JIT.
        - `segregating`: Discard absorbing endpoints at every step.
        - `immigration`: Optional mass vector added after each step, same shape
          as `distribution`. Zero generations leaves the input unchanged.
        - `throw`: Raise on nonfinite output or failed power approximation.
          Under vmap, a failing member may fail the whole batch.

        **Returns:**

        `EpochResult` with evolved mass, an error estimate, and convergence flag.

        **Raises:**

        `ValueError` for invalid shapes or static configuration; `TypeError` for
        unsupported frequency maps. Runtime failures use `equinox.error_if`.

        !!! note "Transforms"

            Enable JAX x64. Geometry is prepared outside JIT; numerical methods
            support JIT and batching. Derivatives follow the concrete algorithm,
            with limitations described by each backend.
        """
        ...


def _epoch_inputs(distribution, immigration, gene_copies, generations, segregating):
    """Normalize shared epoch inputs and enforce their static shape contract."""
    if (
        isinstance(generations, bool)
        or not isinstance(generations, int)
        or generations < 0
    ):
        raise ValueError("generations must be a nonnegative Python integer")
    if not isinstance(segregating, bool):
        raise ValueError("segregating must be a Python bool")
    distribution = jnp.asarray(distribution, dtype=jnp.float64)
    size = gene_copies - 1 if segregating else gene_copies + 1
    if distribution.shape != (size,):
        raise ValueError(f"distribution must have {size} entries")
    immigration = (
        jnp.zeros_like(distribution)
        if immigration is None
        else jnp.asarray(immigration, dtype=jnp.float64)
    )
    if immigration.shape != distribution.shape:
        raise ValueError("immigration must have the same shape as distribution")
    return distribution, immigration


def _epoch_result(distribution, error, tolerance, throw):
    """Apply the common finite-output and approximation-failure policy."""
    converged = (
        jnp.all(jnp.isfinite(distribution)) & jnp.isfinite(error) & (error <= tolerance)
    )
    if throw:
        distribution = eqx.error_if(
            distribution,
            ~converged,
            "epoch failed: nonfinite output or unconverged power action; check inputs "
            "and increase krylov_dimension or use Sequential",
        )
    return EpochResult(distribution, error, converged)


@final
class Sequential(AbstractEpochBackend):
    r"""Evaluate an epoch by repeated FFT or NUFFT transitions.

    This applies the prepared one-generation operator once per generation.
    There is no additional power approximation, so the reported error estimate
    is zero. The underlying transition's numerical errors still apply.
    """

    backend: AbstractTransitionBackend
    gene_copies: int = eqx.field(static=True)

    def __init__(self, backend: AbstractTransitionBackend) -> None:
        r"""Prepare sequential evolution from reusable transition geometry.

        **Arguments:**

        - `backend`: Prepared FFT or NUFFT one-generation backend.

        **Returns:**

        `None`.

        **Raises:**

        `TypeError` if `backend` does not implement `AbstractTransitionBackend`.
        """
        if not isinstance(backend, AbstractTransitionBackend):
            raise TypeError("backend must be an AbstractTransitionBackend")
        self.backend = backend
        self.gene_copies = backend.gene_copies

    @eqx.filter_jit
    def advance(
        self,
        distribution: jax.Array,
        *,
        frequency_map: AbstractFrequencyMap,
        generations: int,
        segregating: bool = False,
        immigration: jax.Array | None = None,
        throw: bool = True,
    ) -> EpochResult:
        r"""Advance with the input and result contracts of `AbstractEpochBackend.advance`."""
        distribution, immigration = _epoch_inputs(
            distribution, immigration, self.gene_copies, generations, segregating
        )
        transition = Transition(self.backend, frequency_map)
        operation = (
            transition.advance_segregating if segregating else transition.advance
        )
        distribution = jax.lax.fori_loop(
            0, generations, lambda _, mass: operation(mass) + immigration, distribution
        )
        return _epoch_result(distribution, jnp.asarray(0.0), 0.0, throw)
