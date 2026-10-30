"""Compact real symmetric pentadiagonal operators and reusable Lineax solves."""

from typing import NamedTuple, final

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx


@final
class SymmetricPentadiagonalOperator(lx.AbstractLinearOperator):
    """A real symmetric matrix stored as its diagonal and two off-diagonals.

    For size n, the band lengths are n, n-1, and max(n-2, 0). Bands must share
    a real floating dtype. Symmetry is structural; positive definiteness is
    checked by the solver's factorization, not assumed by the operator.
    """

    diagonal: jax.Array = eqx.field(converter=jnp.asarray)
    adjacent: jax.Array = eqx.field(converter=jnp.asarray)
    second: jax.Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        """Reject inconsistent band dimensions and unsupported scalar types."""
        if self.diagonal.ndim != 1 or self.diagonal.size < 1:
            raise ValueError("diagonal must be a nonempty vector")
        size = self.diagonal.size
        if self.adjacent.shape != (size - 1,) or self.second.shape != (
            max(size - 2, 0),
        ):
            raise ValueError("band lengths must be n, n-1, and max(n-2, 0)")
        if not jnp.issubdtype(self.diagonal.dtype, jnp.floating):
            raise ValueError("bands must have a real floating dtype")
        if any(
            band.dtype != self.diagonal.dtype for band in (self.adjacent, self.second)
        ):
            raise ValueError("all bands must have the same dtype")

    def mv(self, vector):
        """Apply all five bands in linear work without materializing a matrix."""
        result = self.diagonal * vector
        result = result.at[:-1].add(self.adjacent * vector[1:])
        result = result.at[1:].add(self.adjacent * vector[:-1])
        result = result.at[:-2].add(self.second * vector[2:])
        return result.at[2:].add(self.second * vector[:-2])

    def as_matrix(self):
        """Materialize a dense matrix explicitly for inspection or references."""
        size = self.diagonal.size
        matrix = jnp.diag(self.diagonal)
        first = jnp.arange(size - 1)
        second = jnp.arange(max(size - 2, 0))
        matrix = matrix.at[first, first + 1].set(self.adjacent)
        matrix = matrix.at[first + 1, first].set(self.adjacent)
        matrix = matrix.at[second, second + 2].set(self.second)
        return matrix.at[second + 2, second].set(self.second)

    def transpose(self):
        """Reuse this structurally symmetric real operator."""
        return self

    def in_structure(self):
        """Describe the real vector accepted by this operator."""
        return jax.ShapeDtypeStruct(self.diagonal.shape, self.diagonal.dtype)

    def out_structure(self):
        """Describe the output vector, identical to the input structure."""
        return self.in_structure()


@lx.is_symmetric.register(SymmetricPentadiagonalOperator)
def _is_symmetric(operator):
    """Declare only the symmetry guaranteed by the compact representation."""
    return True


@lx.linearise.register(SymmetricPentadiagonalOperator)
@lx.materialise.register(SymmetricPentadiagonalOperator)
@lx.conj.register(SymmetricPentadiagonalOperator)
def _unchanged_operator(operator):
    """Retain explicit real bands for Lineax transformations and implicit AD."""
    return operator


class _PentadiagonalState(NamedTuple):
    """Two-band Cholesky factors and a reusable positive-pivot validity check."""

    roots: jax.Array
    lower1: jax.Array
    lower2: jax.Array
    valid: jax.Array


def _triangular_solve(rhs, diagonal, adjacent, second):
    """Substitute through two lower bands using only the preceding two values."""

    def step(carry, entries):
        previous, previous_previous = carry
        value, root, first, outer = entries
        result = (value - first * previous - outer * previous_previous) / root
        return (result, previous), result

    zero = jnp.zeros((), dtype=rhs.dtype)
    return jax.lax.scan(step, (zero, zero), (rhs, diagonal, adjacent, second))[1]


@final
class PentadiagonalCholesky(lx.AbstractLinearSolver[_PentadiagonalState]):
    """Solve real SPD pentadiagonal systems in linear storage and work.

    Pass explicitly to lineax.linear_solve. Init factors the bands once; the
    returned state can be reused for multiple right-hand sides and adjoints.
    Nonpositive or nonfinite pivots report a singular-system result.
    """

    def init(self, operator, options):
        """Compute two-band Cholesky factors without dense materialization."""
        del options
        if not isinstance(operator, SymmetricPentadiagonalOperator):
            raise TypeError(
                "PentadiagonalCholesky requires SymmetricPentadiagonalOperator"
            )
        size = operator.diagonal.size
        adjacent = jnp.pad(operator.adjacent, (1, 0))
        second = jnp.pad(operator.second, (2, 0))[:size]

        def factor(carry, entries):
            # Each row depends only on two preceding pivots and one band entry.
            # Leading zero bands make the initial dummy pivots harmless.
            previous, previous_previous, previous_lower = carry
            diagonal, first, outer = entries
            lower2 = outer / previous_previous
            lower1 = (first - lower2 * previous_lower) / previous
            root = jnp.sqrt(diagonal - lower1**2 - lower2**2)
            return (root, previous, lower1), (root, lower1, lower2)

        one = jnp.ones((), dtype=operator.diagonal.dtype)
        zero = jnp.zeros((), dtype=operator.diagonal.dtype)
        _, (roots, lower1, lower2) = jax.lax.scan(
            factor, (one, one, zero), (operator.diagonal, adjacent, second)
        )
        valid = jnp.all(jnp.isfinite(roots) & (roots > 0))
        return _PentadiagonalState(roots, lower1, lower2, valid)

    def compute(self, state, vector, options):
        """Reuse the factors for two banded substitutions and return Lineax status."""
        del options
        roots, lower1, lower2, valid = state
        forward = _triangular_solve(vector, roots, lower1, lower2)
        # Shift lower bands into the upper-row convention before reversing the
        # coordinates, so the same substitution handles the transposed factor.
        upper1 = jnp.pad(lower1[1:], (0, 1))
        upper2 = jnp.pad(lower2[2:], (0, min(2, roots.size)))
        value = _triangular_solve(
            forward[::-1], roots[::-1], upper1[::-1], upper2[::-1]
        )[::-1]
        result = lx.RESULTS.where(valid, lx.RESULTS.successful, lx.RESULTS.singular)
        return value, result, {}

    def transpose(self, state, options):
        """Reuse factors for the same real symmetric system in reverse solves."""
        return state, options

    def conj(self, state, options):
        """Retain the real-valued factorization under complex conjugation."""
        return state, options

    def assume_full_rank(self):
        """Use full-rank implicit derivatives for the required SPD systems."""
        return True
