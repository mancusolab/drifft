"""Rational Krylov epoch actions on reduced Jacobi operators."""

# pattern: Functional Core
import math
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from ._epochs import AbstractEpochBackend, EpochResult, _epoch_inputs, _epoch_result
from ._frequency import AbstractFrequencyMap, Neutral, Underdominance
from ._jacobi import _assemble, _choose_rank, _JacobiGeometry, _padded_rank


def _power_action(matrix: jax.Array, seed: jax.Array, steps: int, dimension: int):
    """Project the original matrix onto a twice-reorthogonalized resolvent basis."""
    size = seed.size
    dimension = min(size, dimension)
    shift = steps / 8
    identity = jnp.eye(size)
    factor = jsp_linalg.lu_factor(identity + shift * (identity - matrix))
    # Safe normalization also keeps zero and happy-breakdown branches finite in AD.
    norm_squared = seed @ seed
    norm = jnp.sqrt(jnp.where(norm_squared > 0, norm_squared, 1.0))
    basis = jnp.zeros((size, dimension)).at[:, 0].set(seed / norm)

    def step(index, basis):
        """Add a resolvent Arnoldi vector, leaving unused columns zero at breakdown."""
        candidate = jsp_linalg.lu_solve(factor, basis[:, index])
        original_squared = candidate @ candidate
        for _ in range(2):
            candidate = candidate - basis @ (basis.T @ candidate)
        squared = candidate @ candidate
        active = squared > (64 * jnp.finfo(jnp.float64).eps) ** 2 * original_squared
        denominator = jnp.sqrt(jnp.where(active, squared, 1.0))
        return basis.at[:, index + 1].set(
            jnp.where(active, candidate / denominator, 0.0)
        )

    basis = jax.lax.fori_loop(0, dimension - 1, step, basis)
    # Project A itself: inverting the projected resolvent gives a different,
    # potentially unstable power approximation for this nonnormal operator.
    projected = basis.T @ matrix @ basis
    coordinates = basis.T @ seed
    result = basis @ (jnp.linalg.matrix_power(projected, steps) @ coordinates)
    defect = matrix @ basis - basis @ projected
    if dimension == size:
        # A complete basis (or invariant-subspace breakdown) needs no truncation
        # estimate. Finiteness is still checked by the public result contract.
        return result, result, defect, projected, coordinates
    coarse_dimension = max(1, dimension // 2)
    coarse = basis[:, :coarse_dimension] @ (
        jnp.linalg.matrix_power(projected[:coarse_dimension, :coarse_dimension], steps)
        @ coordinates[:coarse_dimension]
    )
    return result, coarse, defect, projected, coordinates


@final
class RationalKrylov(AbstractEpochBackend):
    r"""Accelerate constant-size epochs using rational Krylov projection.

    The Jacobi reduction factors a one-generation expectation as $P\approx DE$.
    Writing $H=ED$, the forward action is
    $$(P^T)^g p\approx E^T(H^T)^{g-1}D^T p.$$
    A resolvent Krylov basis $V$ approximates the reduced power by
    $V(V^T H^T V)^{g-1}V^TD^Tp$. Immigration is included through an augmented
    affine system. Neutral and symmetric underdominance maps are supported.

    !!! warning "Experimental approximation"

        The reported error combines nested-space differences with accumulated
        projected recurrence defects. It is an estimate,
        not a rigorous error bound, and excludes Jacobi truncation, quadrature,
        assembly, and floating-point error. Increase rank and Krylov dimension
        and compare against Sequential when establishing accuracy for a new case.

    !!! note "Differentiation"

        Native JAX AD differentiates the evaluated algorithm, including its
        input-dependent basis. This is not a fixed linear operator or a matched
        transpose. Derivatives at zero seeds or changes in Arnoldi breakdown
        are not guaranteed. Reverse-mode encoding may use O(M r) memory.
    """

    gene_copies: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    padded_rank: int = eqx.field(static=True)
    krylov_dimension: int = eqx.field(static=True)
    max_strength: float = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    _geometry: _JacobiGeometry

    def __init__(
        self,
        gene_copies: int,
        *,
        rank: int | None = None,
        krylov_dimension: int = 64,
        max_strength: float = 0.1,
        one_step_tolerance: float = 1e-13,
        rtol: float = 1e-8,
        atol: float = 1e-12,
    ) -> None:
        r"""Prepare selection-independent geometry outside JIT.

        **Arguments:**

        - `gene_copies`: Python integer at least four (twice diploid Ne).
        - `rank`: Interior polynomial count, between one and `gene_copies - 1`.
          If omitted, use an ideal one-step polynomial tail bound.
        - `krylov_dimension`: Maximum basis size, at least two. Capped at the
          reduced system dimension. Increase this if the power estimate fails.
        - `max_strength`: Static upper selection bound in `[0, 1)`, used for padding.
        - `one_step_tolerance`: Planning tolerance in `(0, 1)` for polynomial
          truncation and Jacobi padding separately; not a total error guarantee.
        - `rtol`, `atol`: Nonnegative tolerances for the power-action L1 estimate.

        **Returns:**

        `None`. Runtime geometry consists of JAX arrays; SciPy is used only to
        prepare Gauss–Jacobi nodes, for which JAX has no equivalent routine.

        **Raises:**

        `ValueError` for invalid counts or tolerances. Enable JAX x64 before
        constructing this backend. Dense reduced storage costs O(rank²), and
        LU factorization costs O(rank³) per epoch evaluation.
        """
        if (
            isinstance(gene_copies, bool)
            or not isinstance(gene_copies, int)
            or gene_copies < 4
        ):
            raise ValueError("gene_copies must be a Python integer >= 4")
        if not math.isfinite(max_strength) or not 0 <= max_strength < 1:
            raise ValueError("max_strength must lie in [0, 1)")
        if not math.isfinite(one_step_tolerance) or not 0 < one_step_tolerance < 1:
            raise ValueError("one_step_tolerance must lie in (0, 1)")
        if not all(math.isfinite(value) and value >= 0 for value in (rtol, atol)):
            raise ValueError("rtol and atol must be finite and nonnegative")
        if rank is None:
            rank = _choose_rank(gene_copies, one_step_tolerance)
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or not 1 <= rank < gene_copies
        ):
            raise ValueError("rank must be a Python integer in [1, gene_copies - 1]")
        if (
            isinstance(krylov_dimension, bool)
            or not isinstance(krylov_dimension, int)
            or krylov_dimension < 2
        ):
            raise ValueError("krylov_dimension must be a Python integer >= 2")
        if not jax.config.x64_enabled:
            raise ValueError("RationalKrylov requires jax_enable_x64=True")
        self.gene_copies, self.rank = gene_copies, rank
        self.padded_rank = _padded_rank(
            gene_copies, rank, max_strength, one_step_tolerance
        )
        self.krylov_dimension = krylov_dimension
        self.max_strength, self.rtol, self.atol = max_strength, rtol, atol
        self._geometry = _JacobiGeometry(gene_copies, rank)

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
        r"""Advance with the contracts of `AbstractEpochBackend.advance`.

        Only `Neutral` and `Underdominance` are accepted. Selection must lie in
        `[0, max_strength]`; this domain is checked even when `throw=False`.
        Immigration represents new mass, not recurrent mutation of existing
        alleles. Signed mass is supported without clipping or normalization.
        """
        distribution, source = _epoch_inputs(
            distribution, immigration, self.gene_copies, generations, segregating
        )
        if isinstance(frequency_map, Neutral):
            strength = jnp.asarray(0.0)
        elif isinstance(frequency_map, Underdominance):
            strength = frequency_map.strength
        else:
            raise TypeError("RationalKrylov supports only Neutral and Underdominance")
        # Tie validation to the output dependency, including the identity action.
        distribution = eqx.error_if(
            distribution,
            ~jnp.isfinite(strength) | (strength < 0) | (strength > self.max_strength),
            "selection strength must lie in [0, max_strength]",
        )
        if generations == 0:
            return _epoch_result(distribution, jnp.asarray(0.0), self.atol, throw)
        seed = self._geometry.encode(distribution, strength, segregating)
        if generations == 1:
            result = self._geometry.decode(seed, segregating) + source
            return _epoch_result(result, jnp.asarray(0.0), self.atol, throw)
        matrix, forcing = _assemble(
            strength, self.gene_copies, self.rank, self.padded_rank
        )
        if not segregating:
            extended = jnp.zeros((self.rank + 2, self.rank + 2))
            extended = extended.at[:2, :2].set(jnp.eye(2))
            extended = extended.at[2:, 1].set(forcing)
            matrix = extended.at[2:, 2:].set(matrix)
        matrix = matrix.T
        if immigration is not None:
            # y <- H.T y + D.T b. The final b is added after decoding, so the
            # source passes through exactly g-1, ..., 0 transitions.
            encoded_source = self._geometry.encode(source, strength, segregating)
            augmented = jnp.zeros((seed.size + 1, seed.size + 1))
            augmented = augmented.at[:-1, :-1].set(matrix)
            augmented = augmented.at[:-1, -1].set(encoded_source)
            matrix = augmented.at[-1, -1].set(1)
            seed = jnp.concatenate((seed, jnp.ones(1)))
        fine, coarse, defect, projected, coordinates = _power_action(
            matrix, seed, generations - 1, self.krylov_dimension
        )
        # Diagnostics are nondifferentiable metadata. Decode each defect column
        # sequentially to avoid an M-by-r-by-Krylov-dimension temporary.
        defect, projected, coordinates = jax.tree.map(
            jax.lax.stop_gradient, (defect, projected, coordinates)
        )

        def column_norm(column):
            """Measure a recurrence defect in the output mass norm."""
            auxiliary = jnp.abs(column[-1]) if immigration is not None else 0.0
            physical = column[:-1] if immigration is not None else column
            return (
                jnp.sum(jnp.abs(self._geometry.decode(physical, segregating)))
                + auxiliary
            )

        weights = jax.lax.map(column_norm, defect.T)

        def accumulate(_, carry):
            """Sum decoded local defects along the small projected trajectory."""
            state, total = carry
            return projected @ state, total + weights @ jnp.abs(state)

        _, defect_estimate = jax.lax.fori_loop(
            0, generations - 1, accumulate, (coordinates, jnp.asarray(0.0))
        )
        if immigration is not None:
            # Affine augmentation can amplify an auxiliary-coordinate defect
            # through immigration at every subsequent generation.
            defect_estimate *= 1 + (generations - 1) * jnp.sum(jnp.abs(source))
        if immigration is not None:
            fine, coarse = fine[:-1], coarse[:-1]
        result = self._geometry.decode(fine, segregating) + source
        difference = self._geometry.decode(fine - coarse, segregating)
        error = jnp.maximum(jnp.sum(jnp.abs(difference)), defect_estimate)
        tolerance = self.atol + self.rtol * jnp.sum(jnp.abs(result))
        return _epoch_result(result, error, tolerance, throw)
