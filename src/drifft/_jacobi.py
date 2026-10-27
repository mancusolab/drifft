"""Jacobi geometry, reduced selection operators, and distribution transforms."""

# pattern: Functional Core
import math
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax.scipy.special import gammaln, xlog1py, xlogy
from scipy.special import roots_jacobi

from ._epochs import _epoch_inputs, _identity, _PowerProblem, _prepare_epoch
from ._frequency import AbstractFrequencyMap, Neutral, Underdominance


def _choose_rank(copies: int, tolerance: float) -> int:
    """Bound the ideal one-step polynomial tail, excluding quadrature and roundoff."""
    # Counts/tolerance are host configuration; math computes a concrete rank
    # without storing population-sized planning arrays on the device.
    log_scale = 0.0
    weights = []
    for degree in range(copies - 1):
        if degree:
            ratio = (degree + 1) / copies
            log_scale += math.log1p(-ratio) - math.log1p(ratio)
        weights.append(
            log_scale + math.log((degree + 1) * (degree + 2) * (2 * degree + 3) / 6)
        )
    harmonic = sum(1 / index for index in range(1, copies))
    lift = math.sqrt(3 * copies**2 / (copies**2 - 1) * harmonic)
    target = 2 * math.log(tolerance / ((1 - 1 / copies) * lift))
    tail = -math.inf
    rank = copies - 1
    for index in reversed(range(1, copies - 1)):
        high, low = max(tail, weights[index]), min(tail, weights[index])
        tail = high + math.log1p(math.exp(low - high))
        if tail > target:
            break
        rank = index
    return rank


def _padded_rank(copies: int, rank: int, max_strength: float, tolerance: float) -> int:
    """Choose Jacobi padding using a Bernstein-ellipse assembly bound."""
    # math operates on fixed geometry and the configured strength bound, not
    # the dynamic selection strength differentiated by _assemble.
    ratio = max_strength / (2 - max_strength)
    radius, major, minor = 1.5, 13 / 12, 5 / 12
    image = major * math.sqrt(1 + ratio**2 * minor**2) / (1 - ratio * minor**2)
    growth = image + math.sqrt(image**2 - 1)
    multiplier = (1 + ratio**2 * major**2) / (1 - ratio * minor**2) ** 2
    degree = rank - 1
    endpoint = 0.5 * math.log((degree + 1) * (degree + 2) * (2 * degree + 3) / 6)
    target = math.log(tolerance) - math.log(2) - 0.5 * math.log(rank)
    required = math.ceil(
        (
            math.log(2 * multiplier / (radius - 1))
            + endpoint
            + degree * math.log(growth)
            - target
        )
        / math.log(radius)
    )
    return min(copies - 1, max(3, rank, math.ceil((required + rank) / 2)))


def _polynomials(points: jax.Array, rank: int) -> jax.Array:
    """Evaluate normalized Gegenbauer polynomials for weight 3(1-z²)/4."""
    first = jnp.ones_like(points)
    if rank == 1:
        return first[..., None]
    second = jnp.sqrt(5.0) * points

    def step(carry, degree):
        """Advance the orthonormal three-term recurrence."""
        previous, current = carry
        lower = jnp.sqrt(degree * (degree + 2) / ((2 * degree + 1) * (2 * degree + 3)))
        upper = jnp.sqrt(
            (degree + 1) * (degree + 3) / ((2 * degree + 3) * (2 * degree + 5))
        )
        following = (points * current - lower * previous) / upper
        return (current, following), following

    _, rest = jax.lax.scan(
        step, (first, second), jnp.arange(1, rank - 1, dtype=points.dtype)
    )
    return jnp.moveaxis(jnp.concatenate((first[None], second[None], rest)), 0, -1)


def _assemble(strength: jax.Array, copies: int, rank: int, padded_rank: int):
    """Assemble the reduced interior operator and endpoint-slope forcing in O(rR)."""
    degree = jnp.arange(padded_rank - 1, dtype=jnp.float64)
    beta = jnp.sqrt((degree + 1) * (degree + 3) / ((2 * degree + 3) * (2 * degree + 5)))
    off = beta * jnp.sqrt(1 - ((degree + 2) / copies) ** 2)

    def multiply_coordinate(vector):
        """Apply the symmetric tridiagonal Hahn coordinate matrix."""
        return jnp.pad(off * vector[:-1], (1, 0)) + jnp.pad(off * vector[1:], (0, 1))

    constant, quadratic = 1 - strength / 2, strength / 2
    diagonal = constant + quadratic * (
        jnp.pad(off**2, (1, 0)) + jnp.pad(off**2, (0, 1))
    )
    second_diagonal = quadratic * off[:-1] * off[1:]

    def solve(vector):
        """Solve (a I + b Z²) on its independent even and odd tridiagonals."""
        result = jnp.zeros_like(vector)
        for parity in (0, 1):
            diag = diagonal[parity::2]
            adjacent = second_diagonal[parity::2]
            rhs = vector[parity::2]
            if diag.size == 1:
                answer = rhs / diag
            elif diag.size == 2:
                determinant = diag[0] * diag[1] - adjacent[0] ** 2
                answer = (
                    jnp.stack(
                        (
                            diag[1] * rhs[0] - adjacent[0] * rhs[1],
                            diag[0] * rhs[1] - adjacent[0] * rhs[0],
                        )
                    )
                    / determinant
                )
            else:
                answer = jax.lax.linalg.tridiagonal_solve(
                    jnp.pad(adjacent, (1, 0)),
                    diag,
                    jnp.pad(adjacent, (0, 1)),
                    rhs[:, None],
                )[:, 0]
            result = result.at[parity::2].set(answer)
        return result

    def mapped_coordinate(vector):
        """Apply the rational selected coordinate function to the Jacobi matrix."""
        return solve(multiply_coordinate(vector))

    seed = jnp.zeros(padded_rank).at[0].set(1)
    indices = jnp.arange(2, padded_rank + 1, dtype=jnp.float64)
    scales = jnp.concatenate(
        (
            jnp.ones(1),
            jnp.exp(
                0.5
                * jnp.cumsum(jnp.log1p(-indices / copies) - jnp.log1p(indices / copies))
            ),
        )
    )
    solved = solve(solve(seed))
    first = constant**2 * solved - quadratic**2 * multiply_coordinate(
        multiply_coordinate(solved)
    )
    forcing = scales[:rank] * (quadratic * mapped_coordinate(seed))[:rank]
    if rank == 1:
        return ((1 - 1 / copies) * scales[:rank] * first[:rank])[:, None], forcing
    second = mapped_coordinate(first) / beta[0]

    def step(carry, index):
        """Build each polynomial column without materializing matrix polynomials."""
        previous, current = carry
        following = (mapped_coordinate(current) - beta[index - 1] * previous) / beta[
            index
        ]
        return (current, following), following[:rank]

    _, rest = jax.lax.scan(step, (first, second), jnp.arange(1, rank - 1))
    columns = jnp.concatenate((first[None, :rank], second[None, :rank], rest))
    matrix = (1 - 1 / copies) * scales[:rank, None] * columns.T
    parity = (jnp.arange(rank)[:, None] - jnp.arange(rank)[None, :]) % 2 == 0
    return jnp.where(parity, matrix, 0), forcing


@final
class _JacobiGeometry(eqx.Module):
    """Selection-independent quadrature and bounded binomial stencils."""

    copies: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    polynomials: jax.Array
    weights: jax.Array
    starts: jax.Array
    stencil: jax.Array
    coordinates: jax.Array
    envelope: jax.Array

    def __init__(self, copies: int, rank: int):
        """Prepare Gauss–Jacobi nodes on the host and all runtime geometry in JAX."""
        self.copies, self.rank = copies, rank
        # SciPy supplies a host-only Gaussian quadrature rule absent from JAX.
        node_count = rank + 32
        nodes, weights = roots_jacobi(node_count, 1, 1)
        retained = (node_count + 1) // 2
        nodes = jnp.asarray(nodes[:retained])
        self.weights = 0.75 * jnp.asarray(weights[:retained])
        # Decode both members of each reflected pair. A central node represents
        # itself twice, so halve its weight; odd polynomials vanish there.
        if node_count % 2:
            self.weights = self.weights.at[-1].multiply(0.5)
        self.polynomials = _polynomials(nodes, rank)
        probability = (nodes + 1) / 2
        trials = copies - 2
        # trials is a host count: math produces the fixed stencil width, while
        # the node-dependent stencil values below are computed with JAX.
        radius = min(trials, math.ceil(math.sqrt(trials * math.log(2e15) / 2)) + 2)
        mode = jnp.floor((trials + 1) * probability).astype(jnp.int32)
        self.starts = mode - radius
        raw = self.starts[:, None] + jnp.arange(2 * radius + 1, dtype=jnp.int32)
        count = jnp.clip(raw, 0, trials).astype(jnp.float64)
        log_weight = (
            gammaln(trials + 1)
            - gammaln(count + 1)
            - gammaln(trials - count + 1)
            + xlogy(count, probability[:, None])
            + xlog1py(trials - count, -probability[:, None])
        )
        stencil = jnp.where((raw >= 0) & (raw <= trials), jnp.exp(log_weight), 0)
        self.stencil = stencil / jnp.sum(stencil, axis=1, keepdims=True)
        self.coordinates = 2 * jnp.arange(1, copies, dtype=jnp.float64) / copies - 1
        self.envelope = (1 - self.coordinates) * (1 + self.coordinates)

    def encode(
        self, mass: jax.Array, strength: jax.Array, segregating: bool
    ) -> jax.Array:
        """Apply the transpose of selected polynomial evaluation using O(M+r) storage."""
        coordinates = (
            self.coordinates
            if segregating
            else jnp.concatenate(
                (jnp.array([-1.0]), self.coordinates, jnp.array([1.0]))
            )
        )
        mapped = coordinates / (1 - strength / 2 + strength / 2 * coordinates**2)
        weighted = (1 - 1 / self.copies) * (1 - mapped) * (1 + mapped) * mass
        first, second = jnp.ones_like(mapped), jnp.sqrt(5.0) * mapped

        def step(carry, degree):
            """Contract a polynomial with mass before proceeding to the next degree."""
            previous, current = carry
            lower = jnp.sqrt(
                degree * (degree + 2) / ((2 * degree + 1) * (2 * degree + 3))
            )
            upper = jnp.sqrt(
                (degree + 1) * (degree + 3) / ((2 * degree + 3) * (2 * degree + 5))
            )
            following = (mapped * current - lower * previous) / upper
            return (current, following), weighted @ following

        coefficients = jnp.array([weighted @ first])
        if self.rank > 1:
            _, rest = jax.lax.scan(
                step, (first, second), jnp.arange(1, self.rank - 1, dtype=jnp.float64)
            )
            coefficients = jnp.concatenate(
                (coefficients, jnp.array([weighted @ second]), rest)
            )
        return (
            coefficients
            if segregating
            else jnp.concatenate((jnp.array([mass.sum(), mass @ mapped]), coefficients))
        )

    def decode(self, coefficients: jax.Array, segregating: bool) -> jax.Array:
        """Transpose quadrature interpolation, reconstructing absorbing endpoints if needed."""
        interior_coefficients = coefficients if segregating else coefficients[2:]
        # p_n(-z) = (-1)^n p_n(z): reconstruct both quadrature values without
        # assuming the input distribution is symmetric. Split the matrix-vector
        # product by parity so each retained polynomial entry is used once.
        even = self.polynomials[:, ::2] @ interior_coefficients[::2]
        odd = self.polynomials[:, 1::2] @ interior_coefficients[1::2]
        samples = self.weights * (even + odd)
        reflected = self.weights * (even - odd)
        # Every stencil row is consecutive. Only its starting index is retained;
        # reconstruct int32 indices at application time with cheap arithmetic.
        indices = jnp.clip(
            self.starts[:, None] + jnp.arange(self.stencil.shape[1], dtype=jnp.int32),
            0,
            self.copies - 2,
        )
        interior = (
            jnp.zeros(self.copies - 1).at[indices].add(self.stencil * samples[:, None])
        )
        interior = (
            interior.at[self.copies - 2 - indices].add(
                self.stencil * reflected[:, None]
            )
            / self.envelope
        )
        if segregating:
            return interior
        left = (
            coefficients[0] - coefficients[1] - (1 - self.coordinates) @ interior
        ) / 2
        right = (
            coefficients[0] + coefficients[1] - (1 + self.coordinates) @ interior
        ) / 2
        return jnp.concatenate((left[None], interior, right[None]))


@final
class Jacobi(eqx.Module):
    r"""Reusable reduced Jacobi geometry for neutral and underdominant evolution.

    For gene-copy count $M$ and destination values $v_j$, the binomial expectation is

    $$B_M[v](q)=\sum_{j=0}^M\binom{M}{j}q^j(1-q)^{M-j}v_j.$$

    Write $z=2q-1$. After separating the linear interpolant between $v_0$ and
    $v_M$, approximate the remaining function in the basis
    $(1-z^2)\phi_k(z)$, $k=0,\ldots,r-1$. Here $r$ is `rank` and $\phi_k$
    is a normalized Jacobi polynomial $P_k^{(1,1)}$, orthonormal for weight
    $3(1-z^2)/4$ on $[-1,1]$. The envelope makes every interior basis function
    vanish at the absorbing endpoints. Gauss–Jacobi quadrature and binomial
    stencils map count-space values to these coefficients.

    Let $E$ denote this coefficient map and $D$ evaluation at the offspring
    probabilities $q_i=f(i/M)$. The expectation operator factors as
    $P\approx DE$, so forward mass evolves by its transpose. For $g\geq1$
    generations with a fixed frequency map and no immigration,

    $$p_g\approx E^T(H^T)^{g-1}D^Tp_0,\qquad H=ED.$$

    The reduced operator has $r$ interior coordinates, plus two endpoint
    coordinates for full-state evolution. It is assembled using polynomial
    recurrences and a padded tridiagonal coordinate matrix, without forming
    the population-sized transition matrix. A separate epoch method computes
    the power of $H^T$; representation rank and padding control different
    approximations from the method's power tolerance.

    Supports `Neutral` and `Underdominance`. Construct outside JIT with JAX x64
    enabled; the prepared geometry is independent of the selection strength.
    """

    gene_copies: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    padded_rank: int = eqx.field(static=True)
    max_strength: float = eqx.field(static=True)
    _geometry: _JacobiGeometry

    def __init__(
        self,
        gene_copies: int,
        *,
        rank: int | None = None,
        max_strength: float = 0.1,
        one_step_tolerance: float = 1e-13,
    ) -> None:
        r"""Prepare selection-independent geometry outside JIT.

        **Arguments:**

        - `gene_copies`: Python integer at least four (twice diploid Ne).
        - `rank`: Interior polynomial count, between one and `gene_copies - 1`.
          If omitted, use an ideal one-step polynomial tail bound.
        - `max_strength`: Static upper selection bound in `[0, 1)`, used for padding.
        - `one_step_tolerance`: Planning tolerance in `(0, 1)` for polynomial
          truncation and Jacobi padding separately; not a total error guarantee.

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
        # Validate host planning limits with math; the actual model strength is
        # a dynamic JAX value checked separately when assembling the operator.
        if not math.isfinite(max_strength) or not 0 <= max_strength < 1:
            raise ValueError("max_strength must lie in [0, 1)")
        if not math.isfinite(one_step_tolerance) or not 0 < one_step_tolerance < 1:
            raise ValueError("one_step_tolerance must lie in (0, 1)")
        if rank is None:
            rank = _choose_rank(gene_copies, one_step_tolerance)
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or not 1 <= rank < gene_copies
        ):
            raise ValueError("rank must be a Python integer in [1, gene_copies - 1]")
        if not jax.config.x64_enabled:
            raise ValueError("Jacobi requires jax_enable_x64=True")
        self.gene_copies, self.rank = gene_copies, rank
        self.padded_rank = _padded_rank(
            gene_copies, rank, max_strength, one_step_tolerance
        )
        self.max_strength = max_strength
        self._geometry = _JacobiGeometry(gene_copies, rank)

    def validate(self, frequency_map: AbstractFrequencyMap) -> None:
        """Reject frequency maps unsupported by reduced assembly."""
        if not isinstance(frequency_map, (Neutral, Underdominance)):
            raise TypeError("Jacobi supports only Neutral and Underdominance")

    def _validate_mass(
        self, distribution: jax.Array, frequency_map: AbstractFrequencyMap
    ) -> tuple[jax.Array, jax.Array]:
        """Attach traced selection validation even to identity and boundary-only histories."""
        self.validate(frequency_map)
        if isinstance(frequency_map, Neutral):
            strength = jnp.asarray(0.0)
        else:
            assert isinstance(frequency_map, Underdominance)
            strength = frequency_map.strength
        distribution = eqx.error_if(
            distribution,
            ~jnp.isfinite(strength) | (strength < 0) | (strength > self.max_strength),
            "selection strength must lie in [0, max_strength]",
        )
        return distribution, strength


@_prepare_epoch.register(Jacobi)
def _prepare_jacobi(
    backend, distribution, frequency_map, generations, segregating, immigration
):
    """Encode the reduced action and affine forcing without choosing a power algorithm."""
    distribution, source = _epoch_inputs(
        distribution, immigration, backend.gene_copies, generations, segregating
    )
    distribution, strength = backend._validate_mass(distribution, frequency_map)
    if generations == 0:
        operator = lx.IdentityLinearOperator(
            jax.ShapeDtypeStruct(distribution.shape, distribution.dtype)
        )
        return _PowerProblem(
            operator,
            distribution,
            0,
            _identity,
            jnp.zeros_like(distribution),
            False,
            jnp.asarray(1.0),
        )
    seed = backend._geometry.encode(distribution, strength, segregating)
    decode = eqx.Partial(backend._geometry.decode, segregating=segregating)
    if generations == 1:
        operator = lx.IdentityLinearOperator(
            jax.ShapeDtypeStruct(seed.shape, seed.dtype)
        )
        return _PowerProblem(operator, seed, 0, decode, source, False, jnp.asarray(1.0))
    matrix, forcing = _assemble(
        strength, backend.gene_copies, backend.rank, backend.padded_rank
    )
    if not segregating:
        extended = jnp.zeros((backend.rank + 2, backend.rank + 2))
        extended = extended.at[:2, :2].set(jnp.eye(2))
        extended = extended.at[2:, 1].set(forcing)
        matrix = extended.at[2:, 2:].set(matrix)
    matrix = matrix.T
    augmented = immigration is not None
    if augmented:
        encoded_source = backend._geometry.encode(source, strength, segregating)
        extended = jnp.zeros((seed.size + 1, seed.size + 1))
        extended = extended.at[:-1, :-1].set(matrix)
        extended = extended.at[:-1, -1].set(encoded_source)
        matrix = extended.at[-1, -1].set(1)
        seed = jnp.concatenate((seed, jnp.ones(1)))
    defect_scale = (
        1 + (generations - 1) * jnp.sum(jnp.abs(source))
        if augmented
        else jnp.asarray(1.0)
    )
    return _PowerProblem(
        lx.MatrixLinearOperator(matrix),
        seed,
        generations - 1,
        decode,
        source,
        augmented,
        defect_scale,
    )
