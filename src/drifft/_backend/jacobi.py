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

from .._banded import PentadiagonalCholesky, SymmetricPentadiagonalOperator
from .._frequency import (
    AbstractFrequencyMap,
    AdditiveSelection,
    Composition,
    DiploidSelection,
    DominanceSelection,
    DominantSelection,
    MultiplicativeSelection,
    Mutation,
    Neutral,
    Overdominance,
    RecessiveSelection,
    Underdominance,
)
from .._polynomial import gegenbauer_beta, gegenbauer_moments, gegenbauer_values
from ._base import AbstractTransitionBackend


def _selection_parameters(frequency_map):
    """Resolve supported diploid constraints without constructing new map instances."""
    if isinstance(frequency_map, Neutral):
        return jnp.asarray(0.0), jnp.asarray(0.0)
    if isinstance(frequency_map, DiploidSelection):
        return frequency_map.s_het, frequency_map.s_hom
    if isinstance(frequency_map, Underdominance):
        return frequency_map.strength, jnp.asarray(0.0)
    if isinstance(frequency_map, Overdominance):
        return -frequency_map.strength, jnp.asarray(0.0)
    if isinstance(frequency_map, AdditiveSelection):
        return frequency_map.s_het, 2 * frequency_map.s_het
    if isinstance(frequency_map, DominanceSelection):
        return frequency_map.h * frequency_map.s, frequency_map.s
    if isinstance(frequency_map, RecessiveSelection):
        return jnp.asarray(0.0), frequency_map.s
    if isinstance(frequency_map, DominantSelection):
        return frequency_map.s, frequency_map.s
    if isinstance(frequency_map, MultiplicativeSelection):
        return 1 - frequency_map.r, 1 - frequency_map.r**2
    raise TypeError("Jacobi requires a supported diploid selection rule")


def _map_parameters(frequency_map):
    """Accept one mutation and one diploid rule, preserving biological order."""
    if isinstance(frequency_map, Mutation):
        return jnp.asarray(0.0), jnp.asarray(0.0), frequency_map, True
    if isinstance(frequency_map, Composition):
        if len(frequency_map.maps) == 2:
            first, second = frequency_map.maps
            if isinstance(first, Mutation):
                het, hom = _selection_parameters(second)
                return het, hom, first, True
            if isinstance(second, Mutation):
                het, hom = _selection_parameters(first)
                return het, hom, second, False
        raise TypeError(
            "Jacobi compositions require one Mutation and one diploid selection rule"
        )
    het, hom = _selection_parameters(frequency_map)
    return het, hom, None, True


def _rational_coefficients(frequency_map):
    """Return ascending quadratic coefficients for g(z) = 2 q((z+1)/2)-1."""
    het, hom, mutation, mutation_first = _map_parameters(frequency_map)
    # Substitute x=(z+1)/2 into the diploid rule and collect powers of z.
    # Keeping numerator/denominator coefficients lets assembly evaluate g(Z)
    # through banded matrix actions instead of sampling every population state.
    numerator = jnp.array([-hom / 4, 1 - hom / 2, -hom / 4])
    denominator = jnp.array([1 - het / 2 - hom / 4, -hom / 2, het / 2 - hom / 4])
    if mutation is not None:
        offset = mutation.forward_rate - mutation.backward_rate
        slope = 1 - mutation.forward_rate - mutation.backward_rate
        if mutation_first:

            def substitute(coefficients):
                """Compose a quadratic with the affine mutation coordinate."""
                a, b, c = coefficients
                return jnp.array(
                    [
                        a + b * offset + c * offset**2,
                        slope * (b + 2 * c * offset),
                        c * slope**2,
                    ]
                )

            numerator, denominator = substitute(numerator), substitute(denominator)
        else:
            numerator = offset * denominator + slope * numerator
    return numerator, denominator


def _diploid_padding(copies: int, rank: int, strength: float, tolerance: float) -> int:
    """Plan a conservative ellipse bound for endpoint-subtracted rational columns.

    Fitness reductions are bounded in absolute value by strength. Mutation's
    affine coordinate maps the ellipse into itself. Shrink the ellipse until
    the denominator differs from its positive real-interval value by less than
    half the fitness lower bound. Full Hahn dimension is the exact fallback.
    """
    radius = 1.5
    while True:
        if radius - 1 < 1e-8:
            # The required padding already exceeds any practical dimension;
            # avoid a degenerate floating-point ellipse or log(0).
            return copies - 1
        major, minor = (radius + 1 / radius) / 2, (radius - 1 / radius) / 2
        variation = strength * (0.5 + 1.5 * major) * (minor + major - 1)
        if variation <= (1 - strength) / 2:
            break
        radius = 1 + (radius - 1) / 2
    lower = 1 - strength - variation
    numerator_bound = (
        strength / 4 + (1 + strength / 2) * major + strength * major**2 / 4
    )
    image = max(1.0, numerator_bound / lower)
    growth = image + math.sqrt(image**2 + 1)
    degree = rank - 1
    endpoint = 0.5 * math.log((degree + 1) * (degree + 2) * (2 * degree + 3) / 6)
    # On the ellipse boundary |1-z²| >= minor². Endpoint subtraction
    # contributes at most (1+major) times the real-interval endpoint bound.
    log_bound = (
        endpoint
        + degree * math.log(growth)
        + math.log(2 + image**2 + major)
        - 2 * math.log(minor)
    )
    required = math.ceil(
        (
            math.log(4)
            + log_bound
            - math.log(radius - 1)
            + 0.5 * math.log(rank + 1)
            - math.log(tolerance)
        )
        / math.log(radius)
    )
    return min(copies - 1, max(3, rank, math.ceil((required + rank) / 2)))


def _assemble_diploid(frequency_map, copies: int, rank: int, padded_rank: int):
    """Assemble all r+2 coordinates, including nonabsorbing endpoint coupling."""
    numerator, denominator = _rational_coefficients(frequency_map)
    degree = jnp.arange(padded_rank - 1, dtype=jnp.float64)
    beta = gegenbauer_beta(degree)
    off = beta * jnp.sqrt(1 - ((degree + 2) / copies) ** 2)

    def coordinate(vector):
        """Multiply by the finite-population tridiagonal coordinate matrix Z."""
        return jnp.pad(off * vector[:-1], (1, 0)) + jnp.pad(off * vector[1:], (0, 1))

    d0, d1, d2 = denominator
    # Z is tridiagonal, so d0*I + d1*Z + d2*Z² has only five bands.
    # Factor the denominator once and reuse it for every polynomial column.
    operator = SymmetricPentadiagonalOperator(
        d0 + d2 * (jnp.pad(off**2, (1, 0)) + jnp.pad(off**2, (0, 1))),
        d1 * off,
        d2 * off[:-1] * off[1:],
    )

    solver = PentadiagonalCholesky()
    factor = solver.init(operator, {})

    def solve(rhs):
        """Reuse the banded factorization with Lineax's implicit derivative rules."""
        return lx.linear_solve(operator, rhs, solver, state=factor).value

    def mapped(vector):
        """Apply the rational map g(Z) via one numerator action and banded solve."""
        return solve(
            numerator[0] * vector
            + numerator[1] * coordinate(vector)
            + numerator[2] * coordinate(coordinate(vector))
        )

    seed = jnp.zeros(padded_rank).at[0].set(1)
    endpoint_q = frequency_map.apply(jnp.array([0.0, 1.0]))
    endpoint_g = 2 * endpoint_q - 1
    intercept, slope = jnp.mean(endpoint_g), (endpoint_g[1] - endpoint_g[0]) / 2
    # g = intercept + slope*z + (1-z²)*residual. Polynomial division
    # removes the apparent endpoint poles analytically, before evaluation.
    residual = solve(
        (numerator[0] - intercept * d0) * seed + slope * d2 * coordinate(seed)
    )

    def multiply(state):
        """Multiply endpoint/interior coordinates by g without endpoint division."""
        endpoints, interior = state
        a, b = jnp.mean(endpoints), (endpoints[1] - endpoints[0]) / 2
        product = (
            mapped(interior)
            + a * residual
            + b * coordinate(residual)
            - slope * b * seed
        )
        return endpoints * endpoint_g, product

    indices = jnp.arange(2, padded_rank + 1, dtype=jnp.float64)
    scales = jnp.concatenate(
        (
            jnp.ones(1),
            jnp.exp(
                0.5
                * jnp.cumsum(jnp.log1p(-indices / copies) - jnp.log1p(indices / copies))
            ),
        )
    )[:rank]

    def coefficients(state):
        """Convert endpoint values to intercept/slope and rescale interior modes."""
        endpoints, interior = state
        return jnp.concatenate(
            (
                jnp.array([jnp.mean(endpoints), (endpoints[1] - endpoints[0]) / 2]),
                scales * interior[:rank],
            )
        )

    # Start with h(g), using stable endpoint values 4q(1-q).
    _, squared_residual = multiply((endpoint_g, residual))
    first = (4 * endpoint_q * (1 - endpoint_q), -squared_residual)
    columns = coefficients(first)[None, :]
    if rank > 1:
        second = jax.tree.map(lambda value: value / beta[0], multiply(first))

        def step(carry, index):
            previous, current = carry
            product = multiply(current)
            following = jax.tree.map(
                lambda p, old: (p - beta[index - 1] * old) / beta[index],
                product,
                previous,
            )
            return (current, following), coefficients(following)

        _, rest = jax.lax.scan(step, (first, second), jnp.arange(1, rank - 1))
        columns = jnp.concatenate((columns, coefficients(second)[None, :], rest))
    matrix = jnp.zeros((rank + 2, rank + 2)).at[0, 0].set(1)
    matrix = matrix.at[:, 1].set(coefficients((endpoint_g, residual)))
    return matrix.at[:, 2:].set((1 - 1 / copies) * columns.T)


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


def _assemble(strength: jax.Array, copies: int, rank: int, padded_rank: int):
    """Assemble the reduced interior operator and endpoint-slope forcing in O(rR)."""
    degree = jnp.arange(padded_rank - 1, dtype=jnp.float64)
    beta = gegenbauer_beta(degree)
    off = beta * jnp.sqrt(1 - ((degree + 2) / copies) ** 2)

    def multiply_coordinate(vector):
        """Apply the symmetric tridiagonal Hahn coordinate matrix."""
        return jnp.pad(off * vector[:-1], (1, 0)) + jnp.pad(off * vector[1:], (0, 1))

    constant, quadratic = 1 - strength / 2, strength / 2
    # Symmetric selection gives g(z)=z/(a+b*z²). Z² connects only degrees
    # of the same parity, splitting the denominator into two tridiagonals.
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
    # Squared discrete/continuous normalization factors have ratios (M-k)/(M+k).
    # Accumulate their logarithms before taking the square root to avoid
    # underflow in a long product at large rank.
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
    # h(g(z))/h(z) = (a²-b²*z²)/(a+b*z²)² for h(z)=1-z² and a+b=1.
    # This supplies the first envelope-weighted column with no endpoint poles.
    first = constant**2 * solved - quadratic**2 * multiply_coordinate(
        multiply_coordinate(solved)
    )
    forcing = scales[:rank] * (quadratic * mapped_coordinate(seed))[:rank]
    # The forcing represents g(z)-z = b*z*h(z)/(a+b*z²), needed when the
    # endpoint interpolant has nonzero slope in full-state evolution.
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
    """Process-independent quadrature and bounded binomial stencils."""

    copies: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    polynomials: jax.Array
    weights: jax.Array
    starts: jax.Array
    stencil: jax.Array
    coordinates: jax.Array
    envelope: jax.Array
    endpoint_correction: jax.Array

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
        self.polynomials = gegenbauer_values(nodes, rank)
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
        # Killing endpoints uses E_int on values whose endpoint interpolant
        # is nonzero under mutation. Cache E_int[1] and E_int[z], not M-by-r E.
        transpose = jax.linear_transpose(
            lambda coefficients: self.decode(coefficients, True), jnp.zeros(rank)
        )
        self.endpoint_correction = jnp.stack(
            (transpose(jnp.ones(copies - 1))[0], transpose(self.coordinates)[0]), axis=1
        )

    def encode(
        self,
        mass: jax.Array,
        frequency_map: AbstractFrequencyMap | jax.Array,
        segregating: bool,
    ) -> jax.Array:
        """Apply the transpose of selected polynomial evaluation using O(M+r) storage."""
        coordinates = (
            self.coordinates
            if segregating
            else jnp.concatenate(
                (jnp.array([-1.0]), self.coordinates, jnp.array([1.0]))
            )
        )
        if isinstance(frequency_map, AbstractFrequencyMap):
            probabilities = frequency_map.apply((coordinates + 1) / 2)
            mapped = 2 * probabilities - 1
            weighted = (
                (1 - 1 / self.copies) * 4 * probabilities * (1 - probabilities) * mass
            )
        else:
            # Preserve the strength-only private transform used by benchmark
            # research scripts, without constructing a map inside the kernel.
            mapped = coordinates / (
                1 - frequency_map / 2 + frequency_map / 2 * coordinates**2
            )
            weighted = (1 - 1 / self.copies) * (1 - mapped) * (1 + mapped) * mass
        coefficients = gegenbauer_moments(mapped, weighted, rank=self.rank)
        return (
            coefficients
            if segregating
            else jnp.concatenate((jnp.array([mass.sum(), mass @ mapped]), coefficients))
        )

    def decode(self, coefficients: jax.Array, segregating: bool) -> jax.Array:
        """Transpose quadrature interpolation, reconstructing endpoint mass for full states."""
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
        # Full coordinates retain total mass and its first z moment. Recover
        # the two endpoint masses from those constraints after decoding the
        # interior; this also works for signed inputs without normalization.
        left = (
            coefficients[0] - coefficients[1] - (1 - self.coordinates) @ interior
        ) / 2
        right = (
            coefficients[0] + coefficients[1] - (1 + self.coordinates) @ interior
        ) / 2
        return jnp.concatenate((left[None], interior, right[None]))


@final
class Jacobi(AbstractTransitionBackend):
    r"""Jacobi geometry for binomial expectations and reduced Krylov evolution.

    For gene-copy count $M$ and destination values $v_j$, the binomial expectation is

    $$B_M[v](q)=\sum_{j=0}^M\binom{M}{j}q^j(1-q)^{M-j}v_j.$$

    Write $z=2q-1$. After separating the linear interpolant between $v_0$ and
    $v_M$, approximate the remaining function in the basis
    $(1-z^2)\phi_k(z)$, $k=0,\ldots,r-1$. Here $r$ is `rank` and $\phi_k$
    is a normalized Jacobi polynomial $P_k^{(1,1)}$, orthonormal for weight
    $3(1-z^2)/4$ on $[-1,1]$. The envelope makes every interior basis function
    vanish at the count endpoints. Gauss–Jacobi quadrature and binomial
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

    `Transition` and `Sequential` accept arbitrary valid offspring probabilities
    through the common expectation interface. Reduced Krylov assembly supports
    neutral and diploid selection, standalone mutation, and one mutation
    composed with one supported selection rule in either order. Custom maps and
    longer compositions are not supported by reduced assembly.
    Recurrent mutation couples the endpoint states to the interior; full-state
    evolution retains this coupling, while segregating evolution discards
    endpoint mass each generation. Construct outside JIT with JAX x64 enabled;
    the prepared geometry is independent of the dynamic process parameters.
    Reduced epoch domains are checked at runtime, including for zero generations:
    selection must stay within the configured bound, and mutation rates must be
    finite and in [0, 1]. In `DemographicTransition`, size-changing steps use the
    same backend by default; an explicit FFT or NUFFT `boundary_backend` can be
    supplied instead.
    """

    gene_copies: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    padded_rank: int = eqx.field(static=True)
    max_strength: float = eqx.field(static=True)
    _diploid_padded_rank: int = eqx.field(static=True)
    _geometry: _JacobiGeometry

    def __init__(
        self,
        gene_copies: int,
        *,
        rank: int | None = None,
        max_strength: float = 0.1,
        one_step_tolerance: float = 1e-13,
    ) -> None:
        r"""Prepare process-independent geometry outside JIT.

        **Arguments:**

        - `gene_copies`: Python integer at least four (twice diploid Ne).
        - `rank`: Interior polynomial count, between one and `gene_copies - 1`.
          If omitted, use an ideal one-step polynomial tail bound.
        - `max_strength`: Static bound in `[0, 1)` on the absolute values of
          both diploid fitness reductions, used for reduced Krylov padding.
          In reduced assembly, `Underdominance` and `Overdominance` retain their
          finite, nonnegative strength domains, including in mutation compositions.
        - `one_step_tolerance`: Planning tolerance in `(0, 1)` for polynomial
          truncation and Jacobi padding separately; not a total error guarantee.

        **Returns:**

        `None`. Runtime geometry consists of JAX arrays; SciPy is used only to
        prepare Gauss–Jacobi nodes, for which JAX has no equivalent routine.

        **Raises:**

        `ValueError` for invalid counts or tolerances. Enable JAX x64 before
        constructing this backend. Dense reduced storage costs O(rank²).
        With RationalKrylov, dense LU factorization costs O(rank³) per power action.
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
        self.max_strength = float(max_strength)
        self._diploid_padded_rank = _diploid_padding(
            gene_copies, rank, max_strength, one_step_tolerance
        )
        self._geometry = _JacobiGeometry(gene_copies, rank)

    def expect(self, probabilities: jax.Array, values: jax.Array) -> jax.Array:
        r"""Evaluate the binomial expectation at arbitrary offspring probabilities.

        **Arguments:**

        - `probabilities`: Float64 probability vector, including endpoints.
        - `values`: Finite real destination values of length `gene_copies + 1`.

        **Returns:**

        One expectation per probability. The operation is linear in `values`
        and supports transposition, batching, and polynomial derivatives.
        Frequency-map and `max_strength` restrictions apply only to reduced
        epoch assembly, not this arbitrary-probability operation.

        **Raises:**

        `ValueError` for incompatible input shapes. Probability ranges and
        finiteness are caller preconditions, as for other transition backends.
        """
        probabilities = jnp.asarray(probabilities, dtype=jnp.float64)
        values = jnp.asarray(values, dtype=jnp.float64)
        if probabilities.ndim != 1 or values.shape != (self.gene_copies + 1,):
            raise ValueError(
                "expect requires a probability vector and gene_copies + 1 destination values"
            )

        # Transpose the existing decoder to obtain E v. The linear endpoint
        # interpolant is evaluated directly, preserving exact endpoint values.
        def interior_decode(coefficients):
            return self._geometry.decode(
                jnp.concatenate((jnp.zeros(2), coefficients)), False
            )

        spec = jax.ShapeDtypeStruct((self.rank,), jnp.float64)
        coefficients = jax.linear_transpose(interior_decode, spec)(values)[0]
        polynomial = gegenbauer_moments(
            2 * probabilities - 1, coefficients, rank=self.rank, transposed=True
        )
        return (
            (1 - probabilities) * values[0]
            + probabilities * values[-1]
            + (1 - 1 / self.gene_copies)
            * 4
            * probabilities
            * (1 - probabilities)
            * polynomial
        )

    def validate(self, frequency_map: AbstractFrequencyMap) -> None:
        r"""Check whether a frequency map supports reduced Krylov assembly.

        **Arguments:**

        - `frequency_map`: Neutral, supported diploid selection, mutation, or
          one mutation composed with one supported selection rule in either order.

        **Returns:**

        `None`. Checks the map structure only. Reduced evolution checks dynamic
        parameter domains separately, even for zero generations. One-generation
        `expect` and Sequential do not impose these assembly restrictions.

        **Raises:**

        `TypeError` for unsupported maps or compositions.
        """
        _map_parameters(frequency_map)

    def _reduced_expectation(
        self, frequency_map: AbstractFrequencyMap, *, segregating: bool
    ) -> jax.Array:
        """Assemble ED, including endpoint coupling for full or interior states.

        Preparation validates the map and dynamic parameter bounds first. Keep
        selection-family dispatch, padding, and endpoint corrections here so
        evolution code need not know the representation's assembly details.
        """
        if isinstance(frequency_map, (Neutral, Underdominance)):
            strength = (
                jnp.asarray(0.0)
                if isinstance(frequency_map, Neutral)
                else frequency_map.strength
            )
            matrix, forcing = _assemble(
                strength, self.gene_copies, self.rank, self.padded_rank
            )
            if not segregating:
                extended = jnp.zeros((self.rank + 2, self.rank + 2))
                extended = extended.at[:2, :2].set(jnp.eye(2))
                extended = extended.at[2:, 1].set(forcing)
                matrix = extended.at[2:, 2:].set(matrix)
        else:
            matrix = _assemble_diploid(
                frequency_map,
                self.gene_copies,
                self.rank,
                self._diploid_padded_rank,
            )
            if segregating:
                # Recurrent mutation can produce nonzero endpoint coefficients.
                # Fold their interior contribution into E_int D_int before discarding
                # endpoints; simply slicing the full operator would lose that term.
                matrix = (
                    matrix[2:, 2:] + self._geometry.endpoint_correction @ matrix[:2, 2:]
                )
        return matrix

    def _validate_mass(
        self, distribution: jax.Array, frequency_map: AbstractFrequencyMap
    ) -> tuple[jax.Array, jax.Array]:
        """Check process parameters even for identity and boundary-only histories."""
        self.validate(frequency_map)
        if not isinstance(frequency_map, (Neutral, Underdominance)):
            components = (
                frequency_map.maps
                if isinstance(frequency_map, Composition)
                else (frequency_map,)
            )
            for component in components:
                if isinstance(component, (Underdominance, Overdominance)):
                    distribution = eqx.error_if(
                        distribution,
                        ~jnp.isfinite(component.strength) | (component.strength < 0),
                        "selection strength must lie in [0, max_strength]",
                    )
            het, hom, mutation, _ = _map_parameters(frequency_map)
            distribution = eqx.error_if(
                distribution,
                ~jnp.isfinite(het)
                | ~jnp.isfinite(hom)
                | (jnp.abs(het) > self.max_strength)
                | (jnp.abs(hom) > self.max_strength),
                "diploid selection coefficients must be finite with absolute values <= max_strength",
            )
            if mutation is not None:
                rates = jnp.array([mutation.forward_rate, mutation.backward_rate])
                distribution = eqx.error_if(
                    distribution,
                    jnp.any(~jnp.isfinite(rates) | (rates < 0) | (rates > 1)),
                    "mutation rates must be finite and lie in [0, 1]",
                )
            return distribution, jnp.asarray(0.0)
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
