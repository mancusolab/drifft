"""Prepared, linear hypergeometric observation operators."""

# pattern: Functional Core; host preparation and pure JAX linear actions.
import math
from typing import final

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax.experimental import hijax


def _interpolation_plan(population: int, sample: int, order: int, tolerance: float):
    """Bound the integer-grid Lagrange remainder using hypergeometric differences."""
    order = min(order, sample + 1, population + 1)
    if tolerance == 0:
        intervals = population
    elif order > sample:
        intervals = order - 1
    else:
        log_coefficient = (order - 1) * math.log(2) - math.log(order)
        log_coefficient += sum(
            math.log(sample - k) - math.log(population - k) for k in range(order)
        )
        log_spacing = (math.log(tolerance) - log_coefficient) / order
        spacing = max(1, math.floor(math.exp(min(math.log(population), log_spacing))))
        intervals = max(order - 1, (population + spacing - 1) // spacing)
    if intervals % 2 == 0:
        intervals += 1
    while True:
        if intervals >= population:
            return np.arange(population + 1, dtype=np.int64), 0.0, 1.0
        # Reflect the left half exactly, preserving allele-label symmetry.
        left = np.arange(intervals // 2 + 1, dtype=np.int64) * population // intervals
        knots = np.concatenate((left, population - left[::-1]))
        gaps = np.diff(knots)
        minimum, maximum = int(gaps.min()), int(gaps.max())
        bound = (
            0.0
            if order > sample
            else math.exp(log_coefficient + order * math.log(maximum))
        )
        if bound <= tolerance:
            break
        # Reflection can enlarge the central gap by one. Refine conservatively.
        intervals = max(intervals + 2, (population + maximum - 2) // (maximum - 1))
        if intervals % 2 == 0:
            intervals += 1
    # Bound each Lagrange weight over every possible unit cell in the stencil.
    # Nonuniform integer rounding contributes the maximum/minimum gap ratio.
    lebesgue = max(
        sum(
            math.prod(
                max(abs(cell - k), abs(cell + 1 - k)) for k in range(order) if k != i
            )
            / (math.factorial(i) * math.factorial(order - 1 - i))
            for i in range(order)
        )
        for cell in range(order - 1)
    ) * (maximum / minimum) ** (order - 1)
    return knots, bound, lebesgue


def _barycentric_weights(knots, order):
    """Normalize local coordinates before forming barycentric denominators."""
    stencils = np.lib.stride_tricks.sliding_window_view(knots, order)
    local = (stencils - stencils[:, :1]) / (stencils[:, -1:] - stencils[:, :1])
    difference = local[:, :, None] - local[:, None, :]
    difference[:, np.arange(order), np.arange(order)] = 1
    return 1 / np.prod(difference, axis=-1)


def _hypergeometric_row(population, sample, count, lower, upper):
    """Build normalized retained probabilities by ratios outward from the mode."""
    mode = min(upper, max(lower, (sample + 1) * (count + 1) // (population + 2)))
    right = np.arange(mode, upper, dtype=np.float64)
    right_ratio = ((count - right) / (right + 1)) * (
        (sample - right) / (population - count - sample + right + 1)
    )
    left = np.arange(mode, lower, -1, dtype=np.float64)
    left_ratio = (left / (count - left + 1)) * (
        (population - count - sample + left) / (sample - left + 1)
    )
    weights = np.concatenate(
        (np.cumprod(left_ratio)[::-1], [1.0], np.cumprod(right_ratio))
    )
    return weights / weights.sum()


def _interpolate_counts(
    knots, barycentric, values, *, population, batch_size, transposed
):
    """Gather or scatter local polynomial interpolation with bounded workspace."""
    if knots.size == population + 1:
        return values
    order = barycentric.shape[1]
    batches = (population + batch_size) // batch_size

    def stencil(batch):
        """Generate one batch of interpolation indices and fixed weights."""
        counts = jnp.minimum(batch * batch_size + jnp.arange(batch_size), population)
        # Invert the floor-spaced left half, then reflect its cell index.
        intervals = knots.size - 1
        low = jnp.minimum(counts, population - counts)
        cell = jnp.minimum(((low + 1) * intervals - 1) // population, intervals // 2)
        cell = jnp.where(counts <= population // 2, cell, intervals - 1 - cell)
        start = jnp.clip(cell + 1 - order // 2, 0, knots.size - order)
        indices = start[:, None] + jnp.arange(order)
        nodes = knots[indices]
        span = nodes[:, -1:] - nodes[:, :1]
        differences = (counts[:, None] - nodes) / span
        hit = differences == 0
        terms = barycentric[start] / jnp.where(hit, 1.0, differences)
        exact = jnp.any(hit, axis=-1, keepdims=True)
        weights = terms / jnp.where(exact, 1.0, terms.sum(axis=-1, keepdims=True))
        return indices, jnp.where(exact, hit.astype(values.dtype), weights)

    if transposed:
        padded = jnp.pad(values, (0, batches * batch_size - population - 1)).reshape(
            batches, batch_size
        )

        def scatter(mass, batch):
            """Accumulate into one representative-sized buffer across all batches."""
            index, vector = batch
            indices, weights = stencil(index)
            return mass.at[indices.reshape(-1)].add(
                (weights * vector[:, None]).reshape(-1)
            ), None

        return jax.lax.scan(
            scatter,
            jnp.zeros(knots.size, dtype=values.dtype),
            (jnp.arange(batches), padded),
        )[0]

    def gather(batch):
        """Evaluate one batch of population counts from representative values."""
        indices, weights = stencil(batch)
        return jnp.sum(weights * values[indices], axis=-1)

    return jax.lax.map(gather, jnp.arange(batches)).reshape(-1)[: population + 1]


class _CountInterpolation(hijax.HiPrim):
    r"""Apply fixed count-grid interpolation or its matched mass transpose.

    For prepared knots and barycentric weights, let $I$ interpolate knot values
    onto every population count. The two orientations compute $Iv$ and $I^T p$.
    Hypergeometric expectation and mass propagation can therefore share the
    same represented interpolation map instead of approximating their transposes
    independently.

    Knots and barycentric weights are fixed geometry, and their active tangents
    are rejected. Value JVPs apply the same linear action; the explicit transpose
    regenerates weights in streaming batches. Reverse rules are derived from
    the linearization so they retain geometry without needing input values or
    per-batch interpolation intermediates. This boundary controls memory while
    preserving the adjoint of the actual interpolation used by the primal.
    """

    def __init__(
        self, knots, barycentric, values, *, population, batch_size, transposed
    ):
        """Declare array inputs and the two action shapes."""
        self.in_avals = (knots, barycentric, values)
        self.out_aval = jax.core.ShapedArray(
            knots.shape if transposed else (population + 1,), values.dtype
        )
        self.params = dict(
            population=population, batch_size=batch_size, transposed=transposed
        )
        super().__init__()

    def expand(self, *args):
        """Lower to batched polynomial gathers or scatters."""
        return _interpolate_counts(*args, **self.params)

    def jvp(self, primals, tangents):
        """Differentiate input vectors, with fixed interpolation geometry."""
        knots, barycentric, values = primals
        dknots, dbarycentric, dvalues = tangents
        if not isinstance(dknots, hijax.Zero) or not isinstance(
            dbarycentric, hijax.Zero
        ):
            raise ValueError("sampling interpolation requires fixed geometry")
        tangent = (
            hijax.Zero(self.out_aval.to_tangent_aval())
            if isinstance(dvalues, hijax.Zero)
            else self(knots, barycentric, dvalues)
        )
        return self(*primals), tangent

    lin, linearized = hijax.linearize_from_jvp
    # The linear map needs only geometry in reverse mode, not the input values.
    vjp_fwd, vjp_bwd_retval = hijax.vjp_from_lin

    def transpose(self, out_ct, *maybe_accums):
        """Apply the opposite action without retaining per-batch intermediates."""
        knots, barycentric, values = maybe_accums
        if isinstance(knots, hijax.GradAccum) or isinstance(
            barycentric, hijax.GradAccum
        ):
            raise ValueError("sampling interpolation requires fixed geometry")
        if isinstance(out_ct, hijax.Zero):
            return
        if isinstance(values, hijax.GradAccum):
            values.accum(
                _interpolate(
                    knots,
                    barycentric,
                    out_ct,
                    **(self.params | {"transposed": not self.params["transposed"]}),
                )
            )

    def batch_dim_rule(self, axis_data, dims):
        """Keep the primal unmapped when only its tangent is batched."""
        return 0 if any(dim is not None for dim in dims) else None


def _interpolate(knots, barycentric, values, **config):
    """Bind the linear interpolation primitive to array avals."""
    return _CountInterpolation(
        jax.typeof(knots), jax.typeof(barycentric), jax.typeof(values), **config
    )(knots, barycentric, values)


@final
class HypergeometricSampling(eqx.Module):
    r"""Project population counts to a sample drawn without replacement.

    For diploid population size $N$ and diploid sample size $m$, let $M=2N$
    and $n=2m$ be the corresponding allele-copy counts. For population count $j$,

    $$S_{jk}=\frac{\binom{j}{k}\binom{M-j}{n-k}}{\binom{M}{n}}.$$

    Local polynomial interpolation between representative rows and sparse tail
    truncation approximate $S$. Operations are linear, including for signed
    inputs, and support JIT, AD, and vmap.
    No input-dependent grouping or output renormalization is performed.

    Interpolation preserves the conditional sampling mean before truncation.
    Higher orders can produce small negative entries; outputs are not clipped.
    Truncated rows are normalized during preparation. Endpoint rows are exact. Sampling the entire population is the
    identity; sampling zero copies returns total mass at count zero.

    !!! note "Theory"

        See [hypergeometric sampling](../methods/hypergeometric.md) for the
        compression scheme, accuracy controls, and usage.
    """

    pop_size: int = eqx.field(static=True)
    sample_size: int = eqx.field(static=True)
    accuracy: float = eqx.field(static=True)
    interp_order: int = eqx.field(static=True)
    tail_tol: float = eqx.field(static=True)
    interpolation_bound: float = eqx.field(static=True)
    tail_bound: float = eqx.field(static=True)
    lebesgue_bound: float = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    representative_counts: jax.Array
    _barycentric: jax.Array
    _row: jax.Array
    _count: jax.Array
    _weight: jax.Array

    def __init__(
        self,
        pop_size: int,
        sample_size: int,
        *,
        accuracy: float = 1e-8,
        interp_order: int = 8,
        tail_tol: float | None = None,
        batch_size: int = 256,
    ) -> None:
        """Prepare fixed interpolation and sparse rows outside JIT.

        Host preparation selects integer nodes from a finite-difference
        error bound and builds hypergeometric rows by mode-relative recurrence.
        JAX application stores $O(R W+R p)$ geometry for $R$ representatives,
        row width $W$, and stencil size $p$; interpolation uses bounded batches.
        `interpolation_bound + tail_bound` is the exact-arithmetic error budget.
        Zero accuracy can require $O(Mn)$ storage.

        **Arguments:**

        - `pop_size`: Diploid population size, a positive Python integer.
          The population contains `2 * pop_size` allele-bearing copies.
        - `sample_size`: Diploid sample size, a Python integer in `[0, pop_size]`.
          Draws `2 * sample_size` allele-bearing copies without replacement.
        - `accuracy`: Nominal maximum row half-L1 error, in `[0, 1)`.
          Split between polynomial interpolation and tail truncation; excludes
          floating-point error. Zero uses all rows and their full supports.
        - `interp_order`: Even stencil size from 2 through 16 (degree one less).
          Capped at `2 * sample_size + 1`, the exact polynomial dimension.
        - `tail_tol`: Optional stricter discarded mass per representative row,
          in `[0, 1)`. Zero disables tail truncation. The default accounts for
          amplification by interpolation weights.
        - `batch_size`: Positive integer controlling interpolation workspace.

        **Raises:**

        `ValueError` for invalid counts, controls, or disabled JAX x64;
        `RuntimeError` if row preparation produces invalid probabilities.
        Sizes and geometry are fixed, not differentiable parameters.
        """
        if isinstance(pop_size, bool) or not isinstance(pop_size, int) or pop_size < 1:
            raise ValueError("pop_size must be a positive Python integer")
        if (
            isinstance(sample_size, bool)
            or not isinstance(sample_size, int)
            or not 0 <= sample_size <= pop_size
        ):
            raise ValueError("sample_size must be an integer in [0, pop_size]")
        M, n = 2 * pop_size, 2 * sample_size
        if not math.isfinite(accuracy) or not 0 <= accuracy < 1:
            raise ValueError("accuracy must be finite and in [0, 1)")
        if (
            isinstance(interp_order, bool)
            or not isinstance(interp_order, int)
            or interp_order not in range(2, 17, 2)
        ):
            raise ValueError("interp_order must be an even integer from 2 through 16")
        if tail_tol is not None and (
            not math.isfinite(tail_tol) or not 0 <= tail_tol < 1
        ):
            raise ValueError("tail_tol must be finite and in [0, 1)")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive Python integer")
        if not jax.config.x64_enabled:
            raise ValueError("HypergeometricSampling requires jax_enable_x64=True")
        self.pop_size, self.sample_size = pop_size, sample_size
        self.accuracy = float(accuracy)
        self.interp_order = min(interp_order, n + 1)
        self.batch_size = min(batch_size, M + 1)
        if n == 0 or n == M:
            self.interpolation_bound = self.tail_bound = self.tail_tol = 0.0
            self.lebesgue_bound = 1.0
            self.representative_counts = jnp.empty(0, dtype=jnp.int64)
            self._row = self._count = jnp.empty(0, dtype=jnp.int64)
            self._barycentric = jnp.empty((0, self.interp_order), dtype=jnp.float64)
            self._weight = jnp.empty(0, dtype=jnp.float64)
            return

        knots, self.interpolation_bound, self.lebesgue_bound = _interpolation_plan(
            M, n, self.interp_order, accuracy / 2
        )
        budget = accuracy / (2 * self.lebesgue_bound)
        self.tail_tol = budget if tail_tol is None else min(float(tail_tol), budget)
        self.tail_bound = self.lebesgue_bound * self.tail_tol
        # Exact-row interpolation needs no polynomial coefficient storage.
        barycentric = (
            np.empty((0, self.interp_order))
            if len(knots) == M + 1
            else _barycentric_weights(knots, self.interp_order)
        )
        radius = (
            math.sqrt(0.5 * n * (math.log(2) - math.log(self.tail_tol)))
            if self.tail_tol
            else n
        )
        rows, counts, weights = [], [], []
        for index, count in enumerate(knots):
            mean = n * (int(count) / M)
            lower = max(0, n - (M - int(count)), math.floor(mean - radius))
            upper = min(n, int(count), math.ceil(mean + radius))
            support = np.arange(lower, upper + 1, dtype=np.int64)
            probability = _hypergeometric_row(M, n, int(count), lower, upper)
            total = probability.sum()
            if (
                not np.all(np.isfinite(probability))
                or np.any(probability < 0)
                or not total > 0
            ):
                raise RuntimeError("hypergeometric row preparation failed")
            rows.append(np.full(support.size, index, dtype=np.int64))
            counts.append(support)
            weights.append(probability / total)
        self.representative_counts = jnp.asarray(knots)
        self._barycentric = jnp.asarray(barycentric, dtype=jnp.float64)
        self._row = jnp.asarray(np.concatenate(rows))
        self._count = jnp.asarray(np.concatenate(counts))
        self._weight = jnp.asarray(np.concatenate(weights), dtype=jnp.float64)

    @property
    def _population_copies(self) -> int:
        """Number of allele-bearing copies in the population."""
        return 2 * self.pop_size

    @property
    def _sample_copies(self) -> int:
        """Number of allele-bearing copies drawn without replacement."""
        return 2 * self.sample_size

    def source_size(self) -> int:
        r"""Number of population states (rows of the sampling matrix $S$).

        For diploid population size $N$, allele counts range from zero to $2N$,
        giving $2N+1$ states.

        **Arguments:**

        None.

        **Returns:**

        The number of rows in $S$, as a Python integer.
        """
        return self._population_copies + 1

    def dest_size(self) -> int:
        r"""Number of sample states (columns of the sampling matrix $S$).

        For diploid sample size $m$, allele counts range from zero to $2m$,
        giving $2m+1$ states. An empty sample has one state: zero copies.

        **Arguments:**

        None.

        **Returns:**

        The number of columns in $S$, as a Python integer.
        """
        return self._sample_copies + 1

    @eqx.filter_jit
    def expect(self, values: jax.Array) -> jax.Array:
        r"""Compute expected sample quantities for each population allele count.

        For sample-count values $v_k$, evaluate

        $$ (Sv)_j = \sum_k S_{jk}v_k = \mathbb{E}[v_K \mid J=j], $$

        where $J$ is the population allele count and $K$ is the count in a
        sample drawn without replacement. Evaluation uses the prepared
        approximation to $S$.

        !!! example "Example: Expected sample allele count"

            ```python
            # For each population count, find the expected number of copies in the sample.
            counts = jnp.arange(sampling.dest_size())
            expected_counts = sampling.expect(counts)
            ```

        **Arguments:**

        - `values`: Real vector of length `dest_size()`, converted
          to float64. Signed entries are supported; nonfinite inputs propagate.
          Use vmap for batches.

        **Returns:**

        A float64 vector of length `source_size()`, with one
        conditional expectation per population count, including both endpoints.

        **Raises:**

        `ValueError` for an incompatible vector shape or complex-valued input.
        """
        values = _values(values, self._sample_copies + 1)
        if self._sample_copies == self._population_copies:
            return values
        if self._sample_copies == 0:
            return jnp.full(self._population_copies + 1, values[0])
        row_values = (
            jnp.zeros(self.representative_counts.size)
            .at[self._row]
            .add(self._weight * values[self._count])
        )
        return _interpolate(
            self.representative_counts,
            self._barycentric,
            row_values,
            population=self._population_copies,
            batch_size=self.batch_size,
            transposed=False,
        )

    @eqx.filter_jit
    def advance(self, distribution: jax.Array) -> jax.Array:
        r"""Project population allele-count mass to a sample drawn without replacement.

        For population mass $p_j$, evaluate

        $$ (S^T p)_k = \sum_j S_{jk}p_j. $$

        For a probability distribution $p$, entry $k$ is the probability of
        observing $k$ copies in the sample, averaging over uncertainty in the
        population count. Evaluation uses the transpose of the prepared
        approximation to $S$.

        !!! example "Example: Sampling a singleton"

            ```python
            # With one copy in the population, find the probability of each sample count.
            population = jnp.zeros(sampling.source_size()).at[1].set(1)
            sample_distribution = sampling.advance(population)
            ```

        **Arguments:**

        - `distribution`: Real vector of length `source_size()`,
          converted to float64. Signed and unnormalized mass are supported;
          nonfinite inputs propagate. Use vmap for batches.

        **Returns:**

        A float64 vector of length `dest_size()`, including samples
        with zero copies or all copies of the allele. No renormalization occurs.

        **Raises:**

        `ValueError` for an incompatible vector shape or complex-valued input.
        """
        distribution = _values(distribution, self._population_copies + 1)
        if self._sample_copies == self._population_copies:
            return distribution
        if self._sample_copies == 0:
            return jnp.sum(distribution, keepdims=True)
        mass = _interpolate(
            self.representative_counts,
            self._barycentric,
            distribution,
            population=self._population_copies,
            batch_size=self.batch_size,
            transposed=True,
        )
        return (
            jnp.zeros(self._sample_copies + 1)
            .at[self._count]
            .add(self._weight * mass[self._row])
        )


def _values(values, length):
    """Validate a full, real count vector without discarding imaginary parts."""
    values = jnp.asarray(values)
    if values.shape != (length,) or jnp.iscomplexobj(values):
        raise ValueError(f"expected a real vector of length {length}")
    return values.astype(jnp.float64)
