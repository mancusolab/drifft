"""Prepared, linear hypergeometric observation operators."""

# pattern: Functional Core; host preparation and pure JAX linear actions.
import math
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import hypergeom


def _representative_counts(population: int, sample: int, spacing: float):
    """Space fixed counts by the hypergeometric standard deviation in count units."""
    if sample == 1:
        return np.array([0, population], dtype=np.int64)
    if spacing == 0:
        return np.arange(population + 1, dtype=np.int64)
    counts = [0]
    midpoint = population // 2
    while counts[-1] < midpoint:
        count = counts[-1]
        # Convert the sample-count SD to population-count units via M/n.
        scale = math.sqrt(
            count
            * (population - count)
            * (population - sample)
            / (sample * (population - 1))
        )
        counts.append(min(midpoint, count + max(1, math.floor(spacing * scale))))
    left = np.asarray(counts, dtype=np.int64)
    return np.unique(np.concatenate((left, population - left)))


@final
class HypergeometricSampling(eqx.Module):
    r"""Project population counts to a sample drawn without replacement.

    For population gene-copy count $M$, sample size $n$, and population count $j$,

    $$S_{jk}=\frac{\binom{j}{k}\binom{M-j}{n-k}}{\binom{M}{n}}.$$

    Fixed linear interpolation between representative rows and sparse tail
    truncation approximate $S$. `expect` applies this approximation to sample
    values; `advance` applies its matched transpose to population mass. Both
    are linear, including for signed inputs, and support JIT, AD, and vmap.
    No input-dependent grouping or output renormalization is performed.

    Interpolation preserves the conditional sampling mean before truncation.
    Truncated rows are normalized during preparation, which can perturb that
    mean. Endpoint rows are exact. Sampling the entire population is the
    identity; sampling zero copies returns total mass at count zero.

    !!! note "Theory"

        See [hypergeometric sampling](../methods/hypergeometric.md) for the
        compression scheme, accuracy controls, and usage.
    """

    population_gene_copies: int = eqx.field(static=True)
    sample_gene_copies: int = eqx.field(static=True)
    tv_sd: float = eqx.field(static=True)
    row_eps: float = eqx.field(static=True)
    representative_counts: jax.Array
    _left: jax.Array
    _fraction: jax.Array
    _row: jax.Array
    _count: jax.Array
    _weight: jax.Array

    def __init__(
        self,
        population_gene_copies: int,
        sample_gene_copies: int,
        *,
        tv_sd: float = 0.05,
        row_eps: float = 1e-10,
    ) -> None:
        """Prepare fixed interpolation and sparse rows outside JIT.

        **Arguments:**

        - `population_gene_copies`: Positive Python integer $M$.
        - `sample_gene_copies`: Python integer $n$ between zero and $M$.
        - `tv_sd`: Representative spacing in sample-standard-deviation units,
          in `[0, 1]`. Lower values use more rows; zero disables approximate
          grouping. Sampling one copy is represented exactly by its endpoints.
          This is a grouping heuristic, not a total-variation error bound.
        - `row_eps`: Tail-mass budget per representative row, in `[0, 1)`.
          Zero retains each row's full support. Hoeffding windows bound omitted
          mass in exact arithmetic; floating-point error is additional.

        **Returns:**

        `None`. SciPy prepares row probabilities on the host; application uses
        JAX arrays on the selected device. Storage is $O(M+R W)$ for $R$
        representatives and maximum retained row width $W$. Setting both
        controls to zero can require $O(Mn)$ storage.

        **Raises:**

        `ValueError` for invalid counts, controls, or disabled JAX x64;
        `RuntimeError` if row preparation produces invalid probabilities.
        Sizes and geometry are fixed, not differentiable parameters.
        """
        M, n = population_gene_copies, sample_gene_copies
        if isinstance(M, bool) or not isinstance(M, int) or M < 1:
            raise ValueError("population_gene_copies must be a positive Python integer")
        if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= M:
            raise ValueError("sample_gene_copies must be an integer in [0, M]")
        if not math.isfinite(tv_sd) or not 0 <= tv_sd <= 1:
            raise ValueError("tv_sd must be finite and in [0, 1]")
        if not math.isfinite(row_eps) or not 0 <= row_eps < 1:
            raise ValueError("row_eps must be finite and in [0, 1)")
        if not jax.config.x64_enabled:
            raise ValueError("HypergeometricSampling requires jax_enable_x64=True")
        self.population_gene_copies, self.sample_gene_copies = M, n
        self.tv_sd, self.row_eps = float(tv_sd), float(row_eps)
        if n == 0 or n == M:
            # These exact actions need no prepared population-sized geometry.
            self.representative_counts = jnp.empty(0, dtype=jnp.int64)
            self._left = self._row = self._count = jnp.empty(0, dtype=jnp.int64)
            self._fraction = self._weight = jnp.empty(0, dtype=jnp.float64)
            return

        knots = _representative_counts(M, n, tv_sd)
        widths = np.diff(knots)
        left = np.repeat(np.arange(len(widths), dtype=np.int64), widths)
        left = np.append(left, len(widths) - 1)
        fraction = (np.arange(M + 1) - knots[left]) / widths[left]
        radius = (
            math.sqrt(0.5 * n * (math.log(2) - math.log(row_eps))) if row_eps else n
        )
        rows, counts, weights = [], [], []
        for index, count in enumerate(knots):
            mean = n * (int(count) / M)
            lower = max(0, n - (M - int(count)), math.floor(mean - radius))
            upper = min(n, int(count), math.ceil(mean + radius))
            support = np.arange(lower, upper + 1, dtype=np.int64)
            probability = hypergeom.pmf(support, M, int(count), n)
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
        self._left = jnp.asarray(left)
        self._fraction = jnp.asarray(fraction, dtype=jnp.float64)
        self._row = jnp.asarray(np.concatenate(rows))
        self._count = jnp.asarray(np.concatenate(counts))
        self._weight = jnp.asarray(np.concatenate(weights), dtype=jnp.float64)

    @eqx.filter_jit
    def expect(self, values: jax.Array) -> jax.Array:
        """Return conditional expectations indexed by population count.

        `values` is a real vector of length `sample_gene_copies + 1`, converted
        to float64. The output has length `population_gene_copies + 1`.
        Nonfinite inputs propagate; use vmap for batches.
        """
        values = _values(values, self.sample_gene_copies + 1)
        if self.sample_gene_copies == self.population_gene_copies:
            return values
        if self.sample_gene_copies == 0:
            return jnp.full(self.population_gene_copies + 1, values[0])
        row_values = (
            jnp.zeros(self.representative_counts.size)
            .at[self._row]
            .add(self._weight * values[self._count])
        )
        return (1 - self._fraction) * row_values[
            self._left
        ] + self._fraction * row_values[self._left + 1]

    @eqx.filter_jit
    def advance(self, distribution: jax.Array) -> jax.Array:
        """Return sample-count mass using the transpose of `expect`.

        `distribution` is a real vector of length `population_gene_copies + 1`,
        converted to float64. The output has length `sample_gene_copies + 1`.
        Signed and unnormalized mass are supported; nonfinite inputs propagate.
        Endpoint bins are retained. Use vmap for batches.
        """
        distribution = _values(distribution, self.population_gene_copies + 1)
        if self.sample_gene_copies == self.population_gene_copies:
            return distribution
        if self.sample_gene_copies == 0:
            return jnp.sum(distribution, keepdims=True)
        mass = jnp.zeros(self.representative_counts.size)
        mass = mass.at[self._left].add((1 - self._fraction) * distribution)
        mass = mass.at[self._left + 1].add(self._fraction * distribution)
        return (
            jnp.zeros(self.sample_gene_copies + 1)
            .at[self._count]
            .add(self._weight * mass[self._row])
        )


def _values(values, length):
    """Validate a full, real count vector without discarding imaginary parts."""
    values = jnp.asarray(values)
    if values.shape != (length,) or jnp.iscomplexobj(values):
        raise ValueError(f"expected a real vector of length {length}")
    return values.astype(jnp.float64)
