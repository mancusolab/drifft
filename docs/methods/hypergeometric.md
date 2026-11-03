# Hypergeometric sampling

`HypergeometricSampling` maps a final population distribution to a sample drawn
without replacement. It does not advance a generation or apply selection.

Configuration: [API reference](../api/sampling.md).

## Problem and representation

For $M$ population gene copies and $n$ sampled copies, the observation matrix is

$$
S_{jk}=\frac{\binom{j}{k}\binom{M-j}{n-k}}{\binom{M}{n}},
\qquad j=0,\ldots,M,\quad k=0,\ldots,n.
$$

Its transpose maps population mass to sample mass. Multiplying sample values
by $S$ gives conditional expectations indexed by population count.

## Computation

A full matrix costs $O(Mn)$ to store and apply. Nearby population counts have
similar sampling distributions, and each row concentrates around its mean.
Fixed interpolation and tail truncation exploit these two properties: aggregate
mass at representative counts, then apply their sparse hypergeometric rows.
This costs $O(M+R W)$ per action for $R$ representatives and maximum retained
row width $W$, without constructing the full matrix.

For representative counts $a\le j\le b$, approximate a row by

$$
\widetilde S_{j,:}=(1-t_j)R_{a,:}+t_jR_{b,:},\qquad
 t_j=\frac{j-a}{b-a}.
$$

The weights depend only on counts, not on the incoming distribution. Thus the
approximation is linear, has a matched transpose, and supports derivatives
with respect to input mass or values. Before truncation, interpolation preserves
the conditional sampling mean $nj/M$ because that mean is linear in $j$.

Representative spacing scales with the hypergeometric standard deviation,
converted to population-count units; smaller `tv_sd` retains more rows.
A symmetric grid includes both endpoints. Each representative row retains its
exact support intersected with a Hoeffding window of radius
$\sqrt{(n/2)\log(2/\mathtt{row\_eps})}$ around its mean, rounded outward.
Retained rows are normalized once during preparation.

!!! note "Relation to fastDTWF"

    [Spence et al. (2023)](https://doi.org/10.1093/genetics/iyad168) develop
    total-variation compression and tail truncation for hypergeometric sampling.
    fastDTWF selects representatives using the incoming mass. This implementation
    instead uses fixed interpolation to preserve linearity; the paper's grouping
    error guarantees do not automatically transfer to this scheme.

## Accuracy and controls

- `tv_sd` controls grouping, not a certified total-variation tolerance. Decrease
  it and compare the sample statistic of interest. Zero disables approximate
  grouping. Sampling one copy uses an exact interpolation of the endpoint rows.
- `row_eps` bounds omitted mass per representative row in exact arithmetic.
  Zero retains full support. Normalizing truncated rows preserves total mass
  but can perturb the sampling mean; floating-point error is additional.

Interpolation can assign small probabilities outside a source count's exact
hypergeometric support. It also increases the conditional variance before tail
truncation: preserving the mean does not preserve the whole distribution.
Use `tv_sd=0` when exact support is required, and validate rare-bin likelihoods
with finer grouping rather than relying on mass or mean checks alone.

Both endpoints are exact. Sampling all copies is the identity; sampling zero
copies returns total mass at count zero. These cases need no stored rows.
Setting both controls to zero provides an uncompressed reference for small
problems. Construction uses SciPy on the host; application uses JAX on the
selected device. GPU execution has not yet been validated.

## Usage

```python
import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)
sampling = drifft.HypergeometricSampling(
    population_gene_copies=128, sample_gene_copies=20,
)
population = jnp.zeros(129).at[32].set(1.0)
sample = sampling.advance(population)

values = jnp.arange(21) / 20
expected_frequency = population @ sampling.expect(values)
assert jnp.isclose(expected_frequency, 0.25, rtol=0, atol=1e-10)
```

For a history, pass `model.advance(initial).distribution` to `sampling.advance`.
To work backward, pass `sampling.expect(sample_values)` to `model.expect`;
matched demographic expectations require Sequential.

The operator accepts full count vectors, including endpoints. For a segregating
population spectrum, explicitly pad it with zeros at both ends. Sampling can
produce monomorphic observations even from a segregating population; condition
on sample segregation only after projection, if the observation scheme requires it.

This implementation still reads the full population vector. Direct projection
from Jacobi's reduced state is not implemented. A small measured TV error
controls absolute probability error; rare-bin relative errors and conditional
likelihoods need separate checks.
