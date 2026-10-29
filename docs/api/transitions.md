# Transitions

`Transition` binds a numerical backend to offspring probabilities and provides
expectation, propagation, and expected-visit operations.
See [transitions and frequency maps](../guide/transitions.md) for usage.
`ExpectedVisitsResult` contains the expected-visit solution and its diagnostics.

## Input contracts

The backend specifies destination size. For a frequency map,
`source_gene_copies` specifies the complete source count grid; explicit
probabilities can instead describe sampled source rows. `expect` maps
destination-length values to source-length expectations, and `advance` maps
source mass to destination mass.

Segregating and expected-visits operations require complete, ordered count
grids. Expected visits also requires equal source and destination sizes.
Inputs must have the documented shapes and use float64. Probabilities must be
finite and in `[0, 1]`, and selection parameters must satisfy the
[frequency-map domains](frequency_maps.md#selection). Shape checks raise during
construction or tracing; numerical ranges remain caller preconditions for
FFT/NUFFT transitions. Outputs are not clipped or renormalized.

## Transition

::: drifft.Transition
    options:
        members: [__init__, from_probabilities, probabilities, expect, advance, expect_segregating, advance_segregating, as_lineax, expected_visits]

::: drifft.ExpectedVisitsResult
    options:
        members: [__new__]
