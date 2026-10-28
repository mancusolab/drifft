# Population histories

`Demography` specifies population sizes and interval durations.
`DemographicTransition` combines that history with a frequency map, backend,
and evolution method. Its outputs include whole-history results and interval
diagnostics. See [demographic histories](../guide/demography.md) for usage.

::: drifft.Demography
    options:
        members: [__init__, initial_ne, final_ne, total_generations]

::: drifft.DemographicTransition
    options:
        members: [__init__, advance, advance_segregating, expect]

::: drifft.DemographicResult
    options:
        members: []

::: drifft.IntervalDiagnostics
    options:
        members: []
