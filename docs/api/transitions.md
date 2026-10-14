# Transitions and results

`Transition` combines prepared geometry with a selection model or explicit
probabilities. See [transitions and selection](../guide/transitions.md) for usage.

::: drifft.Transition
    options:
        members: [__init__, from_probabilities, probabilities, expect, advance, expect_segregating, advance_segregating, as_lineax, expected_visits]

::: drifft.Underdominance
    options:
        members: [__call__]

::: drifft.ExpectedVisitsResult

## Demographic histories

See [demographic histories](../guide/demography.md) for interval timing and usage.

::: drifft.Demography
    options:
        members: [initial_ne, final_ne, total_generations]

::: drifft.DemographicTransition
    options:
        members: [__init__, advance, expect]
