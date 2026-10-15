# Transitions and results

`Transition` combines prepared geometry with a frequency map or explicit
probabilities. See [transitions and frequency maps](../guide/transitions.md) for usage.

::: drifft.Transition
    options:
        members: [__init__, from_probabilities, probabilities, expect, advance, expect_segregating, advance_segregating, as_lineax, expected_visits]

::: drifft.ExpectedVisitsResult
    options:
        members: [__new__]

## Frequency maps

??? abstract "`drifft.AbstractFrequencyMap`"

    ::: drifft.AbstractFrequencyMap
        options:
            members: [apply, __call__]

::: drifft.FrequencyMap
    options:
        members: [__init__]

::: drifft.Neutral
    options:
        members: [__init__]

::: drifft.Underdominance
    options:
        members: [__init__]

::: drifft.Mutation
    options:
        members: [__init__]

::: drifft.Composition
    options:
        members: [__init__]

## Demographic histories

See [demographic histories](../guide/demography.md) for interval timing and usage.

::: drifft.Demography
    options:
        members: [__init__, initial_ne, final_ne, total_generations]

::: drifft.DemographicTransition
    options:
        members: [__init__, advance, expect]
