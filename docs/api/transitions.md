# Transitions

`Transition` binds a numerical backend to offspring probabilities and provides
expectation, propagation, and expected-visit operations.
See [transitions and frequency maps](../guide/transitions.md) for usage.
`ExpectedVisitsResult` contains the expected-visit solution and its diagnostics.

::: drifft.Transition
    options:
        members: [__init__, from_probabilities, probabilities, expect, advance, expect_segregating, advance_segregating, as_lineax, expected_visits]

::: drifft.ExpectedVisitsResult
    options:
        members: [__new__]
