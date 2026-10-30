# Transitions

Transitions map allele-count distributions forward by one generation and
destination values to conditional expectations. They also provide segregating
operations and expected-visit solves.

::: drifft.Transition
    options:
        members: [__init__, from_probabilities, probabilities, expect, advance, expect_segregating, advance_segregating, as_lineax, expected_visits]

---

::: drifft.ExpectedVisitsResult
    options:
        members: [__new__]
