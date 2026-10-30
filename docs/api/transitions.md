# Transitions

Transitions map allele-count distributions forward by one generation and
destination values to conditional expectations. They also provide segregating
operations and expected-visit solves.

::: drifft.Transition
    options:
        members: [__init__, probabilities, expect, advance, as_lineax, expected_visits]

---

::: drifft.ExpectedVisitsResult
    options:
        members: [__new__]

---

::: drifft.ExpectedVisitsStatus
    options:
        members: []
