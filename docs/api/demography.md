# Population histories

Population histories specify effective population sizes and interval durations.
A demographic transition combines a history, frequency map, backend, and
evolution method to propagate distributions across the history.

::: drifft.Demography
    options:
        members: [__init__, initial_ne, final_ne, total_generations]

---

::: drifft.DemographicTransition
    options:
        members: [__init__, advance, advance_segregating, expect]

---

::: drifft.DemographicResult
    options:
        members: []

---

::: drifft.IntervalDiagnostics
    options:
        members: []
