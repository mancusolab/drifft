# Evolution methods

Evolution methods propagate allele-count distributions over a constant-size
interval. They apply the prepared transition repeatedly or approximate its
power.

??? abstract "AbstractEpochMethod"

    ::: drifft.AbstractEpochMethod
        options:
            members: [validate, solve, advance]

::: drifft.prepare_epoch

::: drifft.EpochProblem
    options:
        members: [operator, initial, steps, reconstruct, error_norm, result]

::: drifft.EpochResult
    options:
        members: []

---

::: drifft.EvolutionStatus
    options:
        members: []

---

::: drifft.Sequential
    options:
        members: [__init__, advance]

---

::: drifft.PolynomialKrylov
    options:
        members: [__init__, advance]

---

::: drifft.RationalKrylov
    options:
        members: [__init__, advance]
