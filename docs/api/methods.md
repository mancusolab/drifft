# Evolution methods

Evolution methods propagate allele-count distributions over a constant-size
interval. They apply the prepared transition repeatedly or approximate its
power. For direct interval calls and immigration, see
[`AbstractEpochMethod.advance`](#drifft.AbstractEpochMethod.advance).

??? abstract "AbstractEpochMethod"

    ::: drifft.AbstractEpochMethod
        options:
            members: [validate, solve, advance]

---

::: drifft.Sequential
    options:
        members: [__init__]

---

??? abstract "AbstractKrylovMethod"

    ::: drifft.AbstractKrylovMethod
        options:
            members: []
            inherited_members: false

::: drifft.PolynomialKrylov
    options:
        show_bases: true
        members: [__init__]

---

::: drifft.RationalKrylov
    options:
        show_bases: true
        members: [__init__]

::: drifft.EpochResult
    options:
        members: []

---

::: drifft.EvolutionStatus
    options:
        members: []

## Implementing custom methods

Custom methods can focus on the power algorithm while reusing drifft's handling
of population coordinates, backend representations, and immigration.
`prepare_epoch` packages these details into an `EpochProblem`, which supplies
the working operator, initial vector, and exponent, plus methods to reconstruct
population mass and check the result. For a complete walkthrough, see
[custom evolution methods](../extending.md).

::: drifft.prepare_epoch

::: drifft.EpochProblem
    options:
        members: [reconstruct, error_norm, result]
