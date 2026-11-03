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

::: drifft.PolynomialKrylov
    options:
        show_bases: false
        members: [__init__]

---

::: drifft.RationalKrylov
    options:
        show_bases: false
        members: [__init__]

## Results

::: drifft.EpochResult
    options:
        members: []

---

::: drifft.EvolutionStatus
    options:
        members: []

For implementing a method, see [custom evolution methods](../extending.md)
and the [extension API](extensions.md).
