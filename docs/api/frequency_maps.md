# Frequency maps

Frequency maps convert parental allele frequencies into offspring allele
probabilities before binomial sampling. Selection, mutation, and their
compositions use the same interface.

!!! warning "Shared domain contract"

    Every frequency map takes finite frequencies in `[0, 1]` and must return
    finite probabilities in `[0, 1]`. Parameter bounds differ by map and are
    listed in each constructor's arguments; selection coefficients are not
    generally restricted to `[0, 1]`.

    These value constraints are caller preconditions, including under JIT and
    vmap. Built-in constructors check scalar parameter shapes, not value bounds.
    Maps do not clip parameters or clip or normalize outputs. A custom
    [`FrequencyMap`](#drifft.FrequencyMap) must satisfy the same contract, and
    every component of a [`Composition`](#drifft.Composition) must be valid.

!!! note "Jacobi reduced Krylov restrictions"

    [`Jacobi`](backends.md#drifft.Jacobi) imposes additional restrictions when used
    for reduced Krylov evolution: the map must support reduced assembly, and
    both derived diploid selection coefficients must have absolute value at
    most its configured `max_strength`, which is strictly below one. These
    restrictions do not apply to one-generation operations or
    [`Sequential`](methods.md#drifft.Sequential) evolution; the frequency maps'
    own parameter domains still apply.

??? abstract "AbstractFrequencyMap"

    ::: drifft.AbstractFrequencyMap
        options:
            members: [apply, __call__]

## Selection

::: drifft.DiploidSelection
    options:
        members: [__init__]

---

::: drifft.DominanceSelection
    options:
        members: [__init__]

---

::: drifft.AdditiveSelection
    options:
        members: [__init__]

---

::: drifft.RecessiveSelection
    options:
        members: [__init__]

---

::: drifft.DominantSelection
    options:
        members: [__init__]

---

::: drifft.Underdominance
    options:
        members: [__init__]

---

::: drifft.Overdominance
    options:
        members: [__init__]

---

::: drifft.MultiplicativeSelection
    options:
        members: [__init__]

---

::: drifft.Neutral
    options:
        members: [__init__]

## Mutation and composition

::: drifft.Mutation
    options:
        members: [__init__]

---

::: drifft.Composition
    options:
        members: [__init__]

## Custom maps

::: drifft.FrequencyMap
    options:
        members: [__init__]
