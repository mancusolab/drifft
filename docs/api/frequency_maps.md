# Frequency maps

Frequency maps convert parental allele frequencies into offspring allele
probabilities before binomial sampling. Selection, mutation, and their
compositions use the same interface.

## Interface and custom maps

??? abstract "AbstractFrequencyMap"

    ::: drifft.AbstractFrequencyMap
        options:
            members: [apply, __call__]

::: drifft.FrequencyMap
    options:
        members: [__init__]

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
