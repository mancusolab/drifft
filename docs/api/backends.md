# Numerical backends

Numerical backends evaluate binomial expectations and their transposes without
constructing the full transition matrix. They differ in how they approximate
and store this operation.

??? abstract "AbstractTransitionBackend"

    ::: drifft.AbstractTransitionBackend
        options:
            members: [expect]

---

::: drifft.FFT
    options:
        members: [__init__]

---

::: drifft.NUFFT
    options:
        members: [__init__]

---

::: drifft.Jacobi
    options:
        members: [__init__, expect]
