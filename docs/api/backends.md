# Numerical backends

Numerical backends evaluate binomial expectations and their transposes. They
provide the numerical representation used by transitions and evolution methods.

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

---

::: drifft.Dense
    options:
        members: [__init__, expect]
