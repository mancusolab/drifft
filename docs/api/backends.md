# Numerical backends

Numerical backends evaluate binomial expectations and their transposes. They
provide the numerical representation used by transitions and evolution methods.
All implement the shared [`expect`](#drifft.AbstractTransitionBackend.expect)
contract and inherit [`advance`](#drifft.AbstractTransitionBackend.advance) by
transposition unless they provide a specialized implementation. Concrete classes
document their configuration and numerical limitations.

??? abstract "AbstractTransitionBackend"

    ::: drifft.AbstractTransitionBackend
        options:
            members: [expect, advance]

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
        members: [__init__]

---

::: drifft.Dense
    options:
        members: [__init__]
