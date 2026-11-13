# Numerical backends

Numerical backends evaluate binomial expectations and their transposes. They
provide the numerical representation used by transitions and evolution methods.
All implement the shared [`expect`](#drifft.AbstractTransitionBackend.expect)
contract and inherit [`advance`](#drifft.AbstractTransitionBackend.advance) by
transposition unless they provide a specialized implementation. Concrete classes
document their configuration and numerical limitations.

All built-in backends accept `atol` and expose `error_report`. The target is
one-step maximum row L1 error; it is not a total floating-point certificate.
See [accuracy and differentiation](../guide/numerics.md) for gradient targets,
parameter-domain bounds, and the distinction between bounds and budgets.

??? abstract "AbstractTransitionBackend"

    ::: drifft.AbstractTransitionBackend
        options:
            members: [expect, advance, error_report]

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

---

::: drifft.BackendErrorReport

::: drifft.ErrorComponent
