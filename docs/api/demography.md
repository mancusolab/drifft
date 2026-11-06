# Population histories

[`Demography`](#drifft.Demography) defines sizes and durations. Call `history.prepare(...)` outside JIT
to create reusable [`DemographicGeometry`](#drifft.DemographicGeometry), then bind it to a frequency map and
evolution method with [`DemographicTransition`](#drifft.DemographicTransition). Binding is lightweight and can
happen inside `jax.grad`, `jax.jit`, or `jax.vmap`.

Configure `backend_factory` and `boundary_backend_factory` when preparing the
history. The resulting geometry can be shared across frequency maps and evolution
methods. Bind `frequency_map` and `method` when constructing a transition;
parameter-dependent operators are evaluated using that bound process.

::: drifft.Demography
    options:
        members: [__init__, prepare]

---

::: drifft.DemographicGeometry
    options:
        members: []

---

::: drifft.DemographicTransition
    options:
        members: [__init__, advance, expect]

---

::: drifft.DemographicResult
    options:
        members: []

---

::: drifft.IntervalDiagnostics
    options:
        members: []

---

::: drifft.DemographicExpectation
    options:
        members: []
