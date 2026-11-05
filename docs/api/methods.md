# Evolution methods

Evolution methods propagate allele-count distributions over a constant-size
interval. They apply the prepared transition repeatedly or approximate its
power. For direct interval calls, see
[`AbstractEpochMethod.advance`](#drifft.AbstractEpochMethod.advance).

## Source mass

Pass `source_mass` to add the same mass vector after every generation, including
the last. It must have the same shape as the initial distribution: full allele
counts by default, or interior counts with `segregating=True`. In segregating
mode, endpoint mass is discarded before the source is added. `None` adds no
source, and zero generations returns the initial distribution without additions.

For example, introduce singleton mass into an initially empty spectrum:

```python
import jax
import jax.numpy as jnp
import drifft

jax.config.update("jax_enable_x64", True)
backend = drifft.FFT(pop_size=64)
initial = jnp.zeros(127)
source_mass = jnp.zeros_like(initial).at[0].set(128 * 1e-8)
result = drifft.Sequential().advance(
    backend,
    initial,
    frequency_map=drifft.Neutral(),
    generations=20,
    segregating=True,
    source_mass=source_mass,
)
```

`source_mass` is mass per generation, with no automatic population-size scaling
or normalization. For a population history, use
`DemographicTransition.advance(..., singleton_mutation_rate=mu)` to add
`2 * dest_pop_size * mu` at count one after each generation. See
[new mutations through time](../examples/mutation_input.md) for a complete example.

## Method reference

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
of population coordinates, backend representations, and source mass.
`prepare_epoch` packages these details into an `EpochProblem`, which supplies
the working operator, initial vector, and exponent, plus methods to reconstruct
population mass and check the result. For a complete walkthrough, see
[custom evolution methods](../extending.md).

::: drifft.prepare_epoch

::: drifft.EpochProblem
    options:
        members: [reconstruct, error_norm, result]
