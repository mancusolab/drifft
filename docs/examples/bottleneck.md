# A population bottleneck

Does a temporary reduction in population size change allele frequencies when
there is no selection? Compare a bottleneck with a constant-size population,
starting at allele frequency $1/4$. Both histories last 41 generations, begin
and end at 64 diploid individuals, and use neutral reproduction without mutation.

![Final allele-frequency distributions and endpoint probabilities](../_static/examples/bottleneck.svg)

## Calculation

After [installing drifft](../index.md#installation), save the code below as
`bottleneck.py` and run `uv run --with matplotlib python bottleneck.py`.
Matplotlib is needed only for plotting.

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import drifft

jax.config.update("jax_enable_x64", True)
histories = {
    "Constant size": drifft.Demography(ne=[64, 64, 64], generations=[1, 40]),
    "Bottleneck": drifft.Demography(ne=[64, 16, 64], generations=[1, 40]),
}
initial = jnp.zeros(129).at[32].set(1.0)  # Initial frequency is 1/4.
frequencies = jnp.arange(129) / 128
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

for index, (label, history) in enumerate(histories.items()):
    model = drifft.DemographicTransition(
        history,
        backend_factory=drifft.FFT,
        method=drifft.Sequential(),
        frequency_map=drifft.Neutral(),
    )
    result = model.advance(initial)
    assert result.converged
    distribution = result.distribution
    mean = distribution @ frequencies
    variance = distribution @ (frequencies - mean)**2
    assert jnp.isclose(distribution.sum(), 1.0, atol=1e-10)
    assert jnp.isclose(mean, 0.25, atol=1e-10)
    print(f"{label}: mean={mean:.4f}, variance={variance:.4f}, "
          f"loss={distribution[0]:.4f}, fixation={distribution[-1]:.4f}")
    axes[0].step(frequencies, jnp.cumsum(distribution), where="post", label=label)
    offset = (index - 0.5) * 0.35
    axes[1].bar(jnp.array([0.0, 1.0]) + offset,
                jnp.array([distribution[0], distribution[-1]]),
                width=0.35, label=label)

axes[0].set(xlabel="Final allele frequency", ylabel="Cumulative probability")
axes[1].set(xticks=[0, 1], xticklabels=["Loss", "Fixation"], ylabel="Probability")
for ax in axes:
    ax.legend()
fig.tight_layout()
fig.savefig("bottleneck.svg")
plt.show()
```

## Interpretation

The bottleneck increases variance and endpoint probabilities while preserving
mean frequency $1/4$. The CDF includes loss and fixation; the right panel shows
those masses separately.

## When the size changes

Sizes run oldest to newest. Each interval holds its source size until its
final generation, when the next size takes effect. For the bottleneck above:

| Generations | Reproduction |
| --- | --- |
| 1 | Change from 64 to 16 diploid individuals. |
| 2–40 | Reproduce at size 16. |
| 41 | Change from 16 to 64; take the final output. |

Each boundary step includes selection (neutral here) and binomial reproduction.
There is no extra resampling or subsequent evolution at the final size.

The same elapsed time and final allele-count grid make the two outputs directly
comparable. See [population histories](../api/demography.md) for the timing
convention and [the Wright–Fisher model](../guide/model.md) for the probability law.
