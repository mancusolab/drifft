"""Compute and plot sample likelihoods for the bundled demographic history."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import drifft

jax.config.update("jax_enable_x64", True)

# Bundled example history; see the documentation for provenance limitations.
history_data = np.loadtxt(
    Path(__file__).with_name("data") / "schiffels_durbin.tsv",
    delimiter="\t",
    skiprows=1,
)
raw_time, raw_ne = history_data.T

# The default run ends before the large recent expansion.
end_time = 252
keep = raw_time >= end_time
pop_sizes = np.rint(raw_ne[keep]).astype(int).tolist()
boundaries = np.rint(raw_time[keep]).astype(int)
generations = (boundaries[:-1] - boundaries[1:]).tolist()
history = drifft.Demography(pop_sizes=pop_sizes, generations=generations)
print(f"History endpoint: {boundaries[-1]} generations ago")

frequency_map = drifft.Composition(
    drifft.Mutation(forward_rate=1e-8, backward_rate=1e-8),
    drifft.AdditiveSelection(s=2e-4),
)
model = drifft.DemographicTransition(
    history,
    backend_factory=drifft.Jacobi,
    boundary_backend_factory=drifft.FFT,
    method=drifft.RationalKrylov(dimension=128, rtol=1e-10, atol=1e-13),
    frequency_map=frequency_map,
)

# Explicit ancestral condition: the derived allele is absent at the oldest boundary.
initial = jnp.zeros(2 * history.initial_pop_size + 1).at[0].set(1.0)
result = model.advance(initial, throw=False)
# In practice, check result.converged and that population mass is finite and sums to one.
population_probabilities = result.distribution

# Sample 100 gene copies without replacement from the endpoint population.
sample_size = 50
observed_count = 5  # Illustrative observation, not an empirical dataset.
sampling = drifft.HypergeometricSampling(
    pop_size=history.final_pop_size,
    sample_size=sample_size,
    accuracy=1e-10,
)
sample_probabilities = sampling.advance(population_probabilities)
# In practice, check sample mass and positivity before conditioning or taking logs.

# Condition only if the observation scheme selects sample-segregating sites.
segregating = sample_probabilities[1:-1]
conditional_probabilities = segregating / segregating.sum()
likelihood = sample_probabilities[observed_count]
conditional_likelihood = conditional_probabilities[observed_count - 1]
print(f"P(K={observed_count}) = {likelihood:.6g}")
print(f"Log likelihood = {jnp.log(likelihood):.6g}")
print(f"P(K={observed_count} | sample segregating) = {conditional_likelihood:.6g}")

sample_copies = 2 * sample_size
counts = np.arange(1, sample_copies)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(counts / sample_copies, conditional_probabilities, label="Model")
ax.scatter(
    [observed_count / sample_copies],
    [conditional_likelihood],
    color="C1",
    zorder=3,
    label=f"K={observed_count} of n={sample_copies}",
)
ax.legend(frameon=False)
ax.set(
    xlabel="Sample derived-allele frequency",
    ylabel="Conditional count probability",
    yscale="log",
    title=f"Sample at {boundaries[-1]} generations ago",
)
fig.tight_layout()
fig.savefig("schiffels_durbin_likelihood.svg")
plt.show()
