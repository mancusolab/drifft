# drifft

`drifft` computes allele-count distributions under the discrete-time
Wright–Fisher model. Use it to ask how likely an allele is to disappear, how a
population history changes its distribution, or how a predicted quantity
responds to selection. These calculations support differentiation in JAX.

[Try the quickstart](guide/quickstart.md){ .md-button .md-button--primary }
[Install drifft](guide/installation.md){ .md-button }

Begin with the [quickstart](guide/quickstart.md): compute the loss probability of a
new deleterious allele, then differentiate it with respect to selection.
Explore complete examples of [allele loss](examples/allele_loss.md),
a [population bottleneck](examples/bottleneck.md), and
[selection sensitivity](examples/selection_sensitivity.md). For a human-scale
selection-and-mutation calculation, see [sample allele-count likelihoods](examples/schiffels_durbin.md).

## Why the computation can be fast

A dense Wright–Fisher transition stores a binomial distribution for every
possible starting allele count. These rows have structure: nearby offspring
probabilities give similar distributions. `drifft` exploits that structure to
compute transitions without storing the full matrix.

The name brings together genetic drift and the FFT. Fast transforms accelerate
individual transitions; polynomial compression and Krylov methods can reduce
the work needed for long constant-size intervals. Their setup costs make the
best choice depend on the calculation. Start with the default FFT approach;
[choose another algorithm](guide/backends.md) when your workload calls for it.

## From a model to a result

A frequency map specifies selection and mutation. A `Transition` combines
that map with a prepared backend, then propagates a distribution with `advance`
or computes an expectation with `expect`.

For multiple generations, `Demography` records population sizes and durations.
`DemographicTransition` evolves an initial distribution through that history.
The numerical representation and evolution method can be changed separately
from the biological model.

Learn the [Wright–Fisher assumptions](guide/model.md), specify
[selection and mutation](guide/frequency_maps.md), then compute
[transitions](guide/transitions.md) or [population histories](guide/demography.md).
[Accuracy and differentiation](guide/numerics.md) explains how to check results.
Detailed algorithms and API contracts have their own reference sections.
