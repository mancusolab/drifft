# drifft

drifft is a JAX-based Python library for Wright–Fisher population genetics. It
propagates allele-count distributions, computes expectations and expected visits
before absorption, and supports differentiation through selection parameters.
FFT and Chebyshev/NUFFT backends share an Equinox interface.

[Try the quickstart](guide/quickstart.md){ .md-button .md-button--primary }
[Install drifft](guide/installation.md){ .md-button }
[Browse the Python API](api/transitions.md){ .md-button }

## Work with transitions

Prepare a backend for the destination population, bind a selection function with
`Transition`, then call `expect` or `advance`. The model uses `M = 2N` gene copies
for a diploid population of size `N`.

The [user guide](guide/transitions.md) covers custom selection and population-size
changes. Read [backend tradeoffs](guide/backends.md) to choose FFT or NUFFT, and
[expected visits](guide/solves.md) for absorption calculations and solver checks.

## Performance and accuracy

[Validation and benchmarks](benchmarks.md) describe the numerical checks and
comparisons with fastDTWF. Accuracy settings control approximations; solver
convergence alone does not certify exact-model accuracy.

## Support

Report bugs and feature requests in the
[issue tracker](https://github.com/mancusolab/drifft/issues).
For local setup and documentation builds, see [development](contributing.md).
