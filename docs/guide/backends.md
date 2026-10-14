# Choosing a backend

Both backends implement `AbstractTransitionBackend`, an abstract Equinox module.
`Transition` owns selection, source size, propagation, and solver integration;
backends own selection-independent geometry and the expectation approximation.
Choose a backend explicitly and reuse it across selection strengths.

`FFT` uses local log-odds blocks, cached Fourier coefficients, and interpolation.
It favors repeated applications at a fixed size, at the cost of larger geometry
and XLA temporary storage. Its constructor accepts `tail_tol`, `interp_order`,
`min_mode_count`, `block_spacing`, and `balance`. Host NumPy/SciPy preparation
uses binomial quantiles and extended precision for conditioned Fourier weights;
application and differentiation use JAX arrays. Interpolation boundaries can
limit gradient smoothness and should be checked for inference workloads.

`NUFFT` stores a small node vector and regenerates sampling weights, trading
extra work per application for lower stored geometry and XLA temporary storage.
Its polynomial derivative rules remain finite at probability endpoints.
These custom operations use experimental `jax.experimental.hijax.HiPrim` to
preserve explicit JVP, transpose, and batching behavior under JAX transforms.
Hijax does not itself make gradients faster or reduce demographic recompilation:
changing population shapes can require new compilations for either backend.
