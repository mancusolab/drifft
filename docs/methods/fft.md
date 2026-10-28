# Local FFT convolution

The FFT backend computes binomial expectations through local log-odds grids.
For destination values $v_j$, it approximates

$$
B_M[v](q)=\sum_{j=0}^M\binom Mj q^j(1-q)^{M-j}v_j.
$$

Within a retained local grid, the binomial kernel has a diagonal–Toeplitz–diagonal
factorization. The action takes the form

$$
y_r=a_r\sum_j e^{-h(r-(j-j_0))^2/2}\,b_jv_j.
$$

Here $j_0$ is a reference count, $h$ is the log-odds spacing, and $a_r,b_j$
are scaling weights. The sum is a convolution: zero-padded FFTs evaluate it
without constructing a dense block. Local polynomial interpolation evaluates
the result at the required offspring probabilities. Near the endpoints,
direct binomial sums replace the log-odds interpolation.

The Gaussian-shaped kernel in this expression is a factorization of the
binomial kernel, not a normal approximation to the population model. The
numerical errors come from tail truncation, interpolation, and arithmetic.

An individual FFT of length $n$ costs $O(n\log n)$. A complete transition also
includes the local blocks, scaling, interpolation, and endpoint sums; a claim
about one transform is not the total complexity of the transition.

The expectation operation is transposed to propagate mass, using the same
approximation in both directions. Geometry is prepared for fixed population
sizes and numerical settings, while selection changes evaluation probabilities.
See [choosing algorithms](../guide/backends.md) for practical tradeoffs,
[accuracy](../guide/numerics.md) for validation, and the
[FFT API](../api/backends.md#drifft.FFT) for constructor settings.
