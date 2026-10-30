# Computational tasks and approximations

The numerical problem is to propagate allele-count probabilities and compute
expectations without forming a large Wright–Fisher transition matrix. This page
connects those biological quantities to the algorithms in the following chapters;
the [model](../guide/model.md) defines the process and the
[notation](../guide/notation.md) fixes the matrix orientation.

## Three computational tasks

Loss probabilities, demographic likelihoods, and occupation times require
different operations on the same transition. Separating these tasks explains
why choosing a one-generation algorithm does not also determine how to evolve
a long history.

For a fixed population of $M$ gene copies, the main operations are

$$
\begin{aligned}
\text{One-generation expectation:}\quad & Pf,\\
\text{Evolution through }g\text{ generations:}\quad &(P^T)^g v_0,\\
\text{Visits before absorption:}\quad &u=(I-Q^T)^{-1}e_1.
\end{aligned}
$$

For example, choosing $f_j=\mathbf 1_{\{j=0\}}$ gives next-generation loss
probabilities conditional on each initial count. Propagating $v_0$ gives the
full count distribution after an interval. The visit equation assumes the
transience conditions in the [model chapter](../guide/model.md#interior-states-and-expected-visits).
The inverse is mathematical notation: drifft solves the
linear system rather than explicitly forming an inverse.

A dense transition has $(M+1)^2$ entries. Applying it costs $O(M^2)$, and
repeating that operation for $g$ generations costs $O(gM^2)$. Changing sizes
introduces rectangular transitions; a history composes them in time order.
These costs motivate approximating the transition action and, separately,
accelerating its repeated application.

## Representing one generation

A **backend** supplies the numerical representation used to apply one
transition. All backends target the same finite-population, discrete-time
binomial model; their differences are computational approximations.

**Dense** constructs binomial rows directly and provides a small-population
reference. **[FFT](fft.md)** exploits local convolution structure, while
**[NUFFT](nufft.md)** approximates binomial expectations by a Chebyshev polynomial.
Both optimized methods act on population-sized vectors.

**[Jacobi](jacobi.md)** expresses expectations in a polynomial basis with
separate endpoint terms. Its factorization $\widetilde P=DE$ also supplies the
small operator $H_{\mathrm J}=ED$ for evolution in coefficient coordinates.
This is a reduction of the discrete-time transition, not a diffusion model.

Constructing a backend precomputes quantities such as grids, quadrature nodes,
and interpolation weights. The API calls these **prepared geometry**: they
depend on population size and numerical settings and can be reused as selection
or mutation parameters change.

## Evolving an interval or solving for visits

An **evolution method** chooses how to apply a represented transition over many
generations. This choice controls a different source of computational cost from
the backend that represents one generation.

`Sequential` applies the represented transition once per generation.
[Polynomial and rational Krylov methods](jacobi_krylov.md) instead approximate
a power action in a smaller, input-dependent subspace. The **working state**
is the vector evolved by that calculation: population mass for FFT, NUFFT, and
Dense, or encoded coefficients for Jacobi with Krylov. The
[Jacobi chapter](jacobi.md#problem-and-representation) derives the reduced
operator and its relation to the physical interval.

[Expected visits](../guide/solves.md) use a linear solve on the interior
transition rather than a finite-time power. The solver guide describes the
residual checks and convergence requirements.

## Separating the errors

An accurate power calculation can still propagate an inaccurate transition.
Validation must distinguish representation error from power or solve error,
and check the final biological quantity and its gradient.

The transition approximation replaces $P$ by $\widetilde P$. Krylov adds an
approximation to the action of a power of the prepared operator; a visit solve
adds a linear-solve error. Floating-point arithmetic contributes throughout.
Sequential removes the Krylov approximation but retains backend and arithmetic
errors. Neither a power convergence flag nor a small solve residual bounds all
of these contributions.

The [accuracy guide](../guide/numerics.md) gives practical checks. The
[autodiff chapter](autodiff.md) explains how drifft differentiates these
representations, including the distinction between differentiating a matrix
power and differentiating the algorithm used to approximate it.
