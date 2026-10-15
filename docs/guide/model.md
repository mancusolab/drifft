# Wright–Fisher model

This page explains the probability model behind the public operations. For code
examples, see [frequency maps](transitions.md) and [demographic histories](demography.md).
For approximation error and derivatives, see [numerical accuracy](numerics.md).

## One generation

Let the source population contain $M_s$ gene copies and the destination contain
$M_d$. For diploid effective population size $N_e$, the model uses $M=2N_e$.
Source allele count $i$ corresponds to frequency $x_i=i/M_s$.

A frequency map gives the probability $q(x_i)$ of sampling allele 1 in an
offspring gene copy. Conditional on the source count, the destination count is

$$
J \mid I=i \sim \operatorname{Binomial}(M_d, q(x_i)).
$$

Consequently, the transition matrix has $M_s+1$ rows and $M_d+1$ columns:

$$
P_{ij}=\binom{M_d}{j}q(x_i)^j[1-q(x_i)]^{M_d-j}.
$$

`Transition.from_probabilities` instead accepts the row probabilities directly;
those rows need not represent a complete population grid. FFT and NUFFT provide
matrix-free approximations to the same binomial operation.

## Selection and mutation

`Neutral` uses $q(x)=x$. For `Underdominance`, the genotype fitnesses of
0/0, 0/1, and 1/1 are $1$, $1-s$, and $1$. Under random mating, mean fitness is
$1-2sx(1-x)$ and the fitness-weighted allele-1 contribution is $x-sx(1-x)$.
Dividing gives

$$
q_s(x)=\frac{x-sx(1-x)}{1-2sx(1-x)}.
$$

For `Mutation`, let $u$ be the 0 → 1 rate and $v$ the 1 → 0 rate. An allele-1
copy persists with probability $1-v$; an allele-0 copy changes with probability
$u$. The resulting frequency is

$$
m(p)=p(1-v)+(1-p)u.
$$

`Composition(selection, mutation)` therefore evaluates $m(q_s(x))$.
Reversing the arguments evaluates $q_s(m(x))$. These generally differ, so
composition order is part of the biological model.

For mutation alone, $m(0)=u$ and $m(1)=1-v$: both boundary states can be left
when their corresponding mutation rates are positive. Selection and mutation
rates remain differentiable parameters; population grid sizes are fixed metadata.

## Distributions and expectations

Treat an initial mass vector $\mu$ as a column vector. Forward propagation is

$$
\mu_{\mathrm{next}}=P^T\mu.
$$

For destination-state values $f$, conditional source expectations are $Pf$.
These operations satisfy

$$
f^T(P^T\mu)=(Pf)^T\mu.
$$

That identity explains the `.advance()` / `.expect()` pair. The implementation
transposes its approximate expectation operation; it does not independently
approximate propagation or renormalize its output.

## Demographic intervals

Write $P_{a,b}$ for the transition from diploid size $a$ to $b$. An interval of
$g$ generations in our API holds the source size for $g-1$ generations and changes
size on the last generation. Its expectation operator is

$$
A=P_{a,a}^{\,g-1}P_{a,b}.
$$

For successive intervals $A_0,\ldots,A_{K-1}$, the full expectation operator is
$A_0\cdots A_{K-1}$, and propagation applies its transpose. In code, `.advance()`
processes intervals chronologically; `.expect()` applies them in reverse order.
Each interval includes its size-changing generation. No initial equilibrium or
extra resampling operation is implicit.

## Interior states and expected visits

Segregating operations restrict $P$ to interior source and destination counts.
For a constant population, call that square restriction $Q$. Endpoint mass is
discarded rather than conditioned away. Even with recurrent mutation, repeated
interior operations stop trajectories at their first endpoint hit.

Starting from one allele copy, the expected interior occupation vector is

$$
g=\sum_{t=0}^{\infty}(Q^T)^t e_1,
\qquad (I-Q^T)g=e_1,
$$

when the series converges. Here $e_1$ puts unit mass on allele count one,
which is array index zero after removing the endpoints. The $t=0$ term includes
the initial generation. `expected_visits()` solves this system; it does not
compute the stationary distribution of the full recurrent-mutation chain.
See [expected visits](solves.md) for convergence diagnostics and limitations.
