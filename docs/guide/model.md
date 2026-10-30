# Wright–Fisher model

This page explains the probability model behind the public operations. For code
examples, see [frequency maps](frequency_maps.md) and [demographic histories](demography.md).
For approximation error and derivatives, see [numerical accuracy](numerics.md).
Symbols follow the [shared notation](notation.md).

## One generation

One generation combines deterministic changes in allele frequency with
binomial reproduction. This defines the transition every backend approximates.

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

The [transition guide](transitions.md) shows how to apply this operator or
supply offspring probabilities directly.

## Selection and mutation

Selection and mutation determine the offspring sampling probability before
reproduction. Their order is part of the model, since the maps generally do
not commute.

`DiploidSelection(s_het, s_hom)` assigns genotype fitnesses $1$,
$1-s_{\mathrm{het}}$, and $1-s_{\mathrm{hom}}$ to 0/0, 0/1, and 1/1.
Under random mating, weighting genotype frequencies by survival gives

$$
q(x)=\frac{(1-s_{\mathrm{hom}})x^2+(1-s_{\mathrm{het}})x(1-x)}
{(1-x)^2+2(1-s_{\mathrm{het}})x(1-x)+(1-s_{\mathrm{hom}})x^2}.
$$

Negative coefficients give fitness advantages. The supported domain is
$s_{\mathrm{het}}\le1$ and $s_{\mathrm{hom}}<1$, with finite coefficients;
the strict homozygote bound keeps mean fitness positive at fixation.

`DominanceSelection(s, h)` sets $s_{\mathrm{het}}=hs$ and
$s_{\mathrm{hom}}=s$. `AdditiveSelection(s)` fixes $h=1/2$, giving
$s_{\mathrm{het}}=s/2$ and requiring $s<1$. This is the uncapped additive
model, rather than fastDTWF's capped default. The derived coefficient $hs$
must also be finite. `MultiplicativeSelection(r)` instead uses fitnesses
$(1,\rho,\rho^2)$ and gives $q(x)=\rho x/(1-x+\rho x)$ for $\rho>0$
(the API names this ratio `r`). Additive and
multiplicative fitness produce different exact discrete-time updates.

`Neutral` uses $q(x)=x$. For `Underdominance`, the genotype fitnesses of
0/0, 0/1, and 1/1 are $1$, $1-s$, and $1$. Under random mating, mean fitness is
$1-2sx(1-x)$ and the fitness-weighted allele-1 contribution is $x-sx(1-x)$.
Dividing gives

$$
q_s(x)=\frac{x-sx(1-x)}{1-2sx(1-x)}.
$$

Thus `Underdominance(strength=s)` is equivalent to
`DiploidSelection(s_het=s, s_hom=0)`.

With directional mutation rates $\mu_{0\to1}$ and $\mu_{1\to0}$,
`Mutation` gives the offspring frequency

$$
m(x)=x(1-\mu_{1\to0})+(1-x)\mu_{0\to1}.
$$

`Composition(selection, mutation)` therefore evaluates $m(q_s(x))$.
Reversing the arguments evaluates $q_s(m(x))$.

For mutation alone, $m(0)=\mu_{0\to1}$ and $m(1)=1-\mu_{1\to0}$: both boundary states can be left
when their corresponding mutation rates are positive. Selection and mutation
rates remain differentiable parameters; population grid sizes are fixed metadata.

## Distributions and expectations

Propagation moves mass forward; expectation moves an observable backward.
The transpose relation makes these two views of the same transition.

Treat an initial mass vector $v_t$ as a column vector. Forward propagation is

$$
v_{t+1}=P^Tv_t.
$$

For destination-state values $f$, conditional source expectations are $Pf$.
These operations satisfy

$$
f^T(P^Tv_t)=(Pf)^Tv_t.
$$

That identity explains the `.advance()` / `.expect()` pair. The implementation
transposes its approximate expectation operation; it does not independently
approximate propagation or renormalize its output.

## Demographic intervals

Each demographic interval includes its final population-size change. This
convention determines both the matrix product and the number of hold steps.

Write $P_{a,b}$ for the transition from diploid size $a$ to $b$. An interval of
$g$ generations in our API holds the source size for $g-1$ generations and changes
size on the last generation. Its expectation operator is

$$
P_{\mathrm{epoch}}=P_{a,a}^{\,g-1}P_{a,b}.
$$

For successive intervals $P_{\mathrm{epoch},0},\ldots,P_{\mathrm{epoch},K-1}$, the full expectation operator is
$P_{\mathrm{epoch},0}\cdots P_{\mathrm{epoch},K-1}$, and propagation applies its transpose. In code, `.advance()`
processes intervals chronologically; `.expect()` applies them in reverse order.
No initial equilibrium or extra resampling operation is implicit.

## Interior states and expected visits

Expected visits accumulate time spent at segregating counts before the first
endpoint hit. They describe a stopped process rather than a stationary
distribution of the recurrent-mutation model.

Segregating operations restrict $P$ to interior source and destination counts.
For a constant population, call that square restriction $Q$. Endpoint mass is
discarded rather than conditioned away. Even with recurrent mutation, repeated
interior operations stop trajectories at their first endpoint hit.

Starting from one allele copy, the expected interior occupation vector is

$$
u=\sum_{t=0}^{\infty}(Q^T)^t e_1,
\qquad (I-Q^T)u=e_1,
$$

when the series converges. Here $e_1$ puts unit mass on allele count one,
which is array index zero after removing the endpoints. The $t=0$ term includes
the initial generation. `expected_visits()` solves this system.
See [expected visits](solves.md) for convergence diagnostics and limitations.

Continue with [computational tasks and approximations](../methods/overview.md)
for how drifft evaluates these quantities at larger population sizes.
