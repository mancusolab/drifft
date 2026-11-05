# Differentiating Wright–Fisher evolution

`drifft` defines derivatives through the mathematical actions of a transition
or demographic history. Numerical methods evaluate those actions and their
sensitivities. This chapter explains the derivative targets, how forward and
backward calculations relate, and the accuracy limits to check in inference.
For a worked example using `jax.grad`, see
[differentiating selection](../tour.md#differentiate-selection).

## What is differentiable?

Selection and recurrent-mutation parameters, input mass or terminal values,
and singleton source rates are differentiable. Population sizes, generation
counts, prepared geometry, and numerical method settings remain fixed.
Parameter-dependent encoding and reconstruction transforms are included in
the chain rule.

Let $\widetilde P_t(\theta)$ denote the represented one-generation expectation
operator. Its transpose propagates population mass. The tilde distinguishes
the numerical representation from the exact Wright–Fisher operator: an
accurate derivative of the representation can still contain backend error.
A dot denotes a directional derivative, and all inner products here are real.
See the [shared notation](../guide/notation.md) for other conventions.

## Demographic actions

[`DemographicTransition`](../api/demography.md#drifft.DemographicTransition)
differentiates the operation defined by a history. Let
$\widetilde H=\widetilde P_0\cdots\widetilde P_{T-1}$ be its represented
expectation operator, and let $c_H$ be the final mass accumulated from all
singleton sources when the initial mass is zero. Writing `advance(v)` for
its returned distribution, the mathematical targets are

$$
\operatorname{advance}(v)=\widetilde H^T v+c_H,\qquad
\operatorname{expect}(f)=(\widetilde Hf,\;c_H^Tf).
$$

Here `expect` returns `values` and `offset` separately. In particular,
$v^T\operatorname{expect}(f).\mathrm{values}
+\operatorname{expect}(f).\mathrm{offset}$ is the terminal reward.
The same formulas apply to segregating states after restricting every
one-generation kernel to interior counts. An empty history has
$\widetilde H=I$ and $c_H=0$.

### One augmented operator for both orientations

Define the rectangular augmented forward operator

$$
\mathcal A(\theta)=
\begin{pmatrix}\widetilde H(\theta)^T&c_H(\theta)\\0&1\end{pmatrix}.
$$

Forward evolution multiplies this matrix by the initial mass vector with an
extra scalar coordinate equal to one:

$$
\mathcal A(\theta)
\begin{pmatrix}v\\1\end{pmatrix}
=
\begin{pmatrix}\widetilde H(\theta)^T v+c_H(\theta)\\1\end{pmatrix}.
$$

The extra coordinate includes the source contribution. Backward expectation
multiplies the transposed matrix by terminal values with an extra zero:

$$
\mathcal A(\theta)^T
\begin{pmatrix}f\\0\end{pmatrix}
=
\begin{pmatrix}\widetilde H(\theta)f\\c_H(\theta)^T f\end{pmatrix}.
$$

The output contains the backward values and the scalar source reward.
Writing $x$ and $y$ for these augmented input vectors, the product rule
covers both orientations:

$$
\mathrm D[\mathcal A x][\dot\theta,\dot x]
=\dot{\mathcal A}x+\mathcal A\dot x,\qquad
\mathrm D[\mathcal A^T y][\dot\theta,\dot y]
=\dot{\mathcal A}^{\,T}y+\mathcal A^T\dot y.
$$

The vector pullback invokes the opposite orientation. For forward output
cotangent $g$, the initial-mass cotangent is $\widetilde H g$.
For backward output cotangents $(a,b)$, where $b$ weights the scalar offset,
the terminal-value cotangent is $\widetilde H^Ta+b c_H$. Discarding the offset
sets $b=0$; differentiating only the offset sets $a=0$.

Both parameter pullbacks come from the same scalar pairing
$y^T\mathcal A(\theta)x$, holding $x$ and $y$ fixed during that contraction.
This includes selection, recurrent mutation, and singleton-rate dependence.
For example, at singleton rate zero, $c_H=0$ but its rate derivative generally
remains nonzero. The derivative of backward `values` with respect to that rate
is zero; the derivative of `offset` need not be.

## JVPs and pullbacks

A Jacobian–vector product (JVP) computes the change in an output along a
specified input direction. Without a source, the two action rules reduce to

$$
\mathrm D[\widetilde H f]
=\dot{\widetilde H}f+\widetilde H\dot f,\qquad
\mathrm D[\widetilde H^T v]
=\dot{\widetilde H}^{\,T}v+\widetilde H^T\dot v.
$$

The first term accounts for changes in the process; the second accounts for
changes in the input vector. The augmented rules above add source dependence.
[`Transition`](../api/transitions.md#drifft.Transition) follows the same
identities with a single transition in place of the whole history.

A pullback, or vector–Jacobian product (VJP), propagates an output cotangent
back to the inputs. For a scalar objective, that cotangent is the derivative
of the objective with respect to the action's output. The opposite action
provides the vector contribution, and the shared scalar pairing provides the
parameter contribution. JAX composes these rules with the rest of the
objective, including [sampling](hypergeometric.md#differentiation) and a
log-likelihood.

The history rules compose interval and boundary operations without forming
$\widetilde H$ or its derivative matrix. Numerical methods share these
mathematical identities. Their algorithms determine how accurately the
required actions are evaluated.

## Accuracy and limitations

[`Sequential`](../api/methods.md#drifft.Sequential) evaluates derivatives through
the represented transitions and supports higher derivatives where the backend
rules support them.

[`PolynomialKrylov`](../api/methods.md#drifft.PolynomialKrylov) and
[`RationalKrylov`](../api/methods.md#drifft.RationalKrylov) approximate the
mathematical action and its first derivatives. They do not differentiate the
adaptive projection basis. Higher-order Krylov operator sensitivities raise
explicitly.

Forward, backward, and derivative calculations can have different numerical
errors. Their pairings agree up to approximation error. A successful primal
calculation does not certify an accurate gradient, and the numerical error
estimates do not include backend error or provide rigorous error bounds.

A failed sensitivity or pullback raises even with `throw=False`; a failing
member under `vmap` can fail the batch. There is no automatic fallback.
Numerical diagnostics and status fields are not differentiation targets.
Zero inputs and zero source rates retain their mathematical derivatives.

## Checking gradients

Check the final objective and gradient as numerical resolution increases.
For Krylov methods, increase `dimension` and check sensitivity to `rtol` and
`atol`; check backend accuracy separately. A likelihood can amplify small
errors in probabilities. The [accuracy guide](../guide/numerics.md) describes
which settings control each approximation.

For small problems, compare values and derivatives against an independently
assembled dense binomial history. JVP/VJP pairings provide an additional
consistency check, but cannot detect an error shared by both calculations.

Finite differences are useful when the primal is sufficiently resolved.
An underresolved adaptive approximation can have a different derivative from
the mathematical action targeted by the custom rule, so disagreement alone
does not identify an autodiff bug. Validate against a more accurate reference
before interpreting that discrepancy.
