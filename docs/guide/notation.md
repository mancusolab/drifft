# Mathematical notation

These conventions connect the probability model, numerical methods, and
sensitivity calculations. Chapters define their local quantities where used;
API parameter names and Python examples retain their code spelling. Use this
page as a reference: the [model](model.md) introduces the biological process,
then [computational tasks](../methods/overview.md) explains why the numerical
representations are needed.

## Population model

These symbols describe allele counts, biological parameters, and one-generation
transitions. The matrix orientation fixes how propagation and expectation relate.

| Concept | Symbol |
| --- | --- |
| Diploid effective population size | $N_e$ |
| Population gene-copy count | $M=2N_e$; $M_s,M_d$ for source and destination |
| Diploid sample size and sample gene-copy count | $m$; $n=2m$ |
| Source and destination allele counts | $i,j$ |
| Allele frequency | $x$, with $x_i=i/M_s$ |
| Offspring frequency after the full frequency map | $q_\theta(x)$; omit the subscript when parameters are implicit |
| Model parameter collection and its dimension | $\theta,n_\theta$ |
| Selection coefficients and dominance | $s_{\mathrm{het}},s_{\mathrm{hom}},h$ |
| Directional selection coefficient | $s=s_{\mathrm{hom}}$, with $s_{\mathrm{het}}=hs$; additive selection fixes $h=1/2$ |
| Multiplicative fitness ratio | $\rho$ (API parameter `r`) |
| Directional mutation rates | $\mu_{0\to1},\mu_{1\to0}$ |
| One-generation transition and its numerical approximation | $P,\widetilde P$ |
| Population mass column vector | $v_t$ |
| Observable or destination-value vector | $f$ |
| Generation index and interval duration | $t,g$ |
| Exact and represented interior transitions | $Q,\widetilde Q$ |
| Expected-visits vector | $u$ |
| Source mass vector | $b$ |

Rows of $P$ index source states; columns index destination states. Vectors
are columns unless transposed. The [model chapter](model.md) develops the
transition, propagation, and expected-visits equations.

The [fastDTWF paper](https://doi.org/10.1093/genetics/iyad168) uses the same
column-distribution and transpose convention, but names its transition matrix
$\mathbf M$ and its haploid population size $N$. Here those correspond to $P$
and the gene-copy count $M$, respectively. We retain $N_e$ for diploid size.

## Numerical representations

These symbols distinguish computational dimensions from biological sizes and
durations. The [computational overview](../methods/overview.md) introduces
the representations; each algorithm chapter defines its construction.

| Concept | Symbol |
| --- | --- |
| Prepared forward operator; input and output working states | $A;\ z,y$ |
| Represented forward population operator | $\widetilde P^T$ |
| Prepared exponent; reconstruction map and offset | $k;\ R,c$ |
| Working dimension and Krylov basis dimension | $d,m$ |
| Krylov basis and projected operator | $V_m$, $H_m$ |
| Jacobi coefficient transform and evaluation | $E,D$ |
| Reduced Jacobi expectation operator | $H_{\mathrm J}$ |
| Retained interior Jacobi dimension and assembly padding | $r,r_{\mathrm{pad}}$ |
| Polynomial degree and coefficient count | $L,n_{\mathrm{modes}}=L+1$ |
| Quadrature, sampling, and evaluation counts | $n_{\mathrm{quad}},n_{\mathrm{nodes}},n_{\mathrm{eval}}$ |
| Retained binomial stencil width | $W$ |
| Angular coordinate | $\vartheta$ |
| Centered Jacobi coordinate | $\xi=2q-1$ |
| FFT log-odds spacing | $\Delta_{\mathrm{logit}}$ |

Use $k$ for the prepared exponent and $g$ for physical generations. A tilde
marks a numerical approximation. $P$ and $\widetilde P$ act on population
observables; $\widetilde P^T$ propagates population mass. $A$ is the operator
in the coordinates used for the power calculation: $\widetilde P^T$ in
population coordinates, or $H_{\mathrm J}^T$ in reduced Jacobi coordinates.

Polynomial families retain the standard notation $T_j,U_j$ for Chebyshev,
$P_j^{(\alpha,\beta)}$ for Jacobi, and $C_j^{(\lambda)}$ for Gegenbauer;
$\phi_j$ denotes a normalized basis. These conventions follow
[NIST DLMF](https://dlmf.nist.gov/18.3). A polynomial approximation may be called $p_L$; it is not a population mass.

## Derivatives and local notation

A dot denotes a directional derivative; a bar denotes the cotangent of the
same named quantity. A cotangent weights changes in that quantity when
propagating derivatives backward from a scalar objective.

| Concept | Symbol |
| --- | --- |
| Model parameters and scalar objective | $\theta,\ell$ |
| Represented demographic expectation operator | $\widetilde H$ |
| Final mass contributed by singleton sources | $c_H$ |
| Augmented demographic forward operator | $\mathcal A$ |
| Directional changes in the history and its input | $\dot{\widetilde H},\dot v,\dot f$ |

The [autodiff chapter](../methods/autodiff.md) defines the demographic actions
and their derivatives. Local summation indices and auxiliary functions may
vary, but must be defined where they are used.
