# Conventions, and a worklist

This file is a conventions extract plus a list of questions. It is deliberately incomplete: it
carries the definitions and none of the results. Nothing below states a value you are being asked
for, and no answer appears anywhere in this file.

Work from these conventions only. Everything the worklist needs is fixed here, including the
Clebsch-Gordan phase, the Majorana normalisation and the general-spin forms of the transform, so a
disagreement should be traceable to content rather than to something the extract left open. If you
nonetheless find a question underdetermined by what is written here, say so and state the reading
you took rather than choosing one silently.

Items 0(a) and 0(b) exist so that a mismatch can be localised before anything else is judged.
Answer those first.

---

## 2. Setup

### 2.1 The state space and the Majorana dictionary

Let $`V_j = \mathrm{Sym}^{2j}\mathbb{C}^2`$ be the irreducible $`\mathrm{SU}(2)`$ representation of
spin $`j`$ and dimension $`2j+1`$, with orthonormal weight basis $`v_m`$, $`-j \le m \le j`$. The
state space throughout is $`V_3`$, of dimension 7. Writing a state $`u = \sum_m u_m v_m`$ as a
binary sextic $`F`$ in the usual way and taking the six roots of $`F`$ on the Riemann sphere gives
the *Majorana constellation* [Maj] of $`u`$, a multiset of six points determined by the ray $`[u]`$ and
carried by rotations. Statements about constellations below are statements about rays.

### 2.2 Time reversal

Time reversal is the antiunitary $`\Theta`$ on $`V_3`$ acting on coefficients by

```math
\Theta\!\left(\sum_m u_m v_m\right) \;=\; \sum_m (-1)^m\, \overline{u_m}\, v_{-m} ,
```

written in shorthand as $`\Theta v_m = (-1)^m v_{-m}`$ with $`\Theta`$ understood to be antilinear. The
weight reversal and the relative $`(-1)^m`$ pattern are forced; only an overall phase is
conventional. Antilinear intertwiners of $`V_3`$ with
itself are $`\mathrm{Hom}_{\mathrm{SU}(2)}(\overline{V_3}, V_3)`$, which is one-dimensional by
Schur since $`V_3`$ is irreducible and self-dual, so $`\Theta`$ is unique up to a complex scalar. To
identify it, write $`\Theta v_m = c_m v_{-m}`$ with $`\Theta`$ antilinear, and impose $`\Theta J_+ = -J_- \Theta`$. Since
$`J_+ v_m = \alpha_m v_{m+1}`$ with $`\alpha_m = \sqrt{j(j+1) - m(m+1)}`$, and
$`J_- v_{-m} = \alpha'_m v_{-m-1}`$ with $`\alpha'_m = \sqrt{j(j+1) - (-m)(-m-1)}`$, the two
coefficients are equal because $`(-m)(-m-1) = m(m+1)`$, and this reads $`\alpha_m c_{m+1} = -\alpha_m c_m`$, so $`c_{m+1} = -c_m`$ and
$`c_m = c\,(-1)^m`$; antiunitarity fixes $`\lvert c \rvert = 1`$. The global constant is a genuine
convention and the $`m`$-dependence is not, which matters below because it is the $`m`$-dependence
that supplies the alternating sign in the operator asked about below. One computes $`\Theta^2 = (-1)^{2j}`$, so $`\Theta^2 = +1`$ here and $`\Theta^2 = -1`$ at half-integer spin.

Two labels used throughout should be read as representation theory and nothing more. *Spin 3* names
the irreducible $`\mathrm{SU}(2)`$ representation $`V_3`$ and the harmonic level it sits at; it is
not a claim that anything described here has physical spin 3. And $`\Theta`$ is the standard
antiunitary intertwiner of that representation with its conjugate. The equation studied below is
elliptic and stationary, with no time in it, so nothing here establishes a physical time-reversal
symmetry of a dynamics; $`\Theta`$ earns its name from its algebra, not from an evolution it
commutes with.

A ray is *time-reversal invariant* when $`\Theta u = \lambda u`$ for some $`\lambda`$ of modulus one, and
the set of such $`u`$ is $`U(1)\cdot\mathrm{Fix}(\Theta)`$. The condition is projective: for
$`a \in \mathrm{Fix}(\Theta)`$ and any phase, $`u = e^{it}a`$ has $`\Theta u = e^{-2it}u`$, which is in general
neither $`+u`$ nor $`-u`$. In constellation language a ray is time-reversal invariant exactly when
its six Majorana points are antipodally symmetric *as a multiset*, coincidences included.

### 2.3 One transform, and three things it is applied to

For $`0 \le J \le 6`$ let $`[\,\cdot \otimes \cdot\,]_J`$ denote the projection of
$`V_3 \otimes V_3`$ onto its spin-$`J`$ summand. It will be convenient to have a single transform
on matrices. For a $`7 \times 7`$ matrix $`P`$ define

```math
\mathcal{M}_K(P)_N \;=\; \sum_{n+n' = N} \langle 3\,n;\, 3\,n' \mid K\,N \rangle \, (-1)^{n'} P_{n,\,-n'} ,
\qquad 0 \le K \le 6 .
```

This is the state-multipole, or statistical-tensor, expansion of a density matrix in irreducible
tensor operators, standard since [Fa] and used in exactly this form to read multipoles off a
Majorana constellation [RK]. It is written out here because three different arguments are fed to it
below and the paper turns on keeping them apart.

Three specialisations occur below and they must be kept apart.

The **holomorphic square** $`B_J(a) = [a \otimes a]_J`$ takes both arguments to be the same state.
Exchanging two identical slots multiplies the spin-$`J`$ summand of $`V_3 \otimes V_3`$ by
$`(-1)^{3+3-J} = (-1)^J`$, so $`B_J`$ vanishes identically for odd $`J`$.

The **density multipole** $`\rho_K(u) = [u \otimes \Theta u]_K`$ is sesquilinear, linear in $`u`$ and
antilinear in it through $`\Theta u`$. It equals the transform above at the state's own projector,

```math
\rho_K(u) \;=\; \mathcal{M}_K\!\left(u u^{\dagger}\right) ,
```

and it does not vanish for odd $`K`$ in general.

The **right multipole** is the same transform at a different argument, $`R_K = \mathcal{M}_K(P)`$
with $`P`$ the isotypic projector of Section 2.4. Same map, different matrix; the two multiply
rather than merge, and one of the questions below turns on that.

The factorisation the worklist asks about uses this one $`\mathcal{M}_K`$ on both sides, which is what makes
the two multipoles multiply: the time-reversal phase sits inside $`\mathcal{M}_K`$ itself, and
$`\rho_K(u) = \mathcal{M}_K(u u^{\dagger})`$ holds because $`(\Theta u)_{n'} = (-1)^{n'}\overline{u_{-n'}}`$
is exactly the factor the definition carries.

Three notational cautions, all earned. $`\rho_K`$ with a subscript is always a density multipole
and never a representation of $`\Gamma`$; representations are written $`\sigma`$ throughout. And the
index $`K`$ on a multipole is a spin, while the $`J`$ on $`B_J`$ is the same kind of index; no
quantity in this paper is indexed by a binary form degree except where that is said explicitly.
Third, spins do not all live on one axis either: a multipole of the density is indexed by its own
rank, a channel of the nonlinearity by the spin of its target, and these label different
decompositions. One question below works at output spin 8, another at density rank 6 and output spin 3;
the 6 and the 8 are not comparable as indices.


### 2.4 The quotient, its sectors, and the projector convention

Identify $`S^3`$ with $`\mathrm{SU}(2)`$ carrying the round metric. Let
$`\Gamma \subset \mathrm{SU}(2)`$ be the finite subgroup generated, as unit quaternions
$`w + xi + yj + zk`$, by

```math
q_1 \;=\; \tfrac{1}{2}\left(1 + i + j + k\right) ,
\qquad
q_2 \;=\; \tfrac{1}{2}\left(\varphi + \varphi^{-1} i + j\right) ,
\qquad
\varphi \;=\; \tfrac{1+\sqrt{5}}{2} .
```

$`\Gamma`$ acts on $`S^3`$ by right translation and $`X = S^3/\Gamma`$ is the quotient.
**The group is not named here and none of its representation theory is supplied.** Everything
the worklist needs about $`\Gamma`$ is to be derived from these two generators. Item 0(c) is a
sanity check on the generators themselves; do it before anything else, because every later
item inherits whatever they actually generate.

A finite-dimensional unitary representation $`\sigma`$ of $`\Gamma`$ determines a flat bundle
on $`X`$, whose sections are the functions
$`\psi \colon \mathrm{SU}(2) \to \mathbb{C}^{\dim\sigma}`$, written as rows, satisfying
$`\psi(gh) = \psi(g)\,\sigma(h)`$ for $`h \in \Gamma`$. An intertwiner
$`\eta \in \mathrm{Hom}_{\Gamma}(\sigma, V_j)`$ is correspondingly a map with
$`D^j(h)\,\eta = \eta\,\sigma(h)`$, and by Peter-Weyl the sections at level $`\ell = 2j`$ are
$`V_j \otimes \mathrm{Hom}_{\Gamma}(\sigma,\, V_j)`$, realised as

```math
\psi_a(g) \;=\; \sum_{m,n} u_m\, D^j_{mn}(g)\, \eta_{na} ,
```

so the left index is free and the right index carries the sector. The convention is fixed here
because dual-looking Peter-Weyl formulas are both defensible and only one of them matches the
factorisation the worklist asks about. Since $`\sigma`$ is unitary,
$`\lvert\psi(gh)\rvert^2 = \lvert\psi(g)\rvert^2`$, so the density really is a function on
$`X`$.

A *block state* at level 6 is one for which $`j = 3`$, so its left index runs over the
$`V_3`$ of Section 2.1 and its right index over $`\mathrm{Hom}_{\Gamma}(\sigma, V_3)`$. Level 6
is the scope of the main selection questions below; item 16 deliberately ranges over lower
levels and says so. What fixes it is the choice of object: these
are spin-3 states, so the left index is $`V_3`$ and the level is $`2 \cdot 3`$.

**The projector normalisation.** Suppose $`\sigma`$ occurs in $`V_3`$ with multiplicity one, so
that $`\mathrm{Hom}_{\Gamma}(\sigma, V_3)`$ is one-dimensional. For any nonzero intertwiner
$`\eta`$ in it, Schur gives $`\eta^{\dagger}\eta = c\, I_\sigma`$; rescale so that $`c = 1`$, so
that $`P = \eta\eta^{\dagger}`$ is the orthogonal projector onto the $`\sigma`$-summand of
$`V_3`$. Without that normalisation $`P`$ is a multiple of the projector and every quantity
built from it carries an undetermined constant, so it is fixed here once. Whether the
multiplicity is in fact one, and what the summands are, is part of what the worklist asks: it
is a convention about how $`\eta`$ is scaled, not a supplied fact about $`\Gamma`$.

## Conventions this extract fixes

**Clebsch-Gordan phase.** All coefficients $`\langle j_1 m_1; j_2 m_2 \mid J M \rangle`$ are taken
in the Condon-Shortley convention, so they are real and
$`\langle j\,j; j\,{-j} \mid 2j\,0 \rangle > 0`$. This is the phase the worklist assumes; a
different one changes signs throughout and item 0(b) is where that would show.

**Time reversal at general spin.** Section 2.2 fixes $`\Theta`$ on $`V_3`$ up to a global phase.
Where the worklist ranges over other spins, read it as

```math
\Theta_j v_m \;=\; \varepsilon_j\,(-1)^{\,j-m}\, v_{-m} ,
\qquad \lvert\varepsilon_j\rvert = 1 ,
```

which is defined at integer and half-integer $`j`$ alike because $`j - m`$ is an integer in both
cases. Only the $`m`$-dependence is forced; $`\varepsilon_j`$ is conventional.

**The transform at general spin.** Section 2.3 writes $`\mathcal{M}_K`$ on $`7 \times 7`$ matrices,
with a sign $`(-1)^{n'}`$ that is real only when $`n'`$ is an integer. At general spin take

```math
\mathcal{M}^{(j)}_K(P)_N \;=\; \sum_{n+n' = N}
\langle j\,n;\, j\,n' \mid K\,N \rangle \;
\varepsilon_j\,(-1)^{\,j-n'}\, P_{n,\,-n'} ,
```

whose exponent is an integer at half-integer $`j`$ as well, and which reduces to Section 2.3's
transform at $`j = 3`$ with $`\varepsilon_3 = -1`$. Write $`\rho^{(j)}_K`$ and $`M^{(j)}_K`$ for the
corresponding density multipole and cubic self-map.

**A classical fact about binary forms.** For binary forms $`f, g`$ write
$`J(f,g) = f_X g_Y - f_Y g_X`$ for the Jacobian. Over a field of characteristic zero, if
$`f`$ and $`g`$ are nonzero of the same degree $`n > 0`$ and $`J(f,g) = 0`$, then $`f`$ and
$`g`$ are proportional. At unequal degrees $`m = \deg f`$ and $`n = \deg g`$ the same
computation gives the weaker $`f^{\,n} \propto g^{\,m}`$. This is classical and is supplied
as an input, like the Clebsch-Gordan phase above; it is not something the worklist asks you to
prove.

**The Majorana correspondence, concretely.** For $`u = \sum_m u_m v_m \in V_3`$ put

```math
F_u(z) \;=\; \sum_{m=-3}^{3} (-1)^{\,3-m}\sqrt{\binom{6}{3+m}}\; u_m\, z^{\,3-m} .
```

A finite root $`r`$ gives the point with polar angle $`\theta = 2\arctan\lvert r\rvert`$ and
azimuth $`\varphi = \arg r`$; each degree the polynomial falls short of 6 contributes one point at
$`\theta = \pi`$. That fixes absolute coordinates, not merely a shape up to rotation.

---

## Definitions used by the worklist

These are definitions, not results.

For $`0 \le K \le 6`$, and with $`\rho_K`$ the density multipole of Section 2.3,

```math
A_K(u)\,v \;=\; \left[\, \rho_K(u) \otimes v \,\right]_3 ,
\qquad
M_K(u) \;=\; A_K(u)\, u .
```

With $`P`$ the isotypic projector of Section 2.4, $`d = \operatorname{rank} P`$, and
$`R_K = \mathcal{M}_K(P)`$,

```math
\widehat{r}_K([u]) \;=\; \frac{\lVert\rho_K(u)\rVert^2}{\lVert u\rVert^4} .
```

The block state built from $`u`$ and the intertwiner $`\eta`$ is written $`\psi`$. All integrals
below are taken against the Riemannian measure on $`X`$ normalised so that $`\int_X 1 = 1`$,
together with $`\eta^\dagger\eta = I_\sigma`$; without both, the functionals carry an overall scale
and $`Q_d`$ is determined only up to it. The two functionals are

```math
A(u) = \int_X \lvert\psi\rvert^4 , \qquad B(u) = \int_X \lvert\psi\rvert^2 ,
\qquad Q_d([u]) = \frac{A(u)}{B(u)^2} ,
```

On the unit sphere $`Q_d`$ is an expansion in the density multipoles,

```math
Q_d([u]) \;=\; \sum_K w_K\, \widehat{r}_K([u]) ,
```

and $`w_K`$ means the coefficient appearing there, with
$`N = \binom{12}{6}\big/ w_6`$. **A closed form for $`w_K`$ in terms of $`R_K(P)`$ is
deliberately not supplied here; deriving it is item 13.**

Finally $`\mathcal{N}`$ denotes the projection of $`\lvert\psi\rvert^2\psi`$ back onto the
level-6 eigenspace, read as a map on the fibre coordinate $`u`$.

---


## Worklist

Give an exact value for each. Rationals as rationals, radicals as radicals, no decimals except
where a decimal is the only sensible form. Say which items you could not complete rather than
estimating them.

**0. Anchors. Do these first and report them before the rest.**

&nbsp;&nbsp;(a) $`\Theta v_2`$, with the global phase of Section 2.2 taken to be $`+1`$.

&nbsp;&nbsp;(b) $`\langle 3\,3;\, 3\,{-3} \mid 6\,0 \rangle`$.

&nbsp;&nbsp;(c) **The generator packet.** Confirm $`\lVert q_1 \rVert = \lVert q_2 \rVert = 1`$;
compute the order $`\lvert\Gamma\rvert`$ of the group they generate; and report whether
$`\Gamma`$ equals its own derived subgroup. Report all three before continuing. If any of them
comes out other than you expect, stop and say so rather than proceeding: every quotient item
below inherits this group.

**1.** Decompose $`\mathrm{Sym}^3 V_3`$ into $`\mathrm{SU}(2)`$ irreducibles, with
multiplicities.

**2.** $`\dim \mathrm{Hom}_{\mathrm{SU}(2)}\!\left(\mathrm{Sym}^2 V_3 \otimes \overline{V_3},\,
V_J\right)`$ at $`J = 8`$ and at $`J = 3`$.

**3.** $`M_0(u)`$. Show whether it is a multiple of $`u`$, and if so give the constant.

**4.** $`M_6`$ on the weight basis. Is it diagonal there? If so give the diagonal entries as a
sequence indexed by $`m = -3, \dots, 3`$, and describe any structure in them worth
recording.

**5.** At $`u = v_3`$, give $`B_2(u)`$ and $`\rho_2(u)`$ **separately**, as defined in Section
2.3. State whether they agree, and if not, which of the two the definitions in this extract
call for wherever a density multipole is wanted.

**6.** $`\binom{12}{6}\,\lVert\rho_6(u)\rVert^2`$ at the four unit states $`v_3`$, $`v_0`$,
$`(v_2 + v_{-2})/\sqrt{2}`$, $`(v_3 + v_{-3})/\sqrt{2}`$. Separately, establish for each of the
four whether the ray is a critical point of $`\widehat{r}_6`$ on $`\mathbb{P}(V_3)`$, and say by
what argument.

**7.** The full row $`\lVert\rho_K(u)\rVert^2`$ for $`K = 0, \dots, 6`$ at
$`u = (v_2 + v_{-2})/\sqrt{2}`$.

**8.** On the line $`u = \cos t\, v_2 + \sin t\, v_{-3}`$: $`\lVert\rho_6\rVert^2`$ as a function
of $`s = \sin^2 t \in [0,1]`$, **written out in full**; all of its stationary points in the
interior of that interval with the value at each; and separately the values at the two
endpoints.

**9.** On $`u = v_3 + z\, v_0 + v_{-3}`$ with $`z = x + iy`$: the ray invariant
$`\widehat{r}_6([u]) = \lVert\rho_6\rVert^2/\lVert u\rVert^4`$ as a function of $`x, y`$, and its
complete critical set with the value at each point, established by elimination rather than by
evaluating candidates. Separately, and do not conflate the two: the critical set of the
unnormalised $`\lVert\rho_6\rVert^2`$ read as a function on the same affine slice. The two sets
need not agree, since criticality of a homogeneous function on an affine slice is not the same
question as criticality of its projectivisation. Name which of the two you regard as the
statement about rays, and say why.
That parameterisation is a chart on a projective line; say whether it covers the whole line, and
if not, treat what it misses.

**10.** From the generators of Section 2.4 and nothing else: $`\dim (V_K)^{\Gamma}`$ for
$`K = 0, \dots, 6`$. State the method. If you also compute values beyond $`K = 6`$, report them
separately and say so; the range $`K \le 6`$ is what the later items use.

**11.** From the same generators: decompose $`V_3`$ restricted to $`\Gamma`$ into irreducibles,
giving the number of constituents, each multiplicity, and each dimension. Say how you
established the multiplicities rather than reading them off dimensions.

**12.** For each summand found in item 11, build the isotypic projector $`P`$ from the group
itself, and report $`\lVert \mathcal{M}_K(P)\rVert^2`$ for $`K = 0, \dots, 6`$. Build the
projector rather than positing it, and say which construction you used.

**13.** First derive a closed form for $`w_K`$ in terms of $`\lVert R_K(P)\rVert^2`$ and
$`d = \operatorname{rank} P`$, from Peter-Weyl and the normalisations fixed above. State the
derivation, not only the result; the formula is not supplied anywhere in this extract. Then,
using items 11 and 12, give $`w_0`$, the ratio $`w_6/w_0`$ as a function of $`d`$, and $`N`$ in
each sector.

**14.** Write $`\mathcal{N}(u)`$ as a combination of the maps $`M_K`$ of the definitions above.
Give the coefficient of each $`M_K`$, in each sector, including the ones that come out zero.
Where you claim a coefficient is nonzero, say how you established it, since that is a separate
question from whether $`M_K`$ is a multiple of $`M_0`$.

**15.** For each of the four states in item 6, and for the stationary rays found in items 8 and
9, give the Majorana constellation in the convention fixed above: the six points as
coordinates, and the shape. Give each point's latitude and azimuth explicitly; where several
points share a latitude, list their azimuths, and say what that spacing implies about the
configuration.

**16.** For a level-$`\ell = 2j`$ block state, which ranks $`K`$ can $`\rho^{(j)}_K`$ carry?
Combine that with item 12's method and say, for each level $`\ell = 1, \dots, 6`$ and each
summand occurring there, which ranks have $`\lVert\mathcal{M}^{(j)}_K(P)\rVert \neq 0`$. State
what that implies about whether the projected cubic can depend on the direction of $`u`$ at that
level.

**17.** Write $`Q_d([u])`$ as a function of $`\widehat{r}_6([u])`$ in each sector. Are the
functions' critical sets on $`\mathbb{P}(V_3)`$ the same or different? Establish the answer from
the form of those functions, not by enumerating the critical set.

**18.** Suppose $`\mathcal{N}(u) = \beta u`$ at some ray. Normalise so that $`B(u) = 1`$ and give
$`\beta`$ in terms of quantities already defined. Then give the differences between the
sectors' values of $`\beta`$ at the same ray. Be explicit about which normalisation each
quantity in your answer is taken with respect to, since more than one is in play.

**19.** Let $`\mathcal{C}(u)`$ denote the spin-8 component of the trilinear data
$`(u, \Theta u, u)`$, that is the image of $`u \otimes \Theta u \otimes u`$ under the
equivariant projection onto $`V_8`$. Characterise **exactly** the set of nonzero $`u`$ with
$`\mathcal{C}(u) = 0`$. An exact argument is wanted, not a sampled family: give the
characterisation and the proof. That half is insensitive to how $`\mathcal{C}`$ is normalised.

**20.** Not part of this stage.

**21.** The construction above builds the interaction from $`\lvert\psi\rvert^2\psi`$, whose
fibre content is $`u \otimes \Theta u`$. Consider instead the local quantity built from
$`\psi\psi^{T}`$ rather than $`\psi\psi^{\dagger}`$. First establish what it takes for a
quartic built from $`\psi\psi^{T}`$ to be $`\Gamma`$-invariant, and derive which of the
transforms of Section 2.3 its left factor is; do not assume the answer. Call the resulting
level-6 projected maps $`N_J(u)`$. Which $`J`$ survive the same filter that item 10 computes?
Give the span the resulting interaction lands in and its dimension, and say whether it is the
same span as the one carried by the maps with nonzero coefficient in item 14. Do **not** attempt
to determine any finer structure of that second span; say only whether the two spans coincide.

## What to return

A single table: item, exact value, and how you got it in one line. Then, separately, any item where
your convention had to differ from the ones fixed above, with the difference stated. Do not attempt
to guess what the answers are supposed to be, and do not adjust a result to make it look canonical.
