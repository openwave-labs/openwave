# M4.13: q = 1 from Spherical Symmetry and the n = 1 Consistency Test

## Status
DONE (post-hoc)

## Criterion
`Gravity: local metric phenomena`

## Objective
Address the M4.9 assumptions requested by the reviewer: the pair
potential exponent $n = 1$ and the axial compression factor $q = 1$.
Show that $q = 1$ follows from $n = 1$ under spherical symmetry, and
give a three-sector consistency test that selects $n = 1$ from the rest
of the model.

## Method

Executable derivation in a jupytext notebook. It shows symbolically
that for a spherically symmetric EMC density profile

$$\eta(r) = 1 - \frac{A}{r^m}$$

the displacement field $u_r(r) = A/r^m$ produces strain components

$$\varepsilon_{rr} = -\frac{mA}{r^{m+1}}, \qquad \varepsilon_{\theta\theta} = \frac{A}{r^{m+1}}$$

with ratio $\varepsilon_{rr}/\varepsilon_{\theta\theta} = -m$. The
magnitudes are equal only when $m = 1$.

For the EMC model the profile exponent is identified with the pair
potential exponent, $m = n$ where $V(r) \sim 1/r^n$. At $m = 1$ the two
strain magnitudes are equal at every radius, the deformation is
directionally uniform on each shell, and the effective propagation
problem reduces to 1D along any radial direction. That is $q = 1$.

The notebook also runs a three-sector consistency test for $n = 1$:

| Sector | Requires | Fails for |
|---|---|---|
| Strain equality (this notebook) | $m = 1$ | every other $m$ |
| Newtonian force law (M4.10/M4.12) | $m = 1$ | every other $m$ |
| Metric profile form (M4.3–M4.5) | $m = 1$ | every other $m$ |

The force test is numerical: the overlap integral $I(R)$ is computed
for $n = 1$ and $n = 2$, and the extracted force exponent is compared
to $-2$. Only $n = 1$ gives $-2$.

The metric test is algebraic: the Schwarzschild-like form
$(1 - r_s/r)^{-1/2}$ used by M4.3–M4.5 is reproduced by
$(1 - A/r^m)^{-1/2}$ only for $m = 1$; the logarithmic Shapiro-delay
structure is lost for $m \neq 1$.

## Result

Symbolic derivation:

- $\varepsilon_{rr}/\varepsilon_{\theta\theta} = -m$ for general $m$.
- At $m = 1$: ratio $= -1$, equal magnitudes, directional uniformity.
- At $m = 2$: ratio $= -2$, magnitudes differ by factor 2.

Consistency test:

- Strain equality requires $n = 1$; every other value gives unequal
  magnitudes.
- Numerical force test: $n = 1$ gives force exponent $-2$; $n = 2$
  gives a different exponent.
- Metric profile: only $n = 1$ reproduces the Schwarzschild-like form
  used by M4.3–M4.5.

## Interpretation

**What this establishes.** $q = 1$ is not an independent postulate.
It follows from $m = 1$ (the Newtonian density profile) under
spherical symmetry. The reduction from 3D to the 1D chain of M4.9 is
justified, not assumed.

**The n = 1 consistency argument.** $n = 1$ is the EMC pair potential
exponent documented in the manuscript. The consistency test shows
that every other value of $n$ breaks at least one already-verified
sector: the force law (M4.10/M4.12), the metric profile (M4.3–M4.5),
or the strain equality derived here. So $n = 1$ is not a free
parameter; it is uniquely selected by the rest of the chain. This is
stronger than "assumed" but weaker than a dynamical derivation.

**Scope.** This artifact addresses the M4.9 assumptions by derivation
and consistency. The pair potential exponent $n = 1$ is a model input
documented in the manuscript; the consistency test shows it is uniquely
selected by the rest of the chain. No dynamical derivation is claimed,
and none is required for this static criterion. The M4.9 propagation
exponents themselves (the form $\beta = n/(2q)$) are outside the scope
of this artifact, but the notebook justifies the 1D reduction that
M4.9 assumes.

**Net effect on the row.** After this artifact, the M4.9 assumption
list shrinks from $\{n = 1, q = 1\}$ to $\{n = 1\}$ alone, and
$n = 1$ is now supported by a three-sector consistency test rather
than being a standalone postulate.


## Artifacts

- `research/scripts/m4_13_symmetry_derivation.py` — jupytext percent source.
  To convert and run as a notebook:
  `jupytext --to notebook m4_13_symmetry_derivation.py` then open the `.ipynb`.
  To run as a plain script (Markdown cells ignored):
  `python m4_13_symmetry_derivation.py`.
- `research/findings/m4_13_local_metric_geometry.md`

## Reference

Enhanced EWT manuscript, version 5.0.0:
[DOI: 10.5281/zenodo.22540635](https://doi.org/10.5281/zenodo.22540635)

Relevant sections:

- "Bridging the Vector Displacement to the Scalar Refractive Index"
- "Asymptotic Continuous Limit and Schwarzschild Equivalence"
- "Newtonian Force from Interacting EMC Deficits"