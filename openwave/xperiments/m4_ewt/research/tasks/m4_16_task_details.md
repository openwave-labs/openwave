# M4.16 — Scale-Dependent Gravitational Coupling $\alpha_G(r)$

## Status

DONE (post-hoc)

## Criterion

`Running coupling` — coupling varies with scale (onset at the core radius)

## Objective

Demonstrate the purely geometric emergence of the asymptotic dimensionless gravitational coupling $\alpha_G$ from the Body-Centered Cubic (BCC) vacuum lattice chain ($A_\pi$, $\varepsilon_M$, $N_{\text{geom}}$, $K_{\text{WC}}$, $N_{\text{eff}}$) with scale invariance across the empirical anchors (`r_e`, `e`), and model its scale-dependent running $\alpha_G(r)$ and $G(r)$ with onset across the soliton core boundary $r \sim r_e$.

## Outcome

The asymptotic coupling $\alpha_G$ emerges consistently across three equivalent geometric paths without free empirical fitting parameters (the three are one expression rewritten, so their agreement checks the algebra), and scale invariance is established via the rigid ratio $r_e = 100\,r_\nu$ and anchor cancellation. Scale running with onset at $r \sim r_e$ is modeled using a working Gaussian radial density profile ansatz $N(r)$ interpolating between $N_{\text{stat}}$ in the deep core and $N_{\text{eff}}$ in the macroscopic far-field regime; the onset at $r_e$ and the core value $N_{\text{stat}}$ are set by the profile's construction. Numbers, algebraic decompositions, and scale profiles are in the findings note.

## Artifacts

- `research/scripts/m4_16_running_coupling.py`
- `research/findings/m4_16_running_coupling.md`

## Reference

Enhanced EWT manuscript, version 5.4.4 (2026-11-24), § "The Geometric Identity of $G$: Derivation and Consistency". DOI: [10.5281/zenodo.24513845](https://doi.org/10.5281/zenodo.24513845).
