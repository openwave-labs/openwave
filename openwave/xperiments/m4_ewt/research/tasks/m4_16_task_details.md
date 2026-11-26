# M4.16 — Scale-Dependent Gravitational Coupling $\alpha_G(r)$

## Status

DONE (post-hoc)

## Criterion

`Running coupling` — coupling varies with scale (modeled onset around the core radius $r_e$)

## Objective

Demonstrate the purely geometric emergence of the baseline dimensionless gravitational coupling $\alpha_G$ from the Body-Centered Cubic (BCC) vacuum lattice chain ($A_\pi$, $\varepsilon_M$, $N_{\text{geom}}$, $K_{\text{WC}}$, $N_{\text{eff}}$) with scale invariance across the empirical anchors (`r_e`, `e`), and model its scale-dependent running $\alpha_G(r)$ and $G(r)$ across the soliton core scale $r \sim r_e$ along the $N_{\text{eff}} \to N_{\text{stat}}$ transition.

## Outcome

The baseline coupling $\alpha_G$ emerges consistently across three equivalent geometric paths without free empirical fitting parameters (the three are one expression rewritten, so their agreement checks the algebra), and scale invariance is established via the rigid ratio $r_e = 100\,r_\nu$ and anchor cancellation. Scale running across $r \sim r_e$ is modeled as a working ansatz using radial density profiles $N(r)$ interpolating from the diluted core ($N_{\text{eff}}$) to the unperturbed far-field lattice ($N_{\text{stat}}$) across monotonic profiles, EMC-wall variants around $r = r_e$, and regularized local metric/deficit profiles ($\eta_{\text{loc}}$); the transition direction is structurally forced by $\alpha_G \propto N^{-1/2}$, while profile shapes and parameters are set by construction pending full first-principles BCC soliton core dynamics. Numbers, algebraic decompositions, and scale profiles are in the findings note.

## Artifacts

- `research/scripts/m4_16_running_coupling.py`
- `research/scripts/m4_16_running_coupling_all_variants.png`
- `research/findings/m4_16_running_coupling.md`

## Reference

Enhanced EWT manuscript, version 5.4.4 (2026-11-24), § "The Geometric Identity of $G$: Derivation and Consistency". DOI: [10.5281/zenodo.24513845](https://doi.org/10.5281/zenodo.24513845).