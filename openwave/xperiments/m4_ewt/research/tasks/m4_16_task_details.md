# Task: M4.16 Boundary Onset

Model boundary onset of gravitational coupling across $x = r / r_e$:

* Keep the interior ($x < 1$) separate from boundary onset ($x \ge 1$).
* Compute gravitational invariants from the emergence engine.
* Evaluate onset variants and generate the radial plot.e baseline coupling $\alpha_G$ emerges consistently across three equivalent geometric paths without free empirical fitting parameters (the three are one expression rewritten, so their agreement checks the algebra), and scale invariance is established via the rigid ratio $r_e = 100\,r_\nu$ and anchor cancellation. Scale running across $r \sim r_e$ is modeled as a working ansatz using radial density profiles $N(r)$ interpolating from the diluted core ($N_{\text{eff}}$) to the unperturbed far-field lattice ($N_{\text{stat}}$) across monotonic profiles, EMC-wall variants around $r = r_e$, and regularized local metric/deficit profiles ($\eta_{\text{loc}}$); the transition direction is structurally forced by $\alpha_G \propto N^{-1/2}$, while profile shapes and parameters are set by construction pending full first-principles BCC soliton core dynamics. Numbers, algebraic decompositions, and scale profiles are in the findings note.

## Artifacts

- `research/scripts/m4_16_running_coupling.py`


## Reference

Enhanced EWT manuscript, version 5.4.4 (2026-11-24), § "The Geometric Identity of $G$: Derivation and Consistency". DOI: [10.5281/zenodo.24513845](https://doi.org/10.5281/zenodo.24513845).