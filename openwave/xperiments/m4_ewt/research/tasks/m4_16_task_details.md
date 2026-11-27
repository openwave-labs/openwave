# Task: M4.16 Boundary Onset

## Status

DONE (post-hoc)

## Criterion

`Running coupling`: coupling varies with scale (onset at the core radius)

## Objective

Model boundary onset of gravitational coupling across $x = r / r_e$:

* Keep the interior ($x < 1$) separate from boundary onset ($x \ge 1$).
* Compute gravitational invariants from the emergence engine.
* Evaluate onset variants and generate the radial plot.

## Outcome

No local $G(r)$ or $\alpha_G(r)$ is defined inside $r_e$: the interior ($x < 1$) is the push-out zone, excluded by assumption. From the boundary outward, $G(x) = G_{\text{EWT}}\,\Phi(x)$ and $\alpha_G(x) = \alpha_G^{\text{EWT}}\,\Phi(x)$, with four working onset variants for $\Phi$ (`step_boundary`, `emc_wall_step`, `monotonic_shell`, `emc_wall_peak`), none derived from BCC lattice mechanics; at the default parameters `emc_wall_step` equals `step_boundary`. All four converge to the engine invariants in the far field, computed on the engine's route (CODATA $r_e$): $G_{\text{EWT}} = 6.6775199755 \times 10^{-11}$ (0.048169% vs CODATA) and $\alpha_G^{\text{EWT}} = 1.7526511766 \times 10^{-45}$ (0.048018%). The surface-projection reading of $\sqrt{N_{\nu,\text{eff}}}$ is an interpretation, not a derived step. Before [PR #758](https://github.com/openwave-labs/openwave/pull/758), M4.16 used $r_e = 100\,r_\nu$, three algebraically equivalent paths and a radial density profile $N(r)$; that version landed with [PR #755](https://github.com/openwave-labs/openwave/pull/755). Numbers and the radial table are in the findings note.

## Artifacts

* `research/scripts/m4_16_running_coupling.py`
* `research/findings/m4_16_running_coupling.md`
* `research/plots/m4_16_running_coupling.png` (written by the script, not committed)

## Reference

Enhanced EWT manuscript, version 5.4.4 (2026-11-24), § "The Geometric Identity of $G$: Derivation and Consistency". DOI: [10.5281/zenodo.24513845](https://doi.org/10.5281/zenodo.24513845).
