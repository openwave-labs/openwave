# EWT / OpenWave Findings (M4.17)

## Finding 2 (M4.17): Local Medium / Field Parameter Dependence $G(B)$ and $\alpha_G(B)$

## Status

DONE (post-hoc)

## Criterion

`Parametric / field dependence` — effective coupling $G(B)$ and $\alpha_G(B)$ recover the baseline EWT limit for weak local fields ($B \ll B_0$) and exhibit smooth modulation across $B \sim B_0$. No `MODELS.md` criterion.

**Scope**: evaluates a chosen saturation ansatz $f_G$ in normalized units ($B_0 = 1$, $p = 2$) on top of the M4.7 baseline. The $B \to 0$ limit holds by construction ($f_G(0) = 1$). $B$ is not yet identified with a physical field: per the author (review of [PR #755](https://github.com/openwave-labs/openwave/pull/755)), it is a generic local medium/field state parameter, and its identity and the exponent $p$ remain open.

* **Script:** `openwave/xperiments/m4_ewt/research/scripts/m4_17_g_of_b.py`

### 1. Formulation and Physical Scope

Starting from the geometric baseline $G_{\text{EWT}}$ and $\alpha_G^{\text{geom}}$ in `m4_7_ewt_emergence_engine.py`, an effective parameter/field-dependent coupling is modeled via a normalized local medium/field state parameter $x_B = B / B_0$:

$$G(B) = G_{\text{EWT}}\, f_G\!\left(\frac{B}{B_0}\right), \qquad \alpha_G(B) = \alpha_{\text{geom}}\left(\frac{G(B)}{G_{\text{Base}}}\right) = \alpha_G^{\text{geom}}\, f_G\!\left(\frac{B}{B_0}\right)$$

with the saturation profile $f_G(B/B_0) = \left[1 + (B/B_0)^p\right]^{-1}$ ($p = 2.0$, $f_G(0) = 1$). Rather than treating $B$ as an arbitrary alteration of an electron's intrinsic constants, $B$ represents a local medium/field state parameter in the core neighborhood. For $B \ll B_0$ (standard macroscopic or weak-field regimes), $f_G \to 1$ and $G(B) \to G_{\text{EWT}}$.

### 2. Quantitative Behavior Across $B / B_0$

* At $B / B_0 = 1.0000 \times 10^{-3}$: $f_G = 0.999999$, $G(B) = 6.677432 \times 10^{-11}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$ ($1.000 \times 10^{-4}\%$ difference vs. $G_{\text{EWT}}$), $\alpha_G(B) = 1.752631 \times 10^{-45}$
* At $B / B_0 = 1.0000 \times 10^{0}$: $f_G = 0.500000$, $G(B) = 3.338719 \times 10^{-11}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$, $\alpha_G(B) = 8.763165 \times 10^{-46}$
* At $B / B_0 = 1.0000 \times 10^{2}$: $f_G = 9.999000 \times 10^{-5}$, $G(B) = 6.676771 \times 10^{-15}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$, $\alpha_G(B) = 1.752458 \times 10^{-49}$

**Note on profile form:**

The specific profile shape is secondary to the algebraic structure and limiting behavior of the equations; the saturation profile $f_G(B/B_0) = [1 + (B/B_0)^p]^{-1}$ ($p = 2.0$) acts as a phenomenological test ansatz preserving the $B \to 0$ limit ($G_{\text{EWT}}$) rather than a first-principles derivation from BCC lattice dynamics.

---

## Reference

Enhanced EWT manuscript, version 5.4.4 (2026-11-24).

Section:

The Geometric Identity of $G$: Derivation and Consistency:

* Derivation of $G$ from Electron Scaling (The $\hbar$-Independent Base)
* The Base Gravitational Scaling ($\mathbf{G}_{\text{Base}}$)
* The Gravitational Identity Flow: From Electromagnetic Base to Surface Transition
* The Gravitational Coupling Constant
* Formalization of Soliton Geometric Density $\rho(r)$
* Formal Definition of the Operator $\mathcal{U}$: Mapping Geometric Parameters to $G$
* Geometric Emergence of the Planck Length
* The Electron Mass Cancellation in the Planck Length

DOI: [10.5281/zenodo.24513845](https://doi.org/10.5281/zenodo.24513845).
