# EWT / OpenWave Findings (M4.16)

## Finding 1 (M4.16): Scale-Dependent Gravitational Coupling $\alpha_G(r)$ and Geometric Emergence

## Status

DONE (post-hoc)

## Criterion

`Running coupling` — coupling varies with scale (modeled onset around the core radius $r_e$)

**Scope**: reproduces the baseline $\alpha_G$ from the M4.7 chain with $r_e = 100\,r_\nu$, and models what $\alpha_G(r)$ and $G(r)$ do along a transition from the diluted core ($N_{\text{eff}}$) to the unperturbed far-field lattice ($N_{\text{stat}}$) across candidate radial profiles and regularized local metric/deficit variants ($\eta_{\text{loc}}$). It does not derive the profile, the onset radius or the core value from first-principles BCC soliton dynamics: all tested radial profiles are working ansatze (modeled, not independently confirmed), and the factor 100 comes from wave-center counts $K_{\text{WC}} = 10$ and $1$ (manuscript v5.4.4, "Neutrino as the Geometric Anchor"), entered as `100.0` in the engine. No current observation separates $G(r)$ from a constant $G$ at $r \sim 10^{-15}$ m.

* **Script:** `openwave/xperiments/m4_ewt/research/scripts/m4_16_running_coupling.py`
* **Plot:** `openwave/xperiments/m4_ewt/research/scripts/m4_16_running_coupling_all_variants.png`

### 1. Geometric Origin and Equivalent Paths for $\alpha_G$

In the EWT M4 emergence chain, $\alpha_G$ is built from the Body-Centered Cubic (BCC) vacuum lattice geometry without free empirical fitting parameters:

* **Path 1 (Standard Metric Definition):**
  $$\alpha_G = \frac{G_{\text{EWT}}\, m_e^2}{\hbar_{\text{geom}}\, c} = \alpha_{\text{geom}} \left(\frac{G_{\text{EWT}}}{G_{\text{Base}}}\right), \qquad G_{\text{Base}} = \frac{c^2 r_e}{m_e}$$
* **Path 2 (Reduced Geometric Form via $\alpha_{\text{geom}}$):**
  $$\alpha_G = \frac{\alpha_{\text{geom}}}{A_\pi^4 \, N_{\text{geom}}^3 \, K_{\text{WC}} \sqrt{N}}$$
* **Path 3 (Pure BCC Lattice Geometry):**
  $$\alpha_G^{\text{geom}} = \frac{1}{A_\pi^5 \left(1 - \frac{\varepsilon_M}{A_\pi}\right) N_{\text{geom}}^3 \, K_{\text{WC}} \sqrt{N}}$$

Path 2 is Path 1 with $G_{\text{EWT}}/G_{\text{Base}}$ written out from the gravity sector, and Path 3 is Path 2 with $\alpha_{\text{geom}} = 1/(A_\pi - \varepsilon_M)$ substituted. They are one expression rewritten (they agree to $10^{-50}$ at 50 digits), so their agreement checks the algebra. Because $\alpha_G \propto N^{-1/2}$, increasing $N(r)$ from the diluted core ($N_{\text{eff}}$) to the unperturbed far-field lattice ($N_{\text{stat}}$) forces $\alpha_G(r)$ to decrease with radius.

Numerical decomposition from `m4_16_running_coupling.py`:

* $A_\pi = 4\pi^3 + \pi^2 + \pi \approx 137.036303775878$, giving $A_\pi^5 \approx 4.832570 \times 10^{10}$ (5D geometric ladder)
* $1 - \varepsilon_M / A_\pi \approx 0.999999697805$ (vacuum packing correction)
* $N_{\text{geom}}^3 \approx 4.723697 \times 10^8$ (cubic BCC lattice stiffness)
* $K_{\text{WC}} = 10$ (wave-center count)
* $N_{\text{eff}} \approx 6.247379 \times 10^{48}$ (diluted core regime, $r \to 0$) and $N_{\text{stat}} \approx 3.295973 \times 10^{52}$ (unperturbed far-field lattice regime, $r \gg r_e$), with $X_{\text{eff}} = N_{\text{stat}} / N_{\text{eff}} \approx 5275.768324$ (both densities are dimensionless length ratios involving $\lambda_l$, $r_\nu$, and Euler's number)
* Core-limit $\alpha_G(0) \approx 1.752633 \times 10^{-45}$ vs. CODATA 2022 $1.751810 \times 10^{-45}$ ($0.046976\%$ relative difference); far-field $N_{\text{stat}}$ asymptote $\alpha_G(\infty) \approx 2.412949 \times 10^{-47}$ and $G(\infty) \approx 9.193207 \times 10^{-13}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$

### 2. Scale Dependence Across $r_e$ and Profile Variants ($N_{\text{eff}} \to N_{\text{stat}}$)

Using the core-to-vacuum radial effective density profile across $r_e = 100\,r_\nu \approx 2.817935 \times 10^{-15}\ \mathrm{m}$ ($x = r / r_e$):

$$N(r) = N_{\text{eff}} + (N_{\text{stat}} - N_{\text{eff}})\,f(x), \qquad f(0) = 0,\; \lim_{x \to \infty} f(x) = 1$$

with $\alpha_G(r)$ evaluated from $N(r)$ and $G(r) = G_{\text{Base}}\,[\alpha_G(r)/\alpha_{\text{geom}}]$:

* Tested profile families in `m4_16_running_coupling.py`:
  - Monotonic profiles: `gaussian`, `exponential`, `rational_x2`, `woods_saxon`
  - EMC-wall profiles around $x = 1.0$: `wall_gaussian_base`, `wall_rational_base`, `wall_narrow_peak`, `wall_wide_shoulder`, `wall_dip_on_re`
  - Regularized local metric / deficit variants ($\eta_{\text{loc}}$): `eta_loc_p2`, `eta_loc_p1`, `eta_loc_with_wall`, via $\eta_{\text{loc}}(x) = 1 - \delta_{\text{core}}/(1 + x^p) + \text{wall\_term}(x)$ mapped to $f(x) \in [0, 1]$
* Representative values across variants (for $x = r / r_e$):
  - Near core ($x = 0.01$): most variants sit around $f(x) \approx 1.00 \times 10^{-4}$, $\alpha_G \approx 1.418114 \times 10^{-45}$, $G \approx 5.402939 \times 10^{-11}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$, while `exponential` ($f \approx 9.950 \times 10^{-3}$) and `eta_loc_p1` ($f \approx 9.901 \times 10^{-3}$) shift earlier
  - Around core boundary ($x \approx 1.01$): `rational_x2`, `woods_saxon`, and `eta_loc_p2` sit near $f(x) \approx 0.50$ ($\alpha_G \approx 3.40\text{–}3.42 \times 10^{-47}$, $G \approx 1.30 \times 10^{-12}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$); `wall_gaussian_base`, `wall_narrow_peak`, `wall_rational_base`, `wall_wide_shoulder`, and `eta_loc_with_wall` advance faster ($f(x) \approx 0.80\text{–}0.99$), whereas `wall_dip_on_re` slows to $f(x) \approx 0.1538$ ($\alpha_G \approx 6.150 \times 10^{-47}$)
  - Far-field / large $x$ ($x = 100.0$): variants converge to $N \to N_{\text{stat}}$, $\alpha_G \to 2.412949 \times 10^{-47}$, $G \to 9.193207 \times 10^{-13}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$ (`eta_loc_p1` retains a gentler tail with $f(100) \approx 0.9901$, $\alpha_G \approx 2.425 \times 10^{-47}$)

### Scale Invariance and the Absence of Implicit Anchor Dependence

A careful reader might note that the effective density $N_{\text{eff}}$ entering the factored geometric expression for $\alpha_G$ is related to $(r_\nu / \lambda_l)^3$, and since $\lambda_l$ is derived from $r_e$, there appears to be an implicit dependence on the empirical anchors $r_e$ and $e$. However, this is not a dynamical dependence, but a manifestation of *scale invariance*. The anchors $r_e$ and $e$ merely set the absolute unit scale (in metres) and do not dictate the geometric ratios. Specifically, the fundamental geometric link $r_e = 100\,r_\nu$ enforces a rigid ratio, meaning the factor $(r_e / r_\nu)^6$ is a fixed geometric constant ($10^{12}$), while $e$ cancels identically in the geometric fixed-point equation for $r_\nu$. Consequently, if one were to artificially scale the anchor $r_e$ by any factor, the derived Planck length $\lambda_l$ would scale by the exact same factor, leaving the dimensionless density ratios and the coupling $\alpha_G$ completely invariant. The gravitational coupling is therefore a pure, dimensionless geometric ratio of the BCC lattice, entirely independent of the numerical values of the empirical anchors. This paragraph holds on the route of this note, where $r_e = 100\,r_\nu$ is not an anchor, the elementary charge is the only one, and $\alpha_G$ does not depend on it (scaling the charge by 7 leaves $\alpha_G$ unchanged); on the engine's route, with the CODATA $r_e$ held, $\alpha_G \propto (r_e/r_\nu)^6$ exactly, so it moves as the charge to the power $-6$.

**Note on profile form:**

The specific profile shape and auxiliary parameters ($p$, $\delta_{\text{core}}$, wall amplitude/width) are secondary to the algebraic structure and limiting behavior of the equations; all radial profiles $f(x)$ and local metric/deficit variants $\eta_{\text{loc}}(x)$ serve as modeled working ansatze to test scale running across $r_e$, while the core-to-vacuum transition direction ($N_{\text{eff}} \to N_{\text{stat}}$) and BCC geometric factors govern the running behavior pending full first-principles soliton core dynamics.

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