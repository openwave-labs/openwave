# EWT / OpenWave Findings (M4.16)

## Finding 1 (M4.16): Scale-Dependent Gravitational Coupling $\alpha_G(r)$ and Geometric Emergence

## Status

DONE (post-hoc)

## Criterion

`Running coupling` — coupling varies with scale (onset at the core radius)

**Scope**: reproduces the asymptotic $\alpha_G$ from the M4.7 chain with $r_e = 100\,r_\nu$, and shows what $\alpha_G(r)$ and $G(r)$ do under a chosen radial profile. It does not derive the profile, the onset radius or the core value: the profile's width is $r_e$ and its value at $r = 0$ is $N_{\text{stat}}$, so the onset at $r_e$ and the core ratio (which tends to $1/\sqrt{X_{\text{eff}}}$ at $r = 0$) are set by construction. Per the author (review of [PR #755](https://github.com/openwave-labs/openwave/pull/755)), the factor 100 is derived from the wave-center counts $K_{\text{WC}} = 10$ and $1$ (manuscript v5.4.4, "Neutrino as the Geometric Anchor"); the engine enters it as the literal `100.0`. No current observation separates $G(r)$ from a constant $G$ at $r \sim 10^{-15}$ m.

* **Script:** `openwave/xperiments/m4_ewt/research/scripts/m4_16_running_coupling.py`

### 1. Geometric Origin and Equivalent Paths for $\alpha_G$

In the macroscopic asymptotic regime ($r \gg r_e$), $\alpha_G$ emerges from the Body-Centered Cubic (BCC) vacuum lattice geometry without free empirical fitting parameters:

* **Path 1 (Standard Metric Definition):**
  $$\alpha_G = \frac{G_{\text{EWT}}\, m_e^2}{\hbar_{\text{geom}}\, c} = \alpha_{\text{geom}} \left(\frac{G_{\text{EWT}}}{G_{\text{Base}}}\right), \qquad G_{\text{Base}} = \frac{c^2 r_e}{m_e}$$
* **Path 2 (Reduced Geometric Form via $\alpha_{\text{geom}}$):**
  $$\alpha_G = \frac{\alpha_{\text{geom}}}{A_\pi^4 \, N_{\text{geom}}^3 \, K_{\text{WC}} \sqrt{N_{\text{eff}}}}$$
* **Path 3 (Pure BCC Lattice Geometry):**
  $$\alpha_G^{\text{geom}} = \frac{1}{A_\pi^5 \left(1 - \frac{\varepsilon_M}{A_\pi}\right) N_{\text{geom}}^3 \, K_{\text{WC}} \sqrt{N_{\text{eff}}}}$$

Path 2 is Path 1 with $G_{\text{EWT}}/G_{\text{Base}}$ written out from the gravity sector, and Path 3 is Path 2 with $\alpha_{\text{geom}} = 1/(A_\pi - \varepsilon_M)$ substituted. They are one expression rewritten (they agree to $10^{-50}$ at 50 digits), so their agreement checks the algebra.

Numerical decomposition from `m4_16_running_coupling.py`:

* $A_\pi = 4\pi^3 + \pi^2 + \pi \approx 137.036303775878$, giving $A_\pi^5 \approx 4.832570 \times 10^{10}$ (5D geometric ladder)
* $1 - \varepsilon_M / A_\pi \approx 0.999999697805$ (vacuum packing correction)
* $N_{\text{geom}}^3 \approx 4.723697 \times 10^8$ (cubic BCC lattice stiffness)
* $K_{\text{WC}} = 10$ (wave-center count)
* $N_{\text{stat}} \approx 3.295973 \times 10^{52}$, $X_{\text{eff}} = N_{\text{stat}} / N_{\text{eff}} \approx 5275.768324$, $N_{\text{eff}} \approx 6.247379 \times 10^{48}$, $\sqrt{N_{\text{eff}}} \approx 2.499476 \times 10^{24}$ (both densities are dimensionless: `N_nu_statutory = (r_nu / (2 lambda_l EULER))^3` is a ratio of lengths, with Euler's number)
* Asymptotic $\alpha_G^{\text{geom}} = 1.752633 \times 10^{-45}$ vs. CODATA 2022 $1.751810 \times 10^{-45}$ ($0.046976\%$ relative error) and $G_{\text{EWT}} = 6.677439 \times 10^{-11}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$

Since $\alpha_G = \alpha_{\text{geom}}\,G_{\text{EWT}}/G_{\text{Base}}$ is $G_{\text{EWT}}$ times fixed factors, the $0.046976\%$ is essentially the residual of $G_{\text{EWT}}$ on this route in another form ($0.046954\%$ against the engine's `G_CODATA`), the quantity [M4.10.2](../tasks/m4_10_2_task_details.md) recorded on the engine's route ($0.048169\%$ against the same constant). The two routes give two values of $\alpha_G$: this note sets $r_e = 100\,r_\nu$ and gets $1.752633 \times 10^{-45}$; the engine's own `main` uses the CODATA $r_e$ and gets $1.752651 \times 10^{-45}$. They differ by $1.04 \times 10^{-5}$, which is $(R_E/(100\,r_\nu))^6 - 1$.

### 2. Scale Dependence and Onset at the Core Radius ($r \sim r_e$)

Using the radial effective density profile across $r_e = 100\,r_\nu \approx 2.817935 \times 10^{-15}\ \mathrm{m}$:

$$N(r) = N_{\text{eff}} + (N_{\text{stat}} - N_{\text{eff}})\exp\!\left[-\left(\frac{r}{r_e}\right)^2\right]$$

with $\alpha_G(r)$ evaluated from $N(r)$ and $G(r) = G_{\text{Base}}\,[\alpha_G(r)/\alpha_{\text{geom}}]$:

* At $r = 100\,r_e$ (far-field asymptote): $N(r) = 6.247379 \times 10^{48}$, $\alpha_G(r) = 1.752633 \times 10^{-45}$, $G(r) = 6.677439 \times 10^{-11}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$
* At $r = r_e$ (core boundary onset): $N(r_e) = 1.212915 \times 10^{52}$, $\alpha_G(r_e) = 3.977632 \times 10^{-47}$, $G(r_e) = 1.515457 \times 10^{-12}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$
* At $r = 0.1\,r_e$ (deep core): $N(0.1\,r_e) = 3.263183 \times 10^{52}$, $\alpha_G(0.1\,r_e) = 2.425041 \times 10^{-47}$, $G(0.1\,r_e) = 9.239280 \times 10^{-13}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$ (core-to-asymptotic ratio: $1.383656 \times 10^{-2}$)

### Scale Invariance and the Absence of Implicit Anchor Dependence

A careful reader might note that the effective density $N_{\text{eff}}$ entering the factored geometric expression for $\alpha_G$ is related to $(r_\nu / \lambda_l)^3$, and since $\lambda_l$ is derived from $r_e$, there appears to be an implicit dependence on the empirical anchors $r_e$ and $e$. However, this is not a dynamical dependence, but a manifestation of *scale invariance*. The anchors $r_e$ and $e$ merely set the absolute unit scale (in metres) and do not dictate the geometric ratios. Specifically, the fundamental geometric link $r_e = 100\,r_\nu$ enforces a rigid ratio, meaning the factor $(r_e / r_\nu)^6$ is a fixed geometric constant ($10^{12}$), while $e$ cancels identically in the geometric fixed-point equation for $r_\nu$. Consequently, if one were to artificially scale the anchor $r_e$ by any factor, the derived Planck length $\lambda_l$ would scale by the exact same factor, leaving the dimensionless density ratios and the coupling $\alpha_G$ completely invariant. The gravitational coupling is therefore a pure, dimensionless geometric ratio of the BCC lattice, entirely independent of the numerical values of the empirical anchors. This paragraph holds on the route of this note, where $r_e = 100\,r_\nu$ is not an anchor, the elementary charge is the only one, and $\alpha_G$ does not depend on it (scaling the charge by 7 leaves $\alpha_G$ unchanged); on the engine's route, with the CODATA $r_e$ held, $\alpha_G \propto (r_e/r_\nu)^6$ exactly, so it moves as the charge to the power $-6$.

**Note on profile form:**

The specific profile shape is secondary to the algebraic structure and limiting behavior of the equations; the radial profile $N(r)$ with $\exp[-(r/r_e)^2]$ serves as a working model ansatz to test scale running across $r_e$, while the asymptotic $\alpha_G$ value and core boundary scale $r_e$ follow from the BCC geometry.

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
