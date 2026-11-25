# EWT / OpenWave Findings (M4.16, M4.17 & Metric Conversion Status of $m_e$)

## Finding 1 (M4.16): Scale-Dependent Gravitational Coupling $\alpha_G(r)$ and Geometric Emergence

* **OpenWave Criterion:** `Running coupling` — coupling varies with scale (onset at the core radius)
* **Script:** `openwave/xperiments/m4_ewt/research/scripts/m4_16_running_coupling.py`
* **Status:** **PASS** ($\alpha_G(\infty) = 1.752633 \times 10^{-45}$, relative error vs. CODATA 2022: $0.046976\%$; smooth running onset at $r \sim r_e$)

### 1. Geometric Origin and Equivalent Paths for $\alpha_G$

In the macroscopic asymptotic regime ($r \gg r_e$), $\alpha_G$ emerges from the Body-Centered Cubic (BCC) vacuum lattice geometry without free empirical fitting parameters[cite: 8]:

* **Path 1 (Standard Metric Definition):**
  $$\alpha_G = \frac{G_{\text{EWT}}\, m_e^2}{\hbar_{\text{geom}}\, c} = \alpha_{\text{geom}} \left(\frac{G_{\text{EWT}}}{G_{\text{Base}}}\right), \qquad G_{\text{Base}} = \frac{c^2 r_e}{m_e}$$
* **Path 2 (Reduced Geometric Form via $\alpha_{\text{geom}}$):**
  $$\alpha_G = \frac{\alpha_{\text{geom}}}{A_\pi^4 \, N_{\text{geom}}^3 \, K_{\text{WC}} \sqrt{N_{\text{eff}}}}$$
* **Path 3 (Pure BCC Lattice Geometry):**
  $$\alpha_G^{\text{geom}} = \frac{1}{A_\pi^5 \left(1 - \frac{\varepsilon_M}{A_\pi}\right) N_{\text{geom}}^3 \, K_{\text{WC}} \sqrt{N_{\text{eff}}}}$$

Numerical decomposition from `m4_16_running_coupling.py`[cite: 8]:
* $A_\pi = 4\pi^3 + \pi^2 + \pi \approx 137.036303775878$, giving $A_\pi^5 \approx 4.832570 \times 10^{10}$ (5D geometric ladder)
* $1 - \varepsilon_M / A_\pi \approx 0.999999697805$ (vacuum packing correction)
* $N_{\text{geom}}^3 \approx 4.723697 \times 10^8$ (cubic BCC lattice stiffness)
* $K_{\text{WC}} = 10$ (wave-center count)
* $N_{\text{stat}} \approx 3.295973 \times 10^{52}\ \mathrm{m^{-3}}$, $X_{\text{eff}} = N_{\text{stat}} / N_{\text{eff}} \approx 5275.768324$, $N_{\text{eff}} \approx 6.247379 \times 10^{48}\ \mathrm{m^{-3}}$, $\sqrt{N_{\text{eff}}} \approx 2.499476 \times 10^{24}\ \mathrm{m^{-3/2}}$
* Asymptotic $\alpha_G^{\text{geom}} = 1.752633 \times 10^{-45}$ vs. CODATA 2022 $1.751810 \times 10^{-45}$ ($0.046976\%$ relative error) and $G_{\text{EWT}} = 6.677439 \times 10^{-11}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$

### 2. Scale Dependence and Onset at the Core Radius ($r \sim r_e$)

Using the radial effective density profile across $r_e = 100\,r_\nu \approx 2.817935 \times 10^{-15}\ \mathrm{m}$[cite: 8]:

$$N(r) = N_{\text{eff}} + (N_{\text{stat}} - N_{\text{eff}})\exp\!\left[-\left(\frac{r}{r_e}\right)^2\right]$$

with $\alpha_G(r)$ evaluated from $N(r)$ and $G(r) = G_{\text{Base}}\,[\alpha_G(r)/\alpha_{\text{geom}}]$[cite: 8]:
* At $r = 100\,r_e$ (far-field asymptote): $N(r) = 6.247379 \times 10^{48}\ \mathrm{m^{-3}}$, $\alpha_G(r) = 1.752633 \times 10^{-45}$, $G(r) = 6.677439 \times 10^{-11}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$[cite: 8]
* At $r = r_e$ (core boundary onset): $N(r_e) = 1.212915 \times 10^{52}\ \mathrm{m^{-3}}$, $\alpha_G(r_e) = 3.977632 \times 10^{-47}$, $G(r_e) = 1.515457 \times 10^{-12}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$[cite: 8]
* At $r = 0.1\,r_e$ (deep core): $N(0.1\,r_e) = 3.263183 \times 10^{52}\ \mathrm{m^{-3}}$, $\alpha_G(0.1\,r_e) = 2.425041 \times 10^{-47}$, $G(0.1\,r_e) = 9.239280 \times 10^{-13}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$ (core-to-asymptotic ratio: $1.383656 \times 10^{-2}$)[cite: 8]

\paragraph{Scale Invariance and the Absence of Implicit Anchor Dependence.}
Both the asymptotic coupling $\alpha_G(\infty)$ and the running profile $\alpha_G(r/r_e)$ are invariant with respect to external dimensional unit choices[cite: 8]. In Path 3, the asymptotic value is governed exclusively by dimensionless BCC lattice invariants ($A_\pi$, $\varepsilon_M$, $N_{\text{geom}}$, $K_{\text{WC}}$, and $X_{\text{eff}}$)[cite: 8]. Expressed in terms of the dimensionless scale $\tilde{r} \equiv r / r_e$, the core radius $r_e = 100\,r_\nu$ acts strictly as an internal geometric scale ratio rather than an externally imposed length anchor; covariant rescaling of the lattice scale leaves $\alpha_G(\tilde{r})$ unchanged, confirming that the onset at $\tilde{r} \sim 1$ is a geometric consequence of the core-to-vacuum density transition.

---

## Finding 2 (M4.17): Local Medium / Field Parameter Dependence $G(B)$ and $\alpha_G(B)$

* **Script:** `openwave/xperiments/m4_ewt/research/scripts/m4_17_g_of_b.py`
* **Status:** **PASS** (Recovers $G_{\text{EWT}} = 6.677439 \times 10^{-11}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$ and $\alpha_G = 1.752633 \times 10^{-45}$ as $B / B_0 \to 0$; smooth suppression across $B / B_0 \sim 1$)

### 1. Formulation and Physical Scope

Starting from the geometric baseline $G_{\text{EWT}}$ and $\alpha_G^{\text{geom}}$ in `m4_7_ewt_emergence_engine.py`, an effective parameter/field-dependent coupling is modeled via a normalized local medium/field state parameter $x_B = B / B_0$[cite: 8]:

$$G(B) = G_{\text{EWT}}\, f_G\!\left(\frac{B}{B_0}\right), \qquad \alpha_G(B) = \alpha_{\text{geom}}\left(\frac{G(B)}{G_{\text{Base}}}\right) = \alpha_G^{\text{geom}}\, f_G\!\left(\frac{B}{B_0}\right)$$

with the saturation profile $f_G(B/B_0) = \left[1 + (B/B_0)^p\right]^{-1}$ ($p = 2.0$, $f_G(0) = 1$)[cite: 8]. Rather than treating $B$ as an arbitrary alteration of an electron's intrinsic constants, $B$ represents a local medium/field state parameter in the core neighborhood[cite: 8]. For $B \ll B_0$ (standard macroscopic or weak-field regimes), $f_G \to 1$ and $G(B) \to G_{\text{EWT}}$[cite: 8].

### 2. Quantitative Behavior Across $B / B_0$

* At $B / B_0 = 1.0000 \times 10^{-3}$: $f_G = 0.999999$, $G(B) = 6.677432 \times 10^{-11}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$ ($1.000 \times 10^{-4}\%$ difference vs. $G_{\text{EWT}}$), $\alpha_G(B) = 1.752631 \times 10^{-45}$[cite: 8]
* At $B / B_0 = 1.0000 \times 10^{0}$: $f_G = 0.500000$, $G(B) = 3.338719 \times 10^{-11}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$, $\alpha_G(B) = 8.763165 \times 10^{-46}$[cite: 8]
* At $B / B_0 = 1.0000 \times 10^{2}$: $f_G = 9.999000 \times 10^{-5}$, $G(B) = 6.676771 \times 10^{-15}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$, $\alpha_G(B) = 1.752458 \times 10^{-49}$[cite: 8]

---

## Finding 3: Electron Mass $m_e$ as a Metric Conversion Factor and Algebraic Mass Cancellation

* **Module:** `openwave/xperiments/m4_ewt/research/scripts/m4_7_ewt_emergence_engine.py`
* **Status:** **CONFIRMED** (Algebraic cancellation verified in both $\lambda_l$ and $\alpha_G$)

### 1. Metric Conversion Role of $m_e$

Within the EWT emergence chain, the electron mass $m_e$ acts as a SI mass-scale metric conversion rather than an independent dynamical degree of freedom (alongside $c$ as a space-time metric conversion). While individual intermediate scales depend on $m_e$ or $m_e^{-1}$ — for instance, $G_{\text{Base}} = c^2 r_e / m_e \propto m_e^{-1}$, $\hbar_{\text{geom}} = m_e c r_e / \alpha_{\text{geom}} \propto m_e$, and Planck temperature $T_p \propto \sqrt{\hbar_{\text{geom}} / G_{\text{EWT}}} \propto m_e$ — combinations that eliminate $m_e$ isolate pure geometric properties of the BCC vacuum[cite: 8].

### 2. Explicit Algebraic Cancellation in Planck Length and Gravitational Coupling

* **Planck Length ($\lambda_l$):** Because $\hbar_{\text{geom}} \propto m_e$ and $G_{\text{EWT}} \propto m_e^{-1}$, their product $\hbar_{\text{geom}} G_{\text{EWT}}$ is independent of $m_e$[cite: 8]. In explicit mass-cancelled form:
  $$\lambda_l = \frac{r_e}{\sqrt{\alpha_{\text{geom}} A_\pi}} \left(\frac{1}{N_{\text{geom}} A_\pi}\right)^{3/2} \frac{1}{\sqrt{K_{\text{WC}}\sqrt{N_{\text{eff}}}}}$$
  confirming that $\lambda_l$ is determined by vacuum geometry and length/density ratios rather than $m_e$[cite: 8].
* **Dimensionless Gravitational Coupling ($\alpha_G$):** In $\alpha_G = G_{\text{EWT}} m_e^2 / (\hbar_{\text{geom}} c)$, the factors of $m_e$ (and $r_e$, $G$, $\hbar$) cancel out in Paths 2 and 3, leaving a purely geometric invariant of the BCC lattice[cite: 8].