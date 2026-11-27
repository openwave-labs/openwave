# M4.16: Boundary Onset and Scale Emergence of Gravitational Coupling

**Script:** `m4_ewt/research/scripts/m4_16_running_coupling.py`  
**Plot:** `m4_ewt/research/plots/m4_16_running_coupling.png` (written by the script, not committed)  
**Engine dependency:** `m4_7_ewt_emergence_engine.py`  
**Status:** DONE (post-hoc)  
**Criterion:** `Running coupling`: coupling varies with scale (onset at the core radius)

**Scope**: evaluates $G$ and $\alpha_G$ for $r \ge r_e$ as $G_{\text{EWT}}$ and $\alpha_G^{\text{EWT}}$ times a chosen onset factor $\Phi(x)$, on the engine's route (CODATA $r_e$, as in M4.10.2). It does not derive $\Phi$, the onset radius or the interior: the interior is excluded by assumption and the four shapes are ansatzes. Before PR #758, M4.16 used $r_e = 100\,r_\nu$ and reported $\alpha_G = 1.752633 \times 10^{-45}$. No current observation separates $G(r)$ from a constant $G$ at $r \sim 10^{-15}$ m.

---

## 1. Physical Framework & Core Principle

In Enhanced Energy Wave Theory (EWT), gravity is a **boundary-emergent phenomenon** rather than a local interior field $G(r)$ defined inside the soliton core ($r < r_e$):

* **Dominance of Outward Push-Out in the Interior ($x = r / r_e < 1$):** Inside the core radius $r_e$, outward Electromagnetic Core (EMC) push-out driven by $K_{WC} = 10$ interacting wave centers dominates the local medium dynamics, balanced against restoring medium equilibrium. Because outward push-out governs the interior—and asking for a local interior $G(r)$ inside the core is analogous to asking how a uniform statutory vacuum background $N_{\text{stat}}$ self-gravitates—no local gravitational coupling $G(r)$ or $\alpha_G(r)$ exists for $x < 1$ (represented as `NaN` on the radial grid).
* **Boundary Emergence ($x \ge 1$) and the Origin of $\sqrt{N_{\nu,\text{eff}}}$:** Gravity emerges at the spherical soliton interface $r = r_e$ through a surface projection of the cumulative internal volume deficit (a reading, not a derived step: see section 2). Because the effective volume deficit $N_{\nu,\text{eff}} = N_{\nu,\text{statutory}} / X_{\text{eff}}$ is a volumetric count distributed across the 3D core while gravitational emergence acts as a 2D boundary/surface projection on the shell of radius $r_e$ across $K_{WC}$ wave centers, the effective linear/surface projection factor enters as $\left(K_{WC}\sqrt{N_{\nu,\text{eff}}}\right)^{-1}$ in the gravitational constant:
  $$G_{\text{EWT}} = \left(\frac{G_{\text{Base}}}{A_\pi}\right)\left(\frac{1}{N_{\text{geom}} A_\pi}\right)^3 \left(\frac{1}{K_{WC}\sqrt{N_{\nu,\text{eff}}}}\right)$$
  and in the dimensionless gravitational coupling:
  $$\alpha_G^{\text{EWT}} = \frac{\alpha_{\text{geom}}}{A_\pi^4 N_{\text{geom}}^3 K_{WC}\sqrt{N_{\nu,\text{eff}}}}$$

---

## 2. Epistemic Classification (Engine Invariants vs. Working Ansatzes)

| Component | Classification | Origin / Status |
| :--- | :--- | :--- |
| $A_\pi$, $N_{\text{geom}}$, $\alpha_{\text{geom}}$, $X_{\text{eff}}$, $N_{\nu,\text{statutory}}$, $N_{\nu,\text{eff}}$, $G_{\text{EWT}}$, $\alpha_G^{\text{EWT}}$ | **Engine-derived invariants** | Computed directly via `m4_7_ewt_emergence_engine.py` from BCC lattice geometry and electron/neutrino scale relations. |
| Interior exclusion ($x < 1 \to \text{NaN}$) | **Structural model boundary** | Reflects outward EMC push-out dominance inside $r_e$; no local interior gravitational field $G(r)$ is defined. |
| `step_boundary` ($\Phi_{\text{onset}}(x) = 1$ for $x \ge 1$) | **Working ansatz** | Ideal sharp surface onset at $r = r_e$; **not derived** from BCC lattice mechanics. |
| `emc_wall_step` ($\Phi_{\text{onset}}(x) = w_{\text{jump}}$ for $x \ge 1$) | **Working hypothesis** | Models a Degraded EMC Wall producing an abrupt step jump at the boundary $x = 1$; **not derived** from BCC lattice equations. |
| `monotonic_shell` ($\Phi_{\text{onset}}(x) = 1 - e^{-K_{WC}(x-1)}$) | **Working ansatz** | Phenomenological monotonic closure across $x \ge 1$ using $K_{WC}$ as the dimensionless inverse shell thickness; **not derived** from BCC lattice dynamics. |
| `emc_wall_peak` ($\Phi_{\text{onset}}(x) = 1 - e^{-K_{WC}(x-1)} + \eta_{\text{wall}} K_{WC}(x-1)e^{-K_{WC}(x-1)}$) | **Working ansatz** | Phenomenological shell profile with a localized compression bump ($\eta_{\text{wall}} = 0.12$) near $x \sim 1$; **not derived** from BCC lattice dynamics. |
| Surface-projection reading of $\sqrt{N_{\nu,\text{eff}}}$ (section 1) | **Interpretation, not derived** | The factor $\left(K_{WC}\sqrt{N_{\nu,\text{eff}}}\right)^{-1}$ is the engine's; no step here derives the exponent $1/2$ from a projection onto the shell at $r_e$. |

---

## 3. Reference Numerical Values (Verified against `m4_16_running_coupling.py`)

* $K_{WC} = 10$
* $X_{\text{eff}} = 5275.7683235656$
* $N_{\nu,\text{statutory}} = 3.2959040398 \times 10^{52}$
* $N_{\nu,\text{eff}} = 6.2472493819 \times 10^{48}$
* $G_{\text{EWT}} = 6.6775199755 \times 10^{-11}\ \text{m}^3\,\text{kg}^{-1}\,\text{s}^{-2}$ (relative difference vs. CODATA $6.674305 \times 10^{-11}$: $0.048169\%$)
* $\alpha_G^{\text{EWT}} = 1.7526511766 \times 10^{-45}$ (relative difference vs. CODATA $1.751810 \times 10^{-45}$: $0.048018\%$), computed from $\alpha_{\text{geom}}$ as in section 1; $G_{\text{EWT}}\,m_e^2/(\hbar c)$ with CODATA values is larger by a relative $1.92 \times 10^{-6}$

---

## 4. Radial Evaluation Summary ($x = r / r_e$)

| Variant | $x = 0.5$ | $x = 1.0$ | $x = 1.1$ | $x = 1.5$ | $x = 10.0$ | $x = 100.0$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `step_boundary` | `push-out` | $6.678 \times 10^{-11}$ | $6.678 \times 10^{-11}$ | $6.678 \times 10^{-11}$ | $6.678 \times 10^{-11}$ | $6.678 \times 10^{-11}$ |
| `emc_wall_step` | `push-out` | $6.678 \times 10^{-11}$ | $6.678 \times 10^{-11}$ | $6.678 \times 10^{-11}$ | $6.678 \times 10^{-11}$ | $6.678 \times 10^{-11}$ |
| `monotonic_shell` | `push-out` | $0.000 \times 10^{0}$ | $4.221 \times 10^{-11}$ | $6.633 \times 10^{-11}$ | $6.678 \times 10^{-11}$ | $6.678 \times 10^{-11}$ |
| `emc_wall_peak` | `push-out` | $0.000 \times 10^{0}$ | $4.516 \times 10^{-11}$ | $6.660 \times 10^{-11}$ | $6.678 \times 10^{-11}$ | $6.678 \times 10^{-11}$ |

---

## 5. Key Takeaways

1. **Interior Push-Out Dominance ($r < r_e$):** Outward EMC push-out from $K_{WC} = 10$ wave centers dominates the core interior, so no local gravitational coupling $G(r)$ is defined inside $r_e$; gravity exists only from the boundary outward ($x \ge 1$).
2. **Surface Projection Origin of $\sqrt{N_{\nu,\text{eff}}}$:** The square root in $\left(K_{WC}\sqrt{N_{\nu,\text{eff}}}\right)^{-1}$ reflects the geometric projection of the cumulative internal volume deficit onto the spherical boundary surface at $r = r_e$, rather than a bulk local density law. This reading is an interpretation, not a derived step (section 2).
3. **Asymptotic Consistency:** Regardless of whether boundary onset is modeled as an ideal step (`step_boundary`), an abrupt Degraded EMC Wall jump (`emc_wall_step`), or a smooth exponential shell closure (`monotonic_shell`, `emc_wall_peak`), all working onset profiles converge cleanly to $G_{\text{EWT}}$ and $\alpha_G^{\text{EWT}}$ in the far field ($x \gg 1$) without artificial clamping.
