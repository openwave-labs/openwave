# M7 / HydroBoros - Implementation Plan

  ![hydroboros_icon](../images/hydroboros_icon_small.jpg)

> **Purpose.** Build a rigorous **full 3D PDE simulation** of the electron as a
> self-linked toroidal vortex, blending the **hydrodynamics school** (Marc Fleury's toroidal
> electromagnetic electron + the Beltrami / force-free field tradition) with the **Ouroboros
> model** (Paul Werbos's coupled two-vector-field chaoiton, M6). HydroBoros = Hydrodynamics +
> Ouroboros, also evoking the mythological **Hydra**, the water snake.
>
> The work runs headless in
> `research/sandbox_v1..vN/` (Taichi GPU lattice, energy minimizer, matplotlib diagnostics),
> converges to a canonical form, then graduates into the production `medium.py` + `engine1-4` +
> `_launcher.py` for visual rendering, exactly the path M5 followed.
> A primary deliverable is the **HydroBoros (M7) column** of the repo-root
> [`MODELS.md`](../../../../MODELS.md) coverage matrix, every cell backed by a runnable in-platform
> script, the same bar M5 meets (see § 7).
> Rigor standard: [`../../m5_liquid_crystal/research/11a_vortex_loop.md`](../../m5_liquid_crystal/research/11a_vortex_loop.md) (M5.11).
> Theory sources: [`../theory/`](../theory/). Findings as we execute: `research/sandbox_v*/`.

---

## 0. The two parents (and why the blend is non-trivial)

| Model | Substrate | How it is solved today | Charge origin | Key limit |
| --- | --- | --- | --- | --- |
| **Fleury** (hydrodynamics school), [arXiv:2510.22384](https://arxiv.org/abs/2510.22384) | EM field `E, B` confined to a torus | closed-form **analytic** ansatz + Heaviside mask | `ρ = ∇·E` (geometric / divergence charge) | mask unphysical at boundary; energy lands at `0.795 m_e c²`; **no dynamics, no PDE** |
| **Werbos Ouroboros** (M6), [`0d_canonical.md`](../../m6_ouroboros/research/0d_canonical.md) | **double vector field** `(A_μ, J_μ)` | **reduced 1D ODE / BVP** (cylindrical `(α,β)`; spherical `l=1`) | `Q_CS` = Chern-Simons **linking** number | never run as a full **3D** field; electron is only a radial profile (`H/Q = 1.6969`) |

The honest gap both parents share: **neither has ever been evolved as a full 3D nonlinear PDE on
a lattice.** Fleury is analytic; M6 is a 1D radial reduction. That is exactly the gap M5.11 closed
for the tensor model (Taichi-AD lattice + FIRE / L-BFGS minimizer, reproduce Faber's electron from
the relaxed 3D field). **M7's contribution is to build the rigorous 3D PDE simulation neither
parent did**, with the toroidal vortex as the soliton sector.

### The HydroBoros thesis

The Ouroboros "snake eating its tail" is a **self-linked toroidal vortex**. Fleury's toroidal EM
wave is what that vortex looks like in the hydrodynamic / Maxwell reading. The **Beltrami**
(force-free) condition `∇×F = λ F` is the steady, self-sustaining circulation that ties the two
together: it is simultaneously Fleury's monochromatic toroidal eigenmode (`ω = 2c/R₀`) and the
Ouroboros self-confinement. The bridge is already documented inside M5 (Fleury's Navier-Stokes ≡
generalized-Maxwell equivalence, Duda's superfluid mapping): see
[`../../m5_liquid_crystal/research/1b_topological_defect.md`](../../m5_liquid_crystal/research/1b_topological_defect.md)
§ "EM-hydrodynamics formal equivalence".

### Trkalian vs Beltrami: where the charge lives (Marc, 2026-06-30)

A Beltrami field `∇×w = λw` splits on whether the proportionality factor `λ` is constant. Taking the
divergence of both sides (`∇·(∇×w) = 0`) gives `∇λ·w + λ ∇·w = 0`, so:

| Case | `λ` | Divergence | Charge |
| --- | --- | --- | --- |
| **Trkalian** | constant | `∇·w = 0` (solenoidal, forced) | none |
| **general Beltrami** | variable `λ(x)` | `∇·w = −(∇λ·w)/λ ≠ 0` | **yes , the charge IS the variable-λ divergence** |

This is the bridge to Fleury's `charge = ∇·E`: the **electric charge is the non-constant-λ Beltrami
divergence**. The **endorsed build approach** (Marc, 2026-06-30, "the way to build it"): use the
**modern decomposition of Beltrami** (Sato-Yamada, § 3 #11) and **start the code on Trkalian
(constant-λ) exact solutions, then take off the training wheels on variable-λ** , generalize S&Y to
non-constant λ, find the charge-carrying ansatz, and evolve it. The non-constant case is exactly the
hard part ("the math gets very hairy"), so the Trkalian start is the foothold. This maps directly
onto the phases: M7.1-M7.3 stay in the charge-free Trkalian/neutral regime, and **M7.4 takes off the
training wheels , the constant-λ → variable-λ transition where the charge `∇·F` first appears** (§ 6).
Marc may bring in the **Spanish Beltrami school** (Enciso & Peralta-Salas, § 3 #6) as collaborators.

---

## 1. Fleury's toroidal model , the analytic target to reproduce first

dos Santos & Fleury, *An electromagnetic model of the electron*, [arXiv:2510.22384](https://arxiv.org/abs/2510.22384).
The electron is a rotating EM wave confined to a torus:

```text
E = i E₀ H(r−r₀) e^{i(φ−ωt)} [ â_R + i (1 + R/R₀) â_φ ]
B = i B₀ H(r−r₀) e^{i(φ−ωt)} â_z ,        B₀ = E₀/c
phase ψ = (φ − ωt)  ,  one wavelength around the ring  ,  left-handed triad
```

Free parameters: amplitude `E₀`, major radius `R₀`, minor radius `r₀`, frequency `ω`. The model
identifies **charge with the field divergence** (`ρ = ∇·E`, non-zero in free space, the novelty vs
divergence-free Rañada/Irvine knots), and pins the four parameters with three QED constraints:

| Constraint | Equation (thin-torus limit) |
| --- | --- |
| RMS charge | `√2 π² ε₀ E₀ r₀² = e` |
| Magnetic moment | `√2 ε₀ π c E₀ R₀ r₀² (1 + r₀²/2R₀²) = μ_B (1 + α/2π)` |
| Spin | `(1/c) ε₀ E₀² π² R₀² r₀² (1 + r₀²/4R₀²) = ℏ/2` |
| Faraday (fixes ω) | `ω = 2c/R₀` (monochromatic), phase velocity `v_p = 2c` |

Solved numerical values (the **M7.2 reproduction targets**):

```text
E₀ ≈ 0.286 E_S  (Schwinger limit)        R₀ ≈ 1.573 r_c  ≈ 6.073e−13 m  (Compton scale)
r₀ ≈ 0.152 r_c  ≈ 5.854e−14 m            U  ≈ 0.795 m_e c² ≈ 0.406 MeV
ω  ≈ 0.636 ω_D  ≈ 9.86e20 rad/s          (ω_D = Dirac / zitterbewegung frequency)
```

Acknowledged limits (Fleury, § 5.2): the Heaviside mask is unphysical at the torus boundary
(suggests Bessel-function envelopes), and the energy falls short of `m_e c²`. **M7 replaces the
analytic mask with a relaxed lattice field**, which is precisely where the PDE treatment earns its
keep.

Fleury's ansatz is a **rotating wave** in Ceperley's sense (his ref [13], § 3 #14): Ceperley's
circularly-polarized-EM rotating mode `E_r = E₀ e^{i(κz+φ−ωt)}` at `m=1` (his Eq 15) is literally this
torus phase, its **`J_m(κr)` Bessel envelope** is exactly the § 5.2 fix for the Heaviside mask, and
its angular-momentum law **`L_z = m(U/ω)`** (quantized when `U/ω = ℏ`) is the structural origin of the
spin `= ℏ/2` constraint. See [`../theory/ceperley_rotating_waves.md`](../theory/ceperley_rotating_waves.md) § 4b.

---

## 2. Ouroboros (M6) , the self-confinement the blend inherits

Werbos's two coupled vector fields `A_μ, J_μ` on Minkowski `(−,+,+,+)`
([`0d_canonical.md`](../../m6_ouroboros/research/0d_canonical.md), Werbos's
*Evaluating Universe Model Alternatives v5* shared doc):

```text
L = −¼ F_μν F^μν − ¼ G_μν G^μν + m_J² A_μ J^μ − f(J_μ J^μ)
    F = dA  ,  G = dJ  ,  f(s) = (g/4) s²  (canonical electron choice, g = 1.0)
```

In the linear limit (`J→0`) the A-field reproduces Maxwell exactly; the nonlinear `m_J² A·J −
f(J·J)` coupling produces localized time-periodic solitons ("chaoitons"). The electron benchmark is
`H/Q = 1.6969` (within 0.56% of `1.6875`), spin lock-in `2L/Q = 2ω`, charge from the Chern-Simons
linking number `Q_CS = 1`. **All of this exists today only as a 1D radial reduction** (the `(α,β)`
cylindrical ODE for the charged sector; an `l=1` spherical BVP for the neutral sector). M7 carries
the same Lagrangian, `m_J`, `g`, and the `H/Q` benchmark into a full 3D lattice.

---

## 3. Task 0 , collect the theory sources (do this first)

| # | Source | ID / venue | Role in M7 | Status |
| --- | --- | --- | --- | --- |
| 1 | dos Santos & Fleury, *EM model of the electron* | arXiv:**2510.22384** | toroidal ansatz + the 3 QED constraints | ✅ in `theory/` |
| 2 | Fleury & Rousselle, *Critical Review of Zitterbewegung Electron Models* | Symmetry **17**, 360 (2025) | why point-particle zitter models fail; the design brief | 🚧 fetch |
| 3 | Rañada, *Topological theory of the EM field* | Lett. Math. Phys. **18** (1989) | helicity = linking; the Hopfion construction | 🚧 fetch |
| 4 | Rañada, *Knotted solutions of Maxwell in vacuum* | J. Phys. A **23**, L815 (1990) | knotted **divergence-free** EM (the foil Fleury breaks); **helicity = linking = Hopf index** `n = (1/𝒶)∫A·B d³r` (Eq 3), the velocity↔A / vorticity↔B hydrodynamic analogy | ✅ in `electron_beltrami/` |
| 5 | Kedia, Bialynicki-Birula, Peralta-Salas, Irvine, *Tying knots in light fields* | PRL **111**, 150404 (2013) | Bateman construction = the **seeder** for knotted initial data | 🚧 fetch |
| 6 | Enciso & Peralta-Salas, *Knots and links in steady Euler flows* | Ann. Math. **175** (2012) | **Beltrami** fields realize any knot as a steady fluid vortex; the **Spanish Beltrami school** (ICMAT Madrid) Marc may enroll as collaborators | 🚧 fetch |
| 7 | Faddeev & Niemi, *Knots and particles* | Nature **387** (1997); hep-th/9610193 | the 4th-order **Hopfion stabilizer** (Derrick-evading, parallels M5's Skyrme term) | 🚧 fetch |
| 8 | Sutcliffe, *Knots in the Skyrme-Faddeev model* | arXiv:0705.1468 | the canonical **numerical** Hopfion relaxation recipe | 🚧 fetch |
| 9 | Werbos, *Stable Oscillatory Chaoitons in a 2-vector-field theory* | Zenodo 20030162 | the Ouroboros Lagrangian + soliton baseline | 🚧 fetch |
| 10 | Werbos, *Evaluating Universe Model Alternatives v5* | `theory/` `.docx` | Ouroboros / TUFT params, `H/Q = 1.6969` | ✅ present |
| 11 | **Sato & Yamada, *Local Representation and Construction of Beltrami Fields*** (Marc's Beltrami source) | arXiv:**1809.03136** (Physica D, 2019) | the construction recipe for the toroidal-Beltrami **seeder** (eikonal + equal-scale-factor rule); inhomogeneous + non-solenoidal `h` | ✅ [arXiv:1809.03136](https://arxiv.org/abs/1809.03136) (local PDF gitignored) + note [`sato_yamada_beltrami.md`](../theory/sato_yamada_beltrami.md) |
| 12 | Duda superfluid mapping | arXiv:2108.07896 | the EM ≡ hydrodynamics equivalence already cited in M5 | 🚧 fetch |
| 13 | Marc's **2 non-constant-λ (variable-λ) Beltrami papers** = **Kaiser** (*Force-free fields, nonconstant α*, CMP 211, 2000) + **Kravchenko** (*Beltrami fields, nonconstant proportionality factor*, J.Phys.A 36, 2003) | in the corpus (§ 3b) | the **variable-λ** case where the charge lives (`∇·w ≠ 0`); the target to generalize S&Y onto | ✅ both in `electron_beltrami/` (+ the Kravchenko & Oviedo *...on the plane* companion); the core of Q7 / M7.4 |
| 14 | Ceperley, *Rotating Waves* (Fleury's ref [13]) | Am. J. Phys. **60**, 938 (1992); DOI 10.1119/1.17020 | the **phase-vortex / rotating-wave** formalism; **circularly-polarized EM at `m=1` = Fleury's torus** (Eq 15), spin `L_z=mU/ω` + QM bridge (`U/ω=ℏ`), spherical + radiating forms, **Bessel envelope** (= Fleury's § 5.2 fix) | ✅ [DOI 10.1119/1.17020](https://doi.org/10.1119/1.17020) (paywalled; local PDF gitignored) + note [`ceperley_rotating_waves.md`](../theory/ceperley_rotating_waves.md) |
| 15 | Velazco & Ceperley, *Rotating Wave Fields for Microwave Applications* | IEEE Trans. MTT **41**(2), 330 (1993) | applications companion to #14: the **full cylindrical-cavity `E,H` field set** + the `ω_rot=ω/m` derivation | ✅ [DOI 10.1109/22.216476](https://doi.org/10.1109/22.216476) (paywalled; local PDF gitignored) |

Note on #11: Marc shared the Sato-Yamada writeup directly (the Gemini share link is Google-auth
gated and cannot be fetched headless). The paper + Marc's summary are now in `theory/`; the summary
note ties each result to a specific M7 piece.

### 3b. Marc's electron-Beltrami corpus (consolidated + complete , 2026-06-30)

Marc sent his **complete consolidated library**: **48 PDFs** in
[`../theory/electron_beltrami/`](../theory/electron_beltrami/) (year-prefixed `YEAR - Author -
Title.pdf`). This is the **source-of-record** , a superset of the original 20-source NotebookLM list
("HydroBoros: the Beltrami-OpenWave implementation") plus ~28 further papers. **All originals are now
present**, including the ones we could not fetch openly (Kaiser, Dombre, Kravchenko-2003,
Rañada-Maxwell). The earlier lossy Gemini-**ingested text** folder was **removed** (superseded by the
full PDFs); the two originals we had that Marc's set lacked (Fleury *Zilch-Zitter*, Kravchenko & Oviedo
*...on the plane* 2008) were folded into `electron_beltrami/`.

Highlights by M7 relevance (the full 48 are in the folder):

| Theme | Standout papers in `electron_beltrami/` |
| --- | --- |
| **Electron model (toroidal / clock)** | dos Santos *EM Model of the Electron* (FLDB Main + Appendix) + *Poloidal-Toroidal System v8e*; Fleury *Zilch-Zitter Electron*; Ceperley *Rotating Waves*; Hestenes *Zitterbewegung Structure*; **Catillon** *de Broglie particle internal clock* (the measured ZBW); **Wan** *Toroidal Vortices of Light*; Kovacs *What is Inside an Electron* |
| **Force-free / Beltrami / Trkalian** | Sato *Local Rep. of Beltrami Fields*; Woltjer; Marsh *Force-Free Magnetic Fields*; **Kaiser** + **Kravchenko** (the variable-λ pair, both now present) + Kravchenko & Oviedo *...on the plane*; Reed *Beltrami-Trkalian*; Dombre *ABC Flows*; **Enciso** *Beltrami Fields & Knotted Vortex Structures*; Mochizuki *Zero-Poynting EH Beltrami*; Takahashi *Beltrami vortex solutions* |
| **Knotted EM / topological / Clebsch** | Rañada *Knotted Maxwell*; Trueba *Topological EM*; Irvine *Linked & Knotted Beams*; **Arrayás** *Fibration by field lines*; Nastase *Fluid-EM Helicities*; Migdal *Clebsch Confinement*; **Wiegmann** *Chiral Anomaly in Euler Fluid & Beltrami Flow*; Yoshida *Nambu / Clebsch* |
| **Hopfions / chiral solitons** | **Kent** *Hopfions in Magnetic Multilayers* (experimental); Hall *Fusion & Fission of Particle-Like Chiral Structures* |
| **Ball lightning / LENR (applications)** | Rañada *Ball Lightning as Force-Free Knot*; Donoso *Riddle of Ball Lightning*; Biberian *Pd transmutation*; Lewis; Fleury *SuperColdFusion LENR*; Fomitchev *Neutron emission* |
| **Foundations / context** | Cheng *Field & Wave EM*; Barrett *Advanced EM*; Puthoff *Vacuum ZPE*; Celani *Maxwell & Occam*; Hillion; Rapoport *Cartan-Weyl Dirac*; Barbarosie *Divergence-free fields*; Brady *Emergent QM*; Morguer *QED test* |

The corpus confirms and sharpens Marc's framing: **toroidal-EM electron** + **force-free / variable-λ
Beltrami** (where the charge lives) + **knotted-EM / Clebsch** topology, with a **ball-lightning-as-
force-free-knot** thread and an **LENR** applications thread. New high-value finds for the build:
**Catillon** (the measured de Broglie clock, ties to Fleury's `ω`), **Wan** *Toroidal Vortices of
Light* (a toroidal EM structure directly adjacent to the M7 soliton), **Enciso** (the Spanish-school
Beltrami-knot existence results, Q4 / Q6 collaborators), and **Wiegmann** (the chiral-anomaly bridge
between Euler fluid and Beltrami flow). Provenance caveat: the **Reed** piece is a viXra non-reviewed
essay , read critically (seeds-only discipline, § 10).

---

## 4. Task 1 , define the substrate field (DECISION DEFERRED, see Open Questions Q1)

The medium representation, in the M5 / M6 vocabulary (`M5 = 3×3 matrix`, `M6 = double vector
field`). Candidates under consideration:

| Candidate | Field / point | Keeps Fleury (charge = ∇·) | Keeps Ouroboros (linking) | Knot-native | Note |
| --- | --- | --- | --- | --- | --- |
| **A. Riemann-Silberstein** `F = E + icB ∈ ℂ³` | 6 real | ✅ `∇·F` | ✅ helicity `∫A·B` | ✅ (Rañada / Bateman) | single complex field; clean but loses the explicit doublet |
| **B. Ouroboros doublet** `(A_μ, J_μ)` read via RS | 8 real | ✅ | ✅ native | ✅ via Bateman seed | reuses M6 substrate verbatim; current lean |
| C. Faddeev-Niemi `n : ℝ³→S²` | 3 real | ❌ divergence-free → **no charge** | ✅ Hopf charge | ✅✅ | breaks Fleury's central claim |
| D. Clebsch `(α,β)` / `ψ : ℝ³→ℂ` | 2 real | ❌ | partial | ✅✅ | useful as a **seed generator**, not the evolved DOF |

What any acceptable substrate must satisfy (the blend test):

| Requirement | Why it matters |
| --- | --- |
| Fleury's charge `= ∇·E` | the whole point of `2510.22384` vs divergence-free knots |
| Ouroboros charge `= ∫A·B` (linking) | the "snake eats tail" topological charge |
| The two charges are the **same object** (divergence + helicity of one field/pair) | this is the unification HydroBoros claims |
| Beltrami `∇×F = (ω/c) F` reproduces Fleury's `ω = 2c/R₀` | the force-free eigenstate = the steady vortex |
| M6 work carries over (`m_J`, `g`, `H/Q`) | do not re-derive what M6 already validated |
| Derrick stability via a 4th-order term | static stable soliton must exist (see § 5) |

**A-primary ontology (Marc, committed 2026-06-30):** the work starts from an **"A primary"** ontology,
the vector potential `A` is the fundamental field, with `F = dA`, the `E, B` fields and the charge
`∇·E` all derived from it. This favors the potential-primary candidate **B** (the `A_μ` doublet, where
`A` is literally the primary DoF) over the field-primary RS candidate **A**; the Riemann-Silberstein
`F` reading is kept as a derived diagnostic, not the evolved DoF.

The decision between A and B (and whether D enters only as a seeder) is **open**, tracked as **Q1**
in § 9. A full question tracker (M5-style, like
[`../../m6_ouroboros/research/0b_question_tracker.md`](../../m6_ouroboros/research/0b_question_tracker.md))
comes later; for now the open questions live in the table below.

---

## 5. The dynamics (what makes it rigorous, not analytic)

The energy functional blends three pieces, mirroring M5.11's `Skyrme curvature + LdG Higgs`:

```text
E[A,J] =  ∫ ½ (|E|² + c²|B|²)              Maxwell / fluid kinetic       (Fleury)
        + κ ∫ |F × (∇×F)|² / |F|²          Faddeev-Niemi 4th-order       (Derrick-evading stabilizer)
        + ∫ [ m_J² A·J − f(J·J) ]          Ouroboros self-confinement    (Werbos)
```

**Derrick scaling** under `x → λx` (the M5.11 argument, reused): kinetic `~ λ`, 4th-order `~ λ⁻¹`,
potential `~ λ³`. So `E(λ)` has an interior minimum: the 4th-order term provides the outward
pressure that balances collapse. **Static, finite-size, stable toroidal solitons are therefore
allowed**, the same Derrick-evasion that makes Faber's electron exist in M5.11.

Solve it two ways, exactly as M5.11 does:

| Method | Use |
| --- | --- |
| **Reverse-mode Taichi AD** for `δE/δ(A,J)`, validated against a numpy finite-difference gradient to `~1e-13` **before trusting any run** | the gradient for relaxation |
| **FIRE / L-BFGS** relaxation to `‖∇E‖ → 0` | the static soliton (M7.2-M7.4, M7.6, and the Arc B-C coverage phases) |
| **Minkowski leapfrog** (constrained integrator, `∇·B = 0`) | the clock + real-time dynamics (M7.5 stability, M7.11 annihilation) |

---

## 6. Phased sandbox progression (`sandbox_v1` → `sandbox_vN`), each phase gated against a KNOWN result

Each phase is its own iteration `M7.N`, matching its sandbox folder `sandbox_vN`, exactly the M5
convention (M5.11 ↔ `sandbox_v11`). **Every MODELS.md cell (§ 7) is assigned to a phase.** The
program runs in three arcs: **A** builds the electron and earns the column, **B** expands across the
forces and the remaining particle sectors, **C** groups the cells still untested in M5 itself.

### Arc A, the electron and the M7 column (M7.1–M7.7)

| Phase | Sandbox | Build | Validation gate (the credibility anchor) |
| --- | --- | --- | --- |
| **M7.1 , infra** | `sandbox_v1` | A-primary field on a 3D periodic lattice; AD energy gradient; FIRE minimizer; Bateman/Hopf + **Trkalian (constant-λ) Beltrami** seeders (S&Y exact solutions) | AD == numpy grad to `1e-12`; minimizer descends monotonically |
| **M7.2 , reproduce Fleury on the lattice** | `sandbox_v2` | seed the paper's toroidal ansatz; integrate charge, μ, spin, energy | recover `Q_rms = e`, `R₀ ≈ 1.573 r_c`, `E₀ ≈ 0.286 E_S`, `U ≈ 0.795 m_e c²`, `ω = 2c/R₀`. **M7's "reproduce Faber" trust-builder.** |
| **M7.3 , reproduce M6's electron in full 3D** | `sandbox_v3` | switch on the Ouroboros coupling; relax the 3D doublet | recover `H/Q = 1.6969` from the **3D** field (M6 only ever got it from a 1D radial BVP) |
| **M7.4 , the charged soliton (constant-λ → variable-λ) + its Coulomb field (NEW physics)** | `sandbox_v4` | turn the Trkalian seed into a **variable-λ** Beltrami knot (`∇·F ≠ 0` = the charge); relax with the 4th-order stabilizer; read off the far field | stable finite-size soliton; `‖∇E‖ → 0`; helicity / Hopf charge quantized; **charge `= ∇·F` = the variable-λ divergence**; **Coulomb `1/r`** sourced by it (Gauss's law) |
| **M7.5 , the clock + stability** | `sandbox_v5` | Minkowski real-time evolution; add the zitter dressing | persists many periods, no collapse mode; `ω` measured vs Dirac `ω_D` |
| **M7.6 , electron observables** | `sandbox_v6` | mass = field energy; spin `ℏ/2`; `μ_B(1 + α/2π)`; the Klein-Gordon twist sector; **two-charge Coulomb `E(d) ~ 1/d`** | the 4-observable electron (mass, charge, μ, spin) + KG + the two-body force law, from the relaxed field |
| **M7.7 , consolidate the M7 column (MILESTONE)** | `sandbox_v7` | fold the winning recipe into a `0d_canonical.md`-style spec; **add HydroBoros (M7) to MODELS.md** | one runnable canonical script, reproducible first-try; the electron cells land as the new column |

M7.1-M7.3 are the decisive credibility gates (reproduce **both** parents from the same lattice
code). M7.4 is the research core (the thing neither parent did). **M7.7 is the milestone: the M7
column exists in MODELS.md.** Note: **Coulomb rides with the electron**, not as a later forces phase:
once the divergence charge `∇·F` exists (M7.4), its `1/r` field is immediate (Gauss's law), and the
two-body `E(d) ~ 1/d` is confirmed at M7.6, exactly as M5 got Coulomb in its first `m5_1` milestone.

### Arc B, forces and the remaining particle sectors (M7.8–M7.13)

| Phase | Sandbox | Build | Validation gate |
| --- | --- | --- | --- |
| **M7.8 , magnetic force** | `sandbox_v8` | the per-defect magnetic structure carried by the electron's clock (Coulomb already landed in M7.4/M7.6) | magnetic force from the clock's `Γ₀` (pure twist is EM-silent; the M5 mechanism) |
| **M7.9 , gravity** | `sandbox_v9` | the time-axis boost of the field (the M5 4×4 route) | a GEM coupling that vanishes at zero boost; honest pass / fail (Ouroboros stops before gravity, so this is genuinely hard for M7) |
| **M7.10 , nuclear forces** | `sandbox_v10` | strong = the 4th-order short-range roll-off + linking tension; weak = a topology-reconnection (defect-class transition) | running-coupling onset at the core; a reconnection channel; partial, mirroring M5 |
| **M7.11 , antimatter + annihilation** | `sandbox_v11` | seed a soliton + anti-soliton (`Q → −Q`); evolve | charge ledger `±1 → 0`; rest energy released as outgoing waves; pair → vacuum |
| **M7.12 , the lepton + neutrino family** | `sandbox_v12` | vary knot size / linking: charged = self-linked torus, neutrino = the lighter loop | the lepton mass family (μ, τ); light neutral neutrino loops; flavour-rotation mixing |
| **M7.13 , dark matter** | `sandbox_v13` | the **neutral** knot (helicity-only, zero net `∇·F`), inheriting M6's neutral chaoiton | a stable neutral soliton; sub-MeV mass à la M6's `m_χ = 0.460 MeV` |

### Arc C, the composites still untested in M5, grouped (M7.14)

Per the M5 prescription, the cells that remain 🚧 [not yet tested] in M5 itself are grouped into one
later phase, after the column is consolidated. They depend on the electron + force primitives of
Arcs A-B already being in place.

| Phase | Sandbox | Build | Cells |
| --- | --- | --- | --- |
| **M7.14 , composites + atomic structure** | `sandbox_v14` | quark = fractional-charge string segment; baryon / meson = linked / twisted knots; atom = pilot-wave orbital quantization | Quarks · Baryons (p, n) · Mesons (π, K) · Orbital quantization |

One sandbox folder per phase, mirroring M5.11's layout: scripts named `vN_*.py` with checkpoints
under `sandbox_vN/_checkpoints/` (e.g. `sandbox_v1/v1_minimizer.py`, `sandbox_v2/v2_fleury_torus.py`,
… `sandbox_v8/v8_em_forces.py`, … `sandbox_v14/v14_composites.py`).

---

## 7. Filling the MODELS.md comparison table (a primary goal, M5 prescriptions)

A primary deliverable of this program is the **HydroBoros (M7)** column in the repo-root
[`MODELS.md`](../../../../MODELS.md) coverage matrix, evaluated against the same shared criteria as
M5 / M6 / M4. The M5 prescription governs every cell:

| Rule | What it means for M7 |
| --- | --- |
| Every cell is **script-backed** | each filled cell links to a runnable `sandbox_vN/` script (or a research note), reproducible by anyone |
| **Honest status icons** | ✅ validated in-platform · ⚠️ partial / caveats · ❌ tested + failed · 🔶 in progress · 🚧 planned |
| **Negatives are results** | a divergence-ful field that refuses to hold a knot (Q5) lands as a documented ❌, not a silence |
| **The column is earned, cell by cell** | M7 cells start 🚧, go 🔶 during a phase, settle to ✅ / ⚠️ / ❌ when the run lands |
| **New-model governance** | M7 is a new column: open an issue first so a maintainer adds it, then a script-backed PR + DCO + light review, per [`MODELS.md`](../../../../MODELS.md) § Contributing, [`ONBOARDING_MODELS.md`](../../../../ONBOARDING_MODELS.md), [`CONTRIBUTING.md`](../../../../CONTRIBUTING.md) |

Each phase fills specific cells, so the table is the running scoreboard of the program:

| Phase | MODELS.md cells targeted | Backing script |
| --- | --- | --- |
| **M7.1** | (infrastructure, no cell) | `sandbox_v1/` |
| **M7.2** | Charge quantization · Electron rest energy · Magnetic moment μ + spin J · EM waves (Maxwell) · de Broglie clock | `sandbox_v2/v2_fleury_torus.py` |
| **M7.3** | Electron rest energy (`H/Q = 1.6969` in full 3D) · Particle stability | `sandbox_v3/v3_ouroboros_3d.py` |
| **M7.4** | Particle stability (Derrick escape) · Charge quantization (helicity / linking + divergence) · Electric force (Coulomb 1/r, single-charge field) | `sandbox_v4/v4_linked_vortex.py` |
| **M7.5** | de Broglie clock (Zitterbewegung) · Particle stability | `sandbox_v5/v5_clock_stability.py` |
| **M7.6** | Magnetic moment μ + spin J · Spin-½ statistics · Quantum wave equation (Klein-Gordon) · Electric force (Coulomb, two-charge `E(d)~1/d`) | `sandbox_v6/v6_observables.py` |
| **M7.7 (milestone)** | consolidate the column + the M7 deep-dive (a `0d_canonical.md`, the "Per-model results of record" row) | `sandbox_v7/` |
| **M7.8** | Magnetic force | `sandbox_v8/v8_magnetic_force.py` |
| **M7.9** | Gravity | `sandbox_v9/v9_gravity.py` |
| **M7.10** | Strong force / confinement · Weak force | `sandbox_v10/v10_nuclear_forces.py` |
| **M7.11** | Antimatter + annihilation | `sandbox_v11/v11_annihilation.py` |
| **M7.12** | Neutrinos · Lepton mass spectrum (μ, τ) | `sandbox_v12/v12_lepton_neutrino.py` |
| **M7.13** | Dark matter candidate | `sandbox_v13/v13_dark_matter.py` |
| **M7.14** | Quarks · Baryons (p, n) · Mesons (π, K) · Orbital quantization | `sandbox_v14/v14_composites.py` |

All 21 MODELS.md criteria are covered: Arc A (M7.1-M7.7) earns the electron cells, **including
Coulomb** (tied to the electron's charge, M5-style), and consolidates the column at the M7.7
milestone; Arc B (M7.8-M7.13) fills the remaining forces (magnetic, gravity, nuclear) + annihilation
/ neutrinos / dark matter; Arc C (M7.14) groups the cells still 🚧 in M5 (quarks, baryons, mesons,
orbital quantization). Each phase upgrades its cells from 🚧 to a verified icon, as M5's column grew.

---

## 8. Path to production rendering (the M5 architecture is the template)

```text
research/sandbox_v1..vN/   headless Taichi research scripts        (this plan, M7.1–M7.14)
        │  winning recipe →
medium.py                  the (A,J) / RS substrate definition
engine1_seeds.py           Bateman/Hopf + toroidal-Beltrami seeders
engine2_pde.py             the nonlinear PDE evolution + minimizer
engine3_observables.py     charge (∇·F), helicity, energy, spin, μ
engine4_render.py          toroidal field-line / vorticity rendering
_launcher.py               registers HydroBoros for `openwave -x`
```

Reference layout: [`../../m5_liquid_crystal/`](../../m5_liquid_crystal/) (`medium.py`,
`engine1_seeds.py` … `_launcher.py`). Headless first (matplotlib PNG diagnostics in the sandbox);
rendering graduates once the electron is canonical (the M7.7 milestone), and the Arc B-C coverage
phases feed their observables into the renderer as they land, identical to how M5 reached
`_launcher.py`. No GUI / viz work before the physics is canonical.

---

## 9. Open questions (simple tracker for now; full M5-style tracker comes later)

| ID | Question | Status |
| --- | --- | --- |
| **Q1** | **Substrate field:** the Ouroboros doublet `(A_μ, J_μ)` read as Riemann-Silberstein (candidate B) vs single-field RS `F = E + icB` (candidate A); does Clebsch/`ψ` (D) enter only as a knot **seeder**? | 🔶 OPEN , narrowed to **B** by Marc's **A-primary** commitment (`A` fundamental, `F = dA`, charge derived); RS kept as a derived diagnostic; confirm at M7.1 |
| **Q2** | Exact 4th-order stabilizer form: Faddeev-Niemi `\|F×(∇×F)\|²/\|F\|²` vs a Skyrme-Faddeev variant; coefficient `κ` scale | 🚧 OPEN , settle empirically at M7.1/M7.4 |
| **Q3** | Are Fleury's divergence charge and Ouroboros's helicity/linking charge **forced equal**, or independent observables that must be reconciled? | 🚧 OPEN , the conceptual core of the blend |
| **Q4** | Beltrami / ABC source papers + further material from Marc | 🔶 PARTLY IN , Sato-Yamada landed (arXiv:1809.03136 + note in `theory/`, § 3 #11); more Marc material expected (#13) |
| **Q5** | Does a **variable-λ (divergence-ful)** Beltrami field still admit clean, stable knots, or does non-zero `∇·F` destabilize the Hopfion? | 🚧 OPEN , M7.4 answers it (the research question) |
| **Q6** | The `f(J·J)` potential form for M7: keep M6's `(g/4) s²`, or a form better suited to the toroidal sector? | 🚧 OPEN , revisit at M7.3 |
| **Q7** | The **charge-carrying ansatz**: start from exact Trkalian (constant-λ) solutions (S&Y) and **generalize to variable-λ** to introduce `∇·F` = charge; what is the right `λ(x)` profile + how does it evolve? | 🔶 OPEN , **approach endorsed by Marc** (2026-06-30, "the way to build it"); the specific `λ(x)` profile is the open part; the core of M7.4 |

---

## 10. Risks / unknowns

| Risk | Status / mitigation |
| --- | --- |
| 4th-order term cost on a 3D lattice (multi-hour GPU runs) | accepted; same regime as M5.11's vortex-loop relaxation |
| Complex-field / gauge constraints (`∇·B = 0`, RS reality) | constrained integrator + projection, as M5 does for the tensor |
| Variable-λ Beltrami math "gets very hairy" (Marc) | start from exact Trkalian (constant-λ) cases (S&Y), generalize incrementally (Q7); do not jump straight to the charged case |
| Does the variable-λ (divergence-ful) field admit clean knots (Q5) | the open research question; honest pass / fail at M7.4 |
| **Marc's AI-exchange material hallucinates** (his own warning: 2 months / 3 AIs, output not great) | treat it as **seeds only**, never as validated input; every claim re-derived in-platform (the MODELS.md reproducibility bar). A marathon-session review for extractable signal is optional, low priority |
| Masses (M7.6) may stay in tension with data | report honestly, including partial, as M5.11 did |

---

## 11. Cross-references

- Theory (our notes are in-repo; the source PDFs are local-only / gitignored, cited by public DOI/arXiv):
  [arXiv:2510.22384](https://arxiv.org/abs/2510.22384) (Fleury torus) ·
  Werbos *Evaluating Universe Model Alternatives v5* (shared doc, local only) ·
  [`../theory/sato_yamada_beltrami.md`](../theory/sato_yamada_beltrami.md) ([arXiv:1809.03136](https://arxiv.org/abs/1809.03136)) ·
  [`../theory/ceperley_rotating_waves.md`](../theory/ceperley_rotating_waves.md) (Ceperley rotating-wave equations; [AJP DOI 10.1119/1.17020](https://doi.org/10.1119/1.17020) / [IEEE DOI 10.1109/22.216476](https://doi.org/10.1109/22.216476)) ·
  [`../theory/feynman_maxwell_equations.md`](../theory/feynman_maxwell_equations.md) (Feynman Lectures II Ch 18, the Maxwell baseline for M7.2)
- Source corpus (Marc's consolidated library, 48 PDFs): [`../theory/electron_beltrami/`](../theory/electron_beltrami/) (Beltrami / force-free / knotted-EM / ball-lightning / LENR canon, § 3b)
- Rigor standard: [`../../m5_liquid_crystal/research/11a_vortex_loop.md`](../../m5_liquid_crystal/research/11a_vortex_loop.md) (M5.11 vortex-loop)
- Ouroboros canonical spec: [`../../m6_ouroboros/research/0d_canonical.md`](../../m6_ouroboros/research/0d_canonical.md) ·
  background [`../../m6_ouroboros/research/0a_background.md`](../../m6_ouroboros/research/0a_background.md)
- EM ≡ hydrodynamics bridge (already in M5): [`../../m5_liquid_crystal/research/1b_topological_defect.md`](../../m5_liquid_crystal/research/1b_topological_defect.md) § "EM-hydrodynamics formal equivalence"
- Production-engine template: [`../../m5_liquid_crystal/`](../../m5_liquid_crystal/) (`medium.py`, `engine1_seeds.py` … `_launcher.py`)
- Comparison table (the M7 column goal): [`MODELS.md`](../../../../MODELS.md) · new-model flow [`ONBOARDING_MODELS.md`](../../../../ONBOARDING_MODELS.md), [`CONTRIBUTING.md`](../../../../CONTRIBUTING.md)
- Model overview stub: [`../readme.md`](../readme.md)
