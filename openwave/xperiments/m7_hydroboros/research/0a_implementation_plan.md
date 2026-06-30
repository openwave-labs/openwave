# M7 / HydroBoros , Implementation Plan

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
| **Fleury** (hydrodynamics school), [`2510.22384v2`](../theory/2510.22384v2.pdf) | EM field `E, B` confined to a torus | closed-form **analytic** ansatz + Heaviside mask | `ρ = ∇·E` (geometric / divergence charge) | mask unphysical at boundary; energy lands at `0.795 m_e c²`; **no dynamics, no PDE** |
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

---

## 1. Fleury's toroidal model , the analytic target to reproduce first

dos Santos & Fleury, *An electromagnetic model of the electron*, [`arXiv:2510.22384v2`](../theory/2510.22384v2.pdf).
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

---

## 2. Ouroboros (M6) , the self-confinement the blend inherits

Werbos's two coupled vector fields `A_μ, J_μ` on Minkowski `(−,+,+,+)`
([`0d_canonical.md`](../../m6_ouroboros/research/0d_canonical.md),
[`../theory/Evaluating%20Universe%20Model%20Alternativesv5.docx`](../theory/Evaluating%20Universe%20Model%20Alternativesv5.docx)):

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
| 4 | Rañada, *Knotted solutions of Maxwell in vacuum* | J. Phys. A **23**, L815 (1990) | knotted **divergence-free** EM (the foil Fleury breaks) | 🚧 fetch |
| 5 | Kedia, Bialynicki-Birula, Peralta-Salas, Irvine, *Tying knots in light fields* | PRL **111**, 150404 (2013) | Bateman construction = the **seeder** for knotted initial data | 🚧 fetch |
| 6 | Enciso & Peralta-Salas, *Knots and links in steady Euler flows* | Ann. Math. **175** (2012) | **Beltrami** fields realize any knot as a steady fluid vortex | 🚧 fetch |
| 7 | Faddeev & Niemi, *Knots and particles* | Nature **387** (1997); hep-th/9610193 | the 4th-order **Hopfion stabilizer** (Derrick-evading, parallels M5's Skyrme term) | 🚧 fetch |
| 8 | Sutcliffe, *Knots in the Skyrme-Faddeev model* | arXiv:0705.1468 | the canonical **numerical** Hopfion relaxation recipe | 🚧 fetch |
| 9 | Werbos, *Stable Oscillatory Chaoitons in a 2-vector-field theory* | Zenodo 20030162 | the Ouroboros Lagrangian + soliton baseline | 🚧 fetch |
| 10 | Werbos, *Evaluating Universe Model Alternatives v5* | `theory/` `.docx` | Ouroboros / TUFT params, `H/Q = 1.6969` | ✅ present |
| 11 | **Beltrami / ABC sources from Marc's Gemini link** | (link is Google-auth gated, cannot fetch headless) | the specific Beltrami papers + further content Marc is sharing | 🔶 **Rodrigo to fetch + paste IDs** |
| 12 | Duda superfluid mapping | arXiv:2108.07896 | the EM ≡ hydrodynamics equivalence already cited in M5 | 🚧 fetch |

Note on #11: the share link `gemini.google.com/share/9016becf3b08` returns only a sign-in shell to a
headless fetcher. Rodrigo will pull the arXiv IDs (and the additional material Marc is sharing) and
we add them to this table. Until then the standard Beltrami / Euler-knot canon (#6, plus ABC flows)
covers the force-free-field foundation.

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
| **FIRE / L-BFGS** relaxation to `‖∇E‖ → 0` | the static soliton (M7.2-M7.4, M7.6) |
| **Minkowski leapfrog** (constrained integrator, `∇·B = 0`) | the clock + real-time stability (M7.5) |

---

## 6. Phased sandbox progression (`sandbox_v1` → `sandbox_v7`, canonical), each phase gated against a KNOWN result

Each phase is its own iteration `M7.N`, matching its sandbox folder `sandbox_vN`, exactly the M5
convention (M5.11 ↔ `sandbox_v11`).

| Phase | Sandbox | Build | Validation gate (the credibility anchor) |
| --- | --- | --- | --- |
| **M7.1 , infra** | `sandbox_v1` | RS field on a 3D periodic lattice; AD energy gradient; FIRE minimizer; Bateman/Hopf + toroidal-Beltrami **seeders** | AD == numpy grad to `1e-12`; minimizer descends monotonically |
| **M7.2 , reproduce Fleury on the lattice** | `sandbox_v2` | seed the paper's toroidal ansatz; integrate charge, μ, spin, energy | recover `Q_rms = e`, `R₀ ≈ 1.573 r_c`, `E₀ ≈ 0.286 E_S`, `U ≈ 0.795 m_e c²`, `ω = 2c/R₀`. **M7's "reproduce Faber" trust-builder.** |
| **M7.3 , reproduce M6's electron in full 3D** | `sandbox_v3` | switch on the Ouroboros coupling; relax the 3D doublet | recover `H/Q = 1.6969` from the **3D** field (M6 only ever got it from a 1D radial BVP) |
| **M7.4 , the self-linked toroidal soliton (NEW physics)** | `sandbox_v4` | relax a Beltrami self-linked vortex with the 4th-order stabilizer on | stable finite-size soliton; `‖∇E‖ → 0`; helicity / Hopf charge quantized; charge `= ∇·F` |
| **M7.5 , the clock + stability** | `sandbox_v5` | Minkowski real-time evolution; add the zitter dressing | persists many periods, no collapse mode; `ω` measured vs Dirac `ω_D` |
| **M7.6 , observables + spectrum** | `sandbox_v6` | mass = field energy; spin `ℏ/2`; `μ_B(1 + α/2π)`; vary knot / linking | electron observables from the relaxed field; honest pass / fail |
| **M7.7 , canonicalize** | `sandbox_v7` | fold the winning recipe into a `0d_canonical.md`-style spec | one runnable canonical script, reproducible first-try |

M7.1-M7.3 are the decisive credibility gates (reproduce **both** parents from the same lattice code).
M7.4 is the research core (the thing neither parent did). M7.5-M7.6 are the physics payoff.

One sandbox folder per phase, mirroring M5.11's layout: scripts named `vN_*.py` with checkpoints
under `sandbox_vN/_checkpoints/` (e.g. `sandbox_v1/v1_minimizer.py`, `sandbox_v2/v2_fleury_torus.py`,
`sandbox_v3/v3_ouroboros_3d.py`, `sandbox_v4/v4_linked_vortex.py`, `sandbox_v5/v5_clock_stability.py`,
`sandbox_v6/v6_observables.py`, `sandbox_v7/v7_canonical.py`).

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
| **M7.4** | Particle stability (Derrick escape) · Charge quantization (helicity / linking + divergence) | `sandbox_v4/v4_linked_vortex.py` |
| **M7.5** | de Broglie clock (Zitterbewegung) · Particle stability | `sandbox_v5/v5_clock_stability.py` |
| **M7.6** | Magnetic moment μ + spin J · Spin-½ statistics · Lepton mass spectrum | `sandbox_v6/v6_observables.py` |
| **M7.7** | consolidate the column + the M7 deep-dive (a `0d_canonical.md`, the "Per-model results of record" row) | `sandbox_v7/` |

The column lands incrementally: the first credible cells (M7.2 charge / mass / Maxwell) are the
trigger to open the new-column issue and add **HydroBoros (M7)** to the table. Each later phase
upgrades its cells from 🚧 to a verified icon, exactly as M5's column grew.

---

## 8. Path to production rendering (the M5 architecture is the template)

```text
research/sandbox_v1..v7/   headless Taichi research scripts        (this plan, M7.1–M7.7)
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
rendering only after M7.7, identical to how M5 reached `_launcher.py`. No GUI / viz work lands before
the physics is canonical.

---

## 9. Open questions (simple tracker for now; full M5-style tracker comes later)

| ID | Question | Status |
| --- | --- | --- |
| **Q1** | **Substrate field:** the Ouroboros doublet `(A_μ, J_μ)` read as Riemann-Silberstein (candidate B) vs single-field RS `F = E + icB` (candidate A); does Clebsch/`ψ` (D) enter only as a knot **seeder**? | 🔶 OPEN , current lean = B (reuses M6 substrate + keeps both charges); decide at M7.1 |
| **Q2** | Exact 4th-order stabilizer form: Faddeev-Niemi `\|F×(∇×F)\|²/\|F\|²` vs a Skyrme-Faddeev variant; coefficient `κ` scale | 🚧 OPEN , settle empirically at M7.1/M7.4 |
| **Q3** | Are Fleury's divergence charge and Ouroboros's helicity/linking charge **forced equal**, or independent observables that must be reconciled? | 🚧 OPEN , the conceptual core of the blend |
| **Q4** | Beltrami / ABC source papers + further material from Marc (Gemini link is auth-gated) | 🔶 OPEN , Rodrigo to fetch + paste arXiv IDs into § 3 |
| **Q5** | Does a **divergence-ful** field still admit clean, stable knots, or does non-zero `∇·F` destabilize the Hopfion? | 🚧 OPEN , M7.4 answers it (the research question) |
| **Q6** | The `f(J·J)` potential form for M7: keep M6's `(g/4) s²`, or a form better suited to the toroidal sector? | 🚧 OPEN , revisit at M7.3 |

---

## 10. Risks / unknowns

| Risk | Status / mitigation |
| --- | --- |
| 4th-order term cost on a 3D lattice (multi-hour GPU runs) | accepted; same regime as M5.11's vortex-loop relaxation |
| Complex-field / gauge constraints (`∇·B = 0`, RS reality) | constrained integrator + projection, as M5 does for the tensor |
| Beltrami source IDs from Marc's Gemini link unavailable headless | Rodrigo fetches (Q4); standard canon covers the gap meanwhile |
| Does the divergence-ful field admit clean knots (Q5) | the open research question; honest pass / fail at M7.4 |
| Masses (M7.6) may stay in tension with data | report honestly, including partial, as M5.11 did |

---

## 11. Cross-references

- Theory: [`../theory/2510.22384v2.pdf`](../theory/2510.22384v2.pdf) (Fleury torus) ·
  [`../theory/Evaluating%20Universe%20Model%20Alternativesv5.docx`](../theory/Evaluating%20Universe%20Model%20Alternativesv5.docx) (Werbos Ouroboros/TUFT)
- Rigor standard: [`../../m5_liquid_crystal/research/11a_vortex_loop.md`](../../m5_liquid_crystal/research/11a_vortex_loop.md) (M5.11 vortex-loop)
- Ouroboros canonical spec: [`../../m6_ouroboros/research/0d_canonical.md`](../../m6_ouroboros/research/0d_canonical.md) ·
  background [`../../m6_ouroboros/research/0a_background.md`](../../m6_ouroboros/research/0a_background.md)
- EM ≡ hydrodynamics bridge (already in M5): [`../../m5_liquid_crystal/research/1b_topological_defect.md`](../../m5_liquid_crystal/research/1b_topological_defect.md) § "EM-hydrodynamics formal equivalence"
- Production-engine template: [`../../m5_liquid_crystal/`](../../m5_liquid_crystal/) (`medium.py`, `engine1_seeds.py` … `_launcher.py`)
- Comparison table (the M7 column goal): [`MODELS.md`](../../../../MODELS.md) · new-model flow [`ONBOARDING_MODELS.md`](../../../../ONBOARDING_MODELS.md), [`CONTRIBUTING.md`](../../../../CONTRIBUTING.md)
- Model overview stub: [`../readme.md`](../readme.md)
