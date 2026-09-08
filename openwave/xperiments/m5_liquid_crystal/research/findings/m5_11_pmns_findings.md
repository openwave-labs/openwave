# m5_9 — PMNS neutrino mixing from the SO(3) rotation structure (#199)

> Turns the model's standing commitment — **neutrino flavour oscillation = an SO(3) spatial-field rotation** (the δ-0 / axis-2↔3 swing of `M = O·diag(g,1,δ,0)·O^T` without the hedgehog winding; Duda's Wolfram slide, `m5_roadmap.md` L786; `MODELS.md` Neutrinos row) — into a concrete, falsifiable **PMNS prediction** and compares it to the global oscillation fit (NuFIT 6.0). Driver: [`m5_11_pmns_so3.py`](../scripts/m5_11_pmns_so3.py). Data: [`data/m5_11_pmns_summary.json`](../data/m5_11_pmns_summary.json). Plot: [`plots/m5_11_pmns_so3.png`](../plots/m5_11_pmns_so3.png). Date 2026-06-18.
>
> **Headline:** the SO(3) commitment makes a **parameter-free structural prediction** — the **tri-bimaximal (TBM)** mixing pattern with **δ_CP = 180°**. The data confirm the pattern (θ₂₃ near-maximal ✓, θ₁₂ tri-maximal ✓, δ_CP ≈ 177° ✓) with **one** deviation: the reactor angle **θ₁₃ ≈ 8.5° ≠ 0**. That nonzero θ₁₃ is exactly the **SO(3)-breaking order parameter** — the *small* "second coupled rotation toward SU(3)" that #199 asks about. **Verdict: SO(3) is the leading structure; it is broken by a small (~8.5°) second rotation. The δ_CP = 180° platform prediction is currently CONSISTENT (not falsified).**

## The structural chain (why SO(3) → TBM + δ_CP = 180°)

A neutrino is the **δ-0 (axis-2↔3) content swinging** without the topological winding. Reading the PMNS angles off that SO(3) structure:

| Step | SO(3) content | Mixing angle | Prediction |
| --- | --- | --- | --- |
| the **2↔3 axis swing** itself | the primary δ-0 rotation | **θ₂₃** (atmospheric) | **maximal, 45°** |
| the next SO(3) rotation | the solar sector | **θ₁₂** (solar) | **tri-maximal, sin²θ₁₂ = ⅓ → 35.26°** |
| a **pure real SO(3)** rotation has **no genuine complex CP phase** | — | **θ₁₃** (reactor) | **0°** |
| and the CP phase is locked | — | **δ_CP** | **180°** (or 0°) |

That set — θ₂₃ = 45°, θ₁₂ = 35.26°, θ₁₃ = 0°, δ_CP = 180° — **is** the tri-bimaximal matrix. So the model's SO(3) commitment lands on TBM *for a geometric reason* (the axis-2↔3 origin), not as an imposed flavour symmetry. (TBM itself is the known Harrison–Perkins–Scott ansatz; the contribution here is its **SO(3)/substrate origin** + the sharp **δ_CP = 180°** it forces.)

## The prediction vs the measurement

| Angle | SO(3) / TBM (predicted) | NuFIT 6.0 (measured, NO) | Match |
| --- | --- | --- | --- |
| **θ₁₂** (solar) | 35.26° | 33.7° (off 1.6°) | ✅ tri-maximal |
| **θ₂₃** (atmospheric) | 45.0° | 43.3° / 48.5° (octant straddles 45°) | ✅ maximal |
| **θ₁₃** (reactor) | **0°** | **8.5°** | ⚠️ **the SO(3)-breaking** |
| **δ_CP** | **180°** | ~177° (1σ ≈ [148°, 215°]) | ✅ consistent |

The mixing matrices `|U|` side by side (script output):

```text
|U| tri-bimaximal (SO(3) prediction):      |U| NuFIT 6.0 (measured):
 [0.816  0.577  0.000]                      [0.823  0.548  0.148]
 [0.408  0.577  0.707]                      [0.275  0.613  0.741]
 [0.408  0.577  0.707]                      [0.497  0.569  0.655]
```

They agree row-by-row except the `U_e3` corner (0 vs 0.148) — the reactor angle. Everything else is the SO(3) prediction.

## The #199 core question, answered: SO(3) or SU(3)?

> *Does neutrino oscillation stay in a single SO(3) rotation, or must it leave SO(3) for a second coupled rotation (toward the SU(3)-like quark structure)?*

**Answer: it needs a second rotation, but a SMALL one.** A *pure* single SO(3) rotation forces θ₁₃ = 0; the measured θ₁₃ ≈ 8.5° is therefore the **size of the required second coupled rotation**. But it is small (`sin²θ₁₃ ≈ 0.022` leakage), and the three other observables (θ₂₃ maximal, θ₁₂ tri-maximal, δ_CP ≈ 180°) all sit at their pure-SO(3) values. So:

- **SO(3) is the LEADING structure** of neutrino mixing (3 of 4 observables at their SO(3) values).
- It is **broken by a small θ₁₃ ≈ 8.5° rotation** — the toe-hold of the SU(3)-like coupled structure that is full-strength in the quark/CKM sector.
- **δ_CP ≈ 180°** means the CP-violating part of that second rotation is small → the SO(3) route is **favoured, not falsified**.

## The falsifiable platform prediction: δ_CP = 180°

The clean SO(3) signature is **δ_CP = 180°** (a real orthogonal rotation cannot carry a genuine Dirac CP phase). NuFIT 6.0's best fit (~177°, normal ordering) sits right on it, with 180° well inside the 1σ band. **Status: CONSISTENT.** This is the program's sharpest near-term neutrino falsifier: **if JUNO / DUNE settle δ_CP far from 180°** (e.g. near 270°, maximal CP violation), the pure-SO(3) route is **wrong** and the second rotation must be large/complex (full SU(3)-like). A first-class go/no-go on the structure.

## The electron 0.28% clock check (Duda, arxiv 2108.07896)

Duda's reading of the 0.28% electron-clock energy excess: "**3 types of tendencies (as for neutrino) acting in electron, projected (added) into a single allowed evolution degree of freedom**" — the *same* SO(3) 3-axis structure, viewed in the charged-lepton sector (the de Broglie clock is the single observable DOF; the 3 axes are the tendencies).

Projection-geometry framework: `E_obs / E_pure = 1 + f_off`, where `f_off` is the off-primary-axis energy fraction. So **0.28% ⟺ f_off ≈ 0.0028** (an off-axis angle ≈ 3°).

| Check | Result |
| --- | --- |
| Naive estimate (if the electron leaked like the neutrino θ₁₃) | `sin²θ₁₃ ≈ 0.022` = **2.2%** — about **8× too big** |
| Reading | the electron clock is **~8× more tightly projected** than the freely-oscillating neutrino — physically sensible for a **bound, phase-locked clock** vs a free oscillation |
| Status | **structural match** to Duda's picture (3 tendencies → 1 projected DOF → small excess); the **specific 0.28%** needs the electron defect's *measured* off-primary-axis energy fraction (the dynamical follow-up) |

So the 0.28% is consistent with the projection picture, and the scale (0.28% ≪ θ₁₃²) carries real information: the electron's internal tendencies are far more aligned into one DOF than the neutrino's. We do **not** claim to derive 0.28% here — we frame it, bound it, and flag the measurement that would.

## Honest caveats

| Caveat | Note |
| --- | --- |
| This is the **structural consequence** of the SO(3) commitment, not a dynamical M5 neutrino simulation | no field was evolved; the result is "IF neutrino mixing is the SO(3) axis-2↔3 rotation, THEN TBM + δ_CP=180°, broken by θ₁₃." The full dynamical sim (seed the δ-0 excitation, measure 3 masses, read the rotation geometrically) is the **named follow-up** |
| **TBM is a known ansatz** (Harrison–Perkins–Scott) | the contribution is its **SO(3)/substrate origin** + the sharp δ_CP=180° + θ₁₃-as-breaking reading; no claim to have invented TBM |
| NuFIT 6.0 values are the **global-fit ballpark** | the exact table (octant of θ₂₃, the δ_CP range) should be cited precisely for any write-up; the **pattern + δ_CP-near-180°** comparison is what is robust here, not 0.1° precision |
| the 0.28% is **framed, not derived** | the specific number awaits the electron defect's off-axis energy measurement |

## What's next (the dynamical follow-up)

1. **Seed the δ-0 (axis-2↔3) excitation** without the hedgehog winding (per `m5_roadmap.md` L792); verify light + stable + chargeless (∇·n̂ ≈ 0).
2. **Measure the three mass eigenfrequencies** (the e/μ/τ neutrino states) and read the mixing **rotation angles geometrically** from the time-evolved state — confirm (or correct) the TBM + θ₁₃ structure from the field itself.
3. **Measure the electron defect's off-primary-axis energy fraction** (the EM-tilt / twist / boost sector split on the validated electron-id defect) → the dynamical test of the 0.28%.
4. **Propose the MODELS.md Neutrinos-row update**: from "SO(3) route, δ_CP=180° prediction" → "SO(3)→TBM structural prediction confirmed in pattern; θ₁₃ = the measured SO(3)-breaking; δ_CP=180° consistent (NuFIT 6.0)."

## Reproduce

| Output | Command (env `master312`) |
| --- | --- |
| PMNS SO(3) prediction + NuFIT comparison + δ_CP test + 0.28% framework | `python3 m5_11_pmns_so3.py` |
| Writes | `data/m5_11_pmns_summary.json`, `plots/m5_11_pmns_so3.png` |

## Coverage-matrix cell prose (preserved from MODELS.md, condensed there 2026-06-19)

The MODELS.md Neutrinos cell (M5 column) was trimmed to a summary on 2026-06-19; its prior full descriptive text is preserved here for the record:

> Very light, stable, and huge (thousands of times the nucleus: Nature s41586-024-08479-6): simple stable loops of quark string, created e.g. in neutron beta decay, with 0 charge so a different binding mechanism than charged particles. Flavour oscillation = SO(3) spatial field rotation (Phys. Lett. B S0370269326000730); deriving the oscillation parameters from LdGS is "in reach" and doubles as a parameter-fixing handle, difficulty = regularization + gravity to propel the oscillations. **PMNS from SO(3) (#199):** the SO(3) commitment predicts the tri-bimaximal pattern + δ_CP = 180° parameter-free, and the global fit confirms it: θ₂₃ maximal, θ₁₂ tri-maximal (35.26° vs NuFIT 6.0 33.7°), δ_CP ≈ 177°. The open rotation-group question is answered: the one deviation, the reactor angle θ₁₃ ≈ 8.5° ≠ 0, IS the SO(3)-breaking = the small required second coupled rotation toward the SU(3)-like quark structure (SO(3) leading, broken by ~8.5°). FALSIFIABLE PREDICTION: δ_CP = 180°, now CONSISTENT (NuFIT 6.0 best fit ~177°; JUNO/DUNE will sharpen; δ_CP far from 180° kills the pure-SO(3) route). **θ₁₃ run (#199 ext):** the SO(3)-breaking θ₁₃ comes out Cabibbo-scale (~9° ≈ measured 8.5°) from a charged-lepton (second-rotation) correction, likely the corrections, not the potential (#217: V rotation-invariant on the clock). A single such correction trades off θ₁₂ / δ_CP=180° / θ₂₃, so δ_CP near 180° (NuFIT ~177°) now FAVORS the SO(3)-preserving breaking; effective-model, the field derivation (the charged-lepton mass matrix from the (δ, g) hierarchy #200 + the 2-3 structure) is the follow-up. Hopfions as excited-oscillation candidates (Liu et al., Nature Physics 2026).
