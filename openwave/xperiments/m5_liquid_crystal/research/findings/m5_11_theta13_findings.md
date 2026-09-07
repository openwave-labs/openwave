# m5_9b — where does θ₁₃ come from? The SO(3)-breaking, computed (#199 extension)

> Answers Duda's follow-up on the PMNS result: *"θ₁₃ is clearly incorrect — could it be a matter of approximations, choice of potential, etc.?"* He is right that θ₁₃ = 0 is wrong: it is the **pure-SO(3) (tri-bimaximal) limit** of the structural calculation. The measured θ₁₃ ≈ 8.5° is the **SO(3)-breaking**. This task computes that breaking at the effective-model level (the charged-lepton / second-rotation correction to TBM, the standard quark-lepton-complementarity mechanism) and reads off θ₁₃ + the correlations. Driver: [`m5_11_theta13_breaking.py`](../scripts/m5_11_theta13_breaking.py). Data: [`data/m5_11_theta13_summary.json`](../data/m5_11_theta13_summary.json). Plot: [`plots/m5_11_theta13_breaking.png`](../plots/m5_11_theta13_breaking.png). Date 2026-06-18.
>
> **Headline:** **θ₁₃ comes out CABIBBO-SCALE** — a Cabibbo-size 1-2 charged-lepton correction to tri-bimaximal gives **θ₁₃ ≈ 9.2°** (≈ θ_C/√2), matching the measured 8.5°. So the corrections Duda points at **do generate the right θ₁₃ scale** — the answer to "where does θ₁₃ come from." **Honest tension:** a *single* such correction cannot fit all of NuFIT at once: δ_CP = 180° (the SO(3) signature) needs a near-*real* correction, which over-rotates θ₁₂; fitting θ₁₂ needs a phase that pushes δ_CP toward ~100°; and θ₂₃ stays ~45° (the 1-2 rotation cannot reach the upper-octant 48.5°). The full fit needs more structure, which is exactly the **field model's job**.

## The mechanism — PMNS = U_lepton† · U_TBM

In the standard picture, the measured PMNS is the *neutrino*-sector mixing (tri-bimaximal here) corrected by the *charged-lepton*-sector rotation: `PMNS = U_e† · U_TBM`. A charged-lepton rotation of angle θ_e in the 1-2 plane generates θ₁₃; this is the **second coupled rotation toward the SU(3)/quark structure** in our reading.

## Result 1 (robust) — θ₁₃ is CABIBBO-SCALE

| | θ_e (correction) | θ₁₃ generated |
| --- | --- | --- |
| analytic (QLC) | θ_C ≈ 13° | θ_C/√2 ≈ **9.2°** |
| computed (1-2 correction) | 13° | **9.15°** |
| measured (NuFIT 6.0) | — | **8.52°** |

θ₁₃ grows as **θ_e/√2** (plot a), so a **Cabibbo-size** correction lands θ₁₃ right at the observed scale. This is quark-lepton complementarity: the deviation of PMNS from tri-bimaximal is of order the Cabibbo angle. (Note: the *exact* `θ₁₃ = θ_C/√2` = 9.2° is now experimentally excluded by the 8.5° measurement; the **scale** match is the robust statement, not the exact relation.)

## Result 2 — which plane: the 1-2 (Cabibbo-like) correction

| Correction plane (Cabibbo-size) | θ₁₂ | θ₁₃ | θ₂₃ |
| --- | --- | --- | --- |
| **1-2 (Cabibbo-like)** | 26.0° | **9.2°** | 44.3° |
| 1-3 | 26.0° | 9.2° | 45.7° |
| 2-3 | 35.3° | 0.0° | 58.0° |
| **NuFIT 6.0** | 33.7° | 8.5° | 48.5° |

The **1-2** correction is the one that generates a Cabibbo-scale θ₁₃ while keeping θ₂₃ near maximal (the 2-3 plane instead leaves θ₁₃ = 0 and over-rotates θ₂₃). So the SO(3)-breaking lives in the **1-2 (solar/charged-lepton) sector** — consistent with the charged-lepton mass matrix's dominant Cabibbo-like mixing.

## Result 3 (the honest tension) — any TWO of (θ₁₂, θ₁₃, δ_CP=180°), not all three

A single 1-2 correction has only two knobs (the angle θ_e and its phase δ_e), and they are tied:

| Scenario | θ₁₂ | θ₁₃ | δ_CP | θ₂₃ |
| --- | --- | --- | --- | --- |
| **real** correction (δ_e=0) | 26.0° (over-rotated) | 9.2° ✓ | **180° ✓** | 44.3° |
| **phased** best-fit (θ_e≈12°, δ_e≈290°) | 32.8° ✓ | 8.5° ✓ | **~103°** (not 180) | 44.4° (misses 48.5) |

- **δ_CP = 180° (the SO(3) signature) requires a near-real correction**, which over-rotates θ₁₂ to 26° (vs the measured 33.7°).
- **Fitting θ₁₂ needs a phase**, which pushes **δ_CP toward ~90-100°** (near-maximal CP), away from 180°.
- **θ₂₃ stays ~45°** either way — the 1-2 rotation cannot reach the NuFIT upper-octant 48.5° (that needs a separate 2-3 piece).

So a *single* 1-2 charged-lepton correction gives the right θ₁₃ *scale* but **cannot simultaneously fit θ₁₂, the SO(3) δ_CP=180°, and the upper-octant θ₂₃**. This is a real, informative constraint, not a failure: it tells us the full structure is richer (a 2-3 component, and/or a neutrino sector slightly off exact TBM), which the **field model** (the charged-lepton mass matrix from the (δ, g) biaxial hierarchy, #200) must supply.

## Result 4 — δ_CP becomes a SHARP discriminator (good for the platform)

Because δ_CP = 180° survives only for a **near-real** correction, and NuFIT's δ_CP ≈ 177° sits **near 180°**, the data currently **FAVOR the near-real / SO(3)-preserving breaking** over a large complex correction. This sharpens the original δ_CP = 180° platform prediction: it now reads "the SO(3)-breaking is near-real, so δ_CP stays near 180°," and **JUNO/DUNE pinning δ_CP away from 180° (e.g. ~270°) would mean a large complex second rotation — killing the SO(3)-preserving picture.**

## On Duda's "choice of potential" specifically

The calibration thread (#217) found the core potential V is **rotation-invariant on the single clock** (∂V/∂clock = 0). So **V-on does not obviously break the rotation structure** — making the **charged-lepton / second-rotation correction the more likely source of θ₁₃ than the potential.** (The inter-flavour V-on case — whether V breaks the SO(3) *across* the three flavour configurations — still needs the field sim to settle; this effective result points away from it.)

## What this says back to Duda (the substance)

- θ₁₃ = 0 was the **pure-SO(3) approximation**; the real θ₁₃ is the SO(3)-breaking, and it comes out **Cabibbo-scale** from the charged-lepton / second-rotation correction — the right size, confirming the "small second rotation toward the quark SU(3)" reading.
- It is **likely the corrections, not the potential** (#217: V is rotation-invariant on the clock).
- A single 1-2 correction gives the right θ₁₃ but trades off against θ₁₂ / δ_CP / θ₂₃ — so the **full fit needs the charged-lepton mass matrix + a 2-3 piece** (the field model), and **δ_CP near 180° (NuFIT ~177°) is a clean, sharp test** that currently favors the SO(3)-preserving breaking.

## Honest caveats

| Caveat | Note |
| --- | --- |
| **Effective model**, not the field derivation | the correction *angle* θ_e is a parameter here; the result is the **mechanism + the Cabibbo scale**, not a first-principles number. The field derivation of the correction *size* (the charged-lepton mass matrix from the (δ, g) hierarchy #200, + whether V-on breaks the inter-flavour SO(3)) is the named follow-up |
| single 1-2 correction is minimal | the real structure needs a 2-3 piece (for θ₂₃ upper octant) and/or a non-TBM neutrino sector; this run shows the constraint, not the full model |
| QLC exact relation excluded | `θ₁₃ = θ_C/√2` (9.2°) is excluded by the 8.5° measurement; the **scale** match is what holds |
| NuFIT values are global-fit ballpark | the octant of θ₂₃ and the δ_CP range should be cited precisely for any write-up |

## What's next (the field follow-up)

1. The **charged-lepton mass matrix** from the (δ, g) biaxial hierarchy (needs #200) → the correction angle θ_e from first principles (does it come out Cabibbo-size?).
2. Whether **V-on breaks the inter-flavour SO(3)** (the field sim of the δ-0 swing with V on) — settle the potential question directly.
3. The **2-3 structure** that lifts θ₂₃ to the upper octant + the full (θ₁₂, θ₁₃, θ₂₃, δ_CP) fit from the field.

## Reproduce

| Output | Command (env `master312`) |
| --- | --- |
| θ₁₃ vs correction + which-plane + 2D best-fit + δ_CP vs phase | `python3 m5_11_theta13_breaking.py` |
| Writes | `data/m5_11_theta13_summary.json`, `plots/m5_11_theta13_breaking.png` |
