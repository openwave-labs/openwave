# M4.14 — Cosmological Constant from the BCC Lattice

## Status
DONE (post-hoc)

## Criterion
`Cosmology: Λ / cosmic acceleration`

## Objective
Predict the cosmological constant Λ from the BCC lattice chain, with no
continuously adjustable parameter beyond the four dimensional anchors
(`r_e`, `m_e`, `c`, `e`) and one cosmological anchor (`T_cmb`).

## Method
Chain (each step printed in the script's Part numbering):

1. **BCC stiffness** `N_geom = 8π⁴(1−ζ)` from `derive_eps_M_from_BCC`.
2. **Vacuum deficit** `ε_M = 1/(N_geom·π³)` — the same ε_M as in α and AMM.
3. **AMM projection** `O_μ = M_μ·π³·ε_M = M_μ/N_geom`; requiring
   `O_μ = 1/(2π)²` forces `M_μ = N_geom/(2π)²` (ideal `= 2π²`).
4. **Horizon factor** `H = 8·(2π)² = 32π²`. Each of the 8 BCC
   directions contributes one 2D phase unit `(2π)²`. This is a
   phase-unit ansatz, not a first-principles derivation.
5. **Lattice discrimination** under `H(N) = N·(2π)²`: only `N=8` (BCC)
   matches Planck.
6. **EWT trinity** `α_geom`, `ℏ_geom`, `G_geom`, `λ_l` from the engine.
7. **Planck temperature** `T_p = (1/k_B)·√(ℏ·c⁵/G)`.
8. **Horizon radius** `R_H = λ_l·(T_p/T_cmb)²/H`. This recovers the
   Haug relation `T_cmb = (T_p/8π)·√(2λ_l/R_H)` algebraically.
9. **Hubble constant** `H_0 = c/R_H`.
10. **de Sitter Λ** `Λ_dS = 3H_0²/c²`.
11. **Vacuum fraction** `Ω_vac = η_BCC·C_gap` (composition rule =
    model assumption).
12. **Observed Λ** `Λ_vac = Ω_vac·Λ_dS`.

## Results

### Vacuum chain (Parts 1–4)

| Quantity | Value |
|---|---|
| `N_ideal` | 779.2727282720 |
| `ζ` | 6.0339643713×10⁻⁴ |
| `N_geom` | 778.8025178842 |
| `ε_M` | 4.141169769303932×10⁻⁵ |
| `M_μ` (derived) | 19.7272982339 |
| `M_μ` (ideal, `ζ=0`) | 19.7392088022 |
| `H` | 315.8273408349 |
| `H·M_μ = 8·N_geom` | ratio `1.000000000000` |

### EWT trinity (Parts 6–7)

| Quantity | Value |
|---|---|
| `α_geom⁻¹` | 137.036262364181 |
| `ℏ_geom` | 1.054574×10⁻³⁴ J·s |
| `G_geom` | 6.677520×10⁻¹¹ m³·kg⁻¹·s⁻² |
| `λ_l` | 1.616646×10⁻³⁵ m |
| `T_p` | 1.416444×10³² K |

### Horizon and H₀ (Parts 8–9)

| Quantity | Value |
|---|---|
| `R_H^naive` | 4.366379×10²⁸ m |
| `R_H` | 1.382521×10²⁶ m |
| Recovered `T_cmb` | 2.7255000000 K |
| `H_0^EWT` | 66.911313 km/s/Mpc |

Tension: Planck 2018 (−0.83σ), DESI 2024 (−2.59σ), SH0ES 2022 (−5.89σ).

### Lattice discrimination (Part 5, conditional on phase-unit ansatz)

| N | lattice | H₀ [km/s/Mpc] | tension |
|---|---|---|---|
| 4 | tetrahedron | 33.4557 | −62.79σ |
| 6 | simple cubic | 50.1835 | −31.81σ |
| **8** | **BCC** | **66.9113** | **−0.83σ** |
| 12 | FCC | 100.3670 | +61.12σ |
| 16 | hypercubic | 133.8226 | +123.08σ |

### Λ and Ω_vac (Parts 10–12)

| Quantity | Value |
|---|---|
| `Λ_dS` | 1.569560×10⁻⁵² m⁻² |
| `Λ_dS/Λ_Planck` | 1.441285 |
| `η_BCC` | 0.6801747616 |
| `C_gap` | 1.0140759358 |
| `Ω_vac` | 0.6897488579 |
| `Λ_vac` | 1.082602×10⁻⁵² m⁻² |
| `Λ_vac/Λ_Planck` | 0.994125 (rel. diff. `−0.588%`) |

`Ω_vac` vs Planck: TT `+0.737%` (`+0.69σ`), +BAO `+0.123%` (`+0.15σ`).

### Matter correction (Part 13, diagnostic only)

Fit `R_H(Ω_m) = R_H(0)·(1−Ω_m)^p` to `H_0^Planck` at `Ω_m = 0.31`:
`p = 0.018011`, `f(0.31) = 0.993339`. `H_0(corr) = 67.360000 km/s/Mpc`,
`Λ(corr) = 1.590680×10⁻⁵² m⁻²`. **Not part of the derivation** — the
smallness of `p` confirms de Sitter dominates and matter is a ~0.6%
perturbation.

## Interpretation

### Criterion connection: cosmic acceleration / dark energy

The criterion `Cosmology: Λ / cosmic acceleration` is the framework's
target, not a side effect. The chain provides quantitative support in
three senses:

1. **Observation.** `Ω_Λ ≈ 0.69` and `Λ ≈ 1.09×10⁻⁵² m⁻²` are
   observationally established facts. The framework predicts both to
   sub-percent precision: `Ω_vac = 0.6897` at `+0.15σ` of Planck+BAO,
   `Λ_vac = 1.0826×10⁻⁵² m⁻²` at `−0.588%` of Planck 2018.
2. **Mechanism.** The acceleration is not attributed to a mysterious
   substance but to the effective vacuum response of the BCC lattice:
   `Ω_vac = η_BCC·C_gap`. The two ingredients are derived or
   independently tested — `η_BCC` is pure crystallography, `C_gap`
   fixes `M_W` to `0.012%` in the electroweak sector.
3. **Recovery.** The Haug horizon relation is a consequence of the
   BCC horizon model, not an import.

The framework does **not** postulate dark energy as a substance. It
identifies the quantity that Λ-CDM calls the dark-energy fraction `Ω_Λ`
with the effective vacuum response of the BCC lattice. What is **not**
derived is why the composition rule is the product `η_BCC·C_gap` rather
than another combination (sum, or `η_BCC` alone). This is the principal
open objective.

### What the chain earns

- `Λ_vac` is a prediction of the observed cosmological constant, not a
  fit: `−0.588%` against Planck 2018 with **one cosmological anchor**
  (`T_cmb`).
- `H_0 = 66.911 km/s/Mpc` sits on the Planck side of the Hubble tension
  (`−0.83σ`).
- `Ω_vac = 0.6897` matches Planck+BAO `0.6889 ± 0.0056` at `+0.15σ`.
- The Haug relation `T_cmb = (T_p/8π)√(2λ_l/R_H)` is **recovered**, not
  imported, from the BCC horizon model (machine precision).
- Lattice test **conditionally** excludes `N ≠ 8` at tens of σ.

### Where the gaps are (named, per the criterion)

1. **Horizon factor `H = 8·(2π)²` is a phase-unit ansatz.** Each BCC
   direction contributes one `(2π)²`; a first-principles derivation
   is open. The lattice test is a *conditional*
   falsification, not a derivation that other lattices are impossible.
2. **`Ω_vac = η_BCC·C_gap` composition rule is a model assumption.**
   Both ingredients are derived; the *product* is identified, not
   derived from a horizon-balance argument.
3. **Relaxation dynamics `a(t)` is not delivered.** The chain predicts the
   present-day snapshot values but not the cosmic history.

### Falsifications named

- Any future measurement of `Ω_Λ` or `Λ` more than ~1% from the
  predicted values (`Ω_vac = 0.6897`, `Λ_vac = 1.0826×10⁻⁵² m⁻²`)
  rules out the mechanism. Planck+BAO uncertainty is 0.81%.
- Under the phase-unit ansatz, a lattice other than BCC would give
  `H_0` off by tens of σ.
- The relaxation picture predicts a slow time variation of `Λ` at
  cosmological redshifts once the EMC-wall profile is anchored;
  currently not falsifiable.

**`G_geom` cancels in the cosmological combinations.** `ℏ/G` and `ℏ·G`
both reduce to expressions in `r_e`, `m_e`, `c`, `α_geom`, so the
individual CODATA residuals on `G` (0.048%) and `λ_l` (0.028%) do not
propagate to `H_0` or `Λ`. The chain is a test of the BCC geometry at
the level of the products, consistent with Haug's composite-constant
view. A 0.048% `G` residual therefore does not translate to a 0.048%
`H_0` residual.

## Technical notes

- `c` and `k_B` are exact SI 2019 unit-conversion constants, not
  dynamical inputs. `T_cmb` is a measured anchor (COBE/FIRAS).
- `H·M_μ = 8·N_geom` is algebraic: it holds for any `X` in
  `H = 8·X`, `M_μ = N_geom/X`. The non-trivial content is the shared
  `(2π)²` origin (inverse 2D Fourier normalisation), not the algebraic
  cancellation of `X`.
- The lattice test uses `H(N) = N·(2π)²`; this scales `H_0(N)`
  linearly in `N` and is *conditional* on the phase-unit ansatz.
- The `Λ_dS/Λ_Planck = 1.441` ratio decomposes as
  `(H_0^EWT/H_0^Planck)² · (Ω_Λ^Planck)⁻¹ = 0.9867 · 1.4605 = 1.4411`.
  Both factors are as reported by the script; the ratio is structural,
  not a fitted number.

Needs `m4_7_ewt_emergence_engine.py` in the same directory. All numbers
in this finding are taken from the script's console output; the script
does not write data files (it is a diagnostic chain, not a sweep).

## Artifacts

- `research/scripts/m4_14_ewt_cosmological_constant.py`

## Reference

Enhanced EWT manuscript, version 5.3.1 (2026-11-13), § "The Cosmological
Constant from the BCC Lattice" (label `sec:cosmo_summary`). DOI:
[10.5281/zenodo.17654657](https://doi.org/10.5281/zenodo.17654657).

### Cosmological anchors and benchmarks

- Planck Collaboration, Aghanim, N., Akrami, Y., et al. (2020). Planck
  2018 results. VI. Cosmological parameters. *Astronomy & Astrophysics*,
  **641**, A6.
  DOI: [10.1051/0004-6361/201833910](https://doi.org/10.1051/0004-6361/201833910).
- DESI Collaboration, Adame, A. G., Aguilar, J., et al. (2025). DESI 2024
  VI: Cosmological Constraints from the Measurements of Baryon Acoustic
  Oscillations. *Journal of Cosmology and Astroparticle Physics*,
  **2025**(02), 021.
  DOI: [10.1088/1475-7516/2025/02/021](https://doi.org/10.1088/1475-7516/2025/02/021).
- Riess, A. G., Yuan, W., Macri, L. M., et al. (2022). A Comprehensive
  Measurement of the Local Value of the Hubble Constant with 1 km/s/Mpc
  Uncertainty from the Hubble Space Telescope and the SH0ES Team.
  *The Astrophysical Journal Letters*, **934**, L7.
  DOI: [10.3847/2041-8213/ac5c5b](https://doi.org/10.3847/2041-8213/ac5c5b).
- Fixsen, D. J. (2009). The Temperature of the Cosmic Microwave
  Background. *The Astrophysical Journal*, **707**, 916–920.
  DOI: [10.1088/0004-637X/707/2/916](https://doi.org/10.1088/0004-637X/707/2/916).

### Haug's composite-constant framework and the horizon relation

- Haug, E. G. (2020). Collision Space-Time: Unified Quantum Gravity.
  *Physics Essays*, **33**(1), 46–78.
  DOI: [10.4006/0836-1398-33.1.46](https://doi.org/10.4006/0836-1398-33.1.46).
- Haug, E. G. (2020). Finding the Planck Length Multiplied by the Speed
  of Light Without Any Knowledge of G, c, or ℏ, Using a Newton Force
  Spring. *Journal of Physics Communications*, **4**(7), 075001.
  DOI: [10.1088/2399-6528/ab9dd7](https://doi.org/10.1088/2399-6528/ab9dd7).
- Haug, E. G. (2022). Progress in the Composite View of the Newton
  Gravitational Constant and Its Link to the Planck Scale. *Universe*,
  **8**(9), 454.
  DOI: [10.3390/universe8090454](https://doi.org/10.3390/universe8090454).
- Haug, E. G. (2023). Not Relying on the Newton Gravitational Constant
  Gives More Accurate Gravitational Predictions. *Journal of Applied
  Mathematics and Physics*, **11**, 3124–3158.
  DOI: [10.4236/jamp.2023.1110205](https://doi.org/10.4236/jamp.2023.1110205).
- Haug, E. G., & Tatum, E. T. (2024). The Hawking Hubble Temperature as
  a Minimum Temperature, the Planck Temperature as a Maximum Temperature
  and the CMB Temperature as Their Geometric Mean Temperature.
  *Journal of Applied Mathematics and Physics*, **12**, 3328–3340.
  DOI: [10.4236/jamp.2024.1210198](https://doi.org/10.4236/jamp.2024.1210198).
- Haug, E. G. (2024). CMB, Hawking, Planck, and Hubble Scale Relations
  Consistent with Recent Quantization of General Relativity Theory.
  *International Journal of Theoretical Physics*, **63**, 57.
  DOI: [10.1007/s10773-024-05570-6](https://doi.org/10.1007/s10773-024-05570-6).
- Haug, E. G. (2026). Thermodynamic Cosmology from the CMB Temperature
  and the Planck Temperature and the Relation to Quantum Cosmology.
  *Astronomy*, **5**(4), 19.
  DOI: [10.3390/astronomy5040019](https://doi.org/10.3390/astronomy5040019).