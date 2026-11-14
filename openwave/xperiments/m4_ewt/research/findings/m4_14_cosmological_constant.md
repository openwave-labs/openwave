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
   matches Planck. The unit `X = (2π)²` is part of the phase-unit
   ansatz; the test is conditional on it.
6. **EWT trinity** `α_geom`, `ℏ_geom`, `G_geom`, `λ_l` from the engine.
7. **Planck temperature** `T_p = (1/k_B)·√(ℏ·c⁵/G)`.
8. **Horizon radius** `R_H = λ_l·(T_p/T_cmb)²/H`. This is the Haug and
   Tatum relation `T_cmb = (T_p/8π)·√(2λ_l/R_H)`, with its factor
   `32π²` read as `8·(2π)²`.
9. **Hubble constant** `H_0 = c/R_H`.
10. **de Sitter Λ** `Λ_dS = 3H_0²/c²`.
11. **Vacuum fraction** `Ω_vac = η_BCC·C_gap` (composition rule =
    model assumption).
12. **Observed Λ** `Λ_vac = Ω_vac·Λ_dS`.

Steps 1 to 4 do not enter `H_0` directly: `H = 8·(2π)²` and
`O_μ = 1/(2π)²` are fixed numbers, so `ε_M` reaches `H_0` only through
the trinity of step 6.

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
| `T_cmb` from the inverse relation (an identity: it returns the input for any `T_cmb`) | 2.7255000000 K |
| `H_0^EWT` | 66.911313 km/s/Mpc |

Tension: Planck 2018 base (−0.83σ), Planck 2018 with BAO (−1.78σ),
DESI 2024 (−2.59σ), SH0ES 2022 (−5.89σ).

The same relation with CODATA `ℏ` and `G`, and no lattice quantity,
gives `H_0 = 66.8954 km/s/Mpc`. The −0.83σ is therefore the agreement
of the Haug and Tatum relation; the lattice adds the reading of `32π²`
as `8·(2π)²` and the `+0.0238%` carried by the trinity.

### Lattice discrimination (Part 5, conditional on phase-unit ansatz)

| N | lattice | H₀ [km/s/Mpc] | tension |
|---|---|---|---|
| 4 | tetrahedron | 33.4557 | −62.79σ |
| 6 | simple cubic | 50.1835 | −31.81σ |
| **8** | **BCC** | **66.9113** | **−0.83σ** |
| 12 | FCC | 100.3670 | +61.12σ |
| 16 | hypercubic | 133.8226 | +123.08σ |

The unit `X = (2π)²` is part of the phase-unit ansatz; the test is
conditional on it. It discriminates the product `N·X = 32π²`, not `N`
alone: `N = 16` with unit `2π²` gives the same 66.9113.

### Λ and Ω_vac (Parts 10–12)

| Quantity | Value |
|---|---|
| `Λ_dS` | 1.569560×10⁻⁵² m⁻² |
| `Λ_dS/Λ_Planck` | 1.441102 |
| `η_BCC` | 0.6801747616 |
| `C_gap` | 1.0140759358 |
| `Ω_vac` | 0.6897488579 |
| `Λ_vac` | 1.082602×10⁻⁵² m⁻² |
| `Λ_vac/Λ_Planck` | 0.993998 (rel. diff. `−0.600%`) |

`Ω_vac` vs Planck 2018: base (TT,TE,EE+lowE+lensing) `+0.737%`
(`+0.69σ`), with BAO `+0.123%` (`+0.15σ`).

### One Planck 2018 set at a time

The base set is the reference for all three quantities. The set with
BAO is reported for completeness.

| Planck 2018 set | `H_0` | `Ω_vac` | `Λ_vac` |
|---|---|---|---|
| base (TT,TE,EE+lowE+lensing): 67.36 ± 0.54, 0.6847 ± 0.0073 | −0.83σ | +0.69σ | −0.60% |
| with BAO: 67.66 ± 0.42, 0.6889 ± 0.0056 | −1.78σ | +0.15σ | −2.08% |

`Λ_Planck` is `3·Ω_Λ·H_0²/c²` evaluated on each set (1.0891×10⁻⁵² and
1.1056×10⁻⁵² m⁻²); Planck does not publish it separately. The same
relation builds `Λ_vac` from `Ω_vac` and `H_0`, so the `Λ` figure
combines those two comparisons and is not a third test. On the set
with BAO, `Λ` at −2.08% is a tension, not a victory.

## Interpretation

### Criterion connection: cosmic acceleration / dark energy

The criterion `Cosmology: Λ / cosmic acceleration` is the framework's
target, not a side effect. The chain provides quantitative support in
three senses:

1. **Observation.** `Ω_Λ ≈ 0.69` and `Λ ≈ 1.09×10⁻⁵² m⁻²` are
   observationally established facts. The chain reproduces both on the
   Planck 2018 base set under the two stated identifications:
   `Ω_vac = 0.6897` at `+0.69σ`, `Λ_vac = 1.0826×10⁻⁵² m⁻²` at
   `−0.60%`. On the set with BAO they read `+0.15σ` and `−2.08%`.
2. **Mechanism.** The acceleration is not attributed to a mysterious
   substance but to the effective vacuum response of the BCC lattice:
   `Ω_vac = η_BCC·C_gap`. `η_BCC` is pure crystallography. `C_gap`
   is the pure π⁶ operator the engine assigns to neutral vertices;
   using it here is part of the identification. The engine's `M_W`
   prediction at `0.012%` uses the mixed operator `C_gap_mixed`, so
   `M_W` does not test this ingredient.
3. **Horizon relation.** The horizon relation is the Haug and Tatum
   relation, with its factor `32π²` read as `8·(2π)²`.

The framework does **not** postulate dark energy as a substance. It
identifies the quantity that Λ-CDM calls the dark-energy fraction `Ω_Λ`
with the effective vacuum response of the BCC lattice. What is **not**
derived is why the composition rule is the product `η_BCC·C_gap` rather
than another combination (sum, or `η_BCC` alone). This is the principal
open objective.

### What the chain earns

- `Λ_vac` reproduces the observed cosmological constant to `0.60%` on
  the Planck 2018 base set under the two stated identifications (the
  horizon factor and the `Ω_vac` product), with **one cosmological
  anchor** (`T_cmb`). On the set with BAO it is `−2.08%`: a tension,
  not a victory.
- `H_0 = 66.911 km/s/Mpc` sits on the Planck side of the Hubble tension
  (`−0.83σ` on the base set). The same relation with CODATA `ℏ` and `G`
  and no lattice quantity gives `66.8954`.
- `Ω_vac = 0.6897` sits at `+0.69σ` of the Planck 2018 base value
  `0.6847 ± 0.0073` (`+0.15σ` of `0.6889 ± 0.0056` with BAO).
- The relation `T_cmb = (T_p/8π)√(2λ_l/R_H)` is the Haug and Tatum
  relation, with its factor `32π²` read as `8·(2π)²`. The `T_cmb` row
  in the results inverts the relation `R_H` was computed from: an
  identity, not a test.
- Lattice test **conditionally** excludes `N ≠ 8` at tens of σ, the
  condition being the unit `X = (2π)²`.

### Where the gaps are (named, per the criterion)

1. **Horizon factor `H = 8·(2π)²` is a phase-unit ansatz.** Each BCC
   direction contributes one `(2π)²`; a first-principles derivation
   is open. The lattice test is a *conditional*
   falsification, not a derivation that other lattices are impossible.
   The unit `X = (2π)²` is part of the phase-unit ansatz; the test is
   conditional on it.
2. **`Ω_vac = η_BCC·C_gap` composition rule is a model assumption.**
   `η_BCC` is crystallography and `C_gap` is the engine's pure π⁶
   neutral-vertex operator; the choice of that operator and the
   *product* are identified, not derived from a horizon-balance
   argument. The data do not choose between the candidates:

   | Candidate | `Ω` | Planck 2018 base, 0.6847 ± 0.0073 | with BAO, 0.6889 ± 0.0056 |
   |---|---|---|---|
   | `η_BCC` alone | 0.6802 | −0.62σ | −1.56σ |
   | `η_BCC·C_gap` (this chain) | 0.6897 | +0.69σ | +0.15σ |
   | `η_BCC·C_gap_mixed` | 0.6874 | +0.37σ | −0.26σ |

   `C_gap − 1` is 1.41%, and the Planck 1σ on `Ω_Λ` is 1.07% (base) or
   0.81% (with BAO), so the factor that separates the product from
   `η_BCC` alone is about the size of the measurement error.
3. **Relaxation dynamics `a(t)` is not delivered.** The chain predicts the
   present-day snapshot values but not the cosmic history.

### Falsifications named

- Any future measurement of `Ω_Λ` or `Λ` more than ~1% from the
  predicted values (`Ω_vac = 0.6897`, `Λ_vac = 1.0826×10⁻⁵² m⁻²`)
  rules out the mechanism. The falsifier is bound to the Planck 2018
  base set, where `Λ` sits at `−0.60%`, inside the bound. On the set
  with BAO `Λ` is `−2.08%`, stated as a tension. The Planck 1σ on
  `Ω_Λ` is 1.07% (base) and 0.81% (with BAO).
- Under the phase-unit ansatz, a lattice other than BCC would give
  `H_0` off by tens of σ.
- The relaxation picture predicts a slow time variation of `Λ` at
  cosmological redshifts once the EMC-wall profile is anchored;
  currently not falsifiable.

**`G_geom` enters `H_0` as a square root.** `G_geom` is itself an
expression in the anchors and `α_geom`; `H_0 ∝ G_geom^(1/2)`, so its
0.048% residual enters `H_0` at 0.024% and `Λ` at 0.048%. With CODATA
`ℏ` and `G` the same relation gives `H_0 = 66.8954 km/s/Mpc`.

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

## Reproduction

```bash
cd openwave/xperiments/m4_ewt/research/scripts
python3 m4_14_ewt_cosmological_constant.py
```

Needs `m4_7_ewt_emergence_engine.py` in the same directory. All numbers
in this finding are taken from the script's console output; the script
does not write data files (it is a diagnostic chain, not a sweep).

## Artifacts

- `research/scripts/m4_14_ewt_cosmological_constant.py`

## Reference

Enhanced EWT manuscript, version 5.3.1 (2026-11-13), § "The Cosmological
Constant from the BCC Lattice" (label `sec:cosmo_summary`). DOI:
[10.5281/zenodo.24207609](https://doi.org/10.5281/zenodo.24207609).

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
