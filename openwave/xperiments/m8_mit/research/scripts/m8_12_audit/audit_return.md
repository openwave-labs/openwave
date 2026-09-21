# AUDIT — solver_a vs solver_b on the spin-3 quartic r̂₆

I did not import, call or copy any solver script to produce a number. Every value below marked **mine** comes from a script I wrote in this room, built from the definitions in `worklist.md`:

- `aud_core.py`: CG coefficients by lowering |6 6⟩ = v₃⊗v₃ (with an exact J² = 42 check), Θ, ρ₆, N, and the exact gradient and Hessian from the sesquilinear form.
- `aud_acoords.py`: rational weighted coordinates, checked against `aud_core` by an exact polynomial identity.
- Item scripts: `a0 a1 a2 a4 a5 a5num a5b a10 a11 a12 a8min a8max a8chk a_locus a_perturb a_inspect a_consistency`.

I read solver scripts only to understand and grade them (see the manifest).

**Exactness policy.** "Exact" means sympy / `Fraction` arithmetic with no numerical identification. Every exact value I assert also has a high-precision route, with its precision and residual stated. Every PASS-type check I rely on was shown to fail on a planted defect (§5).

---

## 1. The controls (brief step 1)

### Item 11 — u = (v₃ + v₁ − v₋₂)/√3 (`a11.py`)

| quantity | solver_a | solver_b | **mine** (exact; 50-digit check) |
|---|---|---|---|
| 924·r̂₆ | 204 | 204 | **204** |
| ‖tangential grad‖ | 2√1002/297 | 2√1002/297 | **2√1002/297**, minpoly 29403X²−1336; `mp.diff` gradient at 50 digits agrees to all 25 printed digits |
| ‖M_u(iu)‖ | 2√1002/297 | same | **√(1336/29403) = 2√1002/297** |
| ‖M_u(−iJ_x u)‖ | √3·√(5560+416√15)/297 | √((5560+416√15)/29403) | **√((5560+416√15)/29403)**; minpoly 864536409X⁴−326961360X²+28317760 (the same number in both forms) |
| ‖M_u(−iJ_y u)‖ | √3·√(5560−416√15)/297 | √((5560−416√15)/29403) | **√((5560−416√15)/29403)**, same minpoly |
| ‖M_u(−iJ_z u)‖ | 4√921/297 | 4√921/297 | **√(4912/29403) = 4√921/297** |

Verdict: **AGREE**, both solvers, on every number, including the J_x/J_y assignment of the ± branch. I also checked exactly that M_u(iu) = i·g_u, the identity both solvers use.

- *What would have exposed an error:* any residual of 0 here would show a check that cannot fail. A value ‖M_u(iu)‖ ≠ ‖g_u‖ would show a wrong Hessian formula.
- *What I found:* the residuals are 0.21–0.49, non-zero as they must be. At the ten orbits the same residuals are exactly 0 (item 5).

### Item 12 — two routes for H_u (`a12.py`)

| case | exact value (route 1 = exact s²-Taylor of r̂₆(u cos s + e sin s); route 2 = eᵀ(Hess N − 4N·I)e) | route1 − route2 (exact) | numerical residuals vs exact |
|---|---|---|---|
| **mine**: orbit A*, e = P_N w, w = (3,−1,4,1,−5,9,2,−6,5,3,−5,8,9,7) | ≈ 0.04339833887995523343256; minpoly 32692707233713827799475625X⁴ + 20753120733073768580112000X³ − 2683300421416336676697600X² − 465625029456430368645120X + 23448853626230130282496 | **0** | route 1 by `mp.diff`: 2.1e-50 (50 d), 1.7e-80 (80 d). Route 2 by exact quartic interpolation in floating point: 3.7e-47, 4.8e-77. Largest route disagreement: 3.7e-47 (50 d), 4.8e-77 (80 d) |
| solver_a's e₂ at F* (re-computed by me) | (31368148625200 + 1625311542060√2 − 2156748448066√15 + 4831929869904√30)/43841638707405 ≈ 1.181050864353138882016 | **0** | 4.3e-50 / 1.7e-49 (50 d); 3.4e-80 / 6.8e-80 (80 d) |
| solver_b's direction at Wmin (re-computed by me) | ≈ −0.106381984598757117108681; exact form and denominator identical to solver_b's | **0** | 0.0 / 5.7e-48 (50 d); 0.0 / 8.2e-78 (80 d) |

Verdict: **AGREE** for both solvers. My exact value, minimal polynomial and 1.18105… match solver_a. My exact value (common factor 43 cancelled) and −0.1063819845987571171… match solver_b. The two solvers chose different orbits and directions, and the item allows that.

- *What would have exposed an error:* a non-zero exact difference between the routes, or a numerical difference above the working precision.
- *Planted defect:* a direction e + u, which is not tangent, gives route1 − route2 = −2.057 (it fires). A formula without −4N is already caught by the annihilation check at all 10/10 orbits (`a5.py`).

### The locus span{v₃, v₋₁} (`a_locus.py`, `a2.py`)

- **Restriction.** In the chart v₃ + z v₋₁, which omits v₋₁: r̂₆ = (225t² + 450t + 1)/(924(1+t)²), with t = |z|². The restriction is radial: this is exact, and I checked it against the full complex z.
- **Derivative.** dr̂₆/dt = 16/(33(1+t)³) > 0 for every t ≥ 0.
- **Interior critical points: none.** r̂₆ increases strictly from 1/924 at v₃ to 75/308 at v₋₁. The gradient numerators are 224x and 224y, whose only zero is z = 0.
- **Linear coefficient of the restriction (in t = |z|²):**
  - at the v₃ end: **16/33** (the numerator's own linear coefficient is 450/924);
  - at the v₋₁ end (chart v₋₁ + w v₃): numerator 225 + 450s + s², Taylor series 75/308 + **0**·s − (8/33)s². The linear coefficient is exactly **0**, because 450 = 2·225.
- **What would have exposed an error:**
  - A zero of dr̂₆/dt for t > 0 would be a critical circle that both solvers missed. Both claim class B (≅ this locus by R_y(π)) has none, and that holds.
  - A non-zero coefficient at the v₋₁ end would contradict the λ² factor at v1. It is 0, which matches the kernel.
  - As a cross-check against item 5: the v₃-end coefficient 16/33 gives the second derivative 2·16/33 = 32/33 along a unit direction. That is a root of v₃'s exact characteristic polynomial (33λ−32)². It fits.

Neither solver reports this locus under this name. Both report the equivalent class B / L13 = span{v₁, v₋₃}, with the same numbers in the swapped chart: t² + 450t + 225, no circle, a degenerate v1. **AGREE.**

---

## 2. Per-item tables (brief steps 2 and 4)

Verdicts:
- **AGREE**: both solvers agree with my value.
- **DISAGREE**: the solvers differ; I say who is right, or that the difference is a reading.
- **DEFECT**: a demonstrable error.

| item | solver_a | solver_b | mine | verdict |
|---|---|---|---|---|
| 0 | ⟨33;3−3\|60⟩ = 1/√924; ΘΘ = 1; 1977/8624; 6329/33264 | same (in RETURN.md and results.json) | same (`a0.py`). CG by lowering; J² = 42 residual exactly 0; planted sign flip gives residual 19.7. mp 50 d: 3.3e-52 and 0; 80 d: 2.6e-82 and 0 | **AGREE** on values. **Artifact DEFECT in solver_b**: `solver_b/out/item0.json` holds 15207/60368 and 10889/33264 (see §4, D-B1) |
| 1 | 6 line classes, 7 plane classes; other dims 0, 3, 4, 7; stabilizers C3, C4, C4, C5, C6, D2, D3 (read as "acts by a character") | same 6 + 7 classes, same dims; setwise stabilizers; U, W, hex, oct stabilizers not determined | census over C₁…C₁₂, D₂…D₁₂, T, O, I (`a1.py`): 68 dim-1, 12 dim-2, dim 3 = C2 trivial and C3 trivial, dim 4 = C2 nontrivial, dim 7 = C1. The dim-1 spaces fall into 6 invariant values (924r̂₆ = 1, 36, 225, 288, 400, 463); the dim-2 spaces into 7 classes; the three D₂ planes are joined by the exact R₍₁₁₁₎(2π/3) (`a4.py`) | **AGREE** on the classification. The stabilizer column differs by reading (see step 2) |
| 2 | restrictions and critical sets as tabulated | same (charts with non-unit b, rescaled) | identical after rescaling (`a2.py`, exact factorisations and lex Gröbner bases) | **AGREE** |
| 3 | six lines critical; values 1/924, 3/77, 75/308, 100/231, 24/77, 463/924 | same | exact tangential gradient = 0 at all six (`a4.py`) | **AGREE** |
| 4 | 10 orbits; exhibited rotations; v1 ≠ F* by r₁; A* ≠ D* by tr ρ₂³, spectra, stabilizers | 10 orbits; v1 ≠ Umin by \|⟨J⟩\|²; L12circle ≠ L23circle by I₃, stabilizer, spectra | A's rotations cat→F point and xyz→G point verified exactly, and a wrong rotation is rejected. ⟨J⟩ = (0,0,1) at v1 vs 0 at F*. A* vs D* separated by different exact characteristic polynomials (item 5); ⟨J⟩ = ±(0,0,2/5) | **AGREE** |
| 5 | dim N, signatures, 10 characteristic polynomials | identical polynomials (monic form) | all 10 polynomials identical, exact (`a5.py`, rational weighted coordinates). A 50/80-digit c-coordinate eigenvalue route agrees to ≤1.6e-50 / ≤1.7e-80 (`a5num.py`). Annihilation of O_u is exactly 0 at all 10 orbits; planted no-4N fires 10/10 | **AGREE** on values. **Artifact DEFECT in solver_b**: `out/item5.json` is the planted run (D-B2) |
| 5b | basis (iu, u_⊥, iu_⊥): A* −24/55; D* −104/55; F* diag(64/33, 10/11); G* diag(8/11, 920/473); supplementary xyz, cat | basis (iu, ∂ₓû, ∂ᵧû): Wmin diag(920/473, 184/473); Umin diag(8/11, 10/11); L12 −24/55; L23 −104/55; supplementary hex, oct | **both sets reproduced exactly** (`a5b.py`), including the projection factors 23/43, 3/8, 5/8, 5/9, the O_u coefficients of the null tangents, zero off-diagonals and identity Grams | **DISAGREE by reading only**: different tangent bases, both correct (see step 2) |
| 6 | indices n₋/n₊; extrema v3 and cat | same | follows from the item-5 signatures I verified | **AGREE** |
| 7 | kernel span_ℝ{v₋₃, iv₋₃} = T_{v1}B | same (L13) | M_u v₋₃ = M_u(iv₋₃) = 0 exactly as full vectors (control: M_u v₋₂ ≠ 0). Exact λ² factor. The numerical eigenvalue scales with precision: 1.3e-51 at 50 d, 1.0e-81 at 80 d | **AGREE** |
| 8 min | 1/924 at v3, proved (P_L identity) | 1/924 at v3, proved (K spectrum; Fock inequality) | generalised spectrum of N against \|u\|⁴ on degree-2 monomials: {1/924 ×13, 13/154 ×9, 65/84 ×5, 13/7 ×1}, exact (`a8min.py`). This gives r̂₆ ≥ 1/924, attained at v3 | **AGREE** |
| 8 max | 463/924 at cat, **proved** by an exact degree-6 certificate | 463/924 at hex, **not proved** (search only; says so) | **proved independently**: my own equivariant Hermitian-SOS certificate, λ\|u\|⁶ − N\|u\|² with λ = 463/924, exact identity and all blocks PD (`a8max.py`). Planted: λ = 461/924 gives an inconsistent system; a Gram entry perturbed by 1e-6 breaks the identity | values **AGREE**; proof status **DISAGREE**: solver_a is right that a proof exists. solver_b is honest that it has none |
| 9 | all zeros computed; reasons | all zeros computed; reasons; two arithmetic zeros flagged | spot-checked: N∘Θ = N, N∘conj = N, infinitesimal invariance under i, −iJ_{x,y,z} (exact identities; a planted wrong Θ breaks invariance); the 5b off-diagonal zeros exact; v1 kernel exact | **AGREE** (B's stored item-5 residuals: D-B2) |
| 10 | Q-values, H^Q = wH, 5b × w | same logic; 5b × w in its own basis | all Q-values and every 5b×w entry of both solvers reproduced; M^Q − wM = 8(G₀u)(G₀u)ᵀ exactly at F* (`a10.py`) | **AGREE** (5b differences inherited from the basis reading) |
| 11 | see §1 | see §1 | see §1 | **AGREE** |
| 12 | F*: e₁ 64/33, e₂ ≈ 1.18105 | Wmin: ≈ −0.10638 | both re-verified exactly; own choice at A* | **AGREE** (different legitimate choices) |

### Adjudication of the disagreements (step 2)

1. **Item 5b numbers.** solver_a reports F* as diag(64/33, 10/11); solver_b reports Umin, the same orbit, as diag(8/11, 10/11). The same holds for G*/Wmin, cat∈F/hex∈U and xyz∈G/oct∈W. **Both are correct for the basis each used.**
   - A's tangents u_⊥ and iu_⊥ are Hermitian-orthogonal to u, and exactly orthogonal to O_u when they are non-null. I computed |P_N t|²/|t|² = 1.
   - B's chart derivatives ∂ₓû, ∂ᵧû carry an iu (phase) component. I computed the same factors B reports: 3/8 for Umin ∂ₓû, 23/43 for Wmin ∂ᵧû, 5/8 for hex∈U, 5/9 for oct∈W.
   - Because M_u kills O_u, H on a unit unprojected tangent equals factor × (H on the unit projection). For example 184/473 = (23/43)·(8/11) and 8/11 = (3/8)·(64/33); all of these are reproduced in `a5b.py`.
   - The item fixes neither the basis nor whether to normalize before or after projecting. That is why it asks for the Gram matrix. This is a genuine underdetermination, and each solver stated its normalization.
   - The null-tangent coefficients also differ, (1/3, 5/6) for A versus (2/3, 2/3) for B, because iu_⊥ ≠ ∂ᵧû. Both are verified exactly.
2. **Item 8 maximum.** The values agree. solver_a's proof stands (§3 grades). My independent certificate confirms the claim, and B's "not proved" is a correct statement of its own work.
3. **Item 1 stabilizers.** A means "the largest subgroup acting on W by a character" (B-class = C4). B means the setwise stabilizer (L22 = O(2)), and B left U, W, hex and oct undetermined. These are different readings. Neither is wrong, and the class lists coincide. B separated U from W exactly without the stabilizers, and so did I: the exact r̂₆-ranges [75/308, 463/924] and [200/903, 463/924] differ.
4. **Item 12.** Different orbit and direction choices. Both verified.

**Agreed values I spot-checked, by my own route:**
- items 0, 2 (all seven restrictions and critical sets), 3, 4 (rotations, ⟨J⟩ separations), 5 (all ten polynomials, two routes), 5b, 7, 8 (both ends), 10, 11, 12;
- the interior critical points: exact gradient 0 at A*, D*, F*, G*; after a 1e-3 perturbation |grad|² = 3.0e-8, 8.9e-7, 1.1e-7, 4.9e-7 (`a_perturb.py`).

---

## 3. Argument grades (step 3)

| item | solver_a | solver_b |
|---|---|---|
| 1 | **SOUND** | **SOUND** |
| 2 | **SOUND** (elimination) | **SOUND** (elimination) |
| 3 | **SOUND** | **SOUND** |
| 5 | **SOUND** | **SOUND** |
| 7 | **SOUND** | **SOUND** |
| 8 | **SOUND** (both ends) | min **SOUND**; max **INCOMPLETE** (search) |
| 9 | **SOUND** | **SOUND** |

Reasons:

- **Item 1.**
  - *A.* The subgroup classification is named as standard and recalled; every character is found by brute force; n > 12 is handled by an argument (|m−m'| ≤ 6 < n), which I re-derived; the continuous groups are handled by argument; the omission consequences are right. I confirmed that F comes only from D₂, G only from D₃ and D only from C₅ (`a1.py`). Distinctness is exact, via the r̂₆-ranges, which I recomputed.
  - *B.* Same hypotheses, stated as (H1)–(H4). The one numerical identification (rotation angles as rationals with denominator ≤ 60) is declared, cross-checked at 40 digits and by SVD, and backed by exact null spaces. Distinctness uses exact invariant polynomials. I confirmed the first family numerically: spectra {1,4}, {1,9}, {4,4}, {4,9}, {9,9}, {0,0}.
- **Item 2 — completeness on the lines.** Both solvers use an **elimination**, not a solver list, and I worked the cases myself (`a2.py`).
  - For A–E / L12–L33, R_z-invariance makes the restriction radial. ∇f = 2F′(t)(x,y) and F′'s numerator is linear in t: −21(4t−1), −224t, −168(t−1), −35(13t−12), −924(t−1) in my normalisation. The cases z = 0 and F′(t) = 0 with t > 0 are exhaustive.
  - For F/U and G/W, the gradient numerators are x·p and y·q with p − q ∝ (1+t) ≠ 0, so xy ≠ 0 is impossible. The axis cases give 5x² = 3, 3y² = 5 (F) and 5x² = 4, 23y² = 20 (G). My lex Gröbner bases, {x(5x²−11y²−3), y(13x²−3y²+5), xy(2y²+1), y(2y²+1)(3y²−5)} and the G analogue, match B's after the chart rescaling x → √2x.
  - The omitted points are checked in the second chart: the gradient there is 0 exactly.
  - *What a solver alone does:* `sympy.solve` applied to the exact Gröbner bases returned **no** circle points for A, C, D, E and **no real point at all for class B**. It lost the degenerate v1 (`a2.py` output). That shows concretely what both solvers warned about.
- **Item 3.** Both give symmetric criticality with hypotheses listed (isometric action, invariance, a one-dimensional fixed space, degree-0 homogeneity plus phase). Invariance is verified: A by symbolic R_z(θ) plus R₍₁₁₁₎ plus the closed-subgroup argument; B by the infinitesimal identities plus connectedness. I re-verified the infinitesimal identities exactly.
- **Item 5.**
  - Both proofs of H_u(e,e) = eᵀ(Hess N − 4N·I)e are correct. I re-derived D''(0) = 4(‖e‖²−1), which makes the formula valid for non-unit e ∈ T_u, and B's route via ∇f·u = 0.
  - Both proofs of M_u X u = X g_u are correct: differentiate ∇N(w)·Xw ≡ 0 and use Xᵀ = −X.
  - Both place the degree-0 homogeneity correctly and exhibit the failing point (item 11).
- **Item 7.** Both identify the kernel exactly with T_{v1}(B/L13) and correctly separate "the symmetry explains multiplicity 2" from "an arithmetic identity explains the zero". I checked B's weight argument (couplings need m + m′ = 2, and −3 has no partner). The genuine-vs-numerical criteria are sound, and my precision scaling confirms them.
- **Item 8 — the two ends.**
  - *A, min:* an exact identity plus squared norms. Hypotheses M1–M4 are verified, and the equivalent spectrum is confirmed by me.
  - *A, max:* an argument, not a search. Hypotheses and my attempts to break each:
    - **X1** (exact identity): A's cubic polynomials are not stored, so I could not re-expand A's own identity. I read `sos.py`: exact `Fraction` dict arithmetic, with the identity declared only when the difference dict is empty. I could not break it. Stronger: I built an independent certificate of the same form in different coordinates (b_m = c_m/√C(6,3+m)), and it verifies exactly. So the claim holds whatever A's code did.
    - **X2** (PD blocks): I checked all 10 of A's stored rational blocks with my own code. All are symmetric positive definite.
    - **X3** (real coefficients, so each term is a Hermitian square): true by construction; `swap` implements conjugation.
    - **X4** (a-coordinates): I verified N_a(a(c)) = N(c) exactly (`a5.py`).
    - **X5** (r̂₆(cat) = 463/924): verified.
    - The search is correctly labeled as evidence only.
  - *B, min:* SOUND. Argument 1 (the Sym² spectrum) is verified by me. I did not verify argument 2 (the Fock product inequality); it is not needed.
  - *B, max:* **INCOMPLETE**. It is a 400-start BFGS search, correctly described by B as establishing nothing beyond "≥ 463/924". Per the brief, a search is INCOMPLETE regardless of its answer. The answer happens to be right.
- **Item 9.**
  - Both list every zero as computed, with exact reasons. Symmetries used as reasons are shown to preserve r̂₆. I re-verified N∘Θ = N, N∘conj = N and rotation invariance. Arithmetic zeros are honestly labeled (the L13 identity 30+210+210 = 450; B's ⟨32;3−2\|20⟩ = 0).
  - B's stored `out/item5.json` shows non-zero annihilation residuals. That comes from the artifact defect D-B2, not from the argument; B's collected `results.json` records exact 0.

---

## 4. Demonstrated defects

**D-B1 (solver_b, artifact): `solver_b/out/item0.json` contains wrong r̂₆ values.**
- *Demonstration* (`a_consistency.py`): the file records r̂₆(2v₃+v₁−3v₋₂) = 15207/60368 and r̂₆(v₃+iv₀+2v₋₁) = 10889/33264.
- The correct values, by my exact route (`a0.py`, 50/80-digit agreement), are **1977/8624** and **6329/33264**. They are what B's RETURN.md and `results.json` state.
- *Cause* (read in `solver_b/run_all.py`, lines 13–14 and 61–73): `collect.py` is the last regular script, and the planted defects (`cg_sign` on item0, `hess_no4N` on item5, `theta_sign` on item9) run **after** it, writing to the same `out/` paths.

**D-B2 (solver_b, artifact): `solver_b/out/item5.json` is the output of the planted `hess_no4N` run.**
- *Demonstration:*
  - The file gives v3's characteristic polynomial as (77λ−152)²(77λ−75)²(231λ−925)²(231λ−64)²(231λ−43)², and its annihilation residuals as 1/231, √6/462, √6/462, 1/77.
  - The correct polynomial is (λ−4)²(11λ−3)²(11λ−2)²(33λ−65)²(33λ−32)² (`a5.py`; also B's `results.json`), with residuals 0.
  - 1/231 = 4N(v3) is exactly the residual M(iu) = 4N·iu produced by the missing −4N term.
- *Consequences:*
  - Anyone reading B's `out/` directory, or re-running `collect.py` alone, gets wrong numbers for items 0 and 5.
  - B's RETURN.md statement that "each script writes out/<item>.json, and collect.py merges these files into results.json" does not hold for the files left on disk.
  - B's mathematical claims are unaffected.
- *Related minor point:* B's RETURN says each script writes `out/<item>.json`, but `out/item9.json` does not exist and `results.json` has no item-9 section. B's item-9 results exist only in RETURN.md and the log.

**No mathematical defect was found in either solver.** Every value and argument I checked stands.

---

## 5. Claims I tried to break and could not

- **CG convention and the item-0 values.** My CG table, built by lowering, satisfies J² = 42 exactly. A planted sign flip gives residual 19.7.
- **All seven plane restrictions and their critical sets**, including the degenerate v1 and the omitted points.
- **Criticality.** Exact at the six lines and the four interior points; perturbations make the gradient non-zero.
- **Orbit identifications.** Exact rotations cat→F point and xyz→G point (A's), and R₍₁₁₁₎ joining the three D₂ planes. A wrong rotation (R_y(π)) is rejected.
- **The ten characteristic polynomials and signatures.** Two independent routes: rational weighted coordinates with a null-space basis (exact), and c-coordinates with an orthonormal basis (numeric). That the two bases agree confirms basis independence.
- **The v1 kernel.** Exact λ², exact annihilation of v₋₃ and iv₋₃, and a numerical eigenvalue that scales with precision rather than stalling.
- **Item 5b in both bases**, with exact projection factors, Gram matrices and off-diagonal zeros.
- **Item 10 scaling and arithmetic.**
- **Item 11.**
- **Item 12** in three instances (mine, A's, B's).
- **Both ends of item 8.** The maximum was attacked twice: λ = 461/924 makes the system inconsistent, and a perturbed Gram entry breaks the identity. Both fire, and the unperturbed certificate verifies.

**My planted defects** (each fired):
- CG sign flip (J² residual 19.7);
- no −4N (annihilation fails 10/10);
- non-tangent direction in item 12 (route gap −2.057);
- wrong Θ sign (infinitesimal J_x invariance fails);
- control M_u v₋₂ ≠ 0 in item 7;
- λ below the maximum (inconsistent);
- Gram entry +1e-6 (identity fails).

---

## 6. What I could not check, and why

- **solver_a's own degree-6 identity (X1) term by term.** The cubic polynomials c̃ are not stored in its output, and reproducing them would mean running or copying `sos.py`. I checked its stored Gram blocks (PD) and replaced the rest with my own independent certificate.
- **Whether orbits other than cat also attain 463/924.** Neither solver settled this; A says so explicitly. I did not solve the equality conditions of my certificate either.
- **Values I did not recompute:**
  - solver_a's r_L tables (items 1 and 4);
  - tr ρ₂³ = ∓132/42875;
  - the level-1 bound 11/21 and its optimality point;
  - solver_b's second invariant polynomials (λ−54)(λ−30) and (λ−24)(2λ−75)/2;
  - B's I₃ = ∓48/125 (consistent with my ⟨J⟩ = ±(0,0,2/5), but not computed by my code);
  - B's Fock-space argument 2;
  - B's own rotation words for the hex/oct identifications. I verified A's equivalent rotations, which establish the same orbit memberships.
- **Subgroups beyond order 60 and the continuous subgroups.** My census covers C₁₂/D₁₂/T/O/I numerically (float64 SVD, smallest non-zero singular value 0.518, threshold 1e-9). Larger n and the continuous groups rest on the |m−m'| ≤ 6 argument, which I re-derived rather than computed.
- **My own earlier step that was wrong.** The "radial" flag printed by `a2.py` is faulty (a bad substitution test prints False). I did not rely on it. Radiality is visible in the printed polynomials, which depend only on x²+y², and it is checked correctly for span{v₃, v₋₁} in `a_locus.py`.

---

## 7. Consulted-material manifest (relative paths)

**Read by me:**
- `brief.md`, `worklist.md`
- `solver_a/RETURN.md`, `solver_b/RETURN.md`
- `solver_a/sos.py`, `solver_a/item8.py`
- `solver_b/item12.py`, `solver_b/run_all.py`
- `solver_b/core.py`: only the lines containing `RV`, `IDX`, `MS =`, `from_real` or `to_real`, printed by a Python one-liner
- `solver_a/out/item8.json` (structure and the `max` fields)
- `solver_a/out/run_all.log`
- `solver_b/out_run_all.txt` (the SUMMARY portion)

**Loaded by my scripts and partly printed:**
- `solver_a/results.json`, `solver_b/results.json`
- every `out/*.json` of both solvers:
  - solver_a: `defects`, `item0`, `item1`, `item2`, `item3`, `item4`, `item5`, `item5b`, `item8`, `item9`, `item10`, `item11`, `item12`
  - solver_b: `item0`, `item1`, `item2`, `item3`, `item4`, `item5`, `item5b`, `item6_10`, `item7`, `item8`, `item11`, `item12`

**Seen only as names and sizes in a directory listing, not opened:**
- `py`, `.room.sb`, and all other files of `solver_a/` and `solver_b/` (their remaining scripts, `scratch/` files, `.pkl` and `.log` files)

**Loaded without my asking:**
- the Python standard library and sympy, mpmath, numpy and scipy, imported by `./py` from the interpreter's environment outside the room. I did not open those files.
- harness file-state notices.

**Refused action:**
- one shell command, a `./py -c` inline script with `#` after a newline chained with `;`, was denied by the permission layer and not performed. I redid it as the file `a8chk.py`.

**Nothing else.** Nothing outside the room was read deliberately, and no network was used.

**My files:**
- scripts: `aud_core.py`, `aud_acoords.py`, `a0.py`, `a1.py`, `a2.py`, `a4.py`, `a5.py`, `a5num.py`, `a5b.py`, `a8min.py`, `a8max.py`, `a8chk.py`, `a10.py`, `a11.py`, `a12.py`, `a_locus.py`, `a_perturb.py`, `a_inspect.py`, `a_consistency.py`
- outputs: `out_a5.pkl`, `out_a8max.json`
- reports: `AUDIT.md`, `audit.json`
