# Refutation of the solver's stage-1 and stage-2 answers

Nothing refuted a value: every solver answer checked (items 0-21 plus 19 continued) holds. Three things fall short of full confirmation:
- **Item 13, the non-uniqueness note:** true but understated; there are three independent relations among the r̂_K, not one.
- **Checks that cannot fail:** the downstream consistency checks for items 13, 14 and 18 cannot detect a wrong projector.
- **The CS phase:** no solver assertion pins it.

## What I ran

Scripts:
- `audit_ref_compare.py`: 81 exact comparisons of the solver's JSON against mine.
- `audit_ref_newchecks.py`: new routes where we had used the same method or where I had been silent.
- `audit_ref_item13_relations.py`: the complete set of linear relations among the ‖rho_K‖^2.
- `audit_ref_mutsetup.py`: builds the mutated copies of the solver's scripts.

Results are in `audit_ref_compare.json`, `audit_ref_newchecks.json` and `audit_ref_item13_relations.json`, and under the key `refutation` in `audit_results.json`.

The solver's scripts were run only from copies under `ref_runs/`, never inside `solver_copy/`, because they overwrite their own results JSON. Baseline copies reproduced its outputs (self-test, item 0, item 19, stage-2 roots).

Of the 81 comparisons, 80 agree exactly. The one exception is item 20's form "as given", explained below as a convention.

## Per-item verdicts

| Item | Verdict | Basis (mine vs solver, plus any new check) |
| --- | --- | --- |
| 0(a) | CONFIRMED | Both Theta v_2 = v_{-2}. |
| 0(b) | CONFIRMED | Both sqrt231/462. Mutation M2 below shows no solver assertion would have caught a sign error here. |
| 0(c) | CONFIRMED | Norms 1, order 120, perfect, identical element-order histograms, from independent exact closures. |
| 1 | CONFIRMED | Both use weight counting. New route: Weyl integration of the exact character (χ^3 + 3χ(g^2)χ + 2χ(g^3))/6 gives V_9+V_7+V_6+V_5+V_4+2V_3+V_1. |
| 2 | CONFIRMED | Both 1 and 4 by weights. New route: Weyl integration of χ_{Sym^2}·χ_3 gives 1 at J=8 and 4 at J=3. |
| 3 | CONFIRMED | -1/sqrt7 on both sides (different CG code: my Racah formula, its sympy `clebsch_gordan`). |
| 4 | CONFIRMED | lambda_m identical. New identity: lambda_m = p_6 \|\|rho_6(v_m)\|\|^2 with the solver's pairing constant p_6 = -sqrt91/13, verified exactly for all m. |
| 5 | CONFIRMED | B_2(v_3) = 0; rho_2(v_3) = -5 sqrt21/42 at N=0. |
| 6 | CONFIRMED | 1, 400, 288, 463; all four critical (both by exact gradients, with different code). |
| 7 | CONFIRMED | (1/7, 0, 0, 0, 6/11, 0, 24/77). |
| 8 | CONFIRMED | Same polynomial, s = 12/25, value 9/35, endpoints 3/77 and 1/924. Reading difference: the solver lists both rays with s = 12/25 (±sin t); I listed one. New exact check of its (-) ray: r̂_6 = 9/35, M_6(u) ∥ u (critical), ring azimuths ±36°, ±108°, 180°, i.e. the pentagon rotated by π/5 as it states. Not a disagreement: its reading is more complete than mine. |
| 9 | CONFIRMED | Identical exact critical sets (five in the chart plus [v_0]) and identical unnormalised set. Both used lex Groebner bases, so I added a non-Groebner route: a 500-start Newton census in two charts covering the whole projective line found exactly 6 critical rays, with r̂_6 = 0.221483942 (200/903, ×2), 0.311688312 (24/77, ×2), 0.501082251 (463/924) and 0.432900433 (100/231). Nothing is missing. |
| 10 | CONFIRMED | 1,0,0,0,0,0,1; the extras for K = 7..30 are identical. Its half-integer extras (all 0) hold by a one-line argument: -1 ∈ Gamma acts on V_j by (-1)^{2j} = -1. |
| 11 | CONFIRMED | 3 + 4, multiplicity 1 each, FS +1 both. The characters agree class by class when matched by Re g, so it is the same 3-dimensional irrep on both sides despite the different quaternion maps. New route: Burnside, dim span{D^3(g) : g ∈ Gamma} = 25 = 3^2 + 4^2 (exact rank over Q(sqrt5, i)). |
| 12 | CONFIRMED | (9/7, 0^5, 12/7) and (16/7, 0^5, 12/7). The routes differ: my exact Casimir projectors in the monomial basis versus its sympy CG. |
| 13 | PARTIAL | **Holds:** under the Peter-Weyl reading, w_0 = 7, w_6 = 28/39 and 21/52, w_6/w_0 = 12/(13d^2), N = 143 d^2, all identical to mine. **Does not hold as stated:** its underdetermination note says w_K is fixed "only up to adding one common constant". The kernel is 3-dimensional (see the item 13 witness below). |
| 14 | CONFIRMED | Coefficients identical (-d/sqrt7 on M_0, -12/(d sqrt91) on M_6, 0 otherwise). Its c_K = -sqrt(7/(2K+1)) \|\|R_K\|\|^2 differs from my (-1)^{K+1} sqrt(...) form only at odd K, where R_K = 0. Both flag rank{M_0..M_6} = 4. |
| 15 | CONFIRMED | All 10 shared rays have identical point multisets (to 1e-9), and the same shape identifications (sextuple point, two triple points, octahedron, hexagon, pentagonal pyramid, trigonal-antiprism octahedra, D3h prisms). |
| 16 | CONFIRMED | Identical norm tables at every level; ranks 0..2j carried at every level. |
| 17 | CONFIRMED | Q_3 = 1 + (28/39) r̂_6, Q_4 = 1 + (21/52) r̂_6; the same critical sets, by the same affine argument. |
| 18 | CONFIRMED | beta(B=1) identical at the four item-6 rays. At its three extra rays (item 8 and two item-9 rays), recomputing 1 + w_6 r̂_6 with my slopes reproduces its values. Difference coefficient 49/156 on both sides. Its extra unit-fibre-norm variant is a second normalisation, not a disagreement. |
| 19 | CONFIRMED | Same characterisation (TR-invariant rays), same isotypic reading. Normalisation bridge, verified exactly: 720^3 · 504^2 · 286 / 16! = 1296. So its identity ι_8^{-1} J = 504 sqrt286 C_0 and my ratio \|\|C_iso\|\|^2/\|\|J\|\|^2 = 1/1296 are the same fact; its \|\|Pi_8 x\|\| = \|\|C_0\|\| matches my \|\|C_iso(v_3)\|\|^2 = 1/1092 = (sqrt273/546)^2. |
| 19 (continued) | CONFIRMED | sqrt273/1092 on both sides; L_8 = -(1/2) C_0 (its identity) agrees with my \|\|L_8\|\|^2/\|\|C_iso\|\|^2 = 1/4. |
| 20 | CONFIRMED | 12 simple roots, a regular icosahedron, injective. The forms as printed differ; that is a convention, not a disagreement (see below). Its exact rank 7 came from sympy `rank(simplify=True)`; that tool passed my planted-dependency stress tests, and my exact rank over Q(sqrt5, i) also gives 7. |
| 21 | CONFIRMED | Same invariance condition (real type, FS +1), same left factor (B_J = M_J(u (Theta u)^†), re-verified with my code), same surviving J = {0, 6}. After converting N_J bases (my N_0 = -(1/sqrt7) f_0 and N_6 = -(sqrt91/7) f_6, where its N_J = f_J), the coefficients agree exactly in both sectors. rank{M_0, M_6, N_0, N_6} = 4, so the spans do not coincide. |

### Item 13 witness (why the solver's note is only PARTIAL)

The seven quartics \|\|rho_K(u)\|\|^2 span a 4-dimensional space: singular values 5.73, 5.28, 4.68, 3.00, then under 1e-80 at 80 digits. That is forced: SU(2)-invariant Hermitian quartics correspond to End_SU(2)(Sym^2 V_3), of dimension Σ mult^2 = 4 because Sym^2 V_3 = V_6+V_4+V_2+V_0.

Three independent relations, each verified exactly as a polynomial identity in (u, conj u):

```text
-42/11 r0 - 14/11 r1 + 18/11 r2 + 27/11 r3 + r4 = 0
r1 - 2 r3 + r5 = 0
-24/11 r0 + 14/11 r1 - 7/11 r2 + 6/11 r3 + r6 = 0
```

The solver's shift relation (-6, 1, 1, 1, 1, 1, 1) lies in this kernel, but it is only one direction of it.

Witness: for d = 4, w = (1127/143, -147/286, 147/572, -63/286, 0, 0, 0) gives exactly the same Q_4 as the Peter-Weyl w = (7, 0, 0, 0, 0, 0, 21/52), yet has w_6 = 0. So N = 924/w_6 is not intrinsic unless the Peter-Weyl reading is imposed.

My own stage-1 return missed this non-uniqueness entirely. The solver at least flagged part of it. The values agree because both of us used the Peter-Weyl reading, so this is an underdetermination, not a disagreement in value.

### Different readings: disagreement or convention?

| Point | Mine | Solver | Classification |
| --- | --- | --- | --- |
| Quaternion -> SU(2) map | w I - i(x σx + y σy + z σz) | [[w+xi, y+zi], [-y+zi, w-xi]] | Convention. The maps are conjugate by V ∈ SU(2): the nullspace of V U_mine = U_sol V is 1-dimensional (singular values 2.83, 2.83, 2.83, 4.7e-16), the rotation of V is (x,y,z) -> (-z,-y,-x), and D^3(V) P_mine D^3(V)^† = P_solver to 7e-16 in both sectors. All norms, dimensions, characters and coefficients are invariant under this. It moves only the item-20 root positions: my monic form with z -> iz is exactly the solver's form. |
| Item 8: one ray or two per s | One (t ∈ (0, π/2)) | Both signs | Reading. The solver's is more complete; both rays share every reported value. |
| Item 13: non-uniqueness of w_K | Not flagged | Flagged as a 1-dim shift | Reading. Both report the Peter-Weyl coefficients, so the values agree; the true kernel is 3-dimensional. |
| Item 18: normalisation | B = 1 | B = 1 plus unit fibre norm | Convention; its B = 1 values equal mine. |
| Item 19: which V_8 map | Isotypic C_iso | Isotypic, reduced to C_0 = [B_6 (x) Theta u]_8 | Convention: \|\|C_iso\|\| = \|\|C_0\|\| and the zero sets are equal. |
| Item 21: N_J basis | <v, N_J> = <[v (x) u]_J, B_J> | [B_J (x) Theta u]_3 | Convention: N_0^mine = -(1/sqrt7) N_0^sol, N_6^mine = -(sqrt91/7) N_6^sol; the coefficients agree after conversion, and the spans are equal. |

## Checks that cannot fail, and mutation results

Each mutation was applied to a copy under `ref_runs/mut_*`. The setup asserts that each textual edit matched exactly once, so no mutation silently failed to apply.

| # | Check (solver file) | Weakness found | Mutation tested | Fired? |
| --- | --- | --- | --- | --- |
| 1 | `solver_stage2_roots.py`: the saved "multiplicities" line | The string "each 1 (12 distinct exact roots ...)" is written unconditionally. `all_twelve_are_exact_roots` is computed but never asserted. | `mut_fmon`: coefficient -33 z^8 -> -32 z^8 | **No.** It prints "all 12 candidates are exact roots: False", exits 0, and still saves "each 1 ...". The check computes but does not stop, and the saved claim has no computation behind it. |
| 2 | `solver_stage2_roots.py`: hard-coded `fmon` | The exact-root verification runs on a typed-in polynomial, not the computed one. | Compared the hard-coded fmon with its computed `monic_form` | The two are identical (exact), so there is no error, but the link was manual. |
| 3 | `solver_items13_18.py`: eigen-ray and beta(B=1) = Q_d checks | N is built from K ∈ {0, 6} only, and Q_d from w_0 and w_6 only, so the checks agree by construction and cannot see K = 1..5 channels. | `mut_P`: non-invariant diagonal projector (not Theta-closed). `mut_P2`: Theta-closed but not Gamma-invariant projector diag(1,0,1,0,1,0,1), with \|\|R_2\|\|^2, \|\|R_4\|\|^2 ≠ 0. | **No** in both. Every ray is reported as an eigenray with beta = Q_d, exit 0; for example with `mut_P2`, d=4: w = (7, 0, 7/60, 0, 14/99, 0, 112/429). |
| 4 | `solver_items13_18.py`: `fact_ok` (factorisation at a rational g) and `ident_ok` ((d/7)<N,u> = A) | Both are identities valid for any Hermitian P, so they test the algebra, not the projector. | `mut_P`, `mut_P2` | **No**: both True with wrong projectors. |
| 5 | `solver_crosscheck_quadrature.py` | Integrates over SU(2), not X, and asserts nothing about the differences. | `mut_P2` (Theta-closed wrong P); `mut_quadcoef` (c_6 × 1.01); `mut_P` | `mut_P2`: **no**, agreement 1e-13. `mut_quadcoef`: the differences rise from 5e-14 to 9.5e-3, visible but not asserted. `mut_P`: **yes**, but only indirectly, through the Theta-fixed-basis assert (eta eta^† ≠ P). |
| 6 | CS phase of the CG coefficients (`solver_lib.CG`; self-test; item 0) | No assertion pins the CS sign convention. The self-test checks equivariance and rho_K = M_K(uu^†), which any per-J phase satisfies. | `mut_cg6`: CG -> -CG at J = 6 | **No.** The self-test passes; item 0 prints 0(b) = -sqrt231/462 and exits 0; item 19's five checks all pass (kappa -> -504 sqrt286); stage 2 reports (L_8(v_3))_3 = -sqrt273/1092 and exits 0. The reported values are right only because sympy's CG is CS. |
| 7 | `solver_lib` self-test: equivariance | A genuine check. | `mut_cgm`: CG × (-1)^{m1} | **Yes** (assertion at J = 0). |
| 8 | `solver_lib` self-test: rho_K = M_K(uu^†) | A genuine check. | `mut_theta`: Theta phase (-1)^m -> 1 | **Yes.** |
| 9 | `solver_lib` self-test: Theta D = D Theta | A genuine check. | `mut_T`: sign pattern dropped from T | **Yes.** |
| 10 | `solver_item00_anchors.py` | No assertions at all: it reports whatever the generators give. | `mut_gen`: q2 -> (1+i+j-k)/2 | **No** in item 0 (prints \|Gamma\| = 24, perfect False, exits 0; it is left to the reader to stop). **Yes** in `solver_group.py` (assert len(G) == 120). |
| 11 | `solver_item19.py` check 1 (C = kappa ι^{-1}J(F^2, G)) | An identity valid for any vector in the Theta slot, so it cannot detect a wrong Theta. | `mut_item19theta`: signs dropped from Theta | Check 1: **no** (still True). Check 3 (the Fix(Theta) assert): **yes**. |
| 12 | sympy `rank(simplify=True)` (items 14, 20, 21) | Possible weak zero-tests on surd matrices. | `mut_rank`: a perfect square F_w^2 with surds (true rank 6); a square with the nested radical sqrt(5+2 sqrt6) (true rank 6); a disguised-zero column (true rank 2); [M_0, M_6, M_0 + sqrt91 M_6] (true rank 2) | **Yes**, all correct: 6, 6, 2, 2 (7 on its real form). Its rank evidence is sound. |
| 13 | `solver_group.py`: item-10 nullspace cross-check, `commutes_with_D`, `hermitian`, `character_projector_matches` | Recorded but never asserted. | Inspected the JSON | All True / equal in its run. They do not stop on failure; not mutated further. |
| 14 | Dead code (`... if False else ...`, `... if True else ...`) | Harmless. | None | Not applicable. |

Bottom line for the checks: Theta, T, the equivariant CG structure and the generators are genuinely guarded. The CS phase, the Gamma-invariance of the projector in every downstream script, and the item-20 multiplicity string are not. The solver's reported values in those places are nonetheless correct: I reproduced each by independent routes in my own code.

## CONSULTED-MATERIAL MANIFEST

| Kind | Item |
| --- | --- |
| Files read (room) | `solver_copy/solver_stage1_return.md`, `solver_copy/solver_stage2_return.md`, `solver_lib.py`, `solver_group.py`, `solver_items01_09.py`, `solver_items13_18.py`, `solver_item19.py`, `solver_item21.py`, `solver_stage2.py`, `solver_stage2_roots.py`, `solver_stage2_roots.log`, `solver_crosscheck_quadrature.py`, `solver_extra_checks.py`, `solver_item00_anchors.py`, `solver_item15_constellations.py`, `solver_results.json` (read by my scripts); my own `audit_results.json`, `audit_projectors_level6.json`, `audit_ref_*.json`, `log_ref_newchecks.txt`, and the mutation logs under `ref_runs/`. Nothing outside the room. `solver_copy/` was never modified (all runs used copies under `ref_runs/`). |
| Textbook facts from memory | Weyl integration formula for SU(2) class functions; characters of Sym^2 and Sym^3 via power sums; Burnside's theorem (the span of an image equals ⊕ End of the distinct constituents); invariant Hermitian quartics ≅ End_SU(2)(Sym^2 V), of dimension Σ mult^2; symmetric-criticality and Newton's method; stereographic projection; SU(2) conjugation induces an SO(3) rotation of Majorana points; unique factorisation of binary forms. |
| Group theory: DERIVED by computation | Burnside dimension 25; agreement of the solver's and my characters class by class; the conjugating V between the two quaternion maps and its rotation; D(V) P_mine D(V)^† = P_solver. |
| Group theory: RECOGNISED | Only that -1 ∈ Gamma forces zero invariants at half-integer spin, used as the argument confirming the solver's half-integer extras. It follows directly from (-1)^{2j} = -1, and I used it as a derivation. |
