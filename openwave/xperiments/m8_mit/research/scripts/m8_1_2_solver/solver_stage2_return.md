# Solver stage 2 return

Same conventions as stage 1. New scripts: `solver_stage2.py` and `solver_stage2_roots.py` (its
printed output is kept in `solver_stage2_roots.log` in the room). New values are in
`solver_results.json` under the sections `stage2` and `stage2_exact_roots`. Arithmetic is exact
unless a value is labelled FLOAT.

## The table

| Item | Exact value | How (one line) |
| --- | --- | --- |
| 19 (continued) | (L_8(v_3))_3 = sqrt(273)/1092 (FLOAT about 0.0151307); every other component of L_8(v_3) is 0 | rho_6(v_3) has only N = 0, equal to -sqrt(231)/462; CG <6 0; 3 3 \| 8 3> = -sqrt(143)/26; product = sqrt(273)/1092 |
| 19 (consistency) | L_8(u) = -(1/2) C_0(u) identically, with C_0(u) = [[u (x) u]_6 (x) Theta u]_8 | exact polynomial identity; matches stage 1's k_6 = -1/2 and its C_0(v_3) = -sqrt(273)/546 |
| 20: the form | R_6(P) in the d=4 sector, components N = -6..6: 3sqrt5/16, 0, -sqrt66/16, 0, -3sqrt11/16, 0, sqrt231/56, 0, -3sqrt11/16, 0, -sqrt66/16, 0, 3sqrt5/16. In the d=3 sector R_6 = -1 times that. Majorana form, normalised to be monic: F_R(z) = z^12 - (22 sqrt5/5) z^10 - 33 z^8 + (44 sqrt5/5) z^6 - 33 z^4 - (22 sqrt5/5) z^2 + 1, of degree 12 (so no roots at infinity) | transform of the group-built projector; handout Majorana formula with spin 3 replaced by spin 6 (see conventions) |
| 20: multiplicities | All 12 roots are SIMPLE (multiplicity 1 each); no repeated roots | exact: gcd(F_R, F_R') = 1 over Q(sqrt5 + i), discriminant = 5444517870735015415413993718908291383296/390625 (nonzero); 12 distinct exact roots verified |
| 20: root multiset | 12 points, each multiplicity 1: (+-a, 0, +-b), (0, +-b, +-a), (+-b, +-a, 0), with a = sqrt((5 - sqrt5)/10), b = sqrt((5 + sqrt5)/10). As roots z = (x + i y)/(1 + z_3): +-a/(1+b), +-a/(1-b), +-i b/(1+a), +-i b/(1-a), +-b +- i a (FLOAT: +-0.284079, +-3.520147, +-0.557537 i, +-1.793604 i, +-0.850651 +- 0.525731 i) | candidates read off the FLOAT roots, then F_R(r) = 0 verified EXACTLY at all 12; exact pairwise dots {1/sqrt5: 30, -1/sqrt5: 30, -1: 6}, each point has 5 neighbours at dot 1/sqrt5: the vertices of a regular icosahedron |
| 20: injectivity | v -> J(F_R, F_v) (first transvectant, degree 16) is INJECTIVE on V_3 | established from the root multiplicities (proof below); separately, the exact symbolic rank of the 17x7 matrix is 7, and the FLOAT singular values are all at least 1173 |
| 20: role of the multiset | The map has a nonzero kernel iff F_R is a constant times the square of a sextic, iff every root of F_R has even multiplicity. Here every root is simple, so the kernel is 0 | classical Jacobian fact (supplied) plus unique factorisation in C[X,Y] |

## Item 20: injectivity from the multiplicities (the proof)

Homogenise F(X, Y) = Y^deg F(X/Y). The first transvectant of a degree-12 form f and a sextic g is,
up to a nonzero constant, the Jacobian J(f, g) = f_X g_Y - f_Y g_X. The constant does not affect
the kernel.

1. Let v be nonzero in V_3. Then F_v is a nonzero sextic. Suppose J(F_R, F_v) = 0. The supplied
   classical fact at unequal degrees (m = 12, n = 6) gives F_R^6 = c F_v^12 with c nonzero.
2. Factor into linear forms (C[X,Y] is a UFD): F_R = prod l_i^(a_i), F_v = prod l_i^(b_i). Then
   6 a_i = 12 b_i, so every a_i = 2 b_i is even.
3. F_R has 12 simple roots, i.e. every a_i = 1 (exact: gcd(F_R, F_R') = 1). That is a
   contradiction, so J(F_R, F_v) = 0 forces v = 0. The linear map is injective.
4. The converse, which explains the role of the multiset: if all a_i were even, then F_R = c G^2
   with G a sextic, and J(G^2, G) = 2 G J(G, G) = 0, so the v with F_v = G would lie in the kernel.
   A single root of odd multiplicity already forces injectivity.

**Evidence statement.** Injectivity was established from the root multiplicities, as above.
Separately, and as different evidence, the rank was computed two further ways:

- Exact symbolic rank: sympy `rank(simplify=True)` on the exact 17x7 coefficient matrix of
  v -> J(F_R, F_v) returned 7. The equivariant coupling v -> [R_6 (x) v]_8 also returned 7.
- FLOAT: singular values 4298.5, 4196.9, 3437.8, 3099.6, 1826.9, 1819.2, 1173.1.

## Stage-1 answers in the light of stage 2

Nothing in stage 2 shows a stage-1 answer to be wrong, so `solver_stage1_return.md` is unchanged.

- The 19 (continued) value agrees with stage 1: k_6 = -1/2 times C_0(v_3) = -sqrt(273)/546 gives
  sqrt(273)/1092.
- One stage-1 observation now has a derived explanation. The computed R_6(P_3) = -R_6(P_4) follows
  because P_3 + P_4 = I_7 (complementary isotypic projectors) and M_6(I) = 0. That is CG
  orthogonality: M_K(I)_0 = -sqrt7 sum_n <3 n; 3 -n|K 0><3 n; 3 -n|0 0> = -sqrt7 delta_K0. This is
  why ||R_6||^2 = 12/7 came out equal in both sectors in item 12. It also means the root multiset
  is the same in both sectors.

## Conventions that had to differ, or readings I had to choose

| Item | What was chosen | Effect |
| --- | --- | --- |
| 20 | The handout fixes the Majorana correspondence only at spin 3. I used the same formula with 3 replaced by 6: F_R(z) = sum_m (-1)^(6-m) sqrt(C(12, 6+m)) R_m z^(6-m). Both it and the spin-3 formula equal a constant times iota_j(v)(Y, -X), so they are compatible, equivariant identifications | positions of the roots depend on this and on the stage-1 quaternion-to-SU(2) choice; multiplicities and injectivity do not |
| 20 | "First transvectant" read as the Jacobian of the homogenised forms (the standard first transvectant up to a nonzero constant) | none for injectivity |
| 20 | R_6 differs by a sign between sectors; I report the form normalised to be monic | same roots |
| 19 (continued) | CG order as written: <6 N; 3 m \| 8 M> with rho_6 in the first slot | value as reported; swapping the slots would multiply it by (-1)^(6+3-8) = -1 |

## CONSULTED-MATERIAL MANIFEST

| Kind | Item | Use |
| --- | --- | --- |
| File read | `stage2.md` (room) | the task |
| File read | own room files: `solver_lib.py` helpers, `solver_results.json` (stage-1 projectors), `solver_stage2_roots.log` | inputs and outputs |
| Not read | One tool output was too large and the harness saved it automatically to a path outside the room; I did not open it. I reran with output redirected into the room (`solver_stage2_roots.log`) | compliance with rule 1 |
| Context disregarded | instruction and memory files injected by the harness (CLAUDE.md files, memory index) | not consulted |
| Supplied input | classical fact: J(f,g) = 0 implies f^n proportional to g^m (handout) | item 20 proof step 1 |
| Textbook fact (memory) | C[X,Y] is a UFD, and binary forms over C factor into linear forms | item 20 proof step 2 |
| Textbook fact (memory) | a polynomial is squarefree iff gcd(f, f') = 1, iff the discriminant is nonzero | multiplicities |
| Textbook fact (memory) | the first transvectant is the Jacobian up to a constant; stereographic projection and its inverse r = (x + i y)/(1 + z) | item 20 |
| Textbook fact (memory) | CG orthogonality | the R_6(P_3) = -R_6(P_4) explanation |
| Library | sympy (factor_list over Q(sqrt5, i), gcd, discriminant, minimal_polynomial, rank), numpy (FLOAT roots and SVD only) | computation |
| Group theory, DERIVED | R_6(P) in both sectors, their proportionality (-1), squarefreeness, the 12 exact roots and their icosahedral dot-product pattern | all computed from the stage-1 projectors built from q1, q2 |
| Group theory, RECOGNISED | that a degree-12 Gamma-invariant form should be Klein's icosahedral vertex form with 12 simple roots (the rotation group's orbits on the sphere have sizes 12, 20, 30, 60) | expectation only; the multiplicities and positions reported were established by the exact computations above, not by this recognition |
