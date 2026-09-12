# Auditor stage-2 return (items 19 continued and 20)

New room scripts: `audit_stage2_item19.py`, `audit_stage2_item20.py`, `audit_stage2_merge.py`. The new values sit under the keys `stage2_item19` and `stage2_item20` in `audit_results.json`; the stage-1 keys are untouched. `audit_stage1_return.md` has not been modified.

## Table

| Item | Exact value | How (one line) |
| --- | --- | --- |
| 19 (continued) | (L_8(v_3))_3 = sqrt(273)/1092 = 1/sqrt(4368). All other components of L_8(v_3) are 0, and \|\|L_8(v_3)\|\|^2 = 1/4368. | rho_6(v_3) = [v_3 (x) Theta v_3]_6 = -[v_3 (x) v_{-3}]_6 has only the N=0 component, equal to -<3 3; 3 -3 \| 6 0> = -1/sqrt(924). This is multiplied by <6 0; 3 3 \| 8 3> = -sqrt(143)/26 (CS phase, Racah formula). The full coupling routine and this closed-form product agree exactly. |
| 19 (consistency with stage 1) | L_8(u) = 0 exactly on the three time-reversal-invariant test rays and nonzero on the five non-invariant ones. \|\|L_8(u)\|\|^2 / \|\|C_iso(u)\|\|^2 = 1/4 on all five (40 digits). | C_iso is the stage-1 V_8-isotypic component (exact Casimir route). L_8 is therefore a fixed nonzero multiple of an isometric image of C_iso, and the stage-1 characterisation holds unchanged in this normalisation. |
| 20: the form | For d=3: G(z) = (3 sqrt5/16)(z^12 + 1) + (33/8)(z^10 + z^2) - (99 sqrt5/16)(z^8 + z^4) - (33/4) z^6. For d=4: exactly -G (since R_6(P_3) = -R_6(P_4)), so the root multiset is the same. | Route A is exact in Q(sqrt5, i) from the stage-1 projectors: sqrt(C(12,6+N)) R_N = Σ_{k1+k2=N+6} w_mon[k1,k2]. Route B (sympy CG on the orthonormal P) agrees to 1e-30 in every component. \|\|R_6\|\|^2 = 12/7 is re-asserted exactly. |
| 20: root multiset | 12 distinct roots, all simple (multiplicity 1 each). Degree 12 in z, so there is no root at infinity; no root at 0. | Exact: gcd(G, G') computed over Q(sqrt5, i) is constant. Independently, sympy factors G over Q(sqrt5, i) as (3 sqrt5/16)·(z^2 + (sqrt5 - 2 sqrt5 i)/5)(z^2 + (sqrt5 + 2 sqrt5 i)/5)(z^2 - (1+sqrt5) i z + 1)(z^2 + (1+sqrt5) i z + 1)(z^2 + (sqrt5-1) z - 1)(z^2 - (sqrt5-1) z - 1), six distinct quadratics each with multiplicity 1. |
| 20: positions (numeric, 50 digits; the exact values are the roots of the quadratics above) | Real roots: ±0.5575365158350514 (theta = 58.2825255885°) and ±1.7936044933348411 (theta = 121.7174744115°), azimuths 0° and 180°. Imaginary roots: ±0.2840790438404123 i (theta = 31.7174744115°) and ±3.5201470213402020 i (theta = 148.2825255885°), azimuths ±90°. Unit-circle roots: ±0.5257311121191336 ± 0.8506508083520399 i (theta = 90°), azimuths ±58.2825255885° and ±121.7174744115°. | mpmath polyroots at 50 digits, mapped to the sphere. The pairwise dot products are +1/sqrt5 (30 pairs), -1/sqrt5 (30 pairs) and -1 (6 pairs); the minimum chord is 1.0514622242382672. That is a regular icosahedron: 12 vertices in 6 antipodal pairs. |
| 20: injectivity of v -> (G, F_v)_1 | Injective. | Proof from the multiplicities is below. Separate computational evidence: the 17 × 7 matrix of v -> J(G, F_v) has exact rank 7 over Q(sqrt5, i). That is an exact rank computation, not a consequence of the multiplicities. A double-precision SVD gives smallest singular value 213.39, a third and weaker piece of evidence. Control: replacing G by a perfect square F_w^2 drops the exact rank to 6, as the argument predicts. |

## Item 20: why the root multiset decides injectivity

The first transvectant (G, F)_1 of a degree-12 form and a degree-6 form is a nonzero constant times the Jacobian J(G, F) = G_X F_Y - G_Y F_X. So the kernel of v -> (G, F_v)_1 is {v : J(G, F_v) = 0}. The map v -> F_v is a linear bijection from V_3 onto the binary sextics, and G ≠ 0 because R_6 ≠ 0.

Suppose v ≠ 0 and J(G, F_v) = 0. The supplied classical fact at unequal degrees (m = 12, n = 6) gives G^6 ∝ F_v^12. Factor both into linear forms over C (unique factorisation in C[X,Y]). Every root of G then has multiplicity 2 × (its multiplicity in F_v), so G = c·F_v^2.

Conversely, if G = c·H^2 with H a sextic, then J(G, H) = 2cH·J(H, H) = 0, so the v with F_v ∝ H is in the kernel.

Hence the map is injective iff G is not a constant times the square of a sextic. Equivalently, it is injective iff not every root multiplicity of G is even, with a root at infinity counted as the degree deficit. G has 12 simple roots, so the map is injective. The exact rank 7 above agrees with this, and so does the rank-6 control for a square.

## Convention differences and readings

| Point | What I did |
| --- | --- |
| Majorana correspondence at spin 6 (the extract fixes it only for V_3) | I used the direct extension G(z) = Σ_N (-1)^{6-N} sqrt(C(12,6+N)) R_N z^{6-N}, homogenised with z = X/Y. Root multiplicities, and the injectivity verdict, do not depend on this choice: any other SL_2-compatible convention moves the points rigidly and leaves the multiplicities alone. The positions reported refer to this convention and to my quaternion-to-SU(2) map from stage 1. |
| eps_3 | eps_3 = -1 as fixed in the extract. A global sign changes neither roots nor ranks. |
| Sector | R_6(P_3) = -R_6(P_4) exactly, so both sectors give the same root multiset. |
| F_v basis in the rank computation | The columns were taken as X^{3-m} Y^{3+m}. The extract's F_{v_m} differ from these by the nonzero factors (-1)^{3-m} sqrt(C(6,3+m)), which do not change the rank. |
| L_8 | Computed exactly as defined in stage 2: [rho_6(u) (x) u]_8 with the CS couplings. |

## Effect on stage 1

Nothing in stage 2 shows a stage-1 answer to be wrong. Item 19's characterisation, C(u) = 0 iff the ray is time-reversal invariant, is confirmed in the explicit L_8 normalisation: L_8 = (1/2)·(an isometric image of) C_iso, since the norm² ratio is 1/4. The stage-1 values \|\|R_6\|\|^2 = 12/7 and R_6(P_3) = -R_6(P_4) are re-confirmed exactly by a new route.

## CONSULTED-MATERIAL MANIFEST

| Kind | Item |
| --- | --- |
| Files read | `stage2.md`; my own outputs `log_s2_20.txt`, `log_s2_19.txt` (printed by my commands); `audit_projectors_level6.json` and `audit_results.json` (read by my scripts). Nothing outside the room. |
| Textbook facts from memory | Stretched Clebsch-Gordan coefficient <j1 m1; j2 m2 \| j1+j2, m1+m2> = sqrt(C(2j1, j1+m1) C(2j2, j2+m2) / C(2j1+2j2, j1+j2+m1+m2)) in the CS phase (used in route A, cross-checked by route B); the Racah formula; the first transvectant as a constant multiple of the Jacobian; unique factorisation of binary forms into linear factors over C; the Euclidean algorithm for the polynomial gcd; the chain and Leibniz rules; icosahedron geometry (vertex dot products ±1/sqrt5, edge 1.0515 on the unit sphere), used only to name the shape. |
| Supplied as input | The classical Jacobian fact at unequal degrees (f^n ∝ g^m) and the conventions of the extract. |
| Group theory: DERIVED by computation | R_6(P) itself, its exact coefficients, simplicity of all 12 roots, their positions and the pairwise-dot pattern, the exact rank 7, and R_6(P_3) = -R_6(P_4), all from the stage-1 projectors built from the two generators. |
| Group theory: RECOGNISED from prior knowledge | As an expectation only, not used as input: the unique Gamma-invariant in V_6 should be Klein's degree-12 icosahedral form, whose roots are the 12 vertices of a regular icosahedron. The computation matched this. |
