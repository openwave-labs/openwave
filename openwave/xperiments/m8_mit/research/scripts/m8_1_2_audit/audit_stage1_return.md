# Auditor stage-1 return (independent implementation)

Room scripts: `audit_field.py` (exact Q(sqrt5, i) arithmetic, quaternions), `audit_su2lib.py` (Racah CG in the Condon-Shortley phase, Theta, M_K), `audit_item0.py`, `audit_gamma.py`, `audit_su2_items.py`, `audit_item16_ranks.py`, `audit_item19.py`, `audit_quadrature.py`, `audit_derived.py`. Every computed value is in `audit_results.json`, with exact values stored as strings.

Cross-checks built in (each is an assertion that can fail):
- The Racah CG agrees with sympy's CG on all 3 (x) 3 couplings.
- The Gamma objects are exact over Q(sqrt5, i) in the monomial basis of Sym^n, and every Gamma dimension was computed two ways.
- The projectors were built two independent ways and agree exactly.
- ||M_K(P)||^2 was computed exactly with Casimir projectors, and separately numerically with CG tables. The two agree to 1e-12.
- w_K, kappa_K, beta and the item-21 coefficients were each derived exactly, then checked against a direct double-precision Gauss quadrature over SU(2). That quadrature does not use Schur orthogonality, and it agrees to 1e-10 or better.

## Item 0 (reported first; the result matched my expectation, so I continued)

| Item | Exact value | How |
| --- | --- | --- |
| 0(a) | Theta v_2 = v_{-2} | Definition Theta v_m = (-1)^m conj v_{-m}, coded and asserted. The general-spin form with eps_3 = -1 reproduces it for all m. |
| 0(b) | <3 3; 3 -3 \| 6 0> = 1/sqrt(924) = sqrt(231)/462 | Racah formula = sympy CG = sqrt(6!6!/12!), all exact. |
| 0(c) norms | \|\|q1\|\|^2 = 1, \|\|q2\|\|^2 = 1 | Exact in Q(sqrt5). |
| 0(c) order | \|Gamma\| = 120 | Exact BFS closure of <q1, q2> as quaternions over Q(sqrt5). Element orders {1:1, 2:1, 3:20, 4:30, 5:24, 6:20, 10:24}, 9 conjugacy classes, centre of order 2. |
| 0(c) perfect | Yes: the derived subgroup has order 120, so Gamma = [Gamma, Gamma] | Closure of all 120 distinct commutators. |

## Main table

| Item | Exact value | How (one line) |
| --- | --- | --- |
| 1 | Sym^3 V_3 = V_9 + V_7 + V_6 + V_5 + V_4 + 2 V_3 + V_1 (dim 84) | Weight multiplicities of the cubic monomials. |
| 2 | J=8: 1; J=3: 4 | Weight multiplicities of Sym^2 V_3 (x) V_3 (conj V_3 has the same weights). Also Sym^2 V_3 = V_6+V_4+V_2+V_0, and V_8 has multiplicity 2 in V_3^{(x)3}. Check: the four maps f_L = [[u(x)u]_L (x) Theta u]_3 (L=0,2,4,6) have rank 4. |
| 3 | M_0(u) = -(\|\|u\|\|^2/sqrt7) u. It is a multiple of u, with constant -1/sqrt7 times \|\|u\|\|^2. | Exact symbolic identity for general complex u. rho_0(u) = -\|\|u\|\|^2/sqrt7. |
| 4 | Diagonal: M_6(v_m) = lambda_m v_m with lambda_{-3..3} = -(sqrt91/12012)·(1, 36, 225, 400, 225, 36, 1) = -(sqrt91/12012)·C(6,3+m)^2 | Exact. All off-weight components are 0. Structure: all negative, symmetric in m <-> -m, proportional to squared binomials. The linear operator A_6(v_m) is also diagonal, with entries -(sqrt91/12012)(-1)^{k-m} C(6,3+m) C(6,3+k), so the alternating sign sits in A_6 and cancels on the diagonal of M_6. |
| 5 | B_2(v_3) = 0 (all five components). rho_2(v_3) has only N=0 nonzero, equal to -5 sqrt21/42. | Exact. They disagree. The definitions call for rho_K = [u (x) Theta u]_K wherever a density multipole is wanted. B_2(v_3) vanishes by weight alone (weight 6 > 2). |
| 6 | 924 \|\|rho_6\|\|^2 = 1 (v_3), 400 (v_0), 288 ((v_2+v_-2)/sqrt2), 463 ((v_3+v_-3)/sqrt2). All four rays are critical. | Exact. Criticality: the exact Wirtinger gradient of \|\|rho_6\|\|^2 at each point has zero component orthogonal to C·u. Computed dF/d(ubar) = 2 G_6 = 2 c_6 M_6, so critical iff M_6(u) is proportional to u, verified. Symmetric-criticality reading: the ray stabiliser acts on T_[u]P with no fixed vector. |
| 7 | (\|\|rho_K\|\|^2)_{K=0..6} = (1/7, 0, 0, 0, 6/11, 0, 24/77) | Exact. The row sums to 1. |
| 8 | \|\|rho_6\|\|^2 = 3/77 + (10/11) s - (125/132) s^2 (924× = 36 + 840 s - 875 s^2). One interior stationary point: s = 12/25, value 9/35 (924× = 1188/5), a maximum (f'' = -125/66). Endpoints: s=0 gives 3/77 (924× = 36); s=1 gives 1/924 (924× = 1). | Exact. In cos/sin form: 3c^4/77 + 76c^2s^2/77 + s^4/924. The s = 12/25 ray is also critical on all of P(V_3) (M_6 is proportional to u). |
| 9 | r̂_6 = (100x^4 + 200x^2y^2 - 20x^2 + 100y^4 + 148y^2 + 463) / (231 (x^2+y^2+2)^2). Critical set in the chart: (0,0) value 463/924, local max; (0, ±sqrt10/2) value 24/77, saddles; (±sqrt(23/10), 0) value 200/903, local minima. The missing point [v_0] (z = ∞) is also critical, value 100/231, local max. Unnormalised \|\|rho_6\|\|^2 on the slice: critical at (0,0) (value 463/231) and (±1/sqrt10, 0) (value 2, where r̂_6 = 200/441). | Lex Groebner basis {x(10x^2+2y^2-23), y(14x^2+6y^2-15), xy(8y^2+43), y(2y^2-5)(8y^2+43)}. Real solutions follow by cases. The chart covers everything except [v_0], handled in the chart w = 1/z. Hessians exact. Every r̂_6-critical ray is also critical on all of P(V_3). The ray statement is the r̂_6 set; details below. |
| 10 | dim (V_K)^Gamma for K = 0..6: 1, 0, 0, 0, 0, 0, 1 | DERIVED by computation. Method 1: exact character average (1/120) Σ U_{2K}(Re g) over the 120 computed elements. Method 2: exact common fixed space of D(q1), D(q2) (nullspace). The methods agree. Separately, beyond K=6 (character method only): K = 10, 12, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28 give 1; K = 30 gives 2; every other K in 7..30 gives 0. |
| 11 | V_3\|Gamma = sigma_3 + sigma_4: 2 constituents, dimensions 3 and 4, each with multiplicity 1 | DERIVED by computation. The exact commutant of {D(q1), D(q2)} has dimension 2, and (1/120) Σ\|chi_3\|^2 = 2 = Σ m_i^2, so the only option is 1+1. Each summand is irreducible: (1/120) Σ\|chi_sigma\|^2 = 1. Dimensions are the projector ranks. Frobenius-Schur indicator +1 for both. Characters on the classes listed below. |
| 12 | d=3: (\|\|M_K(P)\|\|^2)_{K=0..6} = (9/7, 0, 0, 0, 0, 0, 12/7). d=4: (16/7, 0, 0, 0, 0, 0, 12/7). | P was built from the group: the spectral idempotents of a class sum inside the 2-dim commutant (exact), cross-checked against P = (d/120) Σ conj(chi(g)) D(g). P is idempotent, commutes with both generators and is orthogonal for the invariant form. Norms by exact Casimir projectors; Σ_K = d is asserted. |
| 13 | w_K = 49 \|\|R_K\|\|^2 / ((2K+1) d^2). Both sectors: w_0 = 7, w_1..w_5 = 0. d=3: w_6 = 28/39, w_6/w_0 = 4/39, N = 1287. d=4: w_6 = 21/52, w_6/w_0 = 3/52, N = 2288. As functions of d: w_6/w_0 = 12/(13 d^2) and N = 143 d^2. | Peter-Weyl plus Schur orthogonality (derivation below). The d-formula uses \|\|R_6\|\|^2 = 12/7 in both sectors, which follows from R_6(P_3) = -R_6(P_4) since P_3 + P_4 = I and M_6(I) = 0. Quadrature intercept and slope: 1.0000000000000002 and 0.7179487179487175 (d=3), 1.0000000000000004 and 0.40384615384615086 (d=4). |
| 14 | N(u) = Σ_K kappa_K M_K(u) with kappa_K = (7/d) \|\|R_K\|\|^2 c_K/(2K+1) and c_K = (-1)^{K+1} sqrt((2K+1)/7). d=3: kappa_0 = -3/sqrt7, kappa_1..kappa_5 = 0, kappa_6 = -4/sqrt91. d=4: kappa_0 = -4/sqrt7, kappa_1..kappa_5 = 0, kappa_6 = -3/sqrt91. | c_K is the exact ratio G_K/M_K, found constant over two exact states per K. Nonzero kappa_6 is established by exact \|\|R_6\|\|^2 = 12/7 != 0 and exact c_6 = -sqrt91/7 != 0. The quadrature fit of N(u) on {M_0, M_6} gives -1.13389341902768 and -0.4193139346887685 (d=3) with residual 6e-15. rank{M_0, M_6} = 2, so M_6 is not a multiple of M_0. Caveat: the M_K are not independent (see the notes). |
| 15 | See the constellation section below. | Exact roots of F_u, reduced via F = zeta^a H(zeta^g). Each root checked to 40 digits. |
| 16 | rho^{(j)}_K carries all ranks K = 0..2j (odd included), shown by exact nonzero \|\|rho_K(v_j)\|\|^2 at every level. For levels 1..5 the rep V_j\|Gamma is irreducible, of dimension 2, 3, 4, 5, 6, and \|\|M^{(j)}_K(P = I)\|\|^2 is nonzero only at K = 0 (value 2, 3, 4, 5, 6). Level 6: K = 0 and K = 6 only, in both summands. | Commutant dimension from the generators at each level (1 at levels 1..5, 2 at level 6). Exact Casimir norms. Consequence: at levels 1..5, Q ≡ 1 and N(u) = kappa_0 M_0(u), proportional to \|\|u\|\|^2 u, so the projected cubic cannot depend on the direction of u. The first dependence enters at level 6, through K = 6. |
| 17 | Q_3 = 1 + (28/39) r̂_6, Q_4 = 1 + (21/52) r̂_6 | Identical critical sets on P(V_3), also identical in type, since each is an affine function of the same r̂_6 with positive slope (dQ_d = w_6 dr̂_6, w_6 > 0). Uses w_0 r̂_0 = 7·(1/7) = 1 and w_1..w_5 = 0. |
| 18 | With B(u) = 1 (i.e. \|\|u\|\|^2 = 7/d): beta = A(u) = Q_d([u]) = 1 + w_6^{(d)} r̂_6([u]). beta_3 - beta_4 = (49/156) r̂_6([u]). | <psi_u, \|psi\|^2 psi_u> = A and = (d/7) beta \|\|u\|\|^2 = beta B. Here r̂_6 is scale-invariant; the raw \|\|rho_6(u)\|\|^2 at B = 1 equals (7/d)^2 r̂_6. Checked exactly against the kappa route at the item-6 rays, and by quadrature (<u,N>/\|u\|^2 ÷ (A/B) = 1.0000000000000). Values at the item-6 rays below. |
| 19 | C(u) = 0 exactly when Theta u = lambda u for some \|lambda\| = 1. Equivalently the ray is time-reversal invariant, u ∈ U(1)·Fix(Theta), i.e. the Majorana constellation is antipodally symmetric as a multiset. | Proof below. Exact checks on 17 states (V_8-isotypic projection by the Casimir on V_3^{(x)3}): zero exactly on the TR-invariant rays, nonzero otherwise, and \|\|C\|\|^2 / \|\|J(p_u^2, p_{Theta u})\|\|^2 = 1/1296 in every nonzero case. |
| 21 | Gamma-invariance needs sigma(h) sigma(h)^T = I, which holds iff sigma is of real type (FS indicator +1, true for both level-6 summands) with eta Theta-real. The left factor is the holomorphic square B_J(u); the right factor is again R_J = M_J(P). The filter keeps J = 0, 6 (both even, so B_J survives). Interaction = d N_0 + (12/(13d)) N_6. Its span is span{N_0, N_6}, of dimension 2. It does not coincide with span{M_0, M_6}. | Derivation below. The quadrature confirms the psi psi^T invariance, E = Σ\|\|B_J\|\|^2 \|\|R_J\|\|^2/(2J+1), and the coefficients 3, 4/13 (d=3) and 4, 3/13 (d=4). rank{M_0, M_6, N_0, N_6} = 4 (60-digit SVD: singular values 1.76, 1.17, 0.54, 0.25), whereas each pair has rank 2. |

### Characters of the two level-6 summands (item 11)

Classes are ordered by the real part of a representative:
- -1/2: order 3, size 20
- -1: size 1
- 1/2: order 6, size 20
- 0: order 4, size 30
- (-1-sqrt5)/4: size 12
- (-1+sqrt5)/4: size 12
- (1-sqrt5)/4: size 12
- (1+sqrt5)/4: size 12
- 1: size 1

| Summand | Character values in that order |
| --- | --- |
| sigma_4 | 1, 4, 1, 0, -1, -1, -1, -1, 4 |
| sigma_3 | 0, 3, 0, -1, (1-sqrt5)/2, (1+sqrt5)/2, (1+sqrt5)/2, (1-sqrt5)/2, 3 |

Level 1, 3 and 5 constituents have FS indicator -1 (quaternionic type). Level 2 and 4 constituents have +1.

### Item 9 details

The chart u = v_3 + z v_0 + v_{-3} is the affine chart {a ≠ 0} of the projective line {[a(v_3+v_{-3}) + b v_0]}. It misses only [v_0]. In the chart w = 1/z the function is r̂_6 = (463|w|^4 - 20p^2 + 148q^2 + 100)/(231(2|w|^2+1)^2), with zero gradient at w = 0 and Hessian diag(-40/11, -24/11). The critical set on the line therefore has two minima, two saddles and two maxima, consistent with chi(S^2) = 2.

The r̂_6 set is the statement about rays: r̂_6 is homogeneous of degree 0 and descends to P(V_3). By contrast, \|\|rho_6\|\|^2 on the slice depends on the normalisation u_{±3} = 1 fixed by the chart. Its extra critical points (±1/sqrt10, 0) are chart artefacts (r̂_6 is not stationary there).

### Item 13 derivation

With psi(g) = u^T D(g) eta: \|psi(g)\|^2 = <ubar ubar^†, X(g)>_F, where X = D P D^†.

The map P -> (M_K(P))_K is a Frobenius isometry End V_3 -> ⊕_K V_K (the CG matrix is unitary and the relabelling v_k^* -> (-1)^k v_{-k} is unitary). It is equivariant, M_K(D P D^†) = D^K M_K(P), because that relabelling is the linear form of Theta, which commutes with SU(2). Hence \|psi\|^2 = Σ_K <M_K(ubar ubar^†), D^K(g) R_K>.

The integrand is right-Gamma invariant, so ∫_X = ∫_SU(2) (normalised Haar). Schur orthogonality ∫ D^K_{ab} conj D^{K'}_{cd} = δ δ δ/(2K+1) gives:
- B = <M_0(ubar ubar^†), R_0> = d\|\|u\|\|^2/7, since R_0 = -d/sqrt7.
- A = Σ_K \|\|M_K(ubar ubar^†)\|\|^2 \|\|R_K\|\|^2/(2K+1).

The components of M_K(ubar ubar^†) are the complex conjugates of those of rho_K(u) (the CG coefficients are real and (Theta ubar)_{n'} = conj((Theta u)_{n'})), so the norms agree. Hence

Q_d = A/B^2 = Σ_K [49 \|\|R_K\|\|^2 / ((2K+1) d^2)] r̂_K.

With \|\|R_0\|\|^2 = d^2/7 this gives w_0 = 7, and w_0 r̂_0 = 1.

Item 14 follows the same way:
- <psi_v, psi_u> = (d/7)<v, u>.
- <psi_v, \|psi_u\|^2 psi_u> = Σ_K \|\|R_K\|\|^2/(2K+1) <rho_K(u), [u (x) Theta v]_K> =: Σ_K \|\|R_K\|\|^2/(2K+1) <v, G_K(u)>.
- Hence N = (7/d) Σ_K \|\|R_K\|\|^2 G_K/(2K+1), with G_K = c_K M_K.

### Item 15 constellations

The polar angle is theta = 2 arctan\|r\| as fixed in the extract; latitude = 90° - theta. xyz = (2 Re r, 2 Im r, 1 - \|r\|^2)/(1 + \|r\|^2).

| Ray | F_u | Points |
| --- | --- | --- |
| v_3 | 1 | All six at the south pole (0,0,-1): one sixfold point. |
| v_0 | -2 sqrt5 zeta^3 | Three at the north pole and three at the south pole. |
| (v_2+v_-2)/sqrt2 (item 6; also item 9, y = ±sqrt10/2, up to rotation) | -sqrt3 (zeta^5 + zeta) | North pole, south pole, and four equatorial points (theta = 90°) at azimuths 45°, 135°, -135°, -45°. The 90° spacing makes a square, so the six points form a regular octahedron (pairwise dots: 0 twelve times, -1 three times). |
| (v_3+v_-3)/sqrt2 (item 6; item 9 at z = 0) | (zeta^6 + 1)/sqrt2 | Six equatorial points at azimuths ±30°, ±90°, ±150°. The 60° spacing makes a regular hexagon (D6h). |
| item 8, s = 12/25 | (2 sqrt3/5) zeta^6 - (sqrt78/5) zeta | North pole, plus five points with \|r\| = (13/2)^{1/10}, cos theta = (1 - (13/2)^{1/5})/(1 + (13/2)^{1/5}), theta = 100.66255400100904°, at azimuths 0°, ±72°, ±144°. The 72° spacing makes a regular pentagon just below the equator: a C5v pentagonal pyramid, not a regular polytope (pole-to-vertex and vertex-to-vertex distances differ). |
| item 9, z = -i sqrt10/2 | zeta^6 + 5 sqrt2 i zeta^3 + 1 | Three points at cos theta = 1/sqrt3 (theta = 54.735610317245346°), azimuths 30°, 150°, -90°. Three points at cos theta = -1/sqrt3, azimuths -30°, 90°, -150°. Coordinates: (sqrt2/2, ±sqrt6/6, ±sqrt3/3) and (0, ∓sqrt6/3, ±sqrt3/3), sign patterns as in the JSON. Each triangle is spaced 120° and the two are staggered by 60°. Pairwise dots {0: 12, -1: 3} make a regular octahedron with a 3-fold axis along z. |
| item 9, z = +i sqrt10/2 | zeta^6 - 5 sqrt2 i zeta^3 + 1 | The same octahedron reflected: the upper triangle is at azimuths -30°, 90°, -150° and the lower at 30°, 150°, -90°. |
| item 9, z = +sqrt(23/10) | zeta^6 - sqrt46 zeta^3 + 1 | Upper triangle: \|r\|^6 = 22 - sqrt483, cos theta = (1-a)/(1+a) with a = (22-sqrt483)^{1/3}, theta = 56.04970001896834°, azimuths 0°, 120°, -120°. Lower triangle: \|r\|^6 = 22 + sqrt483, theta = 123.95029998103166°, same azimuths. The triangles are eclipsed and mirror images through the equator (\|r_up\|·\|r_low\| = 1): a D3h triangular prism, not antipodal and not square-faced. |
| item 9, z = -sqrt(23/10) | zeta^6 + sqrt46 zeta^3 + 1 | The same prism rotated by 60°: both triangles at azimuths 60°, 180°, -60°. |
| item 9, [v_0] | (see v_0) | Three at the north pole and three at the south pole. |

### Item 18 values at the item-6 rays (exact; beta = Q_d)

| Ray | beta (d=3) | beta (d=4) | beta_3 - beta_4 |
| --- | --- | --- | --- |
| v_3 | 1288/1287 | 2289/2288 | 7/20592 |
| v_0 | 1687/1287 | 168/143 | 175/1287 |
| (v_2+v_-2)/sqrt2 | 175/143 | 161/143 | 14/143 |
| (v_3+v_-3)/sqrt2 | 1750/1287 | 2751/2288 | 3241/20592 |

### Item 19: characterisation and proof

Claim: for u ≠ 0, C(u) = 0 iff Theta u = lambda u for some lambda (then \|lambda\| = 1). Equivalently u ∈ U(1)·Fix(Theta), Fix(Theta) = {u_{-m} = (-1)^m conj u_m}, i.e. the ray is time-reversal invariant.

Proof.
1. u (x) Theta u (x) u lies in the subspace S ⊂ V_3^{(x)3} that is symmetric in slots 1 and 3, so S ≅ Sym^2 V_3 (x) V_3 as a representation (slot 2 carries w = Theta u as an independent vector). Sym^2 V_3 = V_6+V_4+V_2+V_0, and by the Clebsch-Gordan series V_L (x) V_3 contains V_8 only for L ≥ 5. So V_8 has multiplicity exactly 1 in S (item 2, J = 8).
2. Let Pi_8 be the V_8-isotypic projector of V_3^{(x)3} (a polynomial in the total Casimir, so it commutes with the slot swap and preserves S). C(u) = Pi_8(u (x) Theta u (x) u), and Pi_8\|_S is a nonzero equivariant map onto the single V_8 copy in S.
3. Identify V_j with binary forms of degree 2j (p_u = Σ u_m sqrt(C(6,3+m)) x^{3+m} y^{3-m}; F_u of the extract is p_u(Y, -X), an SL_2 substitution). Define Phi: S -> Sym^16 by Phi(a·b (x) w) = J(p_a p_b, p_w). Multiplication of forms is SL_2-equivariant, and J(f∘A, g∘A) = det(A)·J(f,g)∘A by the chain rule, so Phi is SU(2)-equivariant with image in V_8. Phi ≠ 0: J(x^12, y^6) = 72 x^11 y^5.
4. By Schur, any equivariant map S -> V_8 factors through the V_8-isotypic component of S, which is irreducible. A nonzero such map is injective on that component. Both Pi_8\|_S and Phi are nonzero, so ker Phi = ker Pi_8\|_S. Hence C(u) = 0 iff J(p_u^2, p_{Theta u}) = 0.
5. J(p_u^2, g) = 2 p_u J(p_u, g) (Leibniz). C[x,y] is a domain and p_u ≠ 0, so C(u) = 0 iff J(p_u, p_{Theta u}) = 0.
6. p_u and p_{Theta u} are nonzero (Theta is injective) forms of the same degree 6. By the supplied classical fact, J = 0 iff p_{Theta u} = lambda p_u, i.e. Theta u = lambda u. Conversely, Theta u = lambda u gives J(p_u, lambda p_u) = 0.
7. \|lambda\| = 1 because Theta is norm-preserving. Writing lambda = e^{i alpha}, a = e^{i alpha/2} u satisfies Theta a = a, so u ∈ U(1)·Fix(Theta). By section 2.2 this is antipodal symmetry of the six Majorana points as a multiset. ∎

Reading taken: "the equivariant projection onto V_8" is ambiguous in V_3^{(x)3}, where V_8 has multiplicity 2. For example, coupling slots 1 and 3 to spin 5 first gives a V_8 map that vanishes identically on S. I took the isotypic projection, or equivalently any V_8 map not identically zero on S. Normalisation is irrelevant, as the extract says. Checks: the 1/1296 ratio constancy and the zero pattern in `audit_item19.py`.

### Item 21 derivation

psi psi^T(gh) = psi(g) sigma(h) sigma(h)^T psi(g)^T. For the quartic \|psi psi^T\|^2 to be invariant for all sections we need sigma sigma^T = lambda(h) I with lambda a unitary character. Gamma is perfect (item 0c), so lambda ≡ 1 and sigma must be orthogonal in the eta-basis. That is possible iff FS(sigma) = +1. The computed indicators are +1 for sigma_3 and sigma_4 (and -1 at levels 1, 3, 5).

A Theta-fixed orthonormal eta (columns with Theta e_a = e_a) realises it; the numeric check confirms sigma real orthogonal, while a generic unitary rebasing fails.

Then psi psi^T = Σ u_m u_{m'} (D P' D^T)_{mm'} with P' = eta eta^T:
- The left factor has fibre content u (x) u, i.e. the holomorphic square B_J(u).
- p' = eta eta^T = w(P) for Theta-real eta (checked numerically), so the right factor is [p']_J = M_J(P) = R_J.
- Hence ∫\|psi psi^T\|^2 = Σ_J \|\|B_J(u)\|\|^2 \|\|R_J\|\|^2/(2J+1).

Projecting (psi psi^T) conj(psi) onto level 6 gives (7/d) Σ_J \|\|R_J\|\|^2/(2J+1) N_J(u), where <v, N_J(u)> = <[v (x) u]_J, B_J(u)>, i.e. (N_J)_n = Σ_N <3 n; 3 N-n \| J N> conj(u_{N-n}) B_J(u)_N.

The item-10 filter keeps J = 0 and J = 6. Both are even; odd J vanish twice over (B_J ≡ 0 and R_J = 0), and J = 2, 4 are killed by R_J = 0. The 100-digit solve also gives N_0 = -(1/sqrt7) f_0 and N_6 = -(sqrt91/7) f_6. Per the instruction, I determined nothing further about span{M_0, M_6} for item 21.

## Where my conventions had to differ from, or go beyond, the extract

| Point | What I did |
| --- | --- |
| Quaternion -> SU(2) matrix map (not fixed by the extract) | q = w+xi+yj+zk -> w I - i(x sigma_x + y sigma_y + z sigma_z), a verified homomorphism. Any other identification conjugates Gamma inside SU(2), which leaves every reported dimension, norm, w_K, kappa_K and beta unchanged. |
| Spin-j realisation | v_m = sqrt(C(2j, j+m)) x^{j+m} y^{j-m}, with J_+ = x d/dy giving the Condon-Shortley coefficients (verified). The monomial basis is used internally for exactness. |
| eps_j at half-integer j (item 16) | eps_j = 1. The norms do not depend on eps_j. At j = 3 I used eps_3 = -1 as the extract says. |
| Item 4 "diagonal" | Read as M_6(v_m) proportional to v_m. I also report that A_6(v_m) is a diagonal matrix. |
| Item 14 uniqueness (underdetermined) | M_0..M_6 span only the 4-dim Hom(Sym^2 V_3 (x) conj V_3, V_3) (rank 4 by a 60-digit SVD; M_1, M_3, M_5 have rank 2), so there are 3 linear relations and "the coefficient of each M_K" is not unique in general. I report the coefficients produced by the Peter-Weyl expansion (kappa_K proportional to \|\|R_K\|\|^2). Relative to {M_0, M_6} alone they are unique, since those two are independent. The expansions of every M_K in the f_L basis are in the JSON. |
| Item 19 "projection onto V_8" | V_8-isotypic projection (see the reading above). |
| Item 21 "quartic built from psi psi^T" | Read as \|psi psi^T\|^2, with the interaction taken as the level-6 projection of (psi psi^T) conj(psi) (its conj(psi)-derivative up to a factor 2). |
| Item 9 affine slice | The real (x, y) plane of the chart u_3 = u_{-3} = 1. |

Numbers produced along the way that were wrong or artefacts. All are kept in the JSON; none is used.
- sympy `LUsolve` on the surd matrix returned nonsense fractional-power coefficients for N_6 in the f-basis (flagged `agrees_with_100digit_solve: false`). They are discarded in favour of the 100-digit solve, whose residual is below 1e-100.
- An earlier sympy `evalf` gave a "residual" of 1.5e-19 on an unsimplified surd sum. That was cancellation below evalf's precision cap, not a genuine residual (the later 100-digit solve gave < 1e-100).
- An early run labelled m-order wrongly (descending order under an ascending label). It was fixed before any value was used, and item 0(a) was re-verified.

## CONSULTED-MATERIAL MANIFEST

| Kind | Item |
| --- | --- |
| Files read | `handout.md`; my own outputs `log19.txt`, `logq.txt`, `audit_res_su2.json`. Nothing outside the room. One tool call's stdout was auto-saved by the harness to a file outside the room; I did not open it and re-read the in-room JSON instead. |
| Textbook facts from memory | Racah closed formula for CG coefficients and the Condon-Shortley phase; Clebsch-Gordan series; Casimir-polynomial projectors; SU(2) characters as Chebyshev U_{2j}(cos theta); invariant inner product on Sym^n with <x^k y^{n-k}, same> = 1/C(n,k); Schur's lemma and Schur orthogonality of matrix coefficients; Peter-Weyl; Haar measure in Euler angles and Gauss-Legendre quadrature; Frobenius-Schur indicator theorem (real type iff +1); symmetric criticality (Palais); Wirtinger calculus; chain rule for the Jacobian determinant; Leibniz rule; C[x,y] is a domain; Groebner-basis elimination. |
| Supplied as input by the extract | The classical Jacobian fact for binary forms (used in item 19), and all conventions of the extract. |
| Group theory: DERIVED by computation | \|Gamma\| = 120, perfectness, element orders, classes, characters of every constituent (items 0c, 10, 11, 12, 16), invariant dimensions (item 10, two methods), the decomposition 3 + 4 with multiplicities (item 11, commutant dimension plus character norm), FS indicators, and all \|\|M_K(P)\|\|^2. |
| Group theory: RECOGNISED from prior knowledge | Only as expectations, never used as input: that q1, q2 generate the binary icosahedral group 2I (order 120, perfect), which is why item 0(c) matched expectation; that 2I-invariant binary forms first occur in degrees 12, 20, 30 (Klein), so K = 6 is the first nontrivial invariant; that the l = 3 representation restricts to 3 + 4 under the icosahedral group; and the octahedral/hexagonal stabiliser reasoning behind the symmetric-criticality explanation in item 6. The criticality itself was established by the exact gradient computation. |
