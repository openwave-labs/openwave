# RETURN — spin-3 quartic r̂₆: fixed spaces, critical orbits, Hessians, global extrema

Every number below comes from a script in this room; `./py run_all.py` reruns everything in order
(item0 → item1 → item2 → item3 → item4 → item5 (also produces items 6, 7) → item5b → item8 → item9 →
item10 → item11 → item12 → defects → collect) and rewrites `out/*.json`, `out/*.log` and `results.json`.
Last full run: every script exit 0, 0 FAIL lines, 13/13 planted defects fired (`out/run_all.log`).

Conventions used throughout: basis order `v3, v2, v1, v0, v-1, v-2, v-3`; real coordinates
`x = (Re c, Im c)` so that `Re<u,w>` is the Euclidean dot product; `N(u) = Σ_Q |ρ₆(u)_Q|²`, so
`r̂₆ = N/‖u‖⁴`. Orbit labels used everywhere:

| label | unit representative |
|---|---|
| v3, v2, v1, v0 | the basis vectors |
| xyz | (v2 − v-2)/√2 |
| cat | (v3 + v-3)/√2 |
| A* | (2 v1 + v-2)/√5 |
| D* | (√13 v2 + 2√3 v-3)/5 |
| F* | (√6 v0 + i√5 (v2 + v-2))/4 |
| G* | √(23/43) v0 + i√(10/43) (v3 − v-3) |

**Exactness policy.** An "exact" value was produced by exact arithmetic (sympy rationals and radicals, or
Python `Fraction`) and also agreed with at least one independent high-precision route (mpmath, 50 and
80 digits). No value was identified from numerics alone, so no denominator bounds are needed. Nothing that
was asked for is missing (see item 9).

**Three independent routes to r̂₆ exist in the room:**
(i) the Clebsch–Gordan route, with coefficients built by me;
(ii) a Casimir-projector route that uses no CG coefficients at all, `P₆ = Π_{K≠6}(C−K(K+1))/(42−K(K+1))` on V⊗V;
(iii) the rational "a-coordinates" `a_m = √C(6,3+m)·c_m`, in which
`N = Σ_Q |Σ_{m1} (−1)^{3−Q+m1} a_{m1} ā_{m1−Q}|² / C(12,6+Q)` has rational coefficients. The identity
`Ñ(Sx) = N(x)` is verified exactly as a polynomial identity in `item5.py`.

---

## Item 0

| quantity | value | route / precision |
|---|---|---|
| ⟨3 3; 3 −3 \| 6 0⟩ | **1/√924 = √231/462** (square 1/924) | exact; built by lowering from ⟨33;33\|66⟩=+1 |
| Θ(Θu) | **= u** | exact symbolic: (−1)^{3−m}(−1)^{3+m} = +1 |
| r̂₆(2v3 + v1 − 3v-2) | **1977/8624** (N = 1977/44, ‖u‖² = 14) | exact rational; CG route error 3.3e-52 at 50 digits and 2.6e-82 at 80; Casimir route 6.7e-52 and 7.9e-82 |
| r̂₆(v3 + i v0 + 2v-1) | **6329/33264** (N = 6329/924, ‖u‖² = 6) | exact rational; CG route 0 and 0; Casimir route 6.7e-52 and 2.6e-82 |

**Clebsch–Gordan coefficients.** I built them from the convention alone (`common.cg_lowering`). For each L,
the vector |L L⟩ spans the M = L weight space orthogonal to all |L' L⟩ with L' > L. Its sign is fixed by
⟨3 3; 3 L−3 | L L⟩ > 0, and the lower states come from repeated J₋. The arithmetic is exact rational, done in
the rescaled basis f_m = v_m/√((3+m)!/(3−m)!), where J₋ f_m = f_{m−1}.

The table (231 entries) was checked three ways:
- against Racah's closed formula, which I typed in from memory;
- against sympy's library routine `sympy.physics.quantum.cg.CG`, **called as a lookup**: it agrees on all 231 entries;
- for orthonormality (unitarity).

All L = 6 coefficients equal √(C(6,3+m1)C(6,3+m2)/C(12,6+Q)) > 0. Further checks: Σ_L r_L = 1 and r_0 = 1/7 at both points.

---

## Item 1 — fixed spaces of complex dimension 1 and 2

**How I enumerated.** V^{(gHg⁻¹, χ∘Ad_g⁻¹)} = D(g)V^{(H,χ)}, so one subgroup per conjugacy class suffices.
I took the classification of closed subgroups of SO(3) as known, recalled rather than derived:
{1}, C_n, D_n (n ≥ 2), T, O, I, SO(2), O(2), SO(3).

- **Finite groups:** C_n (n ≤ 12), D_n (2 ≤ n ≤ 12), T, O, I (`item1.py`).
  - The group is generated as 3×3 matrices (orders checked: n, 2n, 12, 24, 60).
  - **Every** 1-dim character is found by brute force over generator values (roots of unity), keeping only
    assignments consistent with every product in the Cayley table. The counts match |H/[H,H]|:
    n, 2 or 4, 3, 2, 1.
  - Dimensions come from the character formula (1/|H|)Σ χ̄(h)(1 + 2cos θ + 2cos 2θ + 2cos 3θ); the largest
    distance from an integer is 1.6e-15 in double precision (these are integer dimensions).
  - For C_n and D_n (axis z, flip R_y(π)), the fixed spaces are also computed **exactly** as null spaces.
- **n > 12:** for C_n with n ≥ 7 the fixed space is span{v_m : m ≡ k mod n}, which is at most one vector
  because |m − m'| ≤ 6 < n. D_n ⊂ C_n·flip, so its fixed spaces are at most 1-dim and lie inside those.
- **Continuous subgroups (covered):**
  - SO(2)_z with χ_k(R_z θ) = e^{−ikθ}: fixed space = ker(J_z − k), i.e. span{v_k} for |k| ≤ 3.
  - O(2)_z: a continuous character must be trivial on SO(2), because R_z(θ) is conjugate to R_z(−θ); ±1 on flips.
    Fixed spaces are 0 for the trivial character and span{v0} for the "det" character.
  - SO(3): only the trivial character; fixed space 0.
  - All of these were checked exactly.
- **Completeness checks (exact):**
  - Every 1-dim C_n/D_n fixed space with n ≤ 12 (66 of them) is literally one of v_{±m}, v2±v-2, v3±v-3.
  - Every 2-dim one is one of the plane representatives below, or an exhibited rotation image of one.
  - T (trivial χ), O (sign χ) and D2 (trivial χ) all give the line [v2 − v-2].

**What would fail if a conjugacy class were omitted:**
- Without D2, class F would be missed; without D3, class G.
- Without C5, class D would be missed; without C3, C4 or C6, classes A, B/C or E.
- Without T or O, nothing new would be lost (the xyz line also comes from D2 and D4), but its full stabilizer O would be mis-stated.
- Without SO(2) or O(2), the v_m lines still appear via C_n (n ≥ 7) and D_n, but their stabilizers would be wrong.
- Omitting I loses nothing, since all its fixed spaces are 0.
- The critical orbits of items 2–4 are exactly the critical points inside these spaces, so a missed space means missed orbits.

**Fixed-space dimensions other than 1 and 2:**

| dimension | where it occurs |
|---|---|
| **0** | T with χ ≠ 1; O with trivial χ; I; SO(3); O(2) with trivial χ; D_n (n ≥ 4) for several characters; C_n (n ≥ 8) for some residues |
| **3** | C2 trivial (span{v2, v0, v-2}); C3 trivial (span{v3, v0, v-3}) |
| **4** | C2 nontrivial (span{v3, v1, v-1, v-3}) |
| **7** | trivial group |

**Classes of complex dimension 2 (7 classes).** Each representative W is verified exactly to equal V^{(H,χ)}.
"Stabilizer" means the largest subgroup acting on W by a character.

| class | representative W | (H, χ) producing it | stabilizer | range of r̂₆ on ℙ(W) (exact) |
|---|---|---|---|---|
| A | span{v1, v-2} | C3_z, χ(R_z 2π/3) = e^{−2πi/3} | C3 | [3/77, 9/35] |
| B | span{v1, v-3} | C4_z, χ(R_z π/2) = −i | C4 | [1/924, 75/308] |
| C | span{v2, v-2} | C4_z, χ(R_z π/2) = −1 | C4 | [3/77, 24/77] |
| D | span{v2, v-3} | C5_z, χ(R_z 2π/5) = e^{−4πi/5} | C5 | [1/924, 9/35] |
| E | span{v3, v-3} | C6_z, χ(R_z π/3) = −1 | C6 | [1/924, 463/924] |
| F | span{v0, (v2+v-2)/√2} | D2 (axes x,y,z), χ(R_z π)=1, χ(R_x π)=χ(R_y π)=−1 | D2 | [75/308, 463/924] |
| G | span{v0, (v3−v-3)/√2} | D3 (axis z, flip R_y π), χ = sign | D3 | [200/903, 463/924] |

Rotation images exhibited exactly:
- R_y(π) maps the C3 (k=2), C4 (k=3) and C5 (k=3) spaces onto A, B and D.
- R_{(111)}(±2π/3) maps F onto the two other D2-character spaces, span{v3+v-3, v1+v-1} and span{v3−v-3, v1−v-1}.

**Classes of complex dimension 1 (6 classes).** Invariants r_L = ‖P_L(u⊗Θu)‖²/‖u‖⁴, L = 1..6, all exact:

| class | (H, χ) | stabilizer | r̂₆ | r_1 … r_5 |
|---|---|---|---|---|
| v3 | SO(2)_z, e^{−3iθ} | SO(2) | 1/924 | 9/28, 25/84, 1/6, 9/154, 1/84 |
| v2 | SO(2)_z, e^{−2iθ} | SO(2) | 3/77 | 1/7, 0, 1/6, 7/22, 4/21 |
| v1 | SO(2)_z, e^{−iθ} | SO(2) | 75/308 | 1/28, 3/28, 1/6, 1/154, 25/84 |
| v0 | O(2)_z, det-type | O(2) | 100/231 | 0, 4/21, 0, 18/77, 0 |
| xyz | O, sign (A2) character; also T trivial, D2 trivial, D4 | O | 24/77 | 0, 0, 0, 6/11, 0 |
| cat | D6, χ(R_z π/3) = −1, χ(R_y π) = +1; also D3 trivial | D6 | 463/924 | 0, 25/84, 0, 9/154, 0 |

Rotations used: v_{−m} = R_y(π)v_m; [v3−v-3] = R_z(π/6)[v3+v-3]; [v2−v-2] = R_z(π/4)[v2+v-2]. All exhibited exactly.

**Stabilizers (argument).**
- Lines [v_p], p ≠ 0: if D(g)u = χu then R(g)⟨J⟩_u = ⟨J⟩_u, because D(g)†J D(g) = R(g)J. Since ⟨J⟩_{v_p} = p ẑ, g ∈ SO(2)_z.
- Line [v0]: ⟨J_iJ_j⟩ = diag(6,6,0), so g preserves the z-axis and g ∈ O(2)_z.
- Planes A–E: scalars on W must fix [v_p], giving g = R_z(θ); then e^{−ipθ} = e^{−iqθ} gives C_{|p−q|}.
- Planes F, G: g ∈ O(2)_z. The exact action on (v2+v-2) or (v3−v-3) leaves D2 or D3 (worked out in `item9`/`item1` comments and checked).
- xyz: the stabilizer contains O, and the only closed subgroups containing O are O and SO(3); SO(3) fixes no line.
- cat: the stabilizer contains D6. A larger closed group would be D_{6k} (k ≥ 2), O(2) or SO(3), whose fixed lines are v_m-type (different invariants) or 0.

**Pairwise distinctness.** If gW = W', then Stab(W') = g Stab(W) g⁻¹, and the image set r̂₆(ℙ(W)) is preserved
(r̂₆ is rotation invariant).
- The seven planes have pairwise non-isomorphic stabilizers (C3, C4, C4, C5, C6, D2, D3), except **B and C, which share C4**.
- **B vs C** — two exact separations:
  - *Group-theoretic.* g must normalize C4_z, so g ∈ O(2)_z, which sends χ to χ or χ̄. χ_B has order 4 (value −i); χ_C has order 2 (value −1).
  - *Invariant.* The r̂₆-ranges [1/924, 75/308] and [3/77, 24/77] differ; they are exact rationals from the complete critical sets of item 2.
- All seven r̂₆-ranges are pairwise distinct, so every plane pair is also separated this way.
- **Lines with the same stabilizer (v3, v2, v1, all SO(2)):** separated exactly by r_1 = 9/28, 1/7, 1/28 (equivalently by |m|). All six lines have pairwise distinct exact (r_1..r_6).

---

## Item 2 — r̂₆ on the planes; complete critical sets

**Chart:** u = a + z b with (a, b) the orthonormal basis in the table above, z = x + iy. The chart omits the point **[b]**.
r̂₆ = N(z)/(1+|z|²)², with N exactly:

| class | 924·N(z) (t = \|z\|²) | critical set on ℙ(W) = S² (chart points + omitted point) |
|---|---|---|
| A | 36t² + 576t + 225 | z=0 [v1], 75/308, local min; **circle t = 1/4**, 9/35, circle of maxima; [v-2], 3/77, local min |
| B | t² + 450t + 225 | z=0 [v1], 75/308, **degenerate** (Hessian 0; r̂₆ = 225/924 − (224/924)t²/(1+t)²), isolated max; [v-3], 1/924, min. No other critical points |
| C | 36t² + 1080t + 36 | z=0 [v2], 3/77, min; **circle t = 1**, 24/77, maxima; [v-2], 3/77, min |
| D | t² + 912t + 36 | z=0 [v2], 3/77, min; **circle t = 12/13**, 9/35, maxima; [v-3], 1/924, min |
| E | t² + 1850t + 1 | z=0 [v3], 1/924, min; **circle t = 1**, 463/924, maxima; [v-3], 1/924, min |
| F | 16(18t² + 71x² + 15y² + 25) | z=0 [v0], 100/231, saddle; z = ±i√(5/3), 75/308, minima; z = ±√(3/5), 463/924, maxima; omitted [(v2+v-2)/√2], 24/77, saddle |
| G | 463t² + 296x² − 40y² + 400 | z=0 [v0], 100/231, local max; z = ±i√(20/23) = ±2i√115/23, 200/903, minima; z = ±2/√5, 24/77, saddles; omitted [(v3−v-3)/√2], 463/924, max |

The chart Hessians, with det and trace, are recorded in `out/item2.json`.

**Why each set is complete (an elimination argument, no solver needed).**
1. *A–E.* W is spanned by two J_z-eigenvectors, and R_z(θ) maps v_p + z v_q to e^{−ipθ}(v_p + e^{−i(q−p)θ}z v_q).
   Rotation and phase invariance therefore make r̂₆ a function f(t) of t = |z|² alone.
   Then ∇f = 2f'(t)(x, y), so the critical points are z = 0 and the zeros of f'(t) with t > 0.
   The numerator of f' is linear in t: −21(4t−1), −224t, −84(t−1), −35(13t−12), −924(t−1).
   So there is at most one circle, and none for B.
2. *F, G.* (1+t)³∂_x f = x·p1 and (1+t)³∂_y f = y·q1, with
   p1 − q1 = (64/33)(1+t) for F and (8/11)(1+t) for G. This is never zero.
   So there are no critical points with xy ≠ 0. On y = 0 the condition is p1(x,0) = 0; on x = 0 it is
   q1(0,y) = 0. Each is linear in x² or y²: F gives 5x² = 3 and 3y² = 5; G gives 5x² = 4 and 23y² = 20.
3. *The omitted point.* In the swapped chart u = b + w a the gradient at w = 0 is exactly 0.

Independent confirmations:
- a lex Gröbner basis of the gradient numerators (an exact elimination algorithm) gives the same solution sets;
- the exact gradient vanishes at every listed point;
- a 60-digit Newton search from 49 seeds in each chart found no |z|² outside the lists.

**What would fail if I relied on a solver alone.**
- A numerical solver only finds the basins it is seeded into. It cannot certify completeness.
- It returns scattered points on the critical circles, which are not isolated.
- It converges poorly at the degenerate point v1 in B (Hessian 0).
- It never sees the omitted point unless a second chart is used.
- A symbolic solver applied blindly to A–E returns positive-dimensional components that must be recognized as circles.

**Index check (Poincaré–Hopf on S²).** The index sum over isolated critical points is 2 in every class; circles contribute
χ(S¹) = 0, and the degenerate v1 in B has index +1 because its gradient is radial.
- *What it can detect:* a missing or extra isolated critical point that changes the index sum. For example,
  dropping one minimum in F makes the sum ≠ 2 (planted defect D4b fires).
- *What it cannot detect:* a missing saddle–extremum pair (net index 0); anything on circles. It also assumes the
  indices are computed correctly — degenerate points need their own index computation, as for v1 in B.

Consistency with item 5b: the chart Hessian at every interior critical point equals H_u on (u_⊥, i u_⊥)/(1+|z|²)²
**exactly** (Fubini–Study factor; `item5b.py`).

---

## Item 3 — the six line classes are critical

Values: v3 **1/924**, v2 **3/77**, v1 **75/308**, v0 **100/231**, xyz **24/77**, cat **463/924**.
The exact tangential gradient ∇N(u) − 4N(u)u is **0** at all six (`item3.py`). Control: at
(v3+v1−v-2)/√3 it is 2√1002/297 ≠ 0.

**Argument (symmetric criticality).** Hypotheses:
- (H1) r̂₆ is C¹ on V∖{0};
- (H2) r̂₆ is invariant under U(1)×SO(3), which acts by unitary, hence Re⟨,⟩-orthogonal, maps (checked exactly in item 9);
- (H3) r̂₆ is homogeneous of degree 0;
- (H4) V^{(H,χ)} = ℂu is exactly one complex dimension (item 1).

Steps:
1. Let K_u = {(χ(h)⁻¹, h)}. Each element acts by an orthogonal map A with Au = u. Differentiating
   r̂₆(Aw) = r̂₆(w) gives ∇r̂₆(Au) = A∇r̂₆(u), so ∇r̂₆(u) is fixed by K_u. Hence ∇r̂₆(u) ∈ V^{(H,χ)} = span_ℝ{u, iu}.
   *Fails without (H2):* for a non-orthogonal action the gradient is not equivariant.
2. By (H3), Re⟨u, ∇r̂₆(u)⟩ = d/dt r̂₆((1+t)u) = 0.
   *Fails without (H3):* a radial component would survive. On the sphere only the tangential part matters, but the argument would need rephrasing.
3. By phase invariance, Re⟨iu, ∇r̂₆(u)⟩ = d/dφ r̂₆(e^{iφ}u) = 0.
   *Fails without phase invariance:* an iu-component could survive.
4. Hence ∇r̂₆(u) = 0, and u is critical on the unit sphere.
   *Fails without (H4):* with a 2-dim fixed space the gradient could point along the extra direction. That is why the planes need item 2.

The same argument, applied with W = V^{(H,χ)} of dimension 2, shows that **a critical point of r̂₆|ℙ(W) is critical
on the whole sphere**: the gradient lies in W ∩ T_u and its W-component vanishes. This is what lets item 4 unite
items 2 and 3.

---

## Item 4 — distinct critical orbits (10)

| orbit | stabilizer | r̂₆ | 924·r̂₆ | occurs as (items 2, 3) |
|---|---|---|---|---|
| v3 | SO(2) | 1/924 | 1 | line class; poles of B, D, E (as v-3) |
| v2 | SO(2) | 3/77 | 36 | line class; poles of A (v-2), C, D |
| v1 | SO(2) | 75/308 | 225 | line class; poles of A, B |
| F* | D2 | 75/308 | 225 | z = ±i√(5/3) in F |
| v0 | O(2) | 100/231 | 400 | line class; z = 0 of F and G |
| xyz | O | 24/77 | 288 | line class; circle of C; omitted point of F; z = ±2/√5 in G |
| cat | D6 | 463/924 | 463 | line class; circle of E; z = ±√(3/5) in F; omitted point of G |
| A* | C3 | 9/35 | 1188/5 | circle \|z\|² = 1/4 of A |
| D* | C5 | 9/35 | 1188/5 | circle \|z\|² = 12/13 of D |
| G* | D3 | 200/903 | 8800/43 | z = ±i√(20/23) in G |

All values are exact and match the 50-digit route to better than 1e-45.

**Same orbit — decided by an exhibited rotation** (each verified exactly as an equality of complex lines, `item4.py`):
- cat → F point z=√(3/5): R_y(π/2)R_z(π/2).
- xyz → G point z = 2/√5: R_z(5π/3)R_y(arccos(1/√3))R_z(π/4), built exactly with cos β = 1/√3.
- z → −z in F: R_z(π/2). z → −z in G: R_z(π/3).
- Omitted point of F → xyz: R_z(π/4). Omitted point of G → cat: R_z(π/6).
- v_{−m} → v_m: R_y(π).
- Circle points: R_z(θ) moves along each circle (checked at θ = 2π/7).
- Control: R_y(π) does *not* map cat onto the F point, so the test can fail.

**Same value, two orbits — decided by a rotation-invariant quantity:**
- **v1 vs F*** (75/308): r_1 = 1/28 vs 0. Exact.
- **A* vs D*** (9/35): all quartic invariants r_1..r_6 coincide (1/175, 12/175, 1/6, 11/350, 172/525, 9/35). They are separated by:
  - the cubic invariant tr(ρ₂³), with ρ₂ the rank-2 part of |u⟩⟨u| (built with the adjoint Casimir): **−132/42875 vs +132/42875**, exact;
  - their exact Hessian spectra (item 5), which also differ;
  - their stabilizers, C3 vs C5. A closed group containing both orders 3 and 5 would be C_{15k}, D_{15k}, I, SO(2), O(2) or SO(3). None of these has a fixed line other than the v_m (I, being perfect, has none).
- Stabilizers of the interior points are exact. A larger stabilizer H' ⊋ H would give V^{(H',χ')} ⊊ W, since W's own stabilizer is exactly H. That forces [u] to be a line class, but the r_L of A*, D*, F*, G* differ from all six line classes.

---

## Item 5 — restricted Hessian at each orbit

**Formula.** For unit u and e ∈ T_u, **H_u(e,e) = eᵀ M_u e** with **M_u = Hess N(u) − 4N(u)·I** (real 14×14, Euclidean = Re⟨,⟩).

*Proof.* Let γ(s) = u cos s + e sin s, so γ(0) = u, γ'(0) = e, γ''(0) = −u, and ‖γ‖² = 1 + (‖e‖²−1)sin² s because Re⟨u,e⟩ = 0.
Write r̂₆(γ) = N(γ)/D(s) with D = ‖γ‖⁴. Then D(0) = 1, D'(0) = 0 and D''(0) = 4(‖e‖²−1).
N is homogeneous of degree 4, so Euler's relation gives ∇N(u)·u = 4N(u). Hence
(N∘γ)''(0) = Hess N(u)[e,e] + ∇N(u)·γ''(0) = Hess N[e,e] − 4N.
The quotient rule with D(0) = 1 and D'(0) = 0 gives r̂₆(γ)''(0) = (N∘γ)'' − N·D''(0) = Hess N[e,e] − 4N‖e‖². ∎

- **Where degree-0 homogeneity enters.** r̂₆ = N/‖u‖⁴ is a degree-4 form over the fourth power of the norm.
  This is what produces the −4N·D'' term, and it is why the curve may leave the sphere when ‖e‖ ≠ 1: only its projectivisation matters.
  The same fact means a straight line u + se gives the same second derivative, which is why my first "planted defect" D11 was not a defect (see below).
- **Annihilation of O_u.** For a skew generator X ∈ {i, −iJ_x, −iJ_y, −iJ_z}, differentiate ∇N(w)·Xw = 0 in w.
  This gives Hess N(u)Xu = X∇N(u), so M_u Xu = X(∇N(u) − 4N(u)u) = X·g_u, with g_u the tangential gradient.
  **At a critical point M_u annihilates O_u; at a non-critical point the residual is ‖X g_u‖.**

**Results.**
- Residual ‖M_u d‖ on O_u: exactly 0 for all four generators at all 10 orbits.
- Numerically ≤ 1.6e-50 at 50 digits and ≤ 1.3e-80 at 80 digits.
- The check fails where it should: at (v3+v1−v-2)/√3 the residuals are 2√1002/297, √(5560+416√15)·√3/297,
  √(5560−416√15)·√3/297 and 4√921/297 (item 11).

Exact characteristic polynomials of H_u|N_u are monic, so each factor below is shown up to its positive leading
constant. They were computed in the a-coordinates over ℚ, ℚ(√10), ℚ(√26) or ℚ(√46) (DomainMatrix). The 50- and
80-digit eigenvalues of M_u restricted to an orthonormal N_u in the original coordinates match the exact roots to
≤ 3.2e-50 and ≤ 1.7e-80.

| orbit | dim O_u | dim N_u | (n₋, n₀, n₊) | characteristic polynomial (monic in λ; factors shown up to positive constants) |
|---|---|---|---|---|
| v3 | 3 | 10 | (0,0,10) | (λ−4)²(11λ−3)²(11λ−2)²(33λ−65)²(33λ−32)² |
| v2 | 3 | 10 | (2,0,8) | (3λ−4)²(11λ−24)²(11λ−20)²(11λ−12)²(33λ+8)² |
| v1 | 3 | 10 | (4,**2**,4) | λ²(3λ+5)²(11λ−3)²(363λ²−374λ−600)² |
| v0 | 3 | 10 | (8,0,2) | (11λ−8)²(11λ+12)²(11λ+20)²(33λ+40)²(33λ+100)² |
| xyz | 4 | 9 | (6,0,3) | (11λ+8)³(11λ+24)³(33λ−40)³ |
| cat | 4 | 9 | (9,0,0) | (λ+4)(11λ+10)²(11λ+15)(11λ+23)(33λ+64)²(33λ+67)² |
| A* | 4 | 9 | (5,0,4) | (55λ+24)(16471125λ⁴+9583200λ³−45776720λ²−23015168λ+998400)² |
| D* | 4 | 9 | (5,0,4) | (55λ+24)²(55λ+104)(165λ−4)²(1815λ²−1012λ−3360)² |
| F* | 4 | 9 | (5,0,4) | (11λ−10)(11λ+7)(33λ−64)(31944λ³+46948λ²−13530λ−7125)² |
| G* | 4 | 9 | (3,0,6) | (11λ−8)(473λ−920)(473λ+696)(1419λ−560)²(671187λ²+438944λ−339200)² |

The exact monic forms with their normalizing constants are in `out/item5.json`. Signatures come from exact
real-root isolation of each rational factor.

---

## Item 5b — tangents of the plane at interior orbit points

**Reading.** "Interior point of a class of complex dimension 2" means a point whose stabilizer is exactly the
plane's: A* ∈ A, D* ∈ D, F* ∈ F, G* ∈ G. The same data is also given for xyz and cat at their places inside C/G and E/F.

**Tangent normalization.** The tangent space of W at u inside T_u is {w ∈ W : Re⟨u,w⟩ = 0}, which has **3 real directions**.
With u = (a+zb)/√(1+|z|²) and u_⊥ = (−z̄ a + b)/√(1+|z|²), I use t1 = iu, t2 = u_⊥, t3 = iu_⊥. These are
Re⟨,⟩-orthonormal and orthogonal to u (checked exactly), i.e. already unit vectors. For every non-null
direction the projection onto N_u also has norm exactly 1, so "unit tangent" and "unit projection" give the same numbers.

| point | t1 = iu | t3 = iu_⊥ | non-null block H_u (exact) | Gram (Re⟨,⟩) |
|---|---|---|---|---|
| A* (z=1/2) | projection 0; = [iu]; H = 0 | projection 0; = (1/3)[iu] + (5/6)[−iJ_z u]; H = 0 | t2: **−24/55** | [1] |
| D* (z=2√39/13) | 0; = [iu]; H = 0 | 0; = (−√39/39)[iu] + (5√39/78)[−iJ_z u]; H = 0 | t2: **−104/55** | [1] |
| F* (z=i√(5/3)) | 0; = [iu]; H = 0 | non-null | (t2,t3): **[[64/33, 0],[0, 10/11]]** | I₂ |
| G* (z=2i√115/23) | 0; = [iu]; H = 0 | non-null | (t2,t3): **[[8/11, 0],[0, 920/473]]** | I₂ |
| xyz in C (z=1) | 0; H = 0 | 0; = (1/2)[−iJ_z u]; H = 0 | t2: −24/11 | [1] |
| cat in E (z=1) | 0; H = 0 | 0; = (1/3)[−iJ_z u]; H = 0 | t2: −4 | [1] |
| xyz in G (z=2/√5) | 0; H = 0 | non-null | [[40/33, 0],[0, −8/11]] | I₂ |
| cat in F (z=√(3/5)) | 0; H = 0 | non-null | [[−10/11, 0],[0, −64/33]] | I₂ |

Every non-null value is a root of the corresponding item-5 characteristic polynomial.
The 50-digit matrix route agrees to ≤ 1e-50.

---

## Item 6 — Morse indices for E = g·r̂₆

The index counts negative directions of the transverse Hessian on N_u. This is the Morse–Bott index of the orbit
in ℙ(V); orbit and phase directions are null by symmetry. For g > 0 the index is n₋; for g < 0 it is n₊. The kernel is unchanged.

| orbit | index, g>0 | index, g<0 | kernel | status |
|---|---|---|---|---|
| v3 | **0** | 10 | 0 | g>0: strict local (and global) **minimum**; g<0: local (global) maximum |
| v2 | 2 | 8 | 0 | saddle |
| v1 | 4 | 4 | 2 | degenerate saddle (both signs have descending and ascending directions) |
| v0 | 8 | 2 | 0 | saddle |
| xyz | 6 | 3 | 0 | saddle |
| cat | 9 | **0** | 0 | g<0: strict local (and global) **minimum**; g>0: local (global) maximum |
| A* | 5 | 4 | 0 | saddle |
| D* | 5 | 4 | 0 | saddle |
| F* | 5 | 4 | 0 | saddle |
| G* | 3 | 6 | 0 | saddle |

Local extrema of E:
- **g > 0:** minimum at v3, maximum at cat.
- **g < 0:** minimum at cat, maximum at v3.
- No other orbit is a local extremum for either sign. v1 has n₋ = n₊ = 4 whatever its kernel does.

Consistency remark, not a proof of completeness: the Morse–Bott sum Σ(−1)^{index}χ(orbit) uses only the orbits
with continuous stabilizers, since χ(SO(3)/Γ) = 0 for finite Γ. v3 (S²): +2; v2 (S²): +2; v0 (ℝP²): +1.
v1 (S²) contributes +2 if its kernel counts as descending — along the kernel r̂₆ decreases quartically (item 7).
That gives 7 = χ(ℂP⁶).

---

## Item 7 — the kernel at v1

- **Kernel (exact):** span_ℝ{v-3, i·v-3}. It is computed as the exact null space of the restricted form and mapped back to c-coordinates.
- **Structural identification:** this is exactly the tangent space at v1 of **class B** = span{v1, v-3}. The check is exact (`item5.py`).
  - Along B: r̂₆ = 225/924 − (224/924)·t²/(1+t)², t = |z|² (exact identity, item 9). The second-order term cancels and r̂₆ decreases at fourth order.
  - The two kernel directions are rotated into each other by the SO(2)_z stabilizer, which acts on the v-3-plane by e^{4iθ}. That symmetry explains the multiplicity 2, but **not** the vanishing.
  - The kernel is not an orbit direction: at v1, O_u = span{iu, −iJ_x u, −iJ_y u} involves only v2 and v0.
- **Genuine kernel vs a numerical one:**
  1. The exact characteristic polynomial over ℚ has the factor λ² (no rounding anywhere).
  2. The null space is computed in exact arithmetic.
  3. The numerical eigenvalues shrink with the working precision — within 5e-51 of 0 at 50 digits and 2e-81 at 80 digits — rather than stalling at a fixed small size.
  4. There is an exact algebraic explanation (the B-restriction identity).

  A numerical artifact would fail 1, 2 and 4, and would typically show a floor under 3.
- No other orbit has a kernel.

---

## Item 8 — global minimum and maximum over the unit sphere

**Minimum = 1/924, attained at the orbit of v3.** Proved; `item8.py`, exact polynomial identities.
- Hypotheses and where each is verified:
  - (M1) With P_L(u) = ‖P_L(u⊗u)‖², the identity **924 N = ‖u‖⁴ + 1715 P0 + 714 P2 + 77 P4** holds. Exact polynomial identity in 14 real variables.
  - (M2) P_L ≥ 0, being squared norms.
  - (M3) P1 = P3 = P5 = 0 and P0 + P2 + P4 + P6 = ‖u‖⁴ (Sym²V = V6⊕V4⊕V2⊕V0, unitarity). Exact identities.
  - (M4) P0 = P2 = P4 = 0 at v3, giving r̂₆(v3) = 1/924. Exact.
- **Conclusion:** r̂₆ ≥ 1/924, with equality exactly when P0 = P2 = P4 = 0; v3 attains it.
- What fails if a hypothesis is omitted:
  - without M1 the coefficients would only be a numerical fit;
  - without M2 there is no inequality;
  - without M3 the bound cannot be normalized;
  - without M4 it is only a lower bound.

**Maximum = 463/924, attained at the orbit of cat.** Proved by an exact degree-6 certificate.
- Why degree 6 was needed:
  - Degree 4 gives only r̂₆ ≤ 11/21: 11/21‖u‖⁴ − N = (29/18)R1 + (9/22)R5 + (7/18)P4, with R_L = ‖P_L(u⊗Θu)‖² (exact identity).
  - Degree 4 cannot do better. Every R_L is an exact linear combination of P0, P2, P4, P6 (verified), and the point
    (P0,P2,P4,P6) = (1/7, 1/3, 0, 11/21) satisfies every degree-4 constraint with R6 = 11/21 (exact weak-duality check).
- The certificate: **463/924·‖u‖⁶ − N(u)‖u‖² = Σ_b Σ_M κ_{L,M}⁻¹ c̃_{b,M}(a)ᵀ Ỹ_b c̃_{b,M}(a)‾**, where:
  - the c̃_{b,M} are cubic polynomials with rational coefficients in (a, ā), coming from highest-weight vectors of
    Sym³V (Sym³V: multiplicities 1:1, 3:2, 4:1, 5:1, 6:1, 7:1, 9:1) and Sym²V⊗V (1:2, 2:2, 3:4, 4:3, 5:3, 6:2, 7:2, 8:1, 9:1),
    built exactly;
  - each coefficient space is reduced by the kernel forced by the cat orbit;
  - the Ỹ_b are 10 rational symmetric blocks of sizes 1, 1, 2, 2, 3, 2, 2, 1, 1, 1, stored in `out/item8.json`.
- Hypotheses and where each is verified:
  - (X1) The identity holds exactly as a polynomial identity in the 14 symbols (a, ā). The difference polynomial has **0 terms**. Independently, it holds to 7e-27 at 5 random points in the original c-coordinates (30 digits).
  - (X2) Every Ỹ_b is positive definite. Exact LDL pivots, all > 0.
  - (X3) The c̃ have real rational coefficients, so conj(c̃)(a) = c̃ with a ↔ ā swapped, and each term is a Hermitian form ≥ 0. True by construction (Fractions).
  - (X4) a_m = √C(6,3+m)c_m is a bijection with Ñ(a) = N(c) (exact identity, item 5) and ‖u‖² = Σ|a_m|²/C(6,3+m).
  - (X5) r̂₆(cat) = 463/924 (items 3 and 4).
- **Conclusion:** r̂₆ ≤ 463/924 on V∖{0}, attained at cat.
- The representation theory, the barrier-method search (margin 0.077 before rounding to denominators ≤ 10⁶) and the facial reduction only *found* the certificate. Soundness rests on X1–X4 alone.
- What fails if a hypothesis is omitted:
  - X1: an approximate identity proves nothing. Planted defect D9 — one entry perturbed by 1e-6 — is caught.
  - X2: negative terms could appear (D10 is caught).
  - X3: the terms need not be |·|².
  - X4: the certificate would concern a different function.
  - X5: only an upper bound would remain.
  - Lowering λ to 463/924 − 1/1000 makes the exact system inconsistent (D8).
- **Search:** 200 random BFGS multistarts in double precision found only 1/924 and 463/924 as minimum and maximum values.
  This is evidence only: it cannot exclude narrow basins. It is now superseded by the two proofs.
- Not established: whether orbits other than cat also attain 463/924. The certificate forces c̃_{b,M}(u) = 0 at any maximizer; I did not solve that system.

---

## Item 9 — every vanishing quantity in items 2–7

Every zero below was **computed and found zero exactly**. None is "missing". Symmetries used as reasons were first
shown to preserve r̂₆, by exact polynomial identities in `item9.py`:
- **S1:** ρ₆(Θu) = ρ₆(u), because ⟨3m1;3m2|6Q⟩ is symmetric and Θ² = 1. So r̂₆∘Θ = r̂₆ and ‖Θu‖ = ‖u‖.
- **S2:** N(R_z(θ)u) = N(u) for symbolic θ; also for R_y(π), for R_{(111)}(2π/3), and for phase e^{iφ}. SO(2)_z together with R_{(111)}(2π/3), which moves the z-axis to the x-axis, generates a closed subgroup that properly contains SO(2)_z and O(2)_z. The only such subgroup is SO(3). Since N is continuous, this proves invariance under every rotation.

| zero | item | computed? | exact reason |
|---|---|---|---|
| gradient of r̂₆\|ℙ(W) at listed points | 2 | yes, exact | they are exactly the solutions of the eliminated system |
| chart Hessian at v1 in B (det = trace = 0) | 2 | yes, exact | identity 924N(v1+zv-3) = 225(1+t)² − 224t². The t-linear term cancels via 2⟨1,−1⟩⟨−3,3⟩ + ⟨1,3⟩² + ⟨−3,−1⟩² = 2⟨1,−1⟩² (30+420 = 450 in units 1/924). Arithmetic, **not** a symmetry |
| zero eigenvalue of the chart Hessian along circles (A, C, D, E) | 2, 5b | yes, exact | the circle is an R_z orbit (S2) |
| tangential gradient at the six lines | 3 | yes, exact | symmetric criticality (item 3) |
| r_1 = r_3 = r_5 = 0 at v0, xyz, cat | 4 | yes, exact | Θu = ±u, so u⊗Θu is symmetric, and odd-L CG coefficients are antisymmetric (both checked) |
| r_1 = 0 at F*, G* | 4 | yes, exact | ⟨J⟩ = 0 exactly: a vector fixed by D2 or D3 is 0; r_1/\|⟨J⟩\|² = 1/28 (checked at v3, v2, v1) |
| r_2 = 0 at v2 | 4 | yes, exact | SO(2) leaves only the Q=0 quadrupole, ∝ ⟨3J_z²−J²⟩ = 0 at m = 2 |
| r_2 = 0 at xyz; tr ρ₂³ = 0 at v2, xyz | 4 | yes, exact | ρ₂ = 0: O fixes \|u⟩⟨u\|, and V_2 has no O-invariant vector (for v2 see previous row) |
| M_u·(O_u) = 0 at all 10 orbits | 5 | yes, exact | M_u Xu = X g_u and g_u = 0 (item 5) |
| n₀ = 0 except v1 | 5 | yes, exact char. poly | — |
| kernel at v1 (λ² factor) | 5, 7 | yes, exact | the B identity above; multiplicity 2 from SO(2) |
| orbit-null controls H_u(iu) = 0, H_u(iu_⊥) = 0 (circle classes) | 5b | yes, exact | direction ∈ O_u (coefficients computed), and M_u annihilates O_u |
| projection of iu, iu_⊥ onto N_u = 0 | 5b | yes, exact | iu is the phase direction; W is J_z-invariant, so iu_⊥ ∈ span{iu, −iJ_z u} |
| off-diagonal H_u(u_⊥, iu_⊥) = 0 at F*, G*, cat∈F, xyz∈G | 5b | yes, exact | σ = (R_z(π/2) or R_z(π/3) or 1)∘Θ, up to sign, fixes u and acts by ±1 with opposite signs on u_⊥ and iu_⊥; it preserves r̂₆ (S1, S2), so H(u_⊥,iu_⊥) = −H(u_⊥,iu_⊥) |
| Gram off-diagonals | 5b | yes | orthonormal basis by construction |

No zero is left unresolved.

---

## Item 10 — replacing r̂₆ by Q_σ = 1 + w₆ r̂₆ (w₆ = 28/39 or 21/52, both > 0)

**What changes and why.**
- Q_σ is degree-0 homogeneous and invariant, and ∇Q = w∇r̂₆. So **the critical set and the orbits of item 4 are unchanged**.
- Orbit values map by the injective affine map v ↦ 1 + w v. Equal values stay equal and distinct values stay distinct, so the item-4 decisions are unchanged.
- The Hessian scales. With N_Q = ‖u‖⁴ + wN one gets M^Q_u = w M_u + 8uuᵀ (verified exactly at F*), and 8uuᵀ vanishes on T_u. So **H^Q_u = w H_u**.
- Consequences: characteristic polynomials become w^n p(λ/w); **signatures, kernels and Morse indices are unchanged**; orbit-null controls stay 0; Gram matrices are unchanged; item 5b entries are multiplied by w.
- For an energy ∝ g Q_σ, the indices and extrema are those of g w r̂₆ with sign(gw) = sign(g). **Item 6 is unchanged**.
- The global min and max of Q_σ are 1 + w/924 and 1 + 463w/924.

**Orbit values Q_σ:**

| orbit | w = 28/39 | w = 21/52 |
|---|---|---|
| v3 | 1288/1287 | 2289/2288 |
| v2 | 147/143 | 581/572 |
| v1 | 168/143 | 2513/2288 |
| F* | 168/143 | 2513/2288 |
| v0 | 1687/1287 | 168/143 |
| xyz | 175/143 | 161/143 |
| cat | 1750/1287 | 2751/2288 |
| A*, D* | 77/65 | 287/260 |
| G* | 5831/5031 | 609/559 |

**Item 5b numbers × w** (Gram matrices unchanged; orbit-null entries 0 × w = 0):

| point | × 28/39 | × 21/52 |
|---|---|---|
| A* | −224/715 | −126/715 |
| D* | −224/165 | −42/55 |
| F* | diag(1792/1287, 280/429) | diag(112/143, 105/286) |
| G* | diag(224/429, 25760/18447) | diag(42/143, 4830/6149) |
| xyz in C | −224/143 | −126/143 |
| cat in E | −112/39 | −21/13 |
| xyz in G | diag(1120/1287, −224/429) | diag(70/143, −42/143) |
| cat in F | diag(−280/429, −1792/1287) | diag(−105/286, −112/143) |

The characteristic polynomials for Q_σ are listed in `results.json` → `item10`.

---

## Item 11 — the non-critical point u = (v3 + v1 − v-2)/√3

| quantity | exact | minimal polynomial | numeric (50 digits) |
|---|---|---|---|
| 924·r̂₆ | **204** (r̂₆ = 17/77) | — | — |
| ‖tangential gradient‖ | **2√1002/297** | 29403X² − 1336 | 0.21316083220665877418 |
| ‖M_u(iu)‖ | **2√1002/297** | 29403X² − 1336 | 0.21316083220665877418 |
| ‖M_u(−iJ_x u)‖ | **√3·√(5560+416√15)/297** | 864536409X⁴ − 326961360X² + 28317760 | 0.49385438776155313541 |
| ‖M_u(−iJ_y u)‖ | **√3·√(5560−416√15)/297** | same | 0.36647038777189921004 |
| ‖M_u(−iJ_z u)‖ | **4√921/297** | 29403X² − 4912 | 0.4087270277574011633 |

**What it tells me about item 5.**
- M_u Xu = X g_u exactly: the script checks ‖M_u Xu‖ = ‖X g_u‖ for all four generators. For X = i this is ‖g_u‖ itself.
- So, given the correct formula, the O_u-annihilation check of item 5 is **equivalent to criticality**.
- Here it fails, with residuals of size ~0.2–0.5, and at the ten critical orbits it passes with exact 0. It is a genuine test that can fail.
- It also tests the formula: without the −4N term the residual at a critical point would be 4N‖Xu‖ ≠ 0 (defect D5).
- It says nothing about the transverse block H_u|N_u. That block is cross-checked by the independent routes of items 5 and 12.

---

## Item 12 — two routes for H_u along a direction in N_u

The orbit is **F***. Two directions:
- e1 = u_⊥, which lies in N_u exactly;
- e2 = the normalized exact projection onto N_u of w = v3 + 2v2 − v1 + 3i v0 + v-2 − 2i v-3 (a generic direction).

| direction | exact H_u(e,e) (both routes symbolically equal) | 50 digits: \|FD − matrix\| | 80 digits: \|FD − matrix\| |
|---|---|---|---|
| e1 | **64/33** | 8.0e-51 | 4.2e-81 |
| e2 | (6273629725040/8768327741481) + (1625311542060√2 − 2156748448066√15 + 4831929869904√30)/43841638707405 ≈ 1.18105086435313888201626959099; minimal polynomial 133920642469381146225X⁴ − 383274233428896216000X³ + 303279115281209333640X² − 35111146438706046720X − 10723003823272339184 | 5.4e-51 | 4.2e-81 |

**Routes and their precision.**
- *Route 1 (second derivative of s ↦ r̂₆(u cos s + e sin s) at 0).* (1a) exact symbolic differentiation; (1b) `mpmath.diff` at 50 and 80 digits, error vs exact ≤ 2.7e-51 and 4.2e-81.
- *Route 2 (eᵀM_u e with M_u = Hess N − 4N·I).* Exact, and at 50 and 80 digits, error vs exact ≤ 5.4e-51 and 2.1e-81.
- **Largest disagreement between the routes: 8.0e-51 at 50 digits, 4.2e-81 at 80 digits.** In exact arithmetic they are identical.

---

## Checks that can fail — planted defects (`defects.py`, 13/13 fired)

Each defect re-runs an item's check on deliberately corrupted input:

| # | corruption | check that fired |
|---|---|---|
| D1 | sign flip of ⟨3 1;3 −1\|6 0⟩ | CG route vs Casimir route (item 0) |
| D1b | same | lowering table vs Racah formula |
| D2 | Θ without its (−1)^{3−m} | invariance under R_{(111)}(2π/3). **Note:** the R_y(π) check alone is blind to this defect, so I added the generic-rotation check to item 9 |
| D3 | point off A's circle | exact zero gradient |
| D4 | F list missing \|z\|² = 5/3 | Newton-search containment |
| D4b | same | Poincaré–Hopf sum |
| D5 | Hessian formula without −4N | O_u annihilation |
| D6 | orbit-null control on u_⊥ | H = 0 |
| D7 | R_y(π)v1 listed as a new orbit | distinct-invariants check |
| D8 | λ = 463/924 − 1/1000 | exact certificate consistency |
| D9 | one Gram entry + 1e-6 | exact identity |
| D10 | Gram blocks negated | exact LDL positivity |
| D11 | central difference with h = 1e-3 | 1e-40 agreement (item 12) |

Other controls built into the item scripts:
- item 3: tangential gradient ≠ 0 at a non-fixed point;
- item 4: a wrong rotation is rejected;
- item 5: O_u residual ≠ 0 at a non-critical point.

D4 and D4b re-evaluate the same predicate on a doctored list rather than re-running the full solver.

---

## Readings I took (underdetermined points)

1. *Item 1:* "classify up to rotation" is about the **subspaces** W, identified when W' = D(g)W. The "stabilizer" of W or of a line is the largest closed subgroup acting on it by a character.
2. *Item 2:* "critical set of the restriction" means critical points of r̂₆ on ℙ(W) ≅ S². Phase circles are divided out.
3. *Item 5b:* "interior point" means stabilizer equal to the plane's (A*, D*, F*, G*). The xyz and cat data are supplementary.
4. *Item 5b:* "normalize the rest" means unit unprojected tangents. Here these coincide with the unit projections.
5. *Item 6:* the Morse index is the transverse (Morse–Bott) index on N_u.
6. *Item 11:* M_u is the c-coordinate matrix of item 5. The item-5 characteristic polynomials were computed in a-coordinates with the metric carried along, which gives the same spectrum; this is verified by the 50/80-digit route in c-coordinates.

## Where my own earlier steps disagreed (reported, then fixed)

- **Item 2:** the Poincaré–Hopf check first FAILED for class B because v1 is a degenerate critical point of the restriction. Fixed by computing its index from the leading order in t (+1), not by relaxing the check.
- **Item 8 exploration:** the first level-2 search claimed feasibility *below* the attained 463/924, which is impossible. Diagnosis: a sign error in my exploration code for tr(YG), found because the identity failed at fresh points. After the fix, the bound is tight at 463/924. The final certificate is built independently and verified exactly.
- **Item 9:** a symmetry check first FAILED for cat∈F and xyz∈G because I had hard-coded which of u_⊥ and iu_⊥ gets the minus sign. Generalized to either sign.
- **Defects:** D2 first MISSED (see above); D11 first MISSED because it was not a defect (homogeneity). Both were replaced; the misses are reported here.
- **run_all.py:** it cannot spawn `./py` from inside `./py`. The room's sandbox refuses nested launches (exit 65), so it runs the scripts in-process with `runpy` under the single `./py` interpreter.

## Tools

- Python 3.12.14 via `./py`, with sympy 1.14.0, mpmath 1.3.0, numpy 2.5.3 and scipy 1.18.1.
- Single thread; no multiprocessing.
- The room also contains exploration scripts in `scratch/` (LP, SDP and restriction explorations). No reported number rests on them alone: everything in this file is regenerated and verified by the item scripts.

---

## Consulted-material manifest

**Files I read** (all inside the room):
- `brief.md` and `worklist.md` (read first, in that order);
- my own `sos.py`, re-read to repair a refactor;
- my own scripts, whose contents the harness echoed back after edits: `common.py`, `item0.py`, `item1.py`, `item2.py`, `item5.py`, `item5b.py`, `item8.py`, `item9.py`, `item11.py`, `item12.py`, `defects.py`, `sos.py`, `scratch/sdp_explore.py`;
- my own outputs `out/item0.json`, `out/item5.json`, `out/item8.json`, `out/item12.json`, printed via `./py`.

**Present but not opened:**
- `py` (426 bytes; used only as the interpreter);
- `.room.sb` (779 bytes; its name appeared in a sandbox error message when a nested `./py` launch was refused).

**Loaded without my asking:**
- the Python standard library and the packages above, from the interpreter's environment outside the room (its path appeared in tracebacks);
- harness notices that output files of two background jobs, and one oversized tool result, had been written outside the room. I did **not** open any of them.

**Looked up (from memory, not derived here):**
- the classification of closed subgroups of SO(3), and the abelianization sizes of C_n, D_n, T, O, I (used only as a check);
- the character formula for fixed-space dimensions and tr D^j(θ);
- Racah's closed CG formula (cross-check only);
- Schur's lemma; the Poincaré–Hopf theorem; the transformation rule D(g)†J D(g) = R(g)J;
- the barrier / facial-reduction technique for semidefinite programs (search only; soundness is checked exactly).

**Library lookup:**
- `sympy.physics.quantum.cg.CG`, called for all 231 coefficients: **agrees** with my own construction on every entry.

**Rule deviations, reported:**
- One early shell command began with a `cd` into this room before `./py`, contrary to the "commands begin with ./py" rule. Nothing outside the room was touched.
- One command (a heredoc) was refused by the permission system and was not retried in that form.
- Two long background computations were started. One finished; the other — an early, slow symbolic CG construction later replaced — may still have been running when I stopped. I could not terminate it: the shell accepts only `./py`, and the sandbox refused process listing.
- No network was used. No links were followed.
