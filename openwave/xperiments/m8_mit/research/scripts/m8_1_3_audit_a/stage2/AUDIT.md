# AUDIT — stage 2

All numbers below were printed by my scripts in `audit/` (stage 2) or `own/` (stage 1, unchanged). Stage 2 imports my stage-1 library (`own/lib_*.py`) read-only and never imports, calls or copies a solver's code; solver files were read as data (`results.json`, logs) and, for grading, as text.

**How to reproduce.** Run `./py audit/a01_anchors.py` … `./py audit/a05_attacks.py`, then `./py audit/a06_collect.py`, which writes `audit.json`. Each script writes `audit/<script>.log`. Running a script with `plant` writes `audit/<script>_plant.log`.

**Final state.**
- Every clean run ends `-- 0 FAIL line(s)`.
- Every planted run fires. The FAIL counts are: a01 11, a02 16, a03 4, a04 10, a05 10.

**Bottom line.** I recomputed every item and every point in my scope by my own route. Nothing is left unchecked. I found:
- **no disagreement** with any reported value of `solver_l6`, `solver_r2` or `solver_rig`;
- **one demonstrable defect**, in the text of `solver_r2/RETURN.md`. It is a false statement about a numerical margin; no value depends on it;
- several weaknesses in the solvers' checks (Section 5).

---

## 1. Verdict tables

Verdicts are AGREE, DISAGREE, DEFECT or UNCHECKED. "Mine" means my stage-1 value where one exists. Otherwise it is my stage-2 recomputation, made by my own route (exact, and a second numerical precision wherever the value is numerical).

### Anchors

| Item | l6 | r2 | rig | Mine | Rooms against each other | Verdict |
|---|---|---|---|---|---|---|
| 0(a) |q₁|, |q₂|, order, classes, characters | 1, 1, 120, 9 classes, table | same | same (PSLQ at 30 and 60 digits, then exact) | same | each room's 9 rows match my 9 rows bijectively, by Re g; sizes and element orders match | AGREE (all three) |
| 0(b) levels ≤ 20 with multiplicities; names | as mine | as mine | as mine | `own/OWN.md` 0(b) | R₂ = 2 (lowest level 7), R₄ = 3 (lowest 6), R₅ = 4 (lowest 6) in all three rooms | AGREE (all three) |
| 0(c) R₄ | mult 1; η; s_N = 3/7 | — | — | mult 1; s_N = 3/7 | — | AGREE |
| 0(c) R₅ | mult 1; η; s_N = 4/7 | — | — | mult 1; s_N = 4/7 | — | AGREE |
| 0(c) R₂ | — | mult 1; η = [e₁, Θe₁]; ρ(q₁), ρ(q₂); s_N = 1/4 | — | mult 1; s_N = 1/4 | — | AGREE |
| 0(d) | √231/462 > 0; Θ² = +1, −1 | same | — | same; my own J₋-lowering route gives √231/462 | l6 = r2 | AGREE |
| 0(e) R₂ | — | C = [[0,−1],[1,0]], CC̄ = −I, map w ↦ −Θw | — | same (stage 1, in my convention) | — | AGREE |

Notes on the anchors:
- **How η was checked (a01).** I tested each solver's η in that solver's own stated convention (quaternion ↦ [[w−iz, −y−ix], [y−ix, w+iz]], acting by substitution). The D matrices were built by my code. The checks, all exact:
  - η^Hη = I;
  - (I − ηη^H)D(qᵢ)η = 0 for both generators;
  - tr ρ(qᵢ) equals my character.
- **η depends on the convention.** l6's R₄ η is *not* an intertwiner in my quaternion convention (informative line in `audit/a01_anchors.log`). This is expected and is not a defect.
- **r2's ρ.** r2's reported ρ(q₁) and ρ(q₂) equal η^H D(qᵢ) η exactly.
- **r2's C and the map.**
  - The solution space of ρ̄X = Xρ is one complex line.
  - The map ψ ↦ ψ̄C equals ψ_{−Θw}, with residual 8.4e−51 at 50 digits over 12 random (w, g ∈ SU(2)) samples.

### Block A

| Item | Slot | Solver's answer | Mine | Verdict |
|---|---|---|---|---|
| A.1 | R₂ (r2) | q₇, q₅, q₃, q₁ = 144/143, 196/143, 28/11, 0 | same (stage 1, exact; quadrature to 1e−38) | AGREE |
| A.1 | R₄ (l6) | q₆, q₄, q₂, q₀ = 1288/1287, 35/33, 14/9, 7/3 | same | AGREE |
| A.1 | R₅ (l6) | q₆, q₄, q₂, q₀ = 2289/2288, 91/88, 21/16, 7/4 | same | AGREE |
| A.2 | R₂, R₄, R₅ | proportional; β = 24/13, 28/39, 21/52 (r2's HS coefficients 1/264, 53/264, 221/264, −13/24) | same | AGREE |
| A.3 | — (l6) | ratio 9/16; difference 49/156 | same | AGREE |
| A.4 | R₂, all 8 points (r2) | see the R₂ table below | same | AGREE (8/8) |
| A.5 | {5/2,−5/2}, {7/2,−7/2}, {3/2,−3/2} (r2) | Q ≡ 168/143, 144/143, 224/143; Q = c(A+B)² | my exact h(αv_a + βv_b) is exactly c(|α|²+|β|²)², same c | AGREE |
| A.6 | R₂ (r2) | q = 144/143; 6.0 %; 0.35 % (ceiling) | q = 144/143; 5.9131…% → 6.0; 0.34965…% → 0.35 | AGREE (reading-sensitive: nearest rounding gives 5.9) |
| A.6 | R₄ (l6) | 1288/1287; 2.0 %; 0.039 % | same | AGREE (not sensitive to the rounding reading) |
| A.6 | R₅ (l6) | 2289/2288; 1.5 %; 0.022 % | same | AGREE (not sensitive to the rounding reading) |

### B.1 (R₄) and B.2 (R₅), l6, all ten census points

**Route (a02).** This is my own exact route:
- exact h, gradient, Hessian, characteristic polynomials and factorisation;
- the signature by Descartes' rule on the N_u polynomial, **and** by 60-digit eigenvalues on an explicit orthonormal basis of N_u;
- the zero multiplicity from the characteristic polynomial, **and** as an algebraic multiplicity from the rank sequence of Ŝᵖ;
- Krein signs at 60 **and** 100 digits.

l6's values agree field by field: critical, Q, 924 r̂₆, signature, k, r_u, zero multiplicity, slow polynomial (exact identity), words, and every Krein sign. In each row, the first number in the zero-multiplicity column is the algebraic multiplicity and the second is the kernel dimension. The kernel dimension is always smaller, so the reported values are algebraic multiplicities, as the worklist requires.

| Point | Q R₄ / R₅ | 924 r̂₆ | Signature | k, r_u | Zero mult. (alg.; ker) | Words | Krein (R₄; R₅ the same, paired) | Verdict |
|---|---|---|---|---|---|---|---|---|
| C3 ray | 77/65 / 287/260 | 1188/5 | (5,0,4) | 4, 2 | 6; 4 | hyperbolic, a complex quartet | + (ν ≈ −1.00362), − (ν ≈ −0.20596) | AGREE |
| D2 ray | 168/143 / 2513/2288 | 225 | (5,0,4) | 4, 4 | 8; 4 | hyperbolic, real pairs | + | AGREE |
| coherent v3 | 1288/1287 / 2289/2288 | 1 | (0,0,10) | 3, 1 | 4; 3 | elliptic, simple | all + | AGREE |
| hexagon | 1750/1287 / 2751/2288 | 463 | (9,0,0) | 4, 4 | 8; 4 | elliptic, one double frequency | −, − (double; cluster dim 4) | AGREE |
| octahedron | 175/143 / 161/143 | 288 | (6,0,3) | 4, 4 | 8; 4 | hyperbolic, real pairs | none | AGREE |
| pentagonal pyramid | 77/65 / 287/260 | 1188/5 | (5,0,4) | 4, 2 | 6; 4 | hyperbolic, a complex quartet | +, − | AGREE |
| trigonal prism | 5831/5031 / 609/559 | 8800/43 | (3,0,6) | 4, 4 | 8; 4 | elliptic, one double frequency | +, + (double; dim 4) | AGREE |
| v1 | 168/143 / 2513/2288 | 225 | (4,2,4) | 3, 1 | 6; 5 | elliptic, simple | ordered by ν (most negative first): +, −, +, − | AGREE |
| v2 | 147/143 / 581/572 | 36 | (2,0,8) | 3, 1 | 4; 3 | elliptic, simple | four +, and − at −3136/1656369 | AGREE |
| zonal v0 | 1687/1287 / 168/143 | 400 | (8,0,2) | 3, 3 | 6; 3 | hyperbolic, real pairs | − (double; dim 4) | AGREE |

The slow polynomials are exactly l6's (`audit/a02_census.log`).

**B.2 ratio.** At all ten points, P_{R₅}(ν) is exactly proportional to P_{R₄}((256/81)ν). So every clause-3 pair has +ξ_{R₅}/+ξ_{R₄} = 9/16, numerically to 1e−40. That is l6's answer: AGREE (10/10).

### B.3 and A.4 (R₂), r2, all eight points

**Route (a03).** I used my own exact route:
- characteristic polynomials over Q, or over Q(√51) at line {7/2, −5/2}, via DomainMatrix;
- for the √2 members, (i+τ′)M̂ with ν = y²/2;
- roots by exact factorisation over the field and polyroots at 80 and 120 digits;
- a match against the squares of 50-digit eigenvalues of τM̂;
- Krein signs at 60 and 100 digits.

r2 agrees on every field of every structure, at every point.

| Point | Q | Signature | k, r_u | Ŝ_u: zero mult. (ker); words; Krein | Members J and k | Members (i+J)/√2 and (i+k)/√2 | Verdict |
|---|---|---|---|---|---|---|---|
| weight 7/2 | 144/143 | (0,0,10) | 5, 1 | 6 (5); elliptic, simple; all + | identical; 8; (20449ν+56448)²(20449ν+62720)²; elliptic with two double roots (not a named word); +, + | identical; 6; elliptic, simple; all + | AGREE |
| weight 5/2 | 168/143 | (4,0,6) | 5, 1 | 6 (5); elliptic, one double frequency; −, −, +, + (double) | identical; 8; hyperbolic, real pairs; + | identical; 6; hyperbolic, a complex quartet; + (double), − | AGREE |
| weight 3/2 | 224/143 | (8,0,2) | 5, 1 | 6 (5); elliptic, simple; −, +, −, −, − | identical; 8; hyperbolic, real pairs; − | identical; 6; hyperbolic, a complex quartet; −, −, − | AGREE |
| weight 1/2 | 168/143 | (4,0,6) | 5, 1 | 6 (5); hyperbolic, not named (1 negative, 2 conjugate pairs); + | identical; 8; hyperbolic, real pairs; + | identical; 6; hyperbolic, not named; + | AGREE |
| line {7/2, −1/2} | 193/143 | (5,0,4) | 6, 2 | 8 (6); hyperbolic, not named (2 conjugate pairs); none | identical to Ŝ_u | identical; 8; hyperbolic, a complex quartet; +, − | AGREE |
| line {7/2, −3/2} | 22/13 | (9,0,0) | 6, 4 | 10 (6); elliptic, simple; −, −, − | identical; 10; elliptic, simple; −, −, − | **different cubics**; 10; elliptic, simple; −, −, − | AGREE |
| line {7/2, −5/2} | 369/247 | (6,0,3) | 6, 2 | 8 (6); hyperbolic, not named; − | **Galois-conjugate quartics over Q(√51)**; 8; hyperbolic, one real pair; −, +, − each | √51 term of opposite sign; 8; hyperbolic, not named; − | AGREE |
| line {5/2, −3/2} | 161/143 | (0,3,6) | 6, 2 | 12 (9); elliptic, simple; +, + | identical; 12; elliptic, simple; +, + | identical cubic; 10; elliptic, simple; +, +, + | AGREE |

In the table, "identical" means the two members of that pair have the same slow polynomial.

**r2's member findings, checked independently.**
- My exact route reproduces the non-rational member polynomials at line {7/2, −5/2}, coefficient by coefficient.
- It also reproduces J ≠ k at {7/2, −5/2} and (i+J) ≠ (i+k) at {7/2, −3/2}.

**Pin(2) reading.** Under the alternative reading of the fibre group (no continuous Sp(1); see Section 4), k drops by 2 at every point: 3 at the weights and 4 on the lines. r_u is unchanged, and n₀ rises by 2. Under that reading the signatures are:

| Point | Signature (Pin(2) reading) |
|---|---|
| weight 7/2 | (0,2,10) |
| weight 5/2 | (4,2,6) |
| weight 3/2 | (8,2,2) |
| weight 1/2 | (4,2,6) |
| line {7/2, −1/2} | (5,2,4) |
| line {7/2, −3/2} | (9,2,0) |
| line {7/2, −5/2} | (6,2,3) |
| line {5/2, −3/2} | (0,5,6) |

### B.5 and Block E

| Item | Solver's answer | Mine | Verdict |
|---|---|---|---|
| B.5 (l6) | proportional, constant 1 | identical functions, constant 1 (stage 1, exact) | AGREE |
| E.1 (rig) | six rigid slots, of dims 1–6: n = 0…5, next level 12, 11, 10, 9, 8, 7, gap 168, 140, 112, 84, 56, 28 | same (stage 1) | AGREE |
| E.2 (rig) | μ^{2d}(μ²+4ω²)^{d−1}(μ²+4ω²+2g₄c²|v|²); 0: algebraic 2d, geometric 2d−1; ±2iω: d−1, d−1; ±i√(…): 1, 1 | stage 1 for d = 1, 2; stage 2 (a04) for d = 1…6 | AGREE (d = 1…6) |

The stage-2 E.2 route builds the real 4d system myself, at a random Gaussian-rational v that is not along a basis vector, and at two parameter sets per d. It computes the exact characteristic polynomial and exact ranks over Q(i).

---

## 2. Defects I can demonstrate

**D1. `solver_r2/RETURN.md`, B.3 "Krein signs": the margin statement is false. No reported value is affected.**

The return says: "The smallest |eigenvalue| of M̂ there is ≥ 0.04 in every case."

r2's own `solver_r2/out_B3_R2.json` (`min_abs_eig` fields) records smaller margins at three places:

| Point | Member(s) | ν | Margin |
|---|---|---|---|
| line {5/2, −3/2} | (i+J)/√2 and (i+k)/√2 | −0.016326… | 0.012563 |
| weight 3/2 | the √2 members | −0.050633… | 0.017596 |
| line {7/2, −5/2} | member k | −0.19196… | 0.037419 |

My independent 60/100-digit computation (`audit/a03_r2.log`) gives the same three margins: 0.01256, 0.0176 and 0.03742. The 0.04 bound holds only for the clusters of Ŝ_u = iM̂ (minimum 0.0435). Those are the only margins printed in `solver_r2/sB3.log`.

The signs themselves are right: every margin is many orders above the kernel singular values (≤ 1e−100 at 100 digits).

---

## 3. Claims I tried to break and could not

- **Criticality is exact, not a tolerance.** This was tested at all 20 census cases and all 8 R₂ points. My exact gradient residual at u is 0. At u + 1e−30·δ, for a fixed generic δ, the residual is nonzero and of order 1e−30 (2.6e−31 to 2.4e−30). A tolerance-level criticality would not show this scaling.
- **Signatures, by an independent decomposition.** Exact Descartes counts on the N_u factor of the characteristic polynomial equal the 60-digit eigen-decomposition of M̂ on an explicit orthonormal basis of N_u in all 28 cases. The smallest nonzero |eigenvalue| was 0.0098 (pentagonal pyramid, R₅). The degenerate signatures were confirmed this way:
  - v1, (4,2,4) in R₄ and R₅: the numeric zero eigenvalues are ≤ 6e−64;
  - line {5/2, −3/2}, (0,3,6).
- **Zero multiplicities are algebraic, not kernel dimensions.** The rank sequences of Ŝᵖ (exact over Q or Q(√51), or 80-digit SVD where entries are surds, with smallest kept singular value ≥ 5.2e−6 against discarded ones ≤ 1e−80) give the same algebraic multiplicity as the characteristic polynomial everywhere. The kernel dimension is strictly smaller everywhere. Examples: octahedron 8 against kernel 4; line {5/2, −3/2} 12 against 9. All Jordan blocks at 0 have size ≤ 2.
- **Krein signs.** At every cluster the sign agrees at 60 and 100 digits, and dim ker(Ŝ² + τ²) = 2 × multiplicity (semisimple). This includes the 4-dimensional clusters (hexagon, trigonal prism, zonal v0, and the R₂ double roots). The thinnest margins are:

  | Point | ν | Margin |
  |---|---|---|
  | pentagonal pyramid, R₅ | ≈ −0.0012255 | 0.0025 |
  | pentagonal pyramid, R₄ | — | 0.0044 |
  | v1, R₅ | −1225/327184 | 0.0056 |
  | line {5/2, −3/2}, √2 members | — | 0.0126 |

  Each is decided with kernel singular values ≤ 1e−100.
- **Exactness of the reported exact values.** Q, the signatures, k, r_u, the zero multiplicities and the slow polynomials (including the Q(√51) ones) come from exact routes. Every slow root also matches the 50-digit eigenvalues of τM̂ (a03).
- **924 r̂₆ at all ten points.** There are two routes:
  - the exact Fischer route;
  - an independent 50-digit route through my Casimir spin-6 projector on V₃ ⊗ V₃.

  They agree to 1e−40 (a05).
- **h is Sp(1)-invariant in R₂.** h(cos t·u + sin t·Θu) = h(u) holds to < 1e−40 at random u. M̂ vanishes exactly on Θu and Θ(iu) at all 8 points. So the Sp(1) reading of the fibre group is consistent with the quartic. A planted Θ without its signs breaks it: relative error 1.7.
- **R₂ member anomalies.** At line {7/2, −5/2}, the irrational member polynomials, J ≠ k, and (i+J) ≠ (i+k) at {7/2, −3/2} are genuine; my exact route reproduces them. They are consistent with this: the phase rotation that carries J to k maps a two-weight point to itself, up to a rotation about z, only for {5/2,−3/2}, {7/2,−1/2} and the weight vectors.
- **C3 ray against the pentagonal pyramid.** l6's observation that these share Q, 924 r̂₆, signature, k, r_u and words but have different slow polynomials is confirmed. Both are exactly critical.

---

## 4. Readings (step 3)

| # | Where | Reading taken | Another reading available? | Would the values change? |
|---|---|---|---|---|
| 1 | R₂ fibre group (0(e), §1.4), decides k and r_u | r2 and I: Γ contains the continuous Sp(1) = {a + bi + cJ + dk}, i.e. the closed group generated by the one-parameter groups e^{αi} and e^{βJ}. K_u gains Θu and Θ(iu) | Yes. The group literally *generated* by the phase and the single map ψ ↦ ψ̄C is Pin(2) = {e^{iα}} ∪ {e^{iα}J}, which is 1-dimensional and contributes only iu | **k:** 5→3 (weights), 6→4 (lines). **r_u:** unchanged. **n₀:** +2 (table in Section 1). **Unchanged:** zero multiplicity, slow polynomials, words and Krein signs, since Ŝ lives on all of E₀; also members and words. The worklist says "generates a group Sp(1)", which supports reading 1; h is Sp(1)-invariant (Section 3) |
| 2 | Charge conjugation in R₄ and R₅ | discrete; adds nothing to K_u (l6, me) | No real alternative: it is a discrete map | — |
| 3 | Words for unnamed patterns | r2: "elliptic/hyperbolic (no listed sub-word)"; me: "(not named)" | Forcing a named word, e.g. "one double frequency" for two doubles | Only the wording; the root data are the same |
| 4 | Krein signs only for negative real ν | all rooms and me | Yes: read "±iτ" with complex τ and apply the definition to the real invariant subspace of a hyperbolic cluster | Every such cluster would read **indefinite**: computed at D2 ray, octahedron, zonal v0, C3 ray and the pentagonal pyramid (R₄), with 2+/2− (3+/3− at the octahedron) (a05). No reported sign changes |
| 5 | A.6, "rounded up to two significant digits" | l6 and r2: ceiling | Nearest rounding | Only R₂'s √((q−1)/2): 6.0 % → 5.9 %. The other five numbers are unchanged |
| 6 | η (0(c)) | each room: Gram–Schmidt of the projector's columns, in its own convention | η ↦ ηU, and the quaternion convention | η and ρ change; s_N, q_J and everything downstream do not |
| 7 | 0(e) | C = [[0,−1],[1,0]] ⇒ map −Θ | C is fixed only up to a phase: −C gives +Θ | Only the sign of the map; the Sp(1) is the same |
| 8 | Irrational member polynomials (R₂, line {7/2, −5/2}) | r2: monic over Q(√51), factored over Q(√51) | The rational degree-8 norm polynomial (product with the Galois conjugate) | That would be a different object, mixing J and k; the words and Krein signs would no longer belong to one member |
| 9 | B.2 "corresponding" | clause 3 | Fixed by clause 3 | 9/16 in every pair; the positive scale 81/256 preserves the sort |
| 10 | E.2 | rig: eigenvalues of the real 4d × 4d generator in the rotating frame, after complexification; |v| with |ψ_v| = |v| | A lab-frame linearisation, or c⁻¹∂_T as the variable | A lab frame shifts the spectrum by ±iω (excluded by the item's own ψ = e^{iωT}(ψ_v + ζ)); a c⁻¹∂_T variable leaves the eigenvalues unchanged |

---

## 5. Their checks (step 4)

### solver_l6

- **A PASS line that cannot fail at any point.** "u is G-orthogonal to K_u" (`s04_blockB.py`, line 123). It holds identically for antisymmetric generators, so it tests the generator matrices, not the point.
- **A planted defect that does not exercise the computation.** The `s05` plant "signature [4, 1, 5]" compares a computed value with a typed constant. It shows that `==` works; it does not exercise the signature route.
  - No plant exercises the Sturm/count route of the signature or the zero-multiplicity route.
  - Those values do rest on the independent CS-coordinate cross-check in `s05`, and that cross-check is genuine.
- **Plants that do exercise the reported values:**
  - the Krein classifier (union of a + and a − cluster);
  - the B.2 ratio test (c = 3/4);
  - criticality (√13 → √14);
  - the K_u vanishing test.
- **A value identified from numerics.** The B.2 common ratio is found from 40-digit numerics by `nsimplify`, and is then verified exactly by proportionality. That is acceptable.
- **Krein signs** are taken at 60 and 90 digits.
- **Fragile but correct here.** The numeric zero count in `s05` (|eig| < 1e−20 at 50 digits) works only because every Jordan block at 0 has size ≤ 2; my rank sequences show this. A block of size 3 would make it fail spuriously.

### solver_r2

- **Krein signs at one precision.** The Krein signs and the semisimplicity checks are numeric at one precision only (50 digits). Their signs are right: my 60- and 100-digit routes agree.
- **D1 above** (the margin statement).
- **A PASS line that cannot fail at any point.** "K_u orthogonal to u" (`sB3.py`, line 275). i, Θ and k, and the rotation generators, are antisymmetric; for half-integer j the matrix E of Θ is antisymmetric.
- **Plants that do exercise the reported values:**
  - Sturm off by one (signature and root counts);
  - a perturbed 50-digit M̂ (exact against numeric);
  - a corrupted tangent vector (k, and M̂K_u = 0);
  - Θ without its signs (τ² = −1);
  - ν shifted (kernel dimension).
- **No plant exercises the Krein sign classifier itself.** All of r2's Krein signs rest on it. My a03 plant (union of a − and a + cluster at weight 5/2) shows this kind of classifier can fail.
- **Self-reported.** r2 itself found and replaced a check that could not fail ("signature sums to 15 − k").

### solver_rig

- **Character values.** These are identified by PSLQ at two precisions (30 and 60 digits) and then verified exactly. This is a sound route.
- **A fragile route.** E.2's geometric multiplicities use sympy `Matrix.rank` on matrices with complex entries. In my own run this routine gave an impossible nullity (0 for a simple eigenvalue at d = 4, ω = 2). I recorded and fixed it in a04 by using exact DomainMatrix ranks over Q(i). rig's values are right, and its numeric finite-difference nullities are an independent cross-check. **Suspicion only:** the route is fragile, but I cannot show it failed in their room.
- **Plants.** cubic, frozen and omega exercise E.2; restrict and next exercise E.1; char, mult, names and others exercise 0(a)–(b). These cover the reported values.

### My own checks

- Every audit script has planted runs that fire (FAIL counts are in the header).
- **Not planted in stage 2:** the "Krein agrees at 60/100" stability line in a02, and the k and r_u comparisons. The Krein classifier is planted in a03, and k is covered by the stage-1 plant (3h for 4h).
- **Fixes made in stage 2:**
  - a02 had a syntax slip;
  - a03's root matching used a plain sort that interleaves conjugate pairs; it is replaced by nearest matching;
  - a04's sympy-rank problem.

  Each was fixed before the final runs. The logs show only corrected runs.

---

## 6. What I could not check

- **Nothing in scope is ungraded.** Every item and every point in scope was recomputed: 0(a)–(e); A.1–A.6; B.1, B.2 and B.3 with all members; B.5; E.1–E.2.
- **Not re-run.** I did not re-run the solvers' own internal numerical routes (their quadratures, CG orthogonality, ladder checks): the brief says running their code is not verification. I reproduced the values those routes support by my own routes instead.
- **Not proved symbolically.** I did not prove E.2 for symbolic (ω, g₄, c, |v|). I checked it exactly at two parameter sets per d, at a generic v; rig's symbolic computation, at v along e₁, covers the general case.

---

## Rule breaches and process notes

- **Tools rule.** My first two shell commands in this stage (`cd <room> && ls -la && cat BRIEF.md worklist.md clauses.md inputs.json`, then `cat BRIEF.md clauses.md; cat .room.sb py`) were not `./py` commands. Both read only files in this room, and the `cd` was to the room's own top level.
- **Harness output.** The harness saved the first command's long output to a tool-results file outside the room. I did not open it.
- **An empty command.** One command began with an empty heredoc (`./py - <<'EOF'`). It ran nothing.
- I never edited anything under `own/`.

## Consulted-material manifest

**Files read in this room:**

- **Top level:** `BRIEF.md`, `worklist.md`, `clauses.md`, `inputs.json`, `.room.sb`, `py`.
- **My stage-1 files.** Read as text: `own/OWN.md`, `own/anchors.json`, `own/own.json`, `own/lib_common.py`, `own/lib_crit.py`, `own/lib_quartic.py`, `own/lib_sym.py`, `own/lib_rep.py`, `own/s05_A4_B.py`, `own/run_all.log`. Loaded by my scripts at run time: `own/group.json`.
- **solver_l6:** `RETURN.md`, `results.json`, `scripts/common.py`, `scripts/pointdata.py`, `scripts/s04_blockB.py` (lines 80–289), `scripts/s05_crosscheck.py` (lines 56–140) and `scripts/s05_crosscheck.log` (its beginning). Searched for check lines: `scripts/s01_group.py`, `scripts/s02_intertwiner.py`, `scripts/s03_blockA.py`, `scripts/s04_blockB.py`, `scripts/s05_crosscheck.py`.
- **solver_r2:** `RETURN.md`, `results.json`, `out_B3_R2.json`, `sB3.py` (lines 181–351), `sB3.log` (one section, plus a search), `sA1.py` (lines 139–169). Searched: `lib.py`, `s0_group.py`, `s0_slot.py`, `sA1.py`, `sA5.py`, `sA4_quad.py`, `sB3.py`, `plant_defects.py`, `tools_patch_b3.py`, `tools_patch_b3b.py`.
- **solver_rig:** `RETURN.md`, `results.json`, `s03_rigid_lin.py` (lines 30–159). Searched: `checks.py`, `s01_group.py`, `s02_rigid_slots.py`, `s03_rigid_lin.py`, `s04_rigid_group.py`.
- **Directory listing:** the names and sizes of every file under `own/`, `solver_l6/`, `solver_r2/`, `solver_rig/`.
- **My stage-2 files:** `audit/*.py`, `audit/*.log`, `audit/census.json`, `audit/r2.json`, `audit/rest.json`.

**Loaded without my asking:**
- the session's system context, including environment details and an account email, which I did not use;
- the Python standard library, numpy, sympy and mpmath, from the interpreter's environment outside the room, as `./py` provides.

**Recalled rather than derived:**
- Descartes' rule, Sylvester inertia and Jordan-structure rank sequences, used as standard methods;
- the recognition of Γ₀ as the binary icosahedral group (stage 1; not used as input).

**Not opened:** the harness's tool-results file outside the room; anything else outside the room. Nothing was looked up, and there was no network.
