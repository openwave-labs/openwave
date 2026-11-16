# RETURN — items 0(a), 0(b), 0(c)[R₂], 0(d), 0(e); A.1, A.2, A.6 [R₂]; A.4; A.5; B.3 (with clause 4)

Every number below was printed by a script in this room. `./py run_all.py` runs them in order:
`s0_group.py`, `s0_slot.py`, `sA1.py`, `sA5.py`, `sA4_quad.py`, `sB3.py`, `make_results.py`, `plant_defects.py`.
Each script's output is kept in `<script>.log` beside it. Shared code is in `lib.py`, `quartic.py` and `quad.py`.
Two kinds of helper file report no numbers:
- `scratch_explore.py` was exploration only.
- `tools_*.py` and `tools_b3_tail.txt` are one-off edit and log-view helpers.
`results.json` holds every reported value. The exact strings there are what the scripts printed.

**Conventions used where Section 2 leaves a choice**
- Quaternion to matrix: q = w + xi + yj + zk ↦ U(q) = w·I − i(xσ_x + yσ_y + zσ_z) = [[w − iz, −y − ix], [y − ix, w + iz]]. This is a homomorphism.
- Dʲ(U) is the action f(x, y) ↦ f((x, y)U) on homogeneous polynomials of degree 2j, with basis e_m = x^{j+m} y^{j−m}/√((j+m)!(j−m)!).
  - This basis is orthonormal, and its ladder operators are exactly Condon–Shortley: J₊ = x∂_y, J₋ = y∂_x.
  - On the weighted coordinates p_m the matrix of Dʲ is polynomial in the entries of U.
  - Checks: tr Dʲ = U_{2j}(Re q) on every class for j ≤ 5/2; D unitary on both generators; Θ commutes with D. The CG vectors obey the CS ladder relation exactly (J = 6, 4, 2 on V₃⊗V₃).
- Free-side translation acts on w by D(g)ᵀ, i.e. the conjugate representation. Its real Lie-algebra span equals that of D, so K_u does not depend on this choice. Nothing I report depends on it.
- Real coordinates are (Re w_m, Im w_m) on the orthonormal CS components. Gradients and Hessians are Euclidean in these coordinates.
- Clebsch–Gordan coefficients: I built them myself from Racah's closed formula, in exact sympy arithmetic.
  - The CS sign ⟨j j; j −j|2j 0⟩ > 0 holds, and so does the ladder relation above.
  - The 64×64 CG matrix of V₇/₂⊗V₇/₂ is orthogonal to 1e−45.
  - **No library CG routine was called.**
- Clause 4 notation: s_N is the constant of § 1.3. ξ is the variable of the characteristic polynomial, and slow polynomials are written in ν = ξ². τ is a cluster's frequency.

---

## Anchors (item 0)

### 0(a) — DERIVED in this room (`s0_group.py`)
- |q₁|² = 1 and |q₂|² = 1, exactly in Q(√5): φ²/4 + φ⁻²/4 + 1/4 = 1.
- **|Γ₀| = 120**, from closing the two generators under multiplication in exact Q(√5) arithmetic.
- **9 conjugacy classes**, computed by conjugating with all 120 elements. Each class has a single real part w = Re q.

| class (element order) | 1 | 2 | 3 | 4 | 5 | 5 | 6 | 10 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| size | 1 | 1 | 20 | 30 | 12 | 12 | 20 | 12 | 12 |
| Re q | 1 | −1 | −1/2 | 0 | (√5−1)/4 | −(√5+1)/4 | 1/2 | (1+√5)/4 | (1−√5)/4 |

- **Irreducible characters.** I derived these by peeling the characters U_{2j}(Re q) of V_j, for j ≤ 20, against the characters already found. A remainder with norm 1 and positive degree is a new irreducible.
  - This gives nine characters with Σ dim² = 120.
  - Row and column orthogonality both hold exactly.
- Columns below are in the class order of the table above (φ = (1+√5)/2, φ' = (1−√5)/2):

| dim (lowest level) | values |
|---|---|
| 1 (0) | 1, 1, 1, 1, 1, 1, 1, 1, 1 |
| 2 (1) | 2, −2, −1, 0, −φ', −φ, 1, φ, φ' |
| **2 (7) = R₂** | 2, −2, −1, 0, −φ, −φ', 1, φ', φ |
| 3 (2) | 3, 3, 0, −1, φ', φ, 0, φ, φ' |
| **3 (6) = R₄** | 3, 3, 0, −1, φ, φ', 0, φ', φ |
| 4 (3) | 4, −4, 1, 0, −1, −1, −1, 1, 1 |
| **4 (6) = R₅** | 4, 4, 1, 0, −1, −1, 1, −1, −1 |
| 5 (4) | 5, 5, −1, 1, 0, 0, −1, 0, 0 |
| 6 (5) | 6, −6, 0, 0, 1, 1, 0, −1, −1 |

Nothing here surprised me. Γ₀ is the binary icosahedral group: I recognised that after deriving it, and nothing above relies on recognising it.

### 0(b) — DERIVED (`s0_group.py`)
Levels ℓ ≤ 20 at which each irreducible occurs in V_{ℓ/2}, written as level(multiplicity):
- dim 1: 0, 12, 20 (all multiplicity 1)
- dim 2 (natural): 1, 11, 13, 19
- **dim 2, R₂:** 7, 13, 17, 19
- dim 3: 2, 10, 12, 14, 18, 20
- **dim 3, R₄:** 6, 10, 14, 16, 18, 20
- dim 4 (spinorial): 3, 9, 11, 13, 15, 17, 19
- **dim 4, R₅:** 6, 8, 12, 14, 16, 18(2), 20
- dim 5: 4, 8, 10, 12, 14, 16(2), 18, 20(2)
- dim 6: 5, 7, 9, 11, 13, 15(2), 17(2), 19(2)

All multiplicities not marked are 1. Each name in § 1.1 (dimension, lowest level) = (2, 7), (3, 6), (4, 6) picks out exactly one irreducible.

### 0(c) for R₂ — DERIVED (`s0_slot.py`)
- **Multiplicity at level 7: 1.** The isotypic projector P = (2/120) Σ χ_R₂(h) Dʲ(h) has rank 2, and P² = P = Pᴴ exactly.
- **Intertwiner η = [e₁, Θe₁]**, with e₁ the normalised column of P at m = −7/2:
  - e₁ = (√15 v₋₇/₂ − √7 v₋₃/₂ + √21 v₁/₂ + √21 v₅/₂)/8
  - e₂ = Θe₁ = (√21 v₋₅/₂ + √21 v₋₁/₂ − √7 v₃/₂ + √15 v₇/₂)/8
  - ηᴴη = I₂, and D(h)η = ηρ(h) holds exactly for both generators.
  - ρ(q₁) = [[(1−i)/2, (1+i)/2], [(−1+i)/2, (1+i)/2]]
  - ρ(q₂) = [[(1−√5)/4, 1/2 − i(1+√5)/4], [−1/2 − i(1+√5)/4, (1−√5)/4]]
  - The traces of ρ on the generators equal χ_R₂.
- **s_N(R₂) = 1/4** = dim ρ/(2j+1).
  - Exact route: Schur orthogonality on SU(2).
  - High-precision route: 50-digit quadrature of ∫|φ_w|². It agrees to 3e−50 at two w.

### 0(d) (`s0_slot.py`)
- ⟨3 3; 3 −3 | 6 0⟩ = √231/462 > 0, so the **sign is +**.
- **Θ² = +1 on V₃** and **Θ² = −1 on V₇/₂**, exactly, as 14×14 and 16×16 real matrices.

### 0(e) (`s0_slot.py`)
- I solved ρ̄(h)C = Cρ(h) for both generators. The solution space is one-dimensional: C = c·[[0, −1], [1, 0]].
  - I report **C = [[0, −1], [1, 0]]** (unitary), with **C C̄ = −I**.
- **The map ψ ↦ ψ̄C, read on w, is w ↦ −Θw.**
  - For a general unitary C' = e^{ia}C it is w ↦ −e^{ia}Θw.
  - Proof steps: D̄(g) = Y D(g) Y⁻¹ with Y_{m,−m} = (−1)^{j−m}, checked exactly on the generators; and Y⁻¹η̄C = η (ηᴴY⁻¹η̄C = I₂).
  - Direct check: ψ̄_w C = ψ_{−Θw} holds at sampled group points to 1e−30.
- So the Sp(1) generated by the phase and ψ ↦ ψ̄C acts on E₀ as the unit quaternions a + b·i + c·J + d·k, with J = Θ and k = J∘i.
  - This is the fibre group I used for K_u in R₂.
  - It means the four members of § 1.4 use exactly the structures generated by the fibre group.

---

## Block A (R₂)

### A.1 — channel coefficients of R₂ (`sA1.py`)
**q₇ = 144/143, q₅ = 196/143, q₃ = 28/11, q₁ = 0.**
- Exact route: Q = ⟨|φ_w|⁴⟩_Haar/(s_N²‖w‖⁴). Schur on D⊗D (multiplicity-free) then gives q_J = tr(Π_J(P⊗P))/((2J+1)s_N²).
- High-precision route: the defining integral, computed by a product quadrature on SU(2) at 50 digits.
  - The rule is exact for these integrands: 32×32 trapezoid nodes in the two phases and 16 Gauss–Legendre nodes in t = |a|².
  - Q was computed at 7 test points (6 random complex points and v₇/₂). It agrees with Σ q_J f_J to ≤ 2.7e−49.
  - q_J solved from four of those points agree with the exact values to 6e−48.
- **Something I got wrong first.** My first quadrature used 16 phase nodes, which is too few. Dʲ(U) is a polynomial in a, c *and their conjugates*, so |φ|⁴ has phases up to 28.
  - With 16 nodes the check fired: differences of about 0.04 at generic points.
  - With 32 nodes the routes agree. The logs show only the corrected run.

### A.2 (`sA1.py`)
**Yes. Q − 1 = β·‖(ww†)₆‖²/‖w‖⁴ with β_R₂ = 24/13.**
- In channel fractions, the HS spin-6 factor is (1/264)f₇ + (53/264)f₅ + (221/264)f₃ − (13/24)f₁.
  - These are exact traces c_J = tr(Π_J Σ_M T_M†⊗T_M)/(2J+1); every term is ±√rational, summed exactly.
  - The identity holds at 7 points to 1e−51.
- (q_J − 1)/c_J = 24/13 in all four channels.
- Q − 1 = (24/13)·HS₆ holds at the 7 points to 1e−45, with HS evaluated directly.

### A.6 (`sA1.py`)
- **q = Q(v₇/₂) = 144/143.** This is exact, and the quadrature agrees to 1e−49.
- (q − 1)/2 = 1/286, so √((q − 1)/2) = √286/286.
- As percentages:
  - √((q−1)/2) = 5.9131…%, **6.0%** rounded up.
  - (q−1)/2 = 0.34965…%, **0.35%** rounded up.
- *Reading:* I took "rounded up to two significant digits" to mean the ceiling at two significant digits. Rounding to nearest would give 5.9% and 0.35%.

### A.4 — R₂'s eight points (`sB3.py`, Q also by quadrature in `sA4_quad.py`)
- All eight are **critical**: ∇h(u) = 4h(u)u holds exactly.
- Q is exact; the 50-digit quadrature of the defining integral agrees to ≤ 7e−49 at every point.
- Signatures are [n₋, n₀, n₊] of M̂_u on N_u.
  - Exact route: Sturm counts on the exact characteristic polynomial of M̂ (rational after the similarity N).
  - High-precision route: 50-digit eigenvalues. The two agree at every point.

| point | critical | Q | signature |
|---|---|---|---|
| weight 7/2 | yes | 144/143 | [0, 0, 10] |
| weight 5/2 | yes | 168/143 | [4, 0, 6] |
| weight 3/2 | yes | 224/143 | [8, 0, 2] |
| weight 1/2 | yes | 168/143 | [4, 0, 6] |
| line {7/2, −1/2} | yes | 193/143 | [5, 0, 4] |
| line {7/2, −3/2} | yes | 22/13 | [9, 0, 0] |
| line {7/2, −5/2} | yes | 369/247 | [6, 0, 3] |
| line {5/2, −3/2} | yes | 161/143 | [0, 3, 6] |

Line {5/2, −3/2} is degenerate: n₀ = 3.

### A.5 (`sA5.py`)
Write A = |α|² and B = |β|². Then [w⊗w]_J has components only at M = 2a, 0, −2a, and Q(αv_a + βv_b) works out to:
- {5/2, −5/2}: Q = (168/143)(A + B)², so **Q ≡ 168/143** on unit vectors.
- {7/2, −7/2}: Q = (144/143)(A + B)², so **Q ≡ 144/143**.
- {3/2, −3/2}: Q = (224/143)(A + B)², so **Q ≡ 224/143**.

Q is constant on each span. It does not depend on |α|/|β|, nor on the phases of α and β. I checked this two ways at two points per span, one of them with a complex phase:
- against the generic exact f_J code;
- against the 50-digit quadrature of the defining integral, agreeing to 1e−40.

---

## Block B: B.3 in R₂ (`sB3.py`)

### Method
- **Stationary group.** Γ is the SU(2) translations plus the Sp(1) = {i, J = Θ, k = J∘i}, which contains the phase (0(e)). So K_u = span{X_a u (3 rotations), iu, Θu, ku}.
  - k is the exact rank of these vectors; the numeric SVD rank agrees.
  - M̂K_u = 0 and K_u ⊥ u are checked exactly at every point.
- **r_u** = k − rank of the Gram matrix σ(x, y) = xᵀ(i·y) on a basis of K_u.
- **Characteristic polynomials.** I computed them exactly over Q, or over Q(√51) where needed, after the similarity N (w = Np), which commutes with i, Θ and k.
- **Roots.**
  - Factorisation and multiplicities are exact.
  - Signs and real/non-real status come from exact Sturm sequences, checked against 80-digit polyroots.
  - Every slow root matches, with multiplicity, the squares of the 50-digit numerical eigenvalues of Ŝ_u or of τM̂.
- **Krein signs.** For each negative root ν = −τ²:
  - I take ker(S² + τ²) from a 50-digit SVD. I check its dimension is 2 × the multiplicity, so every cluster is semisimple.
  - I then check M̂ for definiteness on that kernel. The smallest |eigenvalue| of M̂ there is ≥ 0.04 in every case.
- **Members.** S^τ = τM̂ for τ = J, k, (i+J)/√2, (i+k)/√2. Each satisfies τ² = −1, checked exactly. For the √2 members I used (i+J)M̂ and rescaled ν by 1/2.

**Readings for the slow-flow words**
- A root pattern that is all negative real but fits neither "simple" nor "one double frequency" is reported as "elliptic" with no sub-word. This happens at weight 7/2 for members J and k: two double roots.
- A hyperbolic pattern that fits none of the three listed sub-words is reported as "hyperbolic" with no sub-word.
- Krein signs are defined only for clusters ±iτ, which are the negative real ν. Non-real and positive ν have no Krein sign.

### Ŝ_u = iM̂_u at the eight points

| point | k | r_u | zero mult. | slow polynomial in ν (primitive, factored) | roots (sign, mult.) | words | Krein signs (by τ) |
|---|---|---|---|---|---|---|---|
| weight 7/2 | 5 | 1 | 6 | (20449ν+12544)(20449ν+28224)(20449ν+112896)(20449ν+254016)(20449ν+313600) | 5 negative, simple | elliptic, simple | all +: τ = 112/143, 168/143, 336/143, 504/143, 560/143 |
| weight 5/2 | 5 | 1 | 6 | (20449ν+1600)(20449ν+12544)(20449ν+166464)(20449ν+112896)² | 4 negative; −112896/20449 double | elliptic, one double frequency | τ = 40/143: −; 112/143: −; 408/143: +; 336/143 (double, dim 4): + |
| weight 3/2 | 5 | 1 | 6 | (20449ν+3136)(20449ν+57600)(20449ν+112896)(418161601ν²+3660534592ν+5267275776) | 5 negative, simple | elliptic, simple | τ = 56/143: −; 240/143: +; 336/143: −; 2.634093: −; 1.347379: − |
| weight 1/2 | 5 | 1 | 6 | (20449ν+57600)(418161601ν²−223793856ν+1625702400)(418161601ν²+1346689344ν+8851046400) | 1 negative; 2 conjugate pairs (Re ν = 0.2676 and −1.6102) | hyperbolic (no listed sub-word) | τ = 240/143: + |
| line {7/2, −1/2} | 6 | 2 | 8 | (1254484803ν²+9490953472ν+31470387200)(3763454409ν²+3142275136ν+15735193600) | 2 conjugate pairs (Re ν = −3.7828, −0.4175) | hyperbolic (no listed sub-word) | none (no real negative root) |
| line {7/2, −3/2} | 6 | 4 | 10 | (20449ν+153664)(418161601ν²+6989958976ν+28677390336) | 3 negative, simple | elliptic, simple | all −: τ = 392/143, 3.080180, 2.688572 |
| line {7/2, −5/2} | 6 | 2 | 8 | (43681ν+529984)(7382089ν−22579200)(322457029609ν²−229584540416ν+3540418560000) | 1 negative, 1 positive (22579200/7382089), 1 conjugate pair | hyperbolic (no listed sub-word) | τ = 728/209: − |
| line {5/2, −3/2} | 6 | 2 | 12 | (20449ν+87616)(20449ν+204544) | 2 negative, simple | elliptic, simple | both +: τ = 296/143, 3.162695 |

Exact quadratic roots, as printed in `sB3.log`:
- weight 3/2: ν = (−89504 ± 736√5065)/20449
- line {7/2, −3/2}: ν = (−170912 ± 1568√217)/20449

### The four members at each point
Words, then Krein signs per cluster τ. Full polynomials and roots are in `sB3.log` and `results.json`.

| point | J | k | (i+J)/√2 | (i+k)/√2 |
|---|---|---|---|---|
| weight 7/2 | elliptic, no listed sub-word (two double roots: (20449ν+56448)²(20449ν+62720)²); +, + | same as J | elliptic, simple; +,+,+,+,+ | same as (i+J)/√2 |
| weight 5/2 | hyperbolic, real pairs ((20449ν−32896)²(20449ν+112896)²); + (τ = 336/143) | same as J | hyperbolic, a complex quartet; + (τ = 336/143, double), − (τ = 0.24348) | same as (i+J)/√2 |
| weight 3/2 | hyperbolic, real pairs ((418161601ν²+2698613632ν−8598159360)²); − | same as J | hyperbolic, a complex quartet; −, −, − | same as (i+J)/√2 |
| weight 1/2 | hyperbolic, real pairs ((418161601ν²+2661969024ν−3251404800)²); + | same as J | hyperbolic (no listed sub-word: 1 negative, 2 conjugate pairs); + | same as (i+J)/√2 |
| line {7/2, −1/2} | hyperbolic (no listed sub-word: 2 conjugate pairs); none | same as J | hyperbolic, a complex quartet; + (τ = 1.85594), − (τ = 1.34935) | same as (i+J)/√2 |
| line {7/2, −3/2} | elliptic, simple; −, −, − | same as J | elliptic, simple; −, −, − (cubic 8550986578849ν³+…) | elliptic, simple; −, −, − (a *different* cubic 657768198373ν³+…) |
| line {7/2, −5/2} | hyperbolic, one real pair; τ ≈ 0.4043: −, 1.2546: +, 3.3414: − | hyperbolic, one real pair; τ ≈ 0.4381: −, 1.1970: +, 3.3456: − | hyperbolic (no listed sub-word: 1 negative, 1 positive, 1 pair); − | same pattern as (i+J)/√2; − |
| line {5/2, −3/2} | elliptic, simple; +, + | same as J | elliptic, simple; +, +, + | same as (i+J)/√2 |

Zero multiplicities of the members:
- J and k: 8 at the four weights and at lines {7/2, −1/2} and {7/2, −5/2}; 10 at {7/2, −3/2}; 12 at {5/2, −3/2}.
- The √2 members: 6 at the four weights; 8 at {7/2, −1/2} and {7/2, −5/2}; 10 at {7/2, −3/2} and {5/2, −3/2}.

**Things that looked unusual (reported, not adjusted)**
- At line {7/2, −5/2} the slow polynomials of the four members are **not rational**. Their coefficients lie in Q(√51), which enters through the point's components √(17/38) and √(21/38). Following § 2 ("primitive with integer coefficients where it is rational"), I report these monic over Q(√51), factored over Q(√51).
  - The J and k polynomials differ there: they are Galois conjugates.
  - The √2 members' quartics differ in the sign of their √51 term.
  - Ŝ_u's own slow polynomial at that point is rational.
- At line {7/2, −3/2}, members (i+J)/√2 and (i+k)/√2 have different slow cubics. Their words and Krein signs are the same.
- Several slow roots of Ŝ_u have τ that are rational multiples of 1/143, e.g. τ = 336/143 recurs at weights 7/2, 5/2 and 3/2. I note this without interpreting it.

---

## Checks and planted defects
**Final run** (`./py run_all.py`): 409 PASS and 0 FAIL in total.
- `s0_group` 34, `s0_slot` 25, `sA1` 16, `sA5` 18, `sA4_quad` 8, `sB3` 308.

**How the PASS lines can fail.** Each one is one of these:
- an exact identity, checked in exact arithmetic;
- an exact value against an independent 50-digit route (quadrature of the defining integral, numerical eigen/SVD, or 80-digit polyroots);
- an exact count (rank, Sturm) against a numerical count.

Each fails when either side is wrong.

**Planted defects.** `plant_defects.py` re-runs the scripts with injected defects. The results are in `plant_defects.log`, and **every planted defect fired**:

| defect | what fired |
|---|---|
| q₂ off the unit sphere | the norm check |
| duplicated character | class count, Σd² = 129, row and column orthogonality |
| corrupted η | ηᴴη, the two intertwiner checks, the trace check |
| C = I | both C relations, Y⁻¹η̄C = λη, the sampled-group-point check |
| s = 1/3 | both quadrature checks of s |
| one CG sign flipped | 64×64 orthogonality, deviation 1.91 |
| q₃ + 1e−6 | quadrature vs exact Q (3e−7), solved q_J (1e−6) |
| c₅ + 1e−4 | the HS identity |
| β + 1e−3 | Q − 1 = β·HS₆ |
| extra term in the A.5 formula | 12 checks, generic code and quadrature |
| quadrature point moved by 1e−3 | 7 of 8 points (see note below) |
| Θ without its signs | τ² = −1 checks and Hamiltonian structure (assertion) |
| corrupted tangent vector | M̂K_u = 0 and n₀ ≥ 0 |
| cluster ν shifted by 1e−20 | empty kernels |
| exact matrix perturbed | assertion "odd coefficient nonzero" |
| 50-digit M̂ perturbed by 1e−10 | exact-vs-numeric signature, zero multiplicity, root match |
| Sturm count off by one | 77 FAILs |

Note on the quadrature-point defect: the eighth point, v₇/₂, did not fire, and it should not have. The defect moves the point inside span{v₇/₂, v₋₇/₂}, where A.5 shows Q is constant.

**Real problems the checks caught (fixed, reported):**
- **Too few quadrature nodes** (A.1).
- **Bad `nsimplify` results** in my first HS-factor code: sympy's `nsimplify` turned exact surd sums into garbage. I removed `nsimplify` from every exact path.
- **A check that could never fail.** The first planted-defect pass showed "signature sums to 15 − k" is an identity. I replaced it with n₀, n₊ ≥ 0, which can fail. I also added a check that h is invariant under i, J and k at a random point.
- **A wrong value in results.json.** Building it, I found the 0(d) entry was a different CG value (√210/42): a variable was reused. I fixed it, and `results.json` now carries √231/462, which is what the script prints.

## Incomplete or unverified
- The Krein signs are decided numerically: 50-digit SVD kernels, with a definiteness margin ≥ 0.04. They are not exact.
- Every other value above is exact and agrees with a high-precision route.
- No item of mine is missing.

## Consulted-material manifest
- **Files I read:** `BRIEF.md`, `worklist.md`, `clauses.md`, `inputs.json`, and the text of the `py` wrapper. I printed `py` once to see how the interpreter is invoked.
- **Loaded without my asking:**
  - `.room.sb`, the sandbox profile `./py` hands to `sandbox-exec` on every run. I never opened it.
  - The Python standard library, sympy and mpmath, from the interpreter's own environment outside the room.
- **Rule breaches, reported:**
  - My very first command was not a `./py` command: `cd` into this room's top level, `ls -la`, and `cat` of the four files.
    - The harness saved that command's long output to a file outside the room, which I did not read.
    - The `cd` was to the room's own top level, and no writes were affected.
  - Two later commands were refused by the shell before running and had no effect: a `mkdir` chained with `./py`, and a heredoc patch.
  - The harness also stores a background job's output outside the room. I did not read it; I read the room's own `plant_defects.log` instead.
- **Looked up, i.e. recalled rather than derived:**
  - Racah's closed formula for CG coefficients. I implemented it myself and verified it: CS sign, ladder relation, orthogonality.
  - The SU(2) character formula χ_j = U_{2j}(cos θ). Verified against traces of my Dʲ for j ≤ 5/2.
  - Schur orthogonality and the Haar measure in (t, phases) coordinates. The latter is confirmed by s_N = 1/4 to 50 digits.
  - The Gauss–Legendre construction, Sturm's theorem, and Descartes/Sylvester inertia.
  - No library CG or character table was used.
- **Recognised:** I recognised Γ₀ as the binary icosahedral group after deriving it. The character table and every decomposition in item 0 were DERIVED here.
