# RETURN

Items covered: 0(a)–0(e), with 0(c) for all three slots; C.1 and C.2 in each slot (using clause 4's names); C.4 at every point it names, including `O1`–`O3` in `R₄` and `R₅`. I did not work any other item. Every number below comes from a script in this room. `./py run_all.py` reruns them all in order and writes `<script>.log` beside each script, plus `run_all.log`. `results.json` holds the values, keyed as the brief asks.

## Conventions used (where Section 2 leaves a choice)

* **Quaternion to matrix:** `w + xi + yj + zk ↦ [[w + x·I, y + z·I], [−y + z·I, w − x·I]]`. This is a homomorphism (`i·j = k` is checked).
* **`V_j`:** binary forms of degree 2j in `(x, y)`, with orthonormal basis `e_m = x^{j+m} y^{j−m}/√((j+m)!(j−m)!)`.
  * The weighted coordinate `p_m` is the coefficient of the monomial.
  * `g = [[a,b],[c,d]]` acts by `x ↦ ax + cy`, `y ↦ bx + dy`, and `D^j(g)_{mm'} = ⟨e_m, g·e_{m'}⟩`.
  * `J₋e_m = √((j+m)(j−m+1)) e_{m−1}`, which is the Condon–Shortley phase.
  * Rows and columns are ordered `m = j, …, −j`.
* **Clebsch–Gordan coefficients:** I built them myself from Racah's formula, kept exact as `sign·√(rational)`.
  * The convention is checked by its defining property: `W_J (D^{j1}⊗D^{j2}) = D^J W_J` for every J. This holds to 1e-50 for (3,3), (7/2,7/2) and (6,7/2).
  * I also compared them with `sympy.physics.quantum.cg.CG` (a library lookup). All 1764 coefficients used agree.
* **Θ:** `(Θu)_m = (−1)^{j−m} ū_{−m}`, i.e. `Θu = U ū`. I checked that it commutes with `D^j` for j = 1/2, 3, 7/2.
* **Infinitesimal generators:** the four generators in `inputs.json` span the same space as my phase and su(2) generators (rank 4 of the 8 together, exact), and each is G-skew (exact). So my `D^j` and the bench's realize the same symmetry chart.
* **Measure:** wherever a norm depends on normalization, I use `V = R = 1`. That means normalized Haar measure on SU(2), equal to the measure on X rescaled to volume 1, with `ψ_y` isometric.

---

## Anchors

### 0(a) The group — DERIVED in this room

* **Generator norms:** `|q₁|² = 1` and `|q₂|² = 1`, exactly (arithmetic in ℚ(√5)).
* **Order:** closing `{q₁, q₂}` under multiplication, exactly, gives a group of **order 120**.
* **Conjugacy classes:** found by explicit conjugation. There are 9, and the real part is constant on each class and different between classes.

| class | Re g | size | element order |
|---|---|---|---|
| 0 | 1 | 1 | 1 |
| 1 | (1+√5)/4 | 12 | 10 |
| 2 | 1/2 | 20 | 6 |
| 3 | (√5−1)/4 | 12 | 5 |
| 4 | 0 | 30 | 4 |
| 5 | (1−√5)/4 | 12 | 10 |
| 6 | −1/2 | 20 | 3 |
| 7 | −(1+√5)/4 | 12 | 5 |
| 8 | −1 | 1 | 2 |

**Characters**, with columns in the class order above:

| irrep | values |
|---|---|
| 1 | 1, 1, 1, 1, 1, 1, 1, 1, 1 |
| 2 (= V_{1/2}) | 2, (1+√5)/2, 1, (√5−1)/2, 0, (1−√5)/2, −1, −(1+√5)/2, −2 |
| 2′ | 2, (1−√5)/2, 1, −(1+√5)/2, 0, (1+√5)/2, −1, (√5−1)/2, −2 |
| 3 (= V_1) | 3, (1+√5)/2, 0, (1−√5)/2, −1, (1−√5)/2, 0, (1+√5)/2, 3 |
| 3′ | 3, (1−√5)/2, 0, (1+√5)/2, −1, (1+√5)/2, 0, (1−√5)/2, 3 |
| 4 | 4, −1, 1, −1, 0, −1, 1, −1, 4 |
| 4s (= V_{3/2}) | 4, 1, −1, −1, 0, 1, 1, −1, −4 |
| 5 (= V_2) | 5, 0, −1, 0, 1, 0, −1, 0, 5 |
| 6 (= V_{5/2}) | 6, −1, 0, 1, 0, −1, 0, 1, −6 |

How the table was obtained:
* **Characters of `V_j`:** taken as the trace of the exact `D^j`, which equals the Chebyshev polynomial `U_{2j}(Re g)` (checked for 2j ≤ 8).
* **Irreducibles read directly:** `V_j` is irreducible for 2j = 0..5 (norm 1).
* **The other three:**
  * `2′ = V_{7/2} − 6`.
  * `4 = 2 ⊗ 2′`.
  * `3′ = V_3 − 4`.
* **Completeness checks:** each of these has norm 1, the nine characters are orthonormal, Σ dim² = 120, and there are 9 irreducibles for 9 classes.
* **Galois observation:** 2′ and 3′ are the √5 ↦ −√5 conjugates of 2 and 3. This is checked, not used.

I recognise the group as the binary icosahedral group. Only that name is RECOGNISED; everything else above was DERIVED. Nothing here surprised me.

### 0(b) Levels ℓ ≤ 20 (multiplicities) — DERIVED

| irrep (dim) | levels ℓ: multiplicity |
|---|---|
| 1 (1) | 0:1, 12:1, 20:1 |
| 2 (2) | 1:1, 11:1, 13:1, 19:1 |
| 2′ (2) | 7:1, 13:1, 17:1, 19:1 |
| 3 (3) | 2:1, 10:1, 12:1, 14:1, 18:1, 20:1 |
| 3′ (3) | 6:1, 10:1, 14:1, 16:1, 18:1, 20:1 |
| 4 (4) | 6:1, 8:1, 12:1, 14:1, 16:1, 18:2, 20:1 |
| 4s (4) | 3:1, 9:1, 11:1, 13:1, 15:1, 17:1, 19:1 |
| 5 (5) | 4:1, 8:1, 10:1, 12:1, 14:1, 16:2, 18:1, 20:2 |
| 6 (6) | 5:1, 7:1, 9:1, 11:1, 13:1, 15:2, 17:2, 19:2 |

The multiplicities add up to dim `V_{ℓ/2}` for every ℓ ≤ 20.

Each name in Section 1.1 picks out exactly one irreducible:
* **R₂** (dim 2, lowest level 7) = **2′**
* **R₄** (dim 3, lowest level 6) = **3′**
* **R₅** (dim 4, lowest level 6) = **4**

The 4-dimensional 4s has lowest level 3, so it does not compete for R₅.

### 0(c) Lowest level: multiplicity, η, s_N

* **Multiplicity:** 1 in all three slots. It is the trace of the exact isotypic projector `P = (dim ρ/120) Σ χ̄(g) D^j(g)`.
* **Exact checks on `P`:**
  * Computed exactly over ℚ(√5)(i) in weighted coordinates.
  * `P² = P`, `PD(q_i) = D(q_i)P`, and `P` is Hermitian in the CS basis.
  * P turns out to be real in my convention. Its exact entries are in `results.json` → `0(c)`.
* **η:** I took η as an orthonormal basis of the range of P (Gram–Schmidt on the columns, 50 digits). It is listed at 40 digits in `results.json`.
  * Checks: `η†η = I`, `ηη† = P`, `D(h)η = η ρ(h)` for all 120 h, and `tr ρ(h) = χ(h)` for all h.
  * η is only fixed up to `η ↦ ηV` with V unitary; P is the invariant object.
* **`s_N`:**
  * Exact route: Schur orthogonality gives `s_N = dim ρ/(2j+1)`.
  * High-precision route: exact-degree cubature over SU(2) (Hopf coordinates, Gauss–Legendre in t = sin²η) at 50 digits, at a random w. It agrees to better than 1e-40.

| slot | multiplicity | **s_N** |
|---|---|---|
| R₂ | 1 | **1/4** |
| R₄ | 1 | **3/7** |
| R₅ | 1 | **4/7** |

### 0(d)

* **`⟨3 3; 3 −3 | 6 0⟩ = √231/462 ≈ 0.0328976`, which is positive.** Exact, from Racah's formula; the library value agrees.
* **Θ² = +1 on V₃ and Θ² = −1 on V_{7/2}.** Exact, since U² = (−1)^{2j}.

### 0(e) R₂: the quaternionic structure

* **C:** with my η, the solutions of `ρ̄(h)C = Cρ(h)` form a one-dimensional space (singular values 2.57, 2.0, 1.618, 2.6e-51). Scaled to be unitary, C satisfies the relation for all 120 h.
* **C·C̄ = −1:**
  * Numerically to 1e-51.
  * Exactly, from the Frobenius–Schur indicator of 2′, which equals −1 exactly (`s0_group.log`).
  * By the same exact count, 3′ and 4 have indicator +1, consistent with `C C̄ = 1` in R₄ and R₅.
* **The map on w:**
  * In general `ψ_w ↦ ψ̄_w C` equals `ψ_{cΘw}`.
  * Here c is defined by `Uη̄C = cη`, with |c| = 1 (checked). c depends on C's phase: `C ↦ e^{ia}C` gives `c ↦ e^{ia}c`.
  * Choosing the phase so that c = 1, **the map is exactly Θ**. This was checked directly by comparing `conj(ψ_w)C` with `ψ_{Θw}` at random points of SU(2) (agreement to 6e-51).
  * For that choice, C agrees numerically (to about 2e-51) with `[[0, 1], [−1, 0]]` in my η basis. This is a numerical value, not an exactness claim.
* **The Sp(1) action — this is a reading.** The worklist says the phase and the quaternionic structure "generate a group Sp(1)". Taken literally, the group generated by `e^{iα}` and J = Θ is only U(1) ∪ U(1)J. I read the Sp(1) as the unit quaternions in the real algebra spanned by `1, i, J, iJ`, i.e. the fibre maps `ψ ↦ aψ + bψ̄C` with |a|² + |b|² = 1. On E₀ this acts as **`w ↦ a w + b Θw`**. Here `J² = Θ² = −1` and `Θ i = −i Θ`, so E₀ = V_{7/2} is a left ℍ-module of quaternionic dimension 4, and Sp(1) acts by left multiplication.

---

## C.1 (with clause 4: `s_N`)

* **`s_N`:** R₂: 1/4, R₄: 3/7, R₅: 4/7 (see 0(c)).
* **The point y (my choice, with no symmetry):** `y_m ∝ (k+1) + i·((k² mod 7) − 3)` for the k-th component, k = 0, 1, … in the order m = j..−j. That is:
  * R₂: (1−3i, 2−2i, 3+i, 4−i, 5−i, 6+i, 7−2i, 8−3i)/√234
  * R₄ and R₅: the first seven of these, divided by √161
* **Normalization:** `V = R = 1`.

Method. The structure, checked numerically below, is:
* `|ψ_w|² = s⁻¹ Σ_J [w⊗Θw]_J^T D^J b_J`, and only J = 0, 6 occur (the Γ₀-invariants of `V_J` for J ≤ 2j; checked exactly).
* For L ≠ n this gives `‖Π_L N(ψ_y)‖² = g_K ‖[[y⊗Θy]₆ ⊗ y]_K‖²` with K = L/2. The slot constants `g_K` are exact rationals, computed from P (`s2_constants.log`).

Checks:
* **`g_K` vanishes off the slot's levels:** `g_K = 0` at every level where the slot does not occur (exact).
* **Independent cubature:** every level norm was recomputed at double precision by cubature straight from the definition (projecting `N(ψ_y)` onto each `D^K` on an exact-degree grid). It agrees at every level to ≤ 2e-16. The norms at levels where the slot does not occur come out ≤ 1.5e-29, and Parseval holds.
* **Exactness:** the values are exact, in ℚ(√2, √3, √5, √7) because of the CG surds. The exact strings are in `results.json`; 30 digits are shown here.

| slot | level L | ‖Π_L N(ψ_y)‖² |
|---|---|---|
| R₂ | 13 | 0.0660851469793462121705968871823 |
| R₂ | 17 | 0.0397809087213870680725581240991 |
| R₂ | 19 | 0.0417979907871338345813473000321 |
| R₄ | 10 | 0.0100377065306555069977190887686 |
| R₄ | 14 | 0.00947937756537922034668814396726 |
| R₄ | 16 | 0.00791327396582868385937303808379 |
| R₄ | 18 | 0.0139280082921310614673203703343 |
| R₄ | 20 | 0 (exactly: N has no component above K = 6 + j = 9) |
| R₅ | 8 | 0.00686943882501450947505896604750 |
| R₅ | 12 | 0.00804109190099201777062085737164 |
| R₅ | 14 | 0.00247564101595841245661275188431 |
| R₅ | 16 | 0.00539282011853949969743330840566 |
| R₅ | 18 | 0.00768061260841736324888727118548 |
| R₅ | 20 | 0 (exactly, same reason) |

**Levels where the norm is nonzero:**
* R₂: 13, 17, 19
* R₄: 10, 14, 16, 18
* R₅: 8, 12, 14, 16, 18

**Reading:** "every level above the lowest at which the slot occurs, up to 20" means the levels L with n < L ≤ 20 at which the slot occurs. At every other level, `Π_L N = 0` identically, because N(ψ) is a section of the slot. The cubature confirms this numerically for all levels up to 3n.

## C.2

| slot | lowest level n | next level | λ₁ − λ₀ (units 1/R²) |
|---|---|---|---|
| R₂ | 7 | 13 | **132** |
| R₄ | 6 | 10 | **72** |
| R₅ | 6 | 8 | **32** |

These come from 0(b), with λ_L = L(L+2).

## C.4 Sign of R̃₀(u)

**Structural fact.** Every term of `R̃₀ = −Σ_L ‖Π_L N‖²/(λ_L − λ₀)` has `λ_L > λ₀`. So R̃₀ ≤ 0 always, and R̃₀ < 0 unless `P⊥N(ψ_u) = 0`. The computations below confirm strict negativity at every point.

**Criticality** (needed for "in each slot where it is critical"):
* I write `h = ‖w‖⁴ + c₆‖[w⊗Θw]₆‖²`, where c₆ ≠ 0 is a slot constant. So u is critical exactly when `∂F/∂w̄ = 2F(u)u`, with `F = ‖[w⊗Θw]₆‖²`.
* I tested this exactly (sympy) at each normalized point. All 10 census points (critical in both R₄ and R₅) and all 8 R₂ points are critical.
* The exact test does fire on a non-critical point (planted defect 8).

**Values:**
* R̃₀ is exact (sympy, `V = R = 1`); every value is a negative rational.
* High-precision route: an independent 130-digit real-chart code (`s4_boxes.py`) agrees with all 28 exact values to 2.5e-132.
* Definition route: direct double-precision cubature agrees to ~1e-16.

| slot | point | critical | R̃₀ (exact) | ≈ | sign |
|---|---|---|---|---|---|
| R₄ | C3 ray | yes | −1033699247/643990132500 | −1.6051e-3 | − |
| R₄ | D2 ray | yes | −593698945/544042863936 | −1.0913e-3 | − |
| R₄ | coherent v3 | yes | −608931967/110168679947040 | −5.5273e-6 | − |
| R₄ | hexagon | yes | −1921938515/1377108499338 | −1.3956e-3 | − |
| R₄ | octahedron | yes | −2203040/2833556583 | −7.7748e-4 | − |
| R₄ | pentagonal pyramid | yes | −267786421/175633672500 | −1.5247e-3 | − |
| R₄ | trigonal prism | yes | −336158940460/452437047342423 | −7.4300e-4 | − |
| R₄ | v1 | yes | −30976085/90673810656 | −3.4162e-4 | − |
| R₄ | v2 | yes | −23286907/85006697490 | −2.7394e-4 | − |
| R₄ | zonal v0 | yes | −1871763250/688554249669 | −2.7184e-3 | − |
| R₅ | C3 ray | yes | −15793825383/27476912320000 | −5.7480e-4 | − |
| R₅ | D2 ray | yes | −670232535/967187313664 | −6.9297e-4 | − |
| R₅ | coherent v3 | yes | −122896067/38687492546560 | −3.1766e-6 | − |
| R₅ | hexagon | yes | −75480591711/38687492546560 | −1.9510e-3 | − |
| R₅ | octahedron | yes | −54176703/75561508880 | −7.1699e-4 | − |
| R₅ | pentagonal pyramid | yes | −1599151113/2497901120000 | −6.4020e-4 | − |
| R₅ | trigonal prism | yes | −3697737715/4965015608696 | −7.4476e-4 | − |
| R₅ | v1 | yes | −11034061815/7737498509312 | −1.4261e-3 | − |
| R₅ | v2 | yes | −415251627/2417968284160 | −1.7174e-4 | − |
| R₅ | zonal v0 | yes | −5273625/15112301776 | −3.4896e-4 | − |
| R₂ | line {5/2, −3/2} | yes | −27273813/37780754440 | −7.2190e-4 | − |
| R₂ | line {7/2, −1/2} | yes | −182482085/158679168648 | −1.1500e-3 | − |
| R₂ | line {7/2, −3/2} | yes | −6768561/1092831740 | −6.1936e-3 | − |
| R₂ | line {7/2, −5/2} | yes | −23307740389/14991465809320 | −1.5547e-3 | − |
| R₂ | weight 1/2 | yes | −728595/1889037722 | −3.8570e-4 | − |
| R₂ | weight 3/2 | yes | −28845747/9445188610 | −3.0540e-3 | − |
| R₂ | weight 5/2 | yes | −316750/944518861 | −3.3536e-4 | − |
| R₂ | weight 7/2 | yes | −1629062/33058160135 | −4.9279e-5 | − |

### O₁–O₃ in R₄ and R₅ (interval arithmetic, `mpmath.iv` at 130 digits, `s4_boxes.py`)

**Krawczyk test** of the 19-equation slice system, on each box of `inputs.json`:
* **The box:**
  * Its x-part centre is as given, c = 0, and λ = 4h(x)/|x|²_G from my own h.
  * Radius 1e-60, rounded *inward*, so a zero found lies inside the stated box.
  * x₀ is enclosed from its exact decimal.
  * `k_i = X_i x₀` uses the bench's exact generators.
* **Result:** `K(B) ⊂ int(B)` for all six boxes. The width ratio K/B is ≤ 7.5e-55 and `‖I − Y J(B)‖∞ ≤ 7.5e-55`. So each box contains a unique zero of the slice system.

**Why the zero is a critical point (c = 0):**
* h is invariant and the `X_i` are G-skew. So at any zero, `A c = 0`, where `A_{ik} = ⟨X_i x, G k_k⟩`.
* Over each box, A preconditioned by its midpoint inverse is strictly diagonally dominant (diagonal ≥ 1 − 6e-59, off-diagonal row sums ≤ 2e-58). So A is nonsingular, c = 0, and `∇h = λGx` with `|x|_G = 1`. The certified point is a **critical point** of h.
* The two invariance identities used were also checked as intervals containing 0 over each box.

**Sign:** R̃₀(x/|x|_G) was enclosed over the whole x-part of each box. The enclosures are outward-rounded, at 45 significant digits.

| box | critical | R̃₀ enclosure over the box [lower, upper] | width | sign |
|---|---|---|---|---|
| O1 R4 | yes (certified) | [−0.000863329763239803973508472229496633990783339748, −0.000863329763239803973508472229496633990783339747] | 7.0e-61 | − |
| O1 R5 | yes (certified) | [−0.000786103199105844553814426531720552667356782768, −0.000786103199105844553814426531720552667356782767] | 5.8e-61 | − |
| O2 R4 | yes (certified) | [−0.00156451543861247101284463728995918182412852963, −0.00156451543861247101284463728995918182412852962] | 1.0e-60 | − |
| O2 R5 | yes (certified) | [−0.000596034322637912719136676783441298348027258702, −0.000596034322637912719136676783441298348027258701] | 5.7e-61 | − |
| O3 R4 | yes (certified) | [−0.00157748742119318453653509477632426298977949940, −0.00157748742119318453653509477632426298977949939] | 1.6e-60 | − |
| O3 R5 | yes (certified) | [−0.000576860076001386977530470583141830922381440359, −0.000576860076001386977530470583141830922381440358] | 6.8e-61 | − |

A further check confirms that each printed decimal enclosure contains the computed interval and carries its sign. I added it after my outward-rounding helper first dropped the sign bit (mpmath's mantissa is unsigned). That bug was caught on inspection and fixed before any value was reported.

**The sign of R̃₀ is negative over every box.** I computed the centre λ from my own h, and the six certifications passed with it. That means my h agrees with the function the bench used to place these boxes.

---

## Run summary (`run_all.log`)

```
s0_group.py      PASS 26  FAIL 0
s1_slots.py      PASS 56  FAIL 0
s2_constants.py  PASS 24  FAIL 0
s3_points.py     PASS 28  FAIL 0
s4_boxes.py      PASS 33  FAIL 0
planted.py       10 planted defects, all detectors fired (its FAIL lines are the intended firings)
assemble.py      writes results.json
```

The scripts in `tools/` are one-off editors I used to patch `s4_boxes.py` and similar files. They are not part of the computation and `run_all.py` does not run them.

## Checks that can fail, and planted defects

Every `[PASS]` line is an evaluated comparison. `planted.py` reruns ten of them on deliberately corrupted inputs, and each must print FAIL (`planted.log`):
1. q₂ with φ in place of φ⁻¹: the generator norm check fails.
2. A character with one value shifted: the orthonormality check fails.
3. One CG coefficient with its sign flipped: the product-rule check fails.
4. Θ without its `(−1)^{j−m}` phase: the commutation check fails.
5. A wrong closed form for s_N: the cubature comparison fails.
6. A non-invariant projector: the "c_J only at J ∈ {0,6}" check fails.
7. A g_K perturbed by 1e-6: the cubature level-norm comparison fails.
8. The non-critical point v₂ + v₃/10: the exact criticality test reports it non-critical.
9. Krawczyk with the box centre shifted by 1e-55: fails.
10. Krawczyk with c₆ perturbed by 1e-6: fails.

Two checks also caught real defects during development:
* My first Gauss–Legendre nodes had duplicates (bad starting guesses). The node check printed FAIL and the s_N cubature disagreed. I fixed this by seeding Newton refinement from numpy's nodes.
* My first CG routine used `nsimplify`, which is a guess. I replaced it with the exact `pre·s²` square before reporting any value.

## Anything that looked wrong / open points

* **Nothing in the group or the slots surprised me.**
* **Internal constants of h:** I needed these for C.4 (the slot constants c₀, c₆ of h and g_K). They are in `s2_constants.log`. They are intermediate quantities, **not** answers to A-items, which are not mine.
* **η:** it is not canonical, as explained in 0(c). The 0(e) phase c, and so the matrix C, depend on the choice of η and of C's phase. C·C̄ and the statement "map = Θ for a suitable phase of C" do not.

## Rule incidents (reported, not concealed)

* **My first shell command was `cd <room>; ls -la; cat …; cat inputs.json`.** It did not begin with `./py`, because I issued it before reading the Tools rule in BRIEF.md. It read only files inside this room.
  * The harness saved its long output to a tool-results file outside the room, in the agent's own data directory. I did not open that file.
* **One later compound command (`./py … ; grep …`) was refused by the permission layer and not executed.** I redid that step with a patch script under `tools/`.
* **Background runs:** the harness writes the stdout of my background runs to its own task-output files outside the room. I never read those; I read the in-room `.log` files instead.

## Consulted-material manifest

* **Files read in this room:**
  * `BRIEF.md`, `worklist.md`, `clauses.md`, `inputs.json`
  * `py`, read as text via `./py`
  * my own scripts and logs
  * Directory listings showed `.room.sb`; I did not read it.
* **Loaded without my asking:** the Python standard library, numpy, sympy, mpmath, and their dependencies, through `./py`. scipy is available but not imported.
* **Looked up rather than derived:**
  * `sympy.physics.quantum.cg.CG`, used only as a comparison for my own CG. It agrees on all 1764 coefficients used.
  * From memory, validated in this room:
    * Racah's closed formula for CG coefficients (validated by the product rule).
    * The Haar measure in Hopf coordinates (validated by the exact `s_N`).
    * Krawczyk's theorem.
    * The Frobenius–Schur criterion.
    * The name "binary icosahedral group" (RECOGNISED; the table itself was DERIVED).
* **Nothing outside this room was read, and there was no network access.**
