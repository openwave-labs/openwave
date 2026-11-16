# AUDIT — stage 2 (adversarial)

**Bottom line.** I could not break any reported value in my scope. Every claim in the anchors (all three rooms), C.1–C.5, Block D and B.4 was recomputed by my own code, which is stage 1's library copied unchanged into `audit/`. All of them agree.

The defects I can demonstrate are in **checks, not values**:
* `solver_ca` describes a 130-digit check as "independent" when it is not.
* Several solvers' planted defects do not exercise the checks their values rest on.

These are listed in §4. Verdict counts:
* **DISAGREE: 0.**
* **DEFECT (value): 0.**
* **DEFECT (check / claim about a check): 1** (`solver_ca`).
* **UNCHECKED:** only what §6 lists.

Notation follows clause 4 (`s_N`, `ξ`, `ν = ξ²`, `s_g`). Solvers: **ca** = `solver_ca`, **cb** = `solver_cb`, **cert** = `solver_cert`. "Mine" means my stage-1 value (`own/`, not edited) or a value computed in this stage by `audit/*.py`.

## 0. How I checked (my routes)

* **Library.** The stage-1 library (`core`, `reps`, `s3poly`, `hpoly`, `cgroute`, `exact_pt`, `krawczyk`, `ivutil`, `numrep`, `chk`) was copied unchanged from `own/` into `audit/`. Every run used `./py -B`, so no bytecode was written. `own/` is untouched: its newest file time is the room's creation time.
* **Route E (exact).** S³ harmonic decomposition over Q(√5, i) with exact Haar moments. It needs Gaussian-rational weighted coordinates, up to a scale.
* **Route C (CG coupling).** Coupling of V_j^{⊗3}, run in `mpmath` at 50–70 digits, or in `mpmath.iv` at 90–130 digits.
* **Exact quartic h.** Built from route-E channel coefficients, with exact sympy criticality, charpolys and signatures.
* **Own Krawczyk / Faddeev–LeVerrier / Descartes certifier.** Uses `mpmath.iv` at 130 digits.
* **Solver code.** None of the solvers' code was imported or run. Their scripts were only read, and their returned numbers were parsed for comparison.

| script | what | clean log | planted run (must FAIL) |
|---|---|---|---|
| `a_anchors.py` | 0(a)–0(e) for all three rooms | 63 PASS / 0 FAIL | `plant`: one sign of cert's η flipped → 2 FAIL |
| `a_C1.py` | C.1 at ca's y and at cb's y | 56 / 0 | `plant`: cb L=14 value ×(1+1e-12) → 1 FAIL |
| `a_C4.py` | C.4 at all 28 points + 6 boxes | 102 / 0 | `perturb`: hexagon + 1e-20·v₁ → criticality and ca-match FAIL (4) |
| `a_C5.py` | C.5 at all five cases (exact charpoly + 50/70-digit numerics) | 135 / 0 | — (the comparison with cb is planted in `a_cmp_C5`) |
| `a_cmp_C5.py` | cb's C.5 vs mine, group by group | 55 / 0 | `plant`: one cb value negated per case → 5 FAIL |
| `a_D_all.py` | Block D + B.4 at all six boxes (stage 1 did only O3) | 68 / 0 | `Dbox`: box shifted 1e-40 → Krawczyk FAIL |
| `a_cmp_D.py` | cert's D/B.4 vs mine, enclosure intersection | 72 / 0 | `shift`: cert 924 r̂₆ moved 1e-50 → FAIL |
| `a_quartic.py` | cert's quartic h vs mine, coefficientwise (exact) | 4 / 0 | `plant`: one coefficient + 1e-30 → FAIL |
| `a_reading.py` | C.3 under the two fibre-group readings | 2 / 0 | — (a demonstration, see §3) |
| `a_assemble.py` | cb's C.3 vs my stage-1 C.3; writes `audit.json` | 9 / 0 | `plant`: cb γ_u(v₁, R₅) negated → FAIL |

* **A plant that was too weak.** My first `shift` plant (1e-60) did not fire, because my printed enclosure is wider than that (60 significant digits). I raised it to 1e-50 and it fired. Reported, not hidden.
* **A check that fired on my own bug.** My first `a_anchors.py` run printed FAIL for ca's η. The cause was a bug in **my** rebuild of ca's convention: the normalisation factor was inverted. After the fix it passes. That accident is further evidence the check can fail.

---

## 1. Comparison tables (verdict per item, slot and point)

### Anchors

| item | ca | cb | cert | mine (stage 1) | verdict | how checked |
|---|---|---|---|---|---|---|
| 0(a) norms, order, classes | 1, 1, 120, 9 classes | same | same | same | **AGREE ×3** | Each table mapped to mine through the class real part. Sizes and orders are equal, and the 9 character rows are equal exactly in Q(√5). |
| 0(b) levels ≤ 20, names | R₂ = 2′, R₄ = 3′, R₅ = 4 | 2@7, 3@6, 4@6 | (2,7), (3,6), (4,6) | χ₂, χ₃, χ₆ | **AGREE ×3** | All 9 level tables equal mine. Every slot name maps to my slot irreducible. |
| 0(c) R₂: mult., s_N, η | 1, 1/4, η | 1, 1/4, exact η | — (not in scope) | 1, 1/4 | **AGREE** (ca, cb) | cb: ηηᵀ equals **my exact projector**, exactly. ca: η verified in ca's own convention, which I rebuilt: Dη = ηρ for all 120 h (3e-40), tr ρ = χ, and P_exact = ηη†. |
| 0(c) R₄ | 1, 3/7 | 1, 3/7 | 1, 3/7 | 1, 3/7 | **AGREE ×3** | cb and cert η are identical exact matrices, and both give exactly my projector. |
| 0(c) R₅ | 1, 4/7 | 1, 4/7 | 1, 4/7 | 1, 4/7 | **AGREE ×3** | as R₄ |
| 0(d) | +√231/462; Θ² = +1, −1 | +1/√924; same | +√231/462; same | +1/√924; same | **AGREE ×3** | exact |
| 0(e) R₂ | C C̄ = −1; map = Θ (phase of C chosen) | C = [[0,1],[−1,0]]; map = −Θ | — | C C̄ = −1; map = e^{iα}Θ | **AGREE** (ca, cb) | cb: (Θη)C = −η exactly. ca: U η̄ C = η in ca's convention (5e-51). The three statements differ only by C's free phase. |

### C.1 (level norms) and C.2

Each solver chose its own y, so I recomputed the level norms **at each solver's y**:
* **cb's y** has Gaussian-integer weighted coordinates, so route E applies.
* **ca's y** has Gaussian-integer orthonormal components. I used route C at 60 digits and compared with ca's surd expressions evaluated at 80 digits.

Only reading A of ca's y (orthonormal components) matches; reading B (weighted coordinates) is off by 0.03–0.06. So ca's y is unambiguous.

| slot | ca: nonzero levels / max diff from mine | cb: nonzero levels / comparison | mine (own y) | verdict |
|---|---|---|---|---|
| R₂ | {13, 17, 19} / 3.9e-62 | {13, 17, 19} / equal exactly at all L = 8..20 (and L = 21 = 0) | {13, 17, 19} | **AGREE** (ca, cb) |
| R₄ | {10, 14, 16, 18}; L = 20 → 0 / 2.4e-63 | {10, 14, 16, 18} / equal exactly at all L = 7..20 | {10, 14, 16, 18} | **AGREE** (ca, cb) |
| R₅ | {8, 12, 14, 16, 18}; L = 20 → 0 / 1.2e-63 | {8, 12, 14, 16, 18} / equal exactly | {8, 12, 14, 16, 18} | **AGREE** (ca, cb) |

C.2: next level and λ₁ − λ₀ are 13 / 132, 10 / 72 and 8 / 32 (units 1/R²) in all three returns. **AGREE** (ca, cb).

### C.3 (cb only; compared with my stage-1 exact values, mechanically in `a_assemble.py`)

| point / quantity | cb | mine | verdict |
|---|---|---|---|
| R₂ line {5/2, −3/2}: \|Γ_u\| | 24 | 24 | **AGREE** (numerical in both rooms; see §5) |
| K′, dim | dim 3; e₁: p_{1/2} = 13, p_{−7/2} = 1; e₂(z) | dim 3 | **AGREE**: the spans are equal exactly (rank of the 6 stacked vectors is 3) |
| absolutely irreducible | yes (commutant dim 1) | yes (commutant dim 1) | **AGREE** |
| n₋ (signature) | 0 ([0, 3, 6]) | 0 ([0, 3, 6]) | **AGREE** |
| γ_u | 115020/20333173 R² | 115020/20333173 R² | **AGREE** (exact) |
| v₁ R₄: k, r_u (clause 5) | 3, 1 | 3, 1 | **AGREE** |
| K′, circle weight | C·v₋₃, +4 | m = −3 weight space, 4 | **AGREE** |
| γ_u | −19919/6697728 R² | −19919/6697728 R² | **AGREE** (exact) |
| zero multiplicity off K′; slow polynomial | 4; (20449ν+784)(20449ν+12544)(1656369ν+19600)(1656369ν+1960000) | identical | **AGREE** (exact polynomial identity) |
| v₁ R₅: k, r_u | 3, 1 | 3, 1 | **AGREE** |
| K′, weight | C·v₋₃, +4 | same | **AGREE** |
| γ_u | 224487/41628563 R² | 224487/41628563 R² | **AGREE** |
| zero multiplicity off K′; slow polynomial | 4; (20449ν+3969)(81796ν+30625)(327184ν+1225)(327184ν+3969) | identical | **AGREE** |

### C.4 (ca), V = R = 1

* **Criticality.** I re-checked every point exactly with sympy on my exact quartic. All 28 points are critical, with residual exactly 0.
* **Planted perturbation.** Adding 1e-20·v₁ to the hexagon makes the residual 1.4e-19 in R₄ and 8e-20 in R₅, and the test FAILs. So criticality is exact, not tolerance-level.
* **Values.**
  * "Exact =" means my route E value is **identical** to ca's rational.
  * Every point, exact or not, was also compared with my route C at 60 digits. The worst difference is 1.8e-63.
  * Pentagonal pyramid in R₄: my stage-1 exact cubic-form value −267786421/175633672500 equals ca's.

| slot | point | ca R̃₀ | mine | sign | verdict |
|---|---|---|---|---|---|
| R₄ | C3 ray | −1033699247/643990132500 | −0.0016051476487491056 (CG, \|Δ\| 1.5e-64) | − | **AGREE** |
| R₄ | D2 ray | −593698945/544042863936 | −0.0010912723690643637 (7.6e-64) | − | **AGREE** |
| R₄ | coherent v3 | −608931967/110168679947040 | exact = | − | **AGREE** |
| R₄ | hexagon | −1921938515/1377108499338 | exact = | − | **AGREE** |
| R₄ | octahedron | −2203040/2833556583 | exact = | − | **AGREE** |
| R₄ | pentagonal pyramid | −267786421/175633672500 | exact = (stage 1) | − | **AGREE** |
| R₄ | trigonal prism | −336158940460/452437047342423 | −0.00074299605311848183 (1.5e-64) | − | **AGREE** |
| R₄ | v1 | −30976085/90673810656 | exact = | − | **AGREE** |
| R₄ | v2 | −23286907/85006697490 | exact = | − | **AGREE** |
| R₄ | zonal v0 | −1871763250/688554249669 | exact = | − | **AGREE** |
| R₅ | C3 ray | −15793825383/27476912320000 | −0.00057480350044658875 (7.6e-65) | − | **AGREE** |
| R₅ | D2 ray | −670232535/967187313664 | −0.00069297076743175539 (4.6e-64) | − | **AGREE** |
| R₅ | coherent v3 | −122896067/38687492546560 | exact = | − | **AGREE** |
| R₅ | hexagon | −75480591711/38687492546560 | exact = | − | **AGREE** |
| R₅ | octahedron | −54176703/75561508880 | exact = | − | **AGREE** |
| R₅ | pentagonal pyramid | −1599151113/2497901120000 | −0.00064019792464803411 (0.0 at 60 digits) | − | **AGREE** |
| R₅ | trigonal prism | −3697737715/4965015608696 | −0.00074475852775237601 (2.3e-64) | − | **AGREE** |
| R₅ | v1 | −11034061815/7737498509312 | exact = | − | **AGREE** |
| R₅ | v2 | −415251627/2417968284160 | exact = | − | **AGREE** |
| R₅ | zonal v0 | −5273625/15112301776 | exact = | − | **AGREE** |
| R₂ | line {5/2, −3/2} | −27273813/37780754440 | exact = | − | **AGREE** |
| R₂ | line {7/2, −1/2} | −182482085/158679168648 | −0.0011500065607528 (0.0) | − | **AGREE** |
| R₂ | line {7/2, −3/2} | −6768561/1092831740 | −0.0061935984765596212 (1.8e-63) | − | **AGREE** |
| R₂ | line {7/2, −5/2} | −23307740389/14991465809320 | −0.0015547339189814167 (1.5e-64) | − | **AGREE** |
| R₂ | weight 1/2 | −728595/1889037722 | exact = | − | **AGREE** |
| R₂ | weight 3/2 | −28845747/9445188610 | exact = | − | **AGREE** |
| R₂ | weight 5/2 | −316750/944518861 | exact = | − | **AGREE** |
| R₂ | weight 7/2 | −1629062/33058160135 | exact = | − | **AGREE** |

**The boxes.** ca's C.4 box enclosures were compared with my route C in `mpmath.iv` (90 digits) over the Section-5 box, normalised by |x|⁶_G. Every pair of enclosures intersects and excludes 0.

| box | ca enclosure (lower end) | mine (lower end, width) | sign | verdict |
|---|---|---|---|---|
| O1 R4 | −0.000863329763239803973508472229496633990783339748 | −0.00086332976323980397350847222949663399078333974760954 (7.1e-61) | − | **AGREE** |
| O1 R5 | −0.000786103199105844553814426531720552667356782768 | …6740166 (5.8e-61) | − | **AGREE** |
| O2 R4 | −0.00156451543861247101284463728995918182412852963 | …96228242 (1.05e-60) | − | **AGREE** |
| O2 R5 | −0.000596034322637912719136676783441298348027258702 | …70160609 (6.0e-61) | − | **AGREE** |
| O3 R4 | −0.00157748742119318453653509477632426298977949940 | …93960005 (1.7e-60) | − | **AGREE** (also my stage 1) |
| O3 R5 | −0.000576860076001386977530470583141830922381440359 | …35809291 (7.4e-61) | − | **AGREE** |

### C.5 (cb). Clause 1 (every i = J₀) and clause 4

* **Zero multiplicity.** Mine comes from the **exact charpoly** (sympy), with the radicals of the pentagonal pyramid handled exactly; there the charpoly turns out rational. I also print the kernel dimension, to show the reported number is an *algebraic* multiplicity.
* **Values.** Computed at 50 and 70 digits. They agree to ≤ 2e-48, are invariant under a random change of basis inside the eigenspace (≤ 2e-54), and each s_g² is a root of my exact slow polynomial.
* **Comparison with cb.** Mechanical (`a_cmp_C5.py`): s_g matched to 1e-20, values to relative 1e-18. The worst agreement is 4e-20, i.e. all 20 digits cb prints.

| case | cb zero mult. | mine: algebraic / dim ker Ŝ | my exact slow polynomial | cb groups vs mine | verdict |
|---|---|---|---|---|---|
| pentagonal pyramid in R4 | 6 (k 4, r_u 2, sig (5,0,4)) | **6 / 4** | (846777242025ν²+1018930808896ν+3933798400)(21169431050625ν²+83366483200ν+629407744) | 8 simple groups, including the complex quartet ±0.0417352859837625943 ± 0.0609168609168609169i, where c_g = sgn(Re s)·0.000124233245523991326 − sgn(Im s)·0.000234845365305689488i; the values of ∓1.09518548832466951i and ∓0.0622348690956879072i equal mine | **AGREE** |
| coherent v3 in R4 | 4 | **4 / 3** | (1521ν+12544)(9801ν+19600)(20449ν+784)(184041ν+3136)(1656369ν+802816) | 10 simple groups, equal (and equal to my stage 1) | **AGREE** |
| coherent v3 in R5 | 4 | **4 / 3** | (169ν+441)(1936ν+1225)(20449ν+3136)(81796ν+441)(327184ν+3969) | 10 simple groups, equal; e.g. s_g = −1.615384615384615385i → +0.0009262404663135482546i | **AGREE** |
| weight 7/2 in R2 | 6 (k 5, r_u 1) | **6 / 5** | (20449ν+12544)(20449ν+28224)(20449ν+112896)(20449ν+254016)(20449ν+313600) | 10 simple groups, equal (and equal to my stage 1) | **AGREE** |
| octahedron in R4 | 8 (k 4, r_u 4, sig (6,0,3)) | **8 / 4** | **(552123ν − 250880)³** | 2 groups s_g = ∓0.67408567952638802042, each of multiplicity 3, semisimple. The eigenvalues of C_ab are ±0.00020573325000038580966 (threefold). | **AGREE** |

**The semisimple group (octahedron).** I tested semisimplicity independently: dim ker(Ŝ−s) = dim ker(Ŝ−s)² = number of left eigenvectors = 3. The eigenvalues of C_ab are unchanged under a random complex change of basis of the right eigenspace, with the left basis renormalised so that ℓᵀr = I. So **the normalisation of the eigenvectors does not influence the reported values**; cb's biorthonormalisation (ℓ ← ℓ (ℓᵀr)^{-ᵀ}) is the same as mine.

### Block D (cert; ca's Krawczyk tests counted too) and B.4 (cert)

My certifier (Krawczyk on the inner Section-5 box, then a second Krawczyk step) was run on all six boxes.
* **Signature:** certified by a Descartes count on the interval charpoly. cert used Gershgorin discs after congruence, an independent decomposition.
* **Zero multiplicity:** c₆ of the interval charpoly excludes 0.
* **Comparison with cert:** every enclosure intersects (`a_cmp_D.py`). cert's quartic h equals mine coefficient by coefficient, exactly (263 monomials per slot).

| box | signature | k | r_u | zero mult. | 924 r̂₆ (common digits) | slow roots ν (signs) | words | verdict |
|---|---|---|---|---|---|---|---|---|
| O1 R4 | [4,0,5] | 4 | 2 | 6 | 215.77579250166841349523233893587826824768 | −0.91084042759146767, −0.055255188961110329, −0.0016625682550668705, +0.058530828363685964 (−,−,−,+) | hyperbolic, one real pair | **AGREE** |
| O1 R5 | [4,0,5] | 4 | 2 | 6 | same as R4 | −0.28819560404261282, −0.017483087132226315, −0.00052604698695475200, +0.018519519911947512 | hyperbolic, one real pair | **AGREE** |
| O2 R4 | [6,0,3] | 4 | 2 | 6 | 238.01652892561983471074380165289256198347 | −1.0721089978157267, −0.098742293980773456, −0.0011815740874455854, +0.00094514036983932503 | hyperbolic, one real pair | **AGREE** |
| O2 R5 | [6,0,3] | 4 | 2 | 6 | same | −0.33922198759013229, −0.031242678954854101, −0.00037385742610582975, +0.00029904832014447394 | hyperbolic, one real pair | **AGREE** |
| O3 R4 | [7,0,2] | 4 | 2 | 6 | 238.35685474309527034553212338375548850199 | all four negative (as in my stage 1) | elliptic, simple | **AGREE** |
| O3 R5 | [7,0,2] | 4 | 2 | 6 | same | all four negative | elliptic, simple | **AGREE** |

* **B.4 ratios (clause 3).** At all three points, every one of the four pairs has ξ_{R₅}/ξ_{R₄} enclosed around 0.5625 (9/16). My widths are 3e-103 to 6e-98; cert's printed widths are ≈1e-70, and cert's enclosures contain 9/16. **AGREE** (cert, all 12 pairs).
* **Clause 2.** The enclosures are taken over a certified box containing the certified point, and each has width ≤ 1e-50 (cert's printed widths are ≤ 1e-67).
* **Gram determinants.** O1: 56.995197888787505992…; O2: 44.593740531795367…; O3: 44.464498586148148697…. Equal in both rooms.
* **Krawczyk boxes.**
  * cert ran Krawczyk on a box that **contains** the Section-3 box: outward-rounded, with λ = 4h(x_c)/|x_c|²_G from its own h, which equals mine exactly, and c = 0. It then checked that K(Z) lies **inside the exact decimal box** with margin 1e-150. So the certified point lies in the given box, and every B.4 enclosure is over a refined box ⊂ K(Z).
  * ca ran Krawczyk on an inward-rounded box ⊂ the given box, and proved c = 0 by diagonal dominance.
  * Both procedures are sound. **AGREE** for ca's six certifications too.

---

## 2. Gaps filled by my own route (step 2), and why these samples

Stage 1 had not computed the items below; this stage computed all of them, not a sample.

| what | why it was chosen |
|---|---|
| Block D/B.4 at **O1 and O2** (both slots) | Their words rest on a **single positive root's sign**: ν₄ = +0.000945 (O2 R4) and +0.000299 (O2 R5). Certified by me with widths ~1e-101. |
| C.5 at the **octahedron** | Semisimple multiplicity-3 group; normalisation-sensitive. |
| C.5 at the **pentagonal pyramid** | Complex quartet; the point has radicals; zero multiplicity 6 vs kernel 4. |
| C.5 at **coherent v3 in R5** | — |
| **All 28 C.4 point values** and **5 box signs** | Signs are not in doubt (R̃₀ ≤ 0 structurally), but "exact" is a claim worth testing. |
| C.1 at **both solvers' y** | Each solver chose its own y. |

The R₂ degenerate point (C.3) and v₁ were already exact in stage 1.

---

## 3. Readings where the worklist is underdetermined

1. **The R₂ fibre group (affects C.3 and C.5 in R₂).** The worklist says ψ ↦ ψ̄C "with the phase generates a group Sp(1)". Read literally, the group generated is U(1) ∪ U(1)J.
   * ca, cb and I all took **Sp(1) = {a + bJ}**, acting by w ↦ aw + bΘw.
   * The other reading changes the values. Exact demonstration (`a_reading.log`):

     | reading at the line {5/2, −3/2} point | k | r_u | signature | dim K′ |
     |---|---|---|---|---|
     | Sp(1) (as stated) | 6 | 2 | [0, 3, 6] | 3 |
     | literal: U(1) and J only | 4 | 2 | [0, 5, 6] | 5 |

   * h(Θx) = h(x) holds as an exact polynomial identity, so Θu and iΘu lie in ker M̂ either way.
   * The worklist names the group explicitly, so Sp(1) is the better-supported reading. If the literal one were intended, C.3's K′, |Γ_u|, the irreducibility answer and γ_u would all change, and γ_u might no longer be a multiple of the identity. C.5 at weight 7/2 in R₂ would also be affected. By the same orbit count k would become 3, and the Θ directions would move into N_u. *This C.5 consequence is an expectation, not computed.*
2. **C.1, "every level above the lowest at which the slot occurs, up to 20".**
   * ca read it as the occurrence levels only; cb and I read it as all n < L ≤ 20.
   * The values don't change: the extra levels are exactly 0, as cb and I both computed.
3. **C.1, the choice of y.** Free; each room's y is unambiguous (ca's is checked above). The nonzero-level sets are the same for all three choices.
4. **0(e), "compared with Θ".** C is fixed only up to a phase. The map is Θ times that phase:
   * ca chose the phase so the map is Θ;
   * cb's C gives −Θ;
   * mine gives e^{iα}Θ.

   The same fact in each case; no value depends on it.
5. **C.4 at O1–O3.** ca took the Krawczyk-certified zero, proved c = 0, and enclosed R̃₀ over the whole box. Evaluating at the decimal centre instead would give the same sign.
6. **C.5, what a "slow group" is.**
   * cb and I treat each distinct nonzero eigenvalue as its own group, and list both ±s_g. Grouping a ± pair, or the quartet, as one group would reorganise the table without changing any value.
   * ℓᵀ (not ℓ^H) is used, as written. Using ℓ^H would be wrong for non-real s_g.
   * cb symmetrises D_u; I do not. The values are unchanged.
7. **Block D / B.4, the box for the enclosures (clause 2).** cert, ca and I use a refined certified box. Over the raw Section-5 box, my stage-1 run shows one R₅ slow-root width of 2.2e-50, slightly over the bound. So only the refined-box reading meets clause 2's width.
8. **The meaning of M̂ in the chart.** The operator G⁻¹∂²h − 4h and the form ∂²h − 4hG on the G-orthogonal N_u have the same signature (cert's reading 1). This is not a value difference.
9. **Clause 3, +ξ for ν < 0.** +ξ = i√|ν| (cert and me). The ratio is the same for ±ξ.

---

## 4. The solvers' checks (step 4)

### ca

* **DEFECT (claim about a check), demonstrated by the code.**
  * ca's RETURN calls `s4_boxes.py` "an independent 130-digit real-chart code [that] agrees with all 28 exact values to 2.5e-132".
  * It is not independent:
    * `s4_boxes.py` `R0_interval` (lines 212–242) evaluates the same structural formula ‖Π_K N‖² = g_K‖[[y⊗Θy]₆⊗y]_K‖² as `s3_points.py` `R0_exact` (lines 34–48).
    * It uses the same pickled constants `consts[s]["gK"]` and the same CG table (`cg_rational_square`).
  * So that PASS compares one formula with itself in two arithmetics. It cannot detect a wrong g_K or a wrong formula.
  * The only independent route ca has for C.4 and C.1 is the **double-precision** cubature: absolute tolerance 1e-12, which is about 1e-7 relative for the R̃₀ values. Under the exactness rule, ca's "exact" values were therefore *not* confirmed by an independent high-precision route in ca's own room.
  * **Consequence: none for the values.** My independent routes confirm them, 18 identically and 10 to ≤ 1.8e-63.
* **Same weakness for C.1.** The surd values were checked independently only at about 2e-16.
* **Planted defects that do exercise the reported values:** #7 (g_K × (1+1e-6) → cubature comparison fires), #8 (criticality), #9 and #10 (Krawczyk).
* **Checks with no plant:** the 130-digit comparison, the c = 0 diagonal-dominance check, and the "sign-carrying decimal" check. The last one was introduced after a real sign-bit bug, so it has fired once.
* **Box construction.** ca's inward box uses `mid` of a 130-digit enclosure of the decimal centre. It could poke out of the given box by about 1e-130; this is immaterial, since the zero sits about 1e-89 from the centre. My own stage-1 inner box has the same feature.

### cb

* **C.5 values.**
  * They rest on a 50-vs-70-digit agreement of the **same code**. The planted defect (the μ₀² sign flipped in the 70-digit run only) is built so that this comparison fires.
  * A formula error present in both runs (e.g. the wrong sign of that term in both) would pass every cb check. So the plant does not exercise the property the values rest on.
  * My independent implementation is the guard here, and it agrees.
* **Zero multiplicities (C.3, C.5).** Counted with a tolerance of 10^{−dps/5}, and the check is "the same count at two precisions". That is a numerical count, not an algebraic multiplicity. All of them equal my exact charpoly multiplicities, and the gaps cb reports (e.g. 5e-41 vs 0.109) are consistent with 2×2 Jordan blocks.
* **|Γ_u| = 24.**
  * Found by a numerical search; elements verified at 50 digits.
  * The check "|Γ_u| = number of rotations found" compares a count of deduplicated maps with the number of rotations. It can fail, but only weakly.
  * Completeness rests on the maximality of O among finite subgroups of SO(3) (from knowledge; correct). See §5.
* **Exact γ_u and R̃₀.** These have a real independent check (s04b, exact route A vs identification), and its plant (a wrong λ₀) fires.
* **Values identified from numerics.** The C.3 slow polynomials were identified at 80 and 130 digits with q ≤ 1e30. That is two precisions, as required, and I confirm them exactly.

### cert

* **Planted defects that exercise the code path behind the values:** Krawczyk (#10, #11, the actual `SliceSystem.krawczyk`), `certify_real_roots` (#14), the quartic invariance and the cubature (#8, #9).
* **Plants that test a re-implementation instead of the code used for the values:**
  * The Gershgorin signature plant (#15) and the Gram-determinant plant (#13) re-implement the predicate inline, rather than calling `analyze()`.
  * The double-root plant fires through `polyroots` non-convergence, not through the sign-change test (cert says so itself).
* **No planted defect:** zero multiplicity (a₆ excludes 0), r_u, "K(Z) inside the exact box", and the B.4 sign and ratio checks.
* **A loose PASS line.**
  * `sD_certify.log` prints "refined box is certified by its own Krawczyk test" with steps [True, True, **False**]. The last Krawczyk test, on the 1e-219 box, failed at 220 digits.
  * `refine()` then returns the previous box, which passed (step 2), so the claim in RETURN ("that box passed its own Krawczyk test, width ≈ 2e-115") is true.
  * But the predicate `okr and any(ok_steps[1:])` would print PASS whichever box was returned. Not a defect in the result.
* **Outward rounding of the printed strings.** `out_round` takes the floor or ceiling of a round-to-nearest 220-digit product. That is not directed rounding: it could misround only if an endpoint lay within about 1e-150 (relative) of a 70-digit decimal. A theoretical gap, not a demonstrated error. The interval computations themselves are outward (`mpmath.iv`).

### Comparisons of a quantity with itself

I found none in the three rooms, apart from ca's "independent" 130-digit check above, which is formula-with-itself rather than value-with-itself.

### My own checks (self-audit)

* `a_C5.py`'s 50/70 agreement has cb's weakness. The independent guards are the exact slow-polynomial roots, the exact zero multiplicities and the comparison with cb's independently built code.
* `a_reading.py` has no planted run.

---

## 5. Claims I tried to break and could not

* **Criticality is exact.** All 28 census and R₂ points satisfy ∇h = 4h u exactly in sympy, with residual 0. A 1e-20 perturbation gives a residual of 1e-19 and FAILs. The six O-points are certified by three independent Krawczyk codes (ca, cert, mine).
* **Signatures by independent decompositions:**
  * Block D: Descartes on the interval charpoly (mine) vs Gershgorin after congruence (cert). Same at all six boxes.
  * C.3 and C.5: exact charpoly/Descartes (stage 1 and `a_reading`) vs eigsy at 50 digits (`a_C5`) vs cb's numerics.
* **Zero multiplicity is algebraic, not a kernel dimension.** In C.5 the algebraic multiplicities (exact charpoly) are 6, 4, 4, 6, 8, while dim ker Ŝ is 4, 3, 3, 5, 4. cb reports the algebraic numbers, and so does cert in D (c₆ ≠ 0 with structural zeros below).
* **"Exact" values are exact.**
  * cb's C.1 rationals are identical to my route-E rationals.
  * ca's C.4 rationals are identical to my route-E values at the 18 points where route E applies, and agree to ≤ 1.8e-63 elsewhere.
  * cb's γ_u and slow polynomials are identical to my stage-1 exact values.
  * cert's quartic is identical to mine, coefficient by coefficient.
* **The η of cb and cert are not just "an" intertwiner.** They reproduce **my exact isotypic projector**, exactly. ca's η is valid in ca's own convention.
* **The B.4 words rest on certified signs.** The positive roots at O1 and O2 are certified, and in all six boxes the four slow roots are certified real and simple, in two independent codes.
* **The C.5 values do not depend on the eigenvector normalisation.** I tested this with a random basis change inside every eigenspace (≤ 2e-54).
* **|Γ_u| = 24 at the R₂ line point.**
  * I could not find more elements.
  * cb's completeness argument is valid: k = 6 makes the stabiliser finite, the found set is closed and is O, and O is maximal among finite subgroups of SO(3), since 24 ∤ 60.
  * Neither room certifies the 24 elements exactly.

---

## 6. What I could not check, and why

* **|Γ_u| = 24, exactly.** The stabiliser elements are numerical in both stage 1 and cb (50-digit verification). I did not build exact quaternions for them. The verdict AGREE rests on two independent numerical searches plus cb's maximality argument.
* **ca's statements about C as a matrix** (C ≈ [[0, 1], [−1, 0]] in ca's η basis). Checked numerically at 28 digits only, which matches ca's 30-digit printout. ca calls it "not an exactness claim".
* **Items outside my scope** were not graded: Blocks A, B.1–B.3, B.5, E. Nor were values the solvers list only "for reference" (for example ca's internal c_J, or cb's slow roots printed to 20 digits beyond the polynomial).
* **Strictly rigorous decimal rounding** of cert's 70-digit strings: see §4. Not demonstrated either way.

---

## 7. Process notes (reported, not concealed)

* **My first command broke the tool rule.** It was `cd` into this room plus `ls -la` and `cat` of `BRIEF.md`, `worklist.md`, `clauses.md` and `inputs.json`. It read only in-room files. The harness saved that command's output to a tool-results file outside the room, which I did not open; only its inline preview, showing in-room content, was displayed to me.
* **Two commands were refused by the permission layer and did not run:**
  * a compound command joined with `;`;
  * a command containing a newline followed by `#`.

  After these, every command began with `./py` and was a single command.
* **Concurrent processes.** At one point I had three `./py` processes running at once in the background: `a_C5.py octahedron`, `a_C4.py perturb` and `a_C1.py`. They were not multiprocessing within a script, but they were concurrent load on a shared machine; afterwards I ran everything sequentially.
* **Harness output files.** The harness wrote background-task stdout to files outside the room. I never read them; I read the in-room `.log` files instead.
* **`audit.json`.** Written by `audit/a_assemble.py`. I then added two keys, "checks (step 4)" and "readings (step 3)", with a one-off `./py -c` command; they carry the text of §3 and §4.
* **Superseded outputs.** `audit/a_C5_octahedron.json` and `audit/a_C5_pentagonal.json` come from earlier partial runs. `audit/a_C5.json` and `audit/a_C5.log` are the complete run.

---

## Consulted-material manifest

**Read in this room (top level):**
* `BRIEF.md`, `worklist.md`, `clauses.md`, `inputs.json`
* `py`
* `.room.sb` (read with the file reader; it is also loaded by `sandbox-exec` on every `./py` run)

**Read in `own/` (read only; never written):**
* `own/OWN.md`, `own/BRIEF.md`, `own/own.json`
* `own/core.py`, `own/reps.py`, `own/s3poly.py`, `own/hpoly.py`, `own/cgroute.py`, `own/exact_pt.py`, `own/numrep.py`, `own/chk.py`, `own/krawczyk.py`, `own/ivutil.py` (these were also copied to `audit/`)
* `own/s5_D_B4.py` (template for `audit/a_D_all.py`), `own/s3_C4.py`, `own/s4_C5.py`, `own/s1_C1C2.py`
* `own/s5_D_B4.log`, `own/run_all.log`, `own/s4_C5.log` (first 40 lines)

**Read in `solver_ca/`:**
* `RETURN.md`, `results.json`
* `s4_boxes.py`, `s3_points.py`, `s2_constants.py`
* `planted.log`
* `s0_group.log`, `s1_slots.log`, `s2_constants.log`, `s3_points.log`, `s4_boxes.log` (PASS/FAIL lines)

**Read in `solver_cb/`:**
* `RETURN.md`, `results.json`
* `s05_c5.py`, `s04_c3.py` (lines 60–199 and a search of its check lines), `lib_check.py`
* `s99_planted.log`, `s04_c3.log`
* `s00_group.log`, `s01_cg_theta.log`, `s02_eta.log`, `s02b_eta_exact.log`, `s03_levels.log`, `s04b_c3_exact.log`, `s05_c5.log` (PASS/FAIL lines)

**Read in `solver_cert/`:**
* `RETURN.md`, `results.json`
* `lib_cert.py`, `lib_analyze.py`, `sD_certify.py`, `sB4.py`, `planted_defects.py`
* `planted_defects.log`
* `quartic_R4.json`, `quartic_R5.json`
* `s0a_group.log`, `s0b_levels.log`, `s0c_intertwiner.log`, `s0d_cg_theta.log`, `s1_quartic.log`, `sD_certify.log`, `sB4.log` (PASS/FAIL lines)

**Written by me:** `audit/` (scripts, logs, json), `AUDIT.md` and `audit.json`.

**Loaded without my asking:** the Python standard library, numpy, sympy and mpmath, through `./py`. scipy was not imported.

**Outside the room:**
* Nothing was read: no network, no documents.
* The harness tool-results file from my first command, and the harness background-task output files, were not opened.

**From knowledge (methods, not inputs):**
* Krawczyk's test, Faddeev–LeVerrier, Descartes' rule and Gershgorin's theorem
* the maximality of O among finite subgroups of SO(3) (used to assess cb's argument)
* the name "binary icosahedral group" (recognised, not used)
