# M5.37: The author's 10 November reading: the N4 precondition, the rigid radius ratio, and a pre-registered πN test

> Opened 2026-11-10 from the author's post of 2026-11-10 on discussion #186 ([discussioncomment-19507849](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19507849)), the reply to the [M5.36](m5_36_task_details.md) results post. The post carries no release and asks nothing of this column: it closes the three open items of M5.36, states a structural precondition for the proposed row N4, reports a native nucleon on a frozen card of the author's ("Card C"), states a radius identity for any rigid current, and pre-registers a πN test with the expectation of a miss. Its last line offers the frozen card specification for an independent rung. The task number is the author's own title for the post. The decode of the post is in [`m5_32_convo.md`](m5_32_convo.md).

## TASK PLANNING (2026-11-10; planned and run in one desk pass on the user's go)

**Scope.** A desk ingestion, no release and no run of the author's code. (1) The closures of the M5.36 open items, recorded (§ 1). (2) The N4 precondition, written against the N4 reading of [M5.36](m5_36_task_details.md) § 4 and § 13 (§ 2). (3) The arithmetic of the post checked in a script of this repo, from the printed numbers and literature constants only (§ 3). (4) The asks for the reply (§ 4). (5) The offer of an independent run on the frozen card, recorded with what it would take, not staged (§ 5). Out of scope: any run on Card C (no specification is held here); any S-run; any icon or pointer in `MODELS.md`; the structure of the criteria table.

**Definition of done.** § 1 to § 5 written; [`m5_37_post_checks.py`](../scripts/m5_37_post_checks.py) run with its JSON, each check stating the printed number that would make it fail; a fresh-agent audit on a different model with its own scripts at 0 REFUTED, or the refutations folded in; the convo row; the reply draft in the terminal; `check-docs --changed` and `check_roadmaps.py` clean.

**Gating.** The user's go. Nothing downloaded, nothing executed from outside the repo.

**Parallel shape.** One script of about two seconds on one core, a documented choice: there is no compute to split. One audit agent in the background.

**Consumers.** The N4 precondition is read by any later plan that stages a moment or rotor read on a baryon state of this column: the instrument check it adds is the count of unbroken isospin generators of the exterior tensor, before any inertia is quoted. The arithmetic checks are read by the reply: an ask is sent only on a number an independent agent also failed to derive.

**Blindspot pass.** (i) The post is a claim sheet with no script behind it on this side: every number is a draft under [`AI_HYGIENE.md`](../../../../../AI_HYGIENE.md), and a check here is an arithmetic check of printed numbers, never a reproduction of the card. (ii) A printed number that does not follow from the other printed numbers can be a convention of the card (its own f_π, its own radius), so the finding is "not derivable from the post", an ask, never "wrong". (iii) The 3/5 identity has a premise (one density behind both form factors); the audit is asked to attack the premise against the standard rigid-rotor structure, not only the algebra. (iv) The readiness of the census state for N4 is a reading of stored records; no field was re-read. Unknowns by quadrant: known unknown, where 0.66 μ_N and the 1 to 2.5 percent come from (author-gated, § 4); unknown known, the user's call on the offer (§ 5); machine-checkable, the seven arithmetic items (§ 3).

### 1. What the post closes

| Open item of M5.36 | The author's answer |
| --- | --- |
| Whether `V78` should run on OpenWave | not needed |
| The three README notes (the title, the missing `V84` row, the `V83` line) | accepted: the title reads V1 to V84, `V84` gets a row, the `V83` line gives Σm_ν = 58.69 meV and m_ββ = 1.45 to 3.65 meV; into the next bundle |
| The grades of P1, P2 and N4 as readings | "grey is fine" |

No ask of M5.36 stays open. The P1 reading at adoption (M5.36, action 1) is moot until the board adopts the row.

### 2. The N4 precondition

The author's statement, in the post's words and symbols: a rotor or magnetic read needs a finite isospin inertia; per unit volume the inertia is `c · tr([T, N]ᵀ [T, N])`, finite in the whole space only if the generator `T` commutes with the vacuum tensor `N`; the count of unbroken isospin generators is 0 for a biaxial record exterior, 1 for a uniaxial one, and 3 only when the record vanishes outside the core. The author's own card confines the record with the minimum `s = c(q) · diag(1, δ, 0)`, `c = (1 − q₀) / 2`, for which the exterior record is exactly zero and the inertia moves by less than 2 × 10⁻¹⁴ between the 17³ and 33³ boxes (the author's numbers, not run here).

The reading on this column, from stored records and with nothing computed: the census proton analog ([`MODELS.md`](../../../../../MODELS.md), the baryon row; the M5.22 series) is a vortex column with an equatorial ring on an ordered vacuum, so its order tensor does not vanish outside the core and the author's count gives at most one unbroken generator. The N4 reading of M5.36 § 4 ("a magnetic read on the census state would need the EM reading of the record field applied to a baryon state; not staged") gains a step in front of it: a state whose exterior tensor commutes with all three generators. No such state exists on this column. N4 stays 🚧 with that precondition named; the count for the census vacuum (0 or 1) was not re-read from the fields. [M5.38](m5_38_task_details.md) § 2 carries the read of that count on the stored fields.

### 3. The arithmetic of the post ([`m5_37_post_checks.py`](../scripts/m5_37_post_checks.py), [`m5_37_post_checks.json`](../data/m5_37_post_checks.json))

The rigid ratio in equations. With one density ρ(r) behind both isovector form factors,

```text
G_E(q) ∝ ∫ ρ(r) j0(qr) d³r        = 1 − q² ⟨r²⟩ / 6  + ...
G_M(q) ∝ ∫ ρ(r) 3 j1(qr)/(qr) d³r = 1 − q² ⟨r²⟩ / 10 + ...
r_M² / r_E² = (1/10) / (1/6) = 3/5      for any ρ
```

| # | Item of the post | Printed | On OpenWave, from the printed numbers | Read |
| --- | --- | --- | --- | --- |
| 1 | rigid r_M,V² / r_E,V² for any profile | 3/5 | 3/5 (series of j0 and 3 j1/x); 0.6000 to 0.6001 on a Gaussian, an exponential and a thin shell | ✅ reproduced, on the premise of one density |
| 2 | the rotor moment M_p / Δ at 293 MeV; μ_p − μ_n | 3.20; 4.706 μ_N | 3.2023; 4.70589 | ✅ reproduced |
| 3 | the moment the rotor leaves open | +0.66 μ_N | 1.504 (μ_p − μ_n convention), 0.752 (half) | ⚠️ not derivable from the post |
| 4 | the same from the two radii: a rigid piece at (3/5) r_E,V² plus a piece at 1.5 fm², data ratio 0.887, r_E,V² = 0.823 fm² | +0.66 μ_N | 0.981 on the 3.20 rotor; 1.104 with the total at 4.706 | ⚠️ not derivable from the post |
| 4b | the data ratio behind row 4 (added on the audit) | 0.887 | 0.891 from magnetic radii 0.851 and 0.864 fm (the printed ratio's input set); 0.803 from 0.777 and 0.862 fm, where row 4 gives 0.636; the solve returns exactly 0.66 at a ratio of 0.809 | ⚠️ 0.66 and 0.887 belong to different input sets |
| 5 | the ρ-pole radius 6 / m_ρ²; the rms of 1.5 fm² | 0.39 fm²; about 1.2 fm | 0.3887; 1.225 | ✅ reproduced |
| 6 | Goldberger-Treiman at g_A 0.904, g_πNN 9.55 | 1 to 2.5 percent | 3.5 to 4.4 percent at physical M_N and f_π 92.07 to 93.0 MeV; 1 to 2.5 percent needs f_π 89.8 to 91.2 MeV | ⚠️ not derivable with physical inputs |
| 7 | the Born scaling (9.55 / 13.1)² | 0.53 | 0.5315 | ✅ reproduced |

Rows 3, 4, 4b and 6 are not errors of the post on this evidence: the card has its own radius and may have its own f_π, neither printed. They are the asks of § 4. Row 4 also gives the radius a second piece would need if it carried exactly the open moment of row 3 (1.504 μ_N): 1.23 fm², between the ρ pole and the post's 1.5 fm².

### 4. Asks to the author (held: no reply was posted at the close, the user's decision of 2026-11-10; none blocks anything)

Both asks went to the author on 2026-11-11 inside the [M5.38](m5_38_task_details.md) results post ([discussioncomment-19547011](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19547011)).

1. The +0.66 μ_N of item 4 of the post: at the printed data ratio 0.887 the two-radius solve gives 0.98 to 1.10 μ_N, and it gives 0.66 at a ratio of 0.81 (a proton magnetic radius near 0.78 fm in place of 0.851 fm); which ratio and which r_E,V² the 0.66 is read at.
2. The 1 to 2.5 percent of Goldberger-Treiman: the f_π and the nucleon mass used (the card's or the physical ones).

### 5. The offer: an independent run of the pre-registered test (recorded, not staged)

The test as declared (10 November 2026, before any amplitude): on frozen Card C, with 293 MeV setting only the bare gap, one πN amplitude gives both a Breit-Wigner Δ width of 99 to 135 MeV and m_π³ a₃₃ in 0.18 to 0.24 (Roy-Steiner 0.2115(28)); the method is the Born structure of Hayashi, Saito and Uehara with distorted waves; the author's pre-run expectation is a Born-level miss, both numbers scaling with (9.55 / 13.1)² ≈ 0.53.

What a rung on OpenWave would be: the author's instrument run unchanged from a hashed copy, in the shape of R30 of [M5.32](m5_32_task_details.md), with the two windows above as the pre-registered gates and the author's expectation recorded beside them. It needs the frozen card specification (not held), a script of the amplitude from the author or a reimplementation from the cited structure, and the user's decision: the standing rule of the series is no S-run until a relaxed host binds the Dirac states, and Card C is a baryon card of the author's, outside that rule's wording and outside this column's own states. Nothing is staged.

### 6. The records

The convo row in [`m5_32_convo.md`](m5_32_convo.md); the roadmap row and change-log entry in [`m5_roadmap.md`](../m5_roadmap.md); a pointer from [M5.36](m5_36_task_details.md) § 13 to § 2 of this document. The four papers the post cites (Belushkin, Hammer and Meißner, [doi:10.1103/PhysRevC.75.035202](https://doi.org/10.1103/PhysRevC.75.035202); Bég and Zepeda, [doi:10.1103/PhysRevD.6.2912](https://doi.org/10.1103/PhysRevD.6.2912); Hoferichter et al., [doi:10.1016/j.physrep.2016.02.002](https://doi.org/10.1016/j.physrep.2016.02.002); Hayashi, Saito and Uehara, [doi:10.1103/PhysRevD.43.1520](https://doi.org/10.1103/PhysRevD.43.1520)) are cited here by DOI and were not read for this task; none is in the local corpus.

## EXECUTION RECORD (2026-11-10; the go at 08:52 EST)

### 7. Environment

Python 3.12.14, NumPy 2.5.3, SymPy 1.14.0 on the pinned stack, one thread. `m5_37_post_checks.py --write` runs in about two seconds and regenerates the JSON. Nothing was downloaded and no code of the author's was run.

### 8. The deviations

1. The go carried no reset, ping or fan part. The task is a short attended desk pass with seconds of compute, so the resume ping and the watchdog were not armed and the fan line printed as not needed, each stated in the pre-flight box; the go was not asked a second time.
2. Planning and execution ran in one pass on that go (no staging hand-over): the plan above was written first and nothing in it changed during the run.

### 9. The fresh-agent audit (Sonnet, 2026-11-10, with its own scripts, before anything left the tree; the agent saw the post and, after its own numbers, the script and its JSON)

| # | Claim | Verdict | The agent's number |
| --- | --- | --- | --- |
| 1 | r_M² / r_E² = 3/5 for one density behind both form factors | QUALIFIED: the algebra CONFIRMED, the premise questioned | 0.6000 on three profiles |
| 2 | M_p / 293 = 3.20; μ_p − μ_n = 4.706 | CONFIRMED | 3.2023; 4.7059 |
| 3 | 0.66 is not the rotor's open moment in either convention | CONFIRMED | 1.5036; 0.7518 |
| 4 | 0.66 is not derivable from the post | REFUTED as worded, folded in | 0.981 and 1.104 at the printed 0.887; 0.633 to 0.642 at a ratio near 0.80; 0.66 at 0.809 |
| 5 | 6 / m_ρ² = 0.39 fm²; √1.5 = 1.22 fm | CONFIRMED | 0.3887; 1.2247 |
| 6 | Goldberger-Treiman at physical inputs is outside 1 to 2.5 percent | CONFIRMED | 3.5 to 4.5 percent over M_p or the mean mass, f_π 92.1 to 93; 3.7 to 4.7 on the inverse definition |
| 7 | (9.55 / 13.1)² = 0.53 | CONFIRMED | 0.5315 |

Folded into this record: row 4b of § 3, the reworded ask 1 of § 4, and check 4b of the script (the printed 0.887 is the ratio of the 0.851 fm proton magnetic radius; the 0.66 is the solve at a ratio near 0.81). On claim 1 the agent objected, from memory and without a source, that a rigid hedgehog's isovector magnetic density is not its inertia density; the objection is unverified, and so is the opposite reading (the isovector rotor current built on the inertia density, which is what makes μ_V proportional to the inertia and gives the `V84` identity). The record therefore keeps the identity as reproduced on the premise of one density, the premise being the author's "common-inertia" wording, and claims nothing about which rigid models satisfy it. Not verified by the agent or by this task: the Roy-Steiner value 0.2115(28), the factor conventions of μ_V = M_p / Δ. The agent's notes on the script: check 1 passes by construction for a common density (it tests the algebra, not the premise, as stated above), and check 3 accepts either convention (it fails in both).

## TASK REVIEW (2026-11-10; presented in the terminal at 08:57 EST and approved by the user the same morning, with the decision to record the close only: no reply posted, the asks of § 4 and the offer of § 5 held)

**Task Duration:** 00:05 (from 08:52 EST to 08:57 EST)
**Usage Cap Triggered:** NO

**Results.**

| Item | Label | Result |
| --- | --- | --- |
| The closures of M5.36 (§ 1) | ✅ recorded | `V78` not wanted, the README notes accepted, the grey grades accepted |
| The N4 precondition (§ 2) | ✅ written, a reading | the census state has at most one unbroken generator by the author's count; N4 stays 🚧 |
| The rigid ratio (§ 3, row 1) | ✅ measured | 3/5 by series, 0.6000 to 0.6001 on three profiles, on the premise of one density |
| Rows 2, 5, 7 | ✅ measured | 3.2023 and 4.70589 μ_N, 0.3887 fm², 0.5315 |
| Rows 3, 4, 4b | ⚠️ not at the printed inputs | 0.98 μ_N at the ratio 0.887; 0.66 at 0.809 |
| Row 6 | ⚠️ not at physical inputs | 3.5 to 4.4 percent; 1 to 2.5 needs f_π 89.8 to 91.2 MeV |
| The audit (§ 9) | ✅ folded in | 5 CONFIRMED, 1 QUALIFIED, 1 REFUTED as worded |

**Issues / blockers.** The first wording of row 4 ("not derivable from the post") was refuted by the audit and replaced by row 4b (artifact read as result: one input set tried before the label was written). The premise of row 1 against the standard rigid-rotor structure is unverified in both directions (§ 9).

**Deviations from plan.** The two of § 8.

**Action needed.** None open. Held for a later decision of the user's: the two asks of § 4, and the offer of § 5.

**Scope.** An arithmetic check of printed numbers and a reading of stored records: it reproduces nothing of Card C and computes nothing on this column.

**Findings.** The author's reply closes M5.36 and puts a structural step in front of N4 on this column: no state here has an exterior that commutes with all three isospin generators. The rigid 3/5 identity and four other printed numbers reproduce; the +0.66 μ_N and the Goldberger-Treiman figure do not follow from the printed inputs.

**Research docs created / updated.** [`m5_37_task_details.md`](m5_37_task_details.md); [`m5_37_post_checks.py`](../scripts/m5_37_post_checks.py), [`m5_37_post_checks.json`](../data/m5_37_post_checks.json); [`m5_32_convo.md`](m5_32_convo.md); [`m5_36_task_details.md`](m5_36_task_details.md) § 13; [`m5_roadmap.md`](../m5_roadmap.md).
