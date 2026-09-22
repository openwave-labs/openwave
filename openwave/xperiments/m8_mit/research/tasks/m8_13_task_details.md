# M8.13: AUDIT OF THE UNIQUENESS OF THE TOP-MULTIPOLE MAXIMUM AT SPIN 3, the clause of M8.12's G2 that its run left unresolved

Pre-registration, for review. It governs nothing until the maintainer's go.

## TASK PLANNING

### The question

M8.12 reproduced G2 in its value and attainment: the maximum of `r̂₆` on the unit sphere of `V₃` is `463/924`, argued in one room and certified independently by the audit, and the hexagon orbit attains it. It left G2's uniqueness clause unresolved, since neither room nor the audit settled whether any other orbit attains the maximum ([#582](https://github.com/openwave-labs/openwave/pull/582)). That clause rests on the author's argument in `S0_S3_MAXIMUM.md`, landed with the author package at [#581](https://github.com/openwave-labs/openwave/pull/581) and labelled there as not independently audited.

This task asks one question: **is the hexagon orbit the only maximizer of `r̂₆` on the unit sphere of `V₃`?** It answers it twice, by an auditor's own argument committed before it sees the author's, and by grading the author's argument step by step.

### Standing: the #512 ruling

- **Run-before-write** is met on its plain reading: M8.12's pre-registration ran at #578 and closed with its package at #581 and the correction at #582.
- **One program, one pre-registration**, within the § 12.2 budget.
- **A claim that frozen text is defective** is reproduced by the maintainer with independent code before ratification.

M8.13 is allocated here as the next free ID in creation order, under `ROADMAP_STANDARDS.md` § 6. If a row taking M8.13 lands first, whoever merges second renumbers this one, which edits this document and so moves its hash.

### What this task is not

It audits one proposition about one quartic on one seven-dimensional space. It computes no Hessian and no index, reopens none of M8.12's reproduced values, and makes no stability claim. No `MODELS.md` cell moves and M8.7's gate is unchanged.

### Ownership and run format

The author freezes the claims below and supplies the argument under audit, which is already public. One maintainer-run auditor works in two stages, as M8.11's audit did for its theorem. At stage 1 it receives only the worklist and commits its own answer and argument. Stage 2 has two parts, and each return is committed before the next part's files are handed over: at 2a the auditor receives the grading instructions for 2a and N2's variant, and grades the variant's part 3c; at 2b it receives the author's argument with its checker and log, and grades the argument step by step. Then adjudication.

### Sources of record

| Source | What it supplies |
| --- | --- |
| M8.12 ([#578](https://github.com/openwave-labs/openwave/pull/578), [#582](https://github.com/openwave-labs/openwave/pull/582)) | G2 reproduced in value and attainment, its uniqueness clause unresolved; G1, the minimum and its equality case, reproduced in both rooms |
| `research/scripts/m8_12_author/S0_S3_MAXIMUM.md` ([#581](https://github.com/openwave-labs/openwave/pull/581)) | the argument under audit, SHA-256 `7c634a33fdb0ee05d5a934395a414f6346ba26e049b07f2619961792ab725498` |
| `research/scripts/m8_12_author/check_s3_maximum.py` and its log | its checker, SHA-256 `34054633f9602e864c12d017b035e6517ebbc4d8b0ba6c043dc1ef4adc764d3a`, and its log, `37a00d34168b23200a1f07694627dc2a93dd588b9156be5c55661deffab67b1d`: the inequality exactly, the equality set only by a 20001-point sweep |
| This task | U1 to U3, N1, N2 |

## SETTING

`V₃ = ℂ⁷` with the spin-3 operators `f = (fₓ, f_y, f_z)`, time reversal `(Θu)_m = (−1)^{3−m}·conj(u₋ₘ)`, and the top multipole `ρ₆(u) = [u ⊗ Θu]₆` in the Condon-Shortley convention, fixed by `⟨3 3; 3 3 | 6 6⟩ = +1`. Write `r̂₆(u) = ‖ρ₆(u)‖²/‖u‖⁴`, as in M8.12.

The argument under audit runs in four steps. Step 1 writes `r̂₆` on the unit sphere as a fixed combination of three invariant quartics: the magnetization, the singlet-pair weight and the nematic invariant. Step 2 bounds the first two, each with its equality case. Step 3 bounds the third, `TrN̄² ≤ 171/2`, by reducing it to `λ_max(e₁fₓ² + e₂f_y² + e₃f_z²) ≤ 15/√6` over unit traceless `e`, splitting that operator into three 2×2 blocks and bounding each. Step 4 combines the three equality cases.

**Where uniqueness lives.** Not in step 4, which imports step 3's equality result and imposes step 2's two conditions. It lives in the equality cases, and step 3's is the substantive one: the tracing of where each block reaches `15/√6` on the constraint ellipse, and the passage from there to the states attaining `TrN̄² = 171/2`. Step 4 would read as correct even if step 3 had missed an equality point, so step 3's tracing is what this task grades hardest.

**One property of the equality set that the grading has to respect.** Each of its three points is reached by two of the three blocks at once, so no single block is load-bearing for the set. That is a statement about blocks, not about the note's three traced cases: the case `b = 0` with `a > 0` carries the first point for M₁ and, through the symmetry sentence, for M₂, so omitting it would lose that point. Omitting the third block's case instead leaves every point covered while the argument no longer shows that the third block contributes nothing further. N2 uses exactly that omission, and it scores whether the grader notices it.

## THE FIREWALL

The stage-1 room has no access to: this document; `S0_S3_MAXIMUM.md` and its checker and log; the author's check described below; and M8.12's task doc, method note and author package. It receives the worklist only. At 2a it additionally receives `stage2a_grading.md` and N2's variant, `step3_text.md`, and nothing else: the argument itself would expose the variant's omission by comparison, and its checker names the third block's equality point among its exact checks. At 2b it additionally receives `stage2b_grading.md`, and `S0_S3_MAXIMUM.md` at the hash above with its checker and log.

**Network posture: offline, and this is load-bearing for U1.** The author's uniqueness argument has been public in this repository since #581, so an unrestricted online room could locate the very argument under audit. If network access is enabled and that argument is reached at stage 1 from any source, U1 cannot be scored as an independent reproduction, and stage 1 must be rerun offline. In the literature, Kawaguchi and Ueda's review states the bound `TrN̄² ≤ 171/2` without proof, and Romero et al. probe the maximization numerically without settling it; external material located without exposure to the author's argument is recorded as located, and U1 still passes only on the auditor's own completeness argument. The worklist's standing request for anything looked up is the discriminator.

## DISCLOSURE

- **The argument under audit** was written on 2026-09-19 and has been public since #581, with its hash.
- **Its checker** proves the inequalities exactly and corroborates the equality set only by a 20001-point sweep of the circle. A sweep rejects candidates; it cannot show there are no others.
- **An author-side check, written for this task on 2026-09-21**, `m813_equality.py` with its log, determines step 3's equality set exactly: for each block, the equality condition is a conic whose resultant with the ellipse is not zero, so the two conics share no component and Bézout bounds their common points by four; exact factoring over `ℚ(√6)` then enumerates every candidate, and whether `15/√6` is the block's larger eigenvalue there is decided symbolically. It returns the three points below, each reached by two blocks, each a permutation of `(2, −1, −1)/√6` with a two-dimensional top eigenspace: 25 checks, 0 failures. The gate reruns it and requires its output to equal the log byte for byte. It does not grade the note's own tracing. It is withheld from the auditor at both stages, so that the audit's route is its own, and it lands with the author package after the verdict.
- **Review units** read the argument during the M8.12 redlines. One checked the inequalities and both sums of squares, and stated that it did not check the exhaustiveness of the tracing.
- **The author's interest**: the answer settles, at spin 3, a question the author's fourth bedrock paper records as open, and the author intends to cite it only after this audit.

## CANDIDATE PRE-REGISTERED CLAIMS

### Group P: parents, reproduced rather than new

| ID | Claim | Standing | Pass condition | Fail condition |
| --- | --- | --- | --- | --- |
| P1 | The maximum of `r̂₆` on the unit sphere of `V₃` is `463/924`, and the hexagon orbit attains it | M8.12 G2, value and attainment, #582 | the stage-1 argument reaches the same value | a different value |
| P2 | The minimum is `1/924`, attained exactly on coherent states | M8.12 G1, both rooms | the stage-1 argument reaches the same value and set | a different value or set |

### Group U: uniqueness

| ID | Claim | Standing | Pass condition | Fail condition |
| --- | --- | --- | --- | --- |
| U1 | The maximizers of `r̂₆` on the unit sphere of `V₃` are exactly the hexagon orbit, the orbit of `(v₃ + v₋₃)/√2` under rotations and phase | new at run level; the clause M8.12 left unresolved | the auditor's own stage-1 argument, committed before it sees the author's, establishes the complete set, states why it is complete, and says for each case what fails if it is omitted | another maximizer is found, or the stage-1 argument does not establish completeness |
| U2 | The route of `S0_S3_MAXIMUM.md` steps 1 to 4, with any completion the auditor supplies named explicitly, establishes the maximum and its equality set, with step 3 graded in five parts: the block decomposition; each block's bound; the equality set of `λ_max = 15/√6` on the ellipse; the passage from `‖Q‖ = 15/√6` to a top eigenspace of an extremal operator; and that eigenspace as `span{\|3,3⟩ₙ, \|3,−3⟩ₙ}` | the author's argument, public since #581 | every step and every part of step 3 ESTABLISHED, as written or with the supplied part named | any GAP or DEFECT, recorded with its location |
| U3 | G2's uniqueness clause is recorded as an audited argument | derived from U1 and U2 | U1 and U2 both pass, as M8.11 required of its theorem | either fails; a partial verdict then records which one held |

### Controls

| ID | Claim | Standing | Pass condition | Fail condition |
| --- | --- | --- | --- | --- |
| N1 | The stage-1 return proves the minimum and its complete equality set, the coherent states, where the answer is known | known: M8.12 G1 | the known value and set, with a completeness argument | a different set, or no completeness argument, which would mean the stage-1 procedure cannot certify a complete equality set even where the answer is known. It is not a test that the maximum's method transfers: bounding the invariant form term by term reaches the maximum but not the minimum |
| N2 | Given at 2a, before it sees the argument or its checker, a copy of step 3 whose equality tracing omits the third block's case, although its conclusion survives, the grader notices the omission at the equality-set part | new | the 2a return explicitly notes that the tracing sentence omits the third block's equality case, and addresses it. N2 is scored on that content, not on the label: the case may appear in what the grader supplies, in a GAP note, or in a derivation it writes out, and whatever else it supplies, including a branch analysis, neither earns nor costs the pass. The scale-correct grade is ESTABLISHED, SUPPLIED, with the case supplied. A GAP, or an ESTABLISHED that writes out a derivation from the third block's own bound, also passes N2, since each shows the omission was noticed, but each is recorded as a scale misapplication: the case can be completed, so it is not a GAP, and a derivation the grader writes is a supplied part, so it is not ESTABLISHED as written | the 2a return accepts the complete equality set without addressing the third block's equality case, whatever label it gives, which would mean the grading reads conclusions rather than arguments |

### Diagnostics, not claims

| ID | What it records |
| --- | --- |
| D1 | Each equality point is reached by exactly two blocks, so the top eigenvalue `15/√6` has multiplicity two there, one vector from each block, which is the two-dimensional span in U2's last part. Recorded as a consistency property, not adjudicated |

## FROZEN VALUES

| quantity | value | source |
| --- | --- | --- |
| the maximum | `463/924` | `S0_S3_MAXIMUM.md`; M8.12 G2 |
| the nematic bound | `TrN̄² ≤ 171/2`, equivalently `λ_max ≤ 15/√6` over unit traceless `e` | `S0_S3_MAXIMUM.md` step 3 |
| the constraint ellipse | `(2/3)a² + 8b² = 1`, with `e₃ = 2a/3` and `e₁ − e₂ = 4b` | step 3 |
| the equality points on it | `(√6/2, 0)`, `(−√6/4, √6/8)` and `(−√6/4, −√6/8)` | `m813_equality_log.txt` |
| which blocks reach each | the first by M₁ and M₂; the second by M₁ and M₃; the third by M₂ and M₃ | same |
| the same points as `√6·e` | the three permutations of `(2, −1, −1)` | same |
| the top eigenspace at each | two-dimensional; at `√6·e = (−1, −1, 2)` it is `span{v₃, v₋₃}` | same |
| the maximizing set | the orbit of `(v₃ + v₋₃)/√2` under rotations and phase | `S0_S3_MAXIMUM.md` step 4 |
| the minimum and its set | `1/924`, on coherent states | M8.12 G1 |

## FEASIBILITY, AND THE AUTHOR'S DERIVATION

The argument is one page, its checker runs in under a minute, and the author-side check above runs in about a second. The auditor's stage 1 is a genuine derivation, and it may not close: M8.12's solver A argued the maximum's value and stated explicitly that it had not settled uniqueness. U3's rule makes that outcome a partial verdict rather than a failure of the task.

**The author expects part 3c to grade ESTABLISHED, SUPPLIED, not ESTABLISHED as written.** The note states the sign conditions of M₁'s equality cases, `b = 0` with `a > 0` and `a = −2b` with `b > 0`, without deriving them from the squaring branch `a + 6b > 0`, and it does not show that the other branch, `a + 6b ≤ 0`, contains no equality point. Each of those case lines meets the ellipse twice, and only one point of each pair is an equality point. A grader following the scale should supply that branch analysis. That is still a pass, and it is recorded as one the auditor completed. N2's variant inherits the same omission, so its part 3c is expected to need that branch analysis and the third block's equality case supplied: the two texts are expected to earn the same label, and a matching label is not a failure of the control.

## TO BE FIXED AT GO

- Every file the auditor sees, at each stage, byte-pinned: the worklist; the grading instructions for 2a and 2b; N2's variant; `S0_S3_MAXIMUM.md`; `check_s3_maximum.py` and `check_s3_maximum_log.txt`; and the maintainer's room brief, if there is one. The pre-registration's own instruments are pinned in the amendment below, apart from the two logs, which are posted with it.
- The exactness rule, as #547's: exact means a symbolic derivation, or an identification stating its precision, repeated at a second precision.
- The argument-grading rule, M8.11's with its overlap resolved: one verdict per step, ESTABLISHED, ESTABLISHED, SUPPLIED with the supplied part named, GAP or DEFECT, and a passing argument is recorded as an audited argument, never as a verified or proven theorem.

## AMENDMENT (before go, 2026-09-22)

The maintainer approved and merged this pre-registration at [#583](https://github.com/openwave-labs/openwave/pull/583#pullrequestreview-5278860214), with two questions and three notes, and pinned its seven instruments by hash in the review. This section records what changed in response, before the go. No frozen value moved, and no claim was added or removed.

- **N2 is sequenced, so it cannot be passed by comparing texts.** Handed over together with the argument, the variant differs from it in one clause, so its omission could be found by comparison rather than by reading the variant as an argument. Stage 2 now runs in two parts, each return committed before the next part's files are handed over: at 2a the variant alone, at 2b the argument with its checker and log. The checker waits for 2b as well, since it names the third block's equality point among its exact checks. The variant is renamed `step3_text.md`, byte-identical, so that its file name does not announce it as a variant, and the grading instructions are split to match, with the four verdicts and their rules unchanged.
- **N2 is scored on content, not on the label.** Both texts are expected to earn ESTABLISHED, SUPPLIED, so the label cannot tell them apart. N2's pass condition now scores whether the third block's equality case is addressed, and the pre-registered expectation names what each text is expected to need.
- **The firewall names only what the repository resolves.** An exclusion that named no file in the repository is removed. Stage 1 was already a whitelist, receiving the worklist only.

**The instruments.** Of the seven files the review pinned, two are unchanged and match it byte for byte:

| file | SHA-256 |
| --- | --- |
| `m813_equality.py` | `3577872c112cefe13413eab3a706e834314e635129f7a914f5488d43f3cc090d` |
| `m813_equality_log.txt` | `a30b8198d329d2a038ea3253fce2253b007954db0581fa4b71aeac5d12de0354` |

The gate, its mutation suite and its inventory changed, because this amendment added gates and arms for the sequencing, the scoring rule and the firewall. These hashes supersede the review's for those three files only:

| file | SHA-256 |
| --- | --- |
| `m813_prereg_build.py` | `eb0a440887bb778739464d6fed7bcc350038170f024d7820840a8d1480b66dd6` |
| `m813_prereg_arms.py` | `f068c3d27eac293f424df7069d8559c35e8ee8e224964b9e62938a7206c128b0` |
| `gate_inventory.txt` | `71e9a8887b077c382af0290cfc191f2f455298fb2284326e378619e813b78a6c` |

The gate's log and the mutation suite's log are outputs of runs that read this document, so their hashes cannot be stated here without changing them. They are posted with this amendment, and supersede the review's for those two files only.

## DEFINITION OF DONE

The auditor returns its stage-1 answer and argument, then its 2a grade of N2's variant, then its 2b grades of the argument. Adjudication records U1, U2 and U3, and G2's uniqueness clause moves from unresolved to an audited argument only if U3 passes. If U2 passes with any part graded ESTABLISHED, SUPPLIED, G2's status records what was supplied, for instance "audited argument, with the branch analysis of step 3c supplied by the auditor", so the record says whose argument it was, which is the lesson #582 exists for. No stability claim, no `MODELS.md` cell, and no change to M8.7's gate.
