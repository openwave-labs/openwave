# M8.1.3: FOURTH BLIND RUN, the first dynamics paper (small-amplitude branches on S³/2I)

> Roadmap row: [`../m8_roadmap.md`](../m8_roadmap.md) M8.1.3 (**maintainer-run**), filed by this pull request as a BACKLOG row.
> Parent template: [`m8_1_task_details.md`](m8_1_task_details.md) (the worked blind-run protocol); immediate precedent
> [`m8_1_2_task_details.md`](m8_1_2_task_details.md); the admission: [#512, the answer of 2026-11-15](https://github.com/openwave-labs/openwave/discussions/512#discussioncomment-19632863).
> MIT's dated note on the answer came first, as the question offered and the answer accepted:
> [M8.1.3's page on MIT, at `6bb46e8d`](https://github.com/dmobius3/mode-identity-theory/blob/6bb46e8d7c876b852bddb6a54a25747cc950730e/files/tools/files/openwave/m8-1-3/README.md).
> Status: author-proposed 2026-11-16. Each block is accepted or declined in this pull request's thread before it merges;
> registration and go pending.

## TASK PLANNING (2026-11-16, author-proposed; block acceptance pending)

### Scope

Blind verification, in the M8.1.x sense, "blind verification of what a paper already carries", of the paper *Nonlinear Standing-Wave Branches on the Poincaré Homology Sphere: Ground States and Small-Amplitude Spectral Stability*, deposited 2026-11-14. It studies a defocusing cubic wave equation for fields in the twisted bundles of X = S³/2I. In three soft slots, small-amplitude standing-wave branches are born at the critical orbits of a reduced quartic, and their spectral type is read off the quartic's data. A rigid family exists in the other slots.

The audit comes in five blocks, each accepted or declined on its own in this thread:

| Block | What a room computes, blind |
| --- | --- |
| A. The reduced problems | the quartic's channel coefficients and its spin-6 coefficient in the three soft slots, from the quartic's definition; one slot's eight examined points; the bounds that follow at the top weight vector |
| B. The small-amplitude data, the priority | at each of the 21 orbits and points the paper tabulates: criticality, the signature, the orbit and radical dimensions, the zero multiplicity, the slow polynomial and its flow, and the Krein signs; the second level-6 slot computed from its own coefficients; the link between the paper's spin-6 factor and M8.12's invariant |
| C. Beyond leading order | the levels where the first correction has weight; the next levels and gaps; the two degenerate orbits; the first correction's sign at every tabulated point; and, at five cases, a supplied formula's value |
| D. The certificates | the three orbits outside the census's class, in both level-6 slots: Krawczyk inclusion on the full critical-point system in given boxes, with the rooms' own interval code |
| E. Optional: the rigid families | the linearization on the lowest eigenspace, for each rigid slot |

**What it leaves out**, each for its reason:
- the paper's numerical study, which is this column's own adjudicated record, M8.15 to M8.19;
- model checks that state no value the paper reports;
- the proofs, which would be their own ask.

MIT's own computation of the open items runs on a different instrument, off M8.16.1's pin `8cc2d7e2`, outside M8. No result in the paper rests on it: the paper places it outside OpenWave's review and cites none of its values (p. 23; the answer of 2026-11-15, § 5).

**One departure from the question of 2026-11-15.** The question said that what M8.12, M8.1.2 and M8.10 reproduced blind would be "cited rather than repeated": the ten census orbits, the two level-6 weights, and the first correction at the four pinned rays. This document recomputes them where its blocks meet them. B1 rescores the census orbits' values and signatures, A1 the weights, and C the first correction at any point it shares with M8.10. That is the stronger test, and it is more work than the question priced. A disagreement with a value already on this column's record is read against that record first, under § "A failed claim" below.

### Why this task exists

| Point | Statement |
| --- | --- |
| The column's record | Three blocks reach things this column's closed record took from MIT's pages (the answer of 2026-11-15, § 2). **B:** M8.15 to M8.19 followed the branches born elliptic, and which branches those are is the small-amplitude typing. **C:** the paper's Proposition 4.7 is the Soft-Slot Branches' Proposition N, whose next-order terms M8.16.1 scored its instrument against. **D:** M8.12's task document leaves the three outside orbits "outside the claim"; the third of them is T4, whose rows M8.15 and M8.16.1 to M8.19 carry. A is B's input, and E stands apart |
| Why maintainer-run, by construction | The author and the author's agents wrote, checked and computed this paper. A self-run check is a reproducer, not an independent recompute; blindness is what the platform supplies that the author cannot |
| What it is not | Not a gate on anything running, and not a claim that the platform owes validation. The answer admits the filing and nothing more: until this pull request files nothing is pending, and each block's workload binds only on its acceptance here |

### Ownership and what the author is not doing

The author proposes this task, supplies the sources, the definitions packet and the claims file, answers questions about the packet, and then stops. The author does not run or see the rooms, does not audit them, and takes no part in comparing their results with the claims. If the designer prefers to build the packet from scratch, that is strictly better for independence and the author has no objection.

Protocol authorship by an answer-holding author is platform-native under M8.5-A, which records that protocol authorship does not compromise independence but target-aware implementation does, provided a maintainer reviews and freezes the protocol and a fresh context implements behind a firewall. This task is written to that standard, as M8.1.2's was.

### Sources of record

| Item | Pin |
| --- | --- |
| The paper (deposit) | Zenodo, [10.5281/zenodo.24224845](https://doi.org/10.5281/zenodo.24224845), deposited 2026-11-14, the **version** DOI, so pinned to this text. `standing-wave-branches.pdf`, 891,097 bytes, MD5 `ea9c057e9817d1a09d321ae342c6f0e3` as Zenodo records it, SHA-256 `f2467b7c0166e965a115bcc25b42424072bf0d96651106e0f304e761500f2efe`. A later version is outside this task |
| The definitions packet | `../m8_1_3/conventions-and-worklist.md`, SHA-256 `ddb0b1a75a45f32376f618ca32d571e373a0d8f70c53b2efe2036828a2b75179`, and `../m8_1_3/inputs.json`, SHA-256 `64734303b824abd3667bbd90d5b1d80f6a42af9d7e9b4962ba5a63c164400ed6` |
| The claims file | `m8_1_3_claims.json`, 52,445 bytes, SHA-256 `25fb8a946ec72aa01a6f5e0c5b1ace28ec3269bb92ce7c2dce1469d519abb669`, taken over its serialization: UTF-8 JSON with keys sorted, separators `(',', ':')`, non-ASCII characters kept and one trailing newline. It reaches the maintainer before the rooms open (§ "Where the frozen claims live") |
| The author's scripts (designer-only) | MIT `dmobius3/mode-identity-theory` at `c7e3f46e`, the commit the paper's Data and code names, with its folders `slot-ground-states/`, `soft-slot-branches/` and `hamiltonian-krein-identification/`; and, at the same commit, [Sh]'s certificate file `surviving-ray-v2/sr_v2_certificates.json`, SHA-256 `eb174fa1b15f0ad9b309b7c45fa43814381188a37ef86afeebf7543b77ebe7ff` |

## THE FIREWALL

### Context isolation

The deposit is public, so a withheld title does not make a run blind; isolation does. As in M8.1.2:

| Requirement | Statement |
| --- | --- |
| Repository access | Room and audit contexts have no access to the OpenWave M8 tree, the `mode-identity-theory` repository, the deposited paper, or MIT's pages, in any form |
| Network | Offline for Block D. Otherwise no unrestricted web search; any external reference is a generic representation-theory or computer-algebra source the designer approves in advance |
| Manifest | Each room returns a consulted-material manifest. An empty or unreturned manifest is a protocol failure, recorded as such. Character tables and branching reported as DERIVED or as RECOGNISED, as M8.1.2 graded its Group B, the two group inputs |
| Workspace | Rooms write into their own scratch space, never into the repository; the designer copies their scripts into `research/scripts/` unmodified at FINISH |
| Audit ordering | An auditor commits its own method and results before it sees a room's |

### Where the frozen claims live

The values are not in this file: they are in the claims file, whose SHA-256 this file states. The claims are frozen by the designer at the go, not by the author: the designer may edit the file before the go, and an edited file is recorded at the go with its own SHA-256, beside this one. The file reaches the maintainer out of band before the rooms open, so that the packet can be checked against it. Its integrity is the stated hash (§ 5.1's transport rule). It is committed beside the record at FINISH. The rooms never see it.

Block D's boxes are the one exception to "never". The rooms receive them in the packet's `inputs.json`, as the x-part of each refined centre and its start point. The multipliers are left out, since a Lagrange multiplier there encodes a claimed value. This file holds no values, but it names the paper, its theorem and table numbers, and A1's closed form, which with the packet's own table gives A1's values by evaluation, so the leak path of record is this file as well as the claims file. **A run in which a room could have read this file, its history, the claims file, or MIT's pages is not blind, whatever the packet contained.**

### Withheld terms: attribution, not vocabulary

| Withheld | Permitted |
| --- | --- |
| the author, the model, the repository, the paper's title and its section, theorem and table numbers, the names of MIT's working papers, every claimed constant and every closed form a claim scores | the technical vocabulary the computation needs, including "Krawczyk", "Krein", "Clebsch–Gordan" and "Condon–Shortley" |

The group is given by two generating unit quaternions and is not named, as in M8.1.2. M8.1.2 is exact about what that buys: it removes the strongest search term, but an agent that knows the ADE classification can name the group in one step. So the anchors' character tables are reported DERIVED or RECOGNISED, and scored apart. The census points keep the names M8.12's record gives them; those names are public and print no scored value, though most name their point's symmetry.

### Designer-only material, quarantined from every room

The author holds the pinned scripts and their outputs, the certificate file, the bench checks that built and tested the claims file, the internal reviews of the claims file and the packet, with their adjudications, and the record of one trial run of the packet. **None of it may reach a room or an auditor, and none of it is part of adjudication.** The designer adjudicates against the frozen claims first and records that verdict; only then may the author's material be opened, for provenance-level comparison. As M8.1.2 states it: agreement between the author's code and a room is weak evidence, since a shared convention error survives it; disagreement is strong evidence that something is wrong somewhere.

### Offered instrument (adopt, modify, or discard)

| File | What it is |
| --- | --- |
| [`../m8_1_3/conventions-and-worklist.md`](../m8_1_3/conventions-and-worklist.md) | A conventions extract and a worklist, carrying the definitions and none of the results, in M8.1.2's form. Its sections: the setup; the conventions it fixes; the definitions by block; the worklist, anchors first; the inputs; what to return |
| [`../m8_1_3/inputs.json`](../m8_1_3/inputs.json) | The points the worklist names, Tables 2 and 3's representatives as printed, and Block D's six boxes with their start points and the realization of the four orbit generators |

The packet states each scored quantity's convention, as the answer asks: the form and the space of each signature (the shifted Hessian on the normal space, relative to the metric), the normalization of the eigenvectors behind c_g (ℓᵀr = 1), and the chart and slice of each Block D box. The author's leak check, which scans the packet for every value in the claims file and for numbers that encode one, is a diagnostic only. As M8.1.2's terms say, semantic leakage is the larger risk, so **before the rooms open the designer audits the packet against the claims file, manually or with a maintainer-written gate.**

### Disclosure: the author's prior exercise

The author's scripts compute every claim; they are pinned, not handed over, and a rerun of them is not an independent reproduction. Before proposal, the packet was exercised once by a fresh-context room (§ "Instrument qualification, and what is frozen"). That exercise is not offered as evidence that the paper is free of error: it scored a room against the author's own claims file.

## CANDIDATE PRE-REGISTERED CLAIMS

Seventeen claim groups, A1 to E1. Each carries its page in the PDF, its grade, its status and the matching rules it is scored by, all in the claims file. The status is one of three: a **claim**; an **identity**, which follows from the packet's definitions and is recorded as such (the answer's condition on forced claims, after M8.1.1's conversion identity); or an **input**, given to the rooms. Values the PDF does not print are carried there as the pinned scripts', marked as theirs: a disagreement on one is read for its cause first and fails no claim of the paper. Each claim is frozen at what a computation can establish, as the question put it: at the prism, that Ŝ_u's nonzero eigenvalues are imaginary and Krein-definite and its zero has multiplicity k + r_u, not that the branch is spectrally stable. A theorem's conclusion is recorded beside the score.

### Block A: the reduced problems

| ID | What is scored | Grade, status |
| --- | --- | --- |
| A1 | the spin-6 coefficient β_ρ in each soft slot, and the closed form d(d − r)/(13r); with it the ratio and difference the paper states | exact, claim |
| A2 | the channel coefficients of two slots, each above 1, and the third slot's top coefficient, with the 6j form; the third slot's other coefficients are the script's | exact, claim |
| A3 | one slot's eight points: criticality, the quartic's value, the signature | exact, claim |
| A4 | the quartic constant on three lines, forced by the fibre symmetry | exact, identity |
| A5 | the bounds at the top weight vector, each slot, under the paper's condition that ε is at most the gap | exact, claim |

**Route (the answer's second point, for Block A).** The rooms compute β_ρ and the q_J from the quartic's definition, the integral over X, exactly or to stated digits. The closed forms are not in the packet; they are scored by the rooms' own values matching them. Evaluating them would be the weaker test.

### Block B: the small-amplitude data

| ID | What is scored | Grade, status |
| --- | --- | --- |
| B1 | the ten census orbits in the first level-6 slot: criticality, 924 r̂₆ from M8.12's definition, the signature, k and r_u, the slow flow and the Krein signs; the zero multiplicities that the paper's hypothesis (Z) asks at two of them, v₂ and the prism (the third, at O₃, is B4's) | exact, claim |
| B2 | the same in the second level-6 slot, **computed from its own coefficients**, so that it can fail on its own, and every nonzero slow eigenvalue scored against the first slot's, the paper's Lemma 3.1(c) | exact, claim |
| B3 | the third slot's eight points, at the point and at four members of the stationary family | exact, claim |
| B4 | the three outside orbits in both level-6 slots, at their certified points: the same, Table 2's printed digits, the second slot's slow roots against the first's, and one orbit's interval containing a stated rational | interval, conditional on the certified points unless Block D runs |
| B5 | the paper's spin-6 factor and M8.12's r̂₆ as functions on V₃ | exact, identity |

**The answer's first point.** B2 is computed from the second slot's own coefficients, and Lemma 3.1(c)'s relation is scored as a claim.

**The third link (the answer's third point).** The paper's spin-6 factor is the Hilbert–Schmidt norm of the spin-6 part of ww†; M8.12's r̂₆ is that of [u ⊗ Θu]₆. Both maps are SU(2)-equivariant isometries onto the same isotypic part, so the two are equal, constant 1, under any phase convention, and the paper argues the link itself in its Appendix A. So the third link is labelled an identity, following from the packet's two definitions. Its normalization is still scored where it bites: B1 scores the ten values of 924 r̂₆, computed from M8.12's definition, so a wrong normalization fails B1.

### Block C: beyond leading order

| ID | What is scored | Grade, status |
| --- | --- | --- |
| C1 | the levels where the first correction has weight, in each slot, read from the level norms; and the ratio s | exact, claim |
| C2 | each slot's next level and gap | exact, claim |
| C3 | the two degenerate orbits: the isotropy group and its action, the kernel, and γ_u, in units of R² | exact, claim |
| C4 | the first correction's sign at every tabulated critical point | exact; interval at the outside orbits, conditional on the certified points |
| C5 | at five cases, the zero multiplicity k + r_u; and the value of a supplied formula at each slow group | exact; the formula's values the script's, a 50-digit evaluation, not certified |

**Route (the answer's second point).** The first correction R̃₀, γ_u and D_u are computed from their definitions, as M8.10's rooms computed theirs. The levels of C1 come from the level norms, not from the closed form's constants. For c_g, the coefficient's own definition is the ε² term of a slow eigenvalue, which only a derivation or an instrument reaches. So the packet gives the paper's formula ℓᵀT₁r, marked as a formula to evaluate, and the rooms evaluate it. A pass records that the formula gives the script's value; whether it is the next-order term stays with the proofs. No part asks a room to derive an expansion, so nothing in Block C falls under the ceiling's optional clause. The PDF prints no c_g; the values are the script's, so a disagreement on one fails no claim of the paper.

### Block D: the certificates

| ID | What is scored | Grade, status |
| --- | --- | --- |
| D1 | the three outside orbits in both level-6 slots: Krawczyk inclusion and uniqueness of the full slice system in each given box of radius 10⁻⁶⁰; enclosures of 924 r̂₆ inside Table 2's printed interval; the signature, k, r_u, the zero multiplicity and the slow roots, certified real and simple, with their signs; and the item of the paper's Theorem 4.2 each row rests on | interval: the rooms' own code, offline, on `mpmath` |

The boxes sit in the chart the packet states, about the paper's own refined centres. Those centres come from [Sh]'s certified boxes in the fixed sets, refined by two Newton steps on the full system. [Sh]'s boxes stay designer-side: the Surviving Ray v2's own verifier certifies them. As the question offered, O₃ comes first, no global search is asked, and a decline of D counts against no other block: B4 and C4 then stay conditional on the certified points, as their rows say. The rooms run offline, on the environment as it stands at the go. If a room's interval code needs a library beyond `mpmath`, this document names it before the go.

### Block E, optional: the rigid families

| ID | What is scored | Grade, status |
| --- | --- | --- |
| E1 | for each rigid slot and the control: the lowest level and the next, and the gap; and the linearization on the lowest eigenspace at a family member, its eigenvalues with algebraic and geometric multiplicities, exactly. Off that space the paper's positive conserved form is a proof's statement, recorded beside the score | an exact statement; optional |

### Inputs, each with its task and record

| Input | Record |
| --- | --- |
| the ten census points | M8.12 reproduced them; Table 2's representatives as printed. Their criticality is scored (B1) |
| the third slot's eight points | on no M8 record; Table 3's representatives. Their criticality is scored (A3) |
| the five cases of C5 | from the two lists above |
| Block D's boxes and start points, and the generators' realization | the paper's 10⁻⁶⁰ boxes on the full system; the generators as the bench's certification ran them |
| if Block A is declined, its coefficients | frozen inputs to B, as the question said |

### Diagnostics on the run (not claims about the paper)

The packet's anchors 0(a) to 0(e) are reported first and recorded as diagnostics, never scored against the paper, apart from 0(c)'s s, which C1 scores:
- (a) the generators' norms, the group's order, classes and characters;
- (b) each representation's levels;
- (c) each soft slot's multiplicity, intertwiner and normalizing constant;
- (d) a Clebsch–Gordan sign and Θ²;
- (e) charge conjugation in the quaternionic slot against Θ.

A mismatch there localizes a convention error before anything is scored.

### Feasibility triage

| Block | Arithmetic | Size |
| --- | --- | --- |
| A | exact, or quadrature at 30 digits | three slots, a handful of points each |
| B | exact symbolic linear algebra on 14- and 16-dimensional real spaces; interval at the outside orbits | 21 orbits and points, two slots for the census, four members at each of eight points |
| C | exact or high-precision spectral sums; one formula evaluation | the heaviest: R̃₀ at 34 points, two degenerate orbits, 19 slow groups |
| D | interval arithmetic at about 90 digits | six boxes |
| E | exact, symbolic in ω, g₄, c and \|v\| | six slots |

This is close to M8.1.2's 21 claims in number, but it carries more values and more work, and each block is priced on its own, under § "OBLIGATIONS".

### Instrument coverage

The worklist reaches every claim: each claim ID maps to one or more items, and the author's check fails if one does not. The map is designer-side. The designer should still confirm coverage at the go rather than take this paragraph for it.

### Instrument qualification, and what is frozen

The packet was exercised once before proposal. A fresh-context agent, given only the handout and the inputs in an empty folder, ran four items:
- anchor 0(c);
- A.1 in the first level-6 slot;
- the first correction at one census point;
- one of Block D's boxes.

Scored by the author against the claims file by its own matching rules, all four agreed: three exactly, and the box certified, its enclosure of 924 r̂₆ inside both Table 2's printed interval and the pinned scripts' enclosure, with D1's signature. The room worked from an earlier version of the handout. Its notes found no error that changes a result. Eight places where the handout was underdetermined or loosely worded were then clarified in the version pinned above, and no expected value changed. One room on four items shows the packet is executable where it had been found broken, not that every item is.

**What is frozen.** From the maintainers' launch the packet stops moving on the author's side: no edit because a computed result disagrees with a frozen claim. A wrong answer is a result. This binds the author, not the maintainers: if the designer wants to change the packet, that is a design decision and the designer's to make.

### To be fixed at go time (BEFORE any room opens)

- which blocks are accepted, and the rooms per block;
- the instrument decision: adopt the offered packet after confirming coverage, modify it, or build a fresh one;
- the claims file received, and its SHA-256 matched to this file's, or the designer's edited file recorded with its own;
- where the claims file and this file are held, and what keeps every room from reading them;
- the designer's own audit of the packet against the claims file;
- the environment pin for Block D's offline run, with any library beyond `mpmath` already named in this document;
- the matching rules, frozen as the claims file states them;
- the paper's row in `../../theory/_CITATIONS.md`, which carries its concept DOI, records the version DOI above as M8.1.3's subject, as the Surviving Ray's row records M8.1.2's.

### Which blocks run together, and what each takes from another

- A and B run first and together, since A's coefficients are B's inputs. If A is declined, its coefficients enter B as frozen inputs.
- D can run beside them.
- C follows. It takes only the packet's definitions and its inputs and recomputes what it needs, so it does not wait on A's or B's verdicts.
- E is optional and stands apart.

The order is the author's to state and this is it. Nothing is added to a block after its go.

### A failed claim (the answer's fourth point)

A failed claim is read first for its cause: the packet, the room or the claim. A claim that the frozen text is defective is reproduced by the maintainer, with independent code, before it is ratified, as answer 1 requires. Where a ratified defect touches a value that a closed task's terms froze, or a certificate that a closed task's rows rest on, that task is not reopened and nothing reruns under any ID. The defect is recorded in M8.1.3's own record, which names each closed task it touches and what that task rested on. For M8.16.1, which scored its instrument against the formula C5 evaluates, the ruling of 2026-10-20 already says that a ratified defect "is recorded in M8.16.1's record and does not by itself open a third qualification".

### The verdict's scope (the answer's fifth point)

The verdict attaches to the pinned PDF, named above by its version DOI beside its hash; a later version is outside it. A pass reads as M8.1.2's does: the frozen claims reproduce, blind. It claims no theorem. No `MODELS.md` icon moves on the outcome and M8.7's gate stays as it is, which the Done row states, as each Done row has since M8.12. Whether a cell's text records the outcome is decided separately, after the adjudication.

### Definition of done (skeleton, finalized at go)

| # | Item |
| --- | --- |
| 1 | Room runs for each accepted block, with exact values where the claims ask for them; their scripts and outputs in the repository under `m8_1_3_` prefixes, copied unmodified |
| 2 | An adversarial audit with its own method, committed before it sees a room's work (§ "Context isolation"), with per-claim verdicts |
| 3 | The designer's comparison against the frozen claims file, its SHA-256 matched, every number stated, including any that landed elsewhere; the anchors recorded apart from the claim verdicts, except 0(c)'s s, which C1 scores; a declined block recorded as an outcome |
| 4 | Method note `findings/m8_1_3_method_note.md`: equations first, the equation-to-code map, the audit record, and the rooms' manifests |
| 5 | The author's material opened for provenance comparison only after the verdict is recorded, with the asymmetry between agreement and disagreement stated |
| 6 | Doc sync: canonical, briefing and roadmap row; the Done row carries the verdict's scope; no `MODELS.md` icon moves |
| 7 | Doc checker exit 0; TASK REVIEW presented |

### Scheduling

Maintainer-run at maintainer pace. Not on the author's critical path, and not blocking the column's live front, which is empty. The run-before-write condition is met: M8.19 has run and been adjudicated, and the In Progress and Backlog tables are empty (the answer of 2026-11-15, § 1).

## OBLIGATIONS

Each obligation binds only on its acceptance in this pull request's thread, before merge, against this table. A declined block is a recorded outcome, never a breach. The merge files the BACKLOG row and these terms, and no maintainer work comes due at it: each block's rows bind on that block's acceptance here, and their work comes due at the go, a later deliberate act.

| Obligation | Who discharges it | When |
| --- | --- | --- |
| **Block A**, if accepted: blind rooms behind the firewall above, and their audit, on the quartic's coefficients from its definition, three slots, a handful of points each; adjudication of A1 to A5 | maintainers | after the go |
| **Block B**, if accepted: the same on 21 orbits and points, two slots for the census, four members at each of the third slot's eight points, interval arithmetic at the outside orbits; adjudication of B1 to B5 | maintainers | after the go |
| **Block C**, if accepted: the same on R̃₀ at 34 points, two degenerate orbits and 19 slow groups; adjudication of C1 to C5 | maintainers | after the go |
| **Block D**, if accepted: offline rooms certifying six boxes at about 90 digits; adjudication of D1 | maintainers | after the go |
| **Block E**, optional, if accepted: the same on six slots, the lowest level, the gap and the linearization on the lowest eigenspace, exactly in ω, g₄, c and \|v\|; adjudication of E1 | maintainers | after the go |
| Audit the packet against the claims file, for the accepted blocks | maintainers (the designer) | before the go |
| Fix the go-time items above, among them where the claims file and this file are held | maintainers (the designer) | at the go |
| Reproduce with independent code any claim that frozen text is defective, before it is ratified | maintainers | only if such a claim arises |
| At FINISH: record each room's manifest, copy the rooms' scripts into `research/scripts/` unmodified, commit the claims file beside the record, and write the Done row with the verdict's scope | maintainers | at finish |
| Supply the packet and the claims file, deliver the claims file before the rooms open by a channel the designer names, checked against the hash above, and answer questions on the packet | the author | before and during the run |
| Leave the packet as filed from the launch: no edit because a result disagrees with a frozen claim | the author | from the launch |
| Record the filing on both M8 status rows, the roadmap's "Where is M8?" and "What decides the program?" | the author, in this pull request | at filing |
| Record the outcome on MIT's side, in a dated note | the author | after the adjudication |

No obligation asks for original derivation. The proofs are outside the task, and Block C's formula is evaluated, not derived. So nothing here is optional under the ceiling's last clause. Block E is optional by the author's own offer.

## GO-TIME PRE-REGISTRATION (2026-11-16, go 10:30 EST)

The maintainer's go. The planning sections above are kept as filed, so their status lines ("registration and go pending") describe the document at filing: the blocks were accepted at the merge of #731, and the go was given on 2026-11-16. No value in the claims file is edited. This section records what was fixed before the rooms opened; the packet files were hashed before launch, and this text was written while the rooms ran (deviation 1).

### The go-time checklist, answered

| Item | As run |
| --- | --- |
| Blocks accepted | all five, A to E, in the review of [#731](https://github.com/openwave-labs/openwave/pull/731), merged as filed at `0300b2bc` |
| Instrument | the offered packet, adopted unchanged: `conventions-and-worklist.md` and `inputs.json` at the hashes of Sources of record, given to every room as `worklist.md` and `inputs.json`. Beside them, `clauses.md`, the designer's five clauses, recorded on [#731](https://github.com/openwave-labs/openwave/pull/731#issuecomment-6906779256) before the go, which take precedence where they speak. None adds a value |
| Coverage | confirmed at the designer's audit of the packet against the claims file, recorded in the same comment: complete apart from C1's script constants, with two quantities asked outside their own block (C3's `k` and `r_u` at `v₁`, now clause 5; C4's criticality, read from A.4, B.1, B.2 and Block D) |
| The claims file | received at the author's commit `72077986229a71a485461476fc6bb40f2671a658`, 52,445 bytes, SHA-256 `25fb8a946ec72aa01a6f5e0c5b1ace28ec3269bb92ce7c2dce1469d519abb669`, equal to the pin; re-serialized as stated, it reproduces its own bytes. Not edited |
| Where it is held | in the designer's working folder, outside every room. This file was on `main` at the go (`0300b2bc`). No room can read either (the containment table below) |
| The designer's audit | done before the go and recorded on [#731](https://github.com/openwave-labs/openwave/pull/731#issuecomment-6906779256), with the six readings the adjudication applies |
| Environment | a conda environment kept for rooms alone: Python 3.12.14, numpy 2.5.3, scipy 1.18.1, sympy 1.14.0, mpmath 1.3.0, and the packaging tools that come with the interpreter. No library beyond `mpmath` |
| Matching rules | frozen as the claims file states them |
| Posture | OFFLINE for every block, the stricter of the two this document admits for A, B, C and E |
| Citations row | at FINISH |

### The packet as run

| File | SHA-256 |
| --- | --- |
| `worklist.md` (the merged `conventions-and-worklist.md`, byte-identical) | `ddb0b1a75a45f32376f618ca32d571e373a0d8f70c53b2efe2036828a2b75179` |
| `inputs.json` (the merged file, byte-identical) | `64734303b824abd3667bbd90d5b1d80f6a42af9d7e9b4962ba5a63c164400ed6` |
| [`../m8_1_3/as_run/clauses.md`](../m8_1_3/as_run/clauses.md) | `92162f6f29da705f61070ca9918b264ef5adfc89a4fd67dc4695b7aa12c81b46` |
| [`../m8_1_3/as_run/opening_prompt.md`](../m8_1_3/as_run/opening_prompt.md) | `7083043022280d18b0715bd5cb3cf85e0a2988254b81f9f0eba024eb8d18a248` |

Each room also holds its own brief, `BRIEF.md`, carrying only the room's mechanics and its list of items: [`../m8_1_3/as_run/`](../m8_1_3/as_run/), `brief_<room>.md`. The stage-1 briefs were hashed before launch; the two stage-2 briefs were written and hashed at their own launches.

| Brief | SHA-256 |
| --- | --- |
| `brief_l6.md` | `9f73806f05e8d0bad58e313d6cd1a7295a9e9aad370f6b605ff00a8665933318` |
| `brief_r2.md` | `9d17c59d9c4375ef06b0afb0f51240d2ffd6ab78b859395ccf1935290b1c891a` |
| `brief_cert.md` | `c424114368224057eb7522f32de724b17205a7ab010ab1a33cfdc47d8b3034c2` |
| `brief_ca.md` | `886be315cc3a254784818f8f92218eb147f778cbf0a8a2a63f253317f9afae8d` |
| `brief_cb.md` | `9a01d187e81153fdcc5ae102e163f0e967ae92fb12393af262e1d9ffa13ad93f` |
| `brief_rig.md` | `425abbd8bdb886ab37a19203c6bc2895efd5b84ba2de986edb27ee0a4eef3a95` |
| `brief_xa.md` (audit `a`, stage 1) | `50de5ade642a9d0087cd5b3ea2ada4a26fe36179da4b3fd9948fe1fdba9386aa` |
| `brief_xb.md` (audit `b`, stage 1) | `d797aa93a7d5bb8f4a10b055d25d087001e94b849a2c9fa1067c97efb86db9e7` |
| `brief_xa2.md` (audit `a`, stage 2) | `69799a9f8b2d8a22ee1979d1c4f33f292a5622748b7fa8effef138158227f6e0` |
| `brief_xb2.md` (audit `b`, stage 2) | `b96bf5884f8b5ccb224b6ded511c456d162aec6d6af2622a8f8f0c78fef8f00c` |

**The leak gate.** From the claims file's value fields (1,120 strings; the fields `what`, `on_record`, `source` and `matching_rules` are not read) the gate takes three pattern classes: 79 fractions other than `1/2` to `7/2`, 124 decimals of four or more places and 181 integers of three or more digits. It scans those and 32 withheld terms against each packet file. Small integers (signatures, `k`, levels, gaps) are outside it, as are two scored values that appear only in the skipped fields, `49/156` and `81/256`; none of those appears in a brief or in `clauses.md` either, by reading. `worklist.md` and `clauses.md` match only `924`, the worklist's own factor in `924 r̂₆`. The stage-1 briefs and the opening prompt match nothing, and the two stage-2 briefs, gated when they were written, match nothing ([`gate_stage2_log.txt`](../scripts/m8_1_3_maintainer/gate_stage2_log.txt)). `inputs.json` matches only Block D's start points and centres, which the claims file carries as the same inputs, the component `sqrt(1/4)` (whose `1/4` coincides with C1's `s_N` in `R₂`) and two factorials of the chart. The gate's selftest plants a claims fraction, a claims integer and a term and flags all three. ⚠️ The first run of the gate caught one leak of the designer's own: an example fraction in the briefs' return format was a claims value. It was replaced by placeholders before any room existed.

### Rooms

Six solver rooms and two audit rooms, launched together at 10:35 EST, each a headless one-shot session under the launcher of [`CLEAN_ROOM_STANDARDS.md` § 3.4](../../../../../dev_docs/CLEAN_ROOM_STANDARDS.md#34-launch-headless-or-operated) on the room environment, model Opus 5.5. Each room's init record shows four tools, no MCP server and no skill.

| Room | Items |
| --- | --- |
| `l6` | 0(a) to 0(d); A.1, A.2 and A.6 in `R₄` and `R₅`; A.3; B.1, B.2, B.5 |
| `r2` | 0(a) to 0(e); A.1, A.2 and A.6 in `R₂`; A.4, A.5; B.3 |
| `cert` | 0(a) to 0(d); Block D, all six boxes; then B.4 at the points it certifies |
| `ca` | 0(a) to 0(e); C.1, C.2; C.4 at every point it names |
| `cb` | 0(a) to 0(e); C.1, C.2; C.3; C.5 |
| `rig` | 0(a), 0(b); Block E |
| `xa`, audit, stage 1 | its own route, before it sees any room's work: anchors; A.1 to A.3; A.4 at two points; B.1 and B.2 at three census points; B.3 at two points with the members; B.5; E.1 and E.2 in two slots |
| `xb`, audit, stage 1 | its own route: anchors; C.1, C.2, C.3; C.4 at the five cases and `O3 R4`; C.5 at two cases; Block D and B.4 at `O3` in both slots |

Every claim group has at least one solver room. C.1 and C.2 run in two solver rooms; the anchors run in every room. The audit's second stage is a new launch, after each solver's return and each auditor's stage-1 return are copied out and hashed, with the solvers' rooms copied in beside the auditor's own stage-1 files.

### Containment

| Route to the answers | Guard |
| --- | --- |
| The claims file, this file and every repository document | the session's file tools are confined to the room by `--restricted`, which still admits reads of the harness's own per-session output folders (the harness-pointer route of `CLEAN_ROOM_STANDARDS.md` § 3.4; those hold only the room's own output); code runs only through `./py`, whose deny-by-default sandbox reads the system libraries, the interpreter's install and the room, nothing else |
| Another room | the same: each room reads only itself. The auditors see the solvers' work only in stage 2 |
| Web, MCP, skills, agents | absent: four tools |
| The interpreter's install | the room environment holds only the numerics libraries; the launcher aborts on the platform's name or a locally installed package. It holds the numerics libraries and the packaging tools that come with the interpreter, nothing else |
| What loads unavoidably | a `--restricted` launch loads no instruction file and no memory; the account's email address and the working-directory path reach every session, as § 3.3 of the standard says, and the room paths name nothing |
| Designer-only material | the author's scripts were never fetched. The certificate file was fetched and hashed before the go; its three orbits were compared field by field with the claims file's designer-side copies at that check (deviation 2). Neither reaches a room or the adjudication |

## FINDINGS (run closed 2026-11-16)

Full record: [`../findings/m8_1_3_method_note.md`](../findings/m8_1_3_method_note.md). Returns: [`../scripts/m8_1_3_l6/`](../scripts/m8_1_3_l6/), [`../scripts/m8_1_3_r2/`](../scripts/m8_1_3_r2/), [`../scripts/m8_1_3_cert/`](../scripts/m8_1_3_cert/), [`../scripts/m8_1_3_ca/`](../scripts/m8_1_3_ca/), [`../scripts/m8_1_3_cb/`](../scripts/m8_1_3_cb/), [`../scripts/m8_1_3_rig/`](../scripts/m8_1_3_rig/), the audit rooms [`../scripts/m8_1_3_audit_a/`](../scripts/m8_1_3_audit_a/) and [`../scripts/m8_1_3_audit_b/`](../scripts/m8_1_3_audit_b/), and the designer's comparison [`../scripts/m8_1_3_maintainer/`](../scripts/m8_1_3_maintainer/). The claims file, committed beside the record: [`../m8_1_3/m8_1_3_claims.json`](../m8_1_3/m8_1_3_claims.json).

**Every frozen claim reproduces, blind, in all five blocks, and no claim was found defective.** The comparison against the claims file prints 514 lines under the file's own matching rules, and all 514 pass. Most set a value a room returned against the value the file freezes. Three record cells the degenerate rule leaves unscored and cannot fail; about 30 check a room's values against each other or against a clause (B2's scaling and ratios, B4's widths, ratios and overlaps, two counts); and D1 and B4 score the same enclosure and integers from `cert`, so about 30 lines are repeated between them. Planting one wrong claim value and, separately, one wrong room value turns exactly one line red each time.

### The adjudication

| Group | Verdict | What was compared |
| --- | --- | --- |
| A1 | ✅ reproduced | β in the three slots, exact; the ratio and the difference. The closed form `d(d − r)/(13r)` was not in the packet; the rooms' values satisfy it at all three `(d, r)` |
| A2 | ✅ reproduced | the eight channel coefficients of `R₄` and `R₅` and `R₂`'s `q₇`, exact; `R₂`'s other three agree with the script's values |
| A3 | ✅ reproduced | `R₂`'s eight points: all critical, exactly; `Q` and the signature at each |
| A4 | ✅ identity holds | `Q` constant on the three spans, as polynomials in `\|α\|²` and `\|β\|²` |
| A5 | ✅ reproduced | the two percentages in each slot, rounded up as printed |
| B1 | ✅ reproduced | at the ten census orbits in `R₄`: criticality, `924 r̂₆`, signature, `k`, `r_u`, the zero multiplicity (with (Z)'s at the prism and `v₂`), the slow polynomial (the script's, made monic), the slow-flow words and the Krein signs. At `v₁` the degenerate rule applies: `n₀ = 2` scored with the signature, the words unscored here |
| B2 | ✅ reproduced | the same in `R₅`, computed from `R₅`'s own quartic; at every orbit the room's `R₅` slow polynomial equals its `R₄` polynomial at `256ν/81`, and every paired eigenvalue ratio is exactly 9/16 |
| B3 | ✅ reproduced | `R₂`'s eight points: signature, `k`, `r_u`, zero multiplicity, slow polynomial, the words and Krein sign at the point, and the four members where the file scores them (`definite` at weight 7/2 and the line `{7/2, −3/2}`, `hyperbolic` at weights 3/2 and 5/2). The line `{5/2, −3/2}` is degenerate and goes to C3 |
| B4 | ✅ reproduced, at the points Block D certified | `O₁` to `O₃` in both slots: signature, `k`, `r_u`, zero multiplicity ((Z)'s at `O₃`), words; `924 r̂₆` inside Table 2's printed digits with width far under `10⁻⁵⁰`; `O₂`'s enclosure contains `28800/121`; each paired ratio enclosure contains 9/16, and each `R₅` root enclosure meets `81/256` of exactly one `R₄` root |
| B5 | ✅ identity holds | the Hilbert-Schmidt spin-6 factor and `r̂₆` are the same polynomial on `V₃` |
| C1 | ✅ reproduced, in two rooms | the levels with nonzero norm in each slot; `s_N`. The script's constants `t_J` and `κ_J` have no worklist item and are not scored |
| C2 | ✅ reproduced, in two rooms | next level and gap in each slot |
| C3 | ✅ reproduced | `R₂`'s line point: `\|Γ_u\| = 24`, `dim K′ = 3`, absolutely irreducible, `n₋ = 0`, `γ_u` exact. `v₁` in both slots: `K′ = ℂv₋₃` at circle weight 4, `γ_u` exact in each slot with the pair's Krein sign its sign, `(k, r_u) = (3, 1)` (clause 5), and off `K′` zero multiplicity `4 = k + r_u` with four simple negative slow roots |
| C4 | ✅ reproduced | `R̃₀ < 0` at all 28 tabulated critical points, exactly, and over the six certified boxes by interval; the six values the file carries as the script's agree as exact rationals |
| C5 | ✅ reproduced | the five zero multiplicities; all 19 slow groups' values of the supplied formula agree with the script's values, the largest relative difference `3.6e−12` against the rule's `10⁻⁹`, the octahedron's threefold group included. These are the script's values, not the paper's: a pass records that the formula gives the script's value |
| D1 | ✅ reproduced | all six boxes are certified in the box Section 3 defines (Krawczyk's test on an outward-rounded superset of it, with the image inside the exact box); signature, `k`, `r_u`, zero multiplicity, the slow roots certified real and simple with their signs; `924 r̂₆` inside Table 2's digits and inside the script's enclosure |
| E1 | ✅ reproduced, on `E₀` | the six rigid slots `n = 0` to `5`, their next levels and gaps, and on `E₀` the eigenvalue 0 with algebraic multiplicity `2n + 2` and geometric `2n + 1`, `±2iω` each `n` times, and the simple pair `±i(4ω² + 2g₄c²\|v\|²)^{1/2}`, exactly. Off `E₀` the paper's statement is a proof's and is recorded beside the score, not scored. The script's printed line (its gaps, and a remark on `Sym⁶`) is not scored beyond the levels and gaps it shares with the row |

The anchors 0(a) to 0(e) agree across every room that answered them (order 120, nine classes, the character table, the levels to 20, multiplicity 1 at the lowest level, `s_N`, the sign of `⟨3 3; 3 −3 | 6 0⟩`, `Θ²`), and each room DERIVED its character table; any recognition of the group, before or after, is recorded as unused. 0(e) depends on the phase chosen for `C`: rooms report the map as `Θ` or `−Θ`, which is that phase, and both readings give the same quaternionic structure.

### The adversarial audit

Two audit rooms, each first computing its share of the worklist by its own route, hashed before it saw any room's work, then grading the solvers' returns in a second launch with its own code only. Both recomputed every item and point in their scope. **Neither found a disagreement with any reported value, nor a defect in one.** What they demonstrated is in the rooms' checks and text: a false margin statement in `r2`'s return, a check `ca` calls independent that re-evaluates the same formula, a few PASS lines that cannot fail, and planted defects that do not exercise the route a value rests on. Each value so affected was recomputed by an audit route that can fail. Full record: the [method note](../findings/m8_1_3_method_note.md) § 5.

One reading decides scored values. In `R₂` every room and both audits took the fibre group to be the continuous `Sp(1)` that the worklist names, under which the quartic is invariant. The group the phase and the one map `ψ ↦ ψ̄C` literally generate is smaller, and under it `k` would drop by 2 at every `R₂` point. The claims file's `k` rule names `Sp(1)` in `R₂`, so the scores stand as read.

`|Γ_u| = 24` at `R₂`'s line point rests on two independent numerical searches and on the maximality of the octahedral group among finite subgroups of SO(3), not on 24 elements built exactly.

### The transcript audit

All ten room sessions were classified call by call against their rooms. No session read or wrote outside its room, apart from `cb`'s two reads of harness files holding its own output (deviation 5). Every launcher closing line reports `./py` and its profile unchanged. Every room returned a consulted-material manifest, and the two reads outside a room that the transcripts show, both `cb`'s, are disclosed in its manifest.

### What the record does not claim

No theorem. The verdict attaches to the pinned PDF, version DOI `10.5281/zenodo.24224845`, and reads as M8.1.2's: the frozen claims reproduce, blind. C5's values are the pinned script's, and their passing says nothing about whether the formula is the next-order term. E1 is scored on `E₀` only. No `MODELS.md` icon moves and M8.7's gate is unchanged.

### DEVIATIONS LOG

| # | Deviation |
| --- | --- |
| 1 | The go-time section was written while the rooms ran, not before. The stage-1 packet (worklist, inputs, clauses, opening prompt and the eight stage-1 briefs) was hashed before launch, and the two stage-2 briefs at their own launches; those hashes are what it pins |
| 2 | The certificate file, designer-only material, was fetched and hashed before the go, and at that check its three orbits were compared field by field with the claims file's designer-side copies. Nothing from it reached a room, and it played no part in the adjudication |
| 3 | The leak gate's first run caught a leak of the designer's own: an example fraction in the briefs' return format was a claims value. It was replaced by placeholders before any room existed, and the gate then passed |
| 4 | Every room's first command was a shell read of its own four packet files (`ls` and `cat`, in some rooms after a `cd` into the room itself), outside the `./py` rule; the client admits such reads without a prompt. Audit `a` also read `py` and `.room.sb` with `cat`, in both stages, and `ca` read one of its own logs with `cat`. Several later commands wrapped `./py` in `time` or piped its output through `tail`, `grep` or `cut`. All touched only the room, and the rooms reported them |
| 5 | Room `cb` read two harness files holding copies of its own output, outside the room, by the harness-pointer route, and disclosed both |
| 6 | The designer's comparison first stopped on the octahedron's threefold slow group, whose room records its values under a different key. The adapter now reads every eigenvalue of the group and requires all three to agree, which is stricter than the first version |
| 7 | The audit's second stage launched room by room as its inputs were ready, not together, so as to share the machine with another long run |

## TASK REVIEW (2026-11-16)

Task Duration: 01:38 (from 10:30 to 12:08 EST)
Usage Cap Triggered: NO

| Result | |
| --- | --- |
| ✅ | Every frozen claim reproduces, blind, in all five blocks: A1 to A5, B1 to B5, C1 to C5, D1 and E1, 17 of 17 groups, two of them identities that hold |
| ✅ | Six solver rooms offline; Krawczyk certificates at all six boxes; C5's 19 slow groups within `3.6·10⁻¹²` of the script's values against the rule's `10⁻⁹` |
| ✅ | Two audit rooms, each computing its share by its own route before seeing a room's work: no disagreement with any value, no defect in one |
| ⚠️ | Defects in the rooms' checks and text only: a false margin statement (`r2`), a check called independent that is not (`ca`), PASS lines that cannot fail; every affected value was confirmed by an audit route that can fail |
| ⚠️ | `\|Γ_u\| = 24` rests on two numerical searches and a maximality argument, not on 24 elements built exactly |
| ⚠️ | One reading decides scored values: `R₂`'s fibre group taken as the continuous `Sp(1)` the worklist names and the claims file's `k` rule uses |

Issues: none blocking. The doc checker passes on the designer's documents; its remaining findings are dashes and table pipes inside the verbatim room returns and the as-run briefs, which stay byte-identical to what the rooms produced and received.

Deviations from plan: seven, in the deviations log above. Separately, an adversarial audit of this record found 21 items, one false and the rest overstated or incomplete, all corrected before the record was presented.

Action needed: none outstanding on the platform side. The author's dated note on MIT's side is the author's, as the obligations table states.

Scope: a blind reproduction of the pinned PDF's frozen values, version DOI `10.5281/zenodo.24224845`. No theorem; C5's values are the script's; Block E on `E₀` only; no `MODELS.md` icon moves, and M8.7's gate is unchanged.

### Findings

Every frozen claim of the Standing-Wave Branches reproduces, blind, in all five blocks, on six offline rooms and a two-stage adversarial audit that found no disagreement, and no claim was found defective. Every discrepancy the run produced was in an instrument, a room's check or text or the designer's own comparison, never in a value the paper states.

### Research docs created/updated

- [`tasks/m8_1_3_task_details.md`](m8_1_3_task_details.md) (this file): go-time pre-registration, findings, the adjudication, the audit, the deviations log
- [`findings/m8_1_3_method_note.md`](../findings/m8_1_3_method_note.md): the author-facing record
- [`scripts/m8_1_3_l6/`](../scripts/m8_1_3_l6/), [`scripts/m8_1_3_r2/`](../scripts/m8_1_3_r2/), [`scripts/m8_1_3_cert/`](../scripts/m8_1_3_cert/), [`scripts/m8_1_3_ca/`](../scripts/m8_1_3_ca/), [`scripts/m8_1_3_cb/`](../scripts/m8_1_3_cb/), [`scripts/m8_1_3_rig/`](../scripts/m8_1_3_rig/), [`scripts/m8_1_3_audit_a/`](../scripts/m8_1_3_audit_a/), [`scripts/m8_1_3_audit_b/`](../scripts/m8_1_3_audit_b/), [`scripts/m8_1_3_maintainer/`](../scripts/m8_1_3_maintainer/)
- [`m8_1_3/as_run/`](../m8_1_3/as_run/), [`m8_1_3/m8_1_3_claims.json`](../m8_1_3/m8_1_3_claims.json)
- [`m8_roadmap.md`](../m8_roadmap.md), [`m8_theory_canonical.md`](../m8_theory_canonical.md), [`__M8_model_briefing.md`](../../__M8_model_briefing.md), [`../theory/_CITATIONS.md`](../../theory/_CITATIONS.md)
- [`dev_docs/CLEAN_ROOM_STANDARDS.md`](../../../../../dev_docs/CLEAN_ROOM_STANDARDS.md) § 3.4: the third shell shape the client admits, observed in every room of this run
