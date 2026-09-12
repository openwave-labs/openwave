# M8.5: Quotient-manifold simulation engineering

> Roadmap row: [`../m8_roadmap.md`](../m8_roadmap.md). Status: CLOSED 2026-08-31, row
> retired to Done 2026-09-11. Its gate M8.2 closed ✅ 2026-07-27. **M8.5-A** ✅ REPRODUCED
> 2026-07-31; **M8.5-B** ✅ COMPLETE 2026-08-17 (sealed case `M85B-ADJ-07`, GREEN at both
> rungs); **M8.5-C** ❌ closed 2026-08-31: its first attempt terminated non-adjudicated
> ([#501](https://github.com/openwave-labs/openwave/issues/501)), and the successor
> adjudicated `M8.5-C2-FAILED`, instrument-attributed
> ([#506](https://github.com/openwave-labs/openwave/issues/506),
> [addendum 1](../findings/m8_5c2_protocol.md)). The PLANNING text below is preserved as
> filed and is no longer a live work order. This is a scaffold-stage planning aid written
> by the maintainers (2026-07-21, § "Independent reproduction" added 2026-07-28); the
> author owns the column and may amend everything here.

## PLANNING

### Scope

Two sub-deliverables that the M8.2 lock put in the same task, named and tracked separately
below because they have different owners and different independence rules. They stay inside
M8.5 by agreement (author, [PR #350](https://github.com/openwave-labs/openwave/pull/350)
close-out 2026-07-28), and split into their own task IDs only if ownership or scheduling
diverge enough to make the roadmap unclear:

| # | Sub-deliverable | Owner |
| --- | --- | --- |
| M8.5-A | Independent table and decomposition reproduction (lock § 3) | protocol: author; implementation: maintainers |
| M8.5-B | The quotient backend: the simulation engine for the arena (routes a/b below) | author + platform support |

**A gates any claim that B is certified.** B can be built and can run before A closes; what
it cannot do before then is carry a certified-against-the-McKay-structure claim.

Build and certify the simulation infrastructure M8.4 needs: fields evolving on the
compact quotient S³/2I. No existing OpenWave column runs a curved compact arena, so
this is genuinely new platform ground. Two candidate routes
([`../m8_platform_pointers.md § 6`](../m8_platform_pointers.md)); prototype BOTH far
enough to choose one on evidence.

| Route | Sketch | Known risks |
| --- | --- | --- |
| (a) 2I-equivariant grid | an S³ grid (embedding or intrinsic charts) with the 120-element identification imposed as an equivariance/ghost-cell map | the identification map bookkeeping; chart seams; where the Möbius edge / cone structure of the MIT arena lives on the grid |
| (b) Spectral in 2I-symmetric harmonics | expand fields in S³ harmonics restricted to 2I-invariant (or covariant) subspaces; evolve coefficients | nonlinear terms need convolution handling (cost grows fast with band limit); but the basis IS the McKay representation theory, so slot structure is manifest |

**Route (b) is M8.5-C; its qualification protocol is FILED (2026-08-28), frozen at merge.**
After M8.4 closed unresolved with no target configuration executed and no nontrivial sector
spent, route (b) was chosen to be built as an actual dynamics substrate, target-free by
construction, under its own roadmap row. The protocol is
[`../findings/m8_5c_protocol.md`](../findings/m8_5c_protocol.md), frozen region SHA-256
`e253558b5a767084d4d7777550ac72de5b8a0591ec3d2b847108f04e17c0cc6b`, with the symmetry derivation note beside it, the five
design-input scripts under [`../m8_5c/design_inputs/`](../m8_5c/design_inputs/), and the executed § 14
pre-freeze package under `../m8_5c/s14/`. Attempt A1 was commissioned and TERMINATED
WITHOUT ADJUDICATION 2026-08-29: gate 4's frozen node-drop arm is mathematically dead on
the protocol's parity-pure arenas, the unit stopped with no gate-4 verdict, and the ruling
([#501](https://github.com/openwave-labs/openwave/issues/501)) granted a once-only
post-execution supersession under a new identity; the archived run is
[`../m8_5c/a1/`](../m8_5c/a1/) and the canonical account is the protocol's addendum 1.
The successor M8.5-C2 is FILED 2026-08-29 and frozen at merge: the protocol is
[`../findings/m8_5c2_protocol.md`](../findings/m8_5c2_protocol.md), frozen region SHA-256
`aadf4d9218bc36cf4763b42ce587725c16f673b3a9e4272f19a33b6ec0fbd5ef`, DERIVED from the
ratified C frozen region with the diff and register filed under
[`../m8_5c2/derivation/`](../m8_5c2/derivation/) and the executed § 14 package under
[`../m8_5c2/s14/`](../m8_5c2/s14/). The C2 Build Unit was commissioned per § 12; attempt
C2-A1 terminated 2026-08-30 on the author's halt order at gate 4 after unit misconduct
(seven output-ledger deletions, in-attempt instrument edits including one mid-run, two
files written outside § 12's declared write areas). The room was filed byte-identical
under [`../m8_5c2/a1/`](../m8_5c2/a1/)
([#508](https://github.com/openwave-labs/openwave/pull/508)), and the ruling
([#506](https://github.com/openwave-labs/openwave/issues/506)) adjudicated
`M8.5-C2-FAILED`, instrument-attributed STOP-QUAL: the frozen § 1 FAILED sentence issues,
the spectral route closes, and the M8.4 reopening path closes with the verdict. The
canonical account is the C2 protocol's addendum 1. See also
[`../findings/m8_4_closeout.md`](../findings/m8_4_closeout.md).

### Independent reproduction, M8.5-A (added 2026-07-28)

The M8.2 lock § 3 puts an obligation on this task:
[`../findings/m8_2_preregistration.md`](../findings/m8_2_preregistration.md) requires every
certification table to be reproduced through an INDEPENDENTLY implemented decomposition,
which may compare against
[`../scripts/m8_2_first_occurrence.py`](../scripts/m8_2_first_occurrence.py) but may not call
it, import its tables, or share its derived fixtures. The author added the procedure for it in
the [PR #350](https://github.com/openwave-labs/openwave/pull/350) close-out thread:

| Step | What | Who |
| --- | --- | --- |
| 1 | A frozen reproduction protocol: context firewall, operator conventions, result categories, mutation-tested gates, provenance requirements | the author, submitted as a document before any implementation exists |
| 2 | Implementation of that protocol in a FRESH context with no M8.2 internals loaded | maintainers |

**Step 1 ✅ LOCKED 2026-07-30**: [`../findings/m8_5a_reproduction_protocol.md`](../findings/m8_5a_reproduction_protocol.md),
author-written and filed before any implementation existed, landed through
[PR #380](https://github.com/openwave-labs/openwave/pull/380). It is the binding spec for step 2;
where this planning doc and the protocol differ, the protocol governs. Step 2 is startable.

Why the roles are this way round: the context that produced M8.2 holds the target tables and
the derived fixtures, so it cannot serve as its own reproducer no matter how separately the
second implementation is written. The author raised this rather than being asked.

**The claim ceiling is independent-method reproduction, not blind.** Blind means the verifying
agent has not seen the claimed values
([`ONBOARDING_MODELS.md § 3.2`](../../../../../ONBOARDING_MODELS.md#32-the-maintainer-sequence)
step 5), and that is already spent here twice over: the author's context built the tables, and
the maintainer reproduced all nine rows by explicit quaternions plus Burnside class-sums during
the PR review. An independent method still carries real weight, and the write-up says which one
it is. Do NOT let the word "blind" into the M8.5 deliverable.

#### The three objects, and what M8.5-A's implementer may read

The protocol distinguishes three objects rather than two, because the review-time verification
is now a repository artifact rather than a throwaway (landed 2026-07-28 at the author's request):

| # | Object | Method | Quarantine status for M8.5-A |
| --- | --- | --- | --- |
| 1 | [`../scripts/m8_2_first_occurrence.py`](../scripts/m8_2_first_occurrence.py) | McKay / affine-E8 recursion; irrep labels, dims, distances as literals | FORBIDDEN until step 2's source + raw output are committed |
| 2 | [`../scripts/m8_2_indep_reconstruction.py`](../scripts/m8_2_indep_reconstruction.py) + [note](../findings/m8_2_indep_reconstruction_note.md) | explicit quaternions → conjugacy classes → Burnside class-sums; dims, adjacency, distances derived | FORBIDDEN until step 2's source + raw output are committed |
| 3 | the M8.5-A implementation | the author's frozen protocol, fresh context | the deliverable |

After 3 is committed, 1 and 2 become **adjudication references only**. Agreement of all three
is then reported as **three-way agreement**: it strengthens provenance, and it does NOT move the
claim label off independent-method reproduction.

**What object 2 does not cover** (the author's scope point, 2026-07-28): it reconstructs 2I, its
characters, the McKay distances and the scalar (0-form) first-occurrence table. It does not derive
the coexact one-form entry rule, which stays **ASSERTED**.

**That standing question is answered** (protocol § 6, 2026-07-30). The coexact rule is not a
certification target: it appears only as a separately labeled ASSERTED adjudication module,
optional and pre-declared in the § 9 commitment, with four pre-declared verdicts. The hard rule
is that numerical agreement in any amount never upgrades the rule's standing and is never
reported as independent verification; only a general argument from the operator and
representation structure does, and a pattern interpolated from the computed range does not
qualify however clean the fit.

Object 2 was landed to the requirements the author set for it: source, environment, raw output,
the commit verified against, a short method note, a mutation test for every PASS line, and an
explicit statement of what it does and does not verify.

#### G11 and G12: now part of the frozen floor (addendum 1, 2026-07-30)

Raised in the [PR #380](https://github.com/openwave-labs/openwave/pull/380) review, accepted by
the author into the § 8 minimum set as
[addendum 1](../findings/m8_5a_reproduction_protocol.md#11-addenda-post-freeze-only), landed
via [PR #385](https://github.com/openwave-labs/openwave/pull/385). The protocol is the binding
statement of both gates; this section records only what the exchange decided and corrected.

The gap they close: the original G1-G10 cover the 2I group theory completely and map onto six
of object 2's eight mutation-tested checks, but not the two constructions between the group and
the answer, `τ_σ = Sym²(σ)` and the `V_n` tower. Both corresponding defects
(`sym2_as_square`, `chiv_offbyone`) pass G1-G10 and surface at § 7 as **partial disagreement**
rather than structural failure, so a correct target table reads as a failed reproduction, the
expensive direction.

Three things settled in the exchange, all author-side improvements on the proposal:

| Point | Resolution |
| --- | --- |
| Floor vs implementer-added | into the floor, on scope rather than severity: an implementer-added gate binds one implementation, the floor binds every implementation and every rerun |
| The consumed-object requirement | both gates evaluate THE SAME constructed object the § 5.4 calculation consumes, never a parallel recomputation, with group squaring from the raw multiplication. As first proposed, G11 could have been satisfied by rebuilding an ideal `Sym²σ` and comparing the identity to itself, `m7_trivial_ok` in another costume |
| The dimension-only claim, corrected | the proposal overstated it as "does not discriminate". The addendum has it right: a dimension check catches the defect in the nontrivial 2-dimensional columns (3 against 4 at the identity) but not in the trivial column, where `Sym²(σ)` and `χ_σ²` agree, and it never verifies the classwise construction |

One preference carried into `TASK.md` rather than into any frozen text: build the symmetric
square explicitly and take traces, rather than evaluating the character identity G11 checks.
Where the two routes genuinely differ, G11 is a cross-check; where they coincide, it confirms
the declared contract was applied but not its arithmetic. The route taken is disclosed in the
method note either way.

#### Operational items, settled (2026-07-30)

| Item | Decision |
| --- | --- |
| Clean-room location | OUTSIDE the repository tree, in a directory with no `CLAUDE.md` on any ancestor path and no git relationship to openwave. A git worktree does NOT qualify: it lives inside the tree, so an agent session started there auto-loads `CLAUDE.md` and its M8 pointers on the first turn, before any instruction is read |
| Who commits | a maintainer copies the artifacts out of the clean room and performs the § 9 commit. The consulted-files manifest is generated inside the clean room and copied verbatim, never re-authored outside it |
| Generic references | no web access, the first of the two routes § 4 allows. Generic finite-group and `SU(2)` representation theory comes from the implementer's own knowledge, and the manifest states that no external references were consulted, explicitly rather than by omission. The allowlist route is declined because any query specific enough to help here (`2I` harmonic analysis, the McKay correspondence) returns the answer key, and an allowlist would need a per-URL approval trail proving exclusion. If § 6's module cannot be built without a reference the implementer lacks, that is a `not resolved` verdict for the module, not a reason to open access mid-run |
| The § 7 comparison harness | a second, separately dated commitment, written after the § 9 commitment is filed and § 6.1 is unsealed. Its commit names the § 9 commit it postdates. It may not modify object 3's source or raw output; any change to those is a dated rerun under § 3 |
| Implementer eligibility | any context that has read object 1, object 2, or the pre-registration § 6.1 is permanently disqualified, whatever else it later forgets. That covers every maintainer session that has handled M8.2, M8.5, or this task document |

**The packet's group input is not an open decision.** § 4 permits raw generators, and § 5.3
identifies `Q` from the elements themselves through `χ_Q(g) = 2cos θ(g)` rather than by
declaration, so the element set fixes the `standard` and `galois` column assignment with no
free parameter left to whoever hands it over. The standard 120 icosians (8 units, 16 Hurwitz,
96 golden-set even permutations) are what object 2 built, and its C4 and C8 confirm that set
agrees with § 6.1 on all 9 rows. What remained was provenance rather than correctness, and the
author settled it in [PR #382](https://github.com/openwave-labs/openwave/pull/382)
(2026-07-30): **the author supplies the raw embedded generators; the maintainer audits,
canonicalizes, and hashes the packet.** The audit stays a cross-check rather than a self-audit,
and the packet hash is an addition beyond § 4, which requires only the consulted-files
manifest. Canonicalization is what makes the hash meaningful, since the components are
irrational in `φ` and an uncanonicalized decimal rendering hashes differently for the same
group. The maintainer's own pre-verified pair is retained as a cross-check on the supplied
generators, never as the packet.

**Delivered 2026-07-31** ([PR #386](https://github.com/openwave-labs/openwave/pull/386)
thread), in two deliberately separate archives: the packet, and the author's own verification
evidence marked to stay outside the room. The generators are exact in `Q(φ)` with every
component written `(a + b·φ)/2` over integer `a` and `b`, so no decimal rendering exists to
canonicalize away and the hash pins a group rather than a transcription. The author's
separation of packet from evidence was unprompted, and it is the containment the protocol
wanted; the audit still runs from the packet alone, per
[`../../../../../dev_docs/tasks/t4_task_details.md`](../../../../../dev_docs/tasks/t4_task_details.md).

**Landing map for the block A packet commit** (committed before the room opens, so the
timestamp predates the run; contents subject to the block A leakage scan):

| Artifact | Repo destination |
| --- | --- |
| the canonical packet | `../data/m8_5a_packet.json` |
| the clean-room task file | `../data/m8_5a_cleanroom_task.md` |
| the clean-room group-input file | `../data/m8_5a_cleanroom_group_input.md` |
| the maintainer packet audit | `../scripts/m8_5a_packet_audit.py`, `../data/m8_5a_packet_audit.json` |
| the author's verification evidence, as received | `../data/m8_5a_packet_author_evidence/` |

**Audit result, 2026-07-31.** [`m8_5a_packet_audit.py`](../scripts/m8_5a_packet_audit.py) run
from the packet alone: eight checks green, seven mutations each reddening their target, exit 0.
Exact arithmetic throughout, no tolerance anywhere.

| Quantity | Observed |
| --- | --- |
| incoming SHA-256, as delivered | `e3b0c945bbbb15b4549fa641234c9461062c2337b3d1e372af621b614d4883a9` |
| canonical form | keys sorted, two-space indent, ASCII, LF, one trailing newline |
| authoritative SHA-256 | identical to the incoming hash: the delivered bytes were already canonical, so canonicalization is a no-op here |
| generator norms | exactly `1` in `Q(φ)`, both |
| generator orders | 6 and 4 |
| closure | finite, multiplicatively closed, order exactly 120 |
| center | exactly `{+1, −1}` |
| element-order census | 1, 1, 20, 30, 24, 20, 24 at orders 1, 2, 3, 4, 5, 6, 10 |
| central quotient | order 60, profile 1, 15, 20, 24 at orders 1, 2, 3, 5 |
| leakage scan | no label, dimension, distance or character vocabulary; no key outside the declared set |

The element-order census is the audit's addition rather than a restatement of the author's
gates. A center of order 2 over a quotient with the `A₅` profile is also satisfied by the
direct product `A₅ × C₂`, which is not the binary icosahedral group; the order-4 population
separates them, 30 against none. A finite subgroup of the unit quaternions cannot be
`A₅ × C₂` for an independent reason, since `−1` is the only unit quaternion of order 2 while
that product has 31 involutions, so the census is not the only thing standing between the two.
It is the mechanical version of that argument, and this audit prefers a count to an appeal to a
theorem.

**Cross-check against the author's evidence, run only after the above was complete.** The
author's report states the same packet hash, the same closure order, and its four gates true.
The two runs agree, and they are not fully independent in method: both work exactly over
`Q(φ)` through `fractions.Fraction`, which the packet format effectively forces. The checks
differ, which is where the value is: the element-order census and the leakage scan are the
maintainer side only, and the author's report carries an environment record the audit does not.

**The room, as sealed before launch.** Four files, nothing else:

| File | SHA-256 | Provenance |
| --- | --- | --- |
| `PROTOCOL.md` | `f7370de24ca26f78c4019bf30db30abe54810f83198f9bded39fd0c25e10bf96` | `diff`-identical to [the frozen protocol](../findings/m8_5a_reproduction_protocol.md) |
| `m8_5a_packet.json` | `e3b0c945bbbb15b4549fa641234c9461062c2337b3d1e372af621b614d4883a9` | the author's packet, byte-identical |
| `GROUP_INPUT.md` | `8b5fb2a61cdfe82cfe90f6c57e1e02f283229b1dd30470c331774dc3fb12f837` | maintainer-written wrapper; names the packet, states its hash, restates the packet's own coefficient-format field, quotes the packet verbatim |
| `TASK.md` | `c662fd06c334809f0f4cdeda0903967536564dae6d597d0db308257aa34210c8` | maintainer-written, operational only |

**The group order is withheld, which is stricter than the protocol requires.** Protocol § 4
lists the order, 120, among the permitted construction inputs. `GROUP_INPUT.md` does not supply
it, so G1 stays a check that can fail rather than a restatement of something handed over. The
withholding is stated in that file, so the implementer knows it is deliberate and does not
spend the run wondering whether the packet is incomplete.

**Landing map for the § 9 commitment** (fixed before the run so the copy-out is mechanical;
one commit, nothing unsealed until it lands):

| Clean-room file | Repo destination |
| --- | --- |
| `m8_5a_reproduction.py` | `../scripts/m8_5a_reproduction.py` |
| `raw_output.txt`, `result.json` | `../data/m8_5a_raw_output.txt`, `../data/m8_5a_result.json` |
| `environment.md`, `manifest.md` | folded into `../findings/m8_5a_commitment.md`, one file carrying the environment record, the consulted-files manifest, the raw output's SHA-256, the schema version, and the § 6 declaration (the module RUNS) |
| `method_note_draft.md` | `../findings/m8_5a_method_note.md`, marked DRAFT until the adversarial audit is recorded in it |

The § 7 comparison harness is NOT in this commit: it is written after unsealing, lands as
`../scripts/m8_5a_adjudication.py` in a second, separately dated commit that names the
commitment commit it postdates.

**Planned § 9 declaration: the § 6 coexact module RUNS.** Fixed at the commitment, before
anything is unsealed, and unavailable afterwards. The cost is the implementer deriving the
coexact one-form tower by its own harmonic analysis. The return is that
`structurally derived and reproduced` is the only route that ever moves the rule off ASSERTED,
and § 6's standing rule already blocks a bare numerical match from being oversold. Declining
leaves the rule ASSERTED with no route open.

### The certification benchmark (fix before building)

Certify each prototype on a problem with a KNOWN answer, not on the target problem:
the free Laplacian on S³ has eigenvalues `l(l+2)/R²` with known multiplicities, and on
S³/2I the multiplicities restrict by 2I-invariance (computable independently by
character theory). A prototype that reproduces that spectrum + multiplicity pattern is
certified; one that cannot is refuted before any physics rides on it. This mirrors the
M8.1 gate philosophy one level up.

### Suggested definition of done

| # | Item |
| --- | --- |
| 1 | Both prototypes pass the certification benchmark (spectrum + multiplicities), scripts + JSON in the repo |
| 2 | Trade-off table measured, not argued: accuracy vs cost vs implementation complexity at matched resolution |
| 3 | Route decision recorded with its rationale; the losing prototype kept as the cross-check tool for M8.4 |
| 4 | Prototypes are research scripts (NumPy/SciPy fine); Taichi-first applies only if/when this graduates to production per-frame kernels |

### Blindspots

| Risk | Guard |
| --- | --- |
| Certifying on the target problem (circular) | the benchmark is fixed above, with an independent character-theory multiplicity check |
| Silent symmetry breaking by the grid (route a) | measure the certified spectrum's degeneracy splitting as the resolution ladder climbs; report it |
| Band-limit truncation masquerading as physics (route b) | convergence in the band limit reported for every observable |
| M8.5-A written up as blind, or A's implementer reusing the M8.2 context | the claim ceiling and the firewall are stated above; the protocol (step 1) fixes both before step 2 opens |
| A's implementer reading either existing artifact early | both are named and quarantined above; they open only after A's own source + raw output are committed |
| M8.5-B claiming certification while A is open | A gates that claim, stated in § Scope; B may run, it may not claim |
| A PASS line that cannot go red | mutation-test every gate before it ships ([roadmap § CONVENTIONS](../m8_roadmap.md#conventions); the M8.2 defect) |

### Ownership + gating

M8.5-B is author-driven with platform support. M8.5-A splits: the author writes the
protocol, the maintainers implement it. Gated by M8.2 ✅ (so the engine is built against
locked requirements, not drifting ones); A's step 2 additionally waits on A's step 1, and
any certification claim for B waits on A.

## DEVIATIONS LOG

**2026-07-31, M8.5-A run.** Procedure-level deviations (the stalled permission prompt, the
environment catch, the raw-output path redaction, the scratchpad writes) are logged in
[T4's deviations log](../../../../../dev_docs/tasks/t4_task_details.md) and disclosed in
[`m8_5a_commitment.md`](../findings/m8_5a_commitment.md) § 4-6. No protocol-level deviation:
no mid-run relay, no packet amendment, no unsealing before the commitment.

**2026-08-01. The withheld group order was intentional on both sides, so nothing is recorded
as a deviation.** The block A note asked the author whether omitting it from the packet was
meant, offering to supply 120 and log the change
([#386](https://github.com/openwave-labs/openwave/pull/386) thread, answered on
[#394](https://github.com/openwave-labs/openwave/pull/394#issuecomment-5152324680)). The
author's answer: the packet was limited to the generators so that G1's order result stayed a
derived check rather than a restatement of supplied metadata. The stricter input condition was
the intended one, and the run met it.

## FINDINGS

**M8.5-A adjudication, 2026-07-31: REPRODUCED.** The clean-room implementation
([commitment](../findings/m8_5a_commitment.md) at `dac2b6a1`, merged `c3dc2b5f`) matches the
pre-registration § 6.1 table at `ec877ee0` exactly: 9 label-free `(dim, distance)` signatures
pairwise distinct on both sides, 27 cells equal under exact integer comparison, no tolerance.
Three-way agreement holds: the clean-room result, § 6.1 (object 1's published table), and
object 2's reconstruction agree cell for cell. Harness:
[`m8_5a_adjudication.py`](../scripts/m8_5a_adjudication.py) (its § 6.1 transcription made
independently of object 2's `DOC` fixture, label-to-dim map from object 1's literals, checked
against the transcribed distances before use); record:
[`m8_5a_adjudication.json`](../data/m8_5a_adjudication.json). Both transcription mutations
(`doc_typo` on the target, its mirror on the candidate) redden the comparison, per G10.

| Item | Outcome |
| --- | --- |
| scalar table | REPRODUCED, 27/27 cells, three-way agreement |
| claim label | context-isolated independent-method reproduction, the § 2 ceiling, nothing stronger |
| § 6 coexact module | ran as pre-declared; implementer verdict `structurally derived and reproduced`, echoed not adjudicated; the derivation goes to the block E adversarial audit, and per the standing rule the numerical match upgrades nothing on its own |
| what remains for M8.5-A | ✅ none: block E closed 2026-08-01, see below |

**M8.5-A block E, 2026-08-01: the audit refuted nothing, M8.5-A is complete.** The method
note is finalized ([status block + §§ I-J](../findings/m8_5a_method_note.md): §§ A-H stay
byte-frozen as committed, § I records the adjudication, § J records the audit). The
adversarial audit ran as an independent second agent briefed to refute, with its own group
construction (explicit icosian list checked equal to the packet closure), its own character
table (Burnside class-algebra splitting, the route the implementation deliberately avoids),
and its own tables ([`m8_5a_audit.py`](../scripts/m8_5a_audit.py),
[`m8_5a_audit.json`](../data/m8_5a_audit.json), exit 0, re-run green by the maintainer).

| Audit outcome | Detail |
| --- | --- |
| eight claims attacked, none refuted | both lemmas, the rule table including the `d = 1` case, the coexact tower and its `m²/R²` normalization by an independent Casimir route, the trivial-column scope reading verified faithful against every source statement, realness, peeling completion, and a full 54-cell run-record cross-check |
| two claims strengthened | bipartiteness and realness are theorems here (central `−1`; inverse-closed classes), not merely computed witnesses |
| six unflagged weaknesses recorded | § J table: one informational (the Galois map sends the group to its twin icosian copy, set-stability unneeded but unstated), four minor exposition gaps, one cosmetic; none affects a result |
| standing consequence | `structurally derived and reproduced` stands on the derivation, per the § 6 rule; the numerical match contributed nothing |

M8.5-B (the quotient backend) is the live remainder of M8.5; its certification gate (A
before any claim that B is certified) is discharged.
