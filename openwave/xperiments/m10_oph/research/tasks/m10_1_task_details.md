# M10.1: THE CERTIFICATION GATE, blind recompute of the twelve-direction dispersion

> Roadmap row: [`../m10_roadmap.md`](../m10_roadmap.md) M10.1 (maintainer-run). Spec
> of record: [`../m10_theory_canonical.md`](../m10_theory_canonical.md). Task source:
> the author's own offer in the application
> ([discussion #712](https://github.com/openwave-labs/openwave/discussions/712), Help
> wanted) and the author's definition-only statement,
> [`DEFINITION_ONLY.md`](https://github.com/FloatingPragma/observer-patch-holography/blob/3c4e1e23fc32dd241e8c2bfba75de123fe5c6984/code/openwave_candidate/certification/DEFINITION_ONLY.md)
> at the pinned commit.

## TASK PLANNING (2026-11-13, registered at scaffold; not run)

### Scope

An independent recompute of one bedrock mathematical claim of the M10 column: the
small-momentum expansion of the scalar dispersion defined on the twelve
icosahedral directions. It is the column's certification gate
([`ONBOARDING_MODELS.md`](../../../../../ONBOARDING_MODELS.md) § 3.2 step 5): not a
new result, a check that one existing set of numbers holds up, by someone who has
not seen them.

What it certifies, and what it does not: arithmetic on a declared mathematical
object. The author's definition attaches no physical field and no physical lattice
scale to it. **No `MODELS.md` cell moves on any outcome.** The author named a
second, load-bearing candidate (the hadron fit with its held-out failures); that
one is open to any contributor and is listed in the briefing's Help Wanted.

### The object (the author's definition, restated)

Let `φ = (1 + √5)/2` and take the twelve unit vectors `u` obtained by normalizing
`(0, ±1, ±φ)`, `(±1, ±φ, 0)`, `(±φ, 0, ±1)`, with independent signs. For a positive
length `a`, a unit direction `n` and a scalar momentum `k`:

```text
ω²(k, n, a) = Σ_u [1 − cos(a k (u · n))] / (2 a²)
```

Every direction is used once, with equal weights. `P6` is the degree-six Legendre
polynomial, and `I6(n)` is `Σ_u P6(u · n)` divided by its value when `n` is one of
the twelve unit vectors.

### What is computed (the author's five requests)

| # | Output |
| --- | --- |
| 1 | The exact homogeneous quadratic, quartic and sextic polynomials of the expansion in `k` |
| 2 | Their rotationally invariant parts, and the first nonconstant angular part |
| 3 | The coefficients in the stated `I6` normalization, and their scale-free ratios |
| 4 | The change under `a → a/2` |
| 5 | An independent direct-symbol numerical check over several directions and decreasing `k`, at 60 decimal digits or more, with a bound on the Taylor remainder |

### The independence protocol

Run under [`CLEAN_ROOM_STANDARDS.md`](../../../../../dev_docs/CLEAN_ROOM_STANDARDS.md).

| Role | Sees | Does not see |
| --- | --- | --- |
| Designer (a maintainer) | this task document, the definition above, and the application, which describes the frozen rows in words | the author's frozen numerical values: the prediction register, the ladder page and the comparison file were not opened during the onboarding evaluation, and stay unopened until both agents' outputs are saved |
| Solver agent | a self-contained packet: the definition, the five requests, the acceptance rules | the author's values, the author's repository, the author's name, and every document of this repository |
| Audit agent | the same packet, plus the solver's script and outputs | the same |
| Comparison with the author's values | done by the designer after both outputs are saved and committed | |

### Pre-registered checks (fixed before any numerics)

| ID | Check | Pass | Fail |
| --- | --- | --- | --- |
| C1 (the gate) | The solver's exact quadratic, quartic and sextic polynomials equal the audit agent's, derived by a different method | exact agreement, coefficient by coefficient | any coefficient differs |
| C2 | The direct-symbol check: the exact series reproduces `ω²` at 60 digits or more over several directions and decreasing `k`, inside a stated Taylor-remainder bound | every sample inside its bound | any sample outside |
| C3 | The scaling under `a → a/2` equals what the definition implies term by term | exact | any term off |
| C4 (the control) | Removing one direction, or changing one weight, is detected by the isotropy test | the altered operator is rejected | the altered operator passes |
| C5 (the comparison) | After C1 to C4 are saved: each coefficient and ratio against the author's frozen values | reported as agree or disagree per value, both numbers shown | not a pass or fail of this task: a disagreement is a result |

No-search rule: every computed number is reported, and nothing is chosen by
comparison with any target.

### Definition of done

| # | Item |
| --- | --- |
| 1 | Solver script and outputs saved, with a statement of which files were read |
| 2 | Audit agent's independent derivation and its per-check verdicts |
| 3 | Outputs committed before the author's values are opened; then the C5 comparison |
| 4 | Method note `findings/m10_1_method_note.md`: equations first, equation-to-code map, the audit record |
| 5 | Roadmap row moved to DONE; the briefing's status row updated; no `MODELS.md` cell touched |

### Blindspot pass (what could go wrong, checked in advance)

| Blindspot | Mitigation |
| --- | --- |
| The designer learns a frozen value before the run | the register, ladder and comparison file stay unopened; the canonical's falsifier table withholds the dispersion values until this task has run |
| The two agents share a method, so C1 passes by construction | the packet asks the audit agent for a different route (for example invariant theory of the icosahedral group against a direct Taylor expansion) |
| A normalization convention is read differently from the author's | the `I6` normalization is in the definition; C5 reports both numbers, so a convention gap is visible and is routed to the author as an author-gated question |
| The control cannot fail | C4 is run on a deliberately broken operator, and its rejection is part of the record |
| A value reaches the room through this repository | the solver and the audit agent are told to read no repository document |

## RUN RECORD

### Ordering, declared before launch (2026-11-13)

Three commits on one branch, each pushed before the step it gates.

| # | Commit | Pushed before |
| --- | --- | --- |
| 1 | the packet, the packet check and its output, and this section | the rooms open |
| 2 | the commitment: both rooms' returns, their hashes, the environment records, the manifests and the transcript checks | the author's values are opened |
| 3 | the adjudication: the comparison with the author's values, the method note, the roadmap row | the merge |

**Deviation from [`CLEAN_ROOM_STANDARDS.md`](../../../../../dev_docs/CLEAN_ROOM_STANDARDS.md) § 14, declared here.**
The standard has each of the three commits merged before the next step. Here each
is pushed to the pull request's branch and the branch merges once; the push times
on the branch carry the same ordering. The reason is proportion: this run
certifies arithmetic on a declared object and moves no cell.

### The packet

Four files under [`../m10_1/packet/`](../m10_1/packet/), checked by
[`../scripts/m10_1_packet_check.py`](../scripts/m10_1_packet_check.py)
(output: [`../data/m10_1_packet_check.json`](../data/m10_1_packet_check.json)).
The check scans for withheld names and outcome words, admits only the integers the
definition needs, rejects any decimal literal, and is mutation-tested: five planted
violations, five caught.

| File | SHA-256 | Bytes |
| --- | --- | --- |
| `TASK.md` | `bb5563f1cacea4240bcb0fc57ead7bce92eca649b52e4db6a61d35f1dea80ebe` | 4416 |
| `TASK_AUDIT.md` | `715bf6c2e64028590478cc47d7adba28c3df981c01d4f8227cdb4e435df151f7` | 4450 |
| `OPENING_PROMPT.md` | `95661eee4f73de9e810aa1ad829cb4f9ddf38c51fd01a24e9f5fbbd3f1dc22cd` | 140 |
| `OPENING_PROMPT_AUDIT.md` | `d4ea55e1c9d883f3e5077eafefcaeb71aa56108982ea49202c5a15e6bf620f2e` | 146 |

**Withholdings, stated.** The author's definition-only statement names the
framework and its result files. The packet restates the mathematics without them:
it names neither the framework, its author, this platform, the registered rows,
nor the solid the twelve directions belong to. The audit room receives the solver
room's returns in a `solver/` folder, added at its launch, and nothing else.

**The two routes.** The solver room is told to use a direct symbolic series
expansion. The audit room is told to use a different route (moments of the
direction set, or a fit to high-precision evaluations) and to save its own results
before it opens the solver's.

### Launch

Headless, per the standard's § 3.4, with
[`clean_room_launch.sh`](../../../../../dev_docs/utils/clean_room_launch.sh): one
room per stage, outside every working tree, the room interpreter carrying `sympy`
and `mpmath` and unable to import the platform package. The rooms' records are
added with commit 2.

### Commit 2: the commitment (2026-11-13)

The rooms ran after commit 1 (`f55868ab`). Their returns, the checks C1 to C4,
the deviations, the transcript checks and every hash are in
[`../findings/m10_1_commitment.md`](../findings/m10_1_commitment.md); the
returned files are under [`../m10_1/returns/`](../m10_1/returns/). The author's
values were still unopened when this section was written.

| Check | Verdict at commit 2 |
| --- | --- |
| C1 | ✅ three derivations, two routes, exact agreement ([`../scripts/m10_1_compare_rooms.py`](../scripts/m10_1_compare_rooms.py)) |
| C2 | ✅ the solver's 20 samples and 32 further samples all inside the stated bound |
| C3 | ✅ exact |
| C4 | ✅ four altered operators rejected; the test reads degree two only |
| C5 | pending, commit 3 |

Two audit rooms stopped before their step two, for a cause in the packet's
layout; the step-two checks were done outside the rooms by an independent audit
agent. Both are logged as deviations in the commitment.
