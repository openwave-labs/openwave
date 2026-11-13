# M10.1 commitment: the rooms' returns, before the author's values are opened

> Commit 2 of the M10.1 run ([task document](../tasks/m10_1_task_details.md), run
> record). It follows commit 1, the packet (`f55868ab`, pushed 2026-11-13 18:10 UTC),
> and it is pushed before the author's frozen values are opened. Everything on
> this page was produced without those values: the prediction register, the ladder
> page and the comparison file in the author's repository remain unopened.

## 1. What the rooms returned

Three clean rooms ran headless on 2026-11-13, each a fresh session with four
tools, no instruction file, no memory, no connector and no network, its code
confined to the room by the operating system's sandbox
([`CLEAN_ROOM_STANDARDS.md`](../../../../../dev_docs/CLEAN_ROOM_STANDARDS.md) § 3.4).

| Room | Packet file | Ran (UTC) | What it returned |
| --- | --- | --- | --- |
| solver | `TASK.md` | 18:10:58 to 18:12:41 | complete: script, results, report, manifest |
| audit A | `TASK_AUDIT.md` plus the solver's four files | 18:13:06 to 18:14:51 | step one only: its own derivation, saved before it opened the solver's files |
| audit B | the same | 18:15:29 to 18:18:12 | step one only: its own derivation, saved before it opened the solver's files |

The result the three derivations agree on, by two different routes (a direct
series expansion in the solver room, moments of the direction set in both audit
rooms):

```text
omega_squared = |q|^2  -  (a^2 / 20) |q|^4  +  (a^4 / 840) |q|^6 [ 1 + (16/75) I6(n) ]  +  terms of degree eight and higher
```

| Quantity | Value |
| --- | --- |
| Degree two | `\|q\|^2`, no remainder |
| Degree four | invariant coefficient `-a^2/20`, no remainder |
| Degree six | invariant coefficient `a^4/840`; remainder `(2 a^4/7875) \|q\|^6 I6(n)`; ratio `16/75` |
| First degree with a nonzero remainder | six |
| Under `a` to `a/2` | the degree-2m coefficient is multiplied by `2^-(2m-2)`: 1, 1/4, 1/16; the ratio does not change |
| Sum of `P6` over the twelve directions at one of them | `132/25` |

## 2. The checks, as pre-registered

| Check | Verdict | Evidence |
| --- | --- | --- |
| C1, the gate | ✅ holds | The three result files agree exactly, coefficient by coefficient, and each is internally consistent: [`m10_1_compare_rooms.py`](../scripts/m10_1_compare_rooms.py), output [`m10_1_compare_rooms.json`](../data/m10_1_compare_rooms.json). Three planted disagreements, three caught |
| C2, the direct-symbol check | ✅ holds | The solver's bound (`12 (a k)^8 / (2 a^2 8!)`) is stated in its source before the comparison and is a valid bound; all 20 of its samples are inside it. The stage-two audit ran 32 further samples at 70 digits on 8 directions the solver did not use: all inside, difference over bound from 0.079 to 0.140 |
| C3, the scaling | ✅ exact | On the solver's polynomials, `p(a/2) - 2^-(2m-2) p(a)` is identically zero at degrees two, four and six. A wrong factor (1/8 at degree six) fails |
| C4, the control | ✅ holds, with a limit | The solver's test accepts the original operator and rejects one direction removed and one weight doubled. The stage-two audit broke the operator two more ways (one direction rotated by an exact rational rotation, one direction replaced by a coordinate axis): both rejected. The limit: the test reads degree two only, so it passes six cube axes, whose degree-four sum is not invariant |
| C5, the comparison with the author's values | not run yet | It is commit 3, after this page is pushed |

**What C2 cannot do, recorded.** A degree-six coefficient changed by one part in
a thousand turns the numerical check red in only 3 of 32 samples, at the two
smallest steps. The exact comparisons C1 and C3 carry the coefficients; C2
confirms the series against the definition.

**Findings in the solver's script** (the stage-two audit read it line by line):
a duplicated precision line, harmless; the sphere average `1/(2m+1)` is typed in,
not computed, and is cross-checked by the vanishing remainders at degrees two and
four; no loose tolerance; a re-run of the script regenerates a `results.json`
identical in content to the one shipped.

## 3. Deviations, logged as they happened

| # | What happened | What was done |
| --- | --- | --- |
| 1 | Audit room A finished step one, opened the solver's folder by changing directory into it, and from then on every write was refused: the session's allow rules are written relative to the room, and no longer matched. It stopped and said so, without its step-two checks, its audit report or its manifest | Its step-one files are kept as returned. The room was not messaged and the packet was not amended |
| 2 | Audit room B, a fresh room on the same packet, stopped the same way at the same step | Kept as returned. No third room was launched: the cause is the packet's own layout (a subfolder the room steps into), so a third launch would repeat it |
| 3 | The step-two checks (C2, C3, C4 and the reading of the solver's script) were therefore not done inside a room | They were done by an independent audit agent outside the rooms, with its own script, given only the audit task file and the solver's four files, and told to refute. Its files are under `returns/`: [`STAGE_TWO.md`](../m10_1/returns/STAGE_TWO.md), `stage_two_audit.py`, `stage_two_audit.json`. C1 was decided by the maintainers' exact comparison of the three result files, which the pre-registration assigned to the audit agent |
| 4 | Audit room A listed the names of the files in `solver/` before step one, and opened none of them until its own results were saved | Disclosed. The transcript shows the order |
| 5 | Audit room A's last command, refused and not performed, would have redirected output to a throwaway file outside the room | Disclosed as the one attempted write outside a room. Nothing was written |

The rule this run adds to the clean-room practice: a packet that hands a room a
subfolder should tell it to read the files by path and stay in the room's
directory.

## 4. The transcript checks

Every tool call in each session's transcript was extracted and its paths
classified against the room.

| Room | Tool calls | Paths outside the room | Refused prompts | Order |
| --- | --- | --- | --- | --- |
| solver | 7 (1 read, 3 shell, 3 write) | none | 1: a shell redirection to create its script; it then used the write tool | not applicable |
| audit A | 8 (1 read, 5 shell, 1 write, 1 edit) | one, in a refused command (deviation 5) | 3 | its results were saved at call 5; the solver's files were first read at call 6 |
| audit B | 12 (1 read, 8 shell, 1 write, 2 edit) | none | 4 | its results were saved by call 8; the solver's files were first read at call 9 |

In all three rooms the shell also ran a few commands that were not the room
interpreter (a directory listing, a look at the interpreter wrapper, a `cat` of
files inside the room). Every path in them is inside the room.

## 5. Environment and manifests

| Item | Value |
| --- | --- |
| Sessions | one per room, model `claude-sonnet-5-5`, tools Bash, Edit, Read and Write, no MCP server, no skill |
| Room interpreter | Python 3.12.14 run without site processing, `sympy` 1.14.0, `mpmath` 1.3.0; the platform package does not import from the room |
| Solver's manifest | read `TASK.md` and the interpreter wrapper; nothing outside the room |
| Solver's prior-knowledge statement | it recognised the direction set as the vertices of a regular icosahedron and knew in general terms that the first non-invariant harmonic of that set is at degree six; it recalled none of the coefficients from memory |
| Audit rooms' manifests | not written: both rooms stopped before that step (deviations 1 and 2) |
| Redactions | none: no returned file carries an absolute path |

## 6. Hashes

Computed by the maintainers at copy-out.

| File | SHA-256 | Bytes |
| --- | --- | --- |
| `returns/STAGE_TWO.md` | `fb907984a99bbb212369c3107c1f749789adcae6968e2fd44118e76de1f37e5e` | 2206 |
| `returns/audit_a/audit.py` | `0cb2f842051bc57b088a9b041ba1cd0cd079396971508710abdf92fd3bd51336` | 2859 |
| `returns/audit_a/audit_results.json` | `b4568ad7564398baa31e022b73388f21d59bba371e03f75796319c24c021ab69` | 1706 |
| `returns/audit_b/audit.py` | `60c7eadd6e6e93a166eebfec34c23794c920f04d36a394d3506a8189c4a64d54` | 4675 |
| `returns/audit_b/audit_results.json` | `0a8e5e2ed0da1cd915688d9ff59b42f388cac6394695de859d51cf3f57b1b618` | 1766 |
| `returns/solver/MANIFEST.md` | `16e8aca52f504991b15c55681a521a84bf5437b7cb1580044cecdbe030e20e25` | 1181 |
| `returns/solver/REPORT.md` | `037546577356735227477f9f96bc54ffd18527884b2d37130384d7c1fc1ef4a7` | 4867 |
| `returns/solver/results.json` | `4fc9320c47a85a620c6dd0a551cad6e95f4a7df860e87b1db21a384e9892807d` | 6331 |
| `returns/solver/solve.py` | `a962e69f108f0582fe28f8cec63268d72e4fe7629a1afdb8ba0b8b0114c52b49` | 6219 |
| `returns/stage_two_audit.json` | `8a00b1646e2170143055760f5f383335b964087ea1809b21e539d44f8c1e1d4d` | 2732 |
| `returns/stage_two_audit.py` | `196e2554d22e18427d9a9ec0de4e6562ec7127662d72b8dc5dc45a856345b64a` | 8040 |

The session transcripts stay outside the repository. Their hashes:

| Transcript | SHA-256 |
| --- | --- |
| solver | `da4ae269142099de98bb4430c7b8e5d86a883f6f50712add217725c7abea6e40` |
| audit A | `ee3a921a81a7471e1e93a8df56a813ef08bc137b765408cfb40e5d9b2f85262e` |
| audit B | `563c86bca7898ac2391ec7e0f590b6953736c9f55cbbc787cb3144d19c9e8042` |

## 7. Declared before the unseal

The comparison in commit 3 uses the values on this page and nothing computed
after the author's files are opened. The maintainers make their own transcription
of the author's frozen values at the pinned commit; each value is reported as
agree or disagree with both numbers shown; the comparison is exact where the
author's value is exact; and it carries a mutation on each side.
