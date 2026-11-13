# The context-isolated clean room

> The platform's procedure for running a reproduction in a context that provably could not
> have been handed the answers. Written from an exercise, not in the abstract: the M8.5-A
> reproduction ran this procedure end to end (packet [#389](https://github.com/openwave-labs/openwave/pull/389),
> commitment [#391](https://github.com/openwave-labs/openwave/pull/391), adjudication
> [#392](https://github.com/openwave-labs/openwave/pull/392), audit
> [#394](https://github.com/openwave-labs/openwave/pull/394)), and every rule below either
> held there or was added because something slipped. Planning record and deviations log:
> [T4](tasks/t4_task_details.md); the exercise's own record:
> [`m8_5_task_details.md`](../openwave/xperiments/m8_mit/research/tasks/m8_5_task_details.md).

## 1. The claim ladder

Four terms, a ladder rather than synonyms. Using the wrong rung oversells a result.

| Term | What it names |
| --- | --- |
| clean room | the ENVIRONMENT and procedure: the isolated directory, the audited packet, the ordering record. What this standard specifies |
| context firewall | the RULE SET the room enforces: fresh context, no quarantined access, no web tools, the consulted-files manifest |
| context-isolated independent-method reproduction | the CLAIM a successful run earns, and the ceiling |
| blind | a STRONGER claim the room alone never grants. Blind is a property of what the verifying context has seen, not of the environment; a clean room makes isolation auditable, it cannot erase prior exposure. For an AI implementer the opaque training corpus caps the label structurally: prior exposure to published values cannot be excluded, so isolation buys provenance, never the label |

## 2. When to run one

When a result must be reproduced by a context that the record can show was never handed the
targets: an author-supplied computation reproduced in-platform, a single-implementation
result re-derived before a claim rests on it, any case where the implementing context would
otherwise share state with a context that knows the answers. The cost is real (a sealed
packet, ordered commits, and either an operator at the keyboard or the headless launch of
§ 3.4), so the trigger is the claim, not
habit: if nobody will rely on the isolation, a plain independent implementation is cheaper.

## 3. Building the room

### 3.1 Location, and the ancestor-path check

The room is a directory outside every repository working tree with no agent-instruction file
anywhere on its ancestor path. The check is a procedure, not advice:

```bash
d=/path/to/room
while [ "$d" != "/" ]; do
  for f in "$d/CLAUDE.md" "$d/.claude/CLAUDE.md"; do [ -e "$f" ] && echo "FOUND: $f"; done
  d=$(dirname "$d")
done
```

Read every file it finds, not merely whether one exists. The M8.5-A near-miss: the obvious
sibling-of-the-repository folder sat under a 7-line instruction file whose entire content
told the agent to go read the quarantined tree, on the first turn, before opening the
packet. Nothing inside the room would have revealed the breach. A future room is placed by
running this check, never by copying a path that worked before (the concrete path is a local
choice and stays out of the repository).

### 3.2 The interpreter is a second, independent route

The ancestor-path check covers what loads by instruction; the Python environment is a
separate route into the quarantined tree, and checking one says nothing about the other. An
editable install makes the tree reachable through a bare `import` regardless of where the
room sits, and an environment named after the repository puts that name in the shell prompt.
Verify from the room's shell, before launch:

```bash
python3 -c "import <platform_package>" 2>&1 | tail -1   # must FAIL
pip show <platform_package> | head -3                    # must return nothing
```

Record the interpreter, versions, and available numerics libraries for the environment
record. The M8.5-A launch shell arrived with the platform installed and switched
environments on the spot; the rule exists because the miss was live, not hypothetical.

### 3.3 What loads unavoidably

It depends on the launch mode (§ 3.4).

An operated session loads the user-global instruction file. Grep it for the subject vocabulary
before the run and disclose it in the consulted-files manifest as a named load. Read the
grep's summary with suspicion: "nothing relevant" is the conclusion to be most careful with,
since the file is written for a general working context and nobody edits it with a clean
room in mind. At M8.5-A it carried nothing answer-bearing and still named the platform
repository several times. Project memory is keyed to the working directory, so a fresh
folder starts with none.

A headless launch under `--restricted` loads no instruction file at all. At the 2026-09-13
prototype a canary with every tool disabled was asked to quote a line that exists only in the
user-global file: a plain headless session quoted it, which shows the probe can fire, and the
`--restricted` session answered NONE. Its init record listed no memory path, no MCP server
and no skill. Two things still reach the session in either mode, the account's email address
and the working-directory path, so the room's path must name nothing (§ 3.1). Re-run the
canary pair after any client upgrade before relying on this paragraph.

### 3.4 Launch: headless or operated

Two launch modes are admissible. Both withhold web tools, keep the implementer's writes inside
the room, and leave every commit to a maintainer.

**Headless (the default).** The maintainer launches the room as a separate one-shot session
from a script, and nobody answers prompts:

```bash
cd /path/to/room
claude -p "$(cat OPENING_PROMPT.md)" --restricted --tools "Read,Edit,Write,Bash" \
  --strict-mcp-config --disable-slash-commands --permission-prompts none \
  --allowedTools "Edit(./**)" "Write(./**)" "Bash(./py *)" \
  --no-session-persistence --output-format stream-json --verbose < /dev/null > transcript.jsonl
```

[`utils/clean_room_launch.sh`](utils/clean_room_launch.sh) builds the room interpreter, runs
the § 3.1 and § 3.2 checks, and launches. Each guard below was made to fire at the prototype:

| Route out of the room | Guard |
| --- | --- |
| the file tools | `--restricted` confines Read, Edit and Write to the working directory; an outside path fails as outside the working directory, not as a prompt |
| shell commands | `./py` is allowlisted, and `--permission-prompts none` denies every prompt automatically. The client also admits two shapes without a prompt: a write into the room by redirection (`cat > file <<EOF`), which the `Edit(./**)` and `Write(./**)` rules cover, and a wrapper such as `time ./py`. Both run OUTSIDE the `./py` sandbox, so the transcript check reads every shell command's paths, not only the denials. Observed at M8.11 ([#564](https://github.com/openwave-labs/openwave/pull/564)), every path in the room |
| harness pointers | the client writes a background command's output, and any tool result too large to return inline, to per-session folders OUTSIDE the room, both named after the room's path (a `tasks/` folder under the system temp directory, and a `tool-results/` folder under the client's per-project history), and hands the session the path. `--restricted` allows the read. The folders hold only that session's own output, so nothing answer-bearing is reachable this way, but it is a read outside the room: the transcript check classifies it separately, and the implementer's manifest will usually not list it. Observed at M8.11, and the same route appeared in M8.10's in-session run ([#547](https://github.com/openwave-labs/openwave/pull/547)) |
| code the implementation runs | `./py` executes under macOS `sandbox-exec`, whose profile denies reads and writes outside the room and the interpreter's own install, and every network call, so a script opening an outside file gets `Operation not permitted` from the OS |
| web, MCP connectors, skills, agents, messaging | absent from the session: four tools, no MCP server, no skill |

The approval prompt's job, making a reach outside the room visible, passes to the record:
every automatic denial is listed in the transcript's final `permission_denials` entry, which
the § 7 transcript check reads. A denial is evidence exactly as a declined prompt is (§ 6).
Two nearby options are not this route: `--bare` authenticates only with an API key, and
`dontAsk` mode's denial text invites the model to try other tools.

**After the run.** Once the returns are snapshotted and hashed outside the room and the record
has merged, delete the room and the two harness folders named after its path (the
harness-pointer row above). They hold copies of the room's own output, and a later session
launched from the same path would find them.

**Operated.** `claude --disallowedTools "WebSearch,WebFetch"` in DEFAULT permission mode with
the § 6 runbook, never a bypass mode. The prompts are the guard there, since the filesystem
outside the room is reachable in principle. Use it where the headless guards are unavailable,
such as a platform without `sandbox-exec`, or where a person's reading of each call is wanted.

## 4. The packet

The packet is everything that enters the room. A maintainer who knows the answers assembles
it, which is exactly what the audit anticipates: the containment is that no part is capable
of carrying a target, and every part gets a MECHANICAL audit rather than a judgment call.

| Part | Content rule | Audit |
| --- | --- | --- |
| the frozen protocol | a verbatim copy of the repository file | `diff` against the repository; identity is the whole audit |
| the group/data input | raw inputs only, no derived annotations, no labels, dimensions, or precomputed values | run the checks from the packet alone (closure counts, exact norms, a withheld-vocabulary scan for the derived-object vocabulary), each mutation-tested so no green line is a line that cannot fail |
| the task file | operational instructions only; its mathematical content is generic identities, never values | read against the withheld vocabulary; committed so the record shows what the room was told |

**Generators, not closures.** Hand over the seed, not the built object: an input the
implementer must construct from makes the construction gate a check that can fail, where
handing the full object makes it true by construction. Where the protocol permits supplying
a derivable quantity (M8.5-A: the group order), withholding it is the stricter choice, and
the withholding is STATED in the packet so silence does not read as an incomplete delivery.

**A tightening beyond the protocol is raised with the supplier, not just stated in the
packet.** Both sides agreed to the protocol, so withholding something it permits is the
maintainer's unilateral change until the supplier says otherwise: name it at delivery and
offer to supply the quantity and log a deviation. The run is valid either way, since the
stricter side cannot make a gate easier, and the point of asking is that an unraised
tightening turns into an argument about the input condition after the result exists. At
M8.5-A the author's answer was that the packet had been limited to the generators for the
same reason, which made the condition shared rather than assumed.

**Author-side evidence is not the audit.** An author may ship a verification script and
report alongside the packet; they stay outside the room, and the maintainer audit runs from
the packet alone. An audit that reads the supplier's report and concurs is a review of the
report. Set the two results side by side afterwards as a cross-check.

**Canonicalize, then hash.** Record the INCOMING hash of the delivered bytes, declare a
canonical serialization, and issue the AUTHORITATIVE hash of the canonical form, stating
whether canonicalization changed the bytes (at M8.5-A it was a no-op because the author
supplied exact arithmetic; the chain is recorded either way). A hash over an uncanonical
rendering pins a transcription, not the object.

**The task file's small print, learned the hard way:**

| Instruction to include | Why |
| --- | --- |
| the room boundary stated as an obligation ("this directory is your whole world, and that is a rule, not a description of the machine"), with an out-of-room read named recoverable and its concealment named not | an unstated rule can be neither honored nor breached; the filesystem is not sandboxed |
| protocol links point into a repository the implementer does not have and must not seek | a verbatim protocol copy carries relative links; say they are dead ends |
| print repo-safe relative paths in all output | a script echoing an absolute path forces a disclosed redaction at copy-out, since the room path stays out of the repository |
| fail loud, report the honest outcome, tolerances fixed in source | a disagreement is a finding, not a failure of the task |
| a consulted-files manifest including anything that loaded WITHOUT being asked for | the manifest is corroborated against the transcript at copy-out |
| where the packet hands over a subfolder: read its files by path, and never change directory out of the room's top level | in a headless launch the allow rules are written relative to the room, so after a `cd` into a subfolder every later write needs a prompt and is refused. Observed at M10.1 ([commitment](../openwave/xperiments/m10_oph/research/findings/m10_1_commitment.md) § 3): two audit rooms stopped at their second step |

**Commit before the room opens.** The packet, the task file, the audit script, and the audit
output land in the repository first, so the timestamps predate the run and nothing reads as
retrofitted to the outcome. Prepared artifacts live in the repository, never only in a
scratch directory or a conversation.

## 5. The opening prompt

The message that starts the room session is deliberately near-empty:

> Read `PROTOCOL.md`, `TASK.md`, and `GROUP_INPUT.md` in this directory, in that order, then
> do what `TASK.md` says. Everything you need is here; nothing outside this directory is
> available or needed.

Anything beyond this belongs in the task file, where it enters the audited packet and the
commit record. An instruction delivered as chat text is invisible to the packet hash, absent
from the manifest, and composed by a context holding the targets, which is the one channel
the firewall forbids. The near-empty prompt is a firewall property, not a style preference.

## 6. The operator runbook

In an operated launch (§ 3.4) the human at the room's keyboard is a firewall component. A
headless launch has no keyboard: the prompt rows below are replaced by the automatic denial
and its record, and the rest apply as written. The rules, applied per event:

| Event | Operator action |
| --- | --- |
| at launch, before the opening prompt | verify the room's file hashes against the sealed record; verify the session banner's working directory is the room; record the banner's model for the environment record |
| prompt for a path inside the room, or for running the implementation there | approve |
| prompt for any path outside the room, or any network call | decline, and log the prompt text in the deviations record |
| no blanket grant, ever | never pre-approve tool classes, never "don't ask again", never a bypass mode; each prompt is read before it is answered, and the reading is the guard |
| the implementer asks a question | never answered from the operator's knowledge, however innocuous; relayed to the maintainer session, answered from the protocol text or not at all, logged as a deviation |
| an apparent stall | check for a PENDING PERMISSION PROMPT before interrupting: a frozen token counter with no visible prompt was, at M8.5-A, a prompt that had not surfaced in the terminal. If truly stalled, one interrupt and the single word `continue`, nothing more |
| the implementer finishes | the final message is relayed verbatim; nothing in the room is touched until copy-out |

Remote approval (answering prompts from a phone) is permitted: it relocates the operator's
channel and grants the session nothing. The two rules that erode on a small screen are the
ones to hold consciously: every prompt read in full, and no question ever answered from the
phone.

Information flow between the room and the maintainer session is one-directional: the
maintainer session may see anything from the room (it is already disqualified); the room
receives only the audited packet, fixed before launch and never amended mid-run. A declined
prompt is evidence, not only prevention: a room session that asked for an outside path has
stated something about its behavior under isolation, and the record wants that even when the
answer was no.

## 7. Copy-out and the commitment

Nothing is unsealed until the commitment is committed and merged. The commitment carries:

| Content | Rule |
| --- | --- |
| hashes of every deliverable | computed independently by the maintainer at copy-out and matched against the implementer's own record |
| the environment record and consulted-files manifest | folded in, with the implementer's prior-knowledge disclosure |
| the transcript check | extract every tool call from the session transcript and classify its paths against the manifest; the manifest is corroborated, never merely attested. Session-temp writes of the run's own output are disclosed, not hidden. For a headless launch the transcript is the stream-json file the launch writes, and its final `permission_denials` entry is classified with the tool calls. Reads of the harness's own output folders (§ 3.4) are classified in their own row, and shell writes by redirection are classified by their paths |
| any redaction | disclosed precisely, with BOTH hashes recorded (the original stays the commitment; the transcript retains the original bytes) |
| the pre-declared choices | anything the protocol requires fixed before unsealing (M8.5-A: whether the optional module ran) is declared here and unavailable afterwards |
| the operator log | launch checks, prompts, stalls, deviations |

## 8. Unseal and adjudicate, in a second commit

The comparison lands as its own separately merged change that NAMES the commitment it
postdates, so the ordering is auditable from the record alone: the result was public before
the comparison, or its target transcription, existed anywhere.

| Rule | Why |
| --- | --- |
| the adjudicator makes its OWN transcription of the pinned target | independent of any earlier reconstruction's fixture, so a shared typo cannot be silently self-consistent |
| reference data (a label-to-signature map) may come from the unsealed objects, declared | that is what the protocol opens them to after commitment; check it for internal consistency before any cell comparison |
| the comparison itself carries mutations on BOTH sides | perturb one transcribed target cell and one candidate cell; the comparison must go red both times, so it is a check that can fail |
| exact comparison where the object is exact | a tolerance where an equality belongs hides a class of defect |

## 9. Close-out

The method note keeps the clean-room draft byte-frozen (its hash pinned in the commitment;
the status block says so) and appends the maintainer layer: the adjudication outcome, and
the recorded adversarial audit by an independent second agent briefed to REFUTE with its own
methods (at M8.5-A: its own group construction, its own character-extraction route, its own
tables; eight claims attacked, none refuted, two strengthened, six unflagged weaknesses
recorded).

A communications rule that bit during the exercise: **an edit is not a notification, and a
notification is not a record.** Editing a merged PR body notifies nobody, and a mention
added by edit does not notify either. The durable record goes in the body; anything that
needs a human's answer goes in a NEW comment.

## 10. Teardown, last

Delete the room only after this standard's per-run write-up (or its refutation) is done: the
room is the last place to check a detail the transcript renders ambiguously. A room left on
disk is a room that gets reused, and its second occupant is not context-isolated: by then
the folder holds the packet, the implementation, and the unsealed comparison. Keep the
session transcript; it lives under the agent's project-history directory, or where a headless
launch wrote it, outside the room,
and it is what makes the manifest checkable rather than an attestation.

**Hash the transcript at teardown and record the hash in the task document.** The transcript
itself may not belong in the repo: it embeds whatever instruction files the operator's agent
loads, and a repo that is public, or that may become public, is the wrong home for those. A
SHA-256 discloses nothing and costs one line, and without it the kept transcript is only as
good as the word of whoever produces it later. Record the hash, where the file lives, and
what backs it up. Do not record it in the run's checkpoint file: checkpoints are gitignored
session state and are deleted at close, which is exactly how this rule was discovered.

## 11. What is inherited, and what was M8.5-A's

| Any column inherits | M8.5-A-specific |
| --- | --- |
| the claim ladder, the room build (§ 3), the packet discipline (§ 4), the opening prompt, the operator runbook, the commitment-before-unseal ordering, the two-commit adjudication, the teardown rule | the particular protocol, gates G1-G12, the group input and its `Q(φ)` arithmetic, the § 6 coexact module and its verdict categories, the pinned pre-registration target |
| the mutation-test posture: every green line shown able to go red, including the audit's own checks and the adjudication comparison | the specific mutation lists |
| the reproduce-vs-derive ceiling (§ 12) | the worked instance of it |

## 12. The ceiling: reproduce, not derive

Per [`PR_REVIEW_STANDARDS.md`](PR_REVIEW_STANDARDS.md) Gate B (obligations row): the
platform reproduces computations to a frozen spec and does not derive on request. A part of
a protocol requiring ORIGINAL derivation is optional and declinable per run, and declining
is a recorded outcome, never a breach. The worked instance: the M8.5-A § 6 coexact module
pre-declared `not resolved` as a permitted verdict; the run chose to derive and its
derivation survived the audit, but the permission to decline is what kept the obligation
priced. The claim ceiling for any AI-implemented run stays context-isolated
independent-method reproduction; blind is never granted by machinery (§ 1).

## 13. Failure modes actually hit

Every entry below happened during the first exercise; the standard exists because they did.

| # | Failure mode | Where caught | Guard now |
| --- | --- | --- | --- |
| 1 | the sibling-folder ancestor path carrying an instruction to read the quarantined tree | before launch, by the § 3.1 check | § 3.1, run the check, read the content |
| 2 | the launch shell arriving with the platform importable and the repo named in the prompt | live at launch | § 3.2 verify commands |
| 3 | the user-global instruction file summarized as "nothing relevant" while naming the repository | block A re-read | § 3.3, disclose and read the summary skeptically |
| 4 | prepared artifacts (task file, audit script, prompt) living only in session state | planning review | § 4, commit before the room opens |
| 5 | the audit's withheld-vocabulary check silently skipped when an earlier format check aborted | the audit's own mutation suite | § 4, withheld-vocabulary scan computed first, appended unconditionally |
| 6 | a 15-minute "stall" that was an unsurfaced permission prompt | during the run | § 6 stall row |
| 7 | the implementation printing the room's absolute path into committed output | copy-out | § 4 task-file small print; § 7 redaction rule |
| 8 | the byte-identity double run writing to session temp space against an absolute "writes only inside the room" statement | transcript check | § 7, disclosed session-temp writes |
| 9 | the audit reply landed as a merged-PR body edit, notifying nobody | close-out | § 9 communications rule |
| 10 | the kept transcript's hash written only to the run's gitignored checkpoint | after close-out, by the operator asking why checkpoints are gitignored | § 10, hash at teardown and record it in the task document |

None of the ten reached the result; five were caught by machinery (checks, mutation
suites, the transcript audit), five by a person reading carefully. That ratio is the
argument for keeping both.

Entry 10 generalizes past this procedure: **a disposable file is never the only home for a
fact worth keeping.** Checkpoints exist so a stopped session can resume, they are gitignored
for that reason, and anything in one at close either moves into the tracked record or is
deliberately let go. Writing a terminal summary into a checkpoint reads like closing a task
and is the opposite of it.

## 14. Checklist

| Phase | Gate |
| --- | --- |
| build | ancestor-path check run and contents read; interpreter check green; unavoidable loads grepped and disclosed; room outside every working tree |
| packet | per-part mechanical audits green and mutation-tested; author evidence outside; incoming + authoritative hashes recorded; withholdings stated; task file carries the § 4 small print |
| commit 1 | packet + task file + audit script + output merged BEFORE launch |
| launch | hashes re-verified; headless (the launch script's checks green; the init record's cwd, model and tool list recorded) or operated (banner cwd + model recorded, default permission mode); web tools withheld; near-empty opening prompt, nothing else |
| run | operator runbook applied per event; deviations logged as they happen; packet never amended |
| commit 2 | the commitment: hashes, environment, manifest, transcript check, redactions, pre-declarations, operator log; merged BEFORE unsealing |
| commit 3 | adjudication naming commit 2; own transcription; two-sided comparison mutations; exact comparison |
| close | method note draft frozen + maintainer layer; independent adversarial audit recorded; questions to humans in NEW comments |
| teardown | per-run write-up done, then delete the room; keep the transcript, hash it, record the hash in the task document; nothing of record left only in the run's checkpoint |
