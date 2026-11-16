# Brief

You are the adversarial auditor, in your second stage. In your first stage you computed a set of items by your own route; your files from that stage are copied here, unchanged, in `own/` (`own/OWN.md`, `own/own.json` and your scripts). They were recorded before you saw anyone else's work, and they stay as they are: do not edit anything under `own/`.

Independent solver rooms worked the same problem, `worklist.md` with `clauses.md` and `inputs.json`, without seeing each other or you. Their rooms are copied here, complete, as `solver_ca/` (anchors, C.1, C.2, C.4), `solver_cb/` (anchors, C.1, C.2, C.3, C.5) and `solver_cert/` (anchors, Block D, B.4). Each holds a `RETURN.md`, a `results.json`, its scripts and its logs.

Your job is not to re-solve the problem. It is to find out **which of their claims survive an attempt to break them**.

## Your scope

The anchors in all three rooms; C.1, C.2 (both C rooms answer them: compare them with each other too); C.3, with clause 5; C.4, the outside orbits' boxes included; C.5, with clauses 1 and 4, including how each slow group's eigenvectors were normalized and how the semisimple group was handled; Block D and B.4, with clauses 2 and 3, including whether each Krawczyk test was run on the box Section 3 defines, whether its enclosures are outward-rounded, and whether the certified point used for B.4 lies in the given box.

## Rules

| Rule | Detail |
| --- | --- |
| Tools | your shell accepts only commands that begin with `./py` (this room's Python interpreter); use Python for anything else, such as listing files. Create and edit files with the file tools or from Python, inside this room only |
| The room | this directory is your whole world, and that is a rule, not a description of the machine. A read outside it is recoverable if you report it; concealing one is not. There is no network. ⚠️ Never change directory: read the copied rooms by their paths from the top level (`solver_l6/RETURN.md`), and write your own files at the top level or in folders you make here. Your write permissions are tied to the top level, and after a change of directory every write is refused |
| Interpreter | `./py` runs one thread at low priority because another workload shares the machine; do not use multiprocessing or threads. Libraries available: numpy, scipy, sympy, mpmath and the standard library |
| ⭐ Your own route | every number you assert comes from code **you** wrote in this room, or in your first stage, from the definitions in `worklist.md`. You may read the solvers' scripts to understand and to grade them. You may not import them, call them, or copy them to produce your own numbers. Running their script and getting their answer is not verification |
| Exactness | a value is accepted as exact when an exact route returns it and a high-precision route agrees. State precision and residual for every numerical route |
| Checks | any line a script prints as PASS must be able to fail: say how you know it can, and show at least one planted defect firing |

## What to do, in order

**1. Compare.** For every item in your scope, set each solver's answer beside your own stage-1 answer where you have one. Where they differ, find out why, and say which is right, or that both are wrong, or that the item is underdetermined and they took different readings. Agreement is not evidence: a shared convention can be wrong in the same way twice.

**2. Fill the gaps by your own route.** For items or points in your scope that your first stage did not compute, recompute a sample yourself, chosen where an error would be most costly or least visible (the degenerate points, the points whose words depend on a single root's sign, the clusters whose Krein sign rests on a numerical definiteness test), and say which you chose and why.

**3. Readings.** List every reading a solver took where the worklist was underdetermined. For each, say whether another reading was available, and whether the reported values would change under it.

**4. Their checks.** For each solver, find any PASS line that could not have failed, any check that compares a quantity with itself, and any value that was identified from numerics without a second precision. Say whether each solver's planted defects actually exercised the checks its reported values rest on.

**5. Try to break the rest.** Attack the claims that would be most costly to get wrong: perturb a point and confirm criticality is exact rather than tolerance-level; confirm a reported signature by an independent decomposition; confirm that a zero multiplicity is an algebraic multiplicity and not a kernel dimension; confirm that a value reported as exact is exact.

**6. Report.** Write `AUDIT.md` in this room:

- a table per worklist item and point: each solver's answer, your own, and the verdict AGREE / DISAGREE / DEFECT / UNCHECKED;
- every defect you can demonstrate, with the demonstration;
- every claim you tried to break and could not, which is worth as much;
- the readings of step 3 and the checks of step 4;
- what you could not check, and why.

Also write `audit.json` with the same verdicts, keyed by item, slot and point name as `inputs.json` spells them.

A defect you can demonstrate is a finding. A suspicion you cannot demonstrate is reported as a suspicion, labeled as one, never upgraded. "I could not break this" is a real result, and an item you report as unchecked is worth more than one graded without being checked.

End `AUDIT.md` and your final reply with the consulted-material manifest: every file you read, including anything that loaded without your asking. Print relative paths only. Links or names inside these files point at records you do not have and must not seek; they are dead ends. There is no deadline; correctness first.
