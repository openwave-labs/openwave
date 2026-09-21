# Brief

You are the adversarial auditor. Two independent agents were given the same problem, `worklist.md`, and worked it without seeing each other. Their rooms are copied here as `solver_a/` and `solver_b/`, complete: returns, scripts, logs.

Your job is not to re-solve the problem. It is to find out **which of their claims survive an attempt to break them**, and to grade the arguments they were asked for.

| Rule | Detail |
| --- | --- |
| Tools | your shell accepts only commands that begin with `./py` (this room's Python interpreter); use Python for anything else, such as listing files. Create and edit files with the file tools or from Python, inside this room only |
| The room | this directory is your whole world, and that is a rule, not a description of the machine. A read outside it is recoverable if you report it; concealing one is not. There is no network |
| Interpreter | `./py` is pinned to one thread and low priority because another workload shares the machine; do not use multiprocessing or threads |
| ⭐ Your own route | every number you assert comes from code **you** wrote in this room, from the definitions in `worklist.md`. You may read their scripts to understand and to grade them. You may not import them, call them, or copy them to produce your own numbers. Running their script and getting their answer is not verification |
| Exactness | a value is accepted as exact when an exact route returns it and a high-precision route agrees. State precision and residual for every numerical route |
| Checks | any line a script prints as PASS must be able to fail: say how you know it can, and show at least one planted defect firing |

## What to do, in order

**1. The controls first.** Worklist items 11 and 12 are the two checks whose whole purpose is to be able to come out wrong. Recompute both yourself, exactly, and report whether each solver's numbers match yours. Also settle, by your own route, whether the locus `span{v₃, v₋₁}` carries any interior critical point, and what its restriction's linear coefficient is. A control that cannot fail is worthless, so say for each one what result would have exposed an error, and state what you actually found rather than what the item's placement suggests it should find.

**2. Where they disagree, adjudicate.** List every item where `solver_a` and `solver_b` report different values, different counts, or incompatible readings. For each, compute the answer yourself and say which is right, or that both are wrong, or that the item is genuinely underdetermined and they took different readings. Agreement between them is not evidence for you: they may share a mistake, so spot-check agreed values too, and say which ones you checked.

**3. Grade the arguments.** Items 1, 2, 3, 5, 7, 8 and 9 asked for arguments, not just numbers. For each solver and each of those items, grade:

| Grade | Meaning |
| --- | --- |
| SOUND | the argument establishes the claim; every hypothesis is stated and each is either verified in the work or is standard and named as such |
| INCOMPLETE | no error found, but a step is asserted rather than argued, or a hypothesis is used without being stated |
| DEFECTIVE | there is an error you can demonstrate, or a case the argument misses, and you exhibit it |

Two questions carry the most weight, so answer them explicitly for each solver:

- **Completeness on the lines (item 2).** Does the claim that the critical set is complete rest on an elimination argument, or on a solver returning a list? If the latter, it is not complete, however correct the list. Say which it is, and if the argument is by elimination, check that the cases are exhaustive by working them yourself.
- **The two ends (item 8).** Items asked for an argument rather than a search. A search over starting points, however dense, establishes neither end. If a solver gave a search, say so and grade it INCOMPLETE regardless of whether its answer looks right. If a solver gave an argument, list its hypotheses and try to break each one.

**4. Try to break the rest.** Pick the claims that would be most costly to get wrong, and attack them: perturb a representative and check the reported critical point is really critical; verify a reported signature by an independent route (a different basis, a different decomposition) and confirm it is basis-independent; check a reported kernel direction really is annihilated and is not an artifact of a tolerance; confirm that a quantity reported as zero was computed and not merely unreached.

**5. Report.** Write `AUDIT.md` in this room:

- a table per worklist item: solver_a's answer, solver_b's answer, your own, and the verdict AGREE / DISAGREE / DEFECT;
- the argument grades of step 3 with the reason for each;
- every defect you can demonstrate, with the demonstration;
- every claim you tried to break and could not, which is worth as much;
- what you could not check, and why.

Also write `audit.json` with the same verdicts in machine-readable form.

A defect you can demonstrate is a finding. A suspicion you cannot demonstrate is reported as a suspicion, labeled as one, never upgraded. Reporting "I could not break this" is a real result, and an item you report as unchecked is worth more than one graded without being checked.

End `AUDIT.md` and your final reply with the consulted-material manifest: every file you read, including anything that loaded without your asking. Print relative paths only. Links inside these files point at a repository you do not have and must not seek; they are dead ends. There is no deadline; correctness first.
