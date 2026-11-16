# Brief

You are an independent auditor, and this is your first stage. Three files in this room are the problem: `worklist.md`, a conventions extract and a numbered worklist; `clauses.md`, five clauses that take precedence over `worklist.md` where they speak; and `inputs.json`, the inputs `worklist.md` Section 5 names.

Other rooms, which you never see, are working the worklist now. In a second stage you will be shown their returns and asked to try to break them. This first stage exists so that your own numbers are fixed, and recorded, before you see theirs: compute the items below **by your own route**, from the definitions in `worklist.md`. Prefer a route that differs from the most obvious one where you can (a different basis, a different quadrature or decomposition, an exact route checked by a numerical one), and say what your route is.

## Your items

1. Item 0(a) to 0(e), with 0(c) for all three slots.
2. A.1 and A.2 in all three slots; A.3.
3. A.4 at `line {5/2, -3/2}` and `weight 7/2`.
4. B.1 and B.2 (with clauses 3 and 4) at `v1`, `pentagonal pyramid` and `D2 ray`.
5. B.3 (with clause 4) at `line {5/2, -3/2}` and `weight 7/2`, the four members included.
6. B.5.
7. E.1, and E.2 for two slots of your choice.

Work them in the order listed. Items not listed are not yours: do not work them.

## Rules

| Rule | Detail |
| --- | --- |
| Tools | your shell accepts only commands that begin with `./py` (this room's Python interpreter); use Python for anything else, such as listing files. Create and edit files with the file tools or from Python, inside this room only |
| The room | this directory is your whole world, and that is a rule, not a description of the machine. A read outside it is recoverable if you report it; concealing one is not. There is no network. ⚠️ Never change directory: keep every file at this room's top level or in folders you make here, and read and write them by their path from the top level. Your write permissions are tied to the top level, and after a change of directory every write is refused |
| Interpreter | `./py` runs one thread at low priority because another workload shares the machine; do not use multiprocessing or threads. Libraries available: numpy, scipy, sympy, mpmath and the standard library, nothing else. Interval arithmetic is `mpmath`'s (`mpmath.iv`) or your own |
| Exactness | a value is accepted as exact when an exact route returns it and a high-precision route agrees. If you identify a value from numerics instead, state the precision, the denominator bound and the field, and repeat the identification at a second precision. A value your computation did not reach is reported as missing, never as zero |
| Group data | the group is given by two generators. Build what you need from them, and say for the character table and the decompositions whether you DERIVED them in this room or RECOGNISED them from what you know |
| Scripts | every number you report comes from a script saved in this room, with paths relative to the script, and one script `run_all.py` runs them all in order through `./py`. Keep each script's printed output in a log file beside it |
| Checks | any line a script prints as PASS must be able to fail: say how you know it can, and show at least one planted defect firing |
| Readings | if an item is underdetermined, say so and state the reading you took, then carry on under it |
| Return | write `OWN.md` in this room, item by item: values, route, readings, and anything that looked wrong. Also write `own.json` with every value, keyed first by item (`"0(a)"`, `"A.1"`, `"B.1"`, ...), then by slot (`"R2"`, `"R4"`, `"R5"`) where the item has slots, then by the point or box name exactly as `inputs.json` spells it. Exact values are strings, signatures are `[n_-, n_0, n_+]`, polynomials are strings in `nu`, factored, and enclosures are `["lower", "upper"]` as decimal strings |
| Manifest | end `OWN.md` and your final reply with the consulted-material manifest: every file you read, including anything that loaded without your asking, and anything you looked up rather than derived |

Do not look for where this problem comes from. Any links or names inside `worklist.md` point at records you do not have and must not seek; they are dead ends. Print relative paths only, never an absolute one.

There is no deadline; correctness first. When `OWN.md` and `own.json` are complete, reply with a short summary and stop.
