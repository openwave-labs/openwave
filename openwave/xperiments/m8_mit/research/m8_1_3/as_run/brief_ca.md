# Brief

You are a solver. Three files in this room are the whole problem: `worklist.md`, a conventions extract and a numbered worklist; `clauses.md`, five clauses that take precedence over `worklist.md` where they speak; and `inputs.json`, the inputs `worklist.md` Section 5 names. Other rooms work other items of the same worklist; you never see them, and they never see you.

## Your items

1. Item 0(a) to 0(e), with 0(c) for all three slots.
2. C.1 and C.2 in each slot (with clause 4).
3. C.4, at every point it names, `O1` to `O3` in `R₄` and `R₅` included: for those three, decide criticality and the sign over the boxes of Section 5, by interval arithmetic, with your own Krawczyk test or with an argument that the sign is constant over the box.

Work them in the order listed, the anchors first, and report the anchors before the rest. Items not listed are not yours: do not work them.

## Rules

| Rule | Detail |
| --- | --- |
| Tools | your shell accepts only commands that begin with `./py` (this room's Python interpreter); use Python for anything else, such as listing files. Create and edit files with the file tools or from Python, inside this room only |
| The room | this directory is your whole world, and that is a rule, not a description of the machine. A read outside it is recoverable if you report it; concealing one is not. There is no network. ⚠️ Never change directory: keep every file at this room's top level or in folders you make here, and read and write them by their path from the top level. Your write permissions are tied to the top level, and after a change of directory every write is refused |
| Interpreter | `./py` runs one thread at low priority because another workload shares the machine; do not use multiprocessing or threads. Libraries available: numpy, scipy, sympy, mpmath and the standard library, nothing else. Interval arithmetic is `mpmath`'s (`mpmath.iv`) or your own |
| Exactness | a value is accepted as exact when an exact route returns it and a high-precision route agrees. If you identify a value from numerics instead, state the precision, the denominator bound and the field, and repeat the identification at a second precision. A value your computation did not reach is reported as missing, never as zero |
| Group data | the group is given by two generators. Build what you need from them. For the character table and the decompositions of item 0, say for each whether you DERIVED it in this room or RECOGNISED it from what you know |
| Clebsch–Gordan | build the coefficients yourself from the convention `worklist.md` § 1.2 fixes. You may also call a library's, but then say so in the manifest and report whether the two agree; a library call alone is a lookup, not a derivation |
| Scripts | every number you report comes from a script saved in this room, with paths relative to the script, and one script `run_all.py` runs them all in order through `./py`. Keep each script's printed output in a log file beside it |
| Checks | any line a script prints as PASS must be able to fail: say how you know it can, and show at least one planted defect firing |
| Readings | if an item is underdetermined, say so and state the reading you took, then carry on under it |
| Disagreement | report a disagreement with your own earlier steps rather than smoothing it. An item you report as incomplete is worth more than one filled in and not verified. Do not guess what an answer is supposed to be, and do not adjust a result to make it look canonical |
| Return | write `RETURN.md` in this room, item by item: values, method, readings, and anything that looked wrong. Also write `results.json` with every reported value, keyed first by item (`"0(a)"`, `"A.1"`, `"B.1"`, ...), then by slot (`"R2"`, `"R4"`, `"R5"`) where the item has slots, then by the point or box name exactly as `inputs.json` spells it. Exact values are strings, in a form `sympy` can parse (a reduced rational `"p/q"`, a radical `"sqrt(n)/m"`), signatures are `[n_-, n_0, n_+]`, polynomials are strings in `nu`, factored, and enclosures are `["lower", "upper"]` as decimal strings |
| Manifest | end `RETURN.md` and your final reply with the consulted-material manifest: every file you read, including anything that loaded without your asking, and anything you looked up rather than derived |

Do not look for where this problem comes from. Any links or names inside `worklist.md` point at records you do not have and must not seek; they are dead ends. Print relative paths only, never an absolute one.

There is no deadline; correctness first. When `RETURN.md` and `results.json` are complete, reply with a short summary and stop.
