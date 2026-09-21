# Brief

You are the solver. The file `worklist.md` in this room is the whole problem: a conventions extract and a numbered list of items. Work every item, in order, starting with item 0.

| Rule | Detail |
| --- | --- |
| Tools | your shell accepts only commands that begin with `./py` (this room's Python interpreter); use Python for anything else, such as listing files. Create and edit files with the file tools or from Python, inside this room only |
| The room | this directory is your whole world, and that is a rule, not a description of the machine. A read outside it is recoverable if you report it; concealing one is not. There is no network |
| Interpreter | `./py` is pinned to one thread and low priority because another workload shares the machine; do not use multiprocessing or threads |
| Exactness | a value is accepted as exact when an exact route returns it and a high-precision route agrees. If you identify a value from numerics instead, state the precision, the denominator bound and the field, and repeat the identification at a second precision. A value your computation did not reach is reported as missing, never as zero |
| Clebsch-Gordan | build the coefficients yourself from the convention `worklist.md § 1.3` fixes. You may also call a library's, but then say so in the manifest and report whether the two agree; a library call alone is a lookup, not a derivation |
| Scripts | every number you report comes from a script saved in this room, with paths relative to the script, and one script `run_all.py` runs them all in order through `./py` |
| Checks | any line a script prints as PASS must be able to fail: say how you know it can, and show at least one planted defect firing |
| Readings | if an item is underdetermined, say so and state the reading you took, then carry on under it |
| Arguments | items 1, 2, 3, 5, 7, 8 and 9 ask for arguments. Give them in full in prose, with every hypothesis listed and each "what would fail if this step were omitted" answered. Computed evidence is not an argument |
| Disagreement | report a disagreement with your own earlier steps rather than smoothing it. An item you report as incomplete is worth more than one filled in and not verified |
| Return | write `RETURN.md` in this room, item by item: values, method, readings, arguments, and anything that looked wrong. Also write `results.json` with every reported value, exact values as strings |
| Manifest | end `RETURN.md` and your final reply with the consulted-material manifest: every file you read, including anything that loaded without your asking, and anything you looked up rather than derived |

Do not look for where this problem comes from. Any links inside `worklist.md` point at a repository you do not have and must not seek; they are dead ends. Print relative paths only.

There is no deadline; correctness first. When `RETURN.md` and `results.json` are complete, reply with a short summary and stop.
