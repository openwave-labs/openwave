# TASK (audit)

This directory is your whole world, and that is a rule, not a description of the
machine. Read nothing outside it, fetch nothing, and do not look for any other
source. If you read something outside it by accident, say so in `MANIFEST.md`: an
outside read is recoverable, concealing one is not.

There is no expected answer in this directory, and none is wanted. Your job is to
try to refute another implementer's work, not to confirm it.

## The object

Let `phi = (1 + sqrt(5)) / 2`. Take the twelve unit vectors `u` obtained by
normalizing the vectors `(0, ±1, ±phi)`, `(±1, ±phi, 0)`, `(±phi, 0, ±1)`, with
independent signs. For a positive length `a`, a unit direction `n` and a scalar
momentum `k`, define

```text
omega_squared(k, n, a) = sum over the twelve u of [1 - cos(a * k * (u . n))] / (2 * a * a)
```

Every direction is used once, with equal weights.

Let `P6` be the Legendre polynomial of degree six. Define `I6(n)` as the sum over
the twelve `u` of `P6(u . n)`, divided by the value of that same sum when `n` is
any one of the twelve unit vectors.

## Step one: your own derivation, before you open `solver/`

The folder `solver/` holds another implementer's script and outputs. Do not open
it yet.

Derive, with exact arithmetic (rationals and `sqrt(5)`), the homogeneous
polynomials of degree two, four and six in `q = k * n` of the expansion of
`omega_squared` in powers of `k`, their rotationally invariant parts and their
remainders, the coefficient of `I6(n)` in any nonzero remainder, and the change
of each coefficient under the replacement of `a` by `a / 2`.

Use a route that is not a symbolic series expansion of the cosine sum. Work from
the moments of the direction set, the sums over the twelve `u` of `(u . n)` raised
to the powers two, four and six, each reduced by your own argument, or fit the
coefficients from high-precision evaluations of the definition and then identify
them exactly. Say which route you used.

Write your results to `audit_results.json` with the same keys the solver was
asked for: `degree_2`, `degree_4`, `degree_6` (each with `polynomial`,
`invariant_coefficient`, `remainder`, `remainder_coefficient_in_I6`, `ratio`),
`first_nonzero_remainder_degree`, `scaling_a_to_half`. Save it before step two.

## Step two: the comparison and the audit

Open `solver/`. Then:

| Check | What to do | Verdict |
| --- | --- | --- |
| C1 | compare your polynomials and coefficients with the solver's, coefficient by coefficient, by exact arithmetic | agree, or the list of coefficients that differ, with both values |
| C2 | read the solver's numerical check: was the bound stated before the comparison, is it a valid bound on the truncated remainder, is every sample inside it; then run your own check at sixty digits or more on directions the solver did not use | holds or fails, with your numbers |
| C3 | check the solver's scaling under `a` to `a / 2` against the definition, term by term | exact or not |
| C4 | read the solver's control: was the invariance test defined before it was run, and does it reject both altered operators; then break the operator in a third way of your own and run the solver's test on it | rejects or passes |
| script | read `solver/solve.py` line by line: does it compute what `solver/REPORT.md` says, is any value typed in instead of computed, is any tolerance loose enough to hide a difference | a list of findings, or none |

## Rules

- Run code only as `./py <script>`. That interpreter has `sympy` and `mpmath`.
- Fix every tolerance in the source before running.
- Print and write only paths relative to this directory.
- A disagreement is a finding. Report both numbers and do not reconcile them.
- Do not edit anything under `solver/`.

## What to leave in this directory

| File | Content |
| --- | --- |
| `audit.py` | your derivation and your checks, runnable as `./py audit.py` |
| `audit_results.json` | step one's results, saved before step two |
| `AUDIT.md` | the route you used, then one verdict per check (C1, C2, C3, C4, script) with your own numbers, then anything else you found |
| `MANIFEST.md` | every file you read and when (before or after step one's results were saved), including anything that loaded without being asked for; the interpreter and library versions; and a prior-knowledge statement: whether you recognise this object, and whether you recalled any of its coefficients from memory before computing them |
