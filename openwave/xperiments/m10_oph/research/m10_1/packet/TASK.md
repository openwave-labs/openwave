# TASK

This directory is your whole world, and that is a rule, not a description of the
machine. Read nothing outside it, fetch nothing, and do not look for any other
source. If you read something outside it by accident, say so in `MANIFEST.md`: an
outside read is recoverable, concealing one is not.

There is no expected answer in this directory, and none is wanted. Report what the
mathematics gives. A result that surprises you is a finding, not a failure.

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

These are declared mathematical inputs. No physical field and no physical length
scale is attached to them.

## What to compute

Write the expansion of `omega_squared` in powers of `k` as a sum of terms
homogeneous in the vector `q = k * n`, and work with exact arithmetic (rationals
and `sqrt(5)`), never floating point, for items 1 to 4.

1. The exact homogeneous polynomials in `(q_x, q_y, q_z)` of degree two, four and
   six, each with its power of `a`.
2. For each of the three, its rotationally invariant part (the part proportional
   to a power of `q . q`) and what remains. State at which degree the first
   remainder that is not identically zero appears, and prove that the remainders
   at lower degree vanish exactly.
3. Where a remainder is not zero, express it through `I6(n)` if that is exactly
   possible, and say so if it is not. Report every coefficient exactly. At each
   degree with a nonzero remainder, report the ratio of the remainder's
   coefficient to the invariant part's coefficient.
4. How each coefficient changes under the replacement of `a` by `a / 2`.
5. A direct numerical check that does not reuse the series: evaluate
   `omega_squared` from its definition at sixty decimal digits or more, for at
   least five directions `n` (include one of the twelve `u`, one coordinate axis
   and three generic directions) and for `a * k` in `{1/4, 1/8, 1/16, 1/32}`.
   Compare with the series truncated after degree six, and show that each
   difference is inside a bound on the truncated remainder that you derive and
   state before you run the comparison.

## The control

Build two altered operators: one with a single direction removed, and one with the
weight of a single direction changed from one to two. Define a test of rotational
invariance for the degree-two polynomial before you run it, then run it on the
original and on both altered operators. Report all three outcomes.

## Method

Use a direct symbolic series expansion of the definition for items 1 to 4.

## Rules

- Run code only as `./py <script>`. That interpreter has `sympy` and `mpmath`.
- Fix every tolerance in the source before running. Do not tune anything after
  seeing a result.
- Print and write only paths relative to this directory.
- If two of your own routes disagree, report both numbers and stop. Do not pick.

## What to leave in this directory

| File | Content |
| --- | --- |
| `solve.py` | the whole computation, runnable as `./py solve.py`, regenerating `results.json` |
| `results.json` | every exact result as a string a computer algebra system can parse, under the keys `degree_2`, `degree_4`, `degree_6` (each with `polynomial`, `invariant_coefficient`, `remainder`, `remainder_coefficient_in_I6`, `ratio`), `first_nonzero_remainder_degree`, `scaling_a_to_half`, `numerical_check` (one row per sample: direction, `a*k`, difference, bound, inside), and `control` (three outcomes) |
| `REPORT.md` | equations first: what was computed and how, then the results, then anything that did not go as expected |
| `MANIFEST.md` | every file you read, including anything that loaded without being asked for; the interpreter and library versions; and a prior-knowledge statement: whether you recognise this object, and whether you recalled any of its coefficients from memory before computing them |
