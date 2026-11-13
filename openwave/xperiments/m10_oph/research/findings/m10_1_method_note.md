# M10.1 method note: the twelve-direction scalar dispersion, reproduced in clean rooms

> Task: [M10.1](../tasks/m10_1_task_details.md), the M10 column's certification
> gate. Record of the run: the [commitment](m10_1_commitment.md) (commit 2, pushed
> before the author's values were opened) and this note (commit 3).
>
> **Verdict.** The expansion coefficients the author froze for this object are
> reproduced exactly: seven of seven values agree, by exact arithmetic.
>
> **Scope**: arithmetic on one declared mathematical object, the equal-weight
> cosine sum over the twelve vertex directions of an icosahedron, through degree
> six in the momentum; it licenses "the author's frozen coefficients for this
> object are correct" and does not license any physical identification of the
> object, any statement about the author's other frozen rows, or any `MODELS.md`
> cell.

## 1. The equations

**The object.** With `φ = (1 + √5)/2`, let `u` run over the twelve unit vectors
obtained by normalizing `(0, ±1, ±φ)`, `(±1, ±φ, 0)`, `(±φ, 0, ±1)`. For a positive
length `a`, a unit direction `n` and a momentum `k`, with `q = k n`:

```text
ω²(q, a) = Σ_u [1 − cos(a u·q)] / (2 a²)
```

**The expansion.** Only even powers survive. With the moments
`S_2m(q) = Σ_u (u·q)^(2m)`:

```text
ω² = Σ_{m ≥ 1} (−1)^(m+1) a^(2m−2) S_2m(q) / (2 (2m)!)
```

**The invariant part and the remainder.** The average of `(u·n)^(2m)` over the
sphere is `1/(2m+1)`, so the rotationally invariant part of `S_2m(q)` is
`(12/(2m+1)) |q|^(2m)`. The remainder is `S_2m(q)` minus that, and it has zero
average over directions.

**The angular function.** With `P6` the Legendre polynomial of degree six:

```text
I6(n) = Σ_u P6(u·n) / Σ_u P6(u·u₀)        (u₀ any one of the twelve)
```

**What is reported.** For degrees two, four and six: the homogeneous polynomial in
`(q_x, q_y, q_z)`, the coefficient of its invariant part, its remainder, the
coefficient of `|q|⁶ I6(n)` in the remainder, and the ratio of the two
coefficients. The author's notation for the same quantities:

```text
ω² = k² + C4 k⁴ + B0 k⁶ + B6 k⁶ I6(n) + O(k⁸)
```

**The numerical check.** `|1 − cos t − (t²/2 − t⁴/24 + t⁶/720)| ≤ t⁸/8!`, so with
`|u·n| ≤ 1` the series truncated after degree six is within
`12 (a k)⁸ / (2 a² 8!)` of the definition.

**The control.** The degree-two polynomial is `qᵀ M q / 4` with
`M = Σ_u w_u u uᵀ`. It is rotationally invariant exactly when
`M − (tr M / 3) I` is the zero matrix.

## 2. Equation-to-code map

Three rooms derived the expansion; the solver's script is the reference
implementation, and the two audit rooms' scripts are independent of it.

| Term | Where |
| --- | --- |
| The twelve directions, exact in `Q(√5)` | [`solve.py` L11-L19](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/m10_1/returns/solver/solve.py#L11-L19) |
| The moments `S_2m(q)` | [`solve.py` L31-L37](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/m10_1/returns/solver/solve.py#L31-L37) |
| The series coefficient `(−1)^(m+1) a^(2m−2) / (2 (2m)!)` | [`solve.py` L38-L41](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/m10_1/returns/solver/solve.py#L38-L41) |
| `P6`, the normalization `Σ_u P6(u·u₀)`, and `\|q\|⁶ I6(n)` as a polynomial | [`solve.py` L43-L49](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/m10_1/returns/solver/solve.py#L43-L49) |
| Invariant part, remainder, the coefficient of `I6`, the ratio | [`solve.py` L53-L80](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/m10_1/returns/solver/solve.py#L53-L80) |
| The numerical check and its bound | [`solve.py` L90-L133](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/m10_1/returns/solver/solve.py#L90-L133) |
| The control | [`solve.py` L135-L153](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/m10_1/returns/solver/solve.py#L135-L153) |
| The same moments by the audit rooms' own route | [`audit_a/audit.py` L25-L69](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/m10_1/returns/audit_a/audit.py#L25-L69), [`audit_b/audit.py`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/m10_1/returns/audit_b/audit.py) |
| C1, the exact comparison of the three result files | [`m10_1_compare_rooms.py` L66-L84](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_1_compare_rooms.py#L66-L84) |
| C2 to C4 and the reading of the solver's script | [`stage_two_audit.py` L42-L119](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/m10_1/returns/stage_two_audit.py#L42-L119) |
| C5, the transcription of the author's values and the comparison | [`m10_1_adjudicate.py` L38-L83](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_1_adjudicate.py#L38-L83) |

## 3. Results, each beside its gate

The gates were fixed in the [task document](../tasks/m10_1_task_details.md) before
any numerics.

**The result.**

```text
ω² = |q|² − (a²/20) |q|⁴ + (a⁴/840) |q|⁶ [1 + (16/75) I6(n)] + O(|q|⁸)
```

| Check | Gate | Outcome |
| --- | --- | --- |
| C1 | the solver's polynomials equal the audit's, derived by a different method | ✅ three derivations (one series expansion, two by moments) agree exactly at degrees two, four and six; three planted disagreements caught |
| C2 | the series reproduces the definition at 60 digits or more, inside a stated bound | ✅ the solver's 20 samples at 80 digits and 32 further samples at 70 digits, all inside the bound |
| C3 | the scaling under `a → a/2` is what the definition implies | ✅ exact: factors 1, 1/4, 1/16; a wrong factor fails |
| C4 | an altered operator is rejected by the invariance test | ✅ four altered operators rejected, the original accepted. A limit: the test reads the degree-two term only, so on its own it would pass six cube axes, whose degree-four sum is not invariant |
| C5 | each value against the author's frozen values, both numbers shown | ✅ seven of seven agree exactly (table below); one planted error on each side caught |

| Quantity | The author's frozen value (row FZ-11) | The rooms' value |
| --- | --- | --- |
| `C4` | `−a²/20` | `−a²/20` |
| `B0` | `a⁴/840` | `a⁴/840` |
| `B6` | `2a⁴/7875` | `2a⁴/7875` |
| `B6/C4²` | `32/315` | `32/315` |
| `B0/C4²` | `10/21` | `10/21` |
| `B6/B0` | `16/75` | `16/75` |
| Lowest anisotropic rank | six (ranks one to five vanish) | six: the remainders at degrees two and four are zero, the degree-six remainder is exactly a multiple of `I6`, which is a harmonic of rank six, and an even function of `q` carries no odd rank |

**Ordering.** The rooms' values were fixed in commit 2 (`3f6e4ea0`, committed
2026-11-13 18:26:39 UTC and pushed by the same command). The author's register
was first opened at 18:26:50 UTC by the maintainers' record, which the repository
cannot show, at the pinned commit `3c4e1e2`; its SHA-256 there is `11bff1e4…9980`, and the row's
own content hash is `8ac97d7c…66d8`. The comparison reads only what commit 2
fixed. The register row does not define `I6`; its normalization is the one in the
author's definition-only statement, restated in § 1.

## 4. What was not computed

| Not done | Why it matters |
| --- | --- |
| Any physical identification | The author attaches no physical field, sector, scale or frame to this object, and neither does this note. The result says nothing about whether any physical dispersion has this form |
| The author's row FZ-12 | It is a different object, a sum over the thirty edge directions with a different normalization. Its `C4` and `B0` equal FZ-11's; its rank-six coefficient differs in sign and in size (`B6/B0 = −1/15` against `16/75`). It was not recomputed here |
| Orders above degree six | Not asked, not done |
| The author's own two recomputes and their 90 numerical rows | Not re-run. The author's comparison summary and the register row were read; the two recompute files were not |
| The author's Lean proof of the same expansion | Not built and not read |
| An implementation by a different model family | The three rooms and the two audit agents are sessions of one model family, so a shared error is conceivable. What stands against it: four separately written implementations on two routes agree exactly with each other and with the author's values |
| A `MODELS.md` cell | None moves. The object sits behind no criterion |

**The claim this run earns.** A context-isolated, independent-method
reproduction, in the terms of
[`CLEAN_ROOM_STANDARDS.md`](../../../../../dev_docs/CLEAN_ROOM_STANDARDS.md) § 1. Not
"blind" in the strict sense of that section: the rooms were never handed a
target and the transcripts show it, but the solver recognised the direction set
as an icosahedron's vertices and knew in general terms where its first
anisotropic harmonic sits, and prior exposure of a language model to published
values cannot be excluded. It reported recalling none of the coefficients.

## 5. Deviations

Logged in the [commitment](m10_1_commitment.md) § 3 as they happened: the three
commits were pushed in order to one branch, not merged one by one; two audit
rooms stopped before their second step, for a cause in the packet's layout; the
second-step checks were done outside the rooms by an independent audit agent.
None touches the ordering: every check C1 to C4 was pushed before the author's
values were opened.

## 6. Audit record

An independent second agent, briefed to refute, audited this note and the
adjudication before the merge, with its own script and none of the audited code.

| What it attacked | Outcome |
| --- | --- |
| The mathematics, from scratch (its own exact expansion over `Q(√5)`) | confirmed: the same `C4`, `B0`, `B6`, the three ratios, and `132/25` for the normalization; the truncated series inside the bound at 50 digits |
| The transcription of row FZ-11 | confirmed against the register copy; the register's hash confirmed; the row's content hash is a field of the register, quoted and not recomputed |
| Whether C5 tests what it says | confirmed that the candidate side reads only the file fixed in commit 2, unmodified since; it verified that `I6` is homogeneous of degree six with zero Laplacian, so the rank statement follows |
| The ordering | the commit time confirmed; the time the register was opened cannot be verified from the repository, and this note now says so |
| The equation-to-code map | every range contains its term; one range widened by a line |
| Overclaims | none in the verdict or the scope line. Folded in from its findings: the limit of the C4 control, the source of the `I6` normalization, the size as well as the sign of FZ-12's coefficient, the same-model-family caveat, and one unescaped table cell |

## 7. Minimal inspection set

| # | Artifact | Read it for |
| --- | --- | --- |
| 1 | [`returns/solver/solve.py`](../m10_1/returns/solver/solve.py) | the whole computation, 155 lines, equations in its comments |
| 2 | [`returns/audit_a/audit.py`](../m10_1/returns/audit_a/audit.py) | the same moments by a second route, 77 lines |
| 3 | [`scripts/m10_1_compare_rooms.py`](../scripts/m10_1_compare_rooms.py) | C1, the exact comparison |
| 4 | [`scripts/m10_1_adjudicate.py`](../scripts/m10_1_adjudicate.py) | C5, the target transcription and the comparison |
