# REPORT

## Equations

Twelve unit vectors u = v/|v|, v in {(0,±1,±φ), (±1,±φ,0), (±φ,0,±1)}, |v|² = 2+φ = (5+√5)/2 for all of them.
With q = k n:

    ω² = Σ_u [1 − cos(a u·q)] / (2a²) = Σ_{m≥1} (−1)^{m+1} a^{2m−2} S_m(q) / (2 (2m)!),   S_m(q) = Σ_u (u·q)^{2m}.

Only even powers occur, so (u·q)^{2m} = (v·q)^{2m} / (2+φ)^m and everything stays in Q(√5) (sympy, exact; no floats in items 1–4).

Invariant part of degree 2m: c_m (q·q)^m, with c_m = coef(m)·12/(2m+1) (the sphere average of (u·n)^{2m} is 1/(2m+1)). Remainder = polynomial − invariant part.

I6(n) = Σ_u P6(u·n) / S6, with S6 = Σ_u P6(u·u0) = 132/25 for any vertex u0. As a polynomial: |q|⁶ I6(n) is stored in `results.json` under `I6_polynomial_in_q`.

Remainder-through-I6 route: x⁶ = 1/7 + (l=2, l=4 parts) + (16/231) P6(x). The l=2, l=4 sums vanish for this set, so Σ(u·n)⁶ = 12/7 + (16/231) S6 · I6(n).

## Results

1. Homogeneous polynomials (q² = q_x²+q_y²+q_z²):
   - degree 2: `q_x² + q_y² + q_z²` (a⁰).
   - degree 4: `−(a²/20) (q²)²` (written out in `results.json`).
   - degree 6: `a⁴ [ q_x⁶/900 + q_x⁴q_y²(1/240 − √5/1200) + q_x⁴q_z²(1/240 + √5/1200) + q_x²q_y⁴(1/240 + √5/1200) + q_x²q_z⁴(1/240 − √5/1200) + q_y⁶/900 + q_y⁴q_z²(1/240 − √5/1200) + q_y²q_z⁴(1/240 + √5/1200) + q_z⁶/900 ]`.
     There is no q_x²q_y²q_z² term in this polynomial; the full list is in `results.json`.

2. Invariant parts and remainders.
   - degree 2: invariant coefficient 1; remainder exactly 0.
   - degree 4: invariant coefficient −a²/20; remainder exactly 0.
   - degree 6: invariant coefficient a⁴/840; remainder nonzero.
   - **First nonzero remainder: degree 6.**
   - Proof of vanishing at degrees 2 and 4: the code subtracts c_m(q·q)^m from the exactly expanded polynomial and gets the zero polynomial, coefficient by coefficient in Q(√5). Structurally: the 12 vectors form the vertex set of the icosahedron, a spherical 5-design, so Σ_u (u·q)^j for j ≤ 5 equals its sphere average, which is rotation invariant. This covers j = 2 and 4.
   - At degree 6 the l=6 harmonic component is not killed. The icosahedral group has an invariant harmonic of degree 6 and none of degree 2 or 4.

3. Degree-6 remainder through I6: this is exactly possible. The code verified the identity `remainder = κ |q|⁶ I6(n)` as polynomials.
   - κ = **2a⁴/7875**, found by two routes that agree (coefficient match: 2a⁴/7875; Legendre route: (1/1440)(16/231)(132/25) = 2/7875).
   - Ratio of remainder coefficient to invariant coefficient: (2/7875)/(1/840) = **16/75**.
   - Degrees 2 and 4: remainder coefficient 0, ratio 0.
   - So at degree 6: `ω²_6 = (a⁴/840) |q|⁶ [1 + (16/75) I6(n)]`.
   - At a vertex I6 = 1, which gives (a⁴/840)(91/75) |q|⁶.

4. a → a/2: the degree-2m coefficient is multiplied by 2^{−(2m−2)}: degree 2 by 1, degree 4 by 1/4, degree 6 by 1/16. This applies to the invariant coefficient and to the remainder coefficient alike. The ratio 16/75 does not change.

5. Numerical check: 20 rows (5 directions × a·k ∈ {1/4, 1/8, 1/16, 1/32}) at 80 digits with mpmath. The directions were the vertex u, the z axis, and (1,2,3), (3,−1,2), (1,1,−5), each normalized. I used k = 1 and a = a·k. The bound was fixed in the source before the run.
   - Bound: the Lagrange remainder for 1−cos t after t⁶ is at most t⁸/8!. With |u·n| ≤ 1 this gives |ω² − series₆| ≤ 12 (ak)⁸ / (2a² · 8!).
   - **All 20 differences are inside the bound.** The tightest ratio is about 0.17 (difference/bound). Differences scale as (ak)⁶, as expected from the degree-8 term.
   - Rows are in `results.json` under `numerical_check`.

## Control

Test, fixed before running: the degree-2 polynomial is qᵀMq/4 with M = Σ_u w_u u uᵀ. It is rotationally invariant iff D = M − (tr M/3) I is exactly the zero matrix.

| operator | invariant? | notes |
| --- | --- | --- |
| original | **yes** | D = 0 |
| one direction removed | **no** | D ≠ 0, with entries involving √5 |
| one weight changed 1 → 2 | **no** | D = −(D of removal case) |

The weight-doubled D is the exact negative of the removal D. This follows because the original has D = 0 and the removed or doubled direction contributes ∓uuᵀ.

## Did anything not go as expected

- No disagreement between routes. The two computations of the degree-6 coefficient agree.
- The first Bash call (a heredoc to create `solve.py`) was denied by the permission layer. I wrote the file with the Write tool instead. The final `./py solve.py` ran without trouble.
- `solve.py` contains a duplicated `mp.mp.dps = 80` line. It is harmless.
- Only the original operator has an exactly isotropic degree-2 term. Isotropy at degree 4 was not part of the control; it was shown only for the original.
