**Step 3: the nematic bound, `TrN̄² ≤ 171/2`.** Kawaguchi and Ueda's review states this bound without proof. Write `N̄ = 4I + Q`, with `Q` traceless. Then `TrN̄² = 48 + ‖Q‖²`. For any density matrix,

`‖Q(ρ)‖ = max_E tr(ρ A_E)`, taken over unit real symmetric traceless `E`, where `A_E = Σ Eᵢⱼ{fᵢ, fⱼ}/2`.

Rotating `E` to diagonal form `e = (e₁, e₂, e₃)` does not change `λ_max(A_E)`. So the bound is equivalent to

`λ_max(e₁fₓ² + e₂f_y² + e₃f_z²) ≤ 15/√6` for every unit traceless `e`.

Set `e₃ = 2a/3` and `e₁ − e₂ = 4b`. The constraint becomes `(2/3)a² + 8b² = 1`, and the operator becomes `a(f_z² − 4) + b(f₊² + f₋²)`. The parity of `m` and the flip `m ↦ −m` split it into a zero eigenvalue and three 2×2 blocks:

`M₁ = [[5a, √60·b], [√60·b, −3a + 12b]]`, `M₂` = `M₁` at `−b`, `M₃ = [[0, √240·b], [√240·b, −4a]]`.

The three blocks are bounded one at a time:

- **`M₃`:** `λ_max = −2a + √(30 − 16a²)`. Also `(15/√6 + 2a)² − (30 − 16a²) = 20(a + √6/4)² ≥ 0`, and `15/√6 + 2a ≥ 9/√6 > 0` on the ellipse. So `λ_max(M₃) ≤ 15/√6`.
- **`M₁`:** `λ_max = a + 6b + √(16a² − 48ab + 96b²)`. The right side of `15/√6 − a − 6b ≥ √(…)` is positive, since `a + 6b ≤ √6` on the ellipse, so squaring is valid. Write `N = √((2/3)a² + 8b²)`, which equals 1 on the ellipse. After squaring, and homogenizing with `(15/√6)²N² = 25a² + 300b²`, the bound reads `(3/√6)(a + 6b)·N ≤ a² + 6ab + 24b²`. This holds trivially when `a + 6b ≤ 0`, because the right side is `(a + 3b)² + 15b² ≥ 0`. When `a + 6b > 0`, squaring again gives `(3/2)(a + 6b)²N² ≤ (a² + 6ab + 24b²)²`, and the difference of the two sides is exactly `36b²(a + 2b)² ≥ 0`.
- **`M₂`:** it has `M₁`'s spectrum at `−b`, and the constraint is even in `b`.

So `λ_max ≤ 15/√6` on the whole circle. Tracing the equality cases (`b = 0` with `a > 0` and `a = −2b` with `b > 0`) shows equality only at the three permutations of `(2, −1, −1)/√6`. That is, `E = (3nnᵀ − I)/√6` for a unit vector `n`.

Hence `TrN̄² ≤ 48 + 225/6 = 171/2` for every state. Equality holds iff `u` lies in the top eigenspace of `(3(n·f)² − 12)/√6` for some `n`, that is, iff `u ∈ span{|3,3⟩ₙ, |3,−3⟩ₙ}`.
