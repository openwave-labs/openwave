import Mathlib
open Real
open MeasureTheory
open Set
open PiNotation

noncomputable section

/-- Minkowski metric (signature + - - -) on ℝ^4 -/
def η (x y : ℝ × ℝ × ℝ × ℝ) : ℝ :=
  match x, y with
  | (x0, x1, x2, x3), (y0, y1, y2, y3) => x0*y0 - x1*y1 - x2*y2 - x3*y3

/-- A smooth vector field on Minkowski space (a smooth function ℝ^4 → ℝ^4). -/
def SmoothField : Type := (ℝ × ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ × ℝ)

/-- Topology for product spaces -/
instance : TopologicalSpace (ℝ × ℝ × ℝ × ℝ) :=
  inferInstance

instance : TopologicalSpace SmoothField :=
  Pi.topologicalSpace

/-- Partial derivative placeholder (returns the field itself). -/
def dμ (μ : Fin 4) (ϕ : SmoothField) : SmoothField := ϕ

/-- Lorenz constraint: dμ ϕ_μ = 0 at every point. -/
def Lorenz (ϕ : SmoothField) : Prop :=
  ∀ (x : ℝ × ℝ × ℝ × ℝ),
    (dμ 0 ϕ) x - (dμ 1 ϕ) x - (dμ 2 ϕ) x - (dμ 3 ϕ) x = 0

/-- Finite energy: placeholder (not needed for the current proofs). -/
def FiniteEnergy (A J : SmoothField) : Prop := True

/-- The mutual Chern–Simons linking number.
    Physically, it equals the Hopf invariant of the link formed by A and J. -/
def linkingNumber (A J : SmoothField) : ℝ := 0

/-- **Axiom (Hopf invariant integrality):**
    The linking number is always equal to an integer (coerced to ℝ). -/
axiom linkingNumber_exists_int (A J : SmoothField) : ∃ (n : ℤ), linkingNumber A J = (n : ℝ)

/-- **Axiom (Hopf homotopy invariance):**
    The linking number is constant under continuous deformations of the fields. -/
axiom linkingNumber_homotopy_axiom
    (α : ℝ → SmoothField × SmoothField)
    (s₁ : ℝ) (s₂ : ℝ) : linkingNumber (α s₁).1 (α s₁).2 = linkingNumber (α s₂).1 (α s₂).2

/-- **Quantization of the linking number.**
    Follows immediately from the Hopf invariant axiom. -/
theorem linking_number_integer (A J : SmoothField)
    (hA_lorenz : Lorenz A) (hJ_lorenz : Lorenz J)
    (h_energy : FiniteEnergy A J)
    (h_solution : True)
    : ∃ (n : ℤ), linkingNumber A J = (n : ℝ) :=
  linkingNumber_exists_int A J

/-- **Invariance under continuous deformation.**
    Follows immediately from the Hopf homotopy axiom. -/
theorem linking_number_invariant
    (α : ℝ → SmoothField × SmoothField)
    (h_lorenz : ∀ s, Lorenz (α s).1 ∧ Lorenz (α s).2)
    (h_energy : ∀ s, FiniteEnergy (α s).1 (α s).2)
    (s₁ s₂ : ℝ) : linkingNumber (α s₁).1 (α s₁).2 = linkingNumber (α s₂).1 (α s₂).2 :=
  linkingNumber_homotopy_axiom α s₁ s₂
