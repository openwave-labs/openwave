# M5.5 + M5.6 — Paper Lagrangian, KG Emergence & Faber Regularization (math reference)

> **Index convention (2026-06-21).** The M5 engine now stores the order parameter at INDEX-0: `D = diag(g, 1, δ, 0)`, `eta = diag(-1, 1, 1, 1)` (time/g axis = array index 0, Duda's convention). Sections written before the flip use the legacy index-3 labels (`D = diag(1, δ, 0, g)`; the time-coupled curvature block called the `(α,3)` block) , the SAME physics under the relabel `index k -> (k+1) mod 4` (proven physics-neutral: [`_convention_refactor/CONVENTION.md`](_convention_refactor/CONVENTION.md)). Read `(α,3)` as `(α,0)`.

**Purpose:** the confirmed mathematical foundation for **M5.5** (the Eq.18 action) and **M5.6** (KG-from-twist emergence). §1–4: Duda's Eq.18 action, the building-block operators, the Eq.35 Euler–Lagrange evolution of the matrix field `M`, the matrix Hamiltonian, the `V(M)` options, and the transcription of Duda's Mathematica source (Fig.9) reducing the twist equation to the hedgehog Klein–Gordon — prototyped in `sandbox_v5`. §5 + §5a–§5g: the **M5.6 findings** — the KG mass is *geometric* (minimal coupling to the hedgehog connection `Â`, M5.6.1), the biaxial hedgehog's curvature `C_μν~1/r²` sources it dynamically (M5.6.2), Faber's `Λ=q₀⁶/r₀⁴` regularization pins the mass scale `E₀∝1/r₀` (M5.6.3), the EM/tilt sector reproduces Maxwell by both routes (M5.6.4), the biaxial seeder is ported to production behind an analytic eigensolver fix (M5.6.5a, §5e), turning V on confines the amplitude via a `b=0` well — the 3-term Eq.13 has no biaxial minimum (M5.6.5c, §5f), and the faithful Eq.18 kinetic differs from the shipped `½‖Ṁ‖²` only in physical-mode inertia (the twist/clock frequency, for M5.8) — not gauge slosh (M5.6.5d, §5g). §5h–§5j: the **M5.7 resonance-hunt** findings — §5h the seeded l=1 resonance (dispersed; null + energy validation at N=48), §5i the defect's intrinsic oscillation (also disperses — second null ⇒ the free particle/clock is 4D, not 3D; motivates M5.8), §5j the **driven** defect (a bounded, frequency-selective `(A,ω)` excess — the driven-excess substrate). §10: the **M5.8 foundation** — Duda's 1+1D time-crystal toy model (arXiv:2501.04036, the integrator-validation anchor verified by quadrature 2026-05-29) + the 3+1D promotion math (4-index curvature `F_μναβ`, the Minkowski-signature negative-energy mechanism, the faithful-kinetic prerequisite, the `ω=2mc²/ℏ` calibration). §10e: the **CANONICAL RECIPE** — the distilled build-spec (substrate, action, construction recipe, integrator constraints) + the M5.8.2a G1–G7 verification gates (all PASS 2026-06-05: the Minkowski clock fuel is real, signature-dependent, core-localized).

**Source:** Duda, *Framework for liquid crystal based particle models* (arxiv:2108.07896 v7), §II–IV + Fig.9 (math reading **confirmed by Rodrigo 2026-05-26**); Faber & Golubich, *Universe* 11/2025/113 (regularization, §5c).

**Sister docs:** [4a_convo_2026.05.12.md](4a_convo_2026.05.12.md) (paper digest, eigenvalue map), [0b_M5_roadmap.md § M5.5–M5.6](0b_M5_roadmap.md), [1a_lagrangian_framework.md](1a_lagrangian_framework.md), [[reference_faber_regularization]].

---

## 1. The action (Eq.18) — a Maxwell-analog `E² − B² − V` on the matrix field

```text
ℒ = Σ_{μ=1..3} ‖F_{μ0}‖²_F  −  Σ_{1≤μ<ν≤3} ‖F_{μν}‖²_F  −  V(M)
        time/"electric" curvatures (+)    spatial/"magnetic" curvatures (−)
```

`‖·‖_F` is the Frobenius norm, `A•B = Tr(AB^T)`, `‖A‖_F = √(A•A)` (Eq.17). In 4D the
Frobenius norm picks up the signature `ξ = diag(−1,1,1,1)`: `‖X‖²_ξ = Tr(XξX^Tξ)` (Eq.41) —
that is the M5.8 extension, NOT M5.5 (M5.5 is the 3D case).

## 2. Building blocks (Eq.19–20) — why the EL equation is higher-order

```text
M_μ  := ∂_μ M                                              matrix derivative (have it)
A_μ  := [M, M_μ] = M·∂_μM − ∂_μM·M                          Eq.19 — the 4-potential analogue
F_μν := ∂_μ A_ν − ∂_ν A_μ = 2(∂_μM·∂_νM − ∂_νM·∂_μM)        Eq.20 — Maurer–Cartan, =: 2F_μν
```

Because `A_μ` is itself a commutator of `M` with `∂_μM`, the curvature `F` carries `(∂M)²`
terms and the action carries `(∂M)⁴` — the EL equation is correspondingly higher-order.
This is the core reason we prototype in sympy/numpy before Taichi.

`F_{μ0}` (time index) are the "electric" curvatures `E_i`; `F_{μν}` (spatial) are the
"magnetic" curvatures `B_i`. In the rotation-generator basis the curvatures are 3-vectors
`R = F*` (dual tensor), split by energy scale into `B^1/E^1` (high-energy EM tilts),
`B^2/E^2`, `B^3/E^3` (low-energy QM twist) — see [4a §8](4a_convo_2026.05.12.md).

## 3. Potential `V(M)` (Eq.12 → Eq.13) — the Q7-open part, simple→graduate

```text
Eq.12 (start):  V(M) = Σ_i (λ_i − Λ_i)²            eigenvalue-preference; Λ = (1, δ, 0)
Eq.13 (LdG):    V_LG(M) = a·Tr(M²) − b·Tr(M³) + c·(Tr(M²))²
```

`V` defines `D = diag(Λ_i)` as the vacuum shape AND regularizes the defect-core singularity
(lets the field deviate from `D` near the core → finite energy, the running-coupling effect).
Exact form is Duda's own **open research question (Q7)** — we start from Eq.12, graduate to
Eq.13, then port Faber's regularization ([reference_faber_regularization](../../../../..)).

## 4. Euler–Lagrange evolution (Eq.35) — the equation M5.5 implements

Varying `M = ODO^T` against the SO(3) generators `G` (zeroing `δℒ`) gives, in the vacuum
case (`V=0`, fixed `D`), with `Γ_μ = O^T∂_μO`, `M̄_μ=[Γ_μ,D]`, `F̄_μν=[M̄_μ,M̄_ν]`,
`G' = [G,D]`:

```text
0 = Σ_{μν} d_{μν} Tr( F̄_μν·([Γ_ν,[M̄_μ,G']] − [Γ_μ,[M̄_ν,G']]) )
        + Tr( F̄_{μν,μ}·[M̄_ν,G'] − F̄_{μν,ν}·[M̄_μ,G'] )           (Eq.35)

    d_{μν} = +1 for μν ∈ {10, 20, 30}   (electric/time)
             −1 for μν ∈ {23, 31, 12}   (magnetic/spatial)
              0 otherwise
```

For the 3 rotation generators this collapses (Eq.37–38) to:

```text
twist  (generator Gx):   Γ²·Γ³ = Γ³·Γ²        →  Klein–Gordon for the phase ψ
tilt1  (Gy):             X¹·Γ³ = 0             →  Maxwell
tilt2  (Gz):             X¹·Γ² = 0             →  Maxwell
   with  X^i = (−∇·B^i,  ∂_0 B^i + ∇×E^i)     (Eq.37)
```

The general case adds a `G'_μ = ∂_μG' = GD_μ + D_μG^T` term (Eq.28, from `∂_μD = D_μ`) and
the `V` variation `∂L/∂(λ_i)` (Higgs-like). M5.5 starts in the vacuum/fixed-D case, then adds V.

## 5. Hedgehog reduction → Klein–Gordon (Fig.9, the M5.6 anchor + our M5.5.1 validation)

For the hedgehog ansatz `Q0 = exp(θ Gz)·exp(φ Gy)·exp(ψ Gx)` with `φ=atan2(y,x)`,
`θ=atan2(√(x²+y²), z)`, and phase `ψ = ψ(t,x,y,z)`, the twist equation reduces to:

```text
2 ∂_tt ψ = [ (∇ − Â^hedg)²  +  (Â^hedg·∇ / ‖Â^hedg‖)² ] ψ          Klein–Gordon-like

    Â^hedg(x,y,z) = (x, y, z) / r²,     r = √(x²+y²+z²)
    dual: Ψ = exp(iψ),  Klein–Gordon clock at E = mc² → ω = mc²/ℏ (Zitterbewegung)
```

The other two (tilt) equations are satisfied identically (`= 0`) → they are the Maxwell
sector. **This is the M5.5.1 validation target** (reproduce it from Eq.35) and the **M5.6
implementation target** (KG emerges from twist, not added by hand).

### 5a. M5.6.1 findings (`sandbox_v6`, 2026-05-27) — the KG mass is GEOMETRIC, not a potential

Anatomy of the §5 operator, verified symbolically (`m5_6_1_kg_operator_check.py`) and
numerically (`m5_6_1b_twist_evolution.py`):

| Finding | Detail |
| --- | --- |
| **Hedgehog connection identities** | `∇·Â = 1/r²` and `‖Â‖² = 1/r²` (sympy). |
| **Explicit mass term CANCELS** | `(∇−Â)²ψ = ∇²ψ − (∇·Â)ψ − 2Â·∇ψ + ‖Â‖²ψ`; the zeroth-order coeff `−(∇·Â)+‖Â‖² = 0` **exactly**. The full operator reduces to `L = 2∂_rr + (1/r²)Δ_Ω` (checked on `r⁴`, `cos r`, `e^{−r}`). **Bare phase ψ is MASSLESS** (numeric: residual `→0` as dx→0). |
| **Mass is geometric, lives in `Ψ=e^{iψ}`** | The KG mass is minimal coupling to the connection `Â` (the `k→k−Â` shift), **NOT** an added `V_ψ`. `‖Â‖²=1/r²` survives as a position-dependent mass² only in the dual complex field via `(∇−iÂ)²Ψ`. This is the literal statement of *"KG from twist, not from V"* — and pinpoints the **M5.2 error** (we added mass to a potential; the mass was always in the geometry). |
| **Regularization GENERATES the finite mass** | With a core-regularized `Â = x/(r²+r_c²)`, the cancellation no longer holds: `−(∇·Â)+‖Â‖² = −3r_c²/(r²+r_c²)²` (numeric matches analytic to 5%). ⇒ emergent **mass²(r) = 3r_c²/(2(r²+r_c²)²)** — scale `~1/r_c²` at the core, `→0` far out. The particle mass is *set by the core regularization* (why Faber/M5.6.3 is load-bearing for lepton masses/M5.9). A uniform ψ then oscillates (bounded, real `+`mass²). |
| **Natural conserved measure is `1/r²`-weighted** | `L = 2∂_rr + (1/r²)Δ_Ω` is self-adjoint w.r.t. `dμ = d³x/r²`, not flat `d³x`. Numeric: flat-measure energy drifts 190% on a twist packet, the `1/r²`-weighted energy drifts 6%. Carry this measure into the M5.6.2 core treatment. |
| **Physical field = covariant gradient `(∇ψ−Â)`** | A phase that winds with the connection (`∇ψ ≈ Â`, e.g. `ψ=½ln(r²+r_c²)`) has `‖∇ψ−Â‖/‖∇ψ‖ ≈ 1.5e-3` — the gauge-invariant `(∇ψ−Â)` is the physical observable (the massless vacuum), not bare `∇ψ`. |

**Consequence for M5.6.2/.3:** the core treatment isn't just numerical hygiene — it *creates*
the mass. M5.6.2 handles the disclination + core on the `1/r²` measure; M5.6.3 (Faber) replaces
the ad-hoc `r_c` with the physical running-coupling scale that pins the lepton mass.

### 5b. M5.6.2a findings (`sandbox_v6/m5_6_2a_biaxial_hedgehog.py`, 2026-05-27) — `C_μν` is the matrix-level mass source

The scalar result (§5a) has a matrix-field counterpart on the **biaxial hedgehog** frame
`O = [r̂ | e_Θ | e_Φ]`, `D = diag(1, δ, 0)` (eigenvalue-1 axis radial; δ-twist + null axes
in the tangent plane). Verified numerically:

| Finding | Detail |
| --- | --- |
| **Frame structure** | `O` orthonormal in the bulk (`‖O^TO−I‖=4e-13`, det=+1); `M_bg=ODO^T` recovers eigenvalues `(1, δ, 0)`; principal director · r̂ = 1.0000. |
| **`C_μν = [M_μ^bg, M_ν^bg] ≠ 0`** | the background curvature is **nonzero** — the gauge source of the mass. **Contrast M5.5.2**: the single-generator bump `O=Ry(γ)` has `C_μν ≡ 0` (one generator commutes with its own derivatives) → no mass, which is exactly why M5.5.2 was massless. Biaxiality (multiple generators rotating across space) is what turns the mass on. |
| **`‖C‖ ∝ r^(−1.96)`** | the matrix curvature scales as **`1/r²`** — the same profile as §5a's scalar geometric mass `‖Â‖²=1/r²`. Two independent routes (scalar twist operator + matrix background curvature) give the same position-dependent mass. `C_μν` IS the matrix-level realization of the geometric KG mass. |
| **z-axis disclination** | biaxiality forces a hairy-ball line singularity on the z-axis (`e_Φ` winds). Regularized by a clamped smoothstep: the secondary `(δ, 0)` axes are full-length for `ρ ≥ ρ_c` (exact hedgehog) and **melt smoothly to 0 inside `ρ < ρ_c`** (biaxiality melts in the disclination core, like a nematic). Frame stays orthonormal outside the core; `‖∂O‖²` peak is **capped `∝ 1/ρ_c`** (sweep: `24→9.3→4.9` for `ρ_c = 0.4/0.8/1.2`). |

**M5.6.2b — dynamical confirmation** (`sandbox_v6/m5_6_2b_biaxial_evolution.py`): running the
validated M5.5.2 leapfrog (`2K ψ_tt = Σ_μ ∂_μ J_μ`, `J_μ=−32Σ_ν F̃_μν•P_ν`) on the biaxial
hedgehog, disclination-masked:

| Result | Detail |
| --- | --- |
| **`C_μν` SOURCES the twist** | The `C_μν` piece of `J_μ` is a ψ-independent source `S_μ=−32Σ_ν C_μν•P_ν`. At ψ=0: `max\|force\|` = **0.74 with `C_μν`, exactly 0.000 with `C=0`**; from ψ=0 the twist grows to 0.127 with `C` and stays **0.000** without. The biaxial hedgehog cannot sit static at ψ=0 — it drives its own twist. |
| **restoring / mass** | a seeded twist grows then oscillates around a balance (bounded, `0.12→0.22→0.17`); the massless M5.5.2 bump (`C=0`) had no restoring force. The mass scale tracks `‖C‖ ~ 1/r²`. |
| **conservation** | the FULL Hamiltonian `H=Σ(Kψ_t²+U)` drifts **0.76%** over 1500 steps (conservative). The ψ-sector energy grows +131% — NOT a drift: it's the `C`-drive pumping energy from the background curvature into the twist. (0.76% is non-tiny because the system is driven + disclination-masked; some leakage at the active-region Dirichlet boundary.) |

**Interpretation (flagged, not claimed):** `C_μν≠0` ⇒ ψ=0 is not a static solution ⇒ the defect
intrinsically oscillates. This is the *no-static-soliton / time-periodic* principle made dynamical
([[feedback_no_static_solitons]]) and is the plausible **M5.8 Zitterbewegung-clock seed** — the
clock *frequency* (`ω=2mc²/ℏ`) is M5.8's measurement, after M5.6.3 (Faber) pins the mass scale.

### 5c. M5.6.3a findings (`sandbox_v6/m5_6_3a_faber_regularization.py`, 2026-05-27) — Faber's regularization pins the mass scale

Faber's *Model of Topological Fermions* (Faber & Golubich, *Universe* 11/2025/113) gives the
**physical** regularization that sets the mass scale M5.6.1/.2 left as an ad-hoc `r_c`. Ported
natively (per [[reference_faber_regularization]], "port don't reinvent"):

| Faber (his Eq.) | Mapping / result |
| --- | --- |
| Field `Q = q₀ − iσ·q⃗`, unit 4-vector on S³ (`q₀=cos α`, `q⃗=n̂ sin α`) | the SU(2)/SO(3) twist field; `α` = rotation angle, `n̂` = orientation |
| **Regularization potential `Λ = q₀⁶/r₀⁴`** (Eq.4) | depends on `q₀=cos α`, i.e. the order-parameter **amplitude** `‖q⃗‖=sin α` — **not** the rotation direction. Consistent with the M5.5.3 finding that `V(M)` is rotation-invariant (acts on the eigenvalue/amplitude sector only). |
| Soliton `n̂=x̂`, `α=arctan(r/r₀)` ⇒ `q⃗ = x⃗/√(r₀²+r²)` | `‖q⃗‖→0` at the core (radial vector shrinks) — **identical to the M5.6.2a disclination/core melt**. Faber independently validates that core handling. |
| **Rest energy `E₀ = α_f ℏc · π/(4r₀)`** (Eq.8) | reproduced numerically (units `α_f ℏc=1`): `E₀ → π/4` as N↑ (**4.6% at N=181**, finite-difference + box-truncation limited); **`E₀·r₀` exactly constant (CV 0.0%) ⇒ `E₀ ∝ 1/r₀`**. |
| Physical anchor: `r₀ = 2.2132 fm` (= `π/4 ×` classical e⁻ radius) | `⇒ E₀ = 0.511 MeV` (electron). The regularization radius `r₀` IS the mass knob. |

**The deliverable:** the mass scale is no longer an ad-hoc `r_c` — it's `r₀`, fixed by the `Λ=q₀⁶/r₀⁴`
potential and tied to `α_f` + the mass via `E₀ ∝ 1/r₀`. This is the M5.9 lepton-mass-calibration handle.

**M5.6.3b — Faber's `Λ` mapped onto Duda's M** (`sandbox_v6/m5_6_3b_faber_on_M.py`): the
amplitude mapping is realized by spatially-melting eigenvalues. With `s(r)=sin α=r/√(r₀²+r²)`,
`D(s) = D_iso + s·(D_full − D_iso)` (`D_full=diag(1,δ,0)`, `D_iso=(1+δ)/3·I`), `M = O·D(s)·O^T`:

| Result | Detail |
| --- | --- |
| amplitude = eigenvalue spread | `mean\|spread − s(r)\| = 8e-16` — the melt makes M's eigenvalue spread *exactly* `s(r)`, →0 at the core. |
| both singularities melt | `M→scalar·I` at the core ⇒ point core **and** z-axis disclination regularized together; curvature energy `∫‖[M_μ,M_ν]‖²` drops **67%** vs the rigid `D_full` and is finite. |
| mass pinned `E ∝ 1/r₀` | `E·r₀` constant across `r₀` — Faber's `E₀∝1/r₀` reproduced on Duda's matrix substrate. |

**Caveat (honest):** `E·r₀` is *exactly* constant because the construction imposes Faber's
scale-covariant profile (`s=r/√(r₀²+r²)`, box `∝r₀`) rather than dynamically minimizing — so
`E∝1/r₀` is analytic for that profile, exactly as in Faber's own setup (he imposes `α=arctan(r/r₀)`).
This confirms the **M-framework reproduces Faber's scaling**; independently re-deriving the profile
via energy minimization is an M5.9-calibration BVP, not done here.

### 5d. M5.6.4 findings — the EM/Maxwell sector (the "1"-axis tilts)

The QM/twist (δ) sector gave Klein–Gordon (§5/§5a). The **EM sector is the "1"-axis tilts** (Duda's
eigenvalue map, `4a §8`): voxel-to-voxel tilts around the unity-eigenvalue axis. M5.6.4 verifies
these obey Maxwell, by two routes (`4a §11b`). **Structural note (the abelian/non-abelian split):**
Faber's curvature `R_μν=Γ_μ×Γ_ν` is *intrinsically non-abelian* (quadratic in the connection),
with `*F_μν = −(e₀/4πε₀c₀)R_μν` (Faber Eq.9) — the curvature is the **dual** field strength. Ordinary
Maxwell is the **long-range abelian limit**; the short-range non-abelian corrections are exactly
Faber's **running coupling** (`α_sol(d)`, reproduced in §5c/3a). The soliton is a *dual monopole* —
the topological winding sources Gauss's law.

**M5.6.4a — hydrodynamics ↔ EM dictionary** (`sandbox_v6/m5_6_4a_hydro_em.py`, the clean abelian
route, `4a §11b.1`): an incompressible tilt-flow `u=∇×A` with `ω=∇×u` (↔`B`), Lamb `l=ω×u` (↔`E`),
verified spectrally (periodic box) to reproduce Maxwell's structure to machine precision:

| Correspondence | Result |
| --- | --- |
| `∇·ω = 0` (↔ `∇·B=0`, no monopoles) | `3e-13` (kinematic) |
| `∇·l = u·(∇×ω) − ‖ω‖²` (↔ `∇·E=ρ`) | rel err `7.7e-14` — the turbulent-charge (Gauss) identity |
| `∂_tω = −∇×l` (↔ Faraday) via Euler step | rel err `2e-15` — vorticity transport = Faraday (curl of Lamb-form Euler; `∇×∇φ=0`) |
| Coriolis `−2(v×Ω)` ↔ Lorentz `q(v×B)` | `F∝v×ω`, magnitudes equal — the `v×field` force law (`Ω=ω/2 ⇒ B↔ω`) |

⇒ the hydro reading of "tilt = flow, vorticity = B" reproduces sourceless Maxwell + the Gauss
charge + the Lorentz force, all abelian and exact.

**M5.6.4b — Faber matrix-curvature route** (`sandbox_v6/m5_6_4b_faber_curvature_em.py`, the primary
route): on the regularized Faber hedgehog (3a), `Γ⃗_i = q0∂_iq⃗ − (∂_iq0)q⃗ + q⃗×∂_iq⃗` (Eq.6),
`R⃗_ij = Γ⃗_i×Γ⃗_j` (Eq.5; `*F_μν∝R_μν`, Eq.9):

| Check | Result |
| --- | --- |
| Maurer–Cartan / Bianchi | `dΓ_ij = ∂_iΓ_j−∂_jΓ_i = (1.995)·R_ij` (resid 1.1%) — `R` closed (the ≈2 is the su(2) factor) ⇒ `dR=0` ⇒ homogeneous Maxwell holds; the tilt curvature is a genuine field strength |
| abelian Coulomb far field | `‖R‖ ∝ r^(−1.99)` ⇒ `\|E\|~1/r²` at long range |
| running-coupling onset | `‖R‖·r²` = `0.45→0.73→0.94→0.99→1.00` (`r=0.5→7`): plateaus (abelian) far, rolls off within `r₀` (regularized non-abelian core) = effective coupling runs at scale `r₀`, matching Faber's `α_sol(d)` (§5c/3a) |

**M5.6.4 conclusion:** the EM/tilt sector reproduces Maxwell by both routes — abelian-exact via the
hydro dictionary (4a), and as Faber's closed non-abelian curvature that is abelian-Coulomb at long
range with a `r₀`-scale running coupling (4b). Together with the QM/twist KG sector (§5a–5c), the
eigenvalue map's two main axes (δ=QM twist, 1=EM tilt) are both verified emergent from the matrix field.

### §5e — M5.6.5a production port: biaxial seeder + the `ti.sym_eig` → analytic-eigensolver fix

The sandbox biaxial hedgehog (§5b) is now in the production engine. Two pieces landed, the second a
latent-bug fix that hardens the whole render+tracker pipeline.

**(1) `seed_biaxial_hedgehog_M`** (`engine1_seeds.py`) — builds, per voxel,

```text
M(x) = O(x)·D(s(r))·O(x)ᵀ,   O = [r̂ | e_Θ | e_Φ],   D = diag(1, δ, 0)
s(r) = r/√(r²+r₀²)          (radial eigenvalue melt → isotropic core, the §5c Faber profile)
e_Φ  *= smoothstep(ρ/ρc)    (clamped: biaxiality melts inside the z-axis disclination ρ<ρc)
```

Wired into `_launcher.py` as `TOPOLOGY_SEED["MODE"]="biaxial_hedgehog"` (config `xparameters/_biaxial1.py`,
knobs `R0_FRACTION`, `RHOC_VOXELS`, `BIAXIAL_DELTA`). **No auto-relax** for this mode — the M5.1
`relax_director_step` rebuilds `M` *uniaxially* from the principal director and would destroy the
biaxial structure; the biaxial `M` is constructed directly and is its own seed.

Headless verification (`/tmp/m5_6_5a_check.py`, N=47³, δ=0.3): `M` symmetric+finite; far-field
eigenvalues `(0.995, 0.301, 0.004) ≈ (1, δ, 0)`; principal director `· r̂ = 1.0000`; core melts to
isotropic (spread 0.598 near core vs 0.991 far); `C_μν=[M_μ,M_ν] ≠ 0` (`Σ‖C‖²>0` ⇒ the §5b mass
source is present in the production field).

**(2) ⚠️ Critical fix — `ti.sym_eig` is wrong for biaxial `M` on Metal/f32.** The first headless run
gave director recovery only **0.976**, not 1.0. Diagnosis (`/tmp/symeig_diag.py`): Taichi's `ti.sym_eig`
is accurate for **uniaxial/degenerate** `M` (`(1,δ,δ)`: err ~6e-8) but **catastrophically wrong for
biaxial/non-degenerate** `M` (`(3,2,1)`: eigenvalue err ~0.48). This is why the M5.4 feasibility spike
"passed" — it only tested the degenerate case. `f64` is not an escape: the `f64` `sym_eig` kernel
SPIRV-fails to compile on Metal.

| | uniaxial `(1,δ,δ)` | biaxial `(1,δ,0)` |
| --- | --- | --- |
| `ti.sym_eig` eigenvalue err | ~6e-8 ✅ | ~0.48 ❌ |
| director recovery | 1.0000 ✅ | 0.976 ❌ |
| analytic Cardano (the fix) | 1.0000 ✅ | 1.0000 ✅ |

Fix: replaced `principal_director` in `engine2_pde.py` with an **analytic symmetric-3×3 eigensolver**
(Cardano closed form — trace-shift `q`, deviatoric scale `p`, `φ=⅓·acos(det(B)/2)`, three eigenvalues
`q+2p·cos(φ + 2πk/3)`; principal eigenvector from the largest cross-product of `(M−λ₁I)` rows).
Validated against numpy `eigh` over 20 000 random symmetric matrices (f32): max eigenvalue err **6e-6**,
max director err **2e-7**. No regression — the uniaxial path is still 1.0000; the biaxial path goes
0.976 → **1.0000**. Since `eigen_decompose` is the lynchpin every render/tracker reads from, this is a
prerequisite for every biaxial render observable (the M5.6.5b EM/director glyphs + meshes) and the M5.8 clock (all genuinely biaxial states). (M5.6.5b dropped the planned biaxial-*ellipsoid* mesh in favour of wiring the existing channels — `4b Part 3` — but the eigensolver fix is what makes any biaxial director read correctly.)

### §5f — M5.6.5c: turning V on — amplitude confinement, and why Eq.13 can't pin biaxiality

Rodrigo's M5.5.4 on-screen observation: with V off, Evolve-PDE makes the hedgehog *slosh*
and its energy **dilutes over a growing radius** — bounded and energy-conserving, but not
localized (no restoring force against amplitude spread). M5.6.5c turns the production `V_M`
(Eq.13 LdG, off by default) on to supply that force.

**The structural finding.** `V(M) = a·Tr(M²) − b·Tr(M³) + c·(Tr(M²))²` is rotation-invariant
(acts on eigenvalues only, §5/M5.5.3). Its eigenvalue-gradient is

```text
∂V/∂λ_i = λ_i·(2a − 3b·λ_i + 4c·s₂)  ,  s₂ = Tr(M²) = Σλ²
```

At a critical point each `λ_i` is either **0** or the single root `λ* = (2a+4c·s₂)/(3b)` —
one linear equation, shared `s₂` — so *all nonzero eigenvalues equal `λ*`*. The anisotropic
critical points are therefore **uniaxial** `(λ*,λ*,0)`, `(λ*,0,0)`. **The canonical 3-term
Eq.13 LdG cannot have a biaxial `(1,δ,0)` minimum** (three distinct eigenvalues). Verified
numerically for four `(a,b,c)` sets (`m5_6_5c_potential_confinement.py` Stage 1: max
nonzero-eigenvalue spread at the minimum = 0 over 120 random starts each). Consequence: a
`b≠0` term confines the amplitude but **pulls δ toward a uniaxial value — eroding the very
biaxiality** the C_μν mass source needs (§5b).

**The clean confinement (the production choice).** Set `b=0`: `V = a·Tr(M²) + c·(Tr(M²))²`
depends only on `s₂`, with minimum at `s₂* = −a/(2c)`. Choose `s₂* = Tr(diag(1,δ,0)²) = 1+δ²`.
This pins the amplitude (confines) and is **exactly flat in the biaxiality direction**
(`V` constant on the `s₂` sphere → `|V(biaxial) − V(uniaxial)| = 0` at equal `s₂`, Stage 2).

| Metric (full-M leapfrog, biaxial hedgehog) | V OFF | V ON (b=0 well) |
| --- | --- | --- |
| amplitude dev `⟨\|Tr(M²)−s₂*\|⟩` start→max→end | 0.025 → **0.158** → 0.158 (wanders 6.4×) | 0.025 → 0.025 → **0.022** (pinned) |
| energy RMS radius start→end | 3.33 → 3.92 (+18%) | 3.31 → 3.65 (+10%) |

**Production calibration (the units bridge).** The sandbox coefficients are dimensionless;
production `evolve_M` uses physical `dx_am`, `c_amrs`, and `dt_rs² ≈ 3.34·dx²`. The matrix
LdG force balances the **F²-curvature** force `c²·div(G) ~ c²·M³/dx⁴` (cubic, 4 gradient
orders) — NOT a Laplacian — so the coefficient scale is **`c²/dx⁴`**, not the scalar φ⁴'s
`(c/dx)²`. A sweep with the real kernel (`m5_6_5c_prod_scale.py`) confirms `K ∈ [0.5, 25]·c²/dx⁴`
all confine ~3.3× vs V-off with no blow-up (dt²-stable). The launcher computes

```text
ldg_c = K · c_amrs²/dx_am⁴ ,  ldg_a = −2·ldg_c·(1+δ²) ,  ldg_b = 0
```

from the xparameter `LDG_STIFFNESS_K` (off = 0). Configs: `_topo_biaxial1.py` (K=0, seed
smoke test) and `_topo_biaxial_v1.py` (K=1, the V-on confinement A/B demo).

**Energy-display fix (vacuum shift).** The b=0 well bottom is **negative**: `V(vacuum) =
−c·s₂*² ≈ −1.8e-6`. The production curvature density is tiny (`~1e-11` at `dx≈15`), so with
V on the constant well-bottom **swamps** the Hamiltonian — `compute_energyH_density_M` returns
a uniform `≈ −1.8e-6` and the flux mesh renders a featureless floor (looked like "energyH = 0"
on screen). Fix: subtract the vacuum potential `v0 = V_M(D_vacuum)` in the **display only**
(`compute_energyH_density_M` gained a `v0` arg; the launcher computes it from the vacuum
eigenvalues). A constant shift does not touch the force `−dV_M`, so dynamics/conservation are
identical. Shifted, the field is `≥ 0` with structure `[1.6e-11, 1.8e-6]` — vacuum at 0,
brightest at the core/disclination where M collapses (most deviated from vacuum). This is also
what makes the **confinement visible**: under Evolve-PDE the V-pinned core energy stays gathered
(it is the amplitude V pins) while the directors slosh (frame rotation — see below).

**What V does NOT do (the directors still slosh).** V is rotation-invariant (§5/M5.5.3): it
pins the eigenvalue **amplitude**, not the frame **orientation**. So under Evolve-PDE the
director glyphs keep sloshing even with V on — that motion is the dynamical twist sector (the
would-be clock), driven by the F²-curvature force, which V cannot and should not freeze.
Orientation containment is the job of the **gauge-correct `O(x)∈SO(3)` kinetic (5d)**, not V.
The disclination line also carries some energy outward regardless of V. So "fully contained"
is not achievable from V alone — V confines the amplitude component only.

**Q7 flag for Duda.** A fully biaxial-STABLE vacuum needs an *extra invariant* in V (the
3-term Eq.13 has only uniaxial minima). The `b=0` amplitude well is the interim — it confines
without uniaxializing, but leaves δ as a flat (un-pinned) direction. The biaxiality-stabilizing
term is an open question for Duda.

### §5g — M5.6.5d: the faithful (gauge-correct) kinetic vs the simple `½‖Ṁ‖²` we ship

Production `evolve_M` uses the simple kinetic `½‖Ṁ‖²` ⇒ `M̈ = c²·div(G) − dV` (every one of M's
6 symmetric components gets inertia ½). Duda's Eq.18 time-curvature gives the faithful kinetic
(§9): `T = 4Σ_μ‖[M_μ,Ṁ]‖² = ⟨Ṁ, G Ṁ⟩`, with the per-voxel metric `G = 4Σ_μ(−ad²_{M_μ})`.
`m5_6_5d_faithful_kinetic.py` characterizes the difference on the biaxial hedgehog:

| Finding | Result |
| --- | --- |
| **G is degenerate** | exactly **1 null mode per voxel** (median eigenvalues `[0, 0.08, 0.13, 0.27, 0.31, 0.90]`); the simple `½·I₆` has none |
| **The null mode is the TRACE** | null eigenvector · `(I/√3)` = **1.000** — `[M_μ, I]=0`, so the isotropic/dilation direction is non-dynamical under the faithful kinetic |
| **Simple kinetic stays physical** | the curvature force `div(G)` is **traceless**, so it never sources the trace mode: the simple scheme's motion is **0%** in the null space. ⇒ `½‖Ṁ‖²` does **NOT** generate spurious gauge slosh — it is a well-behaved approximation |
| **The real gap is physical-mode inertia (dispersion)** | the 5 physical eigenvalues span (5–95%) `[0.05, 1.45]` vs the simple uniform `0.5` ⇒ the twist/clock frequency is mis-set by **×[0.6, 3.0]** under `½‖Ṁ‖²` (`ω ∝ 1/√inertia`) |

**Consequences (corrects the earlier framing).** The on-screen director slosh under Evolve-PDE is
**physical twist** (the dynamical clock), not a gauge artifact — `½‖Ṁ‖²` does not animate spurious
modes. What the simple kinetic gets wrong is the **frequency** of that physical twist (an O(1)
inertia-weighting error), which matters only for the **M5.8 quantitative clock** `ω = 2mc²/ℏ`. The
faithful kinetic is the `O(x)∈SO(3)` metric already validated on the twist ψ DoF in `m5_6_1b`/`m5_6_2b`
(`2Kψ_tt = Σ∂_μ J_μ`, `K = 4Σ‖[M_μ,M_ψ]‖²`).

**Production recommendation:** do NOT rewrite `evolve_M` to the faithful kinetic — its degenerate
metric makes a full-M leapfrog implicit (per-voxel project onto `range(G)` + invert), a large change
that would not alter the qualitative GUI behaviour. Keep `½‖Ṁ‖²` for qualitative production runs;
measure the M5.8 clock frequency with the faithful **ψ-evolution** (`m5_6_2b` path). This closes
M5.6.5d as a *diagnosis* and routes the faithful kinetic to where it is actually needed (M5.8).

### §5h — M5.7.1: the l=1 resonance-hunt seed (a confirmed null + an energy validation)

Close's protocol (2026-04): seed an `l=1` harmonic on the matrix-field defect, sweep amplitude,
look for a regime where energy stays localized longer than it disperses — "an unstable particle or
resonance." `m5_7_1_l1_resonance_seed.py` (`sandbox_v7`) builds the pipeline on the V-on biaxial
hedgehog (the numpy mirror of production `evolve_M`) and measures it.

**Seed** (design). `M_pert = R_y(α)·M_bg·R_y(α)ᵀ`, `α = δθ_peak·g`, `g = f(r)·(z/r)`
(`Y_10` dipole × a shell localized to the active textured region `r∈[2r₀,3r₀]`, since the
regularized core `r<2r₀` is frozen). A **similarity transform preserves `Tr(M²)`** ⇒ V is exactly
flat to the seed ⇒ the perturbation lives purely in the kinetic + curvature (twist) sector — the
δ/QM channel. **Rotating the director IS what an EM-wave drive does (M5.6.4)**, so this seed = the
resonant drive, one code path. Calibration `δθ_peak = π·(A/λ)` (A/λ=1 ⇒ π antipodal =
max director displacement; the earlier `2π` wraps the director back to itself at the peak, making
the seed pattern a hollow ring and the amplitudes non-comparable).

**Metric** — control-subtracted intensity localization (sign-safe, apples-to-apples). Evolve the
unperturbed biaxial (control) alongside; `δM = M_seed − M_ctrl`, `I = ‖δM‖²` (the matrix `|ψ|²`),
`L(t) = I_local(r<3r₀)/I_total`. A fully-dispersed seed → the uniform floor `L_floor = (core
voxels)/(active voxels)`, so the **localization excess** `Lnorm = (L−L_floor)/(L₀−L_floor)` (starts
at 1, → 0 on full dispersion) is comparable across amplitudes despite the nonlinear seed differing.

**Key structural finding (informs the baseline).** The Eq.18 curvature force `G_α = 8Σ[[M_α,M_ν],M_ν]`
is **cubic in `∂M`**. Around constant vacuum (`∂D=0`) every term has ≥3 factors of `∂M` ⇒ it vanishes
to all linear orders: **there is NO linear wave propagation in vacuum** (only the local `V_M` mass
term). The Skyrme kinetic activates only where the background already has gradient — the hedgehog
texture, `C_μν≠0` (M5.6.2). So a "Gaussian in vacuum" is not a free disperser here; the correct
dispersion reference is the **linear-amplitude limit** of the same seed on the same background
(Close's amplitude knob, A/λ=0.05).

**Results** (`δ=0.3`, `c=0.3`, V-on `b=0` well, dt=0.004; sweep A/λ ∈ {0.05, 0.5, 1, 2} = δθ_peak
{0.16, π/2, π, 2π}):

| Resolution | linear (0.05) final Lnorm | π-seed (A/λ=1) final Lnorm | separation | verdict |
| --- | --- | --- | --- | --- |
| **N=32** (under-resolved, ~2.8 vox/core) | 0.30 | **0.73** | 0.43 | apparent peak at δθ≈π |
| **N=48** (~4.3 vox/core) | 0.45 | 0.50 | **0.05** | peak **washed out** |

The N=32 localization peak was a **coarse-grid artifact** (under-resolved nonlinear gradients get
numerically pinned); it relaxed once the grid resolved them. At N=48 all four amplitudes disperse
comparably (final Lnorm 0.37–0.50). The `τ` (Lnorm↓50%) metric shows a spread (π-seed τ/τ_lin=3.1)
but it is **confounded** — the linear seed starts more peaked (L₀=0.48 vs 0.38) and is non-monotonic
(disperses then partially re-localizes), so its Lnorm crosses 50% early; by *raw* final L the linear
seed (0.32) ends MORE localized than the π-seed (0.26). **No robust self-trapping** — which confirms
Close's own prediction ("I'd expect dispersion in most cases").

**Energy conservation (bankable).** At N=48 the control total-H drift is **0.01%** over 2000 steps,
and the seed-excess ΔH drift ≤8% — so the production Eq.18 leapfrog (the M5.6 machinery) is solid +
CFL-stable at the finer grid, and the localization is genuine field dynamics, not numerical pumping.
(The +30–38% intensity drift seen at N=32 was itself a coarse-grid artifact — gone at N=48.)

**Where M5.7 goes next.** The single-shell l=1 displacement dispersed; widen the net (M5.7.2):
harden the metric (rank by final-Lnorm/AUC, not τ), try alternative seed geometries (standing-wave/
breather eigenmode, velocity-kick), and weigh the **reframe** — M5.6.2b already showed the biaxial
hedgehog *sources its own twist* (the defect intrinsically oscillates), so the metastable "particle"
may be the defect's **own intrinsic clock** (M5.8), not a *seeded* resonance. The sharpened question
goes to Close (Q11): seeded standing wave vs intrinsic oscillation.

### §5i — M5.7.2: the defect's intrinsic oscillation (a second null → the particle is 4D)

The M5.7.1 seeded perturbation dispersed; M5.6.2b showed the biaxial hedgehog **sources its own
twist** (`C_μν≠0` drives a ψ-independent force — released from rest, the defect oscillates by itself).
So the reframe (Rodrigo 2026-05-28): drop external seeding and ask whether the defect's **own**
oscillation is the long-lived particle. `m5_7_2_intrinsic_oscillation.py` (`sandbox_v7`) evolves the
V-on biaxial hedgehog from rest and measures two things: (1) **localization** — does the *dynamical*
energy `H_dyn = ½‖Ṁ‖² + c²·curv` (the motion, V-well excluded) stay gathered at the core, vs the
uniform floor `E_floor = (core voxels)/(active voxels)`? (2) **coherence** — Hann-windowed FFT of a
near-core director probe `n̂(t)`: a sharp dominant frequency = a clock, broadband = incoherent.

**Result — the intrinsic 3D oscillation ALSO delocalizes (a second null):**

| Metric | N=32 | N=48 | Reading |
| --- | --- | --- | --- |
| localization-excess `(E_end−floor)/(E₀−floor)` | 49% (plateau ~0.25) | **17%** (0.43→0.13, still declining) | the N=32 plateau was a **coarse-grid artifact**; at resolution the energy disperses toward the floor (0.069) — same washout as M5.7.1 |
| dominant osc frequency | 0.25/t | **0.10/t** | **shifted 2.5× with resolution ⇒ not a converged physical clock** (the spectrum *is* concentrated — band±2 89%, peak-bin 57% — but it's a *dispersing* ringing, not a stable oscillator) |
| V-on total-H drift | 0.02% | **0.01%** | leapfrog rock-solid ⇒ the dispersal is real physics, not numerics |

(The repaired coherence metric matters: the first pass reported a "prominence" of `8.6e13×` — a broken
`peak/median` where the median is ~0 float-noise. Replaced by band-power-fraction + `peak/mean`.)

**Combined M5.7.1 + M5.7.2 conclusion.** Both **seeded** and **intrinsic** orientation energy disperse
in pure 3D. Root cause is clean and already on record: **`V` confines the amplitude `Tr(M²)` but is
rotation-invariant, so it does NOT confine the director orientation** (M5.6.5c / §5f). The defect's
energy lives in orientation/twist dynamics → it radiates freely. **⇒ the metastable coherent
particle/clock is NOT a 3D phenomenon.** This is exactly what the framework predicts: Derrick forbids
3D static/oscillatory localization, and the escape is **time-periodicity, which needs the time
dimension** — Duda's stable particle IS the 4D Zitterbewegung clock auto-propelled by the Lorentz
negative-energy structure (Fig.10, §… / M5.8). M5.7 has thus **empirically established that 3D alone
disperses → M5.8 (4D) is necessary, not optional.** Two nuances: (a) the **topological defect itself
is permanent** (winding conserved — only the *excess oscillation energy* disperses); (b) a **driven**
oscillation (continuous EM forcing, the driven-excess case) is a *separate* question this free-dispersal null
does not answer — that is the M5.7.3 preview next step. Caveat on scope: these nulls are for
**Duda's Eq.18** matrix dynamics; **Close's Eq.23** (spin-density, `∇·s=0`) is a different equation
we have not run — an optional cross-check (the 4D explanation already covers our results).

### §5j — M5.7.3: the driven defect (the other half — a bounded `(A, ω)` excess; the driven-excess substrate)

The §5h/§5i nulls are for a *free* defect. M5.7.3 adds a **continuous EM-wave-like drive** to the same
Eq.18 leapfrog — a fixed localized director-rotation forcing `F_drive = A_drive·sin(2π f_d t)·w(x)·[G_y,M_bg]`
(the so(3) rotation tangent about ŷ — **symmetric + traceless**, so it keeps `M` symmetric and acts in the
orientation/EM sector that V leaves flat, shell-localized to the defect; an incident-EM-tilt-wave proxy, the
same channel §5d maps to Maxwell), added to the acceleration `M̈ = c²·div(G) − dV_M + F_drive`. **Result:
a driven defect SUSTAINS a bounded, frequency-selective `(A, ω)` excess** — at the resonant `f_d≈0.10/t`
(the §5i intrinsic mode) it holds the shell excitation at ~3× the free baseline in a steady-state plateau
(H-growth +1%, bounded), resolution-confirmed N=32→N=48 (the gain *grew* 2.7→3.0×, unlike the free nulls
that washed out). ⇒ the free particle/clock is 4D (M5.8) but the **driven-excess** state is a real 3D driven
response — the field-theoretic basis for a maintained driven excess. **This §5j is the Eq.18-evolution
capstone of the M5.7 arc** — §5h seeded null, §5i intrinsic null, §5j driven sustains.

## 6. Matrix Hamiltonian (Eq.23) — the M5.4-carry-over `compute_energyH_density`

```text
ℋ = Σ_{0≤μ<ν} ‖F_μν‖²_F  +  V(M)  +  Σ_μ F_{0μ}•∂_μA_0
```

The last term vanishes in vacuum (integration by parts, as in EM). This replaces the M5.4
placeholder `compute_energyH_density` (which reads the dormant ψ buffer → uniform `¼λ`) and
gets the deferred physical-energy-scaling factor. Lands with the matrix leapfrog in the
launcher's `compute_propagation` no-op.

---

## 7. Duda's Mathematica source (the reference derivation)

The authoritative symbolic derivation lives in Duda's GitHub
`github.com/JarekDuda/liquid-crystals-particle-models`, file **`liquid crystal particles -
3D equations and hedgehog.nb`** (Wolfram, 705 KB / 12.7k lines — open in Mathematica for the
full version). Its operative code is **printed in the paper as Fig.9** (page 10); transcription:

### 7a. 3D evolution derivation (Fig.9 left)

```mathematica
d  = DiagonalMatrix[{1, δ, 0}];              (* ellipsoid shape Λ=(1,δ,0); δ→ℏ *)
Gx = {{0,0,0},{0,0,-1},{0,1,0}};             (* twist generator  *)
Gy = {{0,0,1},{0,0,0},{-1,0,0}};             (* tilt1 generator  *)
Gz = {{0,-1,0},{1,0,0},{0,0,0}};             (* tilt2 generator  *)
Ga = {{1,0,0},{0,0,0},{0,0,0}};              (* 3 elongation generators *)
Gb = {{0,0,0},{0,1,0},{0,0,0}};
Gc = {{0,0,0},{0,0,0},{0,0,1}};
Gpt = Join[Table[G.d + d.Transpose[G], {G, {Gx,Gy,Gz}}], {Ga,Gb,Gc}];  (* G' = GD+DG^T *)
com[A_, B_] := A.B - B.A;                    (* commutator *)
cd   = {{3,2},{1,3},{2,1}};
vect[m_] := Table[m[[ cd[[i,1]], cd[[i,2]] ]], {i,3}];   (* antisym matrix → rotation vector *)
(* Γ_μ affine connection (matrix form), M_μ = com[Γ_μ, …], F_μν = Simplify[com[M_μ, M_ν]] *)
(* vrip = Tr[ F_μν·( com[Γ_μ, M_ν.Gp] − com[Γ_ν, M_μ.Gp] ) + F_{μν,μ}·M_ν.Gp − F_{μν,ν}·M_μ.Gp ]  *)
(*        — integrate-by-parts form of Eq.35, looped over μ,ν, generators Gp *)
vr   = Simplify[Series[vrip, {δ, 0, 0}]];    (* low-order in δ *)
fin  = Table[Sum[…,{μ},{ν}] − Sum[…], {i,3}, {v, vr}];   (* + Lagrangian → evolution terms *)
(* rename R-curvatures → B/E fields → Column shows: ~Klein-Gordon, ~Maxwell1, ~Maxwell2 *)
```

### 7b. Hedgehog application (Fig.9 right)

> **Transcription caveat (M5.5.1, 2026-05-26):** the exact Euler-angle assignment below
> was read off the small Fig.9 image and is **ambiguous** — using `θ→ArcTan[√(x²+y²),z]`
> on `Gz` and `φ→ArcTan[x,y]` on `Gy` as written gave a NON-radial director in the sympy
> mirror (`m5_5_1_evolution_symbolic.py`, `mean|n̂·r̂|=0.40`). The hedgehog is therefore
> built **physics-first** in the sandbox — `O = [r̂ | e_Θ | e_Φ]·Rx(ψ)` with the radial
> director `r̂` as the first column by construction (verified `|n̂·r̂|=1.000`, `det O=+1`,
> connection `Γ_i=O^T∂_iO` antisymmetric + `~1/r`). The precise Euler parameterization is a
> convention detail; any `O` with radial director + ψ-twist about it is the hedgehog. Pull
> the exact angles from the actual `.nb` only if a line-by-line Mathematica match is needed.

```mathematica
sph = {x -> r Cos[θ] Cos[φ], y -> r Cos[θ] Sin[φ], z -> r Sin[θ]};
Q0  = FullSimplify[ MatrixExp[θ Gz].MatrixExp[φ Gy].MatrixExp[ψ Gx] ]
        /. {φ -> ArcTan[x, y], θ -> ArcTan[Sqrt[x^2+y^2], z]};   (* hedgehog *)
Q   = Q0 /. ψ -> ψ[t, x, y, z];                                  (* assume phase dependence *)
BE  = Simplify[Table[ vect[Transpose[Q].D[Q, v]], {v, {t,x,y,z}} ], r > 0];  (* B,E fields *)
(* substitute BE + derivatives into fin[[1;;3]]; first eq → Klein-Gordon-like, 2nd/3rd → 0 *)
A   = {x, y, z}/r^2;                                              (* Â^hedg *)
gmA[f_] := Grad[f, {x,y,z}] - A f;
Adg[f_] := (A + r) . Grad[f, {x,y,z}];
(* verify:  fne[[1]]/r^2  ==  Σ_i gmA[gmA[ψ]][i] + Adg[Adg[ψ]] + 2 ∂_tt ψ   *)
```

---

## 8. sandbox_v5 plan (mirror the Mathematica in sympy/numpy, then Taichi)

| Sub-step | Deliverable | Validates against |
| --- | --- | --- |
| 5.5.0 ✅ | this doc — action + Eq.35 + Fig.9 source transcribed; math confirmed | — |
| 5.5.1 ✅ | sympy (`m5_5_1_evolution_symbolic.py`): operator identity `F_μν=2[M_μ,M_ν]` (Eq.20) ✓; radial hedgehog (director=r̂, det O=+1) ✓; connection `Γ_i=O^T∂_iO` antisymmetric + `~1/r` (the singular `Â^hedg`) ✓ | the symbolic FOUNDATION is verified. Full action→KG EL reduction is NOT done symbolically (a `(∂M)⁴` action) — **moved to numerical validation** (decision 2026-05-26): dispersion `ω(k)` in M5.5.2/M5.6. Also caught + corrected a Fig.9 angle-transcription error (see §7b caveat). |
| 5.5.2 ✅ | numpy (`m5_5_2_twist_evolution.py`): **first numerical evolution of the Eq.18 action** (twist sector, V=0) on a smooth non-singular background. `K(x)=4Σ‖[M_μ^bg,M_ψ]‖²>0` (twist dynamical where the bg varies) ✓; **energy drift 0.21%** over 1500 leapfrog steps ✓; stable. Derived from `ℒ=T−U` via `F_μ0=2[M_μ,Ṁ]`. | **MACHINERY + energy conservation validated.** Static Coulomb already validated (M5.4 page-18 = `¼Σ‖F_μν‖²`). **KG mass gap MOVED to M5.6** — it needs the biaxial HEDGEHOG, whose `δ≠0` perpendicular frame carries a z-axis **DISCLINATION singularity** (`e_Φ~1/ρ`, hairy-ball — unavoidable for biaxial) + the core; disclination handling is M5.6-level. Minor: amplitude grows at the `K→0` active-region edge (metric-degeneracy artifact) → damp in M5.6. |
| 5.5.3 ✅ | `V(M)` Eq.12/13 + regularization (`m5_5_3_potential_regularization.py`). **Finding: V(M) is rotation-invariant** (`V(ODOᵀ)=V(D)`, eigenvalues fixed) → it does NOTHING to the twist/rotation sector (why M5.5.2 needed V=0); V acts ONLY on the eigenvalue-deformation sector = **the regularization**. V defines vacuum Λ (perturbed λ relax to Λ) ✓. **Eq.18 `F²`+V are Derrick-stable**: rigid core `∫‖F‖²` diverges, deformed core finite; `E(L)=E_F/L+E_V·L³` → finite minimum (core size + mass), no extra Skyrme term (F² IS the Skyrme-family kinetic). | exact `Λ=(1,δ,0)` LdG coeffs (Eq.13 a,b,c) + Faber's exact scheme remain **Q7/Q8** (Duda open); baseline shown. |
| 5.5.4 ✅ | **Taichi port** done. `engine2_pde`: `compute_curvature_flux` (G_α=8Σ[[M_α,M_ν],M_ν]) + `evolve_M` (leapfrog) + `V_M`/`dV_M`; `medium`: `curv_flux_*` + `swap_matrix_buffers`; `engine3`: `compute_energyH_density_M`; launcher: `compute_propagation` → matrix leapfrog + `eigen_decompose` (Evolve PDE LIVE), `compute_energyH_density_M` (WAVE_MENU=4 = matrix ℋ). **Energy-conserving** (`m5_5_4_matrix_evolution_check.py`: secular drift 2.15%→1.13%→0.03% as dt→0 = symplectic; the flat 6% osc is the velocity-measurement convention). Integrated path verified (M+director evolve). **Resolves both M5.4 carry-overs.** | e_scale=1 (bare); physical-energy calibration tied to a reference mass → M5.9. Production-scale dynamics speed via SIM_SPEED/c_amrs. **On-screen VERIFIED (Rodrigo 2026-05-26) — "bounded-not-bug"**: a hedgehog under "Evolve PDE" *sloshes* (not the old explode-and-propagate wave); a headless mirror of the GUI scenario (63³, dx≈15.2, V off, 1200 steps) holds **H conserved to 5 digits** with `max‖M−D‖_F` bounded `0.7→2.2→1.6`, finite — correct nonlinear curvature dynamics, NOT a blow-up. The wave-propagation look belonged to the retired *linear* ψ leapfrog. |
| 5.5.5 → **M5.6** | EM-from-tilts cross-check + Faber EM Lagrangian (4a §11b) — **folded into M5.6** (overlaps its Maxwell-sector verification, Eq.37–38) | superfluid-vorticity ↔ Maxwell dictionary |

**Exit criterion (M5.5):** the full Eq.18 action runs; defect dynamics are governed by the
real proposed action (not the M5.1 Frank-energy approximation); KG emerges from the twist mode.

---

## 9. M5.5.2 evolution — EOM derivation + the kinetic-degeneracy finding

**Kinetic term (derived 2026-05-26, confirmed by Rodrigo).** The "electric"/time curvature
parallels Eq.20:

```text
A_0 = [M, Ṁ]                                       (Ṁ = ∂_0 M)
F_μ0 = ∂_μA_0 − ∂_0A_μ = 2[M_μ, Ṁ]                 (the [M,·] terms cancel: ∂_μṀ = ∂_0∂_μM)
```

So the action splits into a proper kinetic + potential:

```text
ℒ = T − U
T = Σ_{μ=1..3} ‖F_μ0‖²_F = 4 Σ_μ ‖[M_μ, Ṁ]‖²_F            kinetic (quadratic in Ṁ)
U = Σ_{μ<ν} ‖F_μν‖²_F + V(M) = 4 Σ_{μ<ν} ‖[M_μ,M_ν]‖²_F + V(M)
EOM:  ∂_0(∂T/∂Ṁ) = ∂T/∂M − ∂U/∂M
```

`M`, `M_μ`, `Ṁ` are all real-symmetric → `[M_μ,Ṁ]` is antisymmetric; `T ≥ 0`.

**THE FINDING — the kinetic metric is DEGENERATE (gauge structure):** `T = 4Σ_μ‖[M_μ,Ṁ]‖²`
vanishes whenever `Ṁ` commutes with every spatial gradient `M_μ`. Consequences that shape
M5.5.2:

| Implication | Detail |
| --- | --- |
| Evolve the rotation DoF, not M's 6 components | The dynamical variable is `O(x) ∈ SO(3)` (3 angles/voxel; `D` frozen). `M`'s 6 free components include gauge/non-dynamical directions — a naive leapfrog on `M` hits the degenerate (non-invertible) metric. Parameterize by `O` (or angular velocity `ω = OᵀȮ`). |
| The twist is dynamical ONLY on a non-uniform director background | A pure single-axis twist `M=Rx(ψ)D Rx(ψ)ᵀ` has `M_x ∝ M'` and `Ṁ ∝ M'` → `[M_x,Ṁ]=0` → **T=0**. So a uniform-axis 1D twist is in the kinetic null space (NOT dynamical). The KG-for-twist emerges only when the twist couples to a position-dependent director — i.e. the **hedgehog**. |
| Minimal KG test is inherently 3D hedgehog + twist | This is exactly the Fig.9 case (twist phase `ψ(t,x,y,z)` on the hedgehog → KG with `Â^hedg`). So M5.5.2's KG-dispersion validation converges with M5.6's headline. |

**Refined M5.5.2 plan:** evolve `O(x) ∈ SO(3)` (gauge-clean DoF) on a small 3D grid, with the
hedgehog as the background + a small twist-phase perturbation; the field-dependent kinetic
metric `T = 4Σ‖[M_μ,Ṁ]‖²` is non-degenerate on the rotation DoF. Validate Eq.23 energy
conservation + KG dispersion `ω(k)` of the twist. (Static Coulomb already validated via the
M5.4 page-18 energy = `¼Σ‖F_μν‖²`.) This is a substantial numerical build — the M5.5↔M5.6 core.

---

## 10. M5.8 foundation — the 1+1D time-crystal toy model + the 4D promotion

**Source:** Duda, *Time crystal φ⁴ kinks by curvature coupling as toy model for mechanism of
oscillations propelled by mass, like observed for electron and neutrinos* (arXiv:2501.04036 v2,
24 Jul 2025; local PDF `../theory/time_crystal_toy_model.pdf`, 3pp). Math **read + verified by quadrature
2026-05-29** (the variational anchors below reproduce to 4–5 digits). This is the mathematical
foundation for the M5.8 build plan in [`0b_M5_roadmap.md § Phase M5.8`](0b_M5_roadmap.md).

**The distilled build-spec is §10e (canonical recipe)** — substrate + action + construction
recipe + integrator constraints + the M5.8.2a G1–G7 gate table. §10a–§10d keep the derivations
and the discovery narrative.

### 10a. The 1+1D toy model — the integrator-validation anchor (M5.8.0)

A clean 1+1D scalar realization of "why a resting particle oscillates at `ω = 2mc²/ℏ`" (the de
Broglie clock / Zitterbewegung). Two real fields: `φ` (the φ⁴ kink, topological) and `ψ` (the
quantum-phase *clock*, `exp(i2πψ)` winded mod 1), coupled **only** through a Lorentz-invariant
curvature `R`. Signature `η = diag(1, −1)`.

```text
ℒ = ∂_μφ ∂^μφ − (1−φ²)² − α R² + (β/3) R⁴                         (Eq.1)
R = ∂₀φ ∂₁ψ − ∂₁φ ∂₀ψ          (= φ₀ψ₁ − φ₁ψ₀ ;  φ₀≡∂_tφ, φ₁≡∂_xφ)

Legendre → Hamiltonian (the βR⁴/3 → βR⁴, since R is degree-1 in velocities):
ℋ = φ₀² + φ₁² + (1−φ²)² − α R² + β R⁴                             (Eq.2)
```

`R` is Lorentz-invariant (it is the 2-form `dφ ∧ dψ` in the `(t,x)` plane — a pseudoscalar; the
paper verifies invariance under a boost `γ`). Note `ψ` carries **no potential and no bare kinetic
term** — it enters the dynamics *exclusively* through `R`.

**Ansatz** (the static-kink clock). `φ ≡ φ(x)` a static kink `−1 → +1` (so `φ₀ = 0`), and
`ψ = ω t` a linear phase (so `ψ₁ = 0`, `ψ₀ = ω`). Then `R = −ω φ_x`, and the energy density
collapses to one line:

```text
ℋ = φ_x² (1 − α ω²) + (1 − φ²)² + β ω⁴ φ_x⁴                       (Eq.3)
```

**The mechanism (why it is a time crystal).** The `−α ω² φ_x²` term is **negative** — turning the
clock on (`ω ≠ 0`) *lowers* the gradient energy. This is the counterintuitive "negative energy
contribution of a time derivative" that *propels* the oscillation; the positive `β ω⁴ φ_x⁴` term
caps it. So the energy-minimizing state has `ω ≠ 0` — a static kink (`ω = 0`) is **not** the
ground state. Minimizing `E = ∫ℋ dx` over `ω`:

```text
ω = √[ α ∫φ_x² dx / (2β ∫φ_x⁴ dx) ]                               (Eq.4)
```

**Two validation anchors** (both at `α = β = 1`):

| Anchor | Profile | ω | E | Notes |
| --- | --- | --- | --- | --- |
| **Analytic** (Eq.5, standard tanh) | `φ = tanh(x/w)`, `w = √(96/61)` | `√(70/61) = 1.0712` | `2.1257` | closed-form via `∫sech⁴ = 4w/3`, `∫sech⁸ = 32w/35`; no fitting |
| **Optimized** (Fig.1, polynomial arg) | `φ = tanh(0.6326x + 0.0198x³ + 0.0203x⁵)` | `1.2898` | `2.0252` | numerically energy-minimized deformation; lower E |

**Our quadrature confirmation (2026-05-29).** With the optimized profile and `α=β=1`: integrals
`∫φ_x² = 1.0126`, `∫φ_x⁴ = 0.3044`, `∫(1−φ²)² = 1.8547` ⇒ `ω* = 1.2897` (paper 1.2898), `E* =
2.0252` (paper 2.0252). The **static** kink (`ω=0`) costs `E = 2.8673` — so the oscillating state
wins by `ΔE = 0.84` ⇒ the time crystal is confirmed: the energy minimum is `ω ≈ 1.29`, not `0`.
This verifies our understanding of the energy functional before any 4D build.

> ⚠️ **Integrator caveat for the dynamical pre-check (M5.8.0b).** The `−αR² + βR⁴` curvature
> coupling makes the conjugate momenta **non-canonical** — `R` mixes `φ₀` and `ψ₀`, so
> `∂ℒ/∂φ₀` depends on `ψ`-gradients and vice versa. A vanilla wave-equation leapfrog does not
> apply; the time-stepper needs the Legendre-inverted acceleration (solve the coupled momentum
> system each step) or a constrained scheme. The Euler–Lagrange equations are left in Mathematica
> in the paper (Fig.3, not transcribed) — derive + confirm them before coding the kernel.

**M5.8.0b-1/0b-2 — the EOM derived + verified (2026-06-04, `sandbox_v8/m5_8_0b_eom_derivation.py`).**
The coupled Euler–Lagrange equations + the Legendre inversion confirm all three checks: the
Hamiltonian reduces to the 0a energy under the static-kink+`ψ=ωt` ansatz ✓, the `ψ`-EL is a pure
conservation law (`∂ℒ/∂ψ=0`, a Noether current) ✓, and `dE/dω=0` recovers `ω*²=αI₂/(2βI₄)` (Eq.4) ✓.
The kernel update is `H·[φ_tt, ψ_tt]ᵀ = b` with the **non-canonical mass matrix** (leading order in the
`−αR²` mechanism):

```text
H ≈ |  2 − 2α ψ_x²     2α φ_x ψ_x |        b = −(spatial + V′ + mixed φ_tx, ψ_tx terms)
    |  2α φ_x ψ_x      −2α φ_x²    |        det H = −4α φ_x²
```

Two structural facts shape the stepper:

- **Off-diagonal `2α φ_x ψ_x ≠ 0`** couples `φ_tt` and `ψ_tt` ⇒ the leapfrog must **invert `H` each
  step** (no vanilla wave update). It is `∝ ψ_x`, so the static-kink+`ψ=ωt` ansatz (`ψ_x=0`)
  **decouples** — why 0a's reduction was clean; the dynamical run is where it bites.
- **`det H ∝ φ_x²`** ⇒ (a) **`ψ` is inertia-less in the vacuum** — it has independent dynamics only
  where `φ_x≠0` (off the core `H` is singular, `ψ` slaved). This is the **1D shadow of "the clock lives
  on the hedgehog where `C_μν≠0`"** (§10b / M5.6.2b) — the kernel masks/regularizes `ψ` off the core.
  (b) **`H[1,1] = −2α φ_x² < 0`** — `ψ`'s inertia is *negative* (the `−αR²` indefinite signature): the
  negative-energy mechanism is right there in the mass matrix (turning the clock on *lowers* energy).
  Well-posed where `det H ≠ 0`.

**M5.8.0b-3/0b-4 — the field leapfrog + the *ghost* finding (2026-06-04, `sandbox_v8/m5_8_0b_toy_leapfrog.py`).**
Building the field stepper from the `H`/`b` above and running it confirmed both structural facts
*numerically* — and surfaced the central one:

- **Energy gate ✓ (the time crystal, dynamically).** With `ε=0` (the pure toy, which auto-localizes the
  `ψ`-energy since `R=0` in the vacuum ⇒ `ℋ_vac=0`), the seeded static-kink+`ψ=ωt` ansatz gives
  `E(ω*)=2.25 < E(0)=2.73` — the oscillating state is the minimum. *(2.25 vs the analytic 2.03 is the
  standard-tanh shape vs the optimized profile, not an error.)* **A regularizing `ε(ψ_t²−ψ_x²)` is the
  WRONG fix** — `ψ=ωt` winds over the whole box, so `ε` adds a spurious box-extensive `ε·ω²·L_box` (we
  saw `E` jump to 14.2). The pure toy localizes `ψ`-energy for free; `ε`-reg breaks it.
- **The `ψ`-sector is a GHOST ⇒ free evolution is ill-posed.** `H = diag(+2, −2αφ_x²)` is **indefinite**
  (one positive, one negative eigenvalue): `ψ` is a **negative-kinetic (ghost) mode** — that *is* the
  negative-energy propulsion made explicit. A naive **explicit** stepper of a ghost is unstable; the
  field leapfrog blows up at `t≈1.2`, exactly where the indefinite-`H` structure predicts. So the
  physical dynamics are **constrained** — they live on the bounded-energy manifold (the `ψ=ωt` ansatz
  the variational 0a uses), not free ghost evolution. This is precisely Duda's "**bounded negative
  energy — mass can't go below zero**" guard (§10c): topology bounds `E` below, so the real motion stays
  on the constraint surface. **Consequence for M5.8.2:** the 4D Minkowski kernel has the *same*
  indefinite `(−,+,+,+)` `Γ₀` signature ⇒ it needs a **constrained / projected** integrator, NOT a
  vanilla explicit leapfrog. Gate (ii) "the clock holds dynamically" is therefore demonstrated with a
  **collective-coordinate** reduction (rigorous, robust), not free field evolution — **done**
  (`m5_8_0b_collective_clock.py`): reducing to `(w,Θ)` makes the ghost `Θ` cyclic ⇒ `p_Θ`
  Noether-conserved ⇒ stable. It reproduces the analytic Eq.5 anchor *exactly* (`ω*=1.0712=√(70/61)`,
  `E*=2.1257`, `E*<E(0)=2.74`); the clock holds at `ω*` with machine-precision energy conservation
  (drift `~10⁻¹⁵`) over `t=60`, robust under a +5% width perturbation.

### 10b. The 3+1D promotion — what M5.8 actually builds (Fig.2 / paper [1] Eq.42)

The toy model's `−αR²` is the 1+1D stand-in for the genuine mechanism: in the full 3+1D
Skyrme + Landau–de Gennes action on the matrix field `M = O D O^T`, the **Minkowski signature**
flips the sign of the time-axis squared-curvature, producing the negative `Γ₀` terms that
auto-propel the clock with no engineered propulsion (paper Fig.2; the same Fig.10 mechanism in
2108.07896).

```text
ℒ = −Σ_{αβμν} F_μναβ F^μναβ − V(M)        (4-index curvature, Eq.42 in arXiv:2108.07896)
F_μναβ = [∂_μ M, ∂_ν M]_αβ
D = diag(g, 1, δ, 0) ,  O ∈ SO(1,3)        (was diag(1,δ,0), O∈SO(3) in 3D M5.4–M5.7)
```

| Item | 3D (M5.4–M5.7) | 4D (M5.8) |
| --- | --- | --- |
| Field algebra | `M = O D O^T`, `D=diag(1,δ,0)`, `O∈SO(3)` | `D=diag(g,1,δ,0)`, `O∈SO(1,3)` (3 rotations + 3 boosts) |
| Metric | Euclidean `+++` (ℋ manifestly ≥0 ⇒ Derrick collapse) | Minkowski `−+++` (indefinite ⇒ negative `Γ₀` curvature terms) |
| Time | external leapfrog parameter | the **0-eigenvalue**, *inside* the algebra — the grid **stays 3D**; time is not a 4th grid axis (4a §6) |
| Storage | 6 independent symmetric comps/voxel | 10 independent symmetric comps/voxel; operators already index-generic (M5.4 design) |

**Faithful-kinetic prerequisite (the load-bearing constraint; §5g + §9).** Measure the clock
frequency with the **faithful** kinetic `T = 4Σ_μ‖[M_μ, Ṁ]‖²` (the `O(x)∈SO(1,3)` metric, the 4D
extension of the validated `m5_6_2b` ψ-evolution `2K ψ_tt = Σ_μ ∂_μ J_μ`), **not** production's
`½‖Ṁ‖²` — the simple kinetic mis-sets physical-mode inertia by `×[0.6, 3.0]`, and the M5.8 exit
is `ω` within 10%. Two facts from §9 shape the build: (1) the metric `K` is **degenerate** (the
trace is its null mode), so evolve the rotation/boost DoF `O`, not M's components; (2) the twist
is dynamical **only on a non-uniform background** (`[M_x, Ṁ] = 0` for a uniform-axis twist) — so
the clock lives on the **hedgehog**, exactly the `m5_6_2b` configuration where `C_μν ≠ 0` already
sources the twist (the defect *cannot* sit at `ψ=0` — the clock seed).

**The `ω = 2mc²/ℏ` calibration sub-question.** The target `ω ≈ 1.55×10²¹ rad/s` is *physical*;
the sim runs in natural units. The clean test is the dimensionless self-consistency ratio
`ω · ℏ / (2 H_rest) → 1` (with `ℏ ↔ δ`, the QM eigenvalue; `H_rest` the measured rest-energy
Hamiltonian), with the absolute Hz set by the Faber `r₀` scale-fix from M5.6.3
(`r₀ = 2.2132 fm → 0.511 MeV`). So M5.8's frequency check overlaps the M5.9 mass-calibration
handle: measure the ratio, let the absolute scale follow `r₀`.

### 10c. Paper re-audit + Duda's list updates (2026-06-01)

Re-read of the toy-model paper (arXiv:2501.04036) against the M5.8 plan, cross-checked with Duda's
2026-06-01 models-of-particles posts. The plan covers the paper's spine faithfully (1+1D Eq.1–5 ✅
M5.8.0; 4-index `F_μναβ` + Minkowski-`Γ₀` ✅ M5.8.2; electron hedgehog+twist+moment ✅ L4/L7/L8;
`ω=2mc²/ℏ` ✅ M5.8.3). One **notable gap** + minor refinements + two list-update clarifications:

- **🔶 GAP — the 2+1D pilot-wave intermediate (paper §III "further work").** The paper lays an
  explicit dimensional ladder **1+1D → 2+1D → 3+1D**; our plan jumps 1D→3D, skipping 2+1D. In 2+1D
  the intrinsic clock oscillation sources **coupled "pilot" waves**, which the paper proposes to use
  to recreate the **hydrodynamic Casimir effect** + **walking-droplet** quantum analogs (interference,
  tunneling, orbit quantization — Couder/Bush) with the external shaker *replaced by the defect's own
  de Broglie clock*. **Reinforced directly by Duda 2026-06-01 email #1** (pilot waves, Casimir,
  hydrodynamic analogs, "replace external shaker with electron de Broglie clock"). Value: a cheaper
  2D stepping stone that would yield a spectacular validation — quantum phenomena emerging from
  classical pilot waves. **Added as roadmap M5.8.6 (breadth/optional, not headline-gating).**
- **Minor refinements (paper §III):** (a) the `ϕ_t ≠ 0` small-perturbation case (we use the static
  `ϕ_t=0`, `ψ=ωt` ansatz) — a M5.8.0 robustness add-on; (b) **neutrino flavor oscillation as 3-mass
  clock-*beating*** (`|ν_j(t)⟩ = e^{−iE_j t/ℏ}|ν_j(0)⟩` — the title's other half) as an explicit
  M5.8.4/M5.9 demo, not just an entry in the mass→ω table; (c) the paper allows **sine-Gordon** `V`
  as an alternative to `(1−φ²)²`.
- **Duda email #1 — `E(ω)` clarifications (the same double-well plot we verified in M5.8.0a):**
  - **Bounded negative energy — mass cannot go below zero.** The negative-`αR²` dip that propels the
    clock is bounded: `E*` stays **positive**. If it went negative there would be *catastrophic
    particle creation* (vacuum runaway). This is an independent physics guardrail on the
    negative-energy mechanism: the floor is lossless but **strictly positive**, never extractable to zero (mass can't go ≤0).
  - **Stabilization (topological) vs frequency-selection (maybe additional).** Duda: the *particle
    stability* is topological (Gauss law counts winding); the *clock-frequency selection* might be a
    **separate, additional** mechanism — he floats **stochastic resonance** (noise + a nonlinear well
    → a preferred resonant frequency). New candidate mechanism for *why* the `E(ω)` minimum locks at
    the de Broglie `ω`; worth probing at M5.8.3 (does the measured `ω` need noise to lock, or does the
    well alone select it?). Matches our split: topology = stability (L4), clock_twist = frequency (L7).
    **(1D answer — M5.8.0d-b, 2026-06-04, `m5_8_0c0d_propulsion_robustness.py`: in the reduced
    collective-coordinate model the well selects `ω*` *deterministically* — five `ω` seeds all relax to
    `ω*=1.0712` with NO noise. So noise is *not* required for frequency-selection in 1D; the full-field
    mode-selection version of the SR question stays open for M5.8.3. Bounded negative energy also
    confirmed: `E>0` everywhere, `E_min=E*=2.1257>0`.)**
  - **Continuous, not bistable.** Local fields perturb the `E(ω)` minimum **slightly and
    continuously** — *not* into "an electron with 2 discrete mass states." Confirms our single-
    continuously-perturbed-clock framing (we never posited a discrete two-mass electron).
- **Duda email #3 — external convergence (Sabine / quadratic gravity).** Mainstream work that adding
  **`R²` (squared-curvature) terms** to the gravitational Lagrangian smooths Big-Bang singularities
  arrives at the *same mathematical structure* our action is built on (Skyrme `F²`, Faber `Λ`, the
  toy-model `−αR²+βR⁴`). Encouraging convergence: gravity theory reaches curvature-squared from the
  *gravity* side; we have it from the *particle/LdG* side (the 4D `g`-axis, `4b §4.7`). Not an action
  item — a confidence signal that squared-curvature Lagrangians are a serious, fertile structure.

### 10d. Complete-model 4D Hamiltonian — explicit form (Wolfram-article full text, 2026-06-04)

The local PDF (`../theory/Time crystal ϕ⁴ kink…Wolfram Community.pdf`, 14pp — the Wolfram-Community
writeup of arXiv:2501.04036, §"Complete model candidate") spells out the **explicit** 4D Hamiltonian
that §10b states as Eq.42. It is the load-bearing M5.8 build target, transcribed here so the
implementation is self-contained (the roadmap M5.8 phase keeps only a summary + a pointer to this).

**The Hamiltonian split (the `(−,+,+,+)` signature does the work).** From `𝓛 = −Σ_{αβμν} F_μναβ F^μναβ
− V(M)`, `F_μναβ = [∂_μ M, ∂_ν M]_αβ` (both `μν` and `αβ` run `0–3`, time included — the `αβ` are the
matrix indices the M5.5 `F_μν=[M_μ,M_ν]` left implicit), the Legendre transform gives:

```text
ℋ = 2 Σ_{0≤μ<ν≤3} [ Σ_{1≤α<β≤3} (F_μναβ)²  −  Σ_{α=1}^{3} (F_μνα0)² ]  +  V(M)
      └──────── positive: spatial–spatial ────┘  └─ negative: temporal (α,0) ─┘
```

- the **positive** spatial–spatial block = the separate EM / QM / GEM curvature energies;
- the **negative** temporal `(α,0)` block (the time-axis components, sign-flipped by the Minkowski
  metric) = the **clock-propulsion fuel** — the genuine 3D/4D analog of the toy model's `−αR²` (§10a).

**Drop `βR⁴` in 3D/4D (confirmed by the article).** The toy added `+βR⁴` *only* to stop infinite `ω`
in 1+1D; the article states that in 3D this is "prevented by the remaining positive squared-curvature
contributions." → **M5.8 must NOT port `βR⁴`** — the 3D spatial curvature regularizes `ω` on its own.

**Generator → force map.** With `D = diag(g, 1, δ, 0)`, `g ≫ 1 ≫ δ_{~ℏ} > 0`, and the `SO(1,3)`
connection `Γ_μ = Oᵀ∂_μO`:

| Generator | Is | Gives |
| --- | --- | --- |
| `Γ¹` (boost into the `g` time axis) | local **boost** | gravitational mass + **GEM** (`gΓ¹Γ²`, `gΓ¹Γ³`) |
| `Γ²`, `Γ³` (tilt of `n̂`) | EM high-energy curvature | **electromagnetism** (`Γ²Γ³`) |
| `Γ¹`-twist (the `δ` low-energy twist) | QM phase, `U(1)→SO(3)` | **QM / clock** (`δΓ¹Γ²`, `δΓ¹Γ³`) |

**Gravity = gravitoelectromagnetism (GEM).** Boost dynamics on the `g`-axis give a *second set of
Maxwell equations* — confirmed by **Gravity Probe B** (frame-dragging):

```text
∇·E_g = −4πG ρ_g
∇·B_g = 0
∇×E_g = −∂_t B_g
∇×B_g = −(4πG/c²) J_g + (1/c²) ∂_t E_g
```

This is the math behind the `4b §4.7` gravity-viz spec and `0c` L3's "gravity = a bend in time." The
clock and gravity **co-arise from one term**: `−(δΓ¹₀ Γ̄¹_μ − δΓ¹_μ Γ̄¹₀)²` energetically prefers BOTH
twist evolution `Γ¹₀` (the clock) AND gravitational mass `Γ̄¹` (the boost).

**Reference electron-field generator (cross-check for `seed_hedgehog` + `clock_twist`).** The article's
Mathematica builds the spinning-hedgehog `M(x)` from the standard `SO(3)` generators `Gx, Gy, Gz`:

```text
Q  = exp(φ·Gz) · exp(θ·Gy) · exp(ψ·Gx)        hedgehog: φ = atan2(y, x), θ = −atan2(√(x²+y²), z)
M  = Q · diag(0.1, 0.01, 0.001) · Qᵀ           ψ = the CLOCK phase, swept 0 → 5π/6 (the animation)
```

Our 4D seeder should reproduce this `M(x)`; the `ψ`-sweep is the clock (the `0c` L7 / `clock.gif`
collective mode). `D = diag(0.1, 0.01, 0.001)` is the article's `(1, δ, 0)`-style hierarchy in demo units.

**Anchors.** (a) the `(E, ω)` ladder `{2.12568, 1.07123}` / `{2.03638, 1.24938}` / `{2.02515, 1.28975}`
(polynomial degree 1/3/5) **confirms §10a / M5.8.0a verbatim**; (b) the de Broglie clock is
**experimentally observed** — Catillon, Cue, Gaillard, Genre, Gouanère, Kirsch, Poizat, Remillieux,
Roussel, Spighel, *"A Search for the de Broglie Particle Internal Clock by Means of Electron
Channeling"*, **Found. Phys. 38 (2008) 659–664** (81 MeV e⁻, ⟨110⟩-Si resonance) — the empirical
target the M5.8 `ω = 2mc²/ℏ` claim points at; (c) **neutrino = a short closed loop** (Abrikosov-vortex-
like) of ellipsoids, excited by `π` (lepton) vs `π/3` (quark `e/3`); muon↔tau (axes 2↔3 along the loop)
is lowest-energy → the dominant oscillation (matches the data) → M5.9.

### 10e. Canonical recipe — the build-spec + the M5.8.2a verification gates (2026-06-05)

The single self-contained statement of **what to build** — substrate, action, conventions,
construction recipe, integrator constraints, gates — distilled from §10a–§10d. A new contributor
(or a future session) reads this subsection to rebuild the model from scratch; the evidentiary
catalog (every headline claim + its runnable script) is the
[`0b_question_tracker.md`](0b_question_tracker.md) § *Empirical validation* table.

#### The substrate

One matrix field per voxel; the grid stays 3D — **time is not a 4th grid axis, it is the 4th
matrix axis** (the 3D→4D promotion table is §10b).

```text
M(x) = O(x) · D · O(x)^T ,   O(x) ∈ SO(1,3) ,   M symmetric 4×4 ,  time = MATRIX INDEX 3
```

Eigenvalue dictionary (`D` is the local ground-state spectrum; `g ≫ 1 ≫ δ > 0`):

| Eigenvalue | Axis meaning | Production value |
| --- | --- | --- |
| `g` | time / boost / gravity axis (index 3) | `LC_G = 8.0` |
| `1` | EM high-energy axis | 1.0 |
| `δ` | QM low-energy twist axis (`δ ~ ℏ`) | `lc_delta = 0.3` |
| `0` | the soft axis | 0.0 |

| Configuration | D | Frame |
| --- | --- | --- |
| Vacuum (uniaxial) | `diag(δ, δ, 1, g)` | director = ẑ everywhere |
| Defect (biaxial hedgehog) | `diag(1, δ, 0, g)` | `O_hh = [r̂ \| e_Θ \| e_Φ] ⊕ 1` (m5_6_2a), point core `r_c` + z-axis disclination `ρ_c` masks |

#### The action + Hamiltonian

```text
ℒ = − Σ_{αβμν} F_μναβ F^μναβ − V(M)          (Duda Eq.42; §10b)
F_μναβ = [∂_μ M, ∂_ν M]_αβ                    (4-index Skyrme curvature:
                                               μν = derivative, αβ = matrix indices)

Legendre ⇒  ℋ = 2 Σ_{0≤μ<ν≤3} [ Σ_{spatial α<β} (F_μναβ)²  −  Σ_{α=0..2} (F_μνα3)² ] + V(M)
                 └── positive: spatial matrix block ──┘   └── NEGATIVE: (α,3) time block ──┘
```

The sign rule (with time = matrix index 3): the sign of every `F²` term is decided by the
**matrix index pair** — spatial pairs positive, `(α,3)` pairs negative — uniformly over all `μν`.
The negative `(α,3)` block is the **clock-propulsion fuel** (the 3+1D genuine analog of the 1D
toy's `−αR²`).

| Piece | Canonical form | Status |
| --- | --- | --- |
| Kinetic (faithful) | `T = 4Σ_μ‖[M_μ, Ṁ]‖²` — the `O(x)∈SO(1,3)` metric | validated on ψ-DoF (`m5_6_2b`); production `½‖Ṁ‖²` is the qualitative visualizer only (inertia off ×[0.6,3.0], §5g) |
| Potential | `V(M)` = Eq.13 LdG (`a·Tr(M²) − b·Tr(M³) + c·Tr(M²)²`), **spatial 3×3 block only** until the time axis couples deliberately | live in production, off by default; `b=0` amplitude well confines ~3.3× |
| Regularization | Faber `Λ = q₀⁶/r₀⁴` → core melts to isotropic, mass pinned `E ∝ 1/r₀` (`r₀ = 2.2132 fm → 0.511 MeV e⁻`) | the M5.9 calibration handle |
| `βR⁴` | **DO NOT PORT** — 1+1D-only regulator; in 3D the positive spatial curvature supplies the cap (via profile response — M5.8.2b) | confirmed by the Wolfram article + the 2a parity trend |

#### The canonical construction recipe (the clock state)

The M5.8.2a/2b scaffold ansatz family — each factor is one physical ingredient:

```text
O(x,t) = O_hh(x) · B(x; b) · R(ωt)            M = O D O^T,  D = diag(1, δ, 0, g)

O_hh   the biaxial hedgehog frame [r̂|e_Θ|e_Φ], embedded 4×4 (time row/col = 1)
B      = exp(b · w(r) · B_a),  B_a = E_{a3} + E_{3a}   — BOOST DRESSING: mixes
         spatial eigen-axis a with the time axis; w(r) core-localized profile
R(ψ)   = exp(ψ · G_pq) — the GLOBAL clock rotation in eigen-plane (p,q); plane
         (δ, 0) (the article's exp(ψ·Gx)). ⚠️ As THE clock mode this global
         rotation is RULED OUT by 2b (ghost saddle-only) — kept as the scan
         scaffold; the physical clock is the localized twist (table below)
```

| Knob | Meaning | Key facts (2a/2b-measured) |
| --- | --- | --- |
| `b = 0` | time axis inert — **the M5.8.1 production state** | fuel block exactly 0; clock costs `+ω²` (the M5.7 null in functional form) |
| `b > 0` | time axis dressed (the boundary M5.8.2 deliberately crosses) | fuel `C(b) < 0` for every plane×axis combo; static `A(b)` dips `5.97→0.39` at `b≈0.2` (the ω=0 GEM effect) then rises, always `> 0` |
| `ω` | clock rate | `E(ω,b) = A(b) + ω²·C(b)` exactly (rigid sweep); `ω_M = 2·ω_clock` (apolar doubling, machine-exact) — the `ω = 2mc²/ℏ` factor 2 |

**The winning recipe (post-2b-1/2b-2, 2026-06-05)** — what each layer of the actual build is:

| Layer | Winning form | Status |
| --- | --- | --- |
| Static ground state | the **boost-dressed** biaxial hedgehog at the GEM dip — `O_hh·B(b*·w(r))`, `b* ≈ 0.13–0.2`, **wide** dressing (`r_w ≈ 3–3.5` in the 48³/L=6 sandbox units): a STABLE center **below** the bare defect (`E* = 2.61 < A(0) = 6.14`), held dynamically (`H` drift 4.9×10⁻⁸) | ✅ 2b-1 |
| Clock mode | **NOT** the global `R(ωt)` (ghost saddle-only — the net global inertia vanishes at the window edges); the **CORE-LOCALIZED twist field** `ψ(x,t)` about the `(δ,0)` plane on the dressed background — m5_6_2b's massive ψ-mode with the Minkowski `(α,3)` signs. **MEASURED (2b-2): a single coherent mode `ω = 5.86` sandbox units (fft/zc agree 0.4%) — the signature coheres the clock** (the Euclidean twin on the identical background is multi-mode, 89% fft/zc split); best core retention (0.66 Mink > 0.59 undressed > 0.46 Euclid, time-matched) | ✅ 2b-2 |
| Ghost handling | per-voxel **stable mask `K(x)>0 ∧ Q(∇ψ) PD`** (77% of the region at `b*`; precomputable from the background): evolve the clock there; the Q-not-PD **fuel shell** runs away at `λ≈15.6/t` in any linearized frozen-background kernel — the LINEAR propulsion signature whose saturation needs compact-orbit nonlinearity + backreaction | ✅ mapped (2b-2) |
| Integrator | conserved-`p_Θ` CC (sandbox ✅) → stable-region linearized field (sandbox ✅) → **the full nonlinear signed system: ONLY the spectral-projection constrained kernel is stable** (2c-1, f64: per-voxel positive-inertia eigenprojection of `A(Ṁ)=4Σ[η[Ṁ,M_i]η,M_i]` + `P`-projection every step + global-(α,3) clamp + bounded-energy guard). **VERDICT (2c-2): any cheap positive inertia (simple `½‖Ṁ‖²`, diagonal faithful-lite) under the signed flux has slow growing modes — heuristics (mask, smoothing, V-pinning) only stretch the blow-up timescale 120→250→700 steps.** **TAICHI KERNEL VALIDATED (Option B spike, `m5_8_2cb_taichi_constrained.py`): one fused kernel (build 10×10 A → own local-matrix Jacobi, NOT `ti.sym_eig` → keep + P-project + `Ṁ=A⁺P`) — machine-exact vs the f64 reference (5.5×10⁻¹⁵ field diff @890 steps), f32 benign (10⁻⁵ @890, zero keep flips), Metal clean, and FAST: the solve costs 8.3 ms/step at 64³ (1.3× rest-of-step, NOT the feared 5–10×; 69 steps/s total). PRODUCTION-WIRED (B-2): `INTEGRATOR_4D: "constrained"` — (M, P) state in `medium.py`, the 6-kernel step in `engine2_pde.py` (the 2c-1 4× convention end-to-end, τ-units `dt_eff = c·dt_rs`, V at `K·0.5/dx⁴`), xperiment `_topo_dressed4d_signed` (`DT_SCALE_4D: 0.007` = the validated `c·dt/dx`); headless §6e 6/6 + the 4-variant 6000-step 63³ Metal repro all bounded (V-on/off × clamp on/off). **GUI-explosion postmortem (B-3): a LAUNCHER dt-wipe, not the kernel — the main loop re-runs `compute_timestep()` every frame, erasing one-shot dt derates; `DT_SCALE_4D` now lives inside `compute_timestep`, the kick is the 2c-1 velocity convention `Ṁ₀ = kick·M_ψ` (dt-independent), and guard (b) auto-pauses on `max\|M\| ≥ 50`** **LONG-HORIZON FINDING (2cb-2): the constrained evolution itself opens a negative-H ghost runaway at ~1150 steps ≈ 2 clock periods — in f32 AND f64 identically (precision exonerated); V/clamp exonerated (V=0 spike fails the same); the (α,3) fuel sector pumps monotonically. **dt-halving VERDICT: onset at FIXED τ (ratio exactly 2.0) ⇒ TRUE DYNAMICS — the QUADRATIC action has NO clock-saturation mechanism at 3+1D field level; the engine is exonerated at machine precision. RESOLVED BY M5.8.2d (the row below)** | ✅ 2c-1/B-1/B-2 (quadratic: ~2 periods) · ❌ quadratic saturation REFUTED · ✅ quartic saturation (2d) |
| **Saturating quartic (NEW LAYER, M5.8.2d)** | `V_u(x) = u + β·u²` on the SIGNED spatial-curvature density `u = 2Σ_(i<j)⟨F_ij,F_ij⟩_s` — the 1D `−αR²+βR⁴` floor transplanted to the channel that actually runs (the (α,3) gradient sector; per-voxel floor `−1/(4β)`). Exactly variational + integrator-invariant: the dU flux integrand × local `(1+2β·u(x))`. β data-anchored: `2β·\|u_min(seed)\| ≈ 3` (β = 1.558 at the 2c-1 24³ seed). **MEASURED: kills the ghost runaway (control −8.6×10⁹ → bounded), H breathes 30–142 over τ = 48 ≈ 45 clock periods (20× the quadratic lifetime), dt-converged, clock alive, floor-bounce real.** ⚠️ Explicit-stepper note: the floor prefactor stiffens forces ~`1+2β\|u\|` ≈ 4× — derate dt accordingly (the phase-3 lesson: full-dt heated numerically past τ≈40; dt/2 clean) | ✅ 2d sandbox (`m5_8_2d_quartic_saturation.py`) · ✅ 2e matrix (`m5_8_2e_invariant_matrix.py`): **Skyrme `β_E·u_E²` also saturates (robustly, align 0.94–0.95) but damps the clock 10× — the SIGNED quartic is the physics-preferred invariant (floors only the fuel; the clock GROWS 7.6e-3 → 2.4e-2 where the full-Euclid twin DECAYS to 6.7e-3 — the signature IS the engine, at full nonlinearity); the saturated state's ω headline RETIRED 2026-06-07 (the "ω₀ = 0.262 + exact comb" was an FFT-window artifact — m5_8_2h W1); the valid measurement: QUASI-periodic, fundamental ω₁ ≈ 1.1 + 2ω₁ harmonic, attractor across starts (2h)**; f64 tracking ~1% · ✅ **G-2c-1 CLOSED (2i, 2026-06-07): the defect HOLDS under the quartic, resolution-ROBUST** — structural decay SLOWS 24³→48³ at every matched t (anti-M5.7), plateau 0.8–0.88, mask-insensitive · ⚠️ **β IS CUTOFF-DEPENDENT** (u_min(seed) deepens 7× at 48³ ⇒ the anchored quartic strength runs with the lattice; ω₁ inherits a scheme systematic, 1.15→0.83 across grids — the EFT-flavored property of this Lagrangian layer, recorded for the calibration error budget) · **the saturated state CLASSIFIED (2n): a MOLTEN CLOCK** — persistent comb + low-dimensional chaotic dressing (λ_max +0.4–0.7/τ, D₂ 2.7–3.0), molten-ness growing with excitation (the DTC "melting" phase) · **the invariant matrix COMPLETE (2l): A4 amplitude quartic GEOMETRICALLY PINNED** (⟨M,M⟩_s constant on the conjugation orbit — no lever arm; the fuel is curvature-sector, which is WHY the signed-u form is the correct invariant class) + the covariant `𝒮+β𝒮²` deferred-with-reason (P cubic in Ṁ; static ≡ validated) · **ω₁ IS RIGID (2m v2): the breathing fundamental is energy-INDEPENDENT** — constant within one FFT bin across a 2.6× rest / 3.9× state-energy dressing family (exponent ω ∝ H^0.03 vs the ZBW law's 1) ⇒ ω belongs to the CORE (scale-free at V=0; the true mass family needs the V-on/Faber-r₀ stack) — the attractor statement extended from start- to energy-independence · **THE EXCITATION FLOOR (2o): a cold defect cannot be prepared by kicking** — the seed's spontaneous release floors excitation at ~1.7× H_rest regardless of kick (un-sittability's preparation-physics face); cooling requires the damped-settle protocol · **THE GROUND CLOCK REGULARIZES (2o settle-route): the coldest measured state (T = 2.36 after a 217× damp-drain) reads near-regular (λ +0.11 ≈ bias floor, D₂ 1.68 vs 2.7–3.0 hot)** — the molten dressing is EXCITATION, not pathology; full coldness forbidden by un-sittability (the self-start mechanism) · **THE CLOCK IS J-NEUTRAL (2p): the rotation Noether charge (gated, ⟨P,Ṁ⟩=2T exact) reads ZERO on the kicked clock** — the breathing channel carries no net frame angular momentum (spin-½, if carried, lives in other sectors; box torque bounds the measurement at 24³) · 🚧 stiffness-aware stepper + fast-clock phase probe + production port |
| **Breathing-state BVP attack (Track C C1+C2-B, M5.8.2f/2f2)** | The M6 v9 transfer (pose the state directly, free the eigenvalue) applied to the reduced `H_eq = V + βQ + p_Θ²/4K_ΘΘ` over radial profile families — made affordable by the EXACT slope-polynomial reduction (`∂_iM = A_i + b′·x̂_i·∂M/∂θ + ψ′·x̂_i·∂M/∂Θ`, all slope-slope commutators vanish ⇒ V,K deg-2 / Q deg-4 in `(b′, ψ′)`; coefficient tables over (ψ,θ,r), surrogate ~0.1 ms/eval). Operating principle: the surrogate GUIDES, the direct quadrature DECIDES (machine-exact canaries; stencil envelope measured; zigzag/cancelling-stack descents excluded by direct DIFFERENTIAL audits). C2-B required the **exact 9-pt phase average** (V,K harmonics ≤ e^{i8ψ}, Q ≤ e^{i16ψ}) — the 2b 4-pt secular set is not shift-invariant under twist | ✅ C1 (`m5_8_2f_breathing_bvp.py`) — **NEGATIVE-DECISIVE for the global-clock ansatz: NO interior ghost minimum at any (β, p_Θ)** — V+βQ rises monotone 8.4 → 1.5×10¹⁰ along the ghost branch (static u positive-dominated, u_min −0.034); the 2b saddle-only verdict survives the quartic VARIATIONALLY · byproducts: 2b-1's E* 2.606 → **2.93 direct** (its Δb=0.1 spline artifact); the β=0 slope-fuel channel opens only at θ≳1.3 and is NOT net-extractable · ✅ C2-B (`m5_8_2f2_localized_clock.py`) — **the static clock-phase twist `R(Θ+ψ(r))` is PURE COST** (the m5_6_2b KG twist-mass, variationally): the twist-fuel pocket is real but 5 orders too weak (`v₀₂=−0.0031`/voxel, GEM-dip window θ∈[0.2,0.5] only); NO twisted state below the t≡0 control; the marginal ΔH=−0.002 candidate refuted by the direct twist-differential dial (a near-cancelling 3-Gauss stack artifact) · **⇒ both static reductions closed ⇒ the breathing state is IRREDUCIBLY TIME-DEPENDENT (`ḃ≠0` IS the bounce)** · ✅ C3 (`m5_8_2f3_breather_orbit.py`) — the reduced CC dynamics (exact-phase kinetic tables, `H = ¼P𝕂⁻¹P + V + βQ`): **`K_bΘ ≡ 0` exact** (2b's cross-term was sampling residue — gyroscopically decoupled) and **THE UN-SITTABLE MINIMUM: `K_bb(a*) = −67.6 < 0` direct with `U″ = +1723 > 0`** — the dressed ground state is energetically minimal but kinetically ghost ⇒ compulsory spontaneous motion (G-2c-3's spontaneity, DERIVED); at anchored β the minimum is fully ghost-kinetic and the orbit exits through `det 𝕂 = 0` where any net-inertia reduction is ill-posed ⇒ **containment is irreducibly many-mode — Track C closes at its natural boundary** · ✅ THE FIELD HANDOFF EXECUTED (`m5_8_2g_spontaneity.py`): **spontaneity CONFIRMED at field level, dt-converged** — a damped-settled config restarted with `P=0` exact regrows T 0 → 5.76 by τ=4000 (H conserved 0.5%, dt/2-reproduced to 4 digits); the clock probe reads `\|s\|~1e-18` machine-zero while the amplitude channel carries O(1) motion (the old G-2c-3 null = probe blindness, unconditional); unkicked reaches the SAME breathing state as kicked (attractor-consistent) · · ✅ THE ω-ATTRACTOR CLOSED (`sandbox_vn/m5_8_2h_omega_attractor.py`, 2026-06-07): dense dt/2 spectra, 4 arms — kicked/EXACTLY-unkicked/jittered share the breathing fundamental **ω₁ ≈ 1.1 (± one bin) + 2ω₁** (detrend-stable across 2.5/5/10 τ filter windows — the dial discipline applied to the FFT) ⇒ ω is a property of the STATE — **G-2c-3 ✅ both halves**; the 2e "strict 0.262 comb" exposed as a window artifact by the pre-registered W1 scrutiny; rod readout: the breather is near-isotropic + core-centered (A_z ≈ 1.1 vs static rod baseline 5.6; frac_rod = volume fraction), the settled arm's motion flows TOWARD the core · remaining: the stiffness-aware stepper for the deep-floor regime + the settled arm's convergence horizon |

**VALIDATION POLICY (the headless-first decision, Rodrigo 2026-06-07):** model validation is HEADLESS — sandbox scripts, gates, npz caches, trend tables, plots; rendering is communication/demo value only and gates nothing scientific. Evidence: the entire M5.8.2 arc (the dt-invariance proof, the quartic saturation, the kill-control, the ω-attractor, Track C, the spontaneity) was produced with zero renders; f32 was never the blocker (f32≡f64 ~1% everywhere checked); the GUI defects that cost real time (launcher dt-wipe, display sync) changed no physics. The rendering stack remains the platform's public face; the physics pipeline does not route through it. (Roadmap: STATUS AT A GLANCE + backlog NG-6, 2026-06-07.)

**The production v1 WORKING recipe (safe — GUI-verified 2026-06-05; `xparameters/_topo_dressed4d.py`):**

| Ingredient | Production form |
| --- | --- |
| Seed | `seed_dressed_hedgehog_M`: biaxial frame + melt × boost dressing `exp(b*·w(r)·B₁)`, `B_STAR=0.13`, `RW_FRACTION=0.29`; clock tangent `M_psi_am` written; optional `CLOCK_KICK` (no `ti.cosh/sinh` — exp identity) |
| Flux | `SIGNED_FLUX_4D: False` (default) ⇒ **Euclidean** flux with the time axis LIVE (the `(α,3)` components evolve; no time-freeze clamp). The Minkowski-SIGNED flux stays OFF pending the constrained-kernel port — it is structurally unstable under cheap inertia |
| V(M) | the dressed well `V = c·(t − t*(x))²` with the per-voxel target `t*(x) = Tr(M_sp_seed²)` (`compute_tstar`, once post-seed) — the seed is the exact V-equilibrium (kills the static off-minimum push the standard well exerts on the dressed shell) |
| Inertia | diagonal faithful-lite `m(x) = 1 + km·dx²·Σ‖∂M‖²` (`KM_INERTIA_4D=30`) — vacuum limit exactly 1 (3D behavior untouched) |
| Guards | coherent-`(α,3)`-drift subtraction (3-mid-plane sampling, Metal-safe); `DT_SCALE_4D=0.5`; stable mask computed + smoothed (diagnostic; zeroed under the safe default) |
| Expected GUI phenomenology | **energyH CONTAINED** (the pinned well) ✅ observed. **energyF dissipates + director glyphs wander — EXPECTED**: V confines amplitude NOT orientation (the M5.6.5c finding, the M5.7 dispersal root cause); under Euclidean dynamics nothing coheres the orientation sector. **Orientation cohesion is precisely what the Minkowski-signed dynamics provides** (2b-2: the signature coheres the clock) — the v1 GUI shows its absence, Option B turns it on |

Reference cross-check: the article's electron-field generator (§10d) — our 4D seeder must
reproduce that `M(x)` with the `ψ`-sweep as the clock. The generator → force dictionary is §10d.

#### The integrator constraints (the ghost lesson)

| Constraint | Why | Source |
| --- | --- | --- |
| NO naive explicit leapfrog of the free 4D field | the `(−,+,+,+)` `Γ₀` terms make the time-axis mode a negative-kinetic GHOST — the 1D toy's free leapfrog blew up at `t≈1.2` | `m5_8_0b_toy_leapfrog.py`, §10a |
| Evolve on the constrained / bounded-energy manifold | collective-coordinate reduction makes the clock phase cyclic ⇒ `p_Θ` Noether-conserved ⇒ stable (1D: drift ~10⁻¹⁵ over t=60) | `m5_8_0b_collective_clock.py` |
| Evolve the `O` rotation/boost DoF, not M's raw components | the faithful metric `K` is degenerate (trace = null mode) | §9 |
| The clock is dynamical only ON the hedgehog | `[M_x, Ṁ] = 0` on a uniform background — the commutator structure auto-localizes the clock energy (the 1D `det H ∝ φ_x²` shadow) | `m5_6_2b` + 2a vacuum control (exactly 0) |
| Bounded energy is the physics guard | topology bounds E below — "mass can't go ≤ 0" (Duda) | §10c; 1D floor `E* = 2.1257 > 0` |

#### Verification gates — M5.8.2a (all PASS, 2026-06-05)

Anchor script: `sandbox_v8/m5_8_2a_4d_hamiltonian.py` (numpy quadrature, 48³, exact §10d ℋ, V=0).

| Gate | Statement | Measured result |
| --- | --- | --- |
| G1 | bare clock costs: `C_neg(b=0) = 0` exactly, `C(0) > 0` — no crystal while the time axis is inert (the M5.7 functional null) | ✅ `C_neg = 0.0` exact; `C_pos = +17.07` |
| G2 | THE FUEL: `∃b` with `C(b) < 0` — Minkowski wins once the time axis is dressed | ✅ all 9 plane×axis combos negative at `b=0.6`; strongest `(1,0)`-plane, `a=2`: `C = −678` |
| G3 | crystal threshold: `E(ω,b) < E(0,0)` beyond finite `ω_c` — the dressed oscillating defect beats the static one | ✅ `ω_c(b=0.4) = 0.555`; at `b=0.2` the dressed static already wins (`A=0.39 < 5.97`) |
| G4 | signature control: Euclidean flip (all blocks +) ⇒ no fuel | ✅ `C_E(b) ∈ [+74.9, +2.34×10⁴] > 0 ∀b` |
| G5 | static mass guard: `A(b) > 0` on a fine scan (§10c at ω=0); the rigid-family ω-unboundedness is the expected toy-without-`βR⁴` artifact — cap = profile response (2b), onset visible | ✅ `A_min = 0.389 > 0`; `C_pos/C_neg`: `0.87 → 0.32 → 0.36 → 0.52 → 0.69 → 0.83` (rises back toward parity) |
| G6 | localization: fuel density core-localized — the clock lives ON the defect | ✅ falls 10 decades by r≈9 (tail/peak < 10⁻⁴); dressed-vacuum control exactly 0 |
| G7 | apolar doubling: `M(ψ)` period = π, not 2π ⇒ `ω_M = 2·ω_clock` | ✅ `max\|M(π)−M(0)\| = 5.6×10⁻¹⁷` vs `\|M(π/2)−M(0)\| = 0.30` |

1D anchors carried in (M5.8.0, all reproduced by quadrature + CC dynamics):

| Anchor | Value |
| --- | --- |
| Analytic kink clock (Eq.5) | `ω* = √(70/61) = 1.0712`, `E* = 2.1257` — CC run holds it exactly, drift 1.88×10⁻¹⁵ |
| Optimized profile (Fig.1) | `ω* = 1.2898`, `E* = 2.0252` |
| Energy floor (0d-c) | `E = 2.1257 > 0` over `(w, ω) ∈ [0.6, 2.5] × [0, 2]` |

#### Open items the recipe still needs

| Item | What it adds | Where |
| --- | --- | --- |
| ✅ M5.8.2b — CC landscape (2026-06-05, `m5_8_2b_cc_clock.py`, 8/8 gates) | the global rigid clock mode RULED OUT (ghost branch saddle-only — net inertia vanishes at the window edges, a mode-choice artifact; `ℒ=−ΣF²` is velocity-quadratic in any CC family, no `βR⁴`-style quartic exists); the dressed defect + slow clock is a stable center BELOW the bare static defect (`E*=2.61 < 6.14`); the ghost runaway documented = the channel 2c must not expose | roadmap M5.8.2b log |
| ✅ M5.8.2b-2 — field-level clock (2026-06-05, `m5_8_2b2_field_clock.py`, 6/6 gates) | the CORE-LOCALIZED twist on the dressed hedgehog WORKS: sourced, bounded, **single coherent mode ω=5.86 (fft/zc agree 0.4%) — the Minkowski signature COHERES the clock** (Euclid twin: multi-mode, 89% split); best core retention (0.66 Mink > 0.59 undressed > 0.46 Euclid). Ghost geography: stable region (K>0 ∧ Q(∇ψ) PD) = 77% at `b*`; the Q-not-PD fuel shell runs away at λ≈15.6/t in the linearized kernel — 2c's constrained job | roadmap M5.8.2b-2 log |
| 🔶 M5.8.2c — production port (Steps 1–3 ✅: 2c-1 spike, Taichi port, headless 27/27 + GUI safe-v1 verified; **Option B spike B-1 ✅ ALL GATES PASS** — the constrained kernel is in Taichi, exact and fast) | the time axis is LIVE in production (safe v1, working recipe above); the constrained spectral-projection kernel **validated in Taichi** (`m5_8_2cb_taichi_constrained.py`: [J1] Jacobi exact, [B1] ≡ numpy machine precision, [B2] f32 benign, [B2m] Metal clean, [B3] physics repro, [B4] 14.5 ms/step at 64³ = 69 steps/s — the 5–10× fear dissolved to 1.3×). **B-2 production wiring ✅ DONE** (`INTEGRATOR_4D: "constrained"`, the (M, P) state machine + 6 kernels + `_topo_dressed4d_signed`; headless §6e 6/6, 1000-step 64³ Metal bounded at 69 steps/s); **remaining: B-3 GUI smoke (`openwave -x _topo_dressed4d_signed` — energyH containment + the glyphs should COHERE). Carries the three open physics gates** (roadmap 2c): G-2c-1 the defect holds ITSELF together (2c-1 first answer: held 900 steps, decisive M5.7-config re-run pending); G-2c-2 fuel-shell saturation (2c-1 + B-1: H plateaued, v-cap never engaged — first positives); G-2c-3 ω self-selection (kicked clock holds, spontaneous start machine-zero — probe/symmetry diagnosis pending) | 🚧 B-3 GUI smoke NEXT |
| M5.8.3 — electron clock `ω` | seed + measure: the dimensionless self-consistency `ω·ℏ/(2H_rest) → 1` (`ℏ ↔ δ`); absolute Hz via the Faber `r₀` scale-fix — the GROUP HEADLINE | after 2c |
