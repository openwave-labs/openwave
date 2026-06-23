# Duda thread study doc — 2026-05-12 → 2026-05-15

Working overview consolidating the multi-day exchange with Jarek Duda on the models-of-particles list (Robert Close + Jeff Yee + Models group also in thread). Captures the physics conclusions reached during the conversation so they can be re-read in one place before being split into proper research files topic-by-topic.

**Status** — substrate gating question CLOSED. Refactor green-lit. Many forward implications captured below.

**Cross-refs (current):** [0c_roadmap.md](0c_roadmap.md), [3b_lagrangian_roadblocks.md](3b_lagrangian_roadblocks.md), [1a_lagrangian_framework.md](1a_lagrangian_framework.md), `../theory/liquid_crystal_model.pdf` (Duda arxiv:2108.07896 v7), `../theory/liquid_crystal_particles.pdf` (Duda slides, 51 pages).

---

## Contents

1. [Duda's responses summarized](#1-dudas-responses-summarized)
2. [Substrate gating question — CLOSED](#2-substrate-gating-question--closed)
3. [The matrix field — what M = ODO^T actually is](#3-the-matrix-field--what-m--odot-actually-is)
4. [The grid stays 3D — matrix size ≠ grid size](#4-the-grid-stays-3d--matrix-size--grid-size)
5. [What the matrix elements represent — NOT ψ, NOT position](#5-what-the-matrix-elements-represent--not-ψ-not-position)
6. [3D vs 4D — what the 4D extension actually adds](#6-3d-vs-4d--what-the-4d-extension-actually-adds)
7. [Force unification — the corrected mapping](#7-force-unification--the-corrected-mapping)
8. [Eigenvalue → physics mapping (Duda's direct text + curvature layer)](#8-eigenvalue--physics-mapping-dudas-direct-text--curvature-layer)
9. [Topology on Close + Yee frameworks](#9-topology-on-close--yee-frameworks)
10. [Refactor strategy — two refactors, sized differently](#10-refactor-strategy--two-refactors-sized-differently)
11. [Slides content (51 pages) — instrumental beyond the paper](#11-slides-content-51-pages--instrumental-beyond-the-paper)
11b. [Couder/Bush walking-droplet deck (../theory/Couder.pdf)](#11b--couderbush-walking-droplet-deck-theorycouderpdf)
12. [Open questions & implications](#12-open-questions--implications)

---

## 1. Duda's responses summarized

### 2026-05-14 first reply — three next directions

Duda acknowledged the M5.1 Coulomb + topological charge result and pointed to three further directions, all of which already line up with our M5 roadmap:

| Pointer | Our phase | Mechanism |
| --- | --- | --- |
| Running coupling (Faber) | M5.5 | regularization via Higgs |
| Cornell potential | M5.9 | quark string + fractional charge |
| Gravitoelectromagnetism | M5.8 | SO(1,3) boost dynamics |

He attached two images: image 1 (Cornell / Abrikosov vortex), image 2 (SO(1,3) affine connection with `M = ODO^T`).

**New reciprocal ask:** "Can you make such topological charge quantization work also for Robert's and Jeff's approaches?"

### 2026-05-15 second reply — eigenvalue clarification + slides

In response to our follow-up, Duda clarified the eigenvalue → physics mapping (direct quotes) and pointed at the slides as the visual source:

| Eigenvalue | Duda's exact wording |
| --- | --- |
| 1 | "EM 'tilts' having the highest Lagrangian contributions" |
| δ | "low energy twists for quantum phase — the ℏc term in QED Lagrangian" |
| g | "energy contributions of boosts for gravity — much larger as e.g. its hedgehog would be a black hole, while spatial hedgehog is just electron" |

So `D = diag(g, 1, δ, 0)` is read as: each eigenvalue tags **which kind of orientation change** dominates the Lagrangian along that axis — tilts (EM), twists (QM), boosts (gravity), null (time).

**Potential V(M) — directly quoted:**

> "There is potential with minimum in this `diag(g, 1, δ, 0)` — getting EM+QM+GEM vacuum dynamics, and activating this potential especially to regularize infinite energy of e.g. charge."

So `D = diag(g, 1, δ, 0)` is **the preferred shape** that minimizes V(M). The role of V is to (a) define this shape as the vacuum state and (b) regularize divergent energies (e.g. the infinite Coulomb self-energy of a point charge) by allowing the field near a defect core to deviate from D smoothly rather than blowing up.

**Two further open notes:**

- V(M) details "most difficult" — could be like in Landau-de Gennes or slightly different, still an open research question
- Potentials are typically effective — there might be an even deeper "anisotropic fluid" beneath the matrix field (this matches OpenWave's existing granule-level picture: matrix would be effective, granules deeper)

---

## 2. Substrate gating question — CLOSED

The question we were holding the refactor on was: matrix `M = ODO^T` (6 DoF) vs traceless Q-tensor (5 DoF)?

**Answer:** full real symmetric matrix `M = ODO^T`, no Q-tensor pivot. Confirmed by:

- Image 2 writes the field explicitly as `M = ODO^T` with `D = diag(g, 1, δ, 0)` (4D form)
- Slides reaffirm the same construction on every page that touches the substrate
- Duda used the same notation in his 2026-05-15 follow-up

Refactor (Vector(3) ψ → 3×3 matrix M) is green-lit with no ambiguity.

---

## 3. The matrix field — what M = ODO^T actually is

### Spectral decomposition

| Symbol | What it is | What it carries |
| --- | --- | --- |
| `M(x)` | real symmetric matrix | the local field state |
| `D` | fixed diagonal | shape (eigenvalues) |
| `O(x)` | rotation matrix | orientation (dynamics) |

Any real symmetric matrix can be written `M = ODO^T` (spectral theorem). The split has physical meaning:

- **D is global, frozen.** Same eigenvalues at every voxel — three (or four) widely-separated scales `g >> 1 >> δ ~ ℏ`. Not stored per voxel.
- **O(x) is the dynamical field.** Per-voxel rotation telling you which way the eigenvalue axes are pointing at that point.
- **M is just a convenient package** of `O(x)` + the global `D`.

The actual degree of freedom per voxel is the orientation `O(x)`. A defect is a point where `O` can't be combed smooth — like the hairy ball theorem in action.

### The eigenvalue hierarchy

```text
D = diag(  g,     1,     δ,     0  )
            │      │      │      │
            │      │      │      └─ time axis (only in 4D, M5.8)
            │      │      └────────  δ ~ ℏ   → QM scale ("twist" axis)
            │      └─────────────── unity    → reference / EM scale
            └─────────────────────── g >> 1  → gravity scale
```

The three (or four) widely-separated scales are physics-motivated, not ad-hoc tuning (Duda 2026-04-19): they map onto three distinct physical regimes — QM, reference/EM, gravity.

### Why symmetric matrices have fewer numbers than they look

Symmetry forces upper-triangle = lower-triangle, so an N×N symmetric matrix has `N(N+1)/2` independent entries:

```text
3×3 symmetric:                 4×4 symmetric:

   [ a   b   c ]                  [ a   b   c   d ]
   [ b   d   e ]                  [ b   e   f   g ]
   [ c   e   f ]                  [ c   f   h   i ]
                                  [ d   g   i   j ]

   diagonal:  a, d, f    (3)      diagonal:    a, e, h, j     (4)
   off-diag:  b, c, e    (3)      off-diag:    b,c,d,f,g,i    (6)
   total:                6        total:                      10
```

Formula: `N(N+1)/2`. 3×3 → 6 entries. 4×4 → 10. 5×5 → 15. Etc.

Why symmetric specifically: it's structural, not an arbitrary choice:

```text
M  =  O · D · O^T
M^T = (O · D · O^T)^T = O · D^T · O^T = O · D · O^T = M
```

Anything that can be written as `rotation × diagonal × rotation^T` is automatically symmetric.

---

## 4. The grid stays 3D — matrix size ≠ grid size

```text
                  Position in 3D space (x, y, z)
                            │
                            ▼
                  ┌──────────────────────┐
                  │  voxel at (1, 4, 2)  │
                  ├──────────────────────┤
                  │   Vector(3) ψ        │   ← M5.0–M5.3 (current)
                  │   = (ψx, ψy, ψz)     │     3 numbers per voxel
                  │                      │
                  │       OR             │
                  │                      │
                  │   3×3 matrix M       │   ← M5.4 target
                  │   = 6 numbers        │     (real symmetric)
                  │                      │
                  │       OR             │
                  │                      │
                  │   4×4 matrix M       │   ← M5.8 target
                  │   = 10 numbers       │     (Minkowski symmetric)
                  └──────────────────────┘
```

The grid stays 3D spatial throughout (voxels at `(x, y, z)`). What changes is the object stored at each voxel:

| Matrix size | Acts on vectors in | What it can describe |
| --- | --- | --- |
| 3×3 | R³ | orientation/biaxiality in 3D space |
| 4×4 | R^(1,3) | orientation in Minkowski spacetime |

The "3" in 3×3 matches the "3" in 3D space because the matrix's job is to rotate/scale 3D vectors at each point. Same dimensional count, different role.

The 4×4 matrix in M5.8 still lives at each 3D voxel — but now it's an operator on 4-vectors. The 4th axis is **time as an algebraic dimension inside the matrix**, not a 4th grid axis.

---

## 5. What the matrix elements represent — NOT ψ, NOT position

The matrix encodes the local **biaxial orientation state** of the medium — analog of a liquid-crystal's nematic order tensor:

| Object | What it is at each voxel |
| --- | --- |
| Position | grid coordinate (fixed, x, y, z) |
| ψ (Vector 3) | one preferred direction |
| M (3×3) | three preferred axes + their orientations |

### Visual analogy — biaxial top at each voxel

```text
Vector(3) ψ at a voxel:           3×3 matrix M at a voxel:

        ↑                                ↑                
        │  "the director                 │  ↗             
        │   points this way"             │ /  ──→         
        │                                │/               
                                                          
        one little arrow                three little arrows,
        pointing somewhere              each with its own
        in 3D                           stiffness, oriented
                                        in a local frame
        = 3 numbers                     = 6 numbers
```

What varies per voxel is **which way the three axes point** — the matrix `O(x)`. The three stiffnesses `(g, 1, δ)` are the same everywhere.

A hedgehog defect = a point where you can't smoothly orient the top — like trying to comb a hairy ball, there's always one point where the orientation has to fail. That failure point IS the particle.

### Why this matters for physics

| Per-voxel quantity | Vector(3) ψ | 3×3 matrix M |
| --- | --- | --- |
| Local preferred axes | 1 | 3 (biaxial) |
| Numbers stored | 3 | 6 |
| Particles supported | 1 mass scale | 3 lepton families |
| Eigenvalue meaning | n/a | stiffness per axis |

Vector(3) ψ has only one direction at each point → defect can only wind around one axis → one mass scale → one particle family.

3×3 matrix M has three directions at each point → defect can wind around any of the three → three mass scales → three lepton families. **That's the physics reason for the substrate upgrade.**

---

## 6. 3D vs 4D — what the 4D extension actually adds

The 3→4 jump is the framework's biggest conceptual move. It's the difference between a field on space and a field on spacetime.

### Three things change simultaneously

| Aspect | M5.4-M5.7 (3D) | M5.8 (4D) |
| --- | --- | --- |
| Where field lives | R³ space | Minkowski R^(1,3) |
| Symmetry group | SO(3) rotations | SO(1,3) Lorentz |
| Metric | Euclidean +++ | Minkowski −+++ |

In plain terms:

- **3D:** time is external — you evolve `M(x, t)` step-by-step via leapfrog. Space and time are separate. No Lorentz invariance.
- **4D:** time is inside the algebra — it's the 0th index of every tensor. Space and time mix under boosts. Manifest Lorentz invariance built in.

### Why anyone wants 4D — three reasons

```text
Reason 1:  Lorentz invariance is real physics
─────────────────────────────────────────────
   Special relativity (length contraction, time
   dilation, E = γmc²) only emerges if you treat
   space + time on equal footing. M5.4-M5.7 are an
   approximation; M5.8 is the genuine article.

Reason 2:  Negative-energy contributions are automatic
──────────────────────────────────────────────────────
   In Euclidean signature (3D), the Hamiltonian is
   manifestly positive — that's why static defects
   collapse (Derrick's theorem).
   In Minkowski signature (4D), the indefinite metric
   creates negative-energy ΓΓ̃ rotation-boost terms.
   These auto-propel the de Broglie clock — the
   "particle is time-periodic" mechanism becomes
   structural, not engineered.

Reason 3:  Gravity comes for free
─────────────────────────────────
   The boost component Γ^0 of the connection is what
   image 2 labels as gravity (gravitoelectromagnetism).
   You only have a boost component once you're in 4D.
   So GEM emerges from the same matrix algebra that
   gives QM (twist) and EM (tilts).
```

### What the "0" eigenvalue means

`D = diag(g, 1, δ, 0)`. The 0 is the time-axis eigenvalue. It makes the time direction null (light-like) in the eigenvalue structure — that's what creates the negative-energy contributions when `O ∈ SO(1,3)` rotates the diagonal into a non-diagonal mixture.

Without the 0, you'd have a regular Lorentzian metric structure. With the 0, the time axis is degenerate and the dynamics get the time-periodic propulsion property as a consequence of the algebra, not as an added force term.

### Two notions of "local time" — easy to conflate

| Notion | Where | What it means |
| --- | --- | --- |
| Per-voxel local-time | relational-time picture | `f = c/λ` varies per voxel |
| M5.8 4D extension | Duda paper §V | Lorentz algebra at each voxel |

These are related but not identical:

- The per-voxel local-time picture says: the rate of change differs per voxel because λ(x) is a scalar field. That's a per-voxel scalar clock-rate.
- M5.8 4D says: the matrix algebra at each voxel becomes Lorentz-covariant. The boost component `Γ^0(x)` varies per voxel, a per-voxel "time-direction tilt".

Both can coexist eventually — they answer different sub-questions about "what does local time mean here?". The grid stays 3D in both; neither makes time a 4th grid axis.

---

## 7. Force unification — the corrected mapping

A subtle but important correction: **QM is not a force.** It's the framework (Schrödinger / Klein-Gordon / Dirac) inside which forces act as potentials or couplings.

```text
   QM (Schrödinger / KG / Dirac)         ← a wave-equation framework
   = how a single particle's wavefunction   for HOW particles move
     evolves in time

           ↓ inside this framework ↓

   Forces (EM, gravity, strong, weak)    ← couplings/potentials added
   = how particles influence each other     INTO the wave equation
```

In the matrix framework: δ-axis twist is the mechanism that produces the QM wave equation as an emergent fact (paper Fig. 9). That's the framework. Forces are produced by other mechanisms.

### All known forces, mapped to matrix mechanisms

| Force | Mechanism in M | Phase |
| --- | --- | --- |
| EM | 1-axis tilts (Maxwell) | M5.5 / M5.6 |
| Gravity | g-axis boosts (GEM) | M5.8 |
| Strong | 1D vortex string + Cornell | M5.9 |
| Weak | defect-class transitions | partial (gap) |

Note these are four different geometric mechanisms in the same matrix field, **not** all four on different axes.

### EM (1-axis tilts)

Voxel-to-voxel tilts around the unity-eigenvalue axis. Radial tilt pattern → electric field. Curl tilt pattern → magnetic field. Maxwell's `F_μν F^μν` term emerges directly.

### Gravity (boosts)

The 4D-extension boost component `Γ^0` — orientation tilts into the time direction. Linearized gravity has the same form as Maxwell (gravitoelectromagnetism), generated by this component instead of by tilts.

### Strong force (topology, not axis)

Not an axis — a different kind of topological defect altogether. Quarks are endpoints of a 1D topological vortex *string* (not a 0D hedgehog). The string carries energy `σ ≈ 1 GeV/fm` per unit length — that's confinement. Cornell potential `V(r) = −α/r + σr` (Coulomb + linear) is what comes out. M5.9 territory.

### Weak force (gap with partial answer)

Duda's paper §III–V doesn't give weak interactions a crisp matrix-mechanism the way it does for EM/gravity/strong. The slides (pages 31–32, 36) show **beta decay as topology reconnection**:

```text
neutron (-+-)  →  shift  →  split (+ -)  →  reconnect  →  proton (+) + electron (-) + neutrino (○)
                            energy release
```

So weak interactions are **topology-changing events** — defect-class transitions via reconnection. Not a force in the EM/gravity sense, but a structural mechanism the framework supports.

Likely full candidates:

- defect-class transitions (e.g. hedgehog → closed vortex loop = beta decay producing a neutrino)
- topology-changing events at the matrix level
- chiral structure of the matrix algebra

Honest answer: still an open research question, partially addressed by the slides.

### Atomic orbital "force" — not a fundamental thing

Atomic orbital binding isn't a separate fundamental force. It's:

```text
Atomic orbital binding  =  EM (long-range 1/d)  +  standing-wave quantization
                           electron ↔ nucleus      de Broglie wave fits orbit
```

The 1/d part is M5.1 territory (already done). The orbital quantization is the M3 near-field standing-wave physics that Jeff Yee flagged carries through three regimes:

| Regime | Same standing-wave mechanism |
| --- | --- |
| intra-particle | binds wave centers into a particle |
| strong residual | binds nucleons inside nucleus |
| atomic orbital | binds electrons to nucleus |

So atomic orbital force lives in M5 as "EM tilts + standing-wave quantization" — no new axis or mechanism. Same applies to chemical bonds and van der Waals — all are EM at different length scales.

### Unification verdict

| Force | Unified in matrix framework? |
| --- | --- |
| EM | yes (tilts) |
| Strong | yes (vortex string + Cornell) |
| Gravity | yes (boost component) |
| Weak | partial — topology reconnection sketched |

Three of four fundamental forces fit cleanly. Weak is the genuine gap (no clean SU(2) chiral mechanism yet), but beta decay does appear as a topology-reconnection event in the slides.

---

## 8. Eigenvalue → physics mapping (Duda's direct text + curvature layer)

Two layers to keep separate:

### 8.1 Duda's direct text (what each eigenvalue means physically)

| Eigenvalue | What it tags | Direct quote |
| --- | --- | --- |
| 1 | EM tilts | "highest Lagrangian contributions" |
| δ ~ ℏ | QM twists | "quantum phase — ℏc term in QED Lagrangian" |
| g >> 1 | gravity boosts | "much larger — hedgehog = black hole" |
| 0 (4D only) | null direction | (clock-propulsion from spacetime signature) |

Read it as: each eigenvalue picks **which kind of local orientation change** carries the Lagrangian weight along that axis. Tilts → EM. Twists → QM. Boosts → gravity. Null → clock.

### 8.2 V(M) shape and regularization (Duda quoted)

The potential V(M) has its **minimum at the preferred shape** `D = diag(g, 1, δ, 0)`. That's how D becomes the vacuum state of the field. Two roles for V:

| Role | What it does |
| --- | --- |
| Vacuum-shape selector | enforces `D = diag(g, 1, δ, 0)` as the ground state |
| Charge regularization | lets the field deviate from D near a defect core to avoid the infinite Coulomb self-energy of a point charge |

The second role is what Duda flagged as "the hardest part" — getting the regularization form right (LdG, or "slightly different", or Faber's variant) is still open research.

### 8.3 Curvature math layer (from slides — one level deeper)

The eigenvalue → curvature mapping that appears in the Lagrangian `F_μναβ F^μναβ` is **combinations** of components, not single eigenvalues. From slide page ~26 (`QM+EM tilts EM` annotation):

| Curvature combination | What physics |
| --- | --- |
| `R^ee_μν = Γ_μ × Γ_ν` (tilt-tilt) | EM (high energy, main curvature) |
| `δ R^ee_μν` (tilt-twist scaled by δ) | QM phase coupling (low energy) |
| `R^gg_μν = Γ^g_μ × Γ^g_ν` (boost-boost) | GEM (gravity) |
| `R^eg_μν = Γ_μ × Γ^g_ν` (tilt-boost cross) | EM-gravity coupling (light bending, time dilation) |

So the eigenvalues act as **prefactors / weights** on the curvature combinations: `1` weights EM tilt-tilt, `δ` weights QM tilt-twist, `g` weights gravity boost-boost. The combinations are mathematically distinct objects (different parts of the antisymmetric `F` tensor), not just labels on the eigenvalues.

### 8.4 Lepton-family reconciliation

The 3 leptons come from **3D axis choice in the spatial part** (the slides explicitly show e/μ/τ as different axis-hedgehog choices). The 4D g eigenvalue is something else — the gravity-scale boost direction, where hedgehog = black hole.

| Mode | What the three λ give |
| --- | --- |
| M5.4–M5.6 (3D) | three spatial λ values → three lepton families |
| M5.8 (4D) | adding g (boost axis) → gravity coupling, not a 4th lepton |

This corrects our prior memory framing that mapped electron=δ, muon=1, tau=g as if each lepton lived on one eigenvalue axis. In the slides' own picture: the 3 leptons are 3 different orientation choices in the **spatial** 3-axis matrix, and the 4D g eigenvalue is a separate physical thing (gravity).

---

## 9. Topology on Close + Yee frameworks

| Approach | Substrate | Topology natural? |
| --- | --- | --- |
| Duda LdGS | matrix M | Yes (Brouwer Q) |
| Close spin-density | Vector(3) s, ∇·s = 0 | Maybe — helicity |
| Jeff EWT | pre-oscillating ψ | Maybe — ellipse axis |

### Robert Close (classicalmatter.org)

- Primary field is **spin density vector `s`** with `∇·s = 0` (Eq. 23). Divergence-free vector field on R³.
- Brouwer degree wants a unit-vector field. Could be done by normalizing `ŝ = s / |s|` — works but throws away magnitude info.
- Better topological invariants for divergence-free fields: **helicity** `H = ∫ s · (∇×s) d³x`, **linking numbers** of flux lines, **hopfion-class** knotted closed loops.
- Compatibility: plausible. Robert's framework is the natural home for **hopfion topology** — knotted divergence-free flux lines.
- Research lift: high. We have his FoP 2025 paper locally; classicalmatter.org adds more accessible material.

### Jeff Yee (energywavetheory.com)

- Substrate is the **pre-oscillating energy-wave medium** at `f₀ ~ 10²⁵ Hz`, `λ₀ ~ 28 am`.
- Particles are K=10 tetrahedron standing-wave interference patterns.
- Topology mapping: extract orientation of the local oscillation ellipse (M4's 6-phasor major axis) → that IS a director field.
- The **pre-oscillation is built in** — no need for Zitterbewegung propulsion via 4D Lorentz negative-energy. The clock is automatic.
- Compatibility: very plausible, arguably **the natural alternative propulsion mechanism** to Duda's 4D-Lorentz solution.
- Research lift: low. We already have M4's 6-phasor data layout.

---

## 10. Refactor strategy — two refactors, sized differently

| Refactor | What changes | Lift |
| --- | --- | --- |
| Vector(3) → 3×3 (M5.4) | substrate type + ops | big |
| 3×3 → 4×4 (M5.8) | +time axis, +Lorentz | medium |

Why the second is smaller:

| 3×3 → 4×4 work | Cost |
| --- | --- |
| Storage type | trivial (4×4 instead of 3×3) |
| Commutator / curvature ops | reusable if index-generic |
| Metric tensor | new (Minkowski) |
| 4-derivative `∂_μ` | new (time becomes index) |
| Leapfrog → covariant | partial rewrite |

If M5.4's operators are built index-generic (parameterize rank, use `ti.Matrix.field` cleanly), most of the matrix algebra carries forward. The 4D refactor becomes mostly type / metric changes, not algorithmic rewrite. Maybe 30–40% of the M5.4 code touches, not 100%.

**Don't pre-build 4D now** — the extra "0" eigenvalue and Lorentz machinery are dead weight for M5.4–M5.7 (spatial-only physics). Premature complexity at a phase that still has to prove its core works.

### Practical recommendation

| Phase | Build |
| --- | --- |
| M5.4 | clean 3×3 matrix substrate |
| M5.4 ops | index-generic where cheap |
| M5.5–M5.7 | stay 3×3, prove physics |
| M5.8 | 4×4 + Minkowski metric + 4-deriv |

The trap to avoid: trying to build a "future-proof 4D-ready" substrate now. Better to do M5.4 cleanly, learn from it, then do M5.8 with eyes open.

The thing to do *right* in M5.4: write operator kernels (commutator, curvature, Frobenius norm) that read matrix dimensions from the field type rather than hard-coding `3`. That's the cheap habit that makes the M5.8 promotion a type change, not an algorithm change.

---

## 11. Slides content (51 pages) — instrumental beyond the paper

Local PDF: `../theory/liquid_crystal_particles.pdf`.

### Top-level confirmations

| Question | Slide answer |
| --- | --- |
| Substrate = M = ODO^T? | confirmed everywhere |
| Full M vs Q-tensor? | full M, not traceless |
| Eigenvalue → physics map | 1=EM, δ=QM, g=gravity (in 4D) |
| 3 leptons mechanism | hedgehog around 3 different λ axes |

### What's instrumental beyond the paper

**1. Working Mathematica code for Coulomb numerical (page 18):**

```text
cos = 1 + (z-d)/Sqrt[(z-d)² + r²] - (z+d)/Sqrt[(z+d)² + r²];
n = {Sqrt[1-cos²]*x/r, Sqrt[1-cos²]*y/r, cos};
M = KroneckerProduct[n, n];           ← matrix from director
dM = {D[M,x], D[M,y], D[M,z]};
H = Sum[(dM[[i]].dM[[j]] - dM[[j]].dM[[i]])², {i,2}, {j,i+1,3}];
                                       ← Hamiltonian from commutator
Es = NIntegrate[H over field];
Fit: V(d) ≈ 1589.56 - 25.16/d         ← analytical Coulomb
```

This is the **M5.4 cross-validation target**. Port to Taichi, expect to reproduce the same `1/d` Coulomb fit on the matrix substrate. The constant `25.16` is the analytical version of our `R² = 0.978` empirical result from M5.1.

**2. Klein-Gordon from hedgehog — explicit closed form (page 32):**

```text
For hedgehog ansatz with phase ψ(t, x, y, z), in vacuum (A = 0):

    A^hedg = (x, y, z) / r²

    2 ∂_tt ψ = [(∇ - A^hedg)² + (A^hedg / |A^hedg| · ∇)²] ψ

    Klein-Gordon-like, dual formulation E ↔ B
```

That's the **M5.6 implementation target as a closed-form equation**. No need to derive — port directly.

**3. Clock-propulsion toy model with numerical values (page 33):**

```text
H = φ² + ψ² + (1 - φ²)² - R² + R⁴
    for curvature R = ∂_0 φ ∂_1 ψ - ∂_1 φ ∂_0 ψ   (Lorentz-invariant)

Numerical solution:
    φ = tanh(0.6326 x + 0.0198 x³ + 0.0203 x⁵)
    energy E ≈ 2.0252
    frequency ω ≈ 1.2898
```

This is the arXiv:2501.04036 kink + clock model with **specific numerical values to validate against**. Direct port to a 1D Taichi sandbox — can verify our infrastructure reproduces `ω` before M5.8.

**4. Eigenvalue → physics mapping (full picture):**

See [§8](#8-eigenvalue--physics-mapping-dudas-direct-text--curvature-layer) above for the consolidated table.

**5. Beta decay as topology reconnection (pages 31–32, 36):**

See [§7 Weak force gap](#weak-force-gap-with-partial-answer) above for the diagram and discussion. Partial answer to the weak-force question — topology-changing events at the matrix level.

**6. Mainstream landscape comparison (page 48):**

Duda explicitly positions his framework relative to:

- Standard liquid crystals (same field + potential, but Lorentz-invariant kinetic)
- Skyrmion models (same kinetic, potential → ~Higgs, topological charge → electric)
- Einstein's teleparallelism (4D liquid-crystal extension; same λᵢ in 3D, then 4D for EM ≫ QM ≫ GEM hierarchy)
- Spin hydrodynamics
- Lattice QCD (Yang-Mills kinetic; quarks as topological vortex strings instead of probability distributions)
- Standard Model (perturbative / effective — creation operators, probability distributions of *what?*)

His pitch: same kinetic and potential structure as standard LC/Skyrmion, plus topological-charge interpretation, plus 4D extension via teleparallelism. SM is the "perturbative approximation" of this.

### What this changes for our M5 phases

| Phase | Slide implication |
| --- | --- |
| M5.4 | port Mathematica code for cross-validation |
| M5.5 | V_LG variants spelled out explicitly |
| M5.6 | KG-from-twist closed-form available |
| M5.8 | clock toy model numerical anchors |
| M5.9 | Cornell + quark-string conflict diagrams |
| M5.4–M5.6 | 3-lepton mass mechanism is spatial axis choice |

---

## 11b — Couder/Bush walking-droplet deck (../theory/Couder.pdf)

Duda's **second** deck (45 slides, *"Hydrodynamical analogues of some quantum phenomena"*) — distinct from the §11 LdGS slides. This one is the walking-droplet / pilot-wave catalog (de Broglie–Bohm, Couder–Fort–Bush). Most of it confirms things we already have; the items below are what's **instrumental beyond our current docs**, mapped to M5 phases.

### 11b.1 Hydrodynamics ↔ EM dictionary (alternative M5.5 route)

The deck gives an explicit superfluid (ν = 0 viscosity) hydrodynamics ↔ electromagnetism dictionary — a *second* derivation route for M5.5's "EM from tilts", independent of the matrix-curvature one:

| Hydrodynamics (superfluid) | Electromagnetism | Analog |
| --- | --- | --- |
| vorticity `ω = ∇ × u` | `B = ∇ × A` | `ω ↔ B`, `u ↔ A` |
| Lamb vector `l = ω × u` | electric field `E` | `l ↔ E` |
| Navier-Stokes `∂u/∂t = −l − ∇φ_p + ν∇²u` | `∂A/∂t = −E − ∇φ` | vector + scalar potential |
| vorticity tendency `∂ω/∂t = −∇×l` | Faraday `∂B/∂t = −∇×E` | — |
| Coriolis force `−m(V × 2Ω)` | Lorentz `q(v × B)` | force law |
| turbulent charge `∇·l = u·∇×ω − \|ω\|² = ρ_n` | `∇·(ε₀E) = ρ_e` | charge density |

Gauge-condition parallel: electrodynamics Lorenz gauge `∇·A + (1/c²)∂φ/∂t = 0` ↔ hydrodynamics `∇·v + (1/c_s²)∂χ/∂t = 0` with `χ = v²/2`. **Relevance:** if the matrix-curvature EM derivation (M5.5) stalls, the superfluid-vorticity reading is a cross-check; both should yield Maxwell. Note also the Zeeman-as-Coriolis empirical plot (the walker Zeeman, §11b.3).

### 11b.2 Faber's explicit quantized-EM Lagrangian (concrete M5.5 target)

The deck states Faber's EM Lagrangian in the local-rotation-axis form — more concrete than what we had, and from **Faber himself** (our M5.6 regularization advisor):

```text
Γ_i = (∂_i u) × u          local rotation-axis "connection"
R_μν = Γ_μ × Γ_ν           curvature
L_EM = −(αℏc / 16π) R_μν · R^μν      with  F_μν ~ R*_μν   (E ↔ B dual)
```

**Relevance:** this is essentially the M5.5 EM term written for the director/rotation field. Pairs with the §11 paper's `A_μ = [M, ∂_μ M]`, `F_μν` curvature — same idea, Faber's normalization with the explicit `αℏc/16π` prefactor. Worth porting alongside the LdG potential in M5.5.

### 11b.3 Walking-droplet path-memory kernel + quantization-law catalog (the standing-wave / orbit-quantization side)

The deck's most reusable artifact is the **wave-memory kernel** that reproduces the entire walker phenomenology (Couder/Fort/Bush simulations):

```text
h(r, t_i) = Σ_p  A / |r − r_p|^(1/2)  ·  exp( −|r − r_p|/δ − (t_i − t_p)/τ )  ·  cos( 2π|r − r_p|/λ_F + φ )
```

sum over past bounce points `r_p` at times `t_p`; `δ` spatial decay, `τ` memory time (`τ ∝ |γ − γ_F|^(-1)` near Faraday instability), `λ_F` Faraday wavelength. With this kernel + free-flight + Coriolis, droplets reproduce a catalog of **standing-wave (orbit) quantization laws**:

| Phenomenon | Quantization law | QM analog |
| --- | --- | --- |
| Cyclotron orbits (rotating bath) | `2R_n ≈ (n + 1/2) λ_F` | Landau levels |
| Two-walker orbits | `d_n = (n − ε₀) λ_F` (ε₀≈0.2) | bound-state quantization |
| Level splitting in rotating cell | `δ_d = C·Ω` (cc vs clockwise split) | Zeeman effect |
| Droplet in harmonic trap (ferrofluid) | double quantization `(n, m)` — radius + angular momentum; Cassini-oval orbits | atomic `(n, ℓ, m)` |
| Lattice of corrals | spontaneous **antiferromagnetic** (Néel) order | spin lattice |
| Circular corral statistics | trajectory histogram = Faraday-mode maxima | quantum corral (STM) |

**Relevance:** this is the concrete realization of the **standing-wave / orbit-quantization** half that Jeff Yee flagged is needed *alongside* topology (the M3-in-M5 retention). If we ever build an orbit-quantization sub-experiment, this kernel is the reference model — and these laws are ready-made validation targets. It is the "Couder side" that complements M5's "Duda topology side."

### 11b.4 MERW — classical max-entropy paths → Born rule ρ = |ψ|² (statistics bridge)

Duda's own **Maximal Entropy Random Walk**: a time-symmetric (Boltzmann) path ensemble whose stationary density is the **QM ground state** with Anderson-like localization. The Born rule emerges as `ρ ∝ ψ²` from combining past (`ρ ∝ ψ`) and future (`ρ ∝ ψ`) trajectory ensembles — verified against STM electron-density maps of Ga₁₋ₓMnₓAs.

**Relevance — two M5 touch-points:** (a) it is Duda's answer to the *"`ρ = |ψ|²` statistics"* question — the classical→Born-rule bridge that complements the topology + clock bridges; (b) it is a natural framing for the per-defect `(A, ω)` ensemble → macroscopic distribution (a max-entropy path-ensemble problem). Candidate tool, not yet on the roadmap.

### 11b.5 The "two-ingredient Schrödinger" framing (conceptual scaffold)

The deck's clean articulation of the classical → QM bridge: **Schrödinger's equation = (1)** a coupled standing wave `ψ₀ e^{iEt/ℏ}` *resonant with the electron's clock* **+ (2)** `ρ = |ψ|²` statistics (MERW). Stable field configurations = solitons (topological); varying particle number = QFT "algebra for particles" / ensemble of Feynman scenarios. **Relevance:** situates M5's three pillars cleanly — the clock (M5.8) supplies ingredient (1), MERW supplies ingredient (2), topology supplies the stable configurations. A good framing paragraph for the M5 overview / any outreach.

### 11b.6 Already covered (no action) + one speculative aside

| Deck content | Where we already have it |
| --- | --- |
| de Broglie clock via electron channeling (Catillon 2008, 81 MeV, ~10²¹ Hz) | M5.8 experimental anchors (`0b_M5_roadmap`) |
| Faber Coulomb Mathematica code, `V(d) ≈ 1589.56 − 25.16/d` | §11 of this doc / M5.4 cross-validation target |
| Topology charge = covering degree / Gauss-Bonnet `∮K dS = 2πχ(S)` | `1a`, `1b` |
| Couder orbit quantization as the droplet analog | `1a` (extensively) |
| Zitterbewegung in BEC (Qu 2013, LeBlanc 2013), Dirac `x_k(t)` ZB term | M5.8 anchors |

Speculative aside (low priority, noted not adopted): the deck floats electron substructure — Dehmelt's Penning-trap bound `R < 10⁻²² m`, the Brodsky–Drell "three heavier fermions" idea, and an `e⁺e⁻` cross-section extrapolation to rest giving `r ≈ 2 fm`. Interesting context for "how big is the electron defect" but not a current M5 target.

### 11b.7 Net adds for the plan

| New item | Informs | Action |
| --- | --- | --- |
| Hydro↔EM dictionary (§11b.1) | M5.5 | cross-check route for EM-from-tilts |
| Faber quantized-EM Lagrangian (§11b.2) | M5.5 | port alongside LdG potential |
| Path-memory kernel + quantization laws (§11b.3) | M3-in-M5 orbit quantization | reference model + validation targets |
| MERW → Born rule (§11b.4) | `ρ=\|ψ\|²` question; (A,ω) ensemble statistics | candidate tool |
| Two-ingredient Schrödinger framing (§11b.5) | M5 overview / outreach | framing paragraph |

---

## 12. Open questions & implications

### What Duda still left open

| Open | Status |
| --- | --- |
| Exact V(M) form | "most difficult" — could be LdG or slightly different |
| Faber regularization specifics | port + adapt, but exact form is open |
| Deeper substrate beneath M | hinted at (~anisotropic fluid) |
| Weak force clean SU(2) mechanism | gap; topology reconnection is partial |

### Memory entries to write (topic-by-topic, later)

| Entry | Captures |
| --- | --- |
| `reference_duda_slides` | 51-page slides at ../theory/, list of ported targets |
| `feedback_eigenvalue_force_map` | 1=EM, δ=QM, g=gravity, 3D leptons from axis choice |
| `reference_kg_from_hedgehog_formula` | closed-form `2∂_tt ψ = …` from slide page 32 |
| `project_duda_thread_2026_05_14_15` | substrate gate closed, three directions, reciprocal ask |
| `feedback_force_unification_4_mechanisms` | tilts=EM, boost=gravity, vortex=strong, reconnect=weak |

### Updates to existing memories (the ones that need refinement)

| Existing memory | What needs to change |
| --- | --- |
| `reference_duda_lcb_paper` | eigenvalue → force-scale mapping (was framed as lepton-mass scale); add slides as companion source |
| `m5-path` (project_m5_path) | substrate decision locked → matrix M (full, not Q-tensor) |
| `(A, ω)` excess | the matrix substrate is the explicit field-theoretic home for the per-defect excitation |
