# Sandbox v8 + vn index-0 flip (the second-wave flip: bring the M5.8 body onto index-0)

Decision (Rodrigo, 2026-06-22): the 2026-06-21 refactor flipped the engine + active sandboxes (v9, v10) to
index-0 and FROZE v1-v8 + vn as index-3 history. On review, the M5.8 4D body (v8 + vn) is treated as LIVING,
not history, so it is flipped to index-0 to match the engine. v1-v5 (pre-4×4) and v7 (3×3 self-contained) and
the 3×3 files of v6 stay as-is; v6's one 4×4 engine-boundary file (`m5_6_5c_prod_scale`) is fixed too.

Canonical convention: [`CONVENTION.md`](CONVENTION.md). First-wave plan: [`00_plan.md`](00_plan.md).

## The flip rule (matrix-eigenvalue index k -> (k+1) mod 4)

| index-3 (was) | index-0 (now) | meaning |
| --- | --- | --- |
| `D = diag(1, δ, 0, g)` | `D = diag(g, 1, δ, 0)` | eigenvalue order; g (time) to index 0 |
| spatial-eigen matrix axes {0,1,2} | {1,2,3} | the 3 nematic eigen-axes |
| time/g matrix axis 3 | 0 | gets the η minus |
| `eta = diag(1,1,1,-1)` | `diag(-1,1,1,1)` | Minkowski signature |
| slice `[...,:3,:3]` (spatial) | `[...,1:4,1:4]` | spatial block |
| `[...,3,3]` (g) | `[...,0,0]` | time/g entry |
| `[...,a,3]`,`[...,3,a]` (boost) | `[...,a,0]`,`[...,0,a]` | a in {1,2,3} |
| eigen-plane `(1,2)`=(δ,0) | `(2,3)` | clock plane |
| boost axis `a` in {0,1,2} | {1,2,3} | a -> a+1 |

**THE DE-CONFLATION (critical, the trap).** Index-3 code reuses ONE constant for two roles because both are
`{0,1,2}` there: the spatial-GRADIENT axis pairs (∂_x,∂_y,∂_z, indexing the gradient list `Mi`) and the
spatial-EIGEN MATRIX αβ pairs. Under index-0 they desync: gradient pairs STAY `{0,1,2}`; matrix-eigen pairs go
to `{1,2,3}`. Each file must split these: keep a `GRAD_PAIRS = [(0,1),(0,2),(1,2)]` for the gradient role and
flip `SP_PAIRS`/`TM_PAIRS` for the matrix role. (Spatial coordinate derivatives `central(M, ax, h)`,
`range(3)` over x,y,z, are NOT matrix indices , they never change.)

## Verification = the sandboxes' own PASS/FAIL gates (their built-in golden master)

Each M5.8 script prints physics GATES (`C_neg=0 exactly`, `fuel localized`, `A(b)>0`, `g-factor≈2`, `PASS`).
The flip is a pure relabel, so EVERY gate + every physics number must be unchanged. A flip error breaks a gate.
**Gate: after flipping, the script PASSes with identical numbers (only relabeled plane/axis IDENTIFIERS may
differ; the eigenvalue-based labels like `(1,0)-plane` are invariant).** Proven on the root below.

## Ordering hazard

Downstream files import index-carrying symbols (`D4`, `boost_field`, `rot4`, `gen4`, `conj`, `SP_PAIRS`,
`TM_PAIRS`) from `m5_8_2a`. Flipping `2a` half-breaks importers until THEY are flipped (their CALLS must pass
the new boost axis a in {1,2,3} and plane `(2,3)`). So flip strictly roots -> importers -> leaves; verify each
tier's gates before the next.

## Dependency-ordered ledger

| Tier | File | 4D? | Status |
| --- | --- | --- | --- |
| 0 root | `sandbox_v8/m5_8_2a_4d_hamiltonian` | yes | ✅ FLIPPED + VERIFIED (gates identical, numbers bit-identical; only a 1e-26 vacuum-control noise wobble) |
| 1 | `sandbox_v8/m5_8_2c1_full_evolution` (imports D4/rot4/gen4/boost_field/conj) | yes | 🚧 |
| 1 | `sandbox_v8/m5_8_2b_cc_clock`, `m5_8_2b2_field_clock`, `m5_8_2c0_von_check` (import D4/SP/TM/...) | yes | 🚧 |
| 1 | `sandbox_v8/m5_8_2f_breathing_bvp` | yes | 🚧 |
| 2 | `sandbox_v8/m5_8_2d_quartic_saturation`, `m5_8_2e_invariant_matrix`, `m5_8_2g_spontaneity` (import 2c1) | yes | 🚧 |
| 2 | `sandbox_v8/m5_8_2cb_taichi_constrained` (imports 2a+2c1) | yes | 🚧 |
| 2 | `sandbox_v8/m5_8_2f2_localized_clock`, `m5_8_2f3_breather_orbit` (import 2f/2a) | yes | 🚧 |
| 3 boundary | `sandbox_v8/m5_8_1_headless_check` (engine + index-3 self-slice) | yes | 🚧 |
| 3 boundary | `sandbox_v8/m5_8_2cb2_signed_gui_repro` (engine + `[:3,:3]`,`[:3,3]`) | yes | 🚧 |
| 3 boundary | `sandbox_v8/m5_8_1_topo_charge`, `m5_8_2c2_gui_repro` (engine drivers, confirm) | yes | 🚧 |
| 3 boundary | `sandbox_v6/m5_6_5c_prod_scale` (engine + `[:3,:3]` -> `[1:4,1:4]`) | yes | 🚧 |
| verify | `sandbox_v8/m5_8_1_4x4_promotion` (already index-0) | yes | 🚧 confirm only |
| skip | `sandbox_v8/m5_8_c_isotropy_gate` (3×3), `m5_8_0b_*` (1D toys) | no | , |
| vn root | `sandbox_vn/m5_8_2q_delta_scaling` (own D4 + imports 2a SP/TM), `m5_8_2r_electron_id` (own D4_of) | yes | 🚧 |
| vn | `m5_8_2h_omega_attractor` (then 2i,2l,2m,2o,2p,2q_omega,2s,2u,2v,2w,2z_settled,3a) | yes | 🚧 |
| vn skip | `2j,2k,2n,2x,2y*,2z,2za,m5_9_pmns_so3,m5_9b_theta13,sine_gordon` (no 4D) | no | , |

~31 files to flip. Gate-verify each tier.

## ✅ STATUS: COMPLETE (2026-06-22)

All v8 + vn 4D files flipped to index-0 + verified physics-neutral (each script's own PASS/FAIL gates + golden
numbers reproduced; only relabeled plane/axis identifiers and sub-1e-25 reorder noise differ). `FROZEN_SANDBOXES.md`
+ `CONVENTION.md` updated.

| Group | Files | Result |
| --- | --- | --- |
| v8 root + chain | `2a`, `2c1`, `2b`, `2b2`, `2c0`, `2f`, `2d`, `2cb`, `2f2`, `2f3`, `2e` | flipped + gates PASS / golden numbers bit-identical (`2a` C=-678.05; `2c1` machine-precision; `2f` F1a -48.8408; `2e` H 66.6→57.9; `2d` onset 1300, align 0.889 = documented) |
| v8 consumers (no edit) | `2g`, `1_topo_charge`, `2c2_gui_repro` | pure engine/kernel consumers, run clean index-0 |
| v8 boundary (engine-importers, the real bug) | `1_headless_check`, `2cb2_signed_gui_repro` | 8 failing checks now PASS; `M[0,0]==g`, spatial spectrum clean of g |
| v6 boundary | `m5_6_5c_prod_scale` | `[:3,:3]->[1:4,1:4]`: spatial spectrum (1,δ,0), g no longer leaks (Tr² 64→1.08) |
| v8 already-index-0 | `1_4x4_promotion` | verified PASS (was written index-0) |
| vn roots | `2q_delta_scaling`, `2r_electron_id` | D4 flip + planes + `W[...,:3,0]->[...,1:4,1]`; gates bit-identical |
| vn own-site | `2s`, `2i`, `2l`, `2m`(no own), `2p` | `2s` spin-1/2 machine-exact; `2i` align 0.962; `2l` c₀=65.09; `2p` G4 generators, pairing 1.000000 |
| vn consumers (no edit) | `2h`, `2q_omega`, `2o`, `2u`, `2v`, `2w`, `2z_settled`, `3a` | run clean; `H_static=16.7379` reproduced across `2u`/`2v`/`3a`; `2o` ω band 1.09-1.60 |

### Data-regen notes (caches, NOT code)

- Stale index-3 **raw-M seed variants** `_m5_8_2cb_ref_{N48,RC1.1,RC1.4,RW2.5,RW4.5}.npz` were DELETED (git-ignored;
  auto-regenerate index-0 on demand via `M58_RW=.. / M58_N=.. python m5_8_2cb_taichi_constrained.py ref`). The
  primary `_m5_8_2cb_ref.npz` was regenerated index-0 (g at diag idx 0).
- Stale index-3 **saved traces** that store DERIVED SCALARS (`data/_m5_8_2h_dense.npz`, `_m5_8_2o_*.npz`,
  `_m5_8_2i_gate_N*.npz`) are relabel-INVARIANT (scalar invariants unchanged), so `analyze` modes still read them
  correctly; regenerate only if a fresh full run is wanted (expensive 48000-step evolutions, NOT done here).
- Missing prerequisite `_m5_8_2g_settled.npz` (absent, pre-existing) blocks `2h run` mode; regenerate from `2g`'s
  settle protocol when needed.
- Fresh index-0 table caches left in place (regenerable): `_m5_8_2f_tables.npz` (1.4M), `_m5_8_2f2_tables.npz`
  (5.5M), `_m5_8_2f3_kin.npz` (1.6M).
