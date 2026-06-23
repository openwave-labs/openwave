# Frozen sandboxes (index-3 era, pre 2026-06-21 flip)

These sandbox directories are HISTORICAL RECORDS of experiments run against the legacy index-3
engine (`D = diag(1, δ, 0, g)`, time axis = array index 3). They were **deliberately NOT edited**
by the 2026-06-21 index-0 flip , editing them would falsify the record of what actually ran
(their results were produced by index-3 code). Perfect provenance: byte-identical to when they ran.

| Frozen dir | Era |
| --- | --- |
| `sandbox_v1` .. `sandbox_v5` | early 3×3 / matrix-feasibility models (pre 4×4 engine) |
| `sandbox_v6` (3×3 files only) | the biaxial-hedgehog seeders etc.; `m5_6_5c_prod_scale` was FLIPPED (engine-boundary, 2026-06-22) |
| `sandbox_v7` | 3×3 self-contained resonance/oscillation, no engine import |

> ⚠️ **`sandbox_v8` + `sandbox_vn` are NO LONGER frozen.** On 2026-06-22 the whole M5.8 4D body (v8 + vn) was
> flipped to index-0 (the second-wave flip), treating it as living rather than history. See
> [`01_sandbox_v8_vn_flip.md`](01_sandbox_v8_vn_flip.md) and the "NOT frozen" section below.

## How to re-run a frozen sandbox faithfully

These scripts import the shared engine, which is now index-0. To run them with the index-3 engine
they were written against, check out the repository at the **last index-3-engine commit** (the
commit immediately BEFORE the convention flip; the flip commit message references
`research/_convention_refactor/`). As of this writing the flip is an uncommitted working-tree
change, so `HEAD` (`6a33d8c`) still carries the index-3 engine:

```text
git stash            # set aside the index-0 flip (if uncommitted)
#   or: git checkout <pre-flip-commit>
python3 research/sandbox_v8/<script>.py
git stash pop        # restore the flip
```

The physics is identical either way (the flip is proven physics-neutral, see
[`CONVENTION.md`](CONVENTION.md) + [`golden_master.py`](golden_master.py)); re-running against the
index-3 commit only reproduces the exact stored arithmetic/labels of the original run.

## NOT frozen (flipped to index-0, active)

`sandbox_v9` (#200 lepton mass) and `sandbox_v10` (neutrino #236) were flipped with the engine on 2026-06-21.
`sandbox_v8` + `sandbox_vn` (the full M5.8 4D body) + `sandbox_v6/m5_6_5c_prod_scale` were flipped on 2026-06-22
(second wave, golden-master-gated by each script's own physics gates: every gate + golden number reproduced).
See [`CONVENTION.md`](CONVENTION.md) and [`01_sandbox_v8_vn_flip.md`](01_sandbox_v8_vn_flip.md).
