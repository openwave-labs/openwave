# Local-only datasets manifest

> AUTO-GENERATED, do not hand-edit the table: `python3 dev_docs/utils/gen_datasets_manifest.py openwave/xperiments/m5_liquid_crystal/research/data/m5_32_r31_s1 --write`

Heavy binary arrays in this folder are **local-only**: gitignored, never deleted (policy 2026-07-20, which supersedes the earlier "delete raw data > 1 MB" rule). They stay on the working machine so later tasks can consume them directly, and they stay OUT of the repo so clones stay light. What IS tracked in git and readable on GitHub: the summary `.json` / `.csv` / `.txt` in this same folder, the plots, and the scripts that rebuild everything here.

**Inventory**: 642 local-only files, 383.08 MB, in 1 task groups.

| Task group | Files | Size | Producing script(s) | Record (regen commands + context) |
| --- | --- | --- | --- | --- |
| `(unprefixed)` | 642 | 383.08 MB | ⚠️ not resolved | ⚠️ not resolved |

**Regeneration**: the exact command + runtime per dataset lives in the task record linked on its row (the task_details / findings doc), which is where the run configuration is already written down. Runs are deterministic from their fixed seeds and configs, so a regenerated array reproduces the original bit-for-bit at the stored precision.
