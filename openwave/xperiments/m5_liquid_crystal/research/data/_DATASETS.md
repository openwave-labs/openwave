# Local-only datasets manifest

> AUTO-GENERATED, do not hand-edit the table: `python3 dev_docs/utils/gen_datasets_manifest.py openwave/xperiments/m5_liquid_crystal/research/data --write`

Heavy binary arrays in this folder are **local-only**: gitignored, never deleted (policy 2026-07-20, which supersedes the earlier "delete raw data > 1 MB" rule). They stay on the working machine so later tasks can consume them directly, and they stay OUT of the repo so clones stay light. What IS tracked in git and readable on GitHub: the summary `.json` / `.csv` / `.txt` in this same folder, the plots, and the scripts that rebuild everything here.

**Inventory**: 210 local-only files, 2275.48 MB, in 12 task groups.

| Task group | Files | Size | Producing script(s) | Record (regen commands + context) |
| --- | --- | --- | --- | --- |
| `m5_21_10` | 3 | 293.47 MB | [`m5_21_10_a_decay64.py`](../scripts/m5_21_10_a_decay64.py) · [`m5_21_10_b_ring.py`](../scripts/m5_21_10_b_ring.py) · [`m5_21_10_c_panel.py`](../scripts/m5_21_10_c_panel.py) (+1 more) | [`m5_21_10_task_details.md`](../tasks/m5_21_10_task_details.md) |
| `m5_21_11` | 42 | 245.85 MB | [`m5_21_11_a_timing.py`](../scripts/m5_21_11_a_timing.py) · [`m5_21_11_b_ladder.py`](../scripts/m5_21_11_b_ladder.py) · [`m5_21_11_c_readers.py`](../scripts/m5_21_11_c_readers.py) (+6 more) | [`m5_21_11_task_details.md`](../tasks/m5_21_11_task_details.md) |
| `m5_21_2b` | 2 | 8.05 MB | [`m5_21_2b_a_instrument.py`](../scripts/m5_21_2b_a_instrument.py) · [`m5_21_2b_audit_check.py`](../scripts/m5_21_2b_audit_check.py) · [`m5_21_2b_b_split.py`](../scripts/m5_21_2b_b_split.py) (+2 more) | [`m5_21_2b_task_details.md`](../tasks/m5_21_2b_task_details.md) |
| `m5_21_4` | 20 | 67.54 MB | [`m5_21_4_a_pair.py`](../scripts/m5_21_4_a_pair.py) · [`m5_21_4_audit_check.py`](../scripts/m5_21_4_audit_check.py) · [`m5_21_4_c_films.py`](../scripts/m5_21_4_c_films.py) (+2 more) | [`m5_21_4_task_details.md`](../tasks/m5_21_4_task_details.md) |
| `m5_21_5` | 3 | 3.13 MB | [`m5_21_5_a_mu.py`](../scripts/m5_21_5_a_mu.py) · [`m5_21_5_b_ladder.py`](../scripts/m5_21_5_b_ladder.py) · [`m5_21_5_c_bridge.py`](../scripts/m5_21_5_c_bridge.py) (+2 more) | [`m5_21_5_task_details.md`](../tasks/m5_21_5_task_details.md) |
| `m5_21_6` | 6 | 166.69 MB | [`m5_21_6_a_decay.py`](../scripts/m5_21_6_a_decay.py) · [`m5_21_6_audit_check.py`](../scripts/m5_21_6_audit_check.py) · [`m5_21_6_c_films.py`](../scripts/m5_21_6_c_films.py) (+1 more) | [`m5_21_6_task_details.md`](../tasks/m5_21_6_task_details.md) |
| `m5_21_9` | 8 | 7.18 MB | [`m5_21_9_a_audit.py`](../scripts/m5_21_9_a_audit.py) · [`m5_21_9_a_negdelta.py`](../scripts/m5_21_9_a_negdelta.py) · [`m5_21_9_b_audit.py`](../scripts/m5_21_9_b_audit.py) (+6 more) | [`m5_21_9_task_details.md`](../tasks/m5_21_9_task_details.md) |
| `m5_22` | 40 | 77.59 MB | [`m5_22_1_a_kick.py`](../scripts/m5_22_1_a_kick.py) · [`m5_22_1_b_deuteron.py`](../scripts/m5_22_1_b_deuteron.py) · [`m5_22_1_c_panels.py`](../scripts/m5_22_1_c_panels.py) (+14 more) | [`m5_22_task_details.md`](../tasks/m5_22_task_details.md) |
| `m5_22_1` | 12 | 22.35 MB | [`m5_22_1_a_kick.py`](../scripts/m5_22_1_a_kick.py) · [`m5_22_1_b_deuteron.py`](../scripts/m5_22_1_b_deuteron.py) · [`m5_22_1_c_panels.py`](../scripts/m5_22_1_c_panels.py) (+1 more) | [`m5_22_1_task_details.md`](../tasks/m5_22_1_task_details.md) |
| `m5_22_2` | 20 | 25.96 MB | [`m5_22_2_a_dive.py`](../scripts/m5_22_2_a_dive.py) · [`m5_22_2_b_decay.py`](../scripts/m5_22_2_b_decay.py) · [`m5_22_2_c_panels.py`](../scripts/m5_22_2_c_panels.py) (+1 more) | [`m5_22_2_task_details.md`](../tasks/m5_22_2_task_details.md) |
| `m5_22_4` | 9 | 7.62 MB | [`m5_22_4_a_fullf.py`](../scripts/m5_22_4_a_fullf.py) · [`m5_22_4_b_omega.py`](../scripts/m5_22_4_b_omega.py) · [`m5_22_4_c_panels.py`](../scripts/m5_22_4_c_panels.py) (+1 more) | [`m5_22_4_task_details.md`](../tasks/m5_22_4_task_details.md) |
| `m5_39` | 45 | 1350.05 MB | [`m5_39_cardc.py`](../scripts/m5_39_cardc.py) · [`m5_39_forecast_pilot.py`](../scripts/m5_39_forecast_pilot.py) · [`m5_39_plots.py`](../scripts/m5_39_plots.py) (+8 more) | [`m5_39_task_details.md`](../tasks/m5_39_task_details.md) |

**Regeneration**: the exact command + runtime per dataset lives in the task record linked on its row (the task_details / findings doc), which is where the run configuration is already written down. Runs are deterministic from their fixed seeds and configs, so a regenerated array reproduces the original bit-for-bit at the stored precision.
