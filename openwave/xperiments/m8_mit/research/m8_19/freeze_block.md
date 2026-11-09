# M8.19: the freeze block

## The terms

- The filed terms, `run/m8_19_task_details.md`: SHA-256 `f2295e0b7f1d086ab46db4c0ad972b11b8b114ed08a2297d97166dbef3c5cff7`.
- The go: https://github.com/openwave-labs/openwave/pull/698#issuecomment-6749289423
- `run/FROZEN.json` written 2026-11-09T19:53:23Z.

## The folder

- MIT `94a679acc0c8ae9b7e002a063f0c31da8f82d538`, `files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/`.
- Its `SHA256SUMS`: 183 files, SHA-256 `1e8f8d0bd56d7ea584362553b1dfe5cc230c7acd8e92e8d6fa49130d09915366`.
- The check, `shasum -a 256 -c SHA256SUMS`, on the copy of the folder the run uses, at 2026-11-09T19:52:58Z, before the freeze: 183 of 183 OK, exit 0.
- The check again, at 2026-11-09T20:24:49Z, after the freeze: 183 of 183 OK, exit 0. Beside the files `SHA256SUMS` lists and `SHA256SUMS` itself, the copy holds only the three the freeze wrote: `run/ENVIRONMENT.json`, `run/FROZEN.json` and `run/FROZEN_PIN`.
- The freeze pin, `run/FROZEN_PIN`: `1e5107d0c6117a7d45c378827b8ee6670c1ce8964b1099992be05e4769486393`.
- The pins of `FROZEN.json`: 16 files, the 6 under `run/` and the 9 under `reference/` that `SHA256SUMS` lists, each at its line there, and `run/ENVIRONMENT.json`.

| file | SHA-256 |
| --- | --- |
| `reference/SOURCES.json` | `d570dec5391c32c17c9b052aaffbf3f0869441847b284697388650a2995bc346` |
| `reference/m8_16_1/run/out/C2a.json` | `0165455b6ba66d462888521a5273fc8f0a744d4430c808e2855f5e9d06d3c24f` |
| `reference/m8_16_1/run/out/C2a_R5.json` | `387ed7b8e04030a2e10569eedb313ca5713bea92bec5a6bac7bd2e6697e668be` |
| `reference/m8_16_1/run/out/RUNLOG.json` | `84f2c9f6667e02d030dfe1d0f759ad183cd0bc6293228bd60e3aa0a2785edadb` |
| `reference/m8_17/run/out/RUNLOG.json` | `7d761506f8bfb05cee50ddc6cbf29b0a97fb20c7d9242d57c6765cff8114196a` |
| `reference/m8_17/run/out/T1_R4.json` | `06e51a090e310a2accf6bab21fdee693ab2806ff6115bb8a7f10b54aae96bcc9` |
| `reference/m8_17/run/out/T1_R5.json` | `e8ecd65f3754e6bd047e0d6c34964d59aaa44bd05bb522a71a884118926d3678` |
| `reference/m8_17/run/out/T4_R4.json` | `2955bc02cb2c653deb28f13bcdf53229b21541521681900661183a632823e512` |
| `reference/m8_17/run/out/T4_R5.json` | `167ddab0d696d2758581a8f7063d0d74283c13bbe2500fb60c91665e1a2ce95c` |
| `run/ENVIRONMENT.json` | `43b7d084a46860ca1700030445569780d87edf5484b1801932b08903cdad3c3f` |
| `run/m819_branch.py` | `7c0d89c1d865fb9dad25c46fc3cbb1414dc92a8a770ea7194ce982456a5349fa` |
| `run/m819_gates.py` | `25373f75ef92e93a92a1efd13edce3703e0ae890a71f08eaad9b0a2242450305` |
| `run/m819_run.py` | `d2bc185a0327e3dc81938a4d9b7585e6a0f1598f60907cb1a6e0c8742751a383` |
| `run/m8_19_task_details.md` | `f2295e0b7f1d086ab46db4c0ad972b11b8b114ed08a2297d97166dbef3c5cff7` |
| `run/record_environment.py` | `1494fb8afd942e74a646671fe16f03ad4d9b095ae62cd1be35a51b2eb3a10059` |
| `run/write_guard.py` | `c6494c56776fc4d91da69a731003d86404dd385318395a677d35479cad7ca171` |

<details><summary>The check's output</summary>

```
$ shasum -a 256 -c SHA256SUMS    # in soft-slot-dynamics-m8-19/, on the bench, 2026-11-09T19:52:58Z
DESIGN.md: OK
DEVLOG.md: OK
dev/compare_qual.py: OK
dev/compare_runs.py: OK
dev/launch_runner.py: OK
dev/m819_branch_vs_m815_branch.diff: OK
dev/runs/001_p5a/p5a.log: OK
dev/runs/001_p5a/p5a/control_1a_arm.json: OK
dev/runs/001_p5a/p5a/job.json: OK
dev/runs/002_standin_gates/gates_C2a_R4.log: OK
dev/runs/002_standin_gates/gates_C2a_R4/gates.json: OK
dev/runs/002_standin_gates/gates_C2a_R4/job.json: OK
dev/runs/002_standin_gates/gates_C2a_R4/states_L2.npz: OK
dev/runs/003_standin_gates/compare_qual.out: OK
dev/runs/003_standin_gates/gates_C2a_R4.log: OK
dev/runs/003_standin_gates/gates_C2a_R4/gates.json: OK
dev/runs/003_standin_gates/gates_C2a_R4/job.json: OK
dev/runs/003_standin_gates/gates_C2a_R4/states_L2.npz: OK
dev/runs/004_final_code.console: OK
dev/runs/004_final_code/PIN.txt: OK
dev/runs/004_final_code/RUNLOG.json: OK
dev/runs/004_final_code/c2a_dur/c2a.json: OK
dev/runs/004_final_code/c2a_dur/console.log: OK
dev/runs/004_final_code/c2a_dur/job.json: OK
dev/runs/004_final_code/c3e3/console.log: OK
dev/runs/004_final_code/c3e3/dynamics_C3_eps3_gates.json: OK
dev/runs/004_final_code/c3e3/job.json: OK
dev/runs/004_final_code/compare_dryrun.out: OK
dev/runs/004_final_code/compare_qual.out: OK
dev/runs/004_final_code/compare_qual_p5a.out: OK
dev/runs/004_final_code/gates_C2a_R4/console.log: OK
dev/runs/004_final_code/gates_C2a_R4/gates.json: OK
dev/runs/004_final_code/gates_C2a_R4/job.json: OK
dev/runs/004_final_code/gates_C2a_R4/states_L2.npz: OK
dev/runs/004_final_code/jobs.json: OK
dev/runs/004_final_code/p5a/console.log: OK
dev/runs/004_final_code/p5a/control_1a_arm.json: OK
dev/runs/004_final_code/p5a/job.json: OK
dev/runs/004_final_code/plan/PLAN.json: OK
dev/runs/004_final_code/plan/console.log: OK
dev/runs/004_final_code/plan/job.json: OK
dev/runs/004_final_code/test_C2a_R4_standin_1/console.log: OK
dev/runs/004_final_code/test_C2a_R4_standin_1/job.json: OK
dev/runs/004_final_code/test_C2a_R4_standin_1/test.json: OK
dev/runs/004_final_code/test_C2a_R4_standin_1/test.npy: OK
dev/runs/dryrun.console: OK
dev/runs/dryrun/RUNLOG.json: OK
dev/runs/dryrun/c2a_dur/c2a.json: OK
dev/runs/dryrun/c2a_dur/console.log: OK
dev/runs/dryrun/c2a_dur/job.json: OK
dev/runs/dryrun/c3e3/console.log: OK
dev/runs/dryrun/c3e3/dynamics_C3_eps3_gates.json: OK
dev/runs/dryrun/c3e3/job.json: OK
dev/runs/dryrun/controls/CONTROLS.json: OK
dev/runs/dryrun/controls/console.log: OK
dev/runs/dryrun/controls/job.json: OK
dev/runs/dryrun/dyn_C1/console.log: OK
dev/runs/dryrun/dyn_C1/dynamics_C1.json: OK
dev/runs/dryrun/dyn_C1/dynamics_C1.npy: OK
dev/runs/dryrun/dyn_C1/job.json: OK
dev/runs/dryrun/dyn_C2a/console.log: OK
dev/runs/dryrun/dyn_C2a/dynamics_C2a.json: OK
dev/runs/dryrun/dyn_C2a/dynamics_C2a.npy: OK
dev/runs/dryrun/dyn_C2a/job.json: OK
dev/runs/dryrun/dyn_C2b/console.log: OK
dev/runs/dryrun/dyn_C2b/dynamics_C2b.json: OK
dev/runs/dryrun/dyn_C2b/dynamics_C2b.npy: OK
dev/runs/dryrun/dyn_C2b/job.json: OK
dev/runs/dryrun/dyn_C3/console.log: OK
dev/runs/dryrun/dyn_C3/dynamics_C3.json: OK
dev/runs/dryrun/dyn_C3/dynamics_C3.npy: OK
dev/runs/dryrun/dyn_C3/job.json: OK
dev/runs/dryrun/gates_C2a_R4/console.log: OK
dev/runs/dryrun/gates_C2a_R4/gates.json: OK
dev/runs/dryrun/gates_C2a_R4/job.json: OK
dev/runs/dryrun/gates_C2a_R4/states_L2.npz: OK
dev/runs/dryrun/p5a/console.log: OK
dev/runs/dryrun/p5a/control_1a_arm.json: OK
dev/runs/dryrun/p5a/job.json: OK
dev/runs/dryrun/plan/PLAN.json: OK
dev/runs/dryrun/plan/console.log: OK
dev/runs/dryrun/plan/job.json: OK
dev/runs/dryrun/rate/console.log: OK
dev/runs/dryrun/rate/dynamics_C3_rate.json: OK
dev/runs/dryrun/rate/dynamics_C3_rate.npy: OK
dev/runs/dryrun/rate/job.json: OK
dev/runs/dryrun/test_C2a_R4_standin_1/console.log: OK
dev/runs/dryrun/test_C2a_R4_standin_1/job.json: OK
dev/runs/dryrun/test_C2a_R4_standin_1/test.json: OK
dev/runs/dryrun/test_C2a_R4_standin_1/test.npy: OK
dev/runs/smoke.console: OK
dev/runs/smoke/RUNLOG.json: OK
dev/runs/smoke/c2a_dur/c2a.json: OK
dev/runs/smoke/c2a_dur/console.log: OK
dev/runs/smoke/c2a_dur/job.json: OK
dev/runs/smoke/c2a_dur/test.npy: OK
dev/runs/smoke/controls/CONTROLS.json: OK
dev/runs/smoke/controls/console.log: OK
dev/runs/smoke/controls/job.json: OK
dev/runs/smoke/dyn_C1/console.log: OK
dev/runs/smoke/dyn_C1/dynamics_C1.json: OK
dev/runs/smoke/dyn_C1/dynamics_C1.npy: OK
dev/runs/smoke/dyn_C1/job.json: OK
dev/runs/smoke/dyn_C2a/console.log: OK
dev/runs/smoke/dyn_C2a/dynamics_C2a.json: OK
dev/runs/smoke/dyn_C2a/dynamics_C2a.npy: OK
dev/runs/smoke/dyn_C2a/job.json: OK
dev/runs/smoke/dyn_C2b/console.log: OK
dev/runs/smoke/dyn_C2b/dynamics_C2b.json: OK
dev/runs/smoke/dyn_C2b/dynamics_C2b.npy: OK
dev/runs/smoke/dyn_C2b/job.json: OK
dev/runs/smoke/dyn_C3/console.log: OK
dev/runs/smoke/dyn_C3/dynamics_C3.json: OK
dev/runs/smoke/dyn_C3/dynamics_C3.npy: OK
dev/runs/smoke/dyn_C3/job.json: OK
dev/runs/smoke_first.console: OK
dev/runs/smoke_first/RUNLOG.json: OK
dev/runs/smoke_first/c2a_dur/c2a.json: OK
dev/runs/smoke_first/c2a_dur/console.log: OK
dev/runs/smoke_first/c2a_dur/job.json: OK
dev/runs/smoke_first/c2a_dur/test.npy: OK
dev/runs/smoke_first/controls/CONTROLS.json: OK
dev/runs/smoke_first/controls/console.log: OK
dev/runs/smoke_first/controls/job.json: OK
dev/runs/smoke_first/dyn_C1/console.log: OK
dev/runs/smoke_first/dyn_C1/dynamics_C1.json: OK
dev/runs/smoke_first/dyn_C1/dynamics_C1.npy: OK
dev/runs/smoke_first/dyn_C1/job.json: OK
dev/runs/smoke_first/dyn_C2a/console.log: OK
dev/runs/smoke_first/dyn_C2a/dynamics_C2a.json: OK
dev/runs/smoke_first/dyn_C2a/dynamics_C2a.npy: OK
dev/runs/smoke_first/dyn_C2a/job.json: OK
dev/runs/smoke_first/dyn_C2b/console.log: OK
dev/runs/smoke_first/dyn_C2b/dynamics_C2b.json: OK
dev/runs/smoke_first/dyn_C2b/dynamics_C2b.npy: OK
dev/runs/smoke_first/dyn_C2b/job.json: OK
dev/runs/smoke_first/dyn_C3/console.log: OK
dev/runs/smoke_first/dyn_C3/dynamics_C3.json: OK
dev/runs/smoke_first/dyn_C3/dynamics_C3.npy: OK
dev/runs/smoke_first/dyn_C3/job.json: OK
dev/setup_sealed.log: OK
dev/setup_sealed.out: OK
dev/setup_sealed.py: OK
dev/smoke.log: OK
dev/smoke.out: OK
dev/smoke.py: OK
dev/smoke_first.out: OK
dev/stub_branch.py: OK
dev/test_freeze.out: OK
dev/test_freeze.py: OK
dev/test_gates.out: OK
dev/test_gates.py: OK
dev/test_gates_dryrun.out: OK
dev/test_launch.out: OK
dev/test_launch.py: OK
dev/test_record.out: OK
dev/test_record.py: OK
dev/test_scope.out: OK
dev/test_scope.py: OK
reference/SOURCES.json: OK
reference/m8_16_1/run/out/C2a.json: OK
reference/m8_16_1/run/out/C2a_R5.json: OK
reference/m8_16_1/run/out/RUNLOG.json: OK
reference/m8_17/run/out/RUNLOG.json: OK
reference/m8_17/run/out/T1_R4.json: OK
reference/m8_17/run/out/T1_R5.json: OK
reference/m8_17/run/out/T4_R4.json: OK
reference/m8_17/run/out/T4_R5.json: OK
registration/m8_19_task_details.md: OK
registration/m8_19_v1_to_v2.diff: OK
registration/m8_19_v2_to_v3.diff: OK
registration/m8_19_v3_to_v4.diff: OK
registration/m8_19_v4_to_v5.diff: OK
registration/m8_19_v5_to_filed.diff: OK
registration/tools/periods.out: OK
registration/tools/periods.py: OK
run/m819_branch.py: OK
run/m819_gates.py: OK
run/m819_run.py: OK
run/m8_19_task_details.md: OK
run/record_environment.py: OK
run/write_guard.py: OK
tools/copy_from_mit.py: OK
exit 0
```

</details>

<details><summary>The check again</summary>

```
$ shasum -a 256 -c SHA256SUMS    # in soft-slot-dynamics-m8-19/, on the bench, 2026-11-09T20:24:49Z
DESIGN.md: OK
DEVLOG.md: OK
dev/compare_qual.py: OK
dev/compare_runs.py: OK
dev/launch_runner.py: OK
dev/m819_branch_vs_m815_branch.diff: OK
dev/runs/001_p5a/p5a.log: OK
dev/runs/001_p5a/p5a/control_1a_arm.json: OK
dev/runs/001_p5a/p5a/job.json: OK
dev/runs/002_standin_gates/gates_C2a_R4.log: OK
dev/runs/002_standin_gates/gates_C2a_R4/gates.json: OK
dev/runs/002_standin_gates/gates_C2a_R4/job.json: OK
dev/runs/002_standin_gates/gates_C2a_R4/states_L2.npz: OK
dev/runs/003_standin_gates/compare_qual.out: OK
dev/runs/003_standin_gates/gates_C2a_R4.log: OK
dev/runs/003_standin_gates/gates_C2a_R4/gates.json: OK
dev/runs/003_standin_gates/gates_C2a_R4/job.json: OK
dev/runs/003_standin_gates/gates_C2a_R4/states_L2.npz: OK
dev/runs/004_final_code.console: OK
dev/runs/004_final_code/PIN.txt: OK
dev/runs/004_final_code/RUNLOG.json: OK
dev/runs/004_final_code/c2a_dur/c2a.json: OK
dev/runs/004_final_code/c2a_dur/console.log: OK
dev/runs/004_final_code/c2a_dur/job.json: OK
dev/runs/004_final_code/c3e3/console.log: OK
dev/runs/004_final_code/c3e3/dynamics_C3_eps3_gates.json: OK
dev/runs/004_final_code/c3e3/job.json: OK
dev/runs/004_final_code/compare_dryrun.out: OK
dev/runs/004_final_code/compare_qual.out: OK
dev/runs/004_final_code/compare_qual_p5a.out: OK
dev/runs/004_final_code/gates_C2a_R4/console.log: OK
dev/runs/004_final_code/gates_C2a_R4/gates.json: OK
dev/runs/004_final_code/gates_C2a_R4/job.json: OK
dev/runs/004_final_code/gates_C2a_R4/states_L2.npz: OK
dev/runs/004_final_code/jobs.json: OK
dev/runs/004_final_code/p5a/console.log: OK
dev/runs/004_final_code/p5a/control_1a_arm.json: OK
dev/runs/004_final_code/p5a/job.json: OK
dev/runs/004_final_code/plan/PLAN.json: OK
dev/runs/004_final_code/plan/console.log: OK
dev/runs/004_final_code/plan/job.json: OK
dev/runs/004_final_code/test_C2a_R4_standin_1/console.log: OK
dev/runs/004_final_code/test_C2a_R4_standin_1/job.json: OK
dev/runs/004_final_code/test_C2a_R4_standin_1/test.json: OK
dev/runs/004_final_code/test_C2a_R4_standin_1/test.npy: OK
dev/runs/dryrun.console: OK
dev/runs/dryrun/RUNLOG.json: OK
dev/runs/dryrun/c2a_dur/c2a.json: OK
dev/runs/dryrun/c2a_dur/console.log: OK
dev/runs/dryrun/c2a_dur/job.json: OK
dev/runs/dryrun/c3e3/console.log: OK
dev/runs/dryrun/c3e3/dynamics_C3_eps3_gates.json: OK
dev/runs/dryrun/c3e3/job.json: OK
dev/runs/dryrun/controls/CONTROLS.json: OK
dev/runs/dryrun/controls/console.log: OK
dev/runs/dryrun/controls/job.json: OK
dev/runs/dryrun/dyn_C1/console.log: OK
dev/runs/dryrun/dyn_C1/dynamics_C1.json: OK
dev/runs/dryrun/dyn_C1/dynamics_C1.npy: OK
dev/runs/dryrun/dyn_C1/job.json: OK
dev/runs/dryrun/dyn_C2a/console.log: OK
dev/runs/dryrun/dyn_C2a/dynamics_C2a.json: OK
dev/runs/dryrun/dyn_C2a/dynamics_C2a.npy: OK
dev/runs/dryrun/dyn_C2a/job.json: OK
dev/runs/dryrun/dyn_C2b/console.log: OK
dev/runs/dryrun/dyn_C2b/dynamics_C2b.json: OK
dev/runs/dryrun/dyn_C2b/dynamics_C2b.npy: OK
dev/runs/dryrun/dyn_C2b/job.json: OK
dev/runs/dryrun/dyn_C3/console.log: OK
dev/runs/dryrun/dyn_C3/dynamics_C3.json: OK
dev/runs/dryrun/dyn_C3/dynamics_C3.npy: OK
dev/runs/dryrun/dyn_C3/job.json: OK
dev/runs/dryrun/gates_C2a_R4/console.log: OK
dev/runs/dryrun/gates_C2a_R4/gates.json: OK
dev/runs/dryrun/gates_C2a_R4/job.json: OK
dev/runs/dryrun/gates_C2a_R4/states_L2.npz: OK
dev/runs/dryrun/p5a/console.log: OK
dev/runs/dryrun/p5a/control_1a_arm.json: OK
dev/runs/dryrun/p5a/job.json: OK
dev/runs/dryrun/plan/PLAN.json: OK
dev/runs/dryrun/plan/console.log: OK
dev/runs/dryrun/plan/job.json: OK
dev/runs/dryrun/rate/console.log: OK
dev/runs/dryrun/rate/dynamics_C3_rate.json: OK
dev/runs/dryrun/rate/dynamics_C3_rate.npy: OK
dev/runs/dryrun/rate/job.json: OK
dev/runs/dryrun/test_C2a_R4_standin_1/console.log: OK
dev/runs/dryrun/test_C2a_R4_standin_1/job.json: OK
dev/runs/dryrun/test_C2a_R4_standin_1/test.json: OK
dev/runs/dryrun/test_C2a_R4_standin_1/test.npy: OK
dev/runs/smoke.console: OK
dev/runs/smoke/RUNLOG.json: OK
dev/runs/smoke/c2a_dur/c2a.json: OK
dev/runs/smoke/c2a_dur/console.log: OK
dev/runs/smoke/c2a_dur/job.json: OK
dev/runs/smoke/c2a_dur/test.npy: OK
dev/runs/smoke/controls/CONTROLS.json: OK
dev/runs/smoke/controls/console.log: OK
dev/runs/smoke/controls/job.json: OK
dev/runs/smoke/dyn_C1/console.log: OK
dev/runs/smoke/dyn_C1/dynamics_C1.json: OK
dev/runs/smoke/dyn_C1/dynamics_C1.npy: OK
dev/runs/smoke/dyn_C1/job.json: OK
dev/runs/smoke/dyn_C2a/console.log: OK
dev/runs/smoke/dyn_C2a/dynamics_C2a.json: OK
dev/runs/smoke/dyn_C2a/dynamics_C2a.npy: OK
dev/runs/smoke/dyn_C2a/job.json: OK
dev/runs/smoke/dyn_C2b/console.log: OK
dev/runs/smoke/dyn_C2b/dynamics_C2b.json: OK
dev/runs/smoke/dyn_C2b/dynamics_C2b.npy: OK
dev/runs/smoke/dyn_C2b/job.json: OK
dev/runs/smoke/dyn_C3/console.log: OK
dev/runs/smoke/dyn_C3/dynamics_C3.json: OK
dev/runs/smoke/dyn_C3/dynamics_C3.npy: OK
dev/runs/smoke/dyn_C3/job.json: OK
dev/runs/smoke_first.console: OK
dev/runs/smoke_first/RUNLOG.json: OK
dev/runs/smoke_first/c2a_dur/c2a.json: OK
dev/runs/smoke_first/c2a_dur/console.log: OK
dev/runs/smoke_first/c2a_dur/job.json: OK
dev/runs/smoke_first/c2a_dur/test.npy: OK
dev/runs/smoke_first/controls/CONTROLS.json: OK
dev/runs/smoke_first/controls/console.log: OK
dev/runs/smoke_first/controls/job.json: OK
dev/runs/smoke_first/dyn_C1/console.log: OK
dev/runs/smoke_first/dyn_C1/dynamics_C1.json: OK
dev/runs/smoke_first/dyn_C1/dynamics_C1.npy: OK
dev/runs/smoke_first/dyn_C1/job.json: OK
dev/runs/smoke_first/dyn_C2a/console.log: OK
dev/runs/smoke_first/dyn_C2a/dynamics_C2a.json: OK
dev/runs/smoke_first/dyn_C2a/dynamics_C2a.npy: OK
dev/runs/smoke_first/dyn_C2a/job.json: OK
dev/runs/smoke_first/dyn_C2b/console.log: OK
dev/runs/smoke_first/dyn_C2b/dynamics_C2b.json: OK
dev/runs/smoke_first/dyn_C2b/dynamics_C2b.npy: OK
dev/runs/smoke_first/dyn_C2b/job.json: OK
dev/runs/smoke_first/dyn_C3/console.log: OK
dev/runs/smoke_first/dyn_C3/dynamics_C3.json: OK
dev/runs/smoke_first/dyn_C3/dynamics_C3.npy: OK
dev/runs/smoke_first/dyn_C3/job.json: OK
dev/setup_sealed.log: OK
dev/setup_sealed.out: OK
dev/setup_sealed.py: OK
dev/smoke.log: OK
dev/smoke.out: OK
dev/smoke.py: OK
dev/smoke_first.out: OK
dev/stub_branch.py: OK
dev/test_freeze.out: OK
dev/test_freeze.py: OK
dev/test_gates.out: OK
dev/test_gates.py: OK
dev/test_gates_dryrun.out: OK
dev/test_launch.out: OK
dev/test_launch.py: OK
dev/test_record.out: OK
dev/test_record.py: OK
dev/test_scope.out: OK
dev/test_scope.py: OK
reference/SOURCES.json: OK
reference/m8_16_1/run/out/C2a.json: OK
reference/m8_16_1/run/out/C2a_R5.json: OK
reference/m8_16_1/run/out/RUNLOG.json: OK
reference/m8_17/run/out/RUNLOG.json: OK
reference/m8_17/run/out/T1_R4.json: OK
reference/m8_17/run/out/T1_R5.json: OK
reference/m8_17/run/out/T4_R4.json: OK
reference/m8_17/run/out/T4_R5.json: OK
registration/m8_19_task_details.md: OK
registration/m8_19_v1_to_v2.diff: OK
registration/m8_19_v2_to_v3.diff: OK
registration/m8_19_v3_to_v4.diff: OK
registration/m8_19_v4_to_v5.diff: OK
registration/m8_19_v5_to_filed.diff: OK
registration/tools/periods.out: OK
registration/tools/periods.py: OK
run/m819_branch.py: OK
run/m819_gates.py: OK
run/m819_run.py: OK
run/m8_19_task_details.md: OK
run/record_environment.py: OK
run/write_guard.py: OK
tools/copy_from_mit.py: OK
exit 0
```

</details>

## The instrument

- The solver commit: MIT `97a8c0884c5e3a0ed25afb345f3d668340c5b537`.
- The manifest, `soft-slot-dynamics-m8-15x/m8_15x_solver/out/MANIFEST.json`: 24 rows, SHA-256 `2b9864ab185e47f98c8572500fb2468c24de9e8346aec4e1f0fb4d077fd3e70c`, each row at its file in that commit's tree. `write_guard.py` and every launch check the solver's 17 `.py` files against it, the manifest against `FROZEN.json`, and `out/freeze_numbers.json` by content against `FROZEN.json`'s `freeze_numbers`. The other 6 rows, `m8_15_math`'s 5 files and `out/orientations.json`, are read only by `m815_freeze.py`, which built the manifest; nothing in the run reads them.

| file | SHA-256 |
| --- | --- |
| `../soft-slot-dynamics/m8_15_math/analyse.py` | `971f1a755cc9b3b329552e299f2e6f09a8851121b3e975e30015ffac9242c5a0` |
| `../soft-slot-dynamics/m8_15_math/core.py` | `3507004f47d3bcc248012335e4e20e70c47cae118e9aa2a87cf85f9cc62f86b3` |
| `../soft-slot-dynamics/m8_15_math/outside3_krein.out` | `ad01a27c993793eac968ec7ccba25499f15eaf42b10ff58234ea7079313a42b1` |
| `../soft-slot-dynamics/m8_15_math/outside3_krein.py` | `94ba627cfc455d20cae34ef761f191bcfbf89e170748545d30f72d3a8b7d89e5` |
| `../soft-slot-dynamics/m8_15_math/outside3_p.npy` | `027a797c805be453c8ea0cc1ca8c9eeb4d276d7e765a7979e1daf09954e82f0d` |
| `angle.py` | `ca557b0d81f403c20b04675e3fb9b24be754f4cbe8b181498121dbd254e212b3` |
| `continuation.py` | `70c4c0354bbd33dfe627a4fbfc8b187bc0073513e56cb674670d27fb837260dd` |
| `fem.py` | `209fba67eecf8c2b6ecaa432aafda3c6a161024d83d14a060809b81ca2e1061d` |
| `geometry.py` | `788e10eb6973668e84370c13a8bd7b16dc9e712f6fa681c7386e15e9b1d1abff` |
| `integrate.py` | `3c137e773f7890478b2495d6b466dc167c2aeffc4a8d179a0b7775b9b6b58096` |
| `irreps.py` | `cbc7f4481706a08d7ca0229914980e773125c642da616e867580fe1620b5a149` |
| `linearize.py` | `7cca177bf0c8efc344cc76d2cbcee94f0a96987ae9f3c0895ae64f2f705eed58` |
| `m815_branch.py` | `7db54e748dfe238bbc21b252ec959748f9354ba681837d0fa4cbf2f0145f56db` |
| `m815_common.py` | `a66fdf091021098497a08f645c54b4cc60340203dcc8f53013c550df88dfafc6` |
| `m815_controls.py` | `dfc7a3c17fa061221d1ea74291047a01cf402d348653397a0b58cabd61b6801c` |
| `m815_freeze.py` | `6a47bf4376aadd589243408b6d925c1910c649080dd1df85075fa5135320a866` |
| `nonlinear.py` | `f00ca41c1406d123dac4384cd2f7336410be49e851426db15cad79da3102a060` |
| `out/freeze_numbers.json` | `c46b102c5ded4a78667dfb0e87e1311d729b006cb1362c6aeb91cbe1b9fe369f` |
| `out/orientations.json` | `21b1f19161058da22a074acdc12150c8650ce60c7cdc9a86bc6622fc452290a1` |
| `persistence.py` | `3d9906d8ea9ace471aec33566a5329e6d68dbf1122677f21b3690d8b9f3fb103` |
| `pinned.py` | `f6e03566b15063529ac65df7b96dddb0c1726f60f418bedac9a0e5f78b936b8f` |
| `standing.py` | `455cfc94f4bb10b97eaf0c426c86cb933e32fd7b199b00fc153bcad88fde5efb` |
| `symmetry.py` | `c6bbb178dd9103ab5b2f168c6df8cd923b4c57bd8432215b1b9d4a5bef5aae19` |
| `verdict.py` | `0dac34197abbf61edd8ede76a0c23d9c0691328ed606c032a9744397b6e29f05` |

## The environment

- Recorded 2026-11-09T19:53:09Z, and read again when this block was built.
- Differences: none.

<details><summary>As recorded</summary>

```json
{
 "interpreter": {
  "build": "3.13.13 | packaged by conda-forge | (main, Apr  8 2026, 02:29:07) [Clang 19.1.7 ]",
  "executable": "/opt/homebrew/Caskroom/miniforge/base/bin/python3",
  "flags": {
   "dont_write_bytecode": 1,
   "isolated": 1,
   "optimize": 0
  },
  "implementation": "CPython",
  "version": "3.13.13"
 },
 "machine": {
  "arch": "arm64",
  "cores": 8,
  "cpu": "Apple M3",
  "efficiency_cores": 4,
  "model": "Mac15,13",
  "performance_cores": 4
 },
 "numpy": {
  "blas": {
   "detection method": "system",
   "found": true,
   "include directory": "unknown",
   "lib directory": "unknown",
   "name": "accelerate",
   "openblas configuration": "unknown",
   "pc file directory": "unknown",
   "version": "unknown"
  },
  "lapack": {
   "detection method": "system",
   "found": true,
   "include directory": "unknown",
   "lib directory": "unknown",
   "name": "accelerate",
   "openblas configuration": "unknown",
   "pc file directory": "unknown",
   "version": "unknown"
  },
  "version": "2.5.0"
 },
 "os": {
  "build": "24H232",
  "kernel": "24.6.0",
  "product": "macOS",
  "version": "15.8.2"
 },
 "recorded": "2026-11-09T19:53:09Z",
 "scipy": {
  "blas": {
   "cython blas ilp64": false,
   "detection method": "extraframeworks",
   "found": true,
   "has ilp64": false,
   "include directory": "unknown",
   "lib directory": "unknown",
   "name": "Accelerate",
   "openblas configuration": "unknown",
   "pc file directory": "unknown",
   "version": "unknown"
  },
  "lapack": {
   "detection method": "extraframeworks",
   "found": true,
   "has ilp64": false,
   "include directory": "unknown",
   "lib directory": "unknown",
   "name": "Accelerate",
   "openblas configuration": "unknown",
   "pc file directory": "unknown",
   "version": "unknown"
  },
  "version": "1.18.0"
 },
 "threading": {
  "MKL_NUM_THREADS": null,
  "OMP_NUM_THREADS": null,
  "OPENBLAS_NUM_THREADS": null,
  "VECLIB_MAXIMUM_THREADS": null
 }
}
```

</details>

<details><summary>Read again</summary>

```json
{
 "interpreter": {
  "build": "3.13.13 | packaged by conda-forge | (main, Apr  8 2026, 02:29:07) [Clang 19.1.7 ]",
  "executable": "/opt/homebrew/Caskroom/miniforge/base/bin/python3",
  "flags": {
   "dont_write_bytecode": 1,
   "isolated": 1,
   "optimize": 0
  },
  "implementation": "CPython",
  "version": "3.13.13"
 },
 "machine": {
  "arch": "arm64",
  "cores": 8,
  "cpu": "Apple M3",
  "efficiency_cores": 4,
  "model": "Mac15,13",
  "performance_cores": 4
 },
 "numpy": {
  "blas": {
   "detection method": "system",
   "found": true,
   "include directory": "unknown",
   "lib directory": "unknown",
   "name": "accelerate",
   "openblas configuration": "unknown",
   "pc file directory": "unknown",
   "version": "unknown"
  },
  "lapack": {
   "detection method": "system",
   "found": true,
   "include directory": "unknown",
   "lib directory": "unknown",
   "name": "accelerate",
   "openblas configuration": "unknown",
   "pc file directory": "unknown",
   "version": "unknown"
  },
  "version": "2.5.0"
 },
 "os": {
  "build": "24H232",
  "kernel": "24.6.0",
  "product": "macOS",
  "version": "15.8.2"
 },
 "scipy": {
  "blas": {
   "cython blas ilp64": false,
   "detection method": "extraframeworks",
   "found": true,
   "has ilp64": false,
   "include directory": "unknown",
   "lib directory": "unknown",
   "name": "Accelerate",
   "openblas configuration": "unknown",
   "pc file directory": "unknown",
   "version": "unknown"
  },
  "lapack": {
   "detection method": "extraframeworks",
   "found": true,
   "has ilp64": false,
   "include directory": "unknown",
   "lib directory": "unknown",
   "name": "Accelerate",
   "openblas configuration": "unknown",
   "pc file directory": "unknown",
   "version": "unknown"
  },
  "version": "1.18.0"
 },
 "threading": {
  "MKL_NUM_THREADS": null,
  "OMP_NUM_THREADS": null,
  "OPENBLAS_NUM_THREADS": null,
  "VECLIB_MAXIMUM_THREADS": null
 }
}
```

</details>

## The reference files

| name | MIT commit | blob | SHA-256 |
| --- | --- | --- | --- |
| reference/m8_17/run/out/RUNLOG.json | `7b66f8a0e207888042e6bb4a9ab636f946cb2a81` | `d3e29ce7d6f5ce64b721e17b188c72db2e174b64` | `7d761506f8bfb05cee50ddc6cbf29b0a97fb20c7d9242d57c6765cff8114196a` |
| reference/m8_17/run/out/T1_R4.json | `7b66f8a0e207888042e6bb4a9ab636f946cb2a81` | `d7480d323c91df14a990aea1cfe3a7f4751d53f9` | `06e51a090e310a2accf6bab21fdee693ab2806ff6115bb8a7f10b54aae96bcc9` |
| reference/m8_17/run/out/T1_R5.json | `7b66f8a0e207888042e6bb4a9ab636f946cb2a81` | `1740687e6215cc0783defdb279db939ef3a3cf95` | `e8ecd65f3754e6bd047e0d6c34964d59aaa44bd05bb522a71a884118926d3678` |
| reference/m8_17/run/out/T4_R4.json | `7b66f8a0e207888042e6bb4a9ab636f946cb2a81` | `2e825c81b3e8dcb52daf25fbfd0f4b09a80f713c` | `2955bc02cb2c653deb28f13bcdf53229b21541521681900661183a632823e512` |
| reference/m8_17/run/out/T4_R5.json | `7b66f8a0e207888042e6bb4a9ab636f946cb2a81` | `1cb6cee89f8dc653f038e6c894336ceb9349d656` | `167ddab0d696d2758581a8f7063d0d74283c13bbe2500fb60c91665e1a2ce95c` |
| reference/m8_16_1/run/out/RUNLOG.json | `292b158cf1a53902489ced3426e1f047699aeb62` | `ccdd58fe8a4498b1d94ebf272ce2c80aaafb193e` | `84f2c9f6667e02d030dfe1d0f759ad183cd0bc6293228bd60e3aa0a2785edadb` |
| reference/m8_16_1/run/out/C2a.json | `292b158cf1a53902489ced3426e1f047699aeb62` | `91564933c228cdc5df300decaae4e62602d42b4b` | `0165455b6ba66d462888521a5273fc8f0a744d4430c808e2855f5e9d06d3c24f` |
| reference/m8_16_1/run/out/C2a_R5.json | `292b158cf1a53902489ced3426e1f047699aeb62` | `af6c5fce0d2befbbade54a375c7db89e391226a2` | `387ed7b8e04030a2e10569eedb313ca5713bea92bec5a6bac7bd2e6697e668be` |
| reference/SOURCES.json | `this tree` | `` | `d570dec5391c32c17c9b052aaffbf3f0869441847b284697388650a2995bc346` |

## `FROZEN.json`

<details><summary>In full</summary>

```json
{
 "frozen": true,
 "go": "https://github.com/openwave-labs/openwave/pull/698#issuecomment-6749289423",
 "written": "2026-11-09T19:53:23Z",
 "terms_sha256": "f2295e0b7f1d086ab46db4c0ad972b11b8b114ed08a2297d97166dbef3c5cff7",
 "solver_commit": "97a8c08",
 "instrument_manifest_sha256": "2b9864ab185e47f98c8572500fb2468c24de9e8346aec4e1f0fb4d077fd3e70c",
 "pins": {
  "reference/SOURCES.json": "d570dec5391c32c17c9b052aaffbf3f0869441847b284697388650a2995bc346",
  "reference/m8_16_1/run/out/C2a.json": "0165455b6ba66d462888521a5273fc8f0a744d4430c808e2855f5e9d06d3c24f",
  "reference/m8_16_1/run/out/C2a_R5.json": "387ed7b8e04030a2e10569eedb313ca5713bea92bec5a6bac7bd2e6697e668be",
  "reference/m8_16_1/run/out/RUNLOG.json": "84f2c9f6667e02d030dfe1d0f759ad183cd0bc6293228bd60e3aa0a2785edadb",
  "reference/m8_17/run/out/RUNLOG.json": "7d761506f8bfb05cee50ddc6cbf29b0a97fb20c7d9242d57c6765cff8114196a",
  "reference/m8_17/run/out/T1_R4.json": "06e51a090e310a2accf6bab21fdee693ab2806ff6115bb8a7f10b54aae96bcc9",
  "reference/m8_17/run/out/T1_R5.json": "e8ecd65f3754e6bd047e0d6c34964d59aaa44bd05bb522a71a884118926d3678",
  "reference/m8_17/run/out/T4_R4.json": "2955bc02cb2c653deb28f13bcdf53229b21541521681900661183a632823e512",
  "reference/m8_17/run/out/T4_R5.json": "167ddab0d696d2758581a8f7063d0d74283c13bbe2500fb60c91665e1a2ce95c",
  "run/ENVIRONMENT.json": "43b7d084a46860ca1700030445569780d87edf5484b1801932b08903cdad3c3f",
  "run/m819_branch.py": "7c0d89c1d865fb9dad25c46fc3cbb1414dc92a8a770ea7194ce982456a5349fa",
  "run/m819_gates.py": "25373f75ef92e93a92a1efd13edce3703e0ae890a71f08eaad9b0a2242450305",
  "run/m819_run.py": "d2bc185a0327e3dc81938a4d9b7585e6a0f1598f60907cb1a6e0c8742751a383",
  "run/m8_19_task_details.md": "f2295e0b7f1d086ab46db4c0ad972b11b8b114ed08a2297d97166dbef3c5cff7",
  "run/record_environment.py": "1494fb8afd942e74a646671fe16f03ad4d9b095ae62cd1be35a51b2eb3a10059",
  "run/write_guard.py": "c6494c56776fc4d91da69a731003d86404dd385318395a677d35479cad7ca171"
 },
 "freeze_numbers": {
  "r_f": {
   "2": 3.970232009242573e-05,
   "3": 1.0397370289375376e-06
  },
  "tau0": {
   "2": 0.0003970232009242573,
   "3": 1.0397370289375377e-05
  },
  "s2": {
   "R4": 0.0006739134989217632,
   "R5": 0.0009547197384733863,
   "R2": 0.002136158705290825
  },
  "d3": 0.00012465800884948396,
  "delta_lev": 0.01,
  "eta": 0.001,
  "eta_ref": 0.001,
  "T": 1000.0,
  "dynamics": {
   "C1": {
    "verdict": "Persists",
    "valid": true,
    "valid_under_v6_energy_rule": false,
    "read_until": 999.4379014555922,
    "max_d": 0.0009996684245826499,
    "max_dperp": 0.0009996684245826499,
    "max_drot": 1.581569074338934e-06,
    "max_dref": 2.372092779873853e-12,
    "charge_drift": 7.240217847948057e-13,
    "energy_drift": 1.8231164843995803e-05,
    "energy_pert": 0.00018986998862574977,
    "energy_dev_vs_pert": 0.09601920227599008,
    "energy_secular_drift": 0.0012570315907074832,
    "t_fail": null,
    "drot_at_fail": null,
    "dt": 0.008158143969827213,
    "r": 8.283304597789254e-16,
    "direction": "split"
   },
   "C2a": {
    "verdict": "Persists",
    "valid": true,
    "valid_under_v6_energy_rule": false,
    "read_until": 999.7728382088599,
    "max_d": 0.0009999042327105863,
    "max_dperp": 0.000999904235973982,
    "max_drot": 1.0646835761910693e-06,
    "max_dref": 6.594720187272686e-13,
    "charge_drift": 1.0326464897283973e-13,
    "energy_drift": 2.388746585864144e-06,
    "energy_pert": 2.477721621508679e-05,
    "energy_dev_vs_pert": 0.09640899789257365,
    "energy_secular_drift": 0.0008435075108459835,
    "t_fail": null,
    "drot_at_fail": null,
    "dt": 0.008186672656923894,
    "r": -7.025630077706069e-17,
    "direction": "split"
   },
   "C2b": {
    "verdict": "Persists",
    "valid": true,
    "valid_under_v6_energy_rule": false,
    "read_until": 999.0145486861588,
    "max_d": 0.000999836836409005,
    "max_dperp": 0.0009998368936998316,
    "max_drot": 1.78783652932435e-06,
    "max_dref": 4.671694869646264e-13,
    "charge_drift": 1.0552442141686022e-13,
    "energy_drift": 1.8971124859240158e-06,
    "energy_pert": 1.9143237478822624e-05,
    "energy_dev_vs_pert": 0.09910092208920844,
    "energy_secular_drift": 0.0011683740106362601,
    "t_fail": null,
    "drot_at_fail": null,
    "dt": 0.008196840682372199,
    "r": -2.3592239273284576e-16,
    "direction": "split"
   },
   "C3": {
    "verdict": "Fails",
    "valid": true,
    "valid_under_v6_energy_rule": false,
    "read_until": 583.2840534605136,
    "max_d": 1.1247219065340623,
    "max_dperp": 0.9040488335256167,
    "max_drot": 0.2443794006976846,
    "max_dref": 3.264796159550024e-11,
    "charge_drift": 1.0727450936121359e-14,
    "energy_drift": 2.4302567234397317e-06,
    "energy_pert": 2.477261984961753e-05,
    "energy_dev_vs_pert": 0.09810253167378472,
    "energy_secular_drift": 0.0009203174559566788,
    "t_fail": 583.2840534605136,
    "drot_at_fail": 4.008264062558979e-05,
    "dt": 0.008186672656923894,
    "r": 1.8995222061946038e-16,
    "direction": "split"
   },
   "C3_rate": {
    "fitted": 0.009740525346941262,
    "linear_L2": 0.009737451477600101,
    "window": [
     346.57460025821615,
     999.7728382088599
    ],
    "points": 655
   }
  },
  "dt": {
   "R2": {
    "omega_max": 195.19715729511552,
    "dt": 0.008196840682372157
   },
   "R3": {
    "omega_max": 196.12304047557632,
    "dt": 0.008158143969827206
   },
   "R4": {
    "omega_max": 195.43959640877944,
    "dt": 0.008186672656923915
   },
   "R5": {
    "omega_max": 195.43959640877995,
    "dt": 0.008186672656923894
   }
  },
  "gates": {
   "cluster": true,
   "linear": true,
   "dynamic": true
  },
  "schedule": {
   "T1/R4": [
    {
     "type": "i",
     "tau": 0.04351204351204351,
     "krein": -1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 0.7832167832167832,
     "krein": 1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 0.9572649572649573,
     "krein": 1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 1.5664335664335665,
     "krein": 1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 1.3053613053613053,
     "krein": 1,
     "first_resolved": 0.25
    }
   ],
   "T1/R5": [
    {
     "type": "i",
     "tau": 0.04351204351204351,
     "krein": -1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 0.7832167832167832,
     "krein": 1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 0.9572649572649573,
     "krein": 1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 1.5664335664335665,
     "krein": 1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 1.3053613053613053,
     "krein": 1,
     "first_resolved": 0.25
    }
   ],
   "T2/R4": [
    {
     "type": "i",
     "tau": 0.8538968639132244,
     "krein": 1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 0.2616912229309043,
     "krein": 1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 0.2616912229309043,
     "krein": 1,
     "first_resolved": 0.25
    }
   ],
   "T2/R5": [
    {
     "type": "i",
     "tau": 0.8538968639132244,
     "krein": 1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 0.2616912229309043,
     "krein": 1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 0.2616912229309043,
     "krein": 1,
     "first_resolved": 0.25
    }
   ],
   "T3/R4": [
    {
     "type": "i",
     "tau": 1.2123005067323727,
     "krein": -1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 0.9533011101877017,
     "krein": -1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 0.9533011101877017,
     "krein": -1,
     "first_resolved": 0.25
    }
   ],
   "T3/R5": [
    {
     "type": "i",
     "tau": 1.2123005067323727,
     "krein": -1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 0.9533011101877017,
     "krein": -1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 0.9533011101877017,
     "krein": -1,
     "first_resolved": 0.25
    }
   ],
   "T4/R4": [
    {
     "type": "i",
     "tau": 1.01209160489,
     "krein": 1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 0.310692742362,
     "krein": -1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 0.139251832964,
     "krein": -1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 0.00573685922309,
     "krein": -1,
     "first_resolved": 1.4142135623730951
    }
   ],
   "T4/R5": [
    {
     "type": "i",
     "tau": 1.01209160489,
     "krein": 1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 0.310692742362,
     "krein": -1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 0.139251832964,
     "krein": -1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 0.00573685922309,
     "krein": -1,
     "first_resolved": 2.0
    }
   ],
   "T5/R2": [
    {
     "type": "i",
     "tau": 2.7412587412587412,
     "krein": -1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 2.688571721359263,
     "krein": -1,
     "first_resolved": 0.25
    },
    {
     "type": "i",
     "tau": 3.0801801129676543,
     "krein": -1,
     "first_resolved": 0.25
    }
   ]
  }
 }
}
```

</details>

## The order

At most four jobs at once, each on one thread, in this order, a job starting as another ends:

1. The controls: `dyn_C1`, `dyn_C2a`, `dyn_C2b`, `dyn_C3`, `rate`, `p5a`, `c3e3`, `c2a_dur`; then `controls`, gate 1.
2. The gates, once gate 1 has passed: `gates_T1_R4`, `gates_T1_R5`, `gates_T4_R4`, `gates_T4_R5`, `coherent_R5`; then `plan`.
3. From the plan, `c2a_long`: C2a/R4 at 1 at the longest `T`, if a test needs more than 2750.
4. From the plan, `test_<stretch>`: each stretch's test, one longer than 2750 only after `c2a_long` has Persisted, valid.
5. The record: `record`.
