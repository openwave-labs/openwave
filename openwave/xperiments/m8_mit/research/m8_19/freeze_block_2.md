# M8.19: the second freeze block

## The ruling

- The maintainer's ruling: https://github.com/openwave-labs/openwave/pull/698#issuecomment-6755529199, at 2026-11-10T00:41:59Z: the freeze written once more, a one-time exception to § 2 of the go, https://github.com/openwave-labs/openwave/pull/698#issuecomment-6749289423.
- The first freeze: recorded by #702's block, [`freeze_block.md`](https://github.com/openwave-labs/openwave/blob/734eb28fe69c05240d237043cc91c915047483d7/openwave/xperiments/m8_mit/research/m8_19/freeze_block.md), which stays as merged; kept as below.
- `run/FROZEN.json` written once more at 2026-11-10T00:55:03Z, with the ruling's link as its `go`, as the ruling's item 4 sets it.

## The first freeze, kept

- Kept beside the folder the run uses, every byte unchanged, in `freeze/first_freeze/run/`, with the refused launch's output, `freeze/launch_1_refused.out`; published with the record in MIT's M8.19 folder.
- Its `FROZEN_PIN` holds `FROZEN.json`'s SHA-256, `1e5107d0c6117a7d45c378827b8ee6670c1ce8964b1099992be05e4769486393`, and its `run/ENVIRONMENT.json` is at `43b7d084a46860ca1700030445569780d87edf5484b1801932b08903cdad3c3f`, the pins #702's block records.
- The refused launch, which refused at 2026-11-09T20:46:43Z before its first job, started no job and wrote no log; it counts toward no job's three launches (the ruling's item 5).

| file | SHA-256 |
| --- | --- |
| `run/ENVIRONMENT.json` | `43b7d084a46860ca1700030445569780d87edf5484b1801932b08903cdad3c3f` |
| `run/FROZEN.json` | `1e5107d0c6117a7d45c378827b8ee6670c1ce8964b1099992be05e4769486393` |
| `run/FROZEN_PIN` | `ea3320b424d620fb7b56c7da10808b7696b66f6e7265f7e6d4d8b034c8df7517` |

## The folder

- MIT `94a679acc0c8ae9b7e002a063f0c31da8f82d538`, `files/framework/files/working/files/scripts/soft-slot-dynamics-m8-19/`; its `SHA256SUMS`: 183 files, SHA-256 `1e8f8d0bd56d7ea584362553b1dfe5cc230c7acd8e92e8d6fa49130d09915366`.
- With the first freeze's three files out of `run/`, the check, `shasum -a 256 -c SHA256SUMS`, at 2026-11-10T00:55:02Z: 183 of 183 OK, exit 0; the copy's files were the commit's tree, nothing else.
- The check again, at 2026-11-10T00:55:05Z, after the second writing: 183 of 183 OK, exit 0. Beside the tree, the copy holds only the three the second writing wrote: `run/ENVIRONMENT.json`, `run/FROZEN.json` and `run/FROZEN_PIN`. No other byte changed.

<details><summary>The check's output</summary>

```
$ shasum -a 256 -c SHA256SUMS
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
$ shasum -a 256 -c SHA256SUMS
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

## The environment

- Recorded once more at 2026-11-10T00:55:03Z, by the launch's route: the run's interpreter with `-I -B`, from a process that had imported the driver, as `main()` runs the check. The exact command, from the folder:

```
/opt/homebrew/Caskroom/miniforge/base/bin/python3 -I -B -c "import os, subprocess, sys; sys.path.insert(0, os.path.abspath('run')); import m819_branch; sys.exit(subprocess.run([sys.executable, '-I', '-B', os.path.join(os.path.abspath('run'), 'record_environment.py')]).returncode)"
```

- The two records differ in the four threading fields and the time alone, the interpreter's path and flags included:

| field | the first record | the second |
| --- | --- | --- |
| `recorded` | `"2026-11-09T19:53:09Z"` | `"2026-11-10T00:55:03Z"` |
| `threading.MKL_NUM_THREADS` | `null` | `"1"` |
| `threading.OMP_NUM_THREADS` | `null` | `"1"` |
| `threading.OPENBLAS_NUM_THREADS` | `null` | `"1"` |
| `threading.VECLIB_MAXIMUM_THREADS` | `null` | `"1"` |

- Read again when this block was built, by the launch's route: differences: none.

<details><summary>The first record, in full</summary>

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

<details><summary>The second record, in full</summary>

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
 "recorded": "2026-11-10T00:55:03Z",
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
  "MKL_NUM_THREADS": "1",
  "OMP_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1",
  "VECLIB_MAXIMUM_THREADS": "1"
 }
}
```

</details>

## `FROZEN.json`

- The second `FROZEN.json`, written at 2026-11-10T00:55:03Z by `write_guard.py`, given the ruling's link, with the terms read against OpenWave's main, as at the first writing; its pin, `run/FROZEN_PIN`: `88940b293d440dd004377d1ba1ad948d945a38cf99e375e4640b1449fe9f445c`. The command, from the folder, with the clone's path given here as a placeholder:

```
/opt/homebrew/Caskroom/miniforge/base/bin/python3 -I -B run/write_guard.py https://github.com/openwave-labs/openwave/pull/698#issuecomment-6755529199 --terms-from <an OpenWave clone>
```

- Its 16 pins: the same 15 files at the same SHA-256 as the first's, each at its `SHA256SUMS` line, and the new record. Its terms' SHA-256, solver commit `97a8c08`, manifest SHA-256 and `freeze_numbers` are the first's.
- The two differ in the environment record's pin, `written` and `go` alone:

| field | the first `FROZEN.json` | the second |
| --- | --- | --- |
| `go` | https://github.com/openwave-labs/openwave/pull/698#issuecomment-6749289423 | https://github.com/openwave-labs/openwave/pull/698#issuecomment-6755529199 |
| `written` | `2026-11-09T19:53:23Z` | `2026-11-10T00:55:03Z` |
| `pins["run/ENVIRONMENT.json"]` | `43b7d084a46860ca1700030445569780d87edf5484b1801932b08903cdad3c3f` | `9416dd934575ab88b778b8cbd0a72b02a0109fe6a07f2f193e75f50bb516a43e` |
| its SHA-256, in `FROZEN_PIN` | `1e5107d0c6117a7d45c378827b8ee6670c1ce8964b1099992be05e4769486393` | `88940b293d440dd004377d1ba1ad948d945a38cf99e375e4640b1449fe9f445c` |

<details><summary>The difference, line by line</summary>

```diff
--- the first: run/FROZEN.json
+++ the second: run/FROZEN.json
@@ -3,2 +3,2 @@
- "go": "https://github.com/openwave-labs/openwave/pull/698#issuecomment-6749289423",
- "written": "2026-11-09T19:53:23Z",
+ "go": "https://github.com/openwave-labs/openwave/pull/698#issuecomment-6755529199",
+ "written": "2026-11-10T00:55:03Z",
@@ -18 +18 @@
-  "run/ENVIRONMENT.json": "43b7d084a46860ca1700030445569780d87edf5484b1801932b08903cdad3c3f",
+  "run/ENVIRONMENT.json": "9416dd934575ab88b778b8cbd0a72b02a0109fe6a07f2f193e75f50bb516a43e",
```

</details>

<details><summary>The second, in full</summary>

```json
{
 "frozen": true,
 "go": "https://github.com/openwave-labs/openwave/pull/698#issuecomment-6755529199",
 "written": "2026-11-10T00:55:03Z",
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
  "run/ENVIRONMENT.json": "9416dd934575ab88b778b8cbd0a72b02a0109fe6a07f2f193e75f50bb516a43e",
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

The first, in full, is in [`freeze_block.md`](https://github.com/openwave-labs/openwave/blob/734eb28fe69c05240d237043cc91c915047483d7/openwave/xperiments/m8_mit/research/m8_19/freeze_block.md).

## The launch's two checks, replayed

- The two calls `main()` makes before `launch()`, replayed without `launch()` at 2026-11-10T00:55:04Z, from the folder:

```
/opt/homebrew/Caskroom/miniforge/base/bin/python3 -I -B -c "import os, subprocess, sys; sys.path.insert(0, os.path.abspath('run')); import m819_branch as D; F = D.validate_frozen(); print('validate_frozen: passed;', len(F['pins']), 'pins; go', F['go'], flush=True); e = subprocess.run([sys.executable, '-I', '-B', os.path.join(os.path.abspath('run'), 'record_environment.py'), '--check'], capture_output=True, text=True); print(e.stdout, end='', flush=True); sys.exit(e.returncode)"
```

- `validate_frozen()` passed, 16 pins, the ruling's link as the go; `record_environment.py --check`, as its subprocess, read no difference.

<details><summary>The output</summary>

```
validate_frozen: passed; 16 pins; go https://github.com/openwave-labs/openwave/pull/698#issuecomment-6755529199
recorded: {
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
  "MKL_NUM_THREADS": "1",
  "OMP_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1",
  "VECLIB_MAXIMUM_THREADS": "1"
 }
}
live: {
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
  "MKL_NUM_THREADS": "1",
  "OMP_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1",
  "VECLIB_MAXIMUM_THREADS": "1"
 }
}
differences: none
exit 0
```

</details>

## The scripts

- The steps, in the ruling's order: `freeze_2.py`, SHA-256 `c3ac9855a3daaea92918e7fc19ff4e7baeb43198b959882b8299eb1638bcc4b1`, run once on the folder; its output, SHA-256 `e06eda9fb719740ef5de89ab70bfbd9fa53734d2bee9f40af349ae46836b63b2`, and each step's, kept with it.
- The checks before the second writing, read-only: `checks_before_write_2.py`, SHA-256 `b4a936553e502a70c48a46109c54b0c90062d89a46bf17b8285d507d59984c5e`, 25 of 25 passed. The read-back after it: `verify_frozen_2.py`, SHA-256 `74c916d1c92ff94da6d1dac786101bdda9cd01934eb804c866c1c554c3298802`, 18 of 18.
- The rehearsals in scratch copies, read for nothing: `rehearse_env.py`, SHA-256 `3ed5d9ab7c5438a9f583763cbd0827ec9387a8d5096893134f85eb3d2f280972`, its output `8f087655b5e38ef2003fb1460561b4827390c9804d28c73230ca10018c2d5c3f`; `rehearse_launch.py`, SHA-256 `6d9ea94d736a8ce126258e9f78f66579b7bb02f8b9e5d7944fa46ee7d582a823`, its output `34233d1e91e32e288ed65cbc0604bdda5dd9e5a4af0556930278d143bc6a2d81`; and `freeze_2.py` run whole on a scratch copy of the folder before the second writing, its output `46f50cb82f1a5b45a2263d10abf688d0b8cc06734b41b62e4d4f603eab96400c`.
- All published with the record in MIT's M8.19 folder. The scripts, and the outputs of `freeze_2.py` and of its rehearsal, name the bench's home folder or a scratch path, so they go out as marked archival copies, the path configurable in a script and a placeholder in an output: each copy names its original's SHA-256, given here, and differs from it only where the path was. The other outputs go out as they are. The first block's builder, `freeze_block.py`, whose original is at `0af0d171…` as #702's pull request names it, goes out the same way.

## Unchanged

- The terms, the instrument, the reference files and the order of the run's jobs are the first block's: every file the freeze pins is at its first SHA-256 but `run/ENVIRONMENT.json`, and `FROZEN.json`'s terms, solver commit, manifest and frozen numbers are the first's.
