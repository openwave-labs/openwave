# M5.40: The Card C payload: the author's 33³ and 55³ fields under the instruments of M5.39

> Opened 2026-11-14 from the author's reply to the results post of [M5.39](m5_39_task_details.md) on discussion #186 ([discussioncomment-19612571](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19612571), 14 November 2026, 12:53 UTC), which announces the capstone v3.7 (Zenodo record [24220797](https://doi.org/10.5281/zenodo.24220797)) and, attached to that record, the file `CardC_payload_recovered.zip`: the two fields, their run records, the energy code and the scripts behind the printed readouts. Roadmap row: [`m5_roadmap.md`](../m5_roadmap.md). Thread record: [`m5_32_convo.md`](m5_32_convo.md). Sources: [`_CITATIONS.md`](../../theory/_CITATIONS.md) (the v3.7 rows). This task is step 2 of the sequence of [M5.39](m5_39_task_details.md) § 7, the comparison held back until the fields arrived.

## TASK PLANNING (2026-11-14; the go is the user's)

**Scope.** (1) The payload filed and verified (§ 1, done at plan time). (2) R1: the evaluator of M5.39 on the author's two fields, with the run records as known answers (§ 3). (3) R2: the readers of M5.39 on those fields, against the printed rows and against the states of M5.39 on the same grids (§ 4). (4) R3: a continuation from each of the author's fields under the card's optimizer settings with a tighter stop, which puts finding (i) of M5.39 (the force gate leaves the energy open) to the author's own states (§ 5). (5) R4: the readout definitions of the deposited scripts written as equations, coded independently, gated against the author's scripts and the printed digits, then read on the mesh ladder of M5.39 (§ 6). Out of scope: the converged background and the zero-spacing calibration (the following task, § 8); the 45³ field, which is not in the recovered archives; the tilt reduction and the charged core ([M5.39](m5_39_task_details.md) § 2, rows 12 and 13); every run of [M5.39](m5_39_task_details.md) § 5; any icon or pointer of [`MODELS.md`](../../../../../MODELS.md), since every run of this task is on the author's Card C and not on this column's certified action.

**Definition of done.** The four rungs run with their known-answer controls (§ 3 to § 6) each reported as met or as the exact failing check; every number in the findings carried by a tracked JSON and a script; the continuation reads taken at two stops; an audit by agents on a different model with every claim either standing or qualified (§ 7); the task review presented in the terminal. A rung whose control fails after its three tries is reported as failed, with its reads kept as numbers.

**Gating.** The payload (arrived, § 1). The user's go.

**Model and effort.** The session's model for the run; the audits on a different, smaller model (two agents in parallel), named at each spawn.

**Research body.** This document (`## FINDINGS` after the run); scripts `../scripts/m5_40_*.py`; data `../data/m5_40_*.json` (tracked) and `../data/m5_40_*.npz` (local-only, indexed by [`_DATASETS.md`](../data/_DATASETS.md)); plots `../plots/m5_40_*.png`, embedded in the findings; in-flight state in `../checkpoints/m5_40/` (local-only).

### 1. The release and the payload as filed (plan time)

The capstone v3.7 (399 pages) and its source archive were filed beside the earlier versions, both matching the checksums of the Zenodo record. Read against v3.6 by hash: 1185 files identical, 3 changed (the text, the contribution map, the README), 13 removed, 49 new (48 under `checks/v37/` and the figures, and one manifest file). The 39 files of `frozen_packets/` that M5.39 hashed are unchanged, so R0 and R1 of [M5.39](m5_39_task_details.md) (§ 11, § 12) stand for v3.7.

The payload (16 files, 33 MB) carries its own digest list, and all 14 listed files verify. Eight of them are the entries of the manifest that v3.4 already printed (`frozen_packets/CardC_payload_manifest/`), with the same digests; the three energy files are byte-identical to the copies in `frozen_packets/STRING_v0.6/sector_b_action/` that M5.39 gated its evaluator against; the six scripts of `definitions/` are new to the record (one of them, `inertia.py`, is the archived audit script already in the release).

| Item of the payload | What it is | Plan-time read |
| --- | --- | --- |
| `bq_c33/residual_edge/final.npz`, `run.json` | the 33³ field (half-box 3.6, h 0.225) and its run record | arrays `state` (33, 33, 33, 13), `q` (…, 4), `record` (…, 3, 3), `coords`; recorded energy 76.05003057685762, residual 9.398 × 10⁻⁴ (winding part 4.938 × 10⁻⁴), margin 0.70071, largest record amplitude 0.39376 |
| `bq_rc76/gate55/final.npz`, `run.json` | the 55³ field (half-box 6.075, h 0.225) and its run record | same arrays at 55³; recorded energy 75.56196067295708 after one step of the recorded method, residual 7.687 × 10⁻⁴ (winding part 6.955 × 10⁻⁴), margin 0.70865, largest record amplitude 0.39547 |
| `bq_c_refine/bq_core.py`, `dn87_relax.py`, `fast_quartic.py` | the Card C energy code | identical to the archived copies |
| `bq_rc76/kinetic_operator.py` | the kinetic operator of the 55³ package | imports a module (`quadrupole_ritz`) that is not in the payload; not used by this task |
| `definitions/static_preflight.py` | the direct axial diagnostic (`axial_twist`) behind the printed g_A | imports only deposited modules |
| `definitions/zero_q2.py`, `inertia.py`, `lattice_source.py` | the two radii and the moment μ_V | `zero_q2.py` and `inertia.py` import only deposited modules |
| `definitions/magnetic_correction.py`, `radius_sensitivity.py` | the isocranking correction and a susceptibility of the radii | read for definitions only |

Three facts of the run records shape the plan. (a) Both records name the seed `smooth`, and both name a method, "normalized raw-residual polish with descending energy", with options (`sobolev`, `damping`, `frame_scale`, `filter_cutoff`) that the deposited driver `dn87_relax.py` does not have: the driver that produced the two fields is not in the payload, and each record starts from an input field (`bq_c33/last_edge/final.npz`, `bq_rc76/final55/final.npz`) that is not in it either. The reply says the same of the 33³ route. So no rerun on the author's route is possible, and the plan replaces it by a continuation from the author's end states (§ 5). (b) The 33³ record holds one history entry, at step 0, and ends with the optimizer's message that its iteration limit was reached. (c) The largest record amplitude of each record equals the largest printed center eigenvalue to three digits, so the records do not show a lump of the kind M5.39 found on its own `smooth` start (a node at 0.96).

### 2. The consumers

| Sub-result | Consumer | Instrument check the consumer needs |
| --- | --- | --- |
| R1: energy, residual in both readings, the length of the winding vector on the author's fields | the gate of the following task (the author's proposed rule: a unit-normalized residual plus a lump check) | the run record's energy returned to 10⁻¹⁰; the record's residual identified with one reading, or reported as not identified |
| R2: the printed rows from the author's fields | the results post; the author's text | each reader's known answer on the author's field before it reads a state of M5.39 |
| R3: the energy and inertia along the continuation | the calibration question (how far a gate-passing field sits from its valley floor, and whether the inertia moves with it) | the reads at two stops; a start that reproduces the recorded energy |
| R4: the readouts on the mesh ladder | the zero-spacing calibration of the following task | the author's script on the same field, where it runs; the printed digits |

### 3. R1: the evaluator on the author's fields

The fields are converted to the unknowns of [`m5_39_cardc.py`](../scripts/m5_39_cardc.py) by a new script, `m5_40_payload.py`, which also repeats the digest check and writes the converted copies as local-only arrays. Nothing of the payload is executed from its filed folder: the author's scripts run from an isolated copy, with the interpreter's isolation flags.

| Control | Known answer | Fails when |
| --- | --- | --- |
| K1, the energy | the recorded energy of each field, within 10⁻¹⁰ relative | the packing of `state`, a constant or the box differs from the reading of M5.39 |
| K2, the record's geometry reads | degree 1; the recorded margin and largest record amplitude to five digits, by the definitions of the deposited driver | the definition is not in the deposited code (then reported as not identified, not as a failure of the field) |
| K3, the residual | the three recorded residuals of each field, within 1 percent, by one of two readings of our gradient: on the winding vector as stored (the gate as M5.39 read it), and projected on the unit sphere and rescaled to unit length | neither reading returns them: the residual is then NOT IDENTIFIED and becomes a question to the author |

Also read: the range of the length of the stored winding vector over the grid. If it is 1 to round-off on the author's fields, the statement of M5.39 that the gate is read on a vector whose length grows to 3.9 holds for the states of M5.39 and not for the author's, and the results post says so.

### 4. R2: the readers on the author's fields

The readers of [`m5_39_reads.py`](../scripts/m5_39_reads.py), unchanged, on the two fields: the energy by term, the center eigenvalues and the branch, the lump count, the six-generator inertia and the gap, the tail amplitude (55³), the isoscalar radius on the winding density, the mirror reads.

| Printed value | From | Word |
| --- | --- | --- |
| center eigenvalues 0.333, 0.376, 0.394; inertia 34.33; gap 311.9 MeV; r²_V 0.4519 fm² | the 33³ field | REPRODUCED when the reader returns the printed digits from the author's field, DIFFERS with the number otherwise |
| inertia 37.44; fitted tail 0.518 fm² | the 55³ field | the same |

The comparison words apply here because the field is the author's own: the reader is the only thing between the field and the printed digit. Then, as numbers without words: each term of the energy of the author's field beside the states of M5.39 on the same grid (33³: the census state and the two extensions; 55³: the five starts), and the radial profiles of the winding density and of the trace of the record's Gram matrix. A node-by-node difference is given only as a diagnostic, since two states of one valley can differ by a lattice symmetry.

### 5. R3: a continuation from the author's fields

Each of the author's fields is continued by the driver of M5.39 ([`m5_39_relax.py`](../scripts/m5_39_relax.py): the card's optimizer settings, the admissibility wall), with the optimizer's stop at a tenth of the gate, since a start that already meets the gate would stop at once.

| Job | Start | Cap | Reads |
| --- | --- | --- | --- |
| 33³ | the author's 33³ field | 10000 iterations | at every saved state: energy, residual in both readings, center eigenvalues; at the first state after 3000 iterations and at the end: inertia, lumps, radii |
| 55³ | the author's 55³ field | 3000 iterations | the same, the inertia at 1500 and at the end |

Recorded before the run, from the census of M5.39 (gate-passing states of one branch 1.2 × 10⁻³ apart on 55³; the 33³ route still falling at 76.05036): the energy falls from the author's value by between 10⁻⁴ and 2 × 10⁻³ on each grid within the caps, the center eigenvalues move by less than 0.01, and the inertia by less than 0.1 percent. The other outcome is as informative and is reported with the same weight: a fall under 10⁻⁵ means the author's field sits at the floor of its valley at this resolution, and finding (i) of M5.39 then describes the routes of M5.39 and not the author's. The start control: the continuation's first evaluation returns K1.

This rung is the one addition to the comparison as first described; it is two background jobs on stored fields and it restarts no rung of the author's.

### 6. R4: the readout definitions

1. The definitions of `static_preflight.py` (`axial_twist`) and `zero_q2.py` (the two radii, the moment) are read and written into the findings as equations, with the normalization each printed value uses.
2. They are coded independently in `m5_40_readouts.py`, on the fields and conventions of [`m5_39_cardc.py`](../scripts/m5_39_cardc.py).
3. Gate (a): the author's scripts, run unmodified from an isolated copy on the author's fields, against `m5_40_readouts.py` on the same fields, within 10⁻⁸ relative. Three tries at making them run without editing them (arguments, working folder, input names); a script that still needs a file or a module outside the payload is reported as not runnable as shipped, and that readout is gated by (b) alone.
4. Gate (b): the printed digits. From the 55³ field: g_A 0.904 at the normalization of the 55³ reset (e = 4.311), and 0.957 at the earlier one the reply names. From the 33³ field: μ_V 3.004, r²_S 0.2303 fm², r²_V 0.4519 fm².
5. The same readouts on the states of M5.39: the mesh ladder at half-box 6.075 (45³, 55³, 65³, 77³) and the box ladder (73³, 91³), as numbers with their trend in h², under the caveats those states carry in [M5.39](m5_39_task_details.md) § 13 and § 17.

### 7. Blindspot pass, the audit and the records

Read against the recurring failure families of the workflow, for this task's instruments.

| Risk | Where it enters | Guard |
| --- | --- | --- |
| Unvalidated instrument | the converter of the fields; the independent readouts | K1 before any other read; gate (a) or (b) before a readout is applied to a state of M5.39 |
| Artifact read as result | the stop and the cap of the continuation | reads at two stops; the fall given with its last-tenth drift |
| Unfalsifiable check | a residual "identified" by a free rescaling | K3 allows the two stated readings only, no fitted factor |
| A definition taken for another | the recorded residual carries the word "normalized" and options the deposited driver lacks | K3's NOT IDENTIFIED outcome; the residual's definition goes to the author when it is not in the deposited code |
| Resolution mistaken for accuracy | the readouts on the mesh ladder | the states' own spread (the census of M5.39) stated beside each number |
| A stored field in another convention | the author's fields may store a unit winding vector where M5.39 relaxes an unnormalized one | the length read of § 3 |

The audit: two agents on a different model, in parallel, each asked to refute. One takes R1 to R3 (every number of the findings against the JSON, the conversion re-derived from the deposited code, the two residual readings recomputed); the other takes R4 (the equations against the author's scripts line by line, one readout recomputed from scratch). The thread record gets its row when the results are posted; the post is the maintainer's, after the review.

### 8. The parallel shape, the estimate and what follows

| Rung | Compute | Shape |
| --- | --- | --- |
| R1, R2, R4 | seconds to minutes per field on stored arrays | one process; a pool would cost more than it saves on a handful of evaluations |
| R3 | 33³: about 0.3 s per iteration, about 50 minutes at its cap; 55³: about 1.4 s per iteration on one thread in the pool of M5.39, about 40 minutes at its cap with the compiled term threaded | two jobs side by side, one thread for the 33³ job and six for the 55³ job, seven busy cores; the one-thread pins of M5.39 on every other library |

Go to review: about an hour and a half. The continuation's fifty minutes set the floor; R1, R2 and R4 run at the desk while it runs, and the two audits run in parallel at the end.

What follows this task, as its own plan: the converged background on the mesh ladder under the rule the author proposes (a gate on the unit-normalized residual plus a lump check), with the zero-spacing calibration the reply agrees should wait for it. Its thresholds are written from K3 and from the continuation of § 5. The release also ships a tilt-texture kit (`checks/v37/LdGS_tilt_texture_kit_v1.2.zip`), an input for the plan of the tilt reduction ([M5.39](m5_39_task_details.md) § 2, row 13); it is not opened by this task.
