# M10.2: charge, localized energy and field/frame rotation

## Source and observables

On a seam $(a,b)$ with fair coin $c\in\{0,1\}$, the signed load record is

$$
d=q_a-q_b,\qquad
j=\frac{d-(2c-1)(|d|\bmod 2)}{2},\qquad
q'=q-j e_a+j e_b.
$$

For an address set $E$, the candidate's coherent branches are

$$
K_{e,c,j}=\frac{1}{\sqrt{2|E|}}
\sum_{z:\,j(z,e,c)=j}|F_{e,c}(z)\rangle\langle z|,
\qquad K_{\mathrm{invalid}}=\Pi_{\mathrm{invalid}}.
$$

Positive real amplitudes are a declared memory law. Fine branches are partial
injections, so their effects, including the invalid identity branch, sum to
identity. Vacuum and zero-current outcomes are included. The observer prepares,
acts on and reads the same coordinate register. The candidate has 240 positions
and 630 seams; its cell-addressed charge/moment assay uses 30 seams. Fresh
preparations assay one source species.

The charge readout and integrated current satisfy

$$
BF=q_{\mathrm{initial}}-q_{\mathrm{terminal}},\qquad
\mathbb E[F_\tau]=B^T L^+q_{\mathrm{initial}}.
$$

The cell's relative integer class is primitive. The experiment distinguishes
two probabilistic integer outcomes from two sharp half-charge daughters. The
full graph has a primitive boundary image; the cell's sphere-face topology is
not asserted for the entire graph.

The full-graph single-source transfer and energy convention are

$$
T=I-L/1260,\qquad H=-1260\log T,\qquad
f=(|4\rangle-|5\rangle-|6\rangle+|7\rangle)/2,\quad Lf=6f.
$$

The constant mode fixes zero energy. The selected ray has
$E_6=-1260\log(209/210)$. A disjoint compact pair has
$E_{12}=-1260\log(104/105)$ on the positive Dirichlet transient kernel.
The inverse-preparation pointer certifies that energy eigenray; every
complementary outcome is retained as unresolved energy. The 511 keV anchor
sets the reporting unit. It is not a predicted electron mass.

For the twelve source axes $v$ with $\phi=(1+\sqrt5)/2$, sum all 60 first-current
alternatives per axis, with $\Delta=j(v_b-v_a)$:

$$
m=\mathbb E[\Delta]=\frac{\phi-3}{15}v,\qquad
C=\mathbb E[(\Delta-m)(\Delta-m)^T],\qquad
C_\parallel=\frac{3-\phi}{9},\quad C_\perp=\frac{\phi+2}{15}.
$$

The executed proper cell rotation $R=\mathrm{diag}(-1,-1,1)$ gives
$m'=-m$, $C'=C$ on the selected antipodal pair; $R^2$ restores the frame.
This is a finite tensor/frame observation. It supplies no spinor phase,
exchange statistic or global continuous rotation theorem.

**Scope**: three finite static experiments on one declared signed-source
instrument, with proposed physical charge, rest-energy and field/frame maps;
no physical clock, free lifetime, spin magnitude, full particle identification
or automatic OpenWave cell award follows.

## Equation-to-code map

All author-package links select commit `c330c6e7f23506b5f87c339cd73906928c1a7cd9`. The platform script
fetches that exact commit, checks every package byte and file-set closure,
then runs the package in disposable copies.

| Object | Implementation |
| --- | --- |
| Endpoint delegation and common graph | [`source.py:1`](https://github.com/FloatingPragma/observer-patch-holography/blob/c330c6e7f23506b5f87c339cd73906928c1a7cd9/code/openwave_static_signed_source/evidence/openwave/shared/static_candidate_join/source.py#L1) |
| Complete coordinate instrument and compact eigen-equations | [`verify.py:17`](https://github.com/FloatingPragma/observer-patch-holography/blob/c330c6e7f23506b5f87c339cd73906928c1a7cd9/code/openwave_static_signed_source/evidence/openwave/shared/compact_static_adjudication/verify.py#L17) and its [interface theorem](https://github.com/FloatingPragma/observer-patch-holography/blob/c330c6e7f23506b5f87c339cd73906928c1a7cd9/code/openwave_static_signed_source/evidence/openwave/shared/compact_static_adjudication/INTERFACE_THEOREM.md) |
| Integer topology and stopped-current response | [`incidence`, `source_branches`, `trajectory`](https://github.com/FloatingPragma/observer-patch-holography/blob/c330c6e7f23506b5f87c339cd73906928c1a7cd9/code/openwave_static_signed_source/evidence/openwave/shared/static_candidate_join/run.py#L92) |
| Independently accumulated current mean and covariance | [`moments`](https://github.com/FloatingPragma/observer-patch-holography/blob/c330c6e7f23506b5f87c339cd73906928c1a7cd9/code/openwave_static_signed_source/evidence/openwave/shared/static_candidate_join/run.py#L157) |
| Failed coherent ancestry join | [`old_pair_map`](https://github.com/FloatingPragma/observer-patch-holography/blob/c330c6e7f23506b5f87c339cd73906928c1a7cd9/code/openwave_static_signed_source/evidence/openwave/shared/static_candidate_join/run.py#L185) |
| Independent reconstruction | [`audit.py`](https://github.com/FloatingPragma/observer-patch-holography/blob/c330c6e7f23506b5f87c339cd73906928c1a7cd9/code/openwave_static_signed_source/audit/audit.py#L1), with [per-claim audit](https://github.com/FloatingPragma/observer-patch-holography/blob/c330c6e7f23506b5f87c339cd73906928c1a7cd9/code/openwave_static_signed_source/audit/AUDIT.md) |

Minimal inspection: source module, full [proof](https://github.com/FloatingPragma/observer-patch-holography/blob/c330c6e7f23506b5f87c339cd73906928c1a7cd9/code/openwave_static_signed_source/evidence/openwave/shared/static_candidate_join/PROOF.md), interface theorem, then the [platform driver](../scripts/m10_2_replay.py).

## Reproduction

Install the package's pinned NumPy/SciPy/SymPy requirements in a Python 3.12 or
3.13 environment, or use the tested versions recorded in the receipt. No
Taichi launcher or GPU is needed. From an OpenWave checkout:

```sh
python3 -B openwave/xperiments/m10_oph/research/scripts/m10_2_replay.py \
  --output openwave/xperiments/m10_oph/research/data/m10_2_replay.json
```

This downloads only the author package via a commit-pinned sparse checkout.
For an existing author checkout at the exact commit, pass `--source-dir`.
The author supplies the runs; review is under the platform's existing rules.

The historical archive contains an interpreter-dependent cache pin. The
documented adapter removes that one pin only in scratch, verifies all 316
published evidence files and creates 150 provider path aliases from identical
hashed bytes. The localized reader's timing receipt is compared and restored
before its downstream consumer. The original science files and manifests are
unchanged. The adapter rejects missing, changed and extra inputs, and rejects
nonfinite comparison values.

The platform's generic provenance checker verifies the delivered file list
and byte counts with zero failures, but reports three warnings: no ordered
concatenation protocol, no provenance-id object, and 54 hash references that
do not identify individual files in this export. Those references include
historical review inputs, ancestral claim digests, the disclosed bytecode
cache, and regenerated stream digests. This is not a strict provenance-archive
clearance. The task-specific checks bind every executed scientific input and
verify the regenerated action/record digests; broader archived claims are not
reproduced by M10.2.

## Results and review boundary

The reproduction gates and their failure conditions are in the
[task record](../tasks/m10_2_task_details.md). Full execution output is in
[m10_2_replay.json](../data/m10_2_replay.json), with the
[validation receipt](../data/m10_2_receipt.json). Packaging findings and dispositions
are recorded in the [submission audit](m10_2_submission_audit.md).

| Test | Result to reproduce | Gate |
| --- | --- | --- |
| Common signed-source execution | 435,080 actual endpoint calls; 264/264 charge comparisons absorb; all 48 changed-law controls fail to absorb | P4: exact records, absolute 1e-10 numerical comparisons |
| Complete source instrument | 82,575,360 pair columns, 645,120 single columns, 144,549,720 continuity checks; altered chart fails 24 columns | P2: every retained result field and original assertion |
| Localized energy | $E_6=6.014331228348673$, $E_{12}=12.057508280349891$; pair 1024.4508487023415 keV at the 511 keV anchor | P2/P3: exact eigen-equations and complete retained outcome laws |
| Tensor/frame | All 720 alternatives; half-turn covariance return and frame reversal; fixed-seam control fails on 4/12 axes | P4: exact radical arithmetic |
| Independent stopping solution | Absorption exactly 1 on all 132 states; expected duration 66–90 events; 3,960 flow entries agree | P5: independent rational chain solve |
| Independent compact and moment calculations | 20,160 pair-support branches, 64 exterior cancellations, zero vacuum amplitude; all 720 symbolic moment alternatives agree | P5: separate implementation |

The second-agent audit confirms these finite claims and qualifies their physical
interpretation. It uses floor/ceiling endpoint averaging, graph-generated
faces, a solved absorbing chain and separately accumulated symbolic moments.
It is a second implementation within the producing group, not external-group
confirmation. The relevant finite Lean statements were inspected with their
hypotheses; no Lean build is claimed in this task.

The constant zero mode and a nonconstant Rayleigh quotient of 1/3 are retained.
The selected compact state is not proved to be the unique lowest charged state.
The source-service error bound is conditional on the declared proper-code
precision, not measured physical hardware accuracy. The positive coherent body
law is supplied as architecture. Its implementation by a narrower passive
repair process is not established.

The old paired-source construction supplies classical ancestry only. It fails
original-coin equality on 12,000 of 79,200 branches. Its retained fine record
destroys a coherence that the signed source preserves: off-diagonal 0 versus
1/4 at branch probability 1/2. None of its coherent successes is transferred.

The review request is whether the displayed carrier/source construction and
operational dictionary satisfy each static criterion. No `MODELS.md` icon is
changed by this submission.
