# M10.2 submission audit

The finite claims were checked by a second Codex agent using a separate
implementation. The [method note](m10_2_method_note.md) links the script, its
input hashes and per-claim report at the author-package commit. The submission
also reruns the original compact-coordinate audit, localized reader and common
signed-source experiment. All producer records remain in the author repository.

**Scope**: this audit checks finite calculations, reproducibility, source
identity and claim wording for M10.2; it is not external-group confirmation,
physical-dictionary acceptance or a platform award.

## Findings and dispositions

| Finding | Disposition |
| --- | --- |
| Export includes an interpreter-dependent cache pin | One named `.pyc` entry omitted from delivery and removed only in a scratch preregistration copy; original manifest retained; every scientific source byte checked |
| Localized provider expects paths represented under exported `standalone/` prefixes | 150 paths reconstructed from identical verified bytes; every alias checked against the original provider pins |
| Localized reader rewrites a timing receipt consumed by the next script | Compare regenerated fields excluding elapsed time, then restore the original receipt in scratch before downstream input checks |
| Absolute-error comparison can accept nonfinite values | Explicit finite-value checks; planted NaN and both infinities rejected |
| Inherited Python optimization could disable archived assertions | Clear `PYTHONOPTIMIZE` in both child environments; exercise the platform run with optimization set in the parent |
| Transport mutations could start from a contaminated package | Verify the unchanged package first and suppress bytecode writes on import |
| Independent audit used byte-exact floating output comparison | Exact keys, integers, labels and hashes retained; finite floats checked at absolute tolerance 1e-10, including nested lists |
| Canonical table's admission-number links could be read as the active candidate | Admission snapshot labelled separately from the M10.2 candidate, with no transfer of other candidates' results |
| Generic provenance checker reports unmatched historical hashes | Three warnings disclosed in the method note; no strict provenance-archive clearance claimed; live replay dependencies separately verified |

The independent finite reconstruction confirms charge topology and stopping
response, localized transfer eigen-equations, and the finite mean/covariance
rotation statement. The physical identifications are qualified. The declared
coherent body law, energy-meter complement, lower charged modes and finite
rotation scope remain explicit.

## Verification

The [platform replay](../data/m10_2_replay.json) includes both producer replay
and the independent calculation. The [transport checks](../data/m10_2_transport_check.json)
reject missing, edited, extra and re-manifested inputs, nonfinite results and
unexpected receipt fields. The source-package audit includes separate mutation
checks of its exact/finite-numeric comparator.

The platform matrix and roadmap linters pass. Python formatting, syntax,
changed-file local-path checks, relative links and diff whitespace are checked
before submission. All platform changes are confined to the M10 directory;
`MODELS.md`, shared engines, policies, dependencies and CI are unchanged.

This contribution uses the platform's existing review process. It introduces
no additional maintainer freeze, execution, derivation or future-work obligation.
Author records describe their own custody; they do not bind platform files.
