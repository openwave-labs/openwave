# M4.15 — Sync M4.7, M4.11 and M4.14 to the v5.4.4 Code Record

## Status

DONE (post-hoc)

## Criterion

`Repository sync` — align M4 scripts, cross-module imports, and reference targets with the latest published manuscript code record (v5.4.4). No `MODELS.md` criterion.

## Objective

Synchronize the M4 research scripts  from the intermediate v5.3.1/v5.3.2 state to the published Enhanced EWT **v5.4.4** manuscript and code archive (`10.5281/zenodo.24525643`), update cross-module import paths to match the `m4_<n>_` repository prefix convention.

## Outcome

The M4 script suite is synchronized with the v5.4.4 code record (`10.5281/zenodo.24525643`) and manuscript (`10.5281/zenodo.24513845`). Direct byte-level SHA-256 checksums do not match the upstream Zenodo archive because OpenWave enforces the `m4_<n>_` filename prefix convention and corresponding internal module imports (`from m4_7_ewt_emergence_engine import ...`), whereas the upstream Zenodo release uses unprefixed standalone script names (`ewt_*.py`). After normalizing those differences, the repository scripts are identical to the archive's or differ only in line wrapping, one comment line, or the archive's header line; the file-by-file table, the normalization rule and the archive's md5 values are in the findings note. M4.11 is unchanged; its Weinberg row and Planck set moved to M4.18.

## Artifacts

- `research/scripts/m4_7_ewt_emergence_engine.py`
- `research/scripts/m4_14_ewt_cosmological_constant.py`
- `research/findings/m4_15_v5_4_4_sync.md`

## Reference

Enhanced EWT manuscript, version 5.4.4 (2026-11-24), DOI: [10.5281/zenodo.24513845](https://doi.org/10.5281/zenodo.24513845).  
Associated v5.4.4 code archive, DOI: [10.5281/zenodo.24525643](https://doi.org/10.5281/zenodo.24525643).
