# M4.14 — Cosmological Constant from the BCC Lattice

## Status
DONE (post-hoc)

## Criterion
`Cosmology: Λ / cosmic acceleration`

## Objective
Derive the cosmological constant Λ from the BCC lattice chain (N_geom,
ε_M, M_μ, horizon factor H, EWT trinity, T_p, R_H, H_0, Λ_dS, Ω_vac),
with no continuously adjustable parameter beyond the four dimensional
anchors (`r_e`, `m_e`, `c`, `e`) and one cosmological anchor (`T_cmb`).

## Outcome

The chain reproduces `Λ` to 0.60% on the Planck 2018 base set (−2.08%
on the set with BAO) under two stated identifications: the horizon
factor `H = 8·(2π)²`, a reading of the `32π²` of the Haug and Tatum
relation, and the product `Ω_vac = η_BCC·C_gap`. Neither is derived
in-platform. Numbers, gaps and falsifiers are in the findings note.

## Maintainer note (review of PR #717, 2026-11-14)

Edits applied at merge, announced in the review and agreed by the author
in the thread of [PR #717](https://github.com/openwave-labs/openwave/pull/717):
the `M_W` credential removed from `C_gap` (the engine's `M_W` prediction
uses `C_gap_mixed`); the `G_geom` paragraph replaced by the square-root
dependence of `H_0`; the horizon relation named as the Haug and Tatum
relation, with the `T_cmb` line labeled an identity and the CODATA figure
66.8954 given at the point of use; the Planck 2018 base set as the single
reference, with the set with BAO beside it; `Λ_Planck` computed as
`3·Ω_Λ·H_0²/c²` in place of the rounded literal; the matter-correction
diagnostic (the former Part 13) removed and the summary renumbered;
the sentence making the lattice test conditional on the unit `(2π)²`;
`T_CMB_CODATA` renamed `T_CMB_COBE_FIRAS`; the version header set to
5.3.1; a Reproduction block in the findings note; the reference pinned
to the v5.3.1 version DOI.

## Artifacts

- `research/scripts/m4_14_ewt_cosmological_constant.py`
- `research/findings/m4_14_cosmological_constant.md`

## Reference

Enhanced EWT manuscript, version 5.3.1 (2026-11-13), § "The Cosmological
Constant from the BCC Lattice" (label `sec:cosmo_summary`). DOI:
[10.5281/zenodo.24207609](https://doi.org/10.5281/zenodo.24207609).
