# M4.9 - Emergent Encoding from Lattice Dynamics

## Status
DONE (post-hoc)

## Criterion
`Gravity: local metric phenomena` — foundational encoding derivation

## Objective
Demonstrate how the macroscopic timelike encoding
($v_{\text{clock}} \propto \sqrt{\eta}$) maps directly to microscopic 1D
lattice dynamics under an assumed $1/r$ EMC pair potential ($n=1$) and
axial lattice compression ($q=1$). The null-probe refractive index
$n_\gamma = \eta^{-1}$ follows from the same strain field through the
null-probe combination $\sqrt{g_{rr}/g_{00}}$ (manuscript v5.1.7), not
from $1/v_{\text{phys}}$.

The required encodings are:

- \(v_{\text{clock}} \propto \sqrt{\eta}\) (measured here, \(\beta = 0.5\))
- \(n_\gamma = \eta^{-1}\) (derived from \(\sqrt{g_{rr}/g_{00}}\), not from
  \(1/v_{\text{phys}}\); manuscript v5.1.7)

## Method

1. Define the microscopic pair potential \(V(r)=V_0/r\).
2. Derive the lattice stiffness from
   \(k(\eta)=d^2V/dr^2|_{r=1/\eta}\).
3. Keep the mass per lattice site fixed at \(m_0\).
4. Use physical wave speed
   \(v_{\text{phys}}=a(\eta)\sqrt{k/m_0}\).
5. Measure the wave-speed exponent \(\beta\).
6. Derive the oscillator-frequency exponent from
   \(f=v_{\text{phys}}/L\).

## Result

- Stiffness exponent: \(\alpha = 3.000002\)
- Wave-speed exponent: \(\beta = 0.501714\)
- Oscillator exponent: \(\gamma = 0.501714\)

All three are consistent with the expected values.

## Interpretation

Given a $1/r$ EMC pair potential ($n=1$) and axial lattice compression ($q=1$),
the physical (timelike) wave speed scales as $v_{\text{phys}} \propto
\eta^{n/(2q)} = \eta^{+1/2}$; this is the clock exponent used in M4.4.
The null-probe refractive index is not \(1/v_{\text{phys}}\): light samples
both the spatial cell stretching and the temporal propagation-period
dilation, giving \(n_\gamma = \sqrt{g_{rr}/g_{00}} = \eta^{-1}\), the
encoding M4.3 and M4.5 use.

## Artifacts

- `research/scripts/m4_9_emergent_encoding.py`
- `research/findings/m4_9_emergent_encoding.md`

## Reference

Enhanced EWT manuscript, version 5.1.7:
[DOI: 10.5281/zenodo.23559858](https://doi.org/10.5281/zenodo.23559858)

Relevant section:

- „Emergent Metric Encodings from Lattice Dynamics"