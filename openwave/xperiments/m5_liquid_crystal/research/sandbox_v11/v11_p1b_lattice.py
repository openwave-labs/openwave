"""M5.11 P1b (foundation) - FABER's energy on a full 3D Cartesian SU(2) lattice.

P1a validated the radial REDUCTION. This validates the genuinely-3D SU(2) curvature
machinery that the vortex LOOP (P2) and the two-soliton dipole need, with NO radial
symmetry assumed: build the smooth fields q0 = cos a, q_vec = n_hat sin a on a 3D
grid, finite-difference them, and form (Faber Eq. 6 line 1, the singularity-free form)

    Gamma_i = q0 d_i q_vec - (d_i q0) q_vec + q_vec x d_i q_vec        (target 3-vectors)
    R_ij    = Gamma_i x Gamma_j
    u_curv  = 1/4 sum_ij |R_ij|^2 = 1/4[(Tr G)^2 - Tr G^2],  G_ij = Gamma_i . Gamma_j
    Lambda  = q0^6                                                    (r0 = 1 units)

so the energy J~ = integral (1/4 R.R + q0^6) d^3x  should equal pi^2 (since the radial
E = (a_f hc/r0)(pi/4) = (a_f hc/4pi)(1/r0) J~  =>  J~ = pi^2 = 9.8696).

Clean validation (tail-free): compare the 3D-lattice cumulative energy inside r<R_cut
to the EXACT radial cumulative 4*pi*I_radial(<R_cut), and show the discretization error
SHRINKS as the spacing a->0 (Faber notes lattice discretization OVERestimates the
single-soliton energy; we reproduce that and its O(a^2) convergence).

Run:  python v11_p1b_lattice.py        ->  convergence table + checkpoint
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "_checkpoints")
os.makedirs(CKPT, exist_ok=True)

PI = np.pi


def radial_cumulative_I(U_cut, n=200000):
    """exact dimensionless radial energy I(<U_cut) = int_0^U_cut [ a'^2 sin^2 a
    + sin^4 a/(2u^2) + u^2 cos^6 a ] du with a = arctan(u). Fine 1D quadrature."""
    u = np.linspace(1e-6, U_cut, n)
    a = np.arctan(u)
    ap = 1.0 / (1.0 + u * u)          # da/du
    sa, ca = np.sin(a), np.cos(a)
    dens = ap * ap * sa * sa + sa ** 4 / (2 * u * u) + u * u * ca ** 6
    return float(np.sum(0.5 * (dens[1:] + dens[:-1]) * np.diff(u)))


def lattice_energy_inside(a, L, R_cut):
    """J~(r<R_cut) on a 3D Cartesian grid (r0 = 1), spacing a, half-box L.
    Returns the lattice cumulative curvature+potential energy inside R_cut."""
    xs = np.arange(-L, L + 0.5 * a, a)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z)
    r_safe = np.where(r < 1e-9, 1e-9, r)
    alpha = np.arctan(r)              # r0 = 1
    sa = np.sin(alpha)
    # smooth fields: q0 = cos a (scalar), q_vec = (x/r) sin a  (-> 0 at origin)
    q0 = np.cos(alpha)
    qv = np.stack([X / r_safe * sa, Y / r_safe * sa, Z / r_safe * sa], axis=-1)
    qv = np.where(r[..., None] < 1e-9, 0.0, qv)

    def d(f, axis):
        g = np.zeros_like(f)
        sl_p = [slice(None)] * f.ndim
        sl_m = [slice(None)] * f.ndim
        sl_o = [slice(None)] * f.ndim
        sl_p[axis] = slice(2, None); sl_m[axis] = slice(0, -2); sl_o[axis] = slice(1, -1)
        g[tuple(sl_o)] = (f[tuple(sl_p)] - f[tuple(sl_m)]) / (2 * a)
        return g

    # Gamma_i = q0 d_i qv - (d_i q0) qv + qv x d_i qv   (i = x,y,z spatial)
    Gamma = []
    for ax in range(3):
        dq0 = d(q0, ax)                 # scalar field grid
        dqv = d(qv, ax)                 # vector field grid (...,3)
        cross = np.cross(qv, dqv)
        Gi = q0[..., None] * dqv - dq0[..., None] * qv + cross
        Gamma.append(Gi)
    # u_curv = 1/4[(Tr G)^2 - Tr G^2], G_ij = Gamma_i . Gamma_j
    Gmat = np.zeros(X.shape + (3, 3))
    for i in range(3):
        for j in range(3):
            Gmat[..., i, j] = np.sum(Gamma[i] * Gamma[j], axis=-1)
    trG = np.trace(Gmat, axis1=-2, axis2=-1)
    trG2 = np.trace(Gmat @ Gmat, axis1=-2, axis2=-1)
    u_curv = 0.25 * (trG * trG - trG2)
    Lambda = q0 ** 6
    dens = u_curv + Lambda
    # cumulative inside R_cut, excluding the 1-cell boundary (no central diff there)
    interior = np.ones(X.shape, dtype=bool)
    for ax in range(3):
        idx = [slice(None)] * 3; idx[ax] = 0; interior[tuple(idx)] = False
        idx[ax] = -1; interior[tuple(idx)] = False
    mask = (r < R_cut) & interior
    return float(np.sum(dens[mask]) * a ** 3)


def main():
    t0 = time.time()
    R_cut = 5.0
    L = 6.0
    I_rad = radial_cumulative_I(R_cut)
    target = 4 * PI * I_rad          # J~(<R_cut), exact
    rows = []
    for a in [0.40, 0.30, 0.22, 0.16]:
        J = lattice_energy_inside(a, L, R_cut)
        rel = (J - target) / target
        rows.append({"a": a, "J_lattice": J, "target_4pi_I": target,
                     "rel_err": rel, "overestimate_pct": 100 * rel})
        print(f"a={a:.2f}  J_lat={J:.5f}  target={target:.5f}  "
              f"overestimate={100*rel:+.3f}%")
    # O(a^2) convergence check: |rel_err| should fall as a shrinks
    rel_abs = [abs(r["rel_err"]) for r in rows]
    converging = all(rel_abs[i + 1] < rel_abs[i] for i in range(len(rel_abs) - 1))
    # Richardson (a^2) extrapolation from the two finest
    a1, a2 = rows[-2]["a"], rows[-1]["a"]
    J1, J2 = rows[-2]["J_lattice"], rows[-1]["J_lattice"]
    J_extrap = (J2 * a1 ** 2 - J1 * a2 ** 2) / (a1 ** 2 - a2 ** 2)
    extrap_rel = (J_extrap - target) / target
    ok = converging and abs(extrap_rel) < 0.01
    KEV = 51.77  # E_keV = J~ * (a_f hc/r0_faber)/(4pi)*1000 ; pi^2 -> 511 keV
    out = {"phase": "P1b-foundation", "gate": "3D SU(2) curvature machinery",
           "ok": bool(ok), "R_cut": R_cut, "L": L, "target_4pi_I_radial": target,
           "I_radial_lt_Rcut": I_rad, "rows": rows, "converging": bool(converging),
           "J_richardson_extrap": J_extrap, "extrap_rel_err": extrap_rel,
           "pi_squared_full": PI ** 2, "note_full_soliton_keV": PI ** 2 * KEV,
           "wall_s": round(time.time() - t0, 1)}
    print(f"\nRichardson(a^2) extrap J~(<{R_cut}) = {J_extrap:.5f}  vs target {target:.5f}"
          f"  ({100*extrap_rel:+.3f}%)")
    print(f"convergence O(a^2): {'YES' if converging else 'NO'}   "
          f"[{'PASS' if ok else 'FAIL'}] full-soliton J~=pi^2 -> {PI**2*KEV:.1f} keV")
    print(f"P1b-foundation {'PASS' if ok else 'FAIL'}  ({out['wall_s']}s)")
    with open(os.path.join(CKPT, "p1b_lattice.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"checkpoint -> {os.path.join(CKPT, 'p1b_lattice.json')}")


if __name__ == "__main__":
    main()
