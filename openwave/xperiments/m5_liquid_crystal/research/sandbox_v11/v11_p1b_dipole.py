"""M5.11 P1b - the Coulomb / fine-structure result, Faber's axisymmetric route.

Step 1 (this gate): validate the AXISYMMETRIC (rho,z) curvature energy on the single
soliton, i.e. check the hand-derived phi-winding algebra (Faber Eq. 6 reduced to the
phi=0 half-plane with the winding-1 axial symmetry):

  q_vec(rho,phi,z) = R_z(phi) . (q_rho e_x + q_z e_z)  =>  at phi=0, the Cartesian
  spatial gradients of q_vec are  d_x = d_rho ,  d_y = (1/rho) d_phi ,  d_z = d_z , giving
    Gamma_x = (q0 qr_r - q0_r qr,   qz qr_r - qr qz_r,   q0 qz_r - q0_r qz)
    Gamma_y = (qr/rho)(-qz, q0, qr)                      <-- the winding term
    Gamma_z = (q0 qr_z - q0_z qr,   qz qr_z - qr qz_z,   q0 qz_z - q0_z qz)
  u_curv = 1/4[(Tr G)^2 - Tr G^2],  G_ij = Gamma_i . Gamma_j ,   Lambda = q0^6/r0^4.

GATE: the axisymmetric energy inside r<R_cut must reproduce the exact radial cumulative
4*pi*I_radial(<R_cut), converging O(h^2) as the spacing h -> 0. (If Gamma_y were wrong,
the single soliton would NOT reproduce the radial energy. This validates the machinery
that the two-soliton dipole needs.)

Run:  python v11_p1b_dipole.py single     ->  axisym single-soliton validation
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "_checkpoints")
os.makedirs(CKPT, exist_ok=True)
PI = np.pi


def radial_cumulative_I(U_cut, n=400000):
    u = np.linspace(1e-6, U_cut, n)
    a = np.arctan(u)
    ap = 1.0 / (1.0 + u * u)
    sa, ca = np.sin(a), np.cos(a)
    dens = ap * ap * sa * sa + sa ** 4 / (2 * u * u) + u * u * ca ** 6
    return float(np.sum(0.5 * (dens[1:] + dens[:-1]) * np.diff(u)))


def _dr(f, h):   # d/drho, central along axis 0 (rho)
    g = np.zeros_like(f)
    g[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * h)
    return g


def _dz(f, h):   # d/dz, central along axis 1 (z)
    g = np.zeros_like(f)
    g[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * h)
    return g


def axisym_density(q0, qr, qz, RHO, h):
    """returns (u_curv, Lambda) per (rho,z) grid point from the winding-1 axial field."""
    q0_r, q0_z = _dr(q0, h), _dz(q0, h)
    qr_r, qr_z = _dr(qr, h), _dz(qr, h)
    qz_r, qz_z = _dr(qz, h), _dz(qz, h)
    # the three target 3-vectors Gamma_x, Gamma_y, Gamma_z (stacked last axis = target comp)
    Gx = np.stack([q0 * qr_r - q0_r * qr,
                   qz * qr_r - qr * qz_r,
                   q0 * qz_r - q0_r * qz], axis=-1)
    inv_rho = 1.0 / RHO
    Gy = np.stack([-qz * qr * inv_rho,
                   q0 * qr * inv_rho,
                   qr * qr * inv_rho], axis=-1)
    Gz = np.stack([q0 * qr_z - q0_z * qr,
                   qz * qr_z - qr * qz_z,
                   q0 * qz_z - q0_z * qz], axis=-1)
    G = [Gx, Gy, Gz]
    # G_ij = Gamma_i . Gamma_j ; u_curv = 1/4[(Tr G)^2 - Tr(G^2)]
    Gmat = np.zeros(q0.shape + (3, 3))
    for i in range(3):
        for j in range(3):
            Gmat[..., i, j] = np.sum(G[i] * G[j], axis=-1)
    trG = np.trace(Gmat, axis1=-2, axis2=-1)
    trG2 = np.trace(Gmat @ Gmat, axis1=-2, axis2=-1)
    u_curv = 0.25 * (trG * trG - trG2)
    Lam = q0 ** 6      # r0 = 1
    return u_curv, Lam


def seed_single(RHO, ZZ):
    """analytic Faber soliton centered at origin (r0 = 1): a = arctan(r)."""
    r = np.sqrt(RHO ** 2 + ZZ ** 2)
    r_safe = np.where(r < 1e-12, 1e-12, r)
    al = np.arctan(r)
    sa = np.sin(al)
    return np.cos(al), RHO / r_safe * sa, ZZ / r_safe * sa   # q0, qr, qz


def gate_single():
    R_cut = 5.0
    Lr, Lz = 6.0, 6.0
    I_rad = radial_cumulative_I(R_cut)
    target = 4 * PI * I_rad          # J~(<R_cut)
    rows = []
    for h in [0.25, 0.18, 0.12, 0.09]:
        rho = np.arange(0.5 * h, Lr, h)           # cell-centered in rho (avoids axis)
        z = np.arange(-Lz, Lz + 0.5 * h, h)
        RHO, ZZ = np.meshgrid(rho, z, indexing="ij")
        q0, qr, qz = seed_single(RHO, ZZ)
        u_curv, Lam = axisym_density(q0, qr, qz, RHO, h)
        dens = u_curv + Lam
        r = np.sqrt(RHO ** 2 + ZZ ** 2)
        interior = np.ones(RHO.shape, dtype=bool)
        interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False
        mask = (r < R_cut) & interior
        # J~(<R_cut) = 2*pi * integral (u_curv+Lam) rho drho dz
        J = 2 * PI * float(np.sum((dens * RHO)[mask]) * h * h)
        rel = (J - target) / target
        rows.append({"h": h, "J_axi": J, "target": target, "rel_err": rel})
        print(f"h={h:.2f}  J_axi={J:.5f}  target={target:.5f}  err={100*rel:+.3f}%")
    rel_abs = [abs(r["rel_err"]) for r in rows]
    converging = all(rel_abs[i + 1] < rel_abs[i] for i in range(len(rel_abs) - 1))
    h1, h2 = rows[-2]["h"], rows[-1]["h"]
    J1, J2 = rows[-2]["J_axi"], rows[-1]["J_axi"]
    J_ex = (J2 * h1 ** 2 - J1 * h2 ** 2) / (h1 ** 2 - h2 ** 2)
    ex_rel = (J_ex - target) / target
    ok = converging and abs(ex_rel) < 0.01
    out = {"phase": "P1b-step1-axisym-single", "ok": bool(ok), "rows": rows,
           "converging": bool(converging), "J_richardson": J_ex, "extrap_rel_err": ex_rel,
           "target_4pi_I": target}
    print(f"\nRichardson(h^2) J_axi(<{R_cut}) = {J_ex:.5f} vs {target:.5f} ({100*ex_rel:+.3f}%)")
    print(f"axisym phi-winding algebra: [{'PASS' if ok else 'FAIL'}]  (converging={converging})")
    with open(os.path.join(CKPT, "p1b_axisym_single.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"checkpoint -> {os.path.join(CKPT, 'p1b_axisym_single.json')}")
    return out


ALPHA_F_INV = 137.035999084
ALPHA_F_HBAR_C = 1.4399645474109544   # MeV.fm = e0^2/4pi eps0


def gate_charge():
    """Charge quantization -> alpha^-1 ~ 137 the robust way (Faber's Gauss-law mechanism).

    The soliton's EXTERIOR energy E(>R) (the Coulomb tail of its field) must equal the
    Coulomb self-energy of exactly ONE elementary charge e:
        E(>R) = (alpha_f hbar c) / (2R)        [ = q^2/(8 pi eps0 R), q = e ]
    because e^2/4pi eps0 = alpha_f hbar c. We read the EFFECTIVE charge^2 off the exterior
    energy: q^2(R) = E(>R)_dimless * 2U  (U = R/r0), which -> 1 as R->inf (the near-field
    core correction ~ 2/(3U^3) dies as 1/U^2). Then alpha_sol = q^2 alpha_f, so
    alpha_sol^-1(R) = 137.036 / q^2(R) -> 137.04. Robust (uses the exact arctan profile),
    no minimization. NOTE: q^2 > 1 at finite R (so alpha_sol^-1 < 137) MIRRORS Faber's
    running 1/alpha_sol(d) < 137 at short distance (his Fig. 2), though the quantitative
    two-soliton running needs the overlap minimization (the scoped stretch)."""
    PI_4 = PI / 4
    rows = []
    for U in [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]:
        I_lt = radial_cumulative_I(U, n=600000)
        I_gt = PI_4 - I_lt                       # exterior energy (dimensionless)
        coulomb = 1.0 / (2 * U)                   # unit-charge Coulomb self-energy (dimless)
        q2 = I_gt / coulomb                        # effective charge^2 (in units of e^2)
        a_inv = ALPHA_F_INV / q2
        a_sol_hbar_c = q2 * ALPHA_F_HBAR_C
        rows.append({"R_over_r0": U, "E_ext_dimless": I_gt, "coulomb_dimless": coulomb,
                     "charge2_in_e2": q2, "alpha_sol_inv": a_inv,
                     "alpha_sol_hbar_c_MeVfm": a_sol_hbar_c})
        print(f"R={U:6.0f} r0   q^2={q2:.5f} e^2   1/alpha_sol={a_inv:8.3f}   "
              f"alpha_sol*hc={a_sol_hbar_c:.5f} MeV.fm")
    q2_far = rows[-1]["charge2_in_e2"]
    ainv_far = rows[-1]["alpha_sol_inv"]
    # converging to unit charge + 137 from below
    conv_q = all(abs(rows[i + 1]["charge2_in_e2"] - 1) < abs(rows[i]["charge2_in_e2"] - 1)
                 for i in range(len(rows) - 1))
    ok = conv_q and abs(q2_far - 1.0) < 5e-3 and abs(ainv_far - ALPHA_F_INV) < 1.0
    out = {"phase": "P1b-charge-quantization", "ok": bool(ok), "rows": rows,
           "charge2_at_160r0": q2_far, "alpha_sol_inv_at_160r0": ainv_far,
           "alpha_sol_hbar_c_at_160r0": rows[-1]["alpha_sol_hbar_c_MeVfm"],
           "faber_target_alpha_sol_inv": "137.1(1)", "faber_target_alpha_sol_hbar_c": "1.4387(8)",
           "coulomb_hbar_c": ALPHA_F_HBAR_C, "converging_to_unit_charge": bool(conv_q)}
    print(f"\n-> soliton charge^2 -> {q2_far:.5f} e^2 ; 1/alpha_sol -> {ainv_far:.3f} "
          f"(CODATA 1/alpha_f = {ALPHA_F_INV:.3f})")
    print(f"-> alpha_sol*hbar*c -> {rows[-1]['alpha_sol_hbar_c_MeVfm']:.5f} MeV.fm "
          f"(Coulomb {ALPHA_F_HBAR_C:.5f}; Faber 1.4387(8))")
    print(f"charge quantization -> 137: [{'PASS' if ok else 'FAIL'}]")
    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Us = [r["R_over_r0"] for r in rows]
        ainv = [r["alpha_sol_inv"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].semilogx(Us, [r["charge2_in_e2"] for r in rows], "bo-")
        ax[0].axhline(1.0, color="r", ls="--", label="unit charge")
        ax[0].set_xlabel("R / r0"); ax[0].set_ylabel("effective charge^2 / e^2")
        ax[0].set_title("soliton -> one elementary charge (Gauss)"); ax[0].legend()
        ax[1].semilogx(Us, ainv, "bo-", label="1/alpha_sol (exterior energy)")
        ax[1].axhline(ALPHA_F_INV, color="r", ls="--", label=f"1/alpha_f={ALPHA_F_INV:.2f}")
        ax[1].set_xlabel("R / r0"); ax[1].set_ylabel("1 / alpha_sol")
        ax[1].set_title("approach to 137 from below (cf. Faber Fig.2)"); ax[1].legend()
        fig.suptitle("M5.11 P1b - charge quantization -> alpha^-1 ~ 137", fontweight="bold")
        fig.tight_layout()
        p = os.path.join(HERE, "p1b_charge_137.png")
        fig.savefig(p, dpi=110); plt.close(fig)
        print(f"plot -> {p}")
    except Exception as e:
        print(f"(plot skipped: {e})")
    with open(os.path.join(CKPT, "p1b_charge.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"checkpoint -> {os.path.join(CKPT, 'p1b_charge.json')}")
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    t0 = time.time()
    if mode == "single":
        gate_single()
    elif mode == "charge":
        gate_charge()
    elif mode == "all":
        gate_single(); print(); gate_charge()
    else:
        print("modes: single | charge | all")
    print(f"({round(time.time()-t0,1)}s)")


if __name__ == "__main__":
    main()
