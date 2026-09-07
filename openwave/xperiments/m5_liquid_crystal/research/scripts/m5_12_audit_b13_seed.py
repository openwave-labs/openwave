"""M5.12 block-13 ADVERSARIAL AUDIT, script 4: seed-independence + solver
adjoint (L4, plus the chain-dependence attack on L1).

Attack A (chain artifact?): build a FRESH bmix seed (the documented breather
class, NOT the warm-start chain), rescale its first harmonic to the r4 rung
a2 = 0.30372, evaluate S0/Q2/omega_bal at seed level with the INDEPENDENT
functional, then relax it with MY OWN minimal hard solver (same math as the
claimant: omega eliminated by the closed form, retraction, LSMR
Gauss-Newton; freshly typed) and compare omega_bal against the chain's r4
endpoint 15.51. Gross disagreement after relaxation = the "law" is a
chain artifact; agreement = seed-independence survives at that rung.

Attack B (linear algebra): the claimant's LinearOperator uses
matvec = FD of the composed map but rmatvec = symmetric-Hessian FD +
rank-one + phase. LSMR requires a consistent adjoint pair. Test
<A v, u> == <v, A^T u> on random vectors at the r4 endpoint, with the
claimant's exact scaling (Dscale) replicated.

Run: /opt/anaconda3/envs/master312/bin/python m5_12_audit_b13_seed.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "..", "data")

ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

import m5_12_audit_b11_lib as lib                                  # noqa: E402
from m5_17_energy import cell_weights, grid_coords                 # noqa: E402
from m5_16_axisym import hedgehog_field, pin_mask                  # noqa: E402
from m5_12_dressed import to_covariant                             # noqa: E402
from m5_12_d3a_bvp import residual, shat                           # noqa: E402
from m5_12_d3b_newton import wscale_at                             # noqa: E402

NR, NZ, H, NT = 32, 64, 1.0, 2
WSCALE = wscale_at(NR, NZ, H, 8.0 * NR / 96)
A2_R4 = 0.30372464684878014
CHAIN_R4 = {"omega": 15.507528337510594, "S0": 44.34207040782907,
            "Q2": -0.18438721312449502}


def fresh_seed():
    scale = NR / 96.0
    rc, w_b = 8.0 * scale, 8.0 * scale
    R, Z = grid_coords(NR, NZ, H)
    M4 = to_covariant(hedgehog_field(R, Z, rc))
    r2 = R ** 2 + Z ** 2
    rr = np.sqrt(r2) + 1e-12
    b2 = np.exp(-r2 / (w_b ** 2))
    A1 = np.zeros_like(M4)
    A1[..., 0, 1] = A1[..., 1, 0] = b2 * R / rr        # eps = 1, rescaled below
    A1[..., 0, 3] = A1[..., 3, 0] = b2 * Z / rr
    X = {"M0": M4, "A": [A1, np.zeros_like(M4)],
         "B": [np.zeros_like(M4), np.zeros_like(M4)]}
    pin = pin_mask(NR, NZ)
    for k in range(NT):
        X["A"][k][pin] = 0.0
        X["B"][k][pin] = 0.0
    return X, pin


# ---------- my own flat-vector plumbing (fresh typed) ----------
class Flat:
    def __init__(self, X, free):
        self.free = free
        self.n = int(np.sum(free))
        self.Nt = len(X["A"])

    def pack(self, X):
        ps = [X["M0"][self.free]]
        for k in range(self.Nt):
            ps += [X["A"][k][self.free], X["B"][k][self.free]]
        return np.concatenate(ps)

    def unpack(self, v, X0):
        X = {"M0": X0["M0"].copy(), "A": [a.copy() for a in X0["A"]],
             "B": [b.copy() for b in X0["B"]]}
        o = 0
        X["M0"][self.free] = v[o:o + self.n]; o += self.n
        for k in range(self.Nt):
            X["A"][k][self.free] = v[o:o + self.n]; o += self.n
            X["B"][k][self.free] = v[o:o + self.n]; o += self.n
        return X

    def rpack(self, Rd, extra):
        ps = [np.where(self.free, Rd["M0"], 0.0)[self.free]]
        for k in range(self.Nt):
            ps += [np.where(self.free, Rd["A"][k], 0.0)[self.free],
                   np.where(self.free, Rd["B"][k], 0.0)[self.free]]
        return np.concatenate(ps + [np.asarray(extra)])


def omega_bal_solverquad(X):
    """the solver's own quadrature (d3a shat, ns=10), for solve parity."""
    s0 = shat(X, 0.0, H, WSCALE)
    q2 = s0 - shat(X, 1.0, H, WSCALE)
    if q2 >= 0 or s0 <= 0:
        return None, s0, q2
    return float(np.sqrt(s0 / -q2)), s0, q2


def relax(X0, free, a2_star, iters=4, lsmr_iters=60):
    V = Flat(X0, free)
    n1 = V.n
    U = X0["A"][0][free]
    U = U / (np.linalg.norm(U) + 1e-300)

    def retract(v):
        a2 = float(np.sum(v[n1:3 * n1] ** 2))
        vv = v.copy()
        vv[n1:3 * n1] *= np.sqrt(a2_star / a2)
        return vv

    def F_of(v):
        X = V.unpack(v, X0)
        w, s0, q2 = omega_bal_solverquad(X)
        if w is None:
            return None, (None, s0, q2)
        Rd, _ = residual(X, w, H, WSCALE)
        c_ph = float(np.sum(X["B"][0][free] * U))
        return V.rpack(Rd, [c_ph]), (w, s0, q2)

    w_cells = cell_weights(NR, NZ, H)
    wfull = np.ones(X0["M0"].shape[:2])
    wfull[: NR - 1, 1:-1] = w_cells
    dblk = np.sqrt(np.broadcast_to(wfull[..., None, None],
                                   X0["M0"].shape)[free])
    Dscale = np.concatenate([dblk] * (1 + 2 * NT))

    z = retract(V.pack(X0))
    F, meta = F_of(z)
    fn0 = float(np.linalg.norm(F))
    traj = [{"iter": 0, "F": fn0, "omega_bal": meta[0], "S0": meta[1],
             "Q2": meta[2]}]
    print(f"  [relax open] |F|0={fn0:.3e} w={meta[0]:.4f} S0={meta[1]:.4f} "
          f"Q2={meta[2]:+.5f}")
    for it in range(1, iters + 1):
        t0 = time.time()
        X = V.unpack(z, X0)
        w, s0, q2 = meta
        R0d, _ = residual(X, 0.0, H, WSCALE)
        R1d, _ = residual(X, 1.0, H, WSCALE)
        R0v = V.rpack(R0d, [0.0])[:-1]
        RQv = R0v - V.rpack(R1d, [0.0])[:-1]
        P = -q2
        gvec = (R0v / P + (s0 / P ** 2) * RQv) / (2.0 * w)
        dRdw = -2.0 * w * RQv
        gph = np.zeros_like(z)
        gph[2 * n1:3 * n1] = U
        eps = 1e-6 * max(1.0, float(np.linalg.norm(z))) / np.sqrt(z.size)

        def jv(vv):
            Fp, _ = F_of(z + eps * vv)
            Fm, _ = F_of(z - eps * vv)
            if Fp is None or Fm is None:
                return np.full(F.size, np.nan)
            return (Fp - Fm) / (2 * eps)

        def rmv(ww):
            wR, wph = ww[:-1], ww[-1]
            Xp = V.unpack(z + eps * wR, X0)
            Xm = V.unpack(z - eps * wR, X0)
            Rp, _ = residual(Xp, w, H, WSCALE)
            Rm, _ = residual(Xm, w, H, WSCALE)
            Hw = (V.rpack(Rp, [0.0])[:-1] - V.rpack(Rm, [0.0])[:-1]) / (2 * eps)
            return Hw + gvec * float(np.dot(dRdw, wR)) + wph * gph

        A = LinearOperator((F.size, z.size),
                           matvec=lambda vv: jv(vv / Dscale),
                           rmatvec=lambda ww: rmv(ww) / Dscale, dtype=float)
        sol = lsmr(A, -F, maxiter=lsmr_iters, atol=1e-8, btol=1e-8)
        s = sol[0] / Dscale
        lam, ok = 1.0, False
        fn = float(np.linalg.norm(F))
        for _ in range(6):
            zt = retract(z + lam * s)
            Ft, mt = F_of(zt)
            if Ft is not None and float(np.linalg.norm(Ft)) < fn:
                z, F, meta, ok = zt, Ft, mt, True
                break
            lam *= 0.5
        traj.append({"iter": it, "F": float(np.linalg.norm(F)),
                     "rel": float(np.linalg.norm(F)) / fn0,
                     "omega_bal": meta[0], "S0": meta[1], "Q2": meta[2],
                     "lam": lam, "accepted": ok,
                     "wall_s": round(time.time() - t0, 1)})
        print(f"  [relax {it}] |F|={traj[-1]['F']:.3e} rel="
              f"{traj[-1]['rel']:.2e} w={meta[0]:.4f} Q2={meta[2]:+.5f} "
              f"lam={lam:g} ok={ok} wall={traj[-1]['wall_s']}s")
        if not ok:
            break
    return traj


def adjoint_test():
    """Attack B on the claimant operator at the r4 endpoint."""
    X0, _ = lib.load_state(os.path.join(DATA, "m5_12_b12_hard_r4_state.npz"))
    free, pin = lib.free_mask(NR, NZ, X0["M0"].shape)
    V = Flat(X0, free)
    n1 = V.n
    U = X0["A"][0][free]
    U = U / (np.linalg.norm(U) + 1e-300)
    z = V.pack(X0)

    def F_of(v):
        X = V.unpack(v, X0)
        w, s0, q2 = omega_bal_solverquad(X)
        Rd, _ = residual(X, w, H, WSCALE)
        c_ph = float(np.sum(X["B"][0][free] * U))
        return V.rpack(Rd, [c_ph]), (w, s0, q2)

    F, (w, s0, q2) = F_of(z)
    R0d, _ = residual(V.unpack(z, X0), 0.0, H, WSCALE)
    R1d, _ = residual(V.unpack(z, X0), 1.0, H, WSCALE)
    R0v = V.rpack(R0d, [0.0])[:-1]
    RQv = R0v - V.rpack(R1d, [0.0])[:-1]
    P = -q2
    gvec = (R0v / P + (s0 / P ** 2) * RQv) / (2.0 * w)
    dRdw = -2.0 * w * RQv
    gph = np.zeros_like(z)
    gph[2 * n1:3 * n1] = U
    eps = 1e-6 * max(1.0, float(np.linalg.norm(z))) / np.sqrt(z.size)

    rng = np.random.default_rng(31)
    out = []
    for trial in range(3):
        v = rng.standard_normal(z.size)
        u = rng.standard_normal(F.size)
        Fp, _ = F_of(z + eps * v)
        Fm, _ = F_of(z - eps * v)
        Av = (Fp - Fm) / (2 * eps)
        wR, wph = u[:-1], u[-1]
        Xp = V.unpack(z + eps * wR, X0)
        Xm = V.unpack(z - eps * wR, X0)
        Rp, _ = residual(Xp, w, H, WSCALE)
        Rm, _ = residual(Xm, w, H, WSCALE)
        Hw = (V.rpack(Rp, [0.0])[:-1] - V.rpack(Rm, [0.0])[:-1]) / (2 * eps)
        Atu = Hw + gvec * float(np.dot(dRdw, wR)) + wph * gph
        lhs = float(np.dot(Av, u))
        rhs = float(np.dot(v, Atu))
        rel = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-300)
        out.append({"lhs_Av_u": lhs, "rhs_v_Atu": rhs, "rel": rel})
        print(f"  [adjoint {trial}] <Av,u>={lhs:+.6e} <v,A'u>={rhs:+.6e} "
              f"rel={rel:.3e}")
    return out


def main():
    res = {"wscale": WSCALE}
    print("== Attack B: adjoint consistency of the claimant operator (r4) ==")
    res["L4_adjoint_r4"] = adjoint_test()

    print("== Attack A: fresh bmix seed at the r4 amplitude ==")
    X, pin = fresh_seed()
    free = np.broadcast_to((~pin)[..., None, None]
                           & np.ones((1, 1, 4, 4), bool), X["M0"].shape)
    a2_unit = float(np.sum(X["A"][0][free] ** 2))
    eps_star = np.sqrt(A2_R4 / a2_unit)
    X["A"][0] *= eps_star
    print(f"  a2(eps=1)={a2_unit:.4f} -> eps*={eps_star:.4f} for a2={A2_R4}")
    S0, Q2, Q2m = lib.my_s0_q2(X, H, WSCALE)
    w0 = float(np.sqrt(S0 / -Q2)) if Q2 < 0 else None
    res["L4_fresh_seed_level"] = {
        "eps_star": eps_star, "a2": A2_R4, "S0": S0, "Q2": Q2,
        "Q2_mix": Q2m, "omega_bal": w0, "chain_r4": CHAIN_R4}
    print(f"  seed-level: S0={S0:.4f} Q2={Q2:+.5f} w_bal={w0} "
          f"(chain r4 endpoint: w={CHAIN_R4['omega']:.3f} "
          f"S0={CHAIN_R4['S0']:.3f} Q2={CHAIN_R4['Q2']:+.5f})")

    print("== Attack A: relax the fresh seed (my own solver, 4 steps) ==")
    traj = relax(X, free, A2_R4, iters=4, lsmr_iters=60)
    res["L4_fresh_seed_relax"] = traj
    with open(os.path.join(DATA, "m5_12_audit_b13_seed.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("-> m5_12_audit_b13_seed.json")


if __name__ == "__main__":
    main()
