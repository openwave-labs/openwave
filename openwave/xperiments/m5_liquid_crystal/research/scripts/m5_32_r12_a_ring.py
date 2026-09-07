"""M5.32 R12 (producer): the charged disclination RING as the electron object.

The author's 2026-08-29 reply names the acceptable objects for a quantized
charge in the biaxial field: "central topological charge 1 in cross-section
(2D), or two 1/2 with charged ring as in your previous simulations". R9 / R10
established that in this order-parameter space (stabilizer of d4 = Klein
four-group, pi_1 = Q8, pi_2 = 0) the protected objects are LINES, and that
the point hedgehog's degree lives on a discontinuity. The ring is the object
whose protection is real here: a closed half-disclination cord (pi_1 element
of order 4) with the hedgehog's far field. The record already holds its seed
(production `engine1_seeds.seed_charged_ring_M`, M5.21.2, ported to numpy
below, same formulas) and its statics verdict "competitive with the point,
instrument-limited" (M5.21.2 census, fwd stencil, free boundary, pre-R10).

EQUATIONS FIRST
---------------
Lattice, action, relaxation: the R10 protocol verbatim (INS4 = the certified
4D stack `m5_21_3_a_4d.py`): cfg = base_cfg(s = -1, g = 8, delta = 0.3),
h = 1.5, vacuum d4 = diag(8, 1, 0.3, 0), E = E_u + V4 with
    E_u = 4 h^3 sum_cells sum_{i<j} <F_ij, F_ij>_eta,  F_ij = [A_i, A_j]_eta,
    V4  = h^3 W1 sum_cells sum_p (tr((M eta)^p) - C_p)^2,
FIRE on the free cells (~pin_shell, depth 1.6), a0 = None, omega = 0,
dt0 = 0.01, dt_max = 0.1, max_iter in {1500, 3000}.
The ring seed (physical units; a = ring radius, w_c = cord melt width, rho_c
= axis melt):
    psi = 1/2 [atan2(-z, rho - a) + atan2(-z, rho + a)] + pi/2
    n = sin(psi) rho_hat + cos(psi) z_hat        (-> r_hat far, -> +-z inside)
    e_phi = azimuth * smoothstep(rho / rho_c) projected perpendicular to n
    e_theta = e_phi x n
    D = (1, delta, 0) on (n, e_theta, e_phi), melted to isotropic
        d_iso = (1 + delta)/3 at the cord: d_k(dc) = d_iso + S(dc/w_c)(D_k - d_iso),
        dc = sqrt((rho - a)^2 + z^2), S = smoothstep
    M = diag(g, M_sp) (static embedding, M_0i = 0)
The clock generator (the M5.21.9 isorotation, generalized so the same rule
serves the hedgehog and the ring): rotation of the internal frame about the
LOCAL leading eigenvector n1(x) of the spatial block,
    a0 = J M - M J,  J_ab = eps_abc n1_c (spatial block), J_0a = 0,
    selftest: on the hedgehog family this equals B8.a0_unit (the record's a0).
Two clock conventions (Q60): RIGID a0, and TAPERED a0 w(r), w = 1 for r <= 12,
linear to 0 at r = 15 (the R10 audit's taper).
    kin = INS4.kin_of(M, a0, cfg) = 4 h^3 sum_i <[a0, A_i]_eta>^2_eta
    fixed J: omega* = J / (2 kin),  E_J = E_stat + J^2 / (4 kin)
Cross-section winding (the M5.19 instrument, meridional plane y = 0): on a
circle of radius r_w about the cord point (a_meas, 0), theta = 1/2 atan2(2
M_xz, M_xx - M_zz) accumulated mod pi/2 on 2 theta; q = total / (4 pi)
(q = 1/2 = a director half-turn). The cord is located as the minimum of the
biaxiality gap (l1 - l2) of the spatial block along z = 0, rho > h.
Far-field charge: the record's leading-eigenvector reader
(m5_22_e_audit.read_charge_from_M) on the cube surface at index 5..n-6,
reported as what it is (R10: not an invariant of this space).

PRE-REGISTERED HYPOTHESES (before any number here)
--------------------------------------------------
H12-a  The seed carries q = 1/2 on three meridional circles (r_w = 2.25, 3,
       3.75) about the cord at both a = 6 and a = 9, reads far-sphere degree
       +-1 with the record's reader, and the generalized a0 matches
       B8.a0_unit on the hedgehog to 1e-8 relative (amended at the first
       selftest read, before any ring number: pointwise up to the eigenvector
       SIGN, i.e. |a0| and kin agree; kin is pointwise quadratic in a0).
H12-b  Under pure statics the ring SHRINKS (cord tension, nothing static sets
       a scale but the core width): a_meas(3000) < a_meas(0) at both seeds,
       monotone across the 1500 / 3000 ladder; the half-winding survives on
       the relaxed cord while a_meas > 2 h (protection is against unwinding,
       not against contraction). Registered alternative: the ring holds at a
       finite radius (|a(3000) - a(1500)| < h): then the ring is a static
       minimum competing with the point, and G6-style existence is met.
H12-c  Clock: the RIGID inertia of the relaxed ring is of the hedgehog's
       order (351 at n 32 L 48) and grows with L; the TAPERED inertia is
       L-independent to < 10 % (by construction, R10) and its shell profile
       peaks at the cord. Fixed J along the seed ladder a in {3, 4.5, 6, 9,
       12}: E_J(a) = E_stat(a) + J^2 / (4 kin_tap(a)); an interior a* exists
       only if kin_tap grows faster than E_stat falls; PREDICT no interior
       minimum with omega* inside the radiation window (omega* < 0.786, R8)
       at J in {50, 200, 800}.

Outputs: ../data/m5_32_r12_ring.json, ../plots/m5_32_r12_ring.png,
         ../checkpoints/m5_32_r12/*.npy (local, gitignored)
Run: nice -n 10 /opt/anaconda3/envs/master312/bin/python3 m5_32_r12_a_ring.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
DATA, PLOTS = os.path.join(RES, "data"), os.path.join(RES, "plots")
CK = os.path.join(RES, "checkpoints", "m5_32_r12")
os.makedirs(CK, exist_ok=True)
OUT = os.path.join(DATA, "m5_32_r12_ring.json")
PNG = os.path.join(PLOTS, "m5_32_r12_ring.png")
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


INS4 = _load("m5_21_3_a_4d", "m5_21_3_a_4d.py")
B8 = _load("m5_21_8_b_lattice", "m5_21_8_b_lattice.py")
E22 = _load("m5_22_e_audit", "m5_22_e_audit.py")
G, DELTA, S = 8.0, 0.3, -1.0
RW = (2.25, 3.0, 3.75)
JS = (50.0, 200.0, 800.0)
OMEGA_RAD = 0.786


def cfg_of(n, L):
    return INS4.base_cfg(s=S, g=G, n=n, L=float(L), delta=DELTA)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ------------------------------ the ring seed ------------------------------
def ring_seed(cfg, a, w_c=3.0, rho_c=3.0):
    n, h = cfg["n"], cfg["h"]
    X, Y, Z = INS4.coords(n, h)
    rho = np.sqrt(X * X + Y * Y)
    ainv = 1.0 / np.sqrt(rho * rho + 1e-12)
    rx, ry = X * ainv, Y * ainv
    psi = 0.5 * (np.arctan2(-Z, rho - a) + np.arctan2(-Z, rho + a)) + 0.5 * np.pi
    sp, cp = np.sin(psi), np.cos(psi)
    nvec = np.stack([sp * rx, sp * ry, cp], axis=-1)
    azim = np.stack([-ry, rx, np.zeros_like(rx)], axis=-1)
    shrink = smoothstep(rho / rho_c)[..., None]
    ephi = azim * shrink
    ephi = ephi - np.sum(ephi * nvec, axis=-1, keepdims=True) * nvec
    etheta = np.cross(ephi, nvec)
    dc = np.sqrt((rho - a) ** 2 + Z * Z)
    smelt = smoothstep(dc / w_c)[..., None, None]
    d_iso = (1.0 + DELTA) / 3.0
    d0 = d_iso + smelt * (1.0 - d_iso)
    d1 = d_iso + smelt * (DELTA - d_iso)
    d2 = d_iso + smelt * (0.0 - d_iso)
    msp = (d0 * nvec[..., :, None] * nvec[..., None, :]
           + d1 * etheta[..., :, None] * etheta[..., None, :]
           + d2 * ephi[..., :, None] * ephi[..., None, :])
    M = np.zeros(X.shape + (4, 4))
    M[..., 1:, 1:] = msp
    M[..., 0, 0] = -cfg["sg"]
    return INS4.sym4(M)


# ------------------------------ the clock generator ------------------------------
def a0_local(M):
    """a0 = J M - M J, J the rotation generator about the local leading
    eigenvector of the spatial block."""
    w, V = np.linalg.eigh(M[..., 1:, 1:])
    n1 = V[..., :, -1]
    J = np.zeros(M.shape)
    J[..., 1, 2], J[..., 2, 1] = -n1[..., 2], n1[..., 2]
    J[..., 1, 3], J[..., 3, 1] = n1[..., 1], -n1[..., 1]
    J[..., 2, 3], J[..., 3, 2] = -n1[..., 0], n1[..., 0]
    return J @ M - M @ J


def taper(cfg, r_in=12.0, r_out=15.0):
    X, Y, Z = INS4.coords(cfg["n"], cfg["h"])
    r = np.sqrt(X * X + Y * Y + Z * Z)
    return np.clip((r_out - r) / (r_out - r_in), 0.0, 1.0)


def kin_pair(M, cfg):
    a0 = a0_local(M)
    w = taper(cfg)[..., None, None]
    return float(INS4.kin_of(M, a0, cfg)), float(INS4.kin_of(M, a0 * w, cfg))


def kin_shells(M, cfg, dr=3.0):
    """h^3-weighted rigid kin per radial shell (from the per-cell density)."""
    a0 = a0_local(M)
    h3 = cfg["h"] ** 3
    X, Y, Z = INS4.coords(cfg["n"], cfg["h"])
    r = np.sqrt(X * X + Y * Y + Z * Z)
    dens = np.zeros(M.shape[:3])
    for br, (A, wt) in INS4.a_fields(M, cfg).items():
        for i in range(3):
            F = INS4.comm_eta(a0, A[i])
            dens += wt * 4.0 * INS4.inner_eta(F, F)
    edges = np.arange(0.0, cfg["L"] / 2 + 1e-9, dr)
    return [{"r_lo": float(edges[i]), "r_hi": float(edges[i + 1]),
             "kin_shell": float(np.sum(dens[(r >= edges[i]) & (r < edges[i + 1])]) * h3)}
            for i in range(len(edges) - 1)]


# ------------------------------ the instruments ------------------------------
def cord_radius(M, cfg):
    """minimum of the biaxiality gap l1 - l2 along the +x axis at z = 0, rho > h."""
    n, h = cfg["n"], cfg["h"]
    c = n // 2
    w = np.linalg.eigvalsh(M[c:, c, c, 1:, 1:])       # x >= center row (y = z = center)
    gap = w[:, 2] - w[:, 1]
    X = INS4.coords(n, h)[0][c:, c, c]
    sel = X > h
    k = int(np.argmin(np.where(sel, gap, np.inf)))
    return float(X[k]), float(gap[k]), gap.tolist(), X.tolist()


def interp_block(M, cfg, x, z):
    """trilinear read of the (xx, xz, zz) entries at physical (x, 0, z)."""
    n, h = cfg["n"], cfg["h"]
    fx = x / h + (n - 1) / 2.0
    fz = z / h + (n - 1) / 2.0
    c = (n - 1) / 2.0
    i0, k0 = int(np.floor(fx)), int(np.floor(fz))
    tx, tz = fx - i0, fz - k0
    j = int(round(c))
    out = np.zeros(3)
    for di, wx in ((0, 1 - tx), (1, tx)):
        for dk, wz in ((0, 1 - tz), (1, tz)):
            ii, kk = np.clip(i0 + di, 0, n - 1), np.clip(k0 + dk, 0, n - 1)
            m = M[ii, j, kk]
            out += wx * wz * np.array([m[1, 1], m[1, 3], m[3, 3]])
    return out


def winding(M, cfg, a_meas, r_w, npts=720):
    ang = np.linspace(0.0, 2.0 * np.pi, npts, endpoint=True)
    xs, zs = a_meas + r_w * np.cos(ang), r_w * np.sin(ang)
    vals = np.array([interp_block(M, cfg, x, z) for x, z in zip(xs, zs)])
    m11, m13, m33 = vals[:, 0], vals[:, 1], vals[:, 2]
    aniso = np.sqrt((m11 - m33) ** 2 + 4.0 * m13 ** 2)
    if float(np.min(aniso)) < 0.02:
        return float("nan"), float(np.min(aniso))
    two_theta = np.arctan2(2.0 * m13, m11 - m33)
    dth = np.diff(two_theta)
    dth = (dth + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.sum(dth) / (4.0 * np.pi)), float(np.min(aniso))


def far_degree(M, cfg):
    n = cfg["n"]
    Q, ncf = E22.read_charge_from_M(M[..., 1:, 1:], 5, n - 6)   # the reader takes the spatial 3x3 block
    return float(Q), int(ncf)


def read_state(M, cfg, tag):
    a_meas, gap_min, gap_prof, xs = cord_radius(M, cfg)
    e_u, e_v = INS4.e_parts(M, cfg)
    ws = {str(rw): winding(M, cfg, a_meas, rw) for rw in RW}
    kr, kt = kin_pair(M, cfg)
    Q, ncf = far_degree(M, cfg)
    rec = {"tag": tag, "E_u": float(e_u), "V4": float(e_v), "E_stat": float(e_u + e_v),
           "a_meas": a_meas, "gap_min": gap_min, "gap_profile_x": xs, "gap_profile": gap_prof,
           "winding": {k: v[0] for k, v in ws.items()}, "aniso_min": {k: v[1] for k, v in ws.items()},
           "kin_rigid": kr, "kin_tapered": kt, "far_degree": Q, "far_degree_conflicts": ncf}
    log(f"{tag}: E_u {e_u:.4f} V4 {e_v:.4f} a_meas {a_meas:.2f} gap {gap_min:.3f} "
        f"q {[round(v[0], 3) if v[0] == v[0] else None for v in ws.values()]} kin {kr:.3f} / tap {kt:.3f} Q {Q:.2f}")
    return rec


# ------------------------------ stages ------------------------------
def stage_selftest():
    cfg = cfg_of(32, 48.0)
    Mh = B8.dressed(cfg, 0.0)
    a_rec = B8.a0_unit(cfg, 0.0)
    a_new = a0_local(Mh)
    rel = float(np.sqrt(np.sum((a_new - a_rec) ** 2)) / np.sqrt(np.sum(a_rec ** 2)))
    rel_neg = float(np.sqrt(np.sum((a_new + a_rec) ** 2)) / np.sqrt(np.sum(a_rec ** 2)))
    # the eigenvector n1 carries a per-cell sign, so a0 is defined up to a per-cell
    # sign; kin is pointwise quadratic in a0, so the physical comparison is |a0|
    rel_abs = float(np.sqrt(np.sum((np.abs(a_new) - np.abs(a_rec)) ** 2)) / np.sqrt(np.sum(a_rec ** 2)))
    kin_rec = float(INS4.kin_of(Mh, a_rec, cfg))
    kin_new = float(INS4.kin_of(Mh, a_new, cfg))
    Mr = ring_seed(cfg, 6.0)
    far = np.linalg.eigvalsh(Mr[2, 2, 2, 1:, 1:])
    out = {"a0_rel_diff_vs_record": min(rel, rel_neg), "sign_flipped": bool(rel_neg < rel),
           "a0_abs_rel_diff_vs_record": rel_abs, "kin_rel_diff": abs(kin_new - kin_rec) / kin_rec,
           "kin_record": kin_rec, "kin_generalized": kin_new,
           "ring_far_corner_spectrum": far.tolist(), "vacuum_spectrum": [0.0, DELTA, 1.0],
           "ring_far_corner_M00": float(Mr[2, 2, 2, 0, 0]),
           "ring_M0i_max": float(np.max(np.abs(Mr[..., 0, 1:])))}
    log(f"selftest: a0 rel diff {out['a0_rel_diff_vs_record']:.2e} (|a0| rel diff {rel_abs:.2e}, sign flipped {out['sign_flipped']}), "
        f"kin {kin_rec:.4f} vs {kin_new:.4f}; ring corner spectrum {far}")
    return out


def relax(cfg, M0, maxit, tag):
    npy = os.path.join(CK, f"{tag}_it{maxit}.npy")
    js = os.path.join(CK, f"{tag}_it{maxit}.json")
    if os.path.exists(npy) and os.path.exists(js):
        return np.load(npy), json.load(open(js))
    free = ~INS4.pin_shell(cfg["n"], cfg["h"])
    t0 = time.time()
    M, info = INS4.fire(M0, cfg, free, max_iter=maxit, log_every=500, tag=f"m5_32_r12_{tag}",
                        dt0=0.01, dt_max=0.1)
    rec = {"stop": info.get("stop"), "wall_s": round(time.time() - t0, 1), "maxit": maxit,
           "rel_move": float(np.sqrt(np.sum((M - M0) ** 2)) / max(np.sqrt(np.sum((M0 - INS4.vac4(cfg)) ** 2)), 1e-300))}
    np.save(npy, M)
    json.dump(rec, open(js, "w"), indent=1)
    return M, rec


def stage_statics():
    out = {}
    for n, L, a in ((32, 48.0, 6.0), (32, 48.0, 9.0), (48, 72.0, 6.0)):
        cfg = cfg_of(n, L)
        tag = f"n{n}_L{L:g}_a{a:g}"
        M0 = ring_seed(cfg, a)
        rec = {"n": n, "L": L, "h": cfg["h"], "a_seed": a, "seed": read_state(M0, cfg, tag + " seed")}
        for it in (1500, 3000):
            M, info = relax(cfg, M0, it, tag)
            r = read_state(M, cfg, f"{tag} it{it}")
            r.update(info)
            rec[f"it{it}"] = r
        rec["shells_it3000_rigid"] = kin_shells(M, cfg)
        out[tag] = rec
    return out


def stage_fixedj():
    cfg = cfg_of(32, 48.0)
    rows = []
    for a in (3.0, 4.5, 6.0, 9.0, 12.0):
        M = ring_seed(cfg, a)
        e_u, e_v = INS4.e_parts(M, cfg)
        kr, kt = kin_pair(M, cfg)
        rows.append({"a": a, "E_stat": float(e_u + e_v), "E_u": float(e_u), "V4": float(e_v),
                     "kin_rigid": kr, "kin_tapered": kt})
        log(f"fixedJ seed a={a}: E_stat {e_u + e_v:.4f} kin {kr:.3f} / tap {kt:.3f}")
    out = {"rows": rows, "J": {}}
    for J in JS:
        ej = [r["E_stat"] + J * J / (4.0 * r["kin_tapered"]) for r in rows]
        om = [J / (2.0 * r["kin_tapered"]) for r in rows]
        k = int(np.argmin(ej))
        interior = 0 < k < len(rows) - 1
        out["J"][str(J)] = {"E_J": ej, "omega_star": om, "argmin_a": rows[k]["a"], "interior": bool(interior),
                            "omega_at_min": om[k], "in_radiation_window": bool(om[k] < OMEGA_RAD)}
    return out


def gates(res):
    st = res["statics"]
    s = res["selftest"]
    ga = s["a0_abs_rel_diff_vs_record"] < 1e-8 and s["kin_rel_diff"] < 1e-10
    for tag, rec in st.items():
        qs = [v for v in rec["seed"]["winding"].values() if v == v]
        ga &= len(qs) == 3 and all(abs(abs(q) - 0.5) < 0.05 for q in qs)
        ga &= abs(abs(rec["seed"]["far_degree"]) - 1.0) < 0.05
    shrink = all(st[t]["it3000"]["a_meas"] < st[t]["seed"]["a_meas"] and st[t]["it3000"]["a_meas"] <= st[t]["it1500"]["a_meas"] for t in st)
    holds = all(abs(st[t]["it3000"]["a_meas"] - st[t]["it1500"]["a_meas"]) < st[t]["h"] for t in st)
    surv = {}
    for t in st:
        r = st[t]["it3000"]
        qs = [v for v in r["winding"].values() if v == v]
        surv[t] = bool(r["a_meas"] > 2 * st[t]["h"] and sum(abs(abs(q) - 0.5) < 0.1 for q in qs) >= 2)
    k32 = st["n32_L48_a6"]["it3000"]["kin_tapered"]
    k48 = st["n48_L72_a6"]["it3000"]["kin_tapered"]
    tap_drift = abs(k48 - k32) / max(abs(k32), 1e-300)
    rig_grows = st["n48_L72_a6"]["it3000"]["kin_rigid"] > st["n32_L48_a6"]["it3000"]["kin_rigid"]
    fj = res["fixedj"]["J"]
    no_interior_window = all(not (v["interior"] and v["in_radiation_window"]) for v in fj.values())
    return {"H12a": bool(ga), "H12b_shrinks": bool(shrink), "H12b_alt_holds": bool(holds),
            "H12b_winding_survives": surv, "H12c_tapered_drift": tap_drift,
            "H12c_tapered_L_independent": bool(tap_drift < 0.10), "H12c_rigid_grows": bool(rig_grows),
            "H12c_no_interior_min_in_window": bool(no_interior_window)}


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for tag, rec in res["statics"].items():
        for key, ls in (("seed", ":"), ("it1500", "--"), ("it3000", "-")):
            ax[0].plot(rec[key]["gap_profile_x"], rec[key]["gap_profile"], ls, label=f"{tag} {key}")
    ax[0].set_xlabel("x (z = 0)"); ax[0].set_ylabel("l1 - l2 (biaxiality gap)"); ax[0].set_title("cord location: gap minimum"); ax[0].legend(fontsize=6)
    for tag, rec in res["statics"].items():
        sh = rec["shells_it3000_rigid"]
        ax[1].plot([0.5 * (s["r_lo"] + s["r_hi"]) for s in sh], [s["kin_shell"] for s in sh], "o-", label=tag)
    ax[1].set_xlabel("r"); ax[1].set_ylabel("rigid kin per shell"); ax[1].set_title("clock inertia shells (relaxed 3000)"); ax[1].legend(fontsize=7)
    rows = res["fixedj"]["rows"]
    for J, v in res["fixedj"]["J"].items():
        ax[2].plot([r["a"] for r in rows], v["E_J"], "o-", label=f"E_J, J = {J}")
    ax[2].plot([r["a"] for r in rows], [r["E_stat"] for r in rows], "k--", label="E_stat")
    ax[2].set_xlabel("seed ring radius a"); ax[2].set_title("fixed-J energy along the seed ladder"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(PNG, dpi=110)


def main():
    res = {"task": "M5.32 R12", "protocol": "R10 (g 8, delta 0.3, h 1.5, FIRE dt0 0.01 dt_max 0.1, pin_shell 1.6)"}
    res["selftest"] = stage_selftest()
    json.dump(res, open(OUT, "w"), indent=1)
    res["fixedj"] = stage_fixedj()
    json.dump(res, open(OUT, "w"), indent=1)
    res["statics"] = stage_statics()
    res["gates"] = gates(res)
    res["runtime_s"] = time.time() - T0
    json.dump(res, open(OUT, "w"), indent=1)
    plot(res)
    log(f"DONE gates {json.dumps(res['gates'])}")


if __name__ == "__main__":
    main()
