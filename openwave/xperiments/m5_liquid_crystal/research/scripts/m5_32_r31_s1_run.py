"""M5.32 R31 = S1: the modes of the run (the functional and the reads live in m5_32_r31_s1_core.py).

EQUATIONS FIRST: see m5_32_r31_s1_core.py. This file holds the protocol:
    controls   R31-0, form level and known answers (minutes): (a) the reduced functional at zero new weights on the
               stored n 32 R23 field returns R23-1's energy and gradient to round-off; (b) the clock generator
               [G_r, M] equals B8.a0_unit on the m5_21_8 analytic family (R6's identity) and vanishes on the uniaxial
               seed exterior; (c) kin_cert >= 0 and the spectral term's kinetic part is zero along a0 (random
               block-diagonal fields); (d) Disc by eigenvalues equals the sympy trace polynomial at small g, the
               polynomial's cancellation noise at g = 8 (the deviation), Disc = 0 with zero gradient on the degenerate
               vacuum and on the uniaxial seed, K_D's closed form; (e) the FD gate of the reduced gradient with every
               term on, two boxes, two parameter points; (f) the hydrogen record's (9Z/8) Tr(dR dR) and the bundle's
               V13 B (ii) evaluated once on the stored R23 field (a record, not a gate, prior claim (i)).
    center     the R23 point on the n 24 box: the five seeds at J = 0 (cold), J_1 = 2 kin_rad(J = 0) omega_G, the
               ladder J_1, 2 J_1 per seed (warm), the closure dE_J*/dJ = omega* at J_1 on rad (R4's G6 gate), the
               cost table (wall per relaxation, cold and warm), the ranges of the new weights set from the relaxed
               states (each term at its range end equals the certified static energy of the relaxed bia state).
    sweep      the pool over the parameter points (the axes through the center first, then the Latin hypercube), one
               point one sequential job (15 relaxations, warm-started as the plan states), stopped by a wall-clock
               budget (--hours) that cuts points, never gates.
    second     the n 32 L 48 box on the candidate points (the same J values: the charge is the invariant).
    collect    the catalog, the five conditions and the three invariants per member, the three states per point, the
               ratio map, Koide's Q, the gate word per reading, the plot.
Parallel shape (the plan): a process pool over points, one thread per worker, no split inside a job at n 24, no GPU.
Out: ../data/m5_32_r31_s1.json (the record), ../data/m5_32_r31_s1/*.npz (local), ../plots/m5_32_r31_s1_map.png.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
PLOTS = os.path.join(HERE, "..", "plots")
OUT_JSON = os.path.join(DATA, "m5_32_r31_s1.json")
OUT_DIR = os.path.join(DATA, "m5_32_r31_s1")
T0 = time.time()


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    argv = sys.argv
    sys.argv = [argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


core = _load("m5_32_r31_s1_core", "m5_32_r31_s1_core.py")
B3, R21, R20, R0, C23 = core.B3, core.R21, core.R20, core.R0, core.C23
CENTER = {"g": 8.0, "delta": 0.3, "w1s": 25.0, "c6": 0.0, "lam_s": 0.0, "lam_sel": 0.0, "eps_sx": 0.0, "eps_sel": 0.0}
LIFT_EPS = {"eps_sx": (0.5, 2.0, 8.2), "eps_sel": (1.0, 5.0, 19.3)}  # in units of the base weight w; 8.2 and 19.3 = the article's radial flip values
BOX1 = (24, 36.0)
BOX2 = (32, 48.0)
TARGETS = (206.7683, 3477.23)
KOIDE = 2.0 / 3.0


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def load_json():
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            return json.load(f)
    return {"task": "M5.32 R31 = S1", "controls": {}, "center": {}, "points": {}, "rows": {}, "second": {}, "collect": {}}


def save_json(J):
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(J, f, indent=1, default=float)
    os.replace(tmp, OUT_JSON)


def job_of(pid, params, seed, J, box):
    n, L = box
    return dict(pid=pid, seed=seed, J=float(J), n=n, L=float(L), g=params["g"], delta=params["delta"], w1s=params["w1s"],
                c6=params["c6"], lam_s=params["lam_s"], lam_sel=params["lam_sel"],
                eps_sx=params.get("eps_sx", 0.0), eps_sel=params.get("eps_sel", 0.0))


def terms_of(box, params):
    return core.Terms(box[0], box[1], params["g"], params["delta"], params["w1s"], params["c6"], params["lam_s"], params["lam_sel"],
                      params.get("eps_sx", 0.0), params.get("eps_sel", 0.0))


# ================= R31-0: controls =================
def controls():
    out = {"versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "scipy": __import__("scipy").__version__, "sympy": __import__("sympy").__version__}}
    # (a) exactness at zero new weights on the stored R23 field
    log("(a) the R23 exactness control")
    M32 = np.load(os.path.join(DATA, "m5_32_r23_1", "rad_pin_d0.3_w25_c0_n32_L48.npz"))["M"]
    T = core.Terms(32, 48.0, 8.0, 0.3, 25.0)
    E, _, parts = T.e_j(M32, 0.0)
    red = core.Reduced(T, M32, 0.0)
    E2, g = red.fun(red.pack(M32))
    ref = json.load(open(os.path.join(DATA, "m5_32_r23_1_cscan.json")))["rows"]["rad_pin_d0.3_w25_c0_n32_L48"]
    refE, refF = ref["E"], ref["chunks"][-1]["fmax_spatial"]
    out["a_r23_exactness"] = {"E_here": E, "E_r23": refE, "rel_diff": abs(E - refE) / abs(refE),
                              "fmax_spatial_here": red.last[1], "fmax_spatial_r23": refF,
                              "rel_diff_fmax": abs(red.last[1] - refF) / refF, "fmax_M00_slaving": red.last[2],
                              "kin_cert_on_the_R23_field": parts["kin_cert"],
                              "a0_norm2_free_r_ge_12": float(np.einsum("...ab,...ab->...", parts["a0"], parts["a0"])[T.free & (T.r >= 12)].sum()),
                              "a0_norm2_total": float(np.einsum("...ab,...ab->...", parts["a0"], parts["a0"]).sum()),
                              "PASS": bool(abs(E - refE) / abs(refE) < 1e-10 and abs(red.last[1] - refF) / refF < 1e-6),
                              "fails_if": "the energy or the reduced gradient on the stored field differs from R23-1's record",
                              "reading": "the clock generator is NOT zero on the relaxed c = 0 exterior: the R23 point's halo fills the box "
                                         "(R23, the quartic flatness), so the field is biaxial out to the shell and kin is box-dependent "
                                         "there; the two-box read of the plan is the guard"}
    # (b) the clock generator against B8.a0_unit on the analytic family, and on the seeds
    log("(b) the clock generator")
    RB = core._load("m5_32_r2_b_bounded", "m5_32_r2_b_bounded.py")
    B8 = RB.B8
    cfg8 = B3.base_cfg(s=-1.0, g=8.0, n=16, L=24.0, delta=0.3)
    Mf = B8.dressed(cfg8, 0.0)
    a0ref = B8.a0_unit(cfg8, 0.0)
    Gr = core.rhat_generator(cfg8)
    a0 = core.a0_of(Mf, Gr)
    # the family's generator G1 acts on indices (2, 3) with its own sign; compare up to the overall sign
    s = np.sign(np.sum(a0 * a0ref)) or 1.0
    diff = float(np.max(np.abs(s * a0 - a0ref)))
    out["b_clock_generator"] = {"family_box": "n 16, L 24, (g, 1, delta, 0) = (8, 1, 0.3, 0)", "max_abs_diff_vs_B8_a0_unit": diff,
                                "max_abs_a0": float(np.max(np.abs(a0ref))), "overall_sign": float(s),
                                "PASS": bool(diff < 1e-6 * max(1.0, float(np.max(np.abs(a0ref))))),
                                "fails_if": "[G_r, M] differs from the family's own clock flow beyond the O(dt^2) of its central difference"}
    T24 = core.Terms(*BOX1, 8.0, 0.3, 25.0)
    rows = {}
    for kind in core.SEEDS:
        M = core.seed_of(T24, kind)
        a0 = core.a0_of(M, T24.Gr)
        a02 = np.einsum("...ab,...ab->...", a0, a0)
        k, _, kp = T24.kin(M, need_grad=False)
        rows[kind] = {"a0_norm2_total": float(a02.sum()), "a0_norm2_r_ge_9": float(a02[T24.r >= 9.0].sum()),
                      "a0_norm2_max_cell": float(a02.max()), "kin_cert": kp["kin_cert"]}
    out["b_clock_generator"]["seeds_n24"] = rows
    out["b_clock_generator"]["reading"] = ("on the uniaxial exterior [G_r, S_ext] = 0 exactly (rad, obl and node carry a0 only where their core "
                                           "is anisotropic and not radial; bia and perm in their cores); kin_cert = 0 on an isotropic melted core")
    # (c) kin_cert >= 0, the spectral kinetic part zero along a0, on random block-diagonal fields
    log("(c) kin sign and the spectral kinetic part")
    rng = np.random.default_rng(7)
    Tc = core.Terms(12, 18.0, 8.0, 0.3, 25.0)
    rows = []
    for trial in range(5):
        M = core.seed_of(Tc, "bia")
        M[..., 1:, 1:] += 0.3 * B3.sym4(rng.normal(size=M[..., 1:, 1:].shape)) * np.exp(-(Tc.r / 4.0) ** 2)[..., None, None]
        M[..., 0, 0] += 0.5 * rng.normal(size=M.shape[:3])
        a0 = core.a0_of(M, Tc.Gr)
        t, pows = core.traces_pows(M)
        kspec = 0.0
        for k in range(1, 5):
            kspec += float(np.sum((np.einsum("...ab,...ab->...", core.dSp_dM(pows, k), a0)) ** 2))
        scale = float(np.sum(a0 * a0)) * float(np.max(np.abs(t[3]))) ** 2
        k, _, kp = Tc.kin(M, need_grad=False)
        rows.append({"kin_cert": kp["kin_cert"], "kin_spec_sum_p_(dS_p.a0)^2": kspec, "scale": scale, "a0_00_max": float(np.abs(a0[..., 0, :]).max())})
    out["c_kin_sign_and_spectral_clock"] = {"trials": rows, "PASS": bool(all(r["kin_cert"] >= 0 and r["kin_spec_sum_p_(dS_p.a0)^2"] < 1e-20 * r["scale"] for r in rows)),
                                            "fails_if": "a random block-diagonal field with kin_cert < 0, or a nonzero dS_p along the internal rotation",
                                            "reading": "a0 is spatial, so <[a0, A_i], [a0, A_i]>_eta is the Frobenius norm of a spatial block; S_p is "
                                                       "invariant under the rotation, so the spectral term is blind to the clock: it enters E_stat only"}
    # (d) Disc: eigen route vs trace polynomial, zeros, K_D
    log("(d) the discriminant")
    rows = {}
    for g in (1.5, 8.0, 32.0):
        Td = core.Terms(12, 18.0, g, 0.3, 25.0)
        M = core.seed_of(Td, "bia")
        M[..., 1:, 1:] += 0.05 * B3.sym4(rng.normal(size=M[..., 1:, 1:].shape)) * Td.free[..., None, None]
        D1, G1 = core.disc_and_grad(M)
        D2, G2 = core.disc_poly_and_grad(M)
        # a brute-force check of the gradient of D1 by FD along one random symmetric direction (3 x 3 block + M_00)
        X = B3.sym4(rng.normal(size=(4, 4))); X[0, 1:] = X[1:, 0] = 0.0
        e = 1e-4
        fd = (core.disc_and_grad(M + e * X, False)[0] - core.disc_and_grad(M - e * X, False)[0]) / (2 * e)
        an = np.einsum("...ab,ab->...", G1, X)
        rows[f"g {g:g}"] = {"max_abs_D_eig_minus_D_poly": float(np.abs(D1 - D2).max()), "max_D": float(D1.max()),
                            "max_abs_G_eig_minus_G_poly": float(np.abs(G1 - G2).max()), "max_abs_G": float(np.abs(G1).max()),
                            "grad_fd_max_abs_err_eig_route": float(np.abs(fd - an).max()), "grad_fd_scale": float(np.abs(an).max()),
                            "D_min": float(D1.min())}
    Tv = core.Terms(12, 18.0, 8.0, 0.3, 25.0)
    Mv = np.broadcast_to(np.diag([8.0, 1.0, 0.3, 0.3]), (12, 12, 12, 4, 4)).copy()
    Dv, Gv = core.disc_and_grad(Mv)
    Ms = core.seed_of(Tv, "rad")
    Ds, Gs = core.disc_and_grad(Ms)
    Mn = np.broadcast_to(np.diag([8.0, 1.0, 0.3, 0.0]), (2, 2, 2, 4, 4)).copy()
    Dn, _ = core.disc_and_grad(Mn, False)
    out["d_discriminant"] = {"routes": rows, "degenerate_vacuum": {"D_max": float(np.abs(Dv).max()), "grad_max": float(np.abs(Gv).max())},
                             "rad_seed_(uniaxial everywhere)": {"D_max": float(np.abs(Ds).max()), "grad_max": float(np.abs(Gs).max()),
                                                                "D_in_the_melted_core_r_lt_3": float(np.abs(Ds[Tv.r < 3.0]).max())},
                             "K_D": {"closed_form_(8, 0.3)": core.k_d_of(8.0, 0.3), "Disc_at_(-8,1,0.3,0)": float(Dn.ravel()[0]),
                                     "rel_diff": abs(core.k_d_of(8.0, 0.3) - float(Dn.ravel()[0])) / core.k_d_of(8.0, 0.3)},
                             "disc_polynomial_in_S_p": core.DISC_STR[:400] + " ...",
                             "PASS": bool(rows["g 1.5"]["max_abs_D_eig_minus_D_poly"] < 1e-9 * max(1.0, rows["g 1.5"]["max_D"])
                                          and float(np.abs(Dv).max()) == 0.0 and float(np.abs(Gv).max()) == 0.0
                                          and abs(core.k_d_of(8.0, 0.3) - float(Dn.ravel()[0])) < 1e-9 * core.k_d_of(8.0, 0.3)),
                             "deviation": "the plan's evaluation route (the sympy polynomial in S_1..S_4) is kept as this identity control only: at g = 8 "
                                          "its monomials (up to S_1^12) cancel to zero on the degenerate vacuum with 1e-6 absolute noise and the "
                                          "reduced gradient failed the FD gate at 1e-2; the run evaluates the same function from the eigenvalues "
                                          "(lambda_0 = -M_00 and the symmetric 3 x 3 block), gradient assembled without division, exact at a coalescing pair",
                             "fails_if": "the two routes disagree at g 1.5, Disc or its gradient nonzero on the degenerate vacuum or the uniaxial seed, K_D off its closed form"}
    # (e) the FD gate of the reduced gradient, every term on
    log("(e) the FD gate")
    rows = {}
    for label, (n, L, g, d, w1s) in {"n24 center": (24, 36.0, 8.0, 0.3, 25.0), "n24 g32 d0.1 w1": (24, 36.0, 32.0, 0.1, 1.0),
                                     "n32 center": (32, 48.0, 8.0, 0.3, 25.0)}.items():
        Te = core.Terms(n, L, g, d, w1s, c6=0.05, lam_s=1e-3, lam_sel=0.3)
        M = core.seed_of(Te, "bia")
        M[..., 1:, 1:] += 1e-2 * B3.sym4(rng.normal(size=M[..., 1:, 1:].shape)) * Te.free[..., None, None]
        red = core.Reduced(Te, M, 50.0)
        fd = red.fd_gate(M, ndir=3, t=1e-4)
        rows[label] = fd
        log(f"   {label}: max rel {fd['max_rel_err']:.2e}")
    out["e_fd_gate"] = {"rows": rows, "weights": {"c6": 0.05, "lam_s": 1e-3, "lam_sel": 0.3, "J": 50.0},
                        "PASS": bool(max(r["max_rel_err"] for r in rows.values()) < 1e-5),
                        "fails_if": "the reduced gradient disagrees with the Richardson pair of central differences beyond 1e-5 relative"}
    # (f) the two selective members on the stored R23 field (a record)
    log("(f) the two selective members on the R23 field")
    S = M32[..., 1:, 1:]
    Sd = S - (np.einsum("...ii->...", S) / 3.0)[..., None, None] * np.eye(3)
    S2 = Sd @ Sd
    trS2 = np.einsum("...ii->...", S2)
    trS3 = np.einsum("...ii->...", S2 @ Sd)
    devS2 = S2 - (trS2 / 3.0)[..., None, None] * np.eye(3)
    R = trS2[..., None, None] * devS2 - trS3[..., None, None] * Sd
    h = T.h
    E_R = 0.0
    for br, wt in B3.branches(T.cfg["stencil"]):
        for i in range(3):
            dR = B3.d1(R, i, h, br)
            E_R += wt * float(np.sum(dR * dR))
    E_R *= h**3 * 9.0 / 8.0
    Tsel = core.Terms(32, 48.0, 8.0, 0.3, 25.0, lam_sel=1.0)
    Eq, _ = Tsel.sel_static(M32, need_grad=False)
    k, _, kp = Tsel.kin(M32, need_grad=False)
    out["f_two_selective_members_on_R23"] = {"hydrogen_(9Z/8)Tr(dR dR)_at_Z_1": E_R,
                                             "V13_B_ii_static_part_at_lam_sel_1": Eq, "V13_B_ii_kin_part_at_lam_sel_1_(a0 = [G_r, M])": kp["kin_sel"],
                                             "K_D": Tsel.K_D, "reading": "a record of how the two members differ on this field, never a gate (prior claim (i))"}
    out["PASS_all"] = bool(all(out[k]["PASS"] for k in ("a_r23_exactness", "b_clock_generator", "c_kin_sign_and_spectral_clock", "d_discriminant", "e_fd_gate")))
    out["wall_s"] = round(time.time() - T0, 1)
    J = load_json()
    J["controls"] = out
    save_json(J)
    print(json.dumps({k: (v.get("PASS") if isinstance(v, dict) else v) for k, v in out.items()}, indent=1))
    return out


# ================= the center point =================
def _relax_worker(args):
    j, src_tag, start, out_dir, max_iter = args
    if src_tag is not None:
        M0 = np.load(os.path.join(out_dir, src_tag + ".npz"))["M"]
    else:
        M0 = start
    ug = core.ups_guard()
    M, row = core.relax(j, M0, out_dir=out_dir, max_iter=max_iter, ug=ug)
    row.pop("chunks_full", None)
    return row


def run_many(argl, workers):
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_relax_worker, a) for a in argl]
        for fut in as_completed(futs):
            rows.append(fut.result())
    return rows


def center(workers=5):
    J = load_json()
    T = core.Terms(*BOX1, CENTER["g"], CENTER["delta"], CENTER["w1s"])
    wG, Lf = core.goldstone_omega(T)
    rec = J.get("center", {}) or {}
    rec.update(box=list(BOX1), omega_G=wG, L_free=Lf, params=CENTER)
    os.makedirs(OUT_DIR, exist_ok=True)
    # stage 1: the five seeds at J = 0, cold
    jobs = [job_of("c0", CENTER, s, 0.0, BOX1) for s in core.SEEDS]
    args = [(j, None, core.seed_of(T, j["seed"]), OUT_DIR, None) for j in jobs]
    log(f"center stage 1: {len(jobs)} cold J = 0 relaxations on {workers} workers")
    rows = run_many(args, workers)
    for r in rows:
        J["rows"][r["tag"]] = r
    rec["J0"] = {r["seed"]: {"tag": r["tag"], "E_J": r["E_J"], "kin": r["final"]["kin"], "wall_s": r["wall_s"], "iters": r["iters"], "label": r["label"]} for r in rows}
    kin_rad = rec["J0"]["rad"]["kin"]
    J1 = 2.0 * kin_rad * wG
    rec["J_1"] = J1
    rec["J_1_definition"] = "J_1 = 2 kin_rad(J = 0) omega_G, omega_G^2 = 3 (2/h sin(pi h / 2 L_free))^2 on the n 24 box; the same J values on every box and point (the charge is the invariant)"
    J["center"] = rec
    save_json(J)
    log(f"center: kin_rad {kin_rad:.4g}, omega_G {wG:.4g}, J_1 {J1:.4g}")
    # stage 2: the ladder J_1, 2 J_1 per seed, warm from the previous J (one sequential job per seed)
    log("center stage 2: the J ladder per seed")

    def ladder_args(seed):
        return (seed, [job_of("c0", CENTER, seed, J1, BOX1), job_of("c0", CENTER, seed, 2 * J1, BOX1)], rec["J0"][seed]["tag"])
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_ladder_worker, ladder_args(s), OUT_DIR) for s in core.SEEDS]
        for fut in as_completed(futs):
            for r in fut.result():
                J["rows"][r["tag"]] = r
    save_json(J)
    # stage 3: the closure at J_1 on rad, and the cost table
    log("center stage 3: the closure dE_J/dJ = omega* on rad at J_1")
    src = core.job_tag(job_of("c0", CENTER, "rad", J1, BOX1))
    cl = [(job_of("c0_closure", CENTER, "rad", f * J1, BOX1), src, None, OUT_DIR, None) for f in (1.05, 0.95)]
    rows = run_many(cl, 2)
    Ep = next(r for r in rows if abs(r["J"] - 1.05 * J1) < 1e-9)
    Em = next(r for r in rows if abs(r["J"] - 0.95 * J1) < 1e-9)
    r1 = J["rows"][src]
    dEdJ = (Ep["E_J"] - Em["E_J"]) / (0.10 * J1)
    rec["closure_at_J_1_rad"] = {"dE_dJ_numeric": dEdJ, "omega_star": r1["final"]["omega"], "rel": abs(dEdJ / r1["final"]["omega"] - 1.0),
                                 "PASS_3_percent": bool(abs(dEdJ / r1["final"]["omega"] - 1.0) <= 0.03),
                                 "rows": {"1.05": {"E_J": Ep["E_J"], "label": Ep["label"]}, "0.95": {"E_J": Em["E_J"], "label": Em["label"]}},
                                 "fails_if": "the envelope slope differs from omega* by more than 3 percent (R4's G6 gate)"}
    for r in rows:
        J["rows"][r["tag"]] = r
    # the cost table
    cost = {}
    for s in core.SEEDS:
        t0 = J["rows"][rec["J0"][s]["tag"]]
        t1 = J["rows"][core.job_tag(job_of("c0", CENTER, s, J1, BOX1))]
        t2 = J["rows"][core.job_tag(job_of("c0", CENTER, s, 2 * J1, BOX1))]
        cost[s] = {"cold_J0": [t0["wall_s"], t0["iters"], t0["label"]], "warm_J1": [t1["wall_s"], t1["iters"], t1["label"]],
                   "warm_2J1": [t2["wall_s"], t2["iters"], t2["label"]]}
    per_point = sum(sum(v[k][0] for k in ("cold_J0", "warm_J1", "warm_2J1")) for v in cost.values())
    rec["cost"] = {"per_seed_wall_s_iters_label": cost, "per_point_sequential_wall_s": per_point,
                   "note": "a sweep point warm-starts its J = 0 relaxations from the center's states, so its cost is bounded by this table"}
    # the ranges of the new weights: each term at its range end equals the certified static energy of the relaxed bia state at J = 0
    Mb = np.load(os.path.join(OUT_DIR, rec["J0"]["bia"]["tag"] + ".npz"))["M"]
    Tb = core.Terms(*BOX1, CENTER["g"], CENTER["delta"], CENTER["w1s"], c6=1.0, lam_s=1.0, lam_sel=1.0)
    E1, _, info = Tb.cert_v4(Mb, False)
    E6, _ = Tb.v6(Mb, False)
    Es, _, per_p = Tb.spec(Mb, need_grad=False)
    Eq, _ = Tb.sel_static(Mb, need_grad=False)
    k, _, kp = Tb.kin(Mb, need_grad=False)
    rng_ = {"E_cert_v4_bia_J0": E1, "unit_values_on_bia_J0": {"V6_at_c6_1": E6, "E_spec_at_lam_s_1": Es, "E_spec_per_p": per_p,
                                                            "E_sel_stat_at_lam_sel_1": Eq, "kin_sel_at_lam_sel_1": kp["kin_sel"], "kin_cert": kp["kin_cert"]},
            "c6_max": abs(E1 / E6) if E6 != 0 else None, "lam_s_max": abs(E1 / Es) if Es != 0 else None, "lam_sel_max": abs(E1 / Eq) if Eq != 0 else None,
            "rule": "each new term at its range end equals the certified static energy (quartic + V4) of the relaxed bia state at J = 0; "
                    "c6 ranges over [-c6_max, c6_max], lam_s and lam_sel over [0, max]; the FD gate (controls (e)) was run at weights inside these ranges"}
    rec["ranges"] = rng_
    rec["wall_s"] = round(time.time() - T0, 1)
    J["center"] = rec
    save_json(J)
    print(json.dumps({"J_1": J1, "closure": rec["closure_at_J_1_rad"], "cost": rec["cost"], "ranges": {k: rng_[k] for k in ("c6_max", "lam_s_max", "lam_sel_max")}}, indent=1, default=float))
    return rec


def _ladder_worker(args, out_dir):
    seed, jobs, src_tag = args
    ug = core.ups_guard()
    M = np.load(os.path.join(out_dir, src_tag + ".npz"))["M"]
    rows = []
    for j in jobs:
        M, row = core.relax(j, M, out_dir=out_dir, ug=ug)
        rows.append(row)
    return rows


# ================= the sweep =================
def points_list(J, n_lhs=200, seed=3105):
    """the axes through the center first, then the Latin hypercube over the box; the ranges from the center record."""
    rg = J["center"]["ranges"]
    c6m, lsm, lselm = rg["c6_max"], rg["lam_s_max"], rg["lam_sel_max"]
    pts = []
    k = 0

    def add(label, **kw):
        nonlocal k
        p = dict(CENTER)
        p.update(kw)
        pts.append((f"{label}", p))
        k += 1
    for g in (12.0, 16.0, 24.0, 32.0):
        add(f"ax_g{g:g}", g=g)
    for d in (0.1, 0.2, 0.45, 0.6):
        add(f"ax_d{d:g}", delta=d)
    for w in (1.0, 5.0, 100.0):
        add(f"ax_w{w:g}", w1s=w)
    for f in (-1.0, -0.3, 0.3, 1.0):
        add(f"ax_c6_{f:+g}", c6=f * c6m)
    for f in (0.1, 0.3, 1.0):
        add(f"ax_ls_{f:g}", lam_s=f * lsm)
    for f in (0.1, 0.3, 1.0):
        add(f"ax_lsel_{f:g}", lam_sel=f * lselm)
    # Latin hypercube over (g, delta, log10 w1s, c6, lam_s, lam_sel)
    rng = np.random.default_rng(seed)
    dims = 6
    cut = (np.arange(n_lhs)[:, None] + rng.uniform(size=(n_lhs, dims))) / n_lhs
    for d in range(dims):
        cut[:, d] = cut[rng.permutation(n_lhs), d]
    for i in range(n_lhs):
        u = cut[i]
        add(f"lhs{i:03d}", g=8.0 + 24.0 * u[0], delta=0.1 + 0.5 * u[1], w1s=float(10 ** (2.0 * u[2])),
            c6=(2.0 * u[3] - 1.0) * c6m, lam_s=u[4] * lsm, lam_sel=u[5] * lselm)
    # the budget cuts points from the END of this list, so the first points cover every axis family once
    # (the comparator axes first: the sextic both signs, the spectral and the selective term at full range),
    # then a second sample per family, then the remaining axis points, then the Latin hypercube
    first = ["ax_c6_+1", "ax_c6_-1", "ax_ls_1", "ax_lsel_1", "ax_g16", "ax_d0.45", "ax_w5",
             "ax_c6_+0.3", "ax_c6_-0.3", "ax_ls_0.3", "ax_g32", "ax_d0.1"]
    rank = {lab: i for i, lab in enumerate(first)}
    pts.sort(key=lambda t: (rank.get(t[0], len(first)),))  # stable: the rest keep their order
    return pts


def _point_worker(args):
    pid, params, J1, box, center_tags, out_dir = args
    ug = core.ups_guard()
    t0 = time.time()
    T = terms_of(box, params)
    Tc = core.Terms(box[0], box[1], CENTER["g"], CENTER["delta"], CENTER["w1s"])
    rows = []
    for seed in core.SEEDS:
        # the J = 0 start: the point's own seed plus the center's relaxation delta of the same seed (the exterior stays the point's)
        M0 = core.seed_of(T, seed)
        if center_tags and seed in center_tags:
            Mc = np.load(os.path.join(out_dir, center_tags[seed] + ".npz"))["M"]
            if Mc.shape[0] == box[0]:
                dlt = Mc[..., 1:, 1:] - core.seed_of(Tc, seed)[..., 1:, 1:]
                M0[..., 1:, 1:] += dlt * T.free[..., None, None]
        M = M0
        for Jv in (0.0, J1, 2.0 * J1):
            j = job_of(pid, params, seed, Jv, box)
            M, row = core.relax(j, M, out_dir=out_dir, ug=ug, log_prefix=f"[{pid}] ")
            rows.append(row)
            if ug is not None and ug.wrap_up():
                break
    return pid, rows, round(time.time() - t0, 1)


def sweep(workers=12, hours=8.0, n_lhs=200, box=BOX1):
    J = load_json()
    J1 = J["center"]["J_1"]
    center_tags = {s: J["center"]["J0"][s]["tag"] for s in core.SEEDS}
    pts = points_list(J, n_lhs)
    done = set(J["points"].keys())
    pending = [(pid, p) for pid, p in pts if pid not in done]
    log(f"sweep: {len(pts)} points planned, {len(pending)} pending, {workers} workers, budget {hours} h")
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {}
        it = iter(pending)
        # keep the pool full; stop submitting when the budget is spent
        def submit_next():
            try:
                pid, p = next(it)
            except StopIteration:
                return False
            futs[ex.submit(_point_worker, (pid, p, J1, box, center_tags, OUT_DIR))] = (pid, p)
            return True
        for _ in range(workers):
            if not submit_next():
                break
        while futs:
            for fut in as_completed(list(futs.keys())):
                pid, p = futs.pop(fut)
                try:
                    pid_, rows, wall = fut.result()
                    Jn = load_json()
                    for r in rows:
                        Jn["rows"][r["tag"]] = r
                    Jn["points"][pid] = {"params": p, "box": list(box), "wall_s": wall, "tags": [r["tag"] for r in rows], "n_rows": len(rows)}
                    save_json(Jn)
                    log(f"point {pid} done in {wall} s ({len(rows)} rows); {len(Jn['points'])} points in the record")
                except Exception as e:  # noqa: BLE001
                    log(f"point {pid} FAILED: {e!r}")
                    Jn = load_json()
                    Jn["points"][pid] = {"params": p, "box": list(box), "status": "FAILED", "error": repr(e)}
                    save_json(Jn)
                if (time.time() - t_start) / 3600.0 < hours:
                    submit_next()
                else:
                    log("budget spent: no new points submitted")
                break
    log("sweep done")


# ================= R31-1b: the author's lifts on the center (chain granularity: one seed's J ladder per job) =================
def _chain_worker(args):
    pid, params, seed, J1, box, center_tags, out_dir = args
    ug = core.ups_guard()
    t0 = time.time()
    T = terms_of(box, params)
    Tc = core.Terms(box[0], box[1], CENTER["g"], CENTER["delta"], CENTER["w1s"])
    M0 = core.seed_of(T, seed)
    if center_tags and seed in center_tags:
        Mc = np.load(os.path.join(out_dir, center_tags[seed] + ".npz"))["M"]
        if Mc.shape[0] == box[0]:
            dlt = Mc[..., 1:, 1:] - core.seed_of(Tc, seed)[..., 1:, 1:]
            M0[..., 1:, 1:] += dlt * T.free[..., None, None]
    rows = []
    M = M0
    for Jv in (0.0, J1, 2.0 * J1):
        j = job_of(pid, params, seed, Jv, box)
        M, row = core.relax(j, M, out_dir=out_dir, ug=ug, log_prefix=f"[{pid} {seed}] ")
        rows.append(row)
        if ug is not None and ug.wrap_up():
            break
    return pid, seed, rows, round(time.time() - t0, 1)


def lift_points():
    pts = []
    for key, vals in LIFT_EPS.items():
        for e in vals:
            p = dict(CENTER)
            p[key] = float(e)
            pts.append((f"ax_{key[4:]}_{e:g}", p))
    return pts


def lifts(workers=12, box=BOX1):
    """the sextic lift (eq. 46) and the selective lift (eq. 48) on the center, one point per eps, the full catalog and ladder;
    30 chains (6 points x 5 seeds) kept on the pool; a chain whose 2 J_1 tag is in the record is skipped (resumable)."""
    J = load_json()
    J1 = J["center"]["J_1"]
    center_tags = {s: J["center"]["J0"][s]["tag"] for s in core.SEEDS}
    chains = []
    for pid, p in lift_points():
        for seed in core.SEEDS:
            if core.job_tag(job_of(pid, p, seed, 2.0 * J1, box)) in J["rows"]:
                continue
            chains.append((pid, p, seed))
    log(f"lifts: {len(lift_points())} points, {len(chains)} chains pending, {workers} workers")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_chain_worker, (pid, p, seed, J1, box, center_tags, OUT_DIR)): (pid, p, seed) for pid, p, seed in chains}
        for fut in as_completed(futs):
            pid, p, seed = futs[fut]
            Jn = load_json()
            rec = Jn["points"].setdefault(pid, {"params": p, "box": list(box), "tags": [], "n_rows": 0, "wall_s": 0.0, "kind": "lift"})
            try:
                _, _, rows, wall = fut.result()
                for r in rows:
                    Jn["rows"][r["tag"]] = r
                rec["tags"] += [r["tag"] for r in rows]
                rec["n_rows"] = len(rec["tags"])
                rec["wall_s"] = round(rec["wall_s"] + wall, 1)
                log(f"chain {pid} {seed} done in {wall} s ({len(rows)} rows); point has {rec['n_rows']} rows")
            except Exception as e:  # noqa: BLE001
                log(f"chain {pid} {seed} FAILED: {e!r}")
                rec.setdefault("failed_chains", []).append({"seed": seed, "error": repr(e)})
            save_json(Jn)
    log("lifts done")


# ================= the second box on the candidates =================
def _second_worker(args):
    pid, params, seed, Jv, tag24, out_dir = args
    ug = core.ups_guard()
    T2 = terms_of(BOX2, params)
    M24 = np.load(os.path.join(out_dir, tag24 + ".npz"))["M"]
    M = core.interp_field(M24, BOX2[0])
    Mext = B3.embed34(T2.S_ext, T2.cfg)
    pin = ~T2.free
    M[pin] = Mext[pin]
    M[..., 0, 1:] = 0.0
    M[..., 1:, 0] = 0.0
    j = job_of(pid + "_n32", params, seed, Jv, BOX2)
    M, row = core.relax(j, M, out_dir=out_dir, ug=ug, log_prefix=f"[{pid} n32] ")
    row["warm_start"] = tag24
    return pid, row


def second(pids, workers=4, keys=("J_ladder_reading",)):
    """the n 32 L 48 box on the three states of each candidate point (the same seed and J: the charge is the invariant),
    warm-started from the n 24 state interpolated to the finer... same spacing, larger box (order-1 zoom, re-pinned).
    Several readings share one pool: their state sets are unioned (a state named by two readings relaxes once)."""
    J = load_json()
    col = J["collect"]["per_point"]
    args = []
    for pid in pids:
        seen = set()
        for key in keys:
            st = col[pid]["states"][key]
            if not st:
                log(f"second: {pid} has no three states in {key}")
                continue
            for tag in st["states"]:
                if tag in seen:
                    continue
                seen.add(tag)
                r = J["rows"][tag]
                if core.job_tag(job_of(pid + "_n32", col[pid]["params"], r["seed"], r["J"], BOX2)) in J["rows"]:
                    log(f"second: {pid} {r['seed']} J {r['J']:.6g} already on the second box, skipped")
                    continue
                args.append((pid, col[pid]["params"], r["seed"], r["J"], tag, OUT_DIR))
    log(f"second box: {len(args)} relaxations on {workers} workers")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_second_worker, a): a for a in args}
        for fut in as_completed(futs):
            pid, row = fut.result()
            Jn = load_json()
            Jn["rows"][row["tag"]] = row
            Jn["second"].setdefault(pid, {"params": col[pid]["params"], "box": list(BOX2), "keys": list(keys), "tags": []})
            Jn["second"][pid]["tags"].append(row["tag"])
            save_json(Jn)
            log(f"second box {pid}: {row['tag']} {row['label']} E_J {row['final']['E_J']:.6g} wall {row['wall_s']}")


# ================= collect =================
RUNAWAY_E_STAT = 1e3  # |E_stat| beyond this is a runaway, not a state (the bounded rows of this box stay under 25); negative c6 runs to -inf


def member_checks(r, rows_of_point):
    """the five conditions and the three invariants on one member; returns (accepted, notes)."""
    f = r["final"]
    notes = []
    deg12 = f.get("cond1_degree_top", {}).get("12", {})
    ok1 = deg12.get("degree") is not None and abs(abs(deg12["degree"]) - 1.0) < 0.1
    if not ok1:
        notes.append(f"cond1: degree on the r 12 cube {deg12.get('degree')} ({deg12.get('reading')})")
    if r["label"] != "AT_GATE":
        notes.append(f"gate: {r['label']} (fmax {r['chunks'][-1].get('fmax_spatial')})")
    if not np.isfinite(f["E_J"]):
        notes.append("E_J not finite (kin <= 0)")
        ok1 = False
    # a runaway: the functional has no minimum along the path taken (negative c6, deviation 9) and the line
    # search stalls at an energy orders above any bounded state of this box (the center's rows stay under 25)
    if abs(f["E_stat"]) > RUNAWAY_E_STAT:
        notes.append(f"runaway: |E_stat| {f['E_stat']:.3g} beyond the {RUNAWAY_E_STAT:g} bar ({r['label']})")
        ok1 = False
    return ok1 and np.isfinite(f["E_J"]), notes


def point_states(J, pid, tags):
    """the three states of one point from its accepted members; the J-free tabulation beside it."""
    rows = [J["rows"][t] for t in tags if t in J["rows"]]
    members = []
    for r in rows:
        acc, notes = member_checks(r, rows)
        f = r["final"]
        members.append({"tag": r["tag"], "seed": r["seed"], "J": r["J"], "E_J": f["E_J"], "E_stat": f["E_stat"], "E_rot": f["E_rot"],
                        "omega": f["omega"], "kin": f["kin"], "label": r["label"], "accepted": bool(acc), "notes": notes,
                        "E_J_over_omega_J": f.get("E_J_over_omega_J"), "biax_eps_max": f["biax_eps_max"],
                        "cond1_rms_r_ge_12": f["cond1_rms_S_minus_Sext_free_r_ge_12"],
                        "cond5_signature": f.get("cond5_K_census", {}).get("signature_histogram"), "cond5_n_neg": f.get("cond5_K_census", {}).get("n_neg"),
                        "degree_r12": f.get("cond1_degree_top", {}).get("12", {}).get("degree"),
                        "frame_jumps": frame_jumps(r["tag"])})
    acc = sorted([m for m in members if m["accepted"]], key=lambda m: m["E_J"])
    out = {"members": members, "n_accepted": len(acc)}

    def three(lst, key):
        if len(lst) < 3:
            return None
        E = [m["E_J"] for m in lst[:3]]
        Q = sum(E) / (sum(np.sqrt(E)) ** 2) if min(E) > 0 else None
        sig = [m["cond5_signature"] for m in lst[:3]]
        nneg = [m["cond5_n_neg"] for m in lst[:3]]
        same_sig = all(s == sig[0] for s in sig) if all(s is not None for s in sig) else None
        nneg_spread = (max(nneg) - min(nneg)) / max(1, max(nneg)) if all(v is not None for v in nneg) else None
        return {"states": [m["tag"] for m in lst[:3]], "E": E, "ratios": [E[1] / E[0], E[2] / E[0]], "koide_Q": Q,
                "invariant_D8_same_K_signature": same_sig, "K_neg_count_spread": nneg_spread,
                "pairwise_exterior_rms_r_ge_12_max": None, "reading_key": key}
    out["J_ladder_reading"] = three(acc, "J ladder in the catalog (the author's answer)")
    accJ0 = [m for m in acc if m["J"] == 0.0]
    out["J_free_reading"] = three(accJ0, "J = 0 members only (the secondary tabulation)")
    # a labeled addition (not in the plan's five conditions): the three lowest DISTINCT states, a member being a
    # duplicate of a lower one when their sorted-spectrum fields agree within DUP_RMS on the free cells
    kept, dups = [], []
    for m in acc:
        lam = spectrum_field(m["tag"])
        if lam is None:
            kept.append(m)
            continue
        same = None
        for k in kept:
            lk = spectrum_field(k["tag"])
            if lk is not None and spectrum_rms(lam, lk) < DUP_RMS:
                same = k["tag"]
                break
        if same is None:
            kept.append(m)
        else:
            dups.append({"tag": m["tag"], "duplicate_of": same})
    out["distinct_reading"] = three(kept, "three lowest distinct states (duplicates by the sorted-spectrum field, a labeled addition)")
    out["duplicates"] = dups
    return out


DUP_RMS = 0.05
_SPEC_CACHE = {}


def spectrum_field(tag):
    """the sorted spatial spectrum per cell of a stored state (orientation-free), cached."""
    if tag in _SPEC_CACHE:
        return _SPEC_CACHE[tag]
    p = os.path.join(OUT_DIR, tag + ".npz")
    if not os.path.exists(p):
        return None
    M = np.load(p)["M"]
    lam = np.sort(np.linalg.eigvalsh(M[..., 1:, 1:]), axis=-1)
    _SPEC_CACHE[tag] = lam
    return lam


def spectrum_rms(a, b):
    n = a.shape[0]
    T = core.Terms(n, n * 1.5, 8.0, 0.3, 25.0)
    d = a - b
    return float(np.sqrt(np.mean(np.sum(d[T.free] ** 2, axis=-1))))


JUMP_TENSOR, JUMP_SPEC_FRAC, JUMP_R = 0.3, 0.5, 14.0  # the article (Sec. 10.2): tensor change > 0.3 with the sorted spectrum moving by about 0.1


def frame_jumps(tag):
    """the author's 3D health check (Release 2.0, Sec. 10.1): the count of nearest-neighbor links with both ends at r < 14 whose
    spatial tensor changes by more than 0.3 in Frobenius norm while the sorted spectrum moves by less than half of that
    (the article's jump links move the spectrum by about a third of the tensor change); the count of large-change links beside it."""
    p = os.path.join(OUT_DIR, tag + ".npz")
    if not os.path.exists(p):
        return None
    M = np.load(p)["M"]
    n = M.shape[0]
    T = core.Terms(n, n * 1.5, 8.0, 0.3, 25.0)
    S = M[..., 1:, 1:]
    lam = np.sort(np.linalg.eigvalsh(S), axis=-1)
    inside = T.r < JUMP_R
    n_links = n_big = n_jump = 0
    for ax in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[ax] = slice(0, n - 1)
        b[ax] = slice(1, n)
        both = inside[tuple(a)] & inside[tuple(b)]
        dS = np.sqrt(np.sum((S[tuple(b)] - S[tuple(a)]) ** 2, axis=(-1, -2)))
        dl = np.sqrt(np.sum((lam[tuple(b)] - lam[tuple(a)]) ** 2, axis=-1))
        big = both & (dS > JUMP_TENSOR)
        jump = big & (dl < JUMP_SPEC_FRAC * dS)
        n_links += int(both.sum())
        n_big += int(big.sum())
        n_jump += int(jump.sum())
    return {"n_links_r_lt_14": n_links, "n_big_change_links": n_big, "n_jump_links": n_jump, "fraction": (n_jump / n_links if n_links else None),
            "definition": f"|dS|_F > {JUMP_TENSOR} and |d sorted lambda|_2 < {JUMP_SPEC_FRAC} |dS|_F, both ends at r < {JUMP_R:g}"}


def vacuum_hessian(params):
    """the parameter-point part of the D8 read (the author's reading of 'the gap unchanged'): the static Hessian of the base
    potential at the point's degenerate vacuum in the 10-dimensional symmetric basis, its zero and negative mode counts;
    the lifts and the sextic are stated, not added (zero Hessian at the vacuum for c6; the lifts stiffen only the pair split)."""
    try:
        D8 = core._load("m5_32_r31_s1_d8", "m5_32_r31_s1_d8.py")
        r = D8.vacuum_d8_read(params["g"], params["delta"], params["w1s"])
        h = dict(r["hessian"])
        h["eigenvalues_nonzero"] = [float(x) for x in h.pop("eigenvalues") if abs(x) > 1e-9]
        h["note"] = "base potential only; c6 has zero Hessian at the vacuum; the lifts add eps w per unit split of the pair"
        return h
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def pairwise_exterior(J, states):
    Ms = []
    for t in states:
        p = os.path.join(OUT_DIR, t + ".npz")
        if os.path.exists(p):
            Ms.append(np.load(p)["M"])
    if len(Ms) < 2:
        return None
    n = Ms[0].shape[0]
    T = core.Terms(n, n * 1.5, 8.0, 0.3, 25.0)
    far = T.free & (T.r >= 12.0)
    mx = 0.0
    for a in range(len(Ms)):
        for b in range(a + 1, len(Ms)):
            d = Ms[a][..., 1:, 1:] - Ms[b][..., 1:, 1:]
            mx = max(mx, float(np.sqrt(np.mean(np.sum(d[far] ** 2, axis=(-1, -2))))))
    return mx


def word_of(reading, both_boxes_ok):
    if reading is None:
        return "NO_THREE_STATES"
    r2, r3 = reading["ratios"]
    Q = reading["koide_Q"]
    if (abs(r2 / TARGETS[0] - 1) < 0.10 and abs(r3 / TARGETS[1] - 1) < 0.10 and Q is not None and abs(Q / KOIDE - 1) < 0.01 and both_boxes_ok):
        return "GREEN_CANDIDATE"
    return "INFORMATIVE_NULL"


def collect():
    J = load_json()
    pts = dict(J["points"])
    pts["c0"] = {"params": CENTER, "box": list(BOX1), "tags": [t for t in J["rows"] if t.startswith("c0_") and "_n24_" in t and not t.startswith("c0_closure")]}
    per_point = {}
    for pid, p in pts.items():
        if p.get("status") == "FAILED":
            continue
        st = point_states(J, pid, p["tags"])
        for key in ("J_ladder_reading", "J_free_reading", "distinct_reading"):
            if st[key]:
                st[key]["pairwise_exterior_rms_r_ge_12_max"] = pairwise_exterior(J, st[key]["states"])
        per_point[pid] = {"params": p["params"], "states": st, "vacuum_hessian": vacuum_hessian(p["params"])}
    # the map: the largest reachable E2 / E1 and E3 / E1 over the points, both readings
    summary = {}
    for key in ("J_ladder_reading", "J_free_reading", "distinct_reading"):
        rows = [(pid, v["states"][key]) for pid, v in per_point.items() if v["states"][key]]
        if not rows:
            summary[key] = {"n_points_with_three_states": 0}
            continue
        best2 = max(rows, key=lambda x: x[1]["ratios"][0])
        best3 = max(rows, key=lambda x: x[1]["ratios"][1])
        closest = min(rows, key=lambda x: abs(np.log(x[1]["ratios"][0] / TARGETS[0])) + abs(np.log(x[1]["ratios"][1] / TARGETS[1])))
        summary[key] = {"n_points_with_three_states": len(rows),
                        "largest_E2_over_E1": {"pid": best2[0], "ratios": best2[1]["ratios"], "Q": best2[1]["koide_Q"]},
                        "largest_E3_over_E1": {"pid": best3[0], "ratios": best3[1]["ratios"], "Q": best3[1]["koide_Q"]},
                        "closest_to_targets_in_log": {"pid": closest[0], "ratios": closest[1]["ratios"], "Q": closest[1]["koide_Q"]},
                        "ratio_E2_E1_min_max": [min(r[1]["ratios"][0] for r in rows), max(r[1]["ratios"][0] for r in rows)],
                        "koide_Q_min_max": [min(r[1]["koide_Q"] for r in rows if r[1]["koide_Q"] is not None), max(r[1]["koide_Q"] for r in rows if r[1]["koide_Q"] is not None)]}
    # the second box: the same three states (seed, J) re-relaxed on n 32 L 48, matched to each n 24 reading it was run for
    second_rows = {}
    for pid, p2 in J.get("second", {}).items():
        if pid not in per_point:
            continue
        second_rows[pid] = {}
        for key in p2.get("keys", [p2.get("key", "J_ladder_reading")]):
            a = per_point[pid]["states"][key]
            if not a:
                continue
            E32, lab32 = [], []
            for tag24 in a["states"]:
                r24 = J["rows"][tag24]
                m = [J["rows"][t] for t in p2["tags"] if J["rows"][t]["seed"] == r24["seed"] and abs(J["rows"][t]["J"] - r24["J"]) < 1e-9]
                E32.append(m[0]["final"]["E_J"] if m else None)
                lab32.append(m[0]["label"] if m else None)
            rec = {"E_n32_in_the_n24_order": E32, "labels_n32": lab32}
            if all(e is not None for e in E32) and len(E32) == 3:
                o = np.argsort(E32)
                Es = [E32[i] for i in o]
                rec["order_kept"] = bool(list(o) == [0, 1, 2])
                rec["ratios_n32"] = [Es[1] / Es[0], Es[2] / Es[0]]
                rec["Q_n32"] = sum(Es) / (sum(np.sqrt(Es)) ** 2) if min(Es) > 0 else None
                rec["ratio_change_n24_to_n32"] = [rec["ratios_n32"][i] / a["ratios"][i] - 1.0 for i in range(2)]
                rec["E_change_n24_to_n32"] = [E32[i] / a["E"][i] - 1.0 for i in range(3)]
                per_point[pid]["states"][key]["second_box"] = rec
            second_rows[pid][key] = rec
    words = {}
    for key in ("J_ladder_reading", "J_free_reading", "distinct_reading"):
        rows = [(pid, v["states"][key]) for pid, v in per_point.items() if v["states"][key]]
        green = [pid for pid, r in rows if word_of(r, r.get("second_box") is not None and max(abs(x) for x in r["second_box"]["ratio_change_n24_to_n32"]) < 0.10) == "GREEN_CANDIDATE"]
        words[key] = {"word": ("GREEN_CANDIDATE" if green else ("INFORMATIVE_NULL" if rows else "NO_THREE_STATES")), "green_points": green,
                      "bound": summary[key].get("largest_E2_over_E1")}
    out = {"per_point": per_point, "summary": summary, "second_box": second_rows, "words": words, "targets": TARGETS, "koide": KOIDE,
           "wall_s": round(time.time() - T0, 1)}
    J["collect"] = out
    save_json(J)
    plot(J)
    print(json.dumps({"summary": summary, "words": words}, indent=1, default=float))
    return out


def plot(J):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = J["collect"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for key, mk, lab in (("J_ladder_reading", "o", "J ladder in the catalog"), ("J_free_reading", "s", "J = 0 members only"),
                         ("distinct_reading", "^", "three lowest distinct states")):
        pts = [(v["params"], v["states"][key]) for v in col["per_point"].values() if v["states"][key]]
        if not pts:
            continue
        r2 = [s["ratios"][0] for _, s in pts]
        r3 = [s["ratios"][1] for _, s in pts]
        Q = [s["koide_Q"] if s["koide_Q"] is not None else np.nan for _, s in pts]
        g = [p["g"] for p, _ in pts]
        d = [p["delta"] for p, _ in pts]
        axs[0].scatter(r2, r3, marker=mk, s=18, alpha=0.7, label=lab)
        axs[1].scatter(g, r2, marker=mk, s=18, alpha=0.7, c=d, cmap="viridis", label=lab)
        axs[2].scatter(r2, Q, marker=mk, s=18, alpha=0.7, label=lab)
    axs[0].axvline(TARGETS[0], color="r", lw=0.8); axs[0].axhline(TARGETS[1], color="r", lw=0.8)
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("E2 / E1"); axs[0].set_ylabel("E3 / E1"); axs[0].legend(fontsize=8)
    axs[1].set_xlabel("g"); axs[1].set_ylabel("E2 / E1 (color: delta)"); axs[1].set_yscale("log")
    axs[2].axhline(KOIDE, color="r", lw=0.8); axs[2].set_xscale("log"); axs[2].set_xlabel("E2 / E1"); axs[2].set_ylabel("Koide Q")
    fig.suptitle("M5.32 R31 = S1: the ratio map over the points run (n 24, L 36; targets in red)")
    fig.tight_layout()
    os.makedirs(PLOTS, exist_ok=True)
    fig.savefig(os.path.join(PLOTS, "m5_32_r31_s1_map.png"), dpi=130)
    log("plot written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--n-lhs", type=int, default=200)
    ap.add_argument("--pids", nargs="*", default=[])
    ap.add_argument("--key", nargs="*", default=["J_ladder_reading"], help="the readings whose three states go to the second box (unioned)")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.mode == "controls":
        controls()
    elif a.mode == "center":
        center(min(a.workers, 5))
    elif a.mode == "sweep":
        sweep(a.workers, a.hours, a.n_lhs)
    elif a.mode == "lifts":
        lifts(a.workers)
    elif a.mode == "second":
        second(a.pids, min(a.workers, 6), tuple(a.key))
    elif a.mode == "collect":
        collect()
    else:
        raise SystemExit(f"unknown mode {a.mode}")


if __name__ == "__main__":
    main()
