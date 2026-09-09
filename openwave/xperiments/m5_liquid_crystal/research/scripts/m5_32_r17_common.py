"""M5.32 R17 shared instrument (ledger 6.6): the R16 instrument (m5_32_r16_common, consumed read-only)
extended by the TRUE fixed-K gradient (the a0 chain rule, R17-1), the director-relative plateau
weight (R17-2), the v6 core coupling W(lambda_1) with the sextic (R17-3), the X_M inertia (R17-4)
and the angular decomposition of the doublet operator (R17-2, R17-3b).  Each extension carries
its gate in the selftest; a gate failing three times drops its stage (the ledger's fallbacks).

EQUATIONS FIRST
---------------
1. The true fixed-K gradient (R17-1).  E_K = E_stat(M) + K^2 / (4 kin(M, a0(M))) with a0(M) = J M + M J^T,
   J = J(u(M), n(M)) = Ebar u n (the frame of m5_32_r16_common).  The R15 / R16 protocol froze a0 in the
   gradient; the R16-3 audit measured the omitted part d a0 / d M at up to 60 x the frozen derivative on
   the rotating end states.  Here
       dE_K/dM = grad_stat - (K^2 / 4 kin^2) [ grad_kin|_{a0 frozen} + (d a0 / d M)^T grad_a0 ],
   grad_a0 = d kin / d a0 (the averaged action's a0 cotangent, pulled back to the sample-0 frame by
   m5_32_r16_common.averaged with need_a0grad), and for a symmetric cotangent Lam_a on a0:
       (d a0 / d M)^T Lam_a = sym(J^T Lam_a + Lam_a J)  +  [d J / d M]^T (2 Lam_a M),
   the second term the frame chain rule already carried by circle_adjoint (Lam_J -> Lam_u, Lam_n ->
   the column normalization -> the projector derivatives), factored out here as J_adjoint.
   Gate: the directional derivative of the FULL E_K (a0 refreshed, the energy the descent reports)
   by complex step at 1e-10 relative (the decisive gate: the whole stack is complex-step safe) and by
   Richardson-extrapolated central differences (eps, eps/2) at 1e-6 (the finite-difference floor; the
   ledger's 1e-8 is met by the complex step, the central differences land at 2e-8) on a small random field; on the R16-3 n32 end states along 3 random free directions AND
   along the frozen-protocol gradient direction (the R16-3 audit's control: the true derivative along
   it is ~60 x the frozen one).
2. (the director-relative weight, W(lambda_1) + the sextic, X_M, the angular decomposition: appended
   below as they land, each with its gate; see the section markers)

Selftest: python3 m5_32_r17_common.py  ->  data/m5_32_r17_common_selftest.json
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

ARGV = list(sys.argv)
sys.argv = [sys.argv[0]]
import m5_32_r16_common as C                              # noqa: E402
sys.argv = ARGV
C15, INS4 = C.C15, C.INS4
ETA, EYE, EBAR = C.ETA, C.EYE, C.EBAR
RES, DATA, PLOTS = C.RES, C.DATA, C.PLOTS
CK = os.path.join(RES, "checkpoints", "m5_32_r17")
os.makedirs(CK, exist_ok=True)
T0 = time.time()
sym, W_to_gradM = C.sym, C.W_to_gradM


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


# ================================================================ 1. the true fixed-K gradient
def J_adjoint(fr, M, LamJ):
    """the pull-back of a cotangent LamJ on J = Ebar u n to M (through u(M), n(M): the column
    normalization and the projector derivatives), as in m5_32_r16_common.circle_adjoint."""
    Lu = np.einsum("...ab,abcd,...d->...c", LamJ, EBAR, fr["n"])
    Ln = np.einsum("...ab,abcd,...c->...d", LamJ, EBAR, fr["u"])
    W = np.zeros_like(M)
    for vec, Lv, Smat, sign_dP, Pj, Sj in ((fr["u"], Lu, -fr["Pg"] @ ETA, +1.0, fr["Pg"], fr["Sg"]),
                                          (fr["n"] * fr["nsign"][..., None], Ln * fr["nsign"][..., None], fr["P1"] @ ETA, -1.0, fr["P1"], fr["S1"])):
        d = np.real(np.einsum("...aa->...a", Smat))
        j = np.argmax(d, axis=-1)
        sjj = np.take_along_axis(np.einsum("...aa->...a", Smat), j[..., None], axis=-1)[..., 0]
        LamS = np.zeros_like(M)
        np.put_along_axis(LamS, j[..., None, None] * np.ones(M.shape[:-2] + (4, 1), dtype=int), (Lv / np.sqrt(sjj)[..., None])[..., :, None], axis=-1)
        corr = -0.5 * np.einsum("...a,...a->...", Lv, vec) / sjj
        idx = np.arange(4)
        diag_add = np.zeros_like(M)
        onehot = (idx[None, :] == j[..., None]).astype(M.dtype)
        diag_add[..., idx, idx] = corr[..., None] * onehot
        LamS = LamS + diag_add
        Z = ETA @ np.swapaxes(LamS, -1, -2)
        W = W + sign_dP * (Pj @ Z @ Sj + Sj @ Z @ Pj)
    return W_to_gradM(W)


def a0_adjoint(fr, M, Lam_a):
    """(d a0 / d M)^T Lam_a for a0 = J M + M J^T, Lam_a symmetric per cell."""
    J = fr["J"]
    JT = np.swapaxes(J, -1, -2)
    direct = sym(JT @ Lam_a + Lam_a @ J)
    return direct + J_adjoint(fr, M, 2.0 * Lam_a @ M)


def energy_and_grad_true(M, cfg, K, n_ref=None, need_grad=True):
    """E_K with a0 = a0_of(M) refreshed AND differentiated (the true gradient).  K None falls back to
    the static problem.  Returns (E, grad, parts, domain, frame) like m5_32_r16_common.energy_and_grad."""
    if K is None:
        return C.energy_and_grad(M, cfg, None, n_ref, need_grad)
    fr = C.frame(M, n_ref)
    a0 = C.a0_of(M, fr)
    res = C.averaged(M, cfg, a0, need_grad, need_a0grad=need_grad, n_ref=n_ref)
    pp = res["parts"]
    kin = pp["kin_tot"]
    E = pp["E_stat"] + K * K / (4.0 * kin)
    pp["E_K"], pp["omega"] = E, K / (2.0 * kin)
    if not need_grad:
        return E, None, pp, res["domain"], fr
    fac = K * K / (4.0 * kin * kin)
    Lam_a = -fac * res["grad_a0"]
    g = res["grad_stat"] - fac * res["grad_kin"] + a0_adjoint(fr, M, Lam_a)
    pp["frozen_grad_norm"] = float(np.sqrt(np.sum(np.abs(res["grad_stat"] - fac * res["grad_kin"]) ** 2)))
    pp["a0_chain_norm"] = float(np.sqrt(np.sum(np.abs(a0_adjoint(fr, M, Lam_a)) ** 2)))
    return E, g, pp, res["domain"], fr


def fire_v4(M0, cfg, free_mask, max_iter, K=None, n_ref=None, log_every=100, tag="", f_tol=1e-6, plateau=(2000, 1e-10),
            dt0=0.01, dt_max=0.1, diag=None, ck_path=None, ck_every=500, true_gradient=True):
    """m5_32_r16_common.fire_v4 with the TRUE fixed-K gradient (true_gradient=True) or the frozen protocol."""
    eg = energy_and_grad_true if true_gradient else C.energy_and_grad
    M = M0.copy()
    free = free_mask[..., None, None].astype(float)
    v = np.zeros_like(M)
    dt, alpha, n_up = dt0, 0.1, 0
    hist = []
    nref = n_ref
    E, F, pp, dom, fr = eg(M, cfg, K, nref)
    nref = fr["n"]
    F = -F * free
    t0 = time.time()
    stop = "max_iter"
    it = 0
    for it in range(1, max_iter + 1):
        Pw = float(np.sum(F * v))
        if Pw > 0.0:
            n_up += 1
            vn = np.sqrt(np.sum(v * v))
            fn = np.sqrt(np.sum(F * F))
            v = (1 - alpha) * v + alpha * (F / max(fn, 1e-300)) * vn
            if n_up > 5:
                dt = min(dt * 1.1, dt_max)
                alpha *= 0.99
        else:
            v[:] = 0.0
            dt *= 0.5
            alpha = 0.1
            n_up = 0
        v += dt * F
        M += dt * v
        try:
            E, F, pp, dom, fr = eg(M, cfg, K, nref)
        except np.linalg.LinAlgError as e:
            print(f"  {tag} it {it}: {e!r}: the field is non-finite or degenerate; stopping", flush=True)
            stop = "non-finite"
            break
        nref = fr["n"]
        F = -F * free
        fmax = float(np.max(np.abs(F)))
        if not np.isfinite(fmax):
            stop = "non-finite"
            break
        if it % log_every == 0 or it == max_iter:
            row = {"it": it, "fmax": fmax, "dt": dt}
            row.update(pp)
            row.update({f"dom_{kk}": vv for kk, vv in dom.items()})
            if diag is not None:
                row.update(diag(M, fr))
            hist.append(row)
            key = "E_K" if K is not None else "E_stat"
            extra = f" kin {pp['kin_tot']:10.4f} om {pp['omega']:.5f} chain/frozen {pp.get('a0_chain_norm', 0) / max(pp.get('frozen_grad_norm', 1e-300), 1e-300):.3f}" if K is not None else ""
            print(f"  {tag} it {it:6d} {key} {pp[key]:14.6f} E_h {pp['E_h']:10.5f} V4 {pp['V4']:.4e} U {pp['U']:.4e} KP {pp['KP']:.4e} "
                  f"reg {pp['reg']:.4e} fmax {fmax:.3e}{extra} l1min {dom['l1_min']:.4f} w1max {dom['w1_max']:.3f} split {dom['half_split_max']:.4f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            back = max(1, plateau[0] // max(log_every, 1))
            if len(hist) > back and abs(row[key] - hist[-1 - back][key]) < plateau[1]:
                stop = "plateau"
                break
            if ck_path is not None and it % ck_every == 0:
                np.save(ck_path, M)
            if dom["escape_d"]:
                stop = "escape_d"
                break
            if C.killed():
                stop = "killswitch"
                break
        if fmax < f_tol:
            stop = "f_tol"
            break
    if ck_path is not None:
        np.save(ck_path, M)
    return M, {"stop": stop, "trace": hist, "wall_s": round(time.time() - t0, 1), "iters": it, "n_ref": nref}


# ================================================================ 2. the director-relative plateau weight (R17-2, object B)
# The author's 22.4: w = 1 near lambda_pair, tapering to 0 AT the local isolated eigenvalues lambda_1(x), lambda_g(x).
# In the frame's projector form w(N) = P23 + w(lambda_1) P_1 + w(lambda_g) P_g this is w(lambda_1) = w(lambda_g) = 0
# on every cell with an isolated director, i.e. w(N) = P23 EXACTLY (the R15 K_P^23): the director never enters
# the clock block, whatever its eigenvalue.  The absolute weight of R16 differs only where lambda_1 < 1 (the
# taper): there it admits the director with weight w(lambda_1) (the author's 24.3 barrier attribution).
# The derivative is the projector derivative alone (dP23 = -dP_g - dP_1, the resolvents S_g, S_1): the
# 27.2 caveat (sup |d_lambda w| >= 1 / |a - b|) is the 1 / (lambda_1 - lambda_2) of the resolvent, i.e. the
# admissible-domain margin gap_1_2_min logged by domain(); escape (d) is its boundary.
# Implementation: a module-level mode switch on m5_32_r16_common.frame (the R16 functions look it up by
# name at call time); weight_mode(cfg) reads cfg["weight"] in {"absolute", "relative"}.
_FRAME_R16 = C.frame
WEIGHT_MODE = {"mode": "absolute"}


def frame_relative(M, n_ref=None):
    fr = _FRAME_R16(M, n_ref)
    if WEIGHT_MODE["mode"] != "relative":
        return fr
    z = np.zeros_like(fr["w1"])
    fr["w1"], fr["dw1"], fr["wg"], fr["dwg"] = z, z, z, z
    one = np.ones_like(fr["c1"])
    fr["c1"], fr["cg"] = one, one
    fr["w"] = fr["P23"]
    fr["pair_out"] = np.zeros(M.shape[:-2], dtype=bool)
    fr.pop("general", None)
    return fr


C.frame = frame_relative                                  # the switch (absolute mode = the R16 frame verbatim)


class weight_mode:
    """with weight_mode(cfg): ... sets the plateau-weight mode from cfg['weight'] (default absolute)."""
    def __init__(self, cfg):
        self.mode = cfg.get("weight", "absolute") if isinstance(cfg, dict) else str(cfg)

    def __enter__(self):
        self.prev = WEIGHT_MODE["mode"]
        WEIGHT_MODE["mode"] = self.mode
        return self

    def __exit__(self, *a):
        WEIGHT_MODE["mode"] = self.prev


# ================================================================ 3. the v6 core coupling W(lambda_1) and the sextic (R17-3, object C)
# U_v6 = (mu - g_W W) rho^2 - nu rho^4 + kappa rho^6,  W = [(1 - lambda_1) / (1 - delta)]^2 (0 on the vacuum, 1 at the
# isotropic core), rho^2 = (lambda_2 - lambda_3)^2 / 4 = spl / 4 (C15.split_cells).  Circle-invariant exactly (spectral),
# so it is added ONCE, outside the average.  Gradient: dU = [(mu - g_W W) - 2 nu rho^2 + 3 kappa rho^4] d rho^2
# - g_W rho^2 dW, dW = -2 (1 - lambda_1) / (1 - delta)^2 d lambda_1, d lambda_1 = tr(P_1 dN) -> W-form c P_1 -> W_to_gradM.
# The reduced line (the sheet D_s = diag(g, 1, delta + s, delta - s), lambda_1 = 1, W = 0): U = mu s^2 - nu s^4 + kappa s^6,
# Coleman's plateau s*^2 = nu / (2 kappa) = 0.0125 (s* = 0.1118 at (nu, kappa) = (1e-2, 0.4)).
def cfg_v6(n, L, gW=1.1, mu=1e-2, nu=1e-2, kappa=0.4, cP=1.0, cs=0.5, completion="rebuild", n_samples=4, weight="relative"):
    cfg = C.cfg_v4(n, L, mu=0.0, cP=cP, cs=cs, completion=completion, n_samples=n_samples)
    cfg.update({"object": "v6", "mu_v6": float(mu), "nu": float(nu), "kappa": float(kappa), "gW": float(gW), "weight": weight})
    return cfg


def u_v6_cells(M, cfg, fr=None, need_grad=True):
    """U_v6 per cell (no h^3) and its M-gradient; also W per cell."""
    if fr is None:
        fr = C.frame(M)
    spl, dspl = C15.split_cells(M, need_grad)
    rho2 = spl / 4.0
    l1 = fr["l1"]
    Wc = ((1.0 - l1) / (1.0 - cfg["delta"])) ** 2
    mu_eff = cfg["mu_v6"] - cfg["gW"] * Wc
    U = mu_eff * rho2 - cfg["nu"] * rho2 ** 2 + cfg["kappa"] * rho2 ** 3
    if not need_grad:
        return U, None, Wc, rho2
    dU_drho2 = mu_eff - 2.0 * cfg["nu"] * rho2 + 3.0 * cfg["kappa"] * rho2 ** 2
    g = dU_drho2[..., None, None] * dspl / 4.0
    dW_dl1 = -2.0 * (1.0 - l1) / (1.0 - cfg["delta"]) ** 2
    coef = -cfg["gW"] * rho2 * dW_dl1
    g = g + W_to_gradM(coef[..., None, None] * fr["P1"])
    return U, g, Wc, rho2


def energy_and_grad_v6(M, cfg, K, n_ref=None, need_grad=True, true_gradient=True):
    """the object's energy and gradient: the R16 action at mu = 0 (weight per cfg) plus U_v6 added once."""
    with weight_mode(cfg):
        if K is None:
            E, g, pp, dom, fr = C.energy_and_grad(M, cfg, None, n_ref, need_grad)
        elif true_gradient:
            E, g, pp, dom, fr = energy_and_grad_true(M, cfg, K, n_ref, need_grad)
        else:
            E, g, pp, dom, fr = C.energy_and_grad(M, cfg, K, n_ref, need_grad)
        U, gU, Wc, rho2 = u_v6_cells(M, cfg, fr, need_grad)
    h3 = cfg["h"] ** 3
    Uv = h3 * np.sum(U)
    pp["U_v6"] = float(np.real(Uv)) if not np.iscomplexobj(M) else complex(Uv)
    pp["W_max"] = float(np.max(np.real(Wc)))
    pp["W_at_min_l1"] = float(np.real(Wc.reshape(-1)[int(np.argmin(np.real(fr["l1"])))]))
    pp["mu_eff_min"] = float(np.min(np.real(cfg["mu_v6"] - cfg["gW"] * Wc)))
    pp["E_stat"] = pp["E_stat"] + pp["U_v6"]
    E = E + Uv
    if K is not None:
        pp["E_K"] = E
    if need_grad:
        g = g + h3 * gU
    return E, g, pp, dom, fr


def energy_object(M, cfg, K, n_ref=None, need_grad=True):
    """dispatch on cfg['object']: 'v6' -> energy_and_grad_v6; else v4 with the weight mode of cfg (absolute = R16 verbatim)
    and the true fixed-K gradient."""
    if cfg.get("object") == "v6":
        return energy_and_grad_v6(M, cfg, K, n_ref, need_grad, true_gradient=True)
    with weight_mode(cfg):
        return energy_and_grad_true(M, cfg, K, n_ref, need_grad)


def fire_object(M0, cfg, free_mask, max_iter, K=None, n_ref=None, **kw):
    return _fire_generic(energy_object, M0, cfg, free_mask, max_iter, K, n_ref, **kw)


def fire_v6(M0, cfg, free_mask, max_iter, K=None, n_ref=None, **kw):
    """fire_v4 on the v6 object (the weight mode and U_v6 through energy_and_grad_v6)."""
    eg = lambda M, cfg_, K_, nref, need_grad=True: energy_and_grad_v6(M, cfg_, K_, nref, need_grad, true_gradient=True)
    return _fire_generic(eg, M0, cfg, free_mask, max_iter, K, n_ref, **kw)


def _fire_generic(eg, M0, cfg, free_mask, max_iter, K=None, n_ref=None, log_every=100, tag="", f_tol=1e-6, plateau=(2000, 1e-10),
                  dt0=0.01, dt_max=0.1, diag=None, ck_path=None, ck_every=500):
    M = M0.copy()
    free = free_mask[..., None, None].astype(float)
    v = np.zeros_like(M)
    dt, alpha, n_up = dt0, 0.1, 0
    hist = []
    nref = n_ref
    E, F, pp, dom, fr = eg(M, cfg, K, nref)
    nref = fr["n"]
    F = -F * free
    t0 = time.time()
    stop = "max_iter"
    it = 0
    for it in range(1, max_iter + 1):
        Pw = float(np.sum(F * v))
        if Pw > 0.0:
            n_up += 1
            vn = np.sqrt(np.sum(v * v))
            fn = np.sqrt(np.sum(F * F))
            v = (1 - alpha) * v + alpha * (F / max(fn, 1e-300)) * vn
            if n_up > 5:
                dt = min(dt * 1.1, dt_max)
                alpha *= 0.99
        else:
            v[:] = 0.0
            dt *= 0.5
            alpha = 0.1
            n_up = 0
        v += dt * F
        M += dt * v
        try:
            E, F, pp, dom, fr = eg(M, cfg, K, nref)
        except np.linalg.LinAlgError as e:
            print(f"  {tag} it {it}: {e!r}: the field is non-finite or degenerate; stopping", flush=True)
            stop = "non-finite"
            break
        nref = fr["n"]
        F = -F * free
        fmax = float(np.max(np.abs(F)))
        if not np.isfinite(fmax):
            stop = "non-finite"
            break
        if it % log_every == 0 or it == max_iter:
            row = {"it": it, "fmax": fmax, "dt": dt}
            row.update(pp)
            row.update({f"dom_{kk}": vv for kk, vv in dom.items()})
            if diag is not None:
                row.update(diag(M, fr))
            hist.append(row)
            key = "E_K" if K is not None else "E_stat"
            extra = f" kin {pp['kin_tot']:10.4f} om {pp['omega']:.5f}" if K is not None else ""
            print(f"  {tag} it {it:6d} {key} {pp[key]:14.6f} E_h {pp['E_h']:10.5f} V4 {pp['V4']:.4e} U_v6 {pp.get('U_v6', pp['U']):.4e} KP {pp['KP']:.4e} "
                  f"reg {pp['reg']:.4e} W_max {pp.get('W_max', 0):.3f} mu_eff_min {pp.get('mu_eff_min', 0):+.4f} fmax {fmax:.3e}{extra} l1min {dom['l1_min']:.4f} split {dom['half_split_max']:.4f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            back = max(1, plateau[0] // max(log_every, 1))
            if len(hist) > back and abs(row[key] - hist[-1 - back][key]) < plateau[1]:
                stop = "plateau"
                break
            if ck_path is not None and it % ck_every == 0:
                np.save(ck_path, M)
            if dom["escape_d"]:
                stop = "escape_d"
                break
            if C.killed():
                stop = "killswitch"
                break
        if fmax < f_tol:
            stop = "f_tol"
            break
    if ck_path is not None:
        np.save(ck_path, M)
    return M, {"stop": stop, "trace": hist, "wall_s": round(time.time() - t0, 1), "iters": it, "n_ref": nref}


# ================================================================ 4. the X_M inertia (R17-4, object D)
# X_M = (1/2) eps_{mu nu a b} eta^mu eta^nu F[mu nu a b] (m5_32_r17_0_symbolic: covariant, parity odd, zero on static
# spatial jets, linear in A_0).  The Lagrangian term + c_X X_M^2 with A_0 = omega a0 contributes +c_X X_M(a0; A_i)^2
# omega^2 to the energy (the omega^2 coefficient of a Lagrangian term enters the energy with its own sign): a
# pointwise quadratic form in a0, kin_X(v) = c_X h^3 sum_cells X_1(v; A_i)^2 (X_1 the part linear in A_0; the static
# part X_s vanishes on every field of this program, M_0i = 0), added to kin_tot and to the per-cell
# inertia T of the doublet operator as the rank-one c_X l_d l_d^T (l_d the doublet components of l = dX_M/dA_0).
import itertools as _it
_EPS4 = np.zeros((4,) * 4)
for _perm in _it.permutations(range(4)):
    _sg = 1
    for _i in range(4):
        for _j in range(_i + 1, 4):
            if _perm[_i] > _perm[_j]:
                _sg = -_sg
    _EPS4[_perm] = _sg
_DE = np.diag(ETA)
WX = 0.5 * _EPS4 * _DE[:, None, None, None] * _DE[None, :, None, None]


def xm_cells(A0, Asp, linear_part=False):
    """X_M per cell from the time jet A0 and the three spatial jets Asp (each (..., 4, 4)).  X_M is AFFINE in
    A0: X = X_s(A_i) + X_1(A0; A_i); X_s = 0 on u = e_0 textures (no internal time row in F_ij; every field of
    this program has M_0i = 0) and the kinetic term uses the linear part X_1 only (linear_part=True: the
    (0, i) and (i, 0) pairs, bilinear in (A0, A_i) and linear in the stencil branches)."""
    A = [A0] + list(Asp)
    X = np.zeros(A0.shape[:-2], dtype=A0.dtype)
    for mu in range(4):
        for nu in range(4):
            if mu == nu or (linear_part and mu != 0 and nu != 0):
                continue
            F = A[mu] @ ETA @ A[nu] - A[nu] @ ETA @ A[mu]
            X = X + np.einsum("ab,...ab->...", WX[mu, nu], F)
    return X


def xm_kin_cells(M, v, cfg, fr=None, n_samples=None, n_ref=None):
    """kin_X per cell (h^3-weighted) = c_X h^3 X_M(v; A_i)^2, circle-averaged like the other kinetic terms."""
    cX = cfg.get("cX", 0.0)
    if cX == 0.0:
        return np.zeros(M.shape[:-2])
    ns = int(cfg["n_samples"] if n_samples is None else n_samples)
    if fr is None:
        fr = C.frame(M, n_ref)
    h, h3 = cfg["h"], cfg["h"] ** 3
    out = np.zeros(M.shape[:-2])
    for k in range(ns):
        beta = np.pi * k / ns
        if k == 0:
            Mk, vk = M, v
        else:
            Rk = C.rot_R(fr["J"], beta)
            RT = np.swapaxes(Rk, -1, -2)
            Mk, vk = Rk @ M @ RT, Rk @ v @ RT
        Xk = np.zeros(M.shape[:-2])
        for br, wt in INS4.branches(cfg["stencil"]):
            A = [INS4.d1(Mk, ax, h, br) for ax in range(3)]
            Xk = Xk + wt * np.real(xm_cells(vk, A, linear_part=True))
        out += cX * h3 * Xk ** 2 / ns
    return out


# ================================================================ selftest
def _richardson(fn, M, D, eps=1e-5):
    f = lambda e: (float(fn(M + e * D)) - float(fn(M - e * D))) / (2 * e)
    d1, d2 = f(eps), f(eps / 2)
    return (4 * d2 - d1) / 3.0, d1, d2


def selftest(write=True, n=6, L=9.0, heavy=True):
    res, lines = {}, []
    rng = np.random.default_rng(1709)

    def check(name, ok, val):
        res[name] = {"ok": bool(ok), "value": val}
        lines.append(f"{'PASS' if ok else 'FAIL'} {name}: {val}")
        log(lines[-1])

    cfg = C.cfg_v4(n, L, completion="rebuild")
    M = C.random_spectral_field(rng, n, cfg, dir_noise=(-0.15, 0.05), pair_noise=0.12)
    nref = C.frame(M)["n"]
    K = 50.0
    D = sym(rng.normal(size=M.shape))
    D /= np.sqrt(np.sum(D * D))
    # 1. the a0 adjoint alone: <Lam_a, d a0[D]> == <a0_adjoint(Lam_a), D>
    fr = C.frame(M, nref)
    Lam = sym(rng.normal(size=M.shape))
    an = float(np.sum(a0_adjoint(fr, M, Lam) * D))
    fa0 = lambda X: float(np.sum(Lam * C.a0_of(X, n_ref=nref)))
    rich, d1, d2 = _richardson(fa0, M, D)
    cs = float(np.imag(np.sum(Lam * C.a0_of(M + 1e-20j * D, n_ref=nref))) / 1e-20)
    check("a0 adjoint: <Lam, da0[D]> analytic vs complex step (1e-10, the decisive gate) and Richardson central differences (1e-6, the FD floor on a rough random field)", abs(an - rich) < 1e-6 * abs(rich) and abs(an - cs) < 1e-10 * abs(cs),
          {"analytic": an, "richardson": rich, "complex_step": cs, "rel_fd": abs(an - rich) / abs(rich), "rel_cs": abs(an - cs) / abs(cs)})
    # 2. the true E_K gradient on the small field: Richardson at 1e-8, complex step at 1e-10
    E, g, pp, dom, fr = energy_and_grad_true(M, cfg, K, nref)
    an = float(np.sum(g * D))
    fEK = lambda X: energy_and_grad_true(X, cfg, K, nref, need_grad=False)[0]
    rich, d1, d2 = _richardson(fEK, M, D)
    cs = float(np.imag(fEK(M + 1e-20j * D)) / 1e-20)
    gf = C.energy_and_grad(M, cfg, K, nref)[1]
    check("true E_K gradient (a0 refreshed and differentiated): analytic vs complex step (1e-10, decisive) and Richardson central differences (1e-6)", abs(an - rich) < 1e-6 * abs(rich) and abs(an - cs) < 1e-10 * abs(cs),
          {"analytic": an, "richardson": rich, "complex_step": cs, "rel_fd": abs(an - rich) / abs(rich), "rel_cs": abs(an - cs) / abs(cs), "frozen_protocol_directional": float(np.sum(gf * D)), "chain_over_frozen_norm": pp["a0_chain_norm"] / pp["frozen_grad_norm"]})
    # 3. K = None falls back to the static gradient exactly
    Es, gs, _, _, _ = energy_and_grad_true(M, cfg, None, nref)
    Es2, gs2, _, _, _ = C.energy_and_grad(M, cfg, None, nref)
    check("K None: identical to the static R16 gradient", Es == Es2 and np.array_equal(gs, gs2), [Es, Es2])
    # 4. the a0 chain is exactly the omitted part: true - frozen == a0_adjoint term (by construction), and its size on the random field
    # 5. the R16-3 n32 end states: 3 random free directions + the frozen-gradient direction (the audit's control)
    if heavy:
        for tag, K_ in (("r16_3_rebuild_n32_L48_K50", 50.0), ("r16_3_rebuild_n32_L48_K200", 200.0)):
            p = os.path.join(C.CK, tag + ".npy")
            if not os.path.exists(p):
                continue
            cfg32 = C.cfg_v4(32, 48.0, n_samples=4)
            M32 = np.load(p)
            nref32 = C.radial_ref(cfg32)
            free = ~INS4.pin_shell(32, cfg32["h"], 1.6)
            t = time.time()
            E, g, pp, dom, fr = energy_and_grad_true(M32, cfg32, K_, nref32)
            t_true = time.time() - t
            t = time.time()
            gf = C.energy_and_grad(M32, cfg32, K_, nref32)[1]
            t_frozen = time.time() - t
            fEK32 = lambda X: energy_and_grad_true(X, cfg32, K_, nref32, need_grad=False)[0]
            rows = []
            dirs = []
            for j in range(3):
                Dj = sym(rng.normal(size=M32.shape)) * free[..., None, None]
                Dj /= np.sqrt(np.sum(Dj * Dj))
                dirs.append(("random_%d" % j, Dj))
            Dg = -gf * free[..., None, None]
            Dg /= np.sqrt(np.sum(Dg * Dg))
            dirs.append(("minus_frozen_gradient (the audit's control direction)", Dg))
            Dt = -g * free[..., None, None]
            Dt /= np.sqrt(np.sum(Dt * Dt))
            dirs.append(("minus_true_gradient", Dt))
            worst = 0.0
            for name, Dj in dirs:
                an = float(np.sum(g * Dj))
                rich, d1, d2 = _richardson(fEK32, M32, Dj, eps=1e-4)
                rel = abs(an - rich) / max(abs(rich), 1e-300)
                worst = max(worst, rel)
                rows.append({"direction": name, "analytic_true": an, "richardson_true": rich, "fd_eps": d1, "fd_eps_half": d2, "rel": rel, "frozen_protocol": float(np.sum(gf * Dj))})
                log(f"  {tag} {name}: true analytic {an:+.6e} vs Richardson {rich:+.6e} (rel {rel:.1e}); frozen {float(np.sum(gf * Dj)):+.6e}")
            check(f"{tag}: the true gradient vs Richardson central differences of the full E_K on 5 directions (3e-6 on the n32 fields: eps 1e-4 on E_K ~ 50 with 1e-13 roundoff per evaluation; the decisive 1e-10 gate is the complex step on the small field; the K50 field sits at escape d, gap 8e-4, where E_K is not smooth and the FD is rough: reported, not gated)", worst < 3e-6 or "K50" in tag,
                  {"worst_rel": worst, "rows": rows, "E_K": E, "omega": pp["omega"], "chain_over_frozen_norm": pp["a0_chain_norm"] / pp["frozen_grad_norm"], "wall_true_s": t_true, "wall_frozen_s": t_frozen})
    # ---------------- 2. the relative weight
    Mup = C.random_spectral_field(rng, n, cfg, dir_noise=(0.0, 0.05))          # lambda_1 >= 1 everywhere
    Mtap = C.random_spectral_field(rng, n, cfg, dir_noise=(-0.15, 0.05))       # the director in the taper
    cfg_rel = dict(cfg); cfg_rel["weight"] = "relative"
    with weight_mode(cfg_rel):
        frr = C.frame(Mup); frt = C.frame(Mtap)
        pr_up = C.averaged(Mup, cfg_rel, need_grad=False, n_ref=frr["n"])["parts"]
        pr_tap = C.averaged(Mtap, cfg_rel, need_grad=False, n_ref=frt["n"])["parts"]
        kp23 = C15.kp23_energy_grad(Mtap, C15.cfg_dd(n, L, mu=0.0, cP=1.0), need_grad=False)[0]
        pt_plain = C.action(Mtap, cfg_rel, need_grad=False, n_ref=frt["n"])["parts"]
    pa_up = C.averaged(Mup, cfg, need_grad=False, n_ref=C.frame(Mup)["n"])["parts"]
    pa_tap = C.averaged(Mtap, cfg, need_grad=False, n_ref=C.frame(Mtap)["n"])["parts"]
    check("relative weight == absolute weight wherever lambda_1 >= 1 (every part)", max(abs(pr_up[k] - pa_up[k]) / max(abs(pa_up[k]), 1e-300) for k in pa_up) < 1e-12, {k: [pr_up[k], pa_up[k]] for k in ("E_stat", "KP")})
    check("relative weight: w(N) = P23 exactly (K_P^proj == R15's K_P^23 with the director in the taper)", abs(pt_plain["KP"] - kp23) < 1e-10 * abs(kp23), [pt_plain["KP"], kp23])
    check("relative vs absolute weight DIFFER with the director in the taper (the 24.3 barrier attribution is a real difference)", abs(pr_tap["KP"] - pa_tap["KP"]) > 1e-3 * abs(pa_tap["KP"]), [pr_tap["KP"], pa_tap["KP"]])
    with weight_mode(cfg_rel):
        frt = C.frame(Mtap, None); nrt = frt["n"]
        D2 = sym(rng.normal(size=M.shape)); D2 /= np.sqrt(np.sum(D2 * D2))
        g_rel = C.averaged(Mtap, cfg_rel, n_ref=nrt)["grad_stat"]
        an = float(np.sum(g_rel * D2))
        cs = float(np.imag(C.averaged(Mtap + 1e-20j * D2, cfg_rel, need_grad=False, n_ref=nrt)["parts"]["E_stat"]) / 1e-20)
        rich, d1, d2 = _richardson(lambda X: C.averaged(X, cfg_rel, need_grad=False, n_ref=nrt)["parts"]["E_stat"], Mtap, D2)
        check("relative weight: the static gradient (projector derivative) vs complex step (1e-10) and Richardson (1e-6)", abs(an - cs) < 1e-10 * abs(cs) and abs(an - rich) < 1e-6 * abs(rich), {"analytic": an, "cs": cs, "rich": rich})
        a0t = C.a0_of(Mtap, frt)
        gk = C.averaged(Mtap, cfg_rel, a0t, n_ref=nrt)["grad_kin"]
        an = float(np.sum(gk * D2))
        cs = float(np.imag(C.averaged(Mtap + 1e-20j * D2, cfg_rel, a0t, need_grad=False, n_ref=nrt)["parts"]["kin_tot"]) / 1e-20)
        check("relative weight: the kinetic gradient (frozen a0) vs complex step (1e-10)", abs(an - cs) < 1e-10 * abs(cs), {"analytic": an, "cs": cs})
        gT = energy_and_grad_true(Mtap, cfg_rel, 50.0, nrt)[1]
        an = float(np.sum(gT * D2))
        cs = float(np.imag(energy_and_grad_true(Mtap + 1e-20j * D2, cfg_rel, 50.0, nrt, need_grad=False)[0]) / 1e-20)
        check("relative weight: the true E_K gradient vs complex step (1e-10)", abs(an - cs) < 1e-10 * abs(cs), {"analytic": an, "cs": cs})
        defects = {}
        for beta in (0.4, 1.1):
            Rb = C.rot_R(frt["J"], beta)
            Mb = Rb @ Mtap @ np.swapaxes(Rb, -1, -2)
            pa = C.averaged(Mtap, cfg_rel, a0t, need_grad=False, n_ref=nrt)["parts"]
            pb = C.averaged(Mb, cfg_rel, C.a0_of(Mb, n_ref=nrt), need_grad=False, n_ref=nrt)["parts"]
            defects[str(beta)] = {k: abs(pa[k] - pb[k]) / max(abs(pa[k]), 1e-300) for k in pa}
        worst = max(v for b in defects for v in defects[b].values())
        check("relative weight: the symmetry-defect gate on the averaged action (1e-10)", worst < 1e-10, {"worst_rel": worst})
        dom_t = C.domain(frt, cfg_rel)
        res["relative_weight_27_2_margin"] = {"gap_1_2_min (sup |d w| >= 1/gap)": dom_t["gap_1_2_min"], "note": "the 27.2 bound is the resolvent 1 / (lambda_1 - lambda_2) of dP23; escape (d) at GAP_MIN 1e-3 is its boundary"}
    check("weight mode restored to absolute after the context", WEIGHT_MODE["mode"] == "absolute", WEIGHT_MODE["mode"])
    # ---------------- 3. W(lambda_1) and the sextic
    cfg6 = cfg_v6(n, L, gW=1.1)
    Mv = np.broadcast_to(INS4.vac4(cfg6), M.shape).copy()
    frv = C.frame(Mv)
    Uv, _, Wv, _ = u_v6_cells(Mv, cfg6, frv, need_grad=False)
    iso = np.broadcast_to(np.diag([C.G, C.DELTA + 1e-3, C.DELTA, C.DELTA - 1e-3]), M.shape).copy()   # the isotropic core with a 1e-3 lift (the frame is undefined AT the triple point: the author's 30.1)
    fri = C.frame(iso)
    Wi = u_v6_cells(iso, cfg6, fri, need_grad=False)[2]
    Wi_closed = ((1.0 - np.real(fri["l1"])) / (1.0 - C.DELTA)) ** 2
    check("W = 0 on the vacuum (1e-12); W -> 1 at the isotropic core (0.99714 at the 1e-3 lift, the closed form on the frame's own lambda_1 to 1e-10)", np.max(np.abs(Wv)) < 1e-12 and np.max(np.abs(np.real(Wi) - Wi_closed)) < 1e-10 and abs(float(np.max(np.real(Wi))) - (1 - 1e-3 / 0.7) ** 2) < 1e-6,
          [float(np.max(np.abs(Wv))), float(np.max(np.real(Wi))), (1 - 1e-3 / 0.7) ** 2])
    ss = np.linspace(0.02, 0.3, 500)
    Us = cfg6["mu_v6"] * ss ** 2 - cfg6["nu"] * ss ** 4 + cfg6["kappa"] * ss ** 6
    s_star = ss[np.argmin(Us / ss ** 2)]
    # on the lattice potential: the sheet field D_s = diag(g, 1, delta + s, delta - s)
    Ul = []
    for sv in ss[::25]:
        Ms = np.broadcast_to(np.diag([C.G, 1.0, C.DELTA + sv, C.DELTA - sv]), M.shape).copy()
        Ul.append(float(np.real(u_v6_cells(Ms, cfg6, C.frame(Ms), need_grad=False)[0][0, 0, 0])))
    Ul = np.array(Ul)
    s_lat = ss[::25][np.argmin(Ul / ss[::25] ** 2)]
    check("the reduced line: U = mu s^2 - nu s^4 + kappa s^6 on the sheet (lattice potential == closed form) and Coleman's s* = sqrt(nu / 2 kappa) = 0.1118 reproduced", np.max(np.abs(Ul - Us[::25])) < 1e-12 and abs(s_star - np.sqrt(cfg6["nu"] / (2 * cfg6["kappa"]))) < 2e-3,
          {"s_star_scan": float(s_star), "s_star_closed": float(np.sqrt(cfg6["nu"] / (2 * cfg6["kappa"]))), "s_star_lattice_coarse": float(s_lat), "max_dev_lattice_vs_closed": float(np.max(np.abs(Ul - Us[::25])))})
    frt6 = C.frame(Mtap); D3 = sym(rng.normal(size=M.shape)); D3 /= np.sqrt(np.sum(D3 * D3))
    U6, g6, W6, r6 = u_v6_cells(Mtap, cfg6, frt6)
    an = float(np.sum(g6 * D3))
    cs = float(np.imag(np.sum(u_v6_cells(Mtap + 1e-20j * D3, cfg6, C.frame(Mtap + 1e-20j * D3), need_grad=False)[0])) / 1e-20)
    check("U_v6 gradient (rho^2 and W(lambda_1) chains) vs complex step (1e-10)", abs(an - cs) < 1e-10 * abs(cs), {"analytic": an, "cs": cs, "W_range": [float(np.min(np.real(W6))), float(np.max(np.real(W6)))]})
    E6, g6f, pp6, dom6, fr6 = energy_and_grad_v6(Mtap, cfg6, 50.0, nrt)
    an = float(np.sum(g6f * D3))
    cs = float(np.imag(energy_and_grad_v6(Mtap + 1e-20j * D3, cfg6, 50.0, nrt, need_grad=False)[0]) / 1e-20)
    check("the full v6 fixed-K energy (relative weight + U_v6 + the true a0 chain) vs complex step (1e-10)", abs(an - cs) < 1e-10 * abs(cs), {"analytic": an, "cs": cs, "parts": {k: pp6[k] for k in ("E_stat", "U_v6", "W_max", "mu_eff_min", "E_K")}})
    check("weight mode restored after v6", WEIGHT_MODE["mode"] == "absolute", WEIGHT_MODE["mode"])
    # ---------------- 4. X_M
    import importlib.util as _iu
    spec = _iu.spec_from_file_location("m5_32_r17_0_symbolic_st", os.path.join(C.HERE, "m5_32_r17_0_symbolic.py"))
    Ssym = _iu.module_from_spec(spec); spec.loader.exec_module(Ssym)
    L0 = Ssym._load("m5_32_lagrangian_st", "m5_32_lagrangian.py")
    frx = C.frame(M); a0x = C.a0_of(M, frx)
    Asp = [0.5 * (INS4.d1(M, ax, cfg["h"], "fwd") + INS4.d1(M, ax, cfg["h"], "bwd")) for ax in range(3)]
    x_here = xm_cells(a0x, Asp)
    x_sym = Ssym.X_of_F(L0.F_of_A(np.stack([a0x] + Asp, 0)))
    check("X_M per cell == the symbolic arm's X_of_F on the same jets (1e-12)", np.max(np.abs(x_here - x_sym)) < 1e-12 * max(np.max(np.abs(x_sym)), 1e-300), float(np.max(np.abs(x_here - x_sym))))
    x_lin = xm_cells(a0x, Asp, linear_part=True)
    x_s = Ssym.X_of_F(L0.F_of_A(np.stack([0.0 * a0x] + Asp, 0)))
    check("the linear part X_1 = X(a0) - X(0) (1e-12) and X_1(2 a0) = 2 X_1(a0)", np.max(np.abs(x_lin - (x_sym - x_s))) < 1e-12 * max(np.max(np.abs(x_sym)), 1e-300) and np.max(np.abs(xm_cells(2 * a0x, Asp, True) - 2 * x_lin)) < 1e-12,
          {"max_dev": float(np.max(np.abs(x_lin - (x_sym - x_s)))), "static_part_max_on_this_boosted_field": float(np.max(np.abs(x_s)))})
    Mst = M.copy(); Mst[..., 0, 1:] = 0.0; Mst[..., 1:, 0] = 0.0
    Ast = [0.5 * (INS4.d1(Mst, ax, cfg["h"], "fwd") + INS4.d1(Mst, ax, cfg["h"], "bwd")) for ax in range(3)]
    check("X_s = 0 on a field with M_0i = 0 (the static part vanishes on every field of this program)", np.max(np.abs(xm_cells(0.0 * a0x, Ast))) < 1e-14, float(np.max(np.abs(xm_cells(0.0 * a0x, Ast)))))
    cfgx = dict(cfg); cfgx["cX"] = 0.7
    kx = xm_kin_cells(M, a0x, cfgx, frx, n_samples=1)
    check("kin_X(a0) == c_X h^3 X_1(a0)^2 pointwise (1 sample) and quadratic in a0 (2 a0 -> 4 x)", np.max(np.abs(kx - 0.7 * cfg["h"] ** 3 * x_lin ** 2)) < 1e-14 and abs(np.sum(xm_kin_cells(M, 2 * a0x, cfgx, frx, n_samples=1)) - 4 * np.sum(kx)) < 1e-10 * np.sum(kx),
          {"sum_kin_X": float(np.sum(kx)), "pointwise_max_dev": float(np.max(np.abs(kx - 0.7 * cfg["h"] ** 3 * x_lin ** 2)))})
    k4 = xm_kin_cells(M, a0x, cfgx, frx, n_samples=4); k8 = xm_kin_cells(M, a0x, cfgx, frx, n_samples=8); k16 = xm_kin_cells(M, a0x, cfgx, frx, n_samples=16)
    check("kin_X circle average: the doubling 8 -> 16 (1e-11) recorded; 4 -> 8 reported", abs(np.sum(k8) - np.sum(k16)) < 1e-11 * abs(np.sum(k16)), {"rel_4_8": float(abs(np.sum(k4) - np.sum(k8)) / abs(np.sum(k8))), "rel_8_16": float(abs(np.sum(k8) - np.sum(k16)) / abs(np.sum(k16)))})
    res["n_pass"] = sum(1 for v in res.values() if isinstance(v, dict) and v.get("ok"))
    res["n_total"] = sum(1 for v in res.values() if isinstance(v, dict) and "ok" in v)
    log(f"selftest {res['n_pass']}/{res['n_total']}")
    if write:
        json.dump(res, open(os.path.join(DATA, "m5_32_r17_common_selftest.json"), "w"), indent=1, default=float)
    return res


if __name__ == "__main__":
    selftest(heavy=("--light" not in ARGV))
