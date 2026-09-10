"""M5.32 R19 pre-registration check: the mixed-bracket proposal, algebra only.

The author's 2026-09-10 proposal (Discussion #186, discussioncomment-18379326,
and the same text to the list): keep the commutator field strength for the
spatial / EM sector and use the ANTICOMMUTATOR for the time / gravity sector,
so that the mediator carries even spin (attractive) where Newton needs it and
odd spin (repulsive) where Coulomb needs it. This script settles, on random
jets and before any relaxation, what the proposal IS in the registry's terms.
No relaxation, no lattice: every line is an identity or a rank statement.

EQUATIONS FIRST
---------------
Jets A_mu = d_mu M (symmetric 4x4), eta = diag(-1, 1, 1, 1),
    P_mu nu = A_mu eta A_nu                       (the product, not symmetric)
    F_mu nu = P_mu nu - P_nu mu                   (the certified commutator)
    G_mu nu = P_mu nu + P_nu mu                   (the anticommutator)
    <X, Y>_eta = sum_ab eta_a eta_b X_ab Y_ab
Densities (the FULL derivative contraction; sum_{mu<nu} is a complete
Lorentz contraction only for objects antisymmetric in (mu, nu), which G is not):
    I1  = (1/2) sum_{mu nu} eta^mu eta^nu <F_mu nu, F_mu nu>_eta   (== the registry's I1)
    GG  = (1/2) sum_{mu nu} eta^mu eta^nu <G_mu nu, G_mu nu>_eta
    PP  = (1/2) sum_{mu nu} eta^mu eta^nu <P_mu nu, P_mu nu>_eta
Three readings of "the time / gravity sector" (the author does not say which):
    (T) the TIME-INDEX split: F for (i, j), G for (0, i) and (0, 0)
    (B) the INTERNAL boost-block split, fixed frame: P_s F P_s on the
        spatial-spatial internal block, G on the time row / column / corner
        (P_t = e_0 e_0^T, P_s = 1 - P_t)
    (A) the pure anticommutator everywhere (GG replaces I1)
Checks (each a number in the JSON, each falsifiable):
    C1  symmetries: F antisymmetric in both pairs, G symmetric in both pairs
        (so G is NOT a 2-form and not the Savvidy structure "antisymmetric
        in the first pair, symmetric in the rest"); (A eta)(B eta) + (B eta)(A eta)
        is not symmetric (the "mixed" result of the author's symbolic test is
        the index placement, the covariant A eta B + B eta A is symmetric)
    C2  the entrywise identity G^2 + F^2 = 2 (P^2 + P^T2), hence GG + I1 = 4 PP
    C3  GG is OUTSIDE the frozen basis {I1..I6, C6a, C6b}: rank 8 -> 9
    C4  GG is Lorentz invariant (rotation and boost); (B) is rotation
        invariant and NOT boost invariant (a covariant version needs the
        field's timelike eigenvector u, the R2 h = eta + 2 u u machinery)
    C5  static sector (A_0 = 0): (T) equals I1 EXACTLY, so no static
        quantity (R0, R3, R11 Newton reads) can change under (T)
    C6  Coulomb sector (A_0 = 0, time row zero): (B) equals I1 EXACTLY, so
        the G1 Coulomb gate is preserved by construction under (B)
    C7  boost sector (static, time row present): (B) = I1 - 2 I(F_t) + 4 I(P_t)
        with I(F_t) the boost block of I1 (negative on every sample: the
        record's boost weight) and I(G_t) indefinite; the fraction of random
        static jets with a NEGATIVE (B) density (the boundedness risk)
    C8  omega degree on the rigid clock A_0 = omega a0: I1 quadratic, GG /
        (T) / (B) QUARTIC (the Legendre bridge E = C omega^2 - A and the
        fixed-J algebra J = 2 kin omega no longer hold as stated)
Out: ../data/m5_32_r19_precheck.json
"""
from __future__ import annotations
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(DATA, "m5_32_r19_precheck.json")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = [argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


LAG = _load("m5_32_lagrangian", "m5_32_lagrangian.py")
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
ETA_D = np.array([-1.0, 1.0, 1.0, 1.0])
W = np.einsum("m,k->mk", ETA_D, ETA_D)
PT = np.zeros((4, 4)); PT[0, 0] = 1.0
PS = np.eye(4) - PT


def sym(rng, n):
    X = rng.normal(size=(n, 4, 4))
    return X + X.swapaxes(-1, -2)


def P_of(A):
    return np.einsum("mnab,bc,kncd->mknad", A, ETA, A)


def ip(X, Y):
    return np.einsum("a,b,mknab,mknab->mkn", ETA_D, ETA_D, X, Y)


def Ifull(X):
    return 0.5 * np.einsum("mk,mkn->n", W, ip(X, X))


def parts(A):
    P = P_of(A)
    F = P - P.swapaxes(0, 1)
    G = P + P.swapaxes(0, 1)
    T = F.copy(); T[0] = G[0]; T[:, 0] = G[:, 0]                  # (T)
    B = PS @ F @ PS + PT @ G @ PS + PS @ G @ PT + PT @ G @ PT       # (B)
    return P, F, G, T, B


def transform(L, A):
    """M -> L M L^T on the internal indices, d_mu -> (L^-T) on the jets."""
    return np.einsum("mk,knab,ca,db->mncd", np.linalg.inv(L).T, A, L, L)


def boost(v):
    Bm = np.eye(4); g = 1.0 / np.sqrt(1.0 - v * v)
    Bm[0, 0] = g; Bm[0, 1] = Bm[1, 0] = -g * v; Bm[1, 1] = g
    return Bm


def rotation(th):
    Rm = np.eye(4); c, s = np.cos(th), np.sin(th)
    Rm[1, 1] = c; Rm[1, 2] = -s; Rm[2, 1] = s; Rm[2, 2] = c
    return Rm


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def in_span(t, cols, names):
    X = np.stack(cols, 1)
    c = np.linalg.lstsq(X, t, rcond=None)[0]
    r = t - X @ c
    return {"rel_resid": float(np.linalg.norm(r) / np.linalg.norm(t)),
            "coef": {n: float(v) for n, v in zip(names, c)}}


def main(n_jets=400, seed=11):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    A = np.stack([sym(rng, n_jets) for _ in range(4)])
    P, F, G, T, B = parts(A)
    out = {"n_jets": n_jets, "seed": seed}
    # C1 symmetries and index placement
    Q = np.einsum("mnab,bc,kncd,de->mknae", A, ETA, A, ETA); Qs = Q + Q.swapaxes(0, 1)
    out["C1"] = {"F_antisym_munu": float(np.abs(F + F.swapaxes(0, 1)).max()),
                 "F_antisym_ab": float(np.abs(F + F.swapaxes(-1, -2)).max()),
                 "G_sym_munu": float(np.abs(G - G.swapaxes(0, 1)).max()),
                 "G_sym_ab": float(np.abs(G - G.swapaxes(-1, -2)).max()),
                 "G_00_max": float(np.abs(G[..., 0, 0]).max()),
                 "F_00_max": float(np.abs(F[..., 0, 0]).max()),
                 "(Aeta)(Beta)+(Beta)(Aeta)_asym_rel": rel(Qs, Qs.swapaxes(-1, -2)) if False else
                 float(np.abs(Qs - Qs.swapaxes(-1, -2)).max() / np.abs(Qs).max()),
                 "F_eta_in_so13": float(np.abs(ETA @ (F @ ETA) + ((F @ ETA).swapaxes(-1, -2)) @ ETA).max() / np.abs(F).max())}
    # C2 identity
    I1 = Ifull(F); GG = Ifull(G); PP = Ifull(P)
    Fa = LAG.F_of_A(A)
    dens = {k: LAG.density_from_K(Fa, LAG.REGISTRY[k]._K()) for k in ["I1", "I2", "I3", "I4", "I5", "I6"]}
    out["C2"] = {"registry_I1_vs_Ifull(F)": rel(dens["I1"], I1), "GG+I1-4PP_rel": rel(GG + I1, 4 * PP)}
    # C3 span
    AE = np.einsum("mnab,bc->mnac", A, ETA); tr2 = np.einsum("mnab,knba->mkn", AE, AE)
    C6a = np.einsum("m,mmn->n", ETA_D, tr2) ** 2
    C6b = np.einsum("mk,mkn->n", W, tr2 ** 2)
    cols = [dens[k] for k in dens] + [C6a, C6b]; names = list(dens) + ["C6a", "C6b"]
    tol = 1e-9 * np.abs(GG).max()
    out["C3"] = {"GG_in_span": in_span(GG, cols, names),
                 "rank_basis": int(np.linalg.matrix_rank(np.stack(cols, 1), tol=tol)),
                 "rank_basis_plus_GG": int(np.linalg.matrix_rank(np.stack(cols + [GG], 1), tol=tol)),
                 "PP_in_span": in_span(PP, cols, names)["rel_resid"]}
    # C4 invariances
    out["C4"] = {}
    for name, L in [("rotation_0.7", rotation(0.7)), ("boost_0.3", boost(0.3))]:
        P2, F2, G2, T2, B2 = parts(transform(L, A))
        out["C4"][name] = {"I1": rel(Ifull(F2), I1), "GG": rel(Ifull(G2), GG),
                           "T_timeindex": rel(Ifull(T2), Ifull(T)), "B_internal_fixed_frame": rel(Ifull(B2), Ifull(B))}
    # C5 static sector
    As = A.copy(); As[0] = 0.0
    Ps_, Fs, Gs, Ts, Bs = parts(As)
    out["C5"] = {"G_0mu_static_max": float(np.abs(Gs[0]).max()),
                 "T_minus_I1_static_max": float(np.abs(Ifull(Ts) - Ifull(Fs)).max()),
                 "I1_static_scale": float(np.abs(Ifull(Fs)).max())}
    # C6 Coulomb sector
    Ac = As.copy(); Ac[:, :, 0, :] = 0.0; Ac[:, :, :, 0] = 0.0
    _, Fc, Gc, Tc, Bc = parts(Ac)
    out["C6"] = {"B_minus_I1_coulomb_max": float(np.abs(Ifull(Bc) - Ifull(Fc)).max()),
                 "I1_coulomb_scale": float(np.abs(Ifull(Fc)).max()),
                 "GG_minus_I1_coulomb_rel": rel(Ifull(Gc), Ifull(Fc))}
    # C7 boost sector
    Ft = PT @ Fs @ PS + PS @ Fs @ PT
    Gt = PT @ Gs @ PS + PS @ Gs @ PT + PT @ Gs @ PT
    Fss = PS @ Fs @ PS
    Ptb = Ifull(PT @ Ps_ @ PS) + Ifull(PS @ Ps_ @ PT) + Ifull(PT @ Ps_ @ PT)
    out["C7"] = {"B=I1-2I(Ft)+4I(Pt)_rel": rel(Ifull(Bs), Ifull(Fs) - 2 * Ifull(Ft) + 4 * Ptb),
                 "I(Ft)_range": [float(Ifull(Ft).min()), float(Ifull(Ft).max())],
                 "I(Gt)_range": [float(Ifull(Gt).min()), float(Ifull(Gt).max())],
                 "I(Fss)_range": [float(Ifull(Fss).min()), float(Ifull(Fss).max())],
                 "frac_B_negative_static": float(np.mean(Ifull(Bs) < 0.0)),
                 "frac_GG_negative_static": float(np.mean(Ifull(Gs) < 0.0)),
                 "frac_I1_negative_static": float(np.mean(Ifull(Fs) < 0.0))}
    # C8 omega degree
    a0 = sym(rng, n_jets); ws = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    V = np.vander(ws, 5, increasing=True)
    out["C8"] = {}
    for which in ["I1", "GG", "T", "B"]:
        vals = []
        for wv in ws:
            Aw = A.copy(); Aw[0] = wv * a0
            Pw, Fw, Gw, Tw, Bw = parts(Aw)
            vals.append({"I1": Ifull(Fw), "GG": Ifull(Gw), "T": Ifull(Tw), "B": Ifull(Bw)}[which])
        coef = np.linalg.solve(V, np.stack(vals, 1).T)
        out["C8"][which] = {f"c{k}_max": float(np.abs(coef[k]).max()) for k in range(5)}
    out["runtime_s"] = time.time() - t0
    # verdict lines (each can fail)
    v = {}
    v["C1_G_not_a_2form"] = out["C1"]["G_sym_munu"] == 0.0 and out["C1"]["F_antisym_munu"] == 0.0
    v["C2_identity"] = out["C2"]["GG+I1-4PP_rel"] < 1e-12
    v["C3_GG_new_operator"] = out["C3"]["rank_basis_plus_GG"] == out["C3"]["rank_basis"] + 1 and out["C3"]["GG_in_span"]["rel_resid"] > 0.1
    v["C4_GG_invariant"] = max(out["C4"][k]["GG"] for k in out["C4"]) < 1e-12
    v["C4_B_not_boost_invariant"] = out["C4"]["boost_0.3"]["B_internal_fixed_frame"] > 0.1 and out["C4"]["rotation_0.7"]["B_internal_fixed_frame"] < 1e-12
    v["C5_T_static_equals_I1"] = out["C5"]["T_minus_I1_static_max"] == 0.0
    v["C6_B_coulomb_equals_I1"] = out["C6"]["B_minus_I1_coulomb_max"] == 0.0
    v["C7_I(Ft)_negative_everywhere"] = out["C7"]["I(Ft)_range"][1] < 0.0
    v["C7_B_indefinite_static"] = 0.0 < out["C7"]["frac_B_negative_static"] < 1.0
    v["C8_I1_quadratic_GG_T_B_quartic"] = out["C8"]["I1"]["c4_max"] < 1e-8 * out["C8"]["I1"]["c2_max"] and all(out["C8"][k]["c4_max"] > 1e-3 * out["C8"][k]["c2_max"] for k in ["GG", "T", "B"])
    out["verdict"] = {k: bool(x) for k, x in v.items()}
    os.makedirs(DATA, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    for k, x in out["verdict"].items():
        print(f"{'PASS' if x else 'FAIL'} {k}")
    print(json.dumps({k: out[k] for k in ["C3", "C5", "C6", "C7"]}, indent=1))
    return out


if __name__ == "__main__":
    main()
