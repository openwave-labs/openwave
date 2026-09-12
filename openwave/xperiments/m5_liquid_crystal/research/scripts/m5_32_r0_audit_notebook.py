"""M5.32 R0 arm (c): INDEPENDENT ADVERSARIAL AUDIT of the author's boost-hedgehog notebook.

Source: ``theory/duda_2026-08-17_newton_for_boost_hedgehogs.pdf`` (2 pages). This script
re-implements the notebook from the PDF alone (the producer's script and JSON were NOT read)
and tries to refute the producer's claims K1-K6.

EQUATIONS (what the notebook does, transcribed from the PDF)
------------------------------------------------------------
Vacuum and generators (4x4, index 0 = time)::

    M0   = diag(g, 0, 0, 0)                       (audit variant K4: diag(g, 1, delta, 0))
    Gb_j = symmetric, entry 1 at (0, j) and (j, 0), j = 1..3   (boost generators)

Hedgehog pair, half separation d, centers at z = -d (o1) and z = +d (o2)::

    f(r) = 1/sqrt(r)  applied to the SQUARED distance, so the profile is 1/R (unit vector)
    o1   = exp(m f(R1^2) (x, y, z+d) . Gb) = 1 + m a1 . Gb + O(m^2),  a1 = (x, y, z+d)/R1
    o2   = exp(m f(R2^2) (x, y, z-d) . Gb) = 1 + m a2 . Gb + O(m^2),  a2 = (x, y, z-d)/R2
    o    = o1 o2,   M = o M0 o^T,   dM_i = d M / d x_i

At O(m) the field is the sum of the two unit-vector hedgehogs a = a1 + a2 and

    M1 = m g (a . Gb)   (entries m g a_b at (0,b), (b,0)),   dM_i = m g (d_i a) . Gb

Plain commutators and the notebook's spatial curvature density (Out[13] at y = 0)::

    com_ij = dM_i dM_j - dM_j dM_i ,  (i,j) in {(x,y), (y,z), (z,x)}
    Hs     = sum_{(i,j)} com_ij[1,2]^2 + com_ij[2,3]^2 + com_ij[3,1]^2      (0-based entries)
           = m^4 g^4 sum_{ij} sum_{b<c} ( d_i a_b d_j a_c - d_j a_b d_i a_c )^2
           = m^4 g^4 |cof(J)|_F^2 ,   J_{ib} = d_i a_b       (sum of the 9 squared 2x2 minors)

Energy (the notebook integrates the half space z > 0 with weight 4 pi x, which by the
z -> -z symmetry of Hs equals the full-space integral)::

    E(d) = int_{x>0, z>0} 4 pi x Hs [x^2 + (z-d)^2 > cut2] dx dz / (m^4 g^4),   cut2 = 1e-3
    fit  E = A + B/d over d = 0.1, 0.2, ..., 3.0   (notebook: A = 863.733, B = 167.668)

Single-hedgehog check: J = (1 - n n)/R gives cof(J) = n n / R^2, so Hs_self = 1/R^4 and the
self energy is 4 pi / sqrt(cut2) per hedgehog, E(inf) = 8 pi / sqrt(cut2) exactly.

Eta-commutator curvature (claim K3)::

    xi   = diag(-1, 1, 1, 1)
    F_ij = dM_i xi dM_j - dM_j xi dM_i
    <F,F>_eta = tr(eta F eta F^T)

Far-field prediction tested in K5: near hedgehog 1 the cross term 2 cof(J1).mixed(J1,J2) scales
as 1/(R^3 d), whose radial integral is logarithmic, so E_int(d) should carry a (ln d)/d piece
with coefficient 2 * 4 pi * (2/3) * (1/d) per hedgehog pair = 32 pi / 3 / d, not a pure 1/d.

Outputs: ``research/data/m5_32_r0_audit_notebook.json``.
Run: ``/opt/anaconda3/envs/master312/bin/python3 m5_32_r0_audit_notebook.py``
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "m5_32_r0_audit_notebook.json"

# --------------------------------------------------------------------------------------
# 1. Symbolic pipeline (faithful to the notebook, sympy instead of Mathematica)
# --------------------------------------------------------------------------------------
x, y, z, d, g, m, delta = sp.symbols("x y z d g m delta", real=True)


def boost_generators():
    gens = []
    for j in (1, 2, 3):
        G = sp.zeros(4, 4)
        G[0, j] = 1
        G[j, 0] = 1
        gens.append(G)
    return gens


def series_matrix(Mx, order):
    """Truncate every entry of a matrix polynomial in m at m**order (Normal[Series[...]])."""
    return Mx.applyfunc(lambda e: sum(sp.expand(e).coeff(m, k) * m**k for k in range(order + 1)))


def build_M(M0, o_order, M_order):
    """o = o1 o2 with o1, o2 = 1 + m X_k (notebook truncation) and o kept to m**o_order,
    then M = o M0 o^T kept to m**M_order."""
    Gb = boost_generators()
    R1 = sp.sqrt(x**2 + y**2 + (z + d) ** 2)
    R2 = sp.sqrt(x**2 + y**2 + (z - d) ** 2)
    X1 = sum((((x, y, z + d)[k] / R1) * Gb[k] for k in range(3)), sp.zeros(4, 4))
    X2 = sum((((x, y, z - d)[k] / R2) * Gb[k] for k in range(3)), sp.zeros(4, 4))
    o1 = sp.eye(4) + m * X1
    o2 = sp.eye(4) + m * X2
    o = series_matrix(o1 * o2, o_order)
    M = series_matrix(o * M0 * o.T, M_order)
    return M


def grads(M):
    return [M.diff(v) for v in (x, y, z)]


def plain_coms(dM):
    return [dM[i] * dM[j] - dM[j] * dM[i] for (i, j) in ((0, 1), (1, 2), (2, 0))]


def eta_coms(dM):
    xi = sp.diag(-1, 1, 1, 1)
    return [dM[i] * xi * dM[j] - dM[j] * xi * dM[i] for (i, j) in ((0, 1), (1, 2), (2, 0))]


def Hs_from_coms(coms):
    return sum((c[1, 2] ** 2 + c[2, 3] ** 2 + c[3, 1] ** 2 for c in coms), sp.Integer(0))


# --------------------------------------------------------------------------------------
# 2. Fast numeric integrand: Hs / (m^4 g^4) = |cof(J)|^2 (weighted for general M0)
# --------------------------------------------------------------------------------------
def jacobian_sum(xx, yy, zz, dd):
    """J_{ib} = d_i (a1 + a2)_b with a_k the unit-vector hedgehogs about z = -d, +d."""
    J = np.zeros((3, 3))
    for zc in (-dd, dd):
        r = np.array([xx, yy, zz - zc])
        R = math.sqrt(r @ r)
        n = r / R
        J += (np.eye(3) - np.outer(n, n)) / R
    return J


def hs_numeric(xx, zz, dd, w=(1.0, 1.0, 1.0)):
    """sum over (i,j) and (b,c) in {(0,1),(1,2),(2,0)} of w_bc^2 * minor^2 at y = 0.
    w_bc = (g + lam_b)(g + lam_c) / g^2 for M0 = diag(g, lam); w = 1 for the notebook."""
    J = jacobian_sum(xx, 0.0, zz, dd)
    tot = 0.0
    for (i, j) in ((0, 1), (1, 2), (2, 0)):
        for wk, (b, c) in zip(w, ((0, 1), (1, 2), (2, 0))):
            mn = J[i, b] * J[j, c] - J[j, b] * J[i, c]
            tot += (wk * mn) ** 2
    return tot


def energy(dd, cut2, w=(1.0, 1.0, 1.0)):
    """E = int_{z>0, x>0} 4 pi x Hs [x^2+(z-d)^2 > cut2] dx dz, in polar coordinates about
    the +d center: x = rho sin th, z = d + rho cos th, rho > sqrt(cut2), z > 0."""
    rc = math.sqrt(cut2)

    def inner(th):
        s, c = math.sin(th), math.cos(th)
        if c < 0:
            rmax = dd / (-c)
            if rmax <= rc:
                return 0.0
        else:
            rmax = np.inf

        def f(rho):
            return 4.0 * math.pi * rho * rho * s * hs_numeric(rho * s, dd + rho * c, dd, w)

        # split at a few radii so the adaptive rule sees the core and the other hedgehog
        pts = [rc, 2 * rc, 10 * rc, dd, 2 * dd, 4 * dd, 10 * dd]
        pts = sorted(set(p for p in pts if rc <= p < rmax))
        val = 0.0
        for a, b in zip(pts, pts[1:] + [rmax]):
            v, _ = integrate.quad(f, a, b, limit=200, epsabs=1e-10, epsrel=1e-10)
            val += v
        return val

    # theta split at pi/2 (the ray toward the other hedgehog is theta = pi)
    val = 0.0
    for a, b in ((0.0, math.pi / 2), (math.pi / 2, 3 * math.pi / 4), (3 * math.pi / 4, math.pi)):
        v, _ = integrate.quad(inner, a, b, limit=200, epsabs=1e-9, epsrel=1e-9)
        val += v
    return val


def fit_lin(ds, es, basis):
    A = np.array([[f(dv) for f in basis] for dv in ds])
    coef, *_ = np.linalg.lstsq(A, np.array(es), rcond=None)
    return [float(c) for c in coef]


# --------------------------------------------------------------------------------------
# 3. Out[13] transcribed from the PDF
# --------------------------------------------------------------------------------------
def out13(xx, zz, dd):
    S1 = math.sqrt(dd**2 + xx**2 - 2 * dd * zz + zz**2)
    S2 = math.sqrt(dd**2 + xx**2 + 2 * dd * zz + zz**2)
    P = S1 * S2
    num = 8 * (
        dd**8
        + dd**6 * (2 * xx**2 - 2 * zz**2 + P)
        + (xx**2 + zz**2) ** 3 * (xx**2 + zz**2 + P)
        + dd**2 * (xx**4 - zz**4) * (2 * xx**2 + 2 * zz**2 + P)
        + dd**4 * (4 * xx**4 + 2 * zz**4 - zz**2 * P + xx**2 * (4 * zz**2 + P))
    )
    den = (dd**2 + xx**2 - 2 * dd * zz + zz**2) ** 3 * (dd**2 + xx**2 + 2 * dd * zz + zz**2) ** 3
    return num / den  # Hs / (m^4 g^4)


def main():
    t0 = time.time()
    res = {"source": "theory/duda_2026-08-17_newton_for_boost_hedgehogs.pdf", "claims": {}}
    rng = np.random.default_rng(20260827)

    # ---- K1 / K6: symbolic pipeline vs numeric cofactor form vs Out[13] ----
    M0 = sp.diag(g, 0, 0, 0)
    M = build_M(M0, o_order=1, M_order=1)
    dM = grads(M)
    Hs_sym = Hs_from_coms(plain_coms(dM))
    Hs_sym_m4 = sp.expand(Hs_sym).coeff(m, 4)
    Hs_fn = sp.lambdify((x, y, z, d, g), Hs_sym_m4, "numpy")
    pts = []
    for _ in range(3):
        px, pz, pd = rng.uniform(0.2, 2.0), rng.uniform(-2.0, 2.0), rng.uniform(0.1, 3.0)
        pg = 32.0
        v_sym = float(Hs_fn(px, 0.0, pz, pd, pg)) / pg**4
        v_num = hs_numeric(px, pz, pd)
        v_o13 = out13(px, pz, pd)
        pts.append(
            {"x": px, "z": pz, "d": pd, "sympy_pipeline": v_sym, "cofactor_form": v_num,
             "out13_pdf": v_o13, "rel_err_sympy_vs_out13": abs(v_sym - v_o13) / abs(v_o13),
             "rel_err_cof_vs_out13": abs(v_num - v_o13) / abs(v_o13)}
        )
    # Hs is homogeneous m^4: any other power present?
    other_orders = [k for k in range(0, 9) if k != 4 and sp.expand(Hs_sym).coeff(m, k) != 0]
    # z -> -z symmetry of Hs at y = 0 (justifies half-space x 2 = full space)
    sym_chk = max(abs(hs_numeric(a, b, c) - hs_numeric(a, -b, c)) / hs_numeric(a, b, c)
                  for a, b, c in [(0.3, 0.7, 0.5), (1.1, -0.2, 1.3), (0.05, 2.0, 0.4)])
    res["claims"]["K1_K6"] = {
        "points": pts, "Hs_orders_in_m_other_than_4": other_orders,
        "Hs_z_reflection_max_rel_asym": sym_chk,
        "single_hedgehog_Hs_times_R4_at_d_large": hs_numeric(0.5, 100.0 + 0.3, 100.0) * (0.5**2 + 0.3**2) ** 2,
    }
    print("K1/K6 points:", json.dumps(pts, indent=1))
    print("orders other than m^4:", other_orders, "| z-reflection asym:", sym_chk)

    # ---- K2 / K5: energies and fits ----
    ds = [round(0.1 * k, 10) for k in range(1, 31)]
    fits = {}
    E_tab = {}
    for cut2 in (1e-3, 1e-4, 1e-2):
        t1 = time.time()
        es = [energy(dv, cut2) for dv in ds]
        A, B = fit_lin(ds, es, [lambda v: 1.0, lambda v: 1.0 / v])
        Einf = 8 * math.pi / math.sqrt(cut2)
        fits[str(cut2)] = {"A": A, "B": B, "E_inf_exact": Einf, "seconds": time.time() - t1}
        E_tab[str(cut2)] = es
        print(f"cut2={cut2:g}: A={A:.3f} B={B:.3f}  E_inf={Einf:.3f}  ({time.time()-t1:.0f}s)")
    A0, B0 = fits["0.001"]["A"], fits["0.001"]["B"]
    res["claims"]["K2"] = {
        "notebook": {"A": 863.733, "B": 167.668},
        "audit": {"A": A0, "B": B0},
        "pct_diff_A": 100 * (A0 - 863.733) / 863.733,
        "pct_diff_B": 100 * (B0 - 167.668) / 167.668,
        "E_table_cut2_1e-3": dict(zip([str(v) for v in ds], E_tab["0.001"])),
    }
    res["claims"]["K5_cutoffs"] = {"fits_0.1_to_3.0": fits, "producer": {"1e-4": 211.4, "1e-3": 167.7, "1e-2": 109.1}}

    # ---- K5 physics: is the far field 1/d? ----
    cut2 = 1e-3
    Einf = 8 * math.pi / math.sqrt(cut2)
    es = E_tab["0.001"]
    outer = [(dv, ev) for dv, ev in zip(ds, es) if dv >= 1.5 - 1e-9]
    od, oe = zip(*outer)
    fit_ABC = fit_lin(od, oe, [lambda v: 1.0, lambda v: 1 / v, lambda v: 1 / v**2])
    fit_ABL = fit_lin(od, oe, [lambda v: 1.0, lambda v: 1 / v, lambda v: math.log(v) / v])
    fit_AB = fit_lin(od, oe, [lambda v: 1.0, lambda v: 1 / v])
    # fixed intercept: E_inf known exactly; log-log slope of E - E_inf
    ll = np.polyfit(np.log(od), np.log(np.array(oe) - Einf), 1)
    # wide range with exact intercept: d E_int vs ln d
    wide_d = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    wide_E = [energy(dv, cut2) for dv in wide_d]
    wide_int = [e - Einf for e in wide_E]
    dEint = [dv * ei for dv, ei in zip(wide_d, wide_int)]
    slope_ln = np.polyfit(np.log(wide_d[2:]), dEint[2:], 1)
    ll_wide = np.polyfit(np.log(wide_d[2:]), np.log(wide_int[2:]), 1)
    # cutoff dependence of d*E_int at fixed large d (log(cut) prediction: 32 pi/3 * ln(rc ratio))
    dE_cut = {}
    for c2 in (1e-4, 1e-2):
        Ei = energy(8.0, c2) - 8 * math.pi / math.sqrt(c2)
        dE_cut[str(c2)] = 8.0 * Ei
    pred = 32 * math.pi / 3
    res["claims"]["K5_far_field"] = {
        "outer_window_d": list(od),
        "fit_A_B_over_d_outer": fit_AB,
        "fit_A_B_over_d_C_over_d2_outer": fit_ABC,
        "fit_A_B_over_d_Clog_over_d_outer": fit_ABL,
        "loglog_slope_E_minus_Einf_outer": float(ll[0]),
        "E_inf_exact": Einf,
        "wide_d": wide_d, "wide_E_int": wide_int, "wide_d_times_E_int": dEint,
        "slope_of_d_Eint_vs_ln_d_(d>=4)": float(slope_ln[0]),
        "predicted_log_coefficient_32pi_over_3": pred,
        "loglog_slope_wide_(d>=4)": float(ll_wide[0]),
        "d_Eint_at_d8_vs_cut2": {"0.001": dEint[3], **dE_cut},
        "predicted_delta_per_decade_of_cut2": pred * math.log(math.sqrt(10)),
    }
    print("outer fits AB:", fit_AB, "ABC:", fit_ABC, "ABlog:", fit_ABL, "loglog:", ll[0])
    print("wide:", list(zip(wide_d, wide_int, dEint)), "slope vs ln d:", slope_ln[0], "pred:", pred)
    print("d*Eint(d=8) vs cut2:", res["claims"]["K5_far_field"]["d_Eint_at_d8_vs_cut2"])

    # ---- K3: time row of F (eta commutator) ----
    def time_row_orders(M0m, o_order, M_order, kmax):
        Mm = build_M(M0m, o_order, M_order)
        dMm = grads(Mm)
        F = eta_coms(dMm)
        subs = {x: sp.Rational(3, 7), y: sp.Rational(-2, 5), z: sp.Rational(5, 9), d: sp.Rational(7, 10),
                g: 32, delta: sp.Rational(3, 10)}
        out = {}
        for k in range(2, kmax + 1):
            mx = 0
            for Fi in F:
                for a in range(4):
                    e0a = sp.expand(Fi[0, a]).coeff(m, k)
                    ea0 = sp.expand(Fi[a, 0]).coeff(m, k)
                    for e in (e0a, ea0):
                        mx = max(mx, abs(float(sp.N(e.subs(subs), 30))))
            out[f"m^{k}"] = mx
        # spatial block antisymmetry and eta contraction vs Hs at m^4
        Fm2 = [Fi.applyfunc(lambda e: sp.expand(e).coeff(m, 2)) for Fi in F]
        anti = max(abs(float(sp.N((Fi[b, c] + Fi[c, b]).subs(subs)))) for Fi in Fm2 for b in range(1, 4) for c in range(1, 4))
        eta = sp.diag(-1, 1, 1, 1)
        contr = sum(((eta * Fi * eta * Fi.T).trace() for Fi in Fm2), sp.Integer(0))
        hs = Hs_from_coms([Fi for Fi in Fm2])  # same entries as plain com up to sign -> squares equal
        ratio = float(sp.N((contr / hs).subs(subs)))
        # plain vs eta spatial block: relation
        dMp = [Mi.applyfunc(lambda e: sp.expand(e).coeff(m, 1)) for Mi in dMm]
        Pc = plain_coms(dMp)
        rel = max(abs(float(sp.N((Fm2[q][b, c] + Pc[q][b, c]).subs(subs)))) for q in range(3) for b in range(1, 4) for c in range(1, 4))
        return out, anti, ratio, rel

    tr_trunc, anti, ratio, rel = time_row_orders(sp.diag(g, 0, 0, 0), 1, 1, 4)
    tr_full, _, _, _ = time_row_orders(sp.diag(g, 0, 0, 0), 2, 2, 4)
    tr_full_gen, _, _, _ = time_row_orders(sp.diag(g, 1, delta, 0), 2, 2, 4)
    res["claims"]["K3"] = {
        "time_row_max_abs_by_order_M_truncated_O(m)": tr_trunc,
        "time_row_max_abs_by_order_M_kept_to_O(m2)_(o as printed in Out[11])": tr_full,
        "time_row_max_abs_by_order_M0_diag(g,1,delta,0)_O(m2)": tr_full_gen,
        "spatial_block_antisymmetry_max_violation_m2": anti,
        "eta_contraction_over_Hs_at_m4": ratio,
        "max_abs(F_eta_spatial + com_plain_spatial)_m2": rel,
    }
    print("K3:", json.dumps(res["claims"]["K3"], indent=1))

    # ---- K4: general vacuum spectrum ----
    M0g = sp.diag(g, 1, delta, 0)
    Mg = build_M(M0g, 1, 1)
    Hs_g = sp.expand(Hs_from_coms(plain_coms(grads(Mg)))).coeff(m, 4)
    Hs_g_fn = sp.lambdify((x, y, z, d, g, delta), Hs_g, "numpy")
    gv, dv_ = 32.0, 0.3
    lam = (1.0, dv_, 0.0)
    w = tuple((gv + lam[b]) * (gv + lam[c]) / gv**2 for (b, c) in ((0, 1), (1, 2), (2, 0)))
    chk = []
    for _ in range(3):
        px, pz, pd = rng.uniform(0.2, 2.0), rng.uniform(-2.0, 2.0), rng.uniform(0.1, 3.0)
        vs = float(Hs_g_fn(px, 0.0, pz, pd, gv, dv_)) / gv**4
        vn = hs_numeric(px, pz, pd, w)
        chk.append(abs(vs - vn) / abs(vs))
    es_g = [energy(dv, 1e-3, w) for dv in ds]
    Ag, Bg = fit_lin(ds, es_g, [lambda v: 1.0, lambda v: 1.0 / v])
    res["claims"]["K4"] = {"g": gv, "delta": dv_, "weights_w_bc": w, "weighted_form_rel_err_vs_sympy": max(chk),
                           "A": Ag, "B": Bg, "sign_B": "positive" if Bg > 0 else "negative"}
    print("K4:", res["claims"]["K4"])

    res["runtime_seconds"] = time.time() - t0
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT, f"in {res['runtime_seconds']:.0f}s")


if __name__ == "__main__":
    main()
