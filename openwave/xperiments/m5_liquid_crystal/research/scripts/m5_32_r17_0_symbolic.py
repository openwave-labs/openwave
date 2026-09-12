"""M5.32 R17-0 (the symbolic arm, ledger 6.6): (a) the dual-reading falsifier's derivation, (e) the
X_M map onto the R1 registry with the section-82.1 divergence identity and the report's three claims
on our cores, (f) the section-66.1 Legendre identity at fixed momentum.  Every author number is a
claim; every derivation here is ours (sympy, exact) or numeric with a stated tolerance.

EQUATIONS FIRST
---------------
(a) The uniaxial identity (the report's 39.1, re-derived).  M_sp = delta I + (1 - delta) n n^T, n a unit
    director; at a point rotate so that n = e_3 and d_i n = a_i e_1 + b_i e_2 (six free symbols).  Then
    A_i = d_i M = (1 - delta)(d_i n n^T + n d_i n^T), F_ij = A_i A_j - A_j A_i (the 3 x 3 static sector, G = I),
    and the registry's static quartic I1 = sum_{i<j} tr(F_ij F_ij^T) satisfies EXACTLY
        I1 = 2 (1 - delta)^4 sum_{i<j} Omega_ij^2,   Omega_ij = n . (d_i n x d_j n) = a_i b_j - a_j b_i.
    On the unit hedgehog n = x / r: d_i n_a = (delta_ia - n_i n_a) / r, sum_{i<j} Omega_ij^2 = 1 / r^4, so the
    certified static density E_h = 4 I1 = 8 (1 - delta)^4 / r^4 =: A / r^4 (A = 1.92080 at delta 0.3): the
    tail amplitude the record arm measures at ratio 0.999 on our fields.
    The pair.  Read the tail as a Coulomb field energy: density A / r^4 = (1/2) E^2 with E = q / r^2 fixes
    q^2 = 2 A (this is the report's step, q^2 = 16 (1 - delta)^4).  The interaction energy of two such
    fields is the superposition cross term of the SAME density,
        U = int E_1 . E_2 d^3x = q_1 q_2 int (r - x_1).(r - x_2) / (|r - x_1|^3 |r - x_2|^3) d^3x = 4 pi q_1 q_2 / d
    (checked here by quadrature to 1e-6; the identity int grad(1/r_1) . grad(1/r_2) = 4 pi / d), so
        U = 4 pi q^2 / d = 8 pi A / d = 64 pi (1 - delta)^4 / d = 48.27 / d   at delta 0.3,
    the M5.21.4 form 64 pi c2 / d with c2 = A / 8.  The report writes U = q^2 / (4 pi d) with the SAME q
    (the Heaviside-Lorentz potential attached to a Gaussian field E = q / r^2 and a half-square density):
    the three conventions mix, and the result 4 (1 - delta)^4 / (pi d) = 0.30570 / d is the consistent
    answer divided by 16 pi^2.  In any one convention (Gaussian: density E^2 / 8 pi, U = q^2 / d;
    Heaviside-Lorentz: density E^2 / 2 with E = q / 4 pi r^2, U = q^2 / 4 pi d) the pair / tail ratio is 8 pi.
    Verdicts: NORMALIZATION_SUPERPOSITION if our derivation gives pair / tail = 8 pi (the report's
    1 / 2 pi is off by 16 pi^2), NORMALIZATION_REPORT if 1 / 2 pi, NEITHER otherwise.  The texture-level
    caveat is stated, not assumed: the field-level superposition is the report's own reading; the
    director texture of a like-charge pair is the charge-2 texture with an escape tube (M5.21.4), so the
    1/d law at the texture level is a separate question, which the record answers as a string form.
(e) X_M := (1/2) sum eps_{mu nu a b} eta^mu eta^nu F[mu, nu, a, b] (the R1 slot rule for one epsilon over F:
    eps-derivative pair eta, eps-internal pair delta; the E-family's pattern with ONE F: X_M is LINEAR
    in F, a pseudoscalar).  Checks: covariance under random SO(1,3) maps (drift), parity flip under one
    reflection, X_M = 0 on static purely spatial jets (F[ij, a, b] has no internal time row), the
    relation E1 = -2 X_M R with R = sum F[m, n, n, m] the double trace (so E1 is X_M times I6's root);
    X_M^2 against span{I1..I6} on random jets (least squares; the R0 audit's C3 theorem predicts IN_SPAN:
    two epsilons contract to a generalized delta, every resulting pairing is an R0 pairing) and its
    coefficients; the section-82.1 identity in our convention,
        J^mu := sum eps_{mu nu a b} eta^mu eta^nu (M eta A_nu)[a, b],   d_mu J^mu = X_M
    (exact because d_mu d_nu M is symmetric while eps is antisymmetric in mu nu), verified by sympy on a
    random quadratic polynomial M(t, x, y, z) and numerically on lattice jets.  On our cores: X_M per
    cell with A_0 = omega a0 (linear in omega by construction), int X_M d^3x and int |X_M| d^3x on the
    R16-1 n32 core and the R16-3 end states; l := dX_M / dA_0 per cell (X_M is linear in A_0: l_ab = X_M
    at A_0 = E_ab), its Frobenius norm, its entry pattern at the core cell and at the max-split cell,
    and the clock projection l . a0 / |a0|.  Pre-registered: on the RADIAL HEDGEHOG (a uniaxial texture
    with curl n = 0) l_ab = 4 sum_ik eps_ijk (A_i)_bk symmetrized over (a, b) VANISHES (the hedgehog's jets
    give an antisymmetric l), so on a near-radial core l is set by the split and by the departure from the
    radial texture: near zero on the R16-1 static core, nonzero on the R16-3 rotating end states.
    AUDIT CORRECTION (2026-09-08, R17-0 audit E3 REFUTED as first stated): the first docstring said "on a
    uniaxial texture"; the audit's own continuum evaluation shows l != 0 on a random uniaxial director with
    curl n != 0 (|l| up to 27 on jets of scale 2), the symmetric part of T[a, c] = -(1 - delta)[(grad n_c x n)_a
    + n_c (curl n)_a] vanishing only for the curl-free radial hedgehog.  The lattice-error attribution below
    therefore holds to the extent the relaxed core is radial and curl-free (R16-1: UNIAXIAL_RADIAL, the
    analytic seed as the control), not for uniaxial textures in general.
(f) L_c = L_0 + c R^2 with R = r + T . v affine in the velocities, L_0 = (1/2) v^T K_0 v - V.  Momentum
    p = K_0 v + 2 c R T, so R = R_0 / (1 + 2 c I) with R_0 = r + T^T K_0^-1 p, I = T^T K_0^-1 T, and the exact
    Hamiltonian at fixed p is H_c = H_0 - c R_0^2 / (1 + 2 c I), H_0 = (1/2) p^T K_0^-1 p + V (sympy, 2 dof).
    With r = 0 on one clock coordinate (K_0 = I_0, T = l, p = K): H = K^2 / (2 (I_0 + 2 c l^2)): the square
    adds 2 c l^2 to the inertia.  Rank-one caveat: with several coordinates only the projection of l on
    the clock direction acts on the clock's inertia.

usage: python3 m5_32_r17_0_symbolic.py
out:   data/m5_32_r17_0_symbolic.json, checkpoints/m5_32_r17/r17_0_symbolic.log
"""
from __future__ import annotations
import importlib.util
import json
import os
import sys
import time

import numpy as np
import sympy as sp

ARGS = list(sys.argv[1:])
sys.argv = [sys.argv[0]]
import m5_32_r16_common as C                              # noqa: E402
import m5_32_r16_4_symbol as S4                           # noqa: E402

C15, INS4 = C.C15, C.INS4
ETA = C.ETA
RES, DATA = C.RES, C.DATA
CK16 = C.CK
CK = os.path.join(RES, "checkpoints", "m5_32_r17")
os.makedirs(CK, exist_ok=True)
T0 = time.time()
LOG = open(os.path.join(CK, "r17_0_symbolic.log"), "a")
DELTA = C.DELTA
OUT = {"rung": "R17-0 symbolic"}


def log(m):
    line = f"[{time.time() - T0:8.1f}s] {m}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


def rel(p):
    return os.path.relpath(p, RES)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(C.HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ================================================================ (a)
def stage_a():
    log("(a) the uniaxial identity and the pair normalization")
    out = {}
    d = sp.Symbol("delta", positive=True)
    a = sp.symbols("a1:4", real=True)
    b = sp.symbols("b1:4", real=True)
    e1, e2, e3 = (sp.Matrix([1, 0, 0]), sp.Matrix([0, 1, 0]), sp.Matrix([0, 0, 1]))
    n = e3
    dn = [a[i] * e1 + b[i] * e2 for i in range(3)]
    A = [(1 - d) * (dn[i] * n.T + n * dn[i].T) for i in range(3)]
    I1 = 0
    Om2 = 0
    for i in range(3):
        for j in range(i + 1, 3):
            F = A[i] * A[j] - A[j] * A[i]
            I1 += (F * F.T).trace()
            Om = n.dot(dn[i].cross(dn[j]))
            Om2 += Om ** 2
    ident = sp.simplify(sp.expand(I1 - 2 * (1 - d) ** 4 * Om2))
    out["I1_minus_2(1-delta)^4_Omega2_pointwise"] = str(ident)
    out["identity_exact"] = bool(ident == 0)
    log(f"  I1 - 2 (1 - delta)^4 sum Omega_ij^2 = {ident} (exact: {ident == 0})")
    # the hedgehog: sum Omega_ij^2 = 1 / r^4
    x, y, z = sp.symbols("x y z", real=True)
    r = sp.sqrt(x ** 2 + y ** 2 + z ** 2)
    nv = sp.Matrix([x, y, z]) / r
    dnv = [nv.diff(v) for v in (x, y, z)]
    Om2h = 0
    for i in range(3):
        for j in range(i + 1, 3):
            Om2h += nv.dot(dnv[i].cross(dnv[j])) ** 2
    hedge = sp.simplify(Om2h * r ** 4)
    out["hedgehog_Omega2_r4"] = str(hedge)
    out["A_tail_8(1-delta)^4_at_0.3"] = 8.0 * (1.0 - DELTA) ** 4
    log(f"  hedgehog: sum Omega_ij^2 r^4 = {hedge}; E_h = 4 I1 = 8 (1 - delta)^4 / r^4, A = {8 * (1 - DELTA) ** 4:.5f}")
    # the cross term int E_1 . E_2 d^3x for unit charges a distance d apart, by quadrature: with charge 1 at the
    # origin, E_1 = r_hat / r^2 and the angular integral of E_2 . r_hat over the sphere of radius r is the FLUX of E_2
    # through it, Phi(r) = 4 pi for r > d and 0 for r < d (Gauss); so int E_1 . E_2 = int_0^inf Phi(r) / r^2 dr = 4 pi / d.
    # Both steps are checked numerically: Phi(r) by theta-quadrature at several r, the radial integral by quad.
    from scipy.integrate import quad
    dd = 1.0

    def flux(rr):
        f = lambda th: 2 * np.pi * rr * rr * np.sin(th) * (rr - dd * np.cos(th)) / (rr * rr + dd * dd - 2 * rr * dd * np.cos(th)) ** 1.5
        return quad(f, 0.0, np.pi, epsabs=1e-11, epsrel=1e-11, limit=200)[0]
    fl = {f"{rr:g}": float(flux(rr)) for rr in (0.3, 0.7, 0.95, 1.05, 1.5, 3.0, 10.0)}
    inner = quad(lambda rr: flux(rr) / (rr * rr), 0.0, dd, epsabs=1e-9, epsrel=1e-9, limit=200)[0]
    outer = quad(lambda rr: flux(rr) / (rr * rr), dd, np.inf, epsabs=1e-9, epsrel=1e-9, limit=200)[0]
    val = inner + outer
    out["cross_term_quadrature"] = {"d": dd, "flux_of_E2_through_sphere_r": fl, "inner_r_lt_d": float(inner), "outer_r_gt_d": float(outer), "value": float(val), "4pi_over_d": 4 * np.pi / dd,
                                    "rel_dev_from_4pi_over_d": float(abs(val - 4 * np.pi / dd) / (4 * np.pi / dd))}
    log(f"  int E_1 . E_2 d^3x (unit charges, d = 1): flux(r) {fl}; inner {inner:.3e} + outer {outer:.6f} = {val:.6f} vs 4 pi / d = {4 * np.pi:.6f} (rel dev {out['cross_term_quadrature']['rel_dev_from_4pi_over_d']:.2e})")
    A = 8.0 * (1.0 - DELTA) ** 4
    conv = {"report_chain": {"density": "A / r^4 = q^2 / (2 r^4)", "q2": 2 * A, "potential_written": "q^2 / (4 pi d)", "U_coefficient": 2 * A / (4 * np.pi), "equals_4(1-delta)^4/pi": 4 * (1 - DELTA) ** 4 / np.pi},
            "gaussian": {"density": "E^2 / (8 pi) with E = q / r^2", "q2": 8 * np.pi * A, "potential": "q^2 / d", "U_coefficient": 8 * np.pi * A},
            "heaviside_lorentz": {"density": "E^2 / 2 with E = q / (4 pi r^2)", "q2": 32 * np.pi ** 2 * A, "potential": "q^2 / (4 pi d)", "U_coefficient": 32 * np.pi ** 2 * A / (4 * np.pi)},
            "half_square_with_E_q_over_r2 (the report's field and density)": {"density": "E^2 / 2 with E = q / r^2", "q2": 2 * A, "potential (superposition int E1.E2)": "4 pi q^2 / d", "U_coefficient": 4 * np.pi * 2 * A}}
    for k in conv:
        conv[k]["pair_over_tail"] = conv[k]["U_coefficient"] / A
    out["conventions"] = conv
    ratio_ours = 8 * np.pi
    ratio_report = 1.0 / (2 * np.pi)
    out["pair_over_tail_ours"] = ratio_ours
    out["pair_over_tail_report"] = ratio_report
    out["factor"] = ratio_ours / ratio_report
    out["U_ours_at_0.3"] = 8 * np.pi * A
    out["U_report_at_0.3"] = 4 * (1 - DELTA) ** 4 / np.pi
    num_ratio = val * (2 * A) / A                   # U = q^2 int E1.E2 with q^2 = 2A (the report's own q), divided by A
    if abs(num_ratio - 8 * np.pi) < 1e-3 * 8 * np.pi:
        v = "NORMALIZATION_SUPERPOSITION"
    elif abs(num_ratio - ratio_report) < 1e-3 * ratio_report:
        v = "NORMALIZATION_REPORT"
    else:
        v = "NEITHER"
    out["numeric_pair_over_tail"] = float(num_ratio)
    out["verdict"] = v
    out["statement"] = ("the report's tail 8(1-delta)^4 is our E_h normalization exactly (record arm, ratio 0.999); its pair coefficient 4(1-delta)^4/(pi d) attaches a Heaviside-Lorentz potential q^2/(4 pi d) to a Gaussian field E = q/r^2 "
                        "with a half-square density; the superposition cross term of the same density gives 8 pi A / d = 48.27 / d at delta 0.3, so the report's number is the consistent one divided by 16 pi^2; "
                        "the record's certified like-charge pair is not a 1/d coefficient at all (it rises with d), so 'both numbers are already measured' is answered by what IS measured: the tail (yes, 1.9208) and a string-form like-charge pair (no 1/d)")
    log(f"  VERDICT {v}: pair / tail = 8 pi = {8 * np.pi:.4f} by our derivation (numeric {num_ratio:.4f}); the report's 1 / (2 pi) = {ratio_report:.5f}; factor {out['factor']:.3f} = 16 pi^2 = {16 * np.pi ** 2:.3f}")
    OUT["a"] = out


# ================================================================ (e)
def xm_weight():
    """W[mu, nu, a, b] = (1/2) eps_{mu nu a b} eta_mu eta_nu (the R1 rule: eps-derivative pair eta, eps-internal pair delta)."""
    import itertools
    E = np.zeros((4,) * 4)
    for perm in itertools.permutations(range(4)):
        sgn = 1
        for i in range(4):
            for j in range(i + 1, 4):
                if perm[i] > perm[j]:
                    sgn = -sgn
        E[perm] = sgn
    de = np.diag(ETA)
    return 0.5 * E * de[:, None, None, None] * de[None, :, None, None]


WX = xm_weight()


def X_of_F(F):
    return np.einsum("mnab,...mnab->...", WX, F)


def X_of_jets(A):
    """A: (4, ..., 4, 4) jets -> X_M per point."""
    L0 = STAGE_E["L0"]
    return X_of_F(L0.F_of_A(A))


STAGE_E = {}


def stage_e():
    log("(e) the X_M map onto the R1 registry")
    L0 = _load("m5_32_lagrangian_r17s", "m5_32_lagrangian.py")
    X = _load("m5_32_terms_ext_r17s", "m5_32_terms_ext.py")
    STAGE_E["L0"] = L0
    out = {"definition": "X_M = (1/2) sum_{mu nu a b} eps_{mu nu a b} eta^mu eta^nu F[mu, nu, a, b], F_mu_nu = A_mu eta A_nu - A_nu eta A_mu (slots 0,1 derivative, 2,3 internal; eps-d pair eta, eps-i pair delta: the R1 E-family rule with one F)"}
    rng = np.random.default_rng(1707)
    p = L0.default_params(s=-1.0, g=32.0)
    A, M = L0._random_jets(rng, 200, p)
    F = L0.F_of_A(A)
    x0 = X_of_F(F)
    sc = float(np.max(np.abs(x0)))
    drift = 0.0
    for kind in ("boost", "boost", "boost", "rotation", "rotation", "rotation"):
        Lm = L0._lorentz(rng, kind)
        Ap, Mp = X._transform(Lm, A, M)
        drift = max(drift, float(np.max(np.abs(X_of_F(L0.F_of_A(Ap)) - x0)) / sc))
    Pref = np.diag([1.0, -1.0, 1.0, 1.0])
    Ap, Mp = X._transform(Pref, A, M)
    x1 = X_of_F(L0.F_of_A(Ap))
    out["covariance_drift_SO13_SO3"] = drift
    out["parity_flip_x_reflection"] = float(np.max(np.abs(x1 + x0)) / sc)
    out["parity_keep_x_reflection"] = float(np.max(np.abs(x1 - x0)) / sc)
    log(f"  covariance drift {drift:.2e}; parity: |X' + X| / scale {out['parity_flip_x_reflection']:.2e} (odd), |X' - X| {out['parity_keep_x_reflection']:.2e}")
    # static purely spatial jets: X_M = 0
    As = A.copy(); As[0] = 0.0
    As[1:, :, 0, :] = 0.0; As[1:, :, :, 0] = 0.0
    out["X_on_static_spatial_jets_max"] = float(np.max(np.abs(X_of_F(L0.F_of_A(As)))))
    # E1 = -2 X_M R ?
    R = np.einsum("...mnnm->...", F)
    e1 = X.REGISTRY_EXT["E1"].density(A, M, p)
    out["E1_vs_minus2_X_R"] = {"max_rel_dev": float(np.max(np.abs(e1 + 2 * x0 * R)) / max(np.max(np.abs(e1)), 1e-300)), "note": "E1 = F[pqrs] F[xyxy] eps[pqrs]: the second factor is minus the double trace R (I6 = R^2), the first is 2 X_M"}
    log(f"  static spatial jets: max |X| {out['X_on_static_spatial_jets_max']:.2e}; E1 = -2 X_M R to {out['E1_vs_minus2_X_R']['max_rel_dev']:.2e}")
    # X_M^2 vs span{I1..I6}
    names = ["I1", "I2", "I3", "I4", "I5", "I6"]
    V = np.array([L0.REGISTRY[n].density(A, M, p) for n in names]).T
    scale = np.max(np.abs(V), axis=0)
    c, *_ = np.linalg.lstsq(V / scale, x0 ** 2, rcond=None)
    res = float(np.max(np.abs((V / scale) @ c - x0 ** 2)) / max(np.max(np.abs(x0 ** 2)), 1e-300))
    coeff = {n: float(c[i] / scale[i]) for i, n in enumerate(names)}
    out["X2_on_span_I1_I6"] = {"coefficients": coeff, "max_rel_residual": res, "verdict": "IN_SPAN" if res < 1e-8 else "NEW_TERM"}
    # the E-family: X_M^2 against E1..E3 alone (parity even vs odd: must fail) and the joint span
    VE = np.array([X.REGISTRY_EXT[n].density(A, M, p) for n in ("E1", "E2", "E3")]).T
    cE, *_ = np.linalg.lstsq(VE / np.max(np.abs(VE), axis=0), x0 ** 2, rcond=None)
    resE = float(np.max(np.abs((VE / np.max(np.abs(VE), axis=0)) @ cE - x0 ** 2)) / max(np.max(np.abs(x0 ** 2)), 1e-300))
    out["X2_on_span_E1_E3_control"] = {"max_rel_residual": resE, "expected": "fails (X_M^2 is parity even, E1..E3 odd)"}
    # X_M itself against E1..E3 (linear vs quadratic in F: must fail) and R1's basis hash for the pattern
    cX, *_ = np.linalg.lstsq(VE / np.max(np.abs(VE), axis=0), x0, rcond=None)
    out["X_on_span_E1_E3_control"] = {"max_rel_residual": float(np.max(np.abs((VE / np.max(np.abs(VE), axis=0)) @ cX - x0)) / sc), "expected": "fails (X_M is linear in F, the E-terms quadratic)"}
    out["R1_registry_hashes"] = {n: X.REGISTRY_EXT[n].hash for n in ("E1", "E2", "E3")}
    log(f"  X_M^2 on span(I1..I6): residual {res:.2e} -> {out['X2_on_span_I1_I6']['verdict']}; coefficients {coeff}; control on E1..E3 residual {resE:.2e}")
    # the divergence identity by sympy on a random quadratic polynomial M(t, x, y, z)
    t, x, y, z = sp.symbols("t x y z", real=True)
    co = (t, x, y, z)
    rr = np.random.default_rng(5).integers(-3, 4, size=(4, 4, 15))
    mons = [1, t, x, y, z, t * t, x * x, y * y, z * z, t * x, t * y, t * z, x * y, x * z, y * z]
    Msym = sp.zeros(4, 4)
    for a_ in range(4):
        for b_ in range(a_, 4):
            e = sum(int(rr[a_, b_, k]) * mons[k] for k in range(15))
            Msym[a_, b_] = e; Msym[b_, a_] = e
    ETAs = sp.diag(-1, 1, 1, 1)
    Aj = [Msym.diff(v) for v in co]
    Fs = {}
    for mu in range(4):
        for nu in range(4):
            Fs[(mu, nu)] = Aj[mu] * ETAs * Aj[nu] - Aj[nu] * ETAs * Aj[mu]
    Xs = 0
    Js = [0, 0, 0, 0]
    for mu in range(4):
        for nu in range(4):
            for a_ in range(4):
                for b_ in range(4):
                    w = WX[mu, nu, a_, b_]
                    if w == 0.0:
                        continue
                    Xs += sp.Rational(int(round(2 * w)), 2) * Fs[(mu, nu)][a_, b_]
                    Js[mu] += sp.Rational(int(round(2 * w)), 2) * (Msym * ETAs * Aj[nu])[a_, b_]
    divJ = sum(Js[mu].diff(co[mu]) for mu in range(4))
    ident = sp.expand(2 * divJ - Xs)
    ident_half = sp.expand(divJ - Xs)
    out["divergence_identity_sympy"] = {"d_mu J_report^mu - X_M": str(ident), "exact": bool(ident == 0), "J_report_definition": "J^mu = sum eps_{mu nu a b} eta^mu eta^nu (M eta A_nu)[a, b] (the report's J, no 1/2; our slot rule)",
                                       "with_the_half_weighted_J": f"d_mu (J/2)^mu - X_M = {str(ident_half)[:80]}... (nonzero: the factor 2 is the antisymmetrization of F, F = A eta A - (A eta A)^T)",
                                       "note": "exact for any smooth M: d_mu d_nu M is symmetric, eps antisymmetric in (mu, nu); the report's 82.1 statement holds in our convention with the report's J"}
    out["divergence_identity_verdict"] = "TOTAL_DERIVATIVE" if ident == 0 else "NOT_A_TOTAL_DERIVATIVE"
    log(f"  d_mu J^mu - X_M = {ident} -> {out['divergence_identity_verdict']}")
    OUT["e"] = out
    # ---------------- on our cores
    cores = {}
    fields = [("analytic_seed_n32_CONTROL", None, 32, 48.0, None), ("analytic_seed_n64_CONTROL", None, 64, 48.0, None),
              ("r16_1_end_n32", os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"), 32, 48.0, None), ("r16_1_end_n64", os.path.join(CK16, "r16_1_rebuild_n64_L48_analytic.npy"), 64, 48.0, None)]
    r163 = json.load(open(os.path.join(DATA, "m5_32_r16_3.json")))
    for tag, rec in r163["runs"].items():
        n = rec["n"]; L = rec["L"]
        om = rec["end_parts_8"]["omega"] if "end_parts_8" in rec else rec["dE_dK"]["omega_end"]
        fields.append((tag, os.path.join(CK16, tag + ".npy"), n, L, om))
    basis = []
    labels = []
    for a_ in range(4):
        for b_ in range(a_, 4):
            E = np.zeros((4, 4)); E[a_, b_] = E[b_, a_] = 1.0
            basis.append(E); labels.append(f"{a_}{b_}")
    for lab, p_, n, L, om in fields:
        cfg = C.cfg_v4(n, L, n_samples=8)
        M = C15.seed_uniaxial(cfg) if p_ is None else np.load(p_)
        h3 = cfg["h"] ** 3
        Xc, Yc, Zc = INS4.coords(n, cfg["h"])
        r = np.sqrt(Xc * Xc + Yc * Yc + Zc * Zc)
        fr = C.frame(M, C.radial_ref(cfg))
        a0 = C.a0_of(M, fr)
        Asp = S4.jets_at(M, cfg)
        rec = {"field": rel(p_) if p_ else "C15.seed_uniaxial (the exact uniaxial hedgehog: l = 0 in the continuum; what remains is the lattice error of the central-difference jets)", "omega_record": om, "h": cfg["h"]}
        # X_M per cell at omega in {0, 0.1, 0.2} (+ the record omega)
        oms = [0.0, 0.1, 0.2] + ([om] if om else [])
        xs = {}
        for w in oms:
            Aj4 = np.stack([w * a0] + Asp, 0)
            xc = X_of_F(L0.F_of_A(Aj4))
            xs[f"{w:.4f}"] = {"int_X_h3": float(h3 * np.sum(xc)), "int_abs_X_h3": float(h3 * np.sum(np.abs(xc))), "max_abs_X": float(np.max(np.abs(xc))), "r_at_max": float(r.reshape(-1)[int(np.argmax(np.abs(xc)))])}
        rec["X_by_omega"] = xs
        x1, x2 = xs["0.1000"]["int_abs_X_h3"], xs["0.2000"]["int_abs_X_h3"]
        rec["linear_in_omega_check_ratio_0.2_over_0.1"] = float(x2 / max(x1, 1e-300))
        # l = dX / dA_0 per cell (linear): l_ab = X at A_0 = E_ab (symmetric unit)
        lmat = np.zeros(M.shape)
        for E, lb in zip(basis, labels):
            Aj4 = np.stack([np.broadcast_to(E, M.shape)] + Asp, 0)
            xc = X_of_F(L0.F_of_A(Aj4))
            a_, b_ = int(lb[0]), int(lb[1])
            lmat[..., a_, b_] = xc
            lmat[..., b_, a_] = xc
        # off-diagonal entries: the unit E_ab (a != b) has both entries 1, so l_ab (the derivative w.r.t. the symmetric pair) is xc; kept as the pair derivative
        lnorm = np.sqrt(np.sum(lmat * lmat, axis=(-1, -2)))
        a0n = np.sqrt(np.sum(a0 * a0, axis=(-1, -2)))
        proj = np.sum(lmat * a0, axis=(-1, -2)) / np.maximum(a0n, 1e-300)
        spl, _ = C15.split_cells(M, need_grad=False)
        half = np.sqrt(np.maximum(np.real(spl), 0.0)) / 2.0
        idx_core = int(np.argmin(r)); idx_split = int(np.argmax(half))
        sh = M.shape[:3]
        cc, cs_ = np.unravel_index(idx_core, sh), np.unravel_index(idx_split, sh)
        rec["l_norm_max"] = float(np.max(lnorm)); rec["r_at_l_norm_max"] = float(r.reshape(-1)[int(np.argmax(lnorm))])
        rec["l_at_core_cell"] = {"r": float(r[cc]), "half_split": float(half[cc]), "entries": {labels[k]: float(lmat[cc][int(labels[k][0]), int(labels[k][1])]) for k in range(10)}, "norm": float(lnorm[cc]), "clock_projection": float(proj[cc]), "a0_norm": float(a0n[cc])}
        rec["l_at_max_split_cell"] = {"r": float(r[cs_]), "half_split": float(half[cs_]), "entries": {labels[k]: float(lmat[cs_][int(labels[k][0]), int(labels[k][1])]) for k in range(10)}, "norm": float(lnorm[cs_]), "clock_projection": float(proj[cs_]), "a0_norm": float(a0n[cs_])}
        # the pre-registered proportionality l ~ split: correlation of |l| with the half split over the free cells
        free = ~INS4.pin_shell(n, cfg["h"], 1.6)
        rec["corr_l_norm_vs_half_split"] = float(np.corrcoef(lnorm[free].reshape(-1), half[free].reshape(-1))[0, 1])
        rec["l_norm_over_half_split_median_where_split_gt_1e-3"] = float(np.median((lnorm / np.maximum(half, 1e-300))[free & (half > 1e-3)])) if np.any(free & (half > 1e-3)) else None
        rec["int_l_dot_a0hat_h3"] = float(h3 * np.sum(proj))
        rec["int_l_norm_h3"] = float(h3 * np.sum(lnorm))
        for a_, b_ in ((3.0, 6.0), (6.0, 12.0)):
            mk = free & (r >= a_) & (r < b_)
            rec[f"median_l_norm_r_{a_:g}_{b_:g}"] = float(np.median(lnorm[mk]))
        rec["l_norm_on_free_max"] = float(np.max(lnorm[free]))
        cores[lab] = rec
        log(f"  {lab}: int|X| at omega 0.1 {x1:.3e}, 0.2 {x2:.3e} (ratio {rec['linear_in_omega_check_ratio_0.2_over_0.1']:.3f}); int X at omega 0.1 {xs['0.1000']['int_X_h3']:+.3e}; |l| max {rec['l_norm_max']:.3e} at r {rec['r_at_l_norm_max']:.2f}; "
            f"core cell |l| {rec['l_at_core_cell']['norm']:.3e} proj {rec['l_at_core_cell']['clock_projection']:.3e} (split {rec['l_at_core_cell']['half_split']:.4f}); max-split cell |l| {rec['l_at_max_split_cell']['norm']:.3e} proj {rec['l_at_max_split_cell']['clock_projection']:.3e} (split {rec['l_at_max_split_cell']['half_split']:.4f}); corr(|l|, split) {rec['corr_l_norm_vs_half_split']:.3f}")
    ctrl32, ctrl64 = cores["analytic_seed_n32_CONTROL"], cores["analytic_seed_n64_CONTROL"]
    OUT["e_lattice_error_of_l"] = {"median_l_norm_r_3_6_h1.5": ctrl32["median_l_norm_r_3_6"], "median_l_norm_r_3_6_h0.75": ctrl64["median_l_norm_r_3_6"], "ratio_h1.5_over_h0.75": ctrl32["median_l_norm_r_3_6"] / max(ctrl64["median_l_norm_r_3_6"], 1e-300),
                                   "statement": "l = dX_M / dA_0 vanishes identically on the RADIAL (curl-free) hedgehog in the continuum (the antisymmetric l of its jets; the R17-0 audit refuted the broader 'any uniaxial texture' wording: curl n != 0 gives l != 0); on the lattice the central-difference jets leave a residual that falls with h (the ratio above), so on a near-radial core |l| at the level of the control rows is discretization, not a clock coupling"}
    OUT["e_on_our_cores"] = cores
    OUT["e_claims"] = {"X_M_zero_on_static_hedgehog": "confirmed (structural: static spatial jets have no internal time row)",
                       "X_over_omega_linear": "linear by construction (F_0i is linear in A_0 = omega a0); the report's 5.7135e-2 is a value on the report's own configuration, not reproducible without it",
                       "dX_dA0_nonzero_at_12_13_23_value_0.3479": "checked on OUR cores above (the entry pattern and the clock projection); the report's value belongs to its own configuration"}


# ================================================================ (f)
def stage_f():
    log("(f) the fixed-momentum Legendre identity")
    c, r_, V = sp.symbols("c r V", real=True)
    k11, k12, k22 = sp.symbols("k11 k12 k22", real=True)
    t1, t2 = sp.symbols("t1 t2", real=True)
    v1, v2, p1, p2 = sp.symbols("v1 v2 p1 p2", real=True)
    K0 = sp.Matrix([[k11, k12], [k12, k22]])
    T = sp.Matrix([t1, t2]); v = sp.Matrix([v1, v2]); p = sp.Matrix([p1, p2])
    R = r_ + (T.T * v)[0]
    L = (sp.Rational(1, 2) * (v.T * K0 * v)[0] - V) + c * R ** 2
    pv = sp.Matrix([sp.diff(L, v1), sp.diff(L, v2)])
    sol = sp.solve([pv[0] - p1, pv[1] - p2], [v1, v2], dict=True)[0]
    H = ((p.T * v)[0] - L).subs(sol)
    K0i = K0.inv()
    I_ = (T.T * K0i * T)[0]
    R0 = r_ + (T.T * K0i * p)[0]
    H0 = sp.Rational(1, 2) * (p.T * K0i * p)[0] + V
    claim = H0 - c * R0 ** 2 / (1 + 2 * c * I_)
    diff = sp.simplify(H - claim)
    out = {"H_c_minus_claim": str(diff), "exact": bool(diff == 0), "claim": "H_c = H_0 - c R_0^2 / (1 + 2 c I), I = T^T K_0^-1 T, R_0 = r + T^T K_0^-1 p"}
    # the clock case: r = 0, one coordinate, K_0 = I_0, T = l, p = K
    I0, l, K = sp.symbols("I0 l K", positive=True)
    Hc = (K ** 2 / (2 * I0)) - c * (l * K / I0) ** 2 / (1 + 2 * c * l ** 2 / I0)
    out["clock_case"] = {"H": str(sp.simplify(Hc)), "equals_K2_over_2(I0+2cl2)": bool(sp.simplify(Hc - K ** 2 / (2 * (I0 + 2 * c * l ** 2))) == 0)}
    out["rank_one_caveat"] = "with several velocity coordinates the added inertia is 2 c l l^T (rank one); only the projection of l on the clock direction changes the clock's inertia, the rest couples the clock to the tilt directions"
    log(f"  H_c - [H_0 - c R_0^2 / (1 + 2cI)] = {diff} (exact {diff == 0}); clock case H = K^2 / (2 (I_0 + 2 c l^2)): {out['clock_case']['equals_K2_over_2(I0+2cl2)']}")
    OUT["f"] = out


if __name__ == "__main__":
    stage_a()
    json.dump(OUT, open(os.path.join(CK, "r17_0_symbolic_partial.json"), "w"), indent=1, default=float)
    stage_e()
    json.dump(OUT, open(os.path.join(CK, "r17_0_symbolic_partial.json"), "w"), indent=1, default=float)
    stage_f()
    OUT["wall_s"] = time.time() - T0
    json.dump(OUT, open(os.path.join(DATA, "m5_32_r17_0_symbolic.json"), "w"), indent=1, default=float)
    log(f"written data/m5_32_r17_0_symbolic.json ({OUT['wall_s']:.0f} s)")
