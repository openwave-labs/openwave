"""
M5.8.2f2 — Track C C2 (option B): the CORE-LOCALIZED clock as a phase-twisted
eigenframe at a single global ω — the breathing/clock state's second
variational posing, on the manifold C1 proved necessary.

C1 (m5_8_2f) closed the rigid-global-clock + radial-boost-dressing ansatz
NEGATIVE-DECISIVE (no interior ghost minimum at any (β, p_Θ) — the 2b
net-cancellation survives the quartic variationally). C2-B changes the
manifold, not the machinery:

    W(x,t) = O_hh(x) · B(b(r)) · R_(δ,0)(Θ(t) + ψ(r)) ,   M = W D₄ Wᵀ

a STATIC radial clock-phase twist ψ(r) under a RIGID rotation Θ(t)=ωt —
strictly periodic at one ω (the 2e-measured signature: ω₀ + exact harmonic
comb), every radius at the same rate. Option A (differential rotation
R(Θ·φ(r))) is NOT a steady ansatz — its spatial twist Θ·φ′ winds up linearly
in time — and is deferred to a winding-cost diagnostic.

EXACT STRUCTURE (what makes C2 cheap on the C1 machinery):
  ∂_iM = A_i + s·x̂_i·Mb + t·x̂_i·M_T ,   s = b′(r) ,  t = ψ′(r)
with [Mb,Mb] = [M_T,M_T] = 0 and the s·t cross-commutators cancelling
⇒ F_ij is exactly LINEAR in (s,t) jointly ⇒
  v (signed spatial curvature): 6 coeffs over {1,s,t,s²,st,t²}
  q = u² = 4v²:                 15 coeffs (total degree ≤ 4)
  kd (clock inertia):           3 coeffs (1,s,s²) — t DROPS OUT ⇒ K[b] ≡ C1's K
Phase-averaging washes the absolute phase ⇒ only t(r) = ψ′ enters — and t is
a FREE function (any ψ′ realizable; ψ = ∫t). C1's extractability obstruction
(s = b′ couples radii) VANISHES for the twist channel:

    THE C2 MECHANISM: if v₀₂ < 0 in ANY realizable bin, the β=0 potential is
    GENUINELY unbounded in t (per-bin extractable) — and βQ floors it
    ⇒ the interior-minimum mechanism C1 lacked. At θ=0, M_T is
    spatial-block ⇒ v₀₂ ≥ 0 (pure KG twist-mass cost, m5_6_2b) — twist-fuel
    requires boost dressing: the dressed defect's twist is the fuel carrier.

H_eq[b, t; p_Θ] = V[b,t] + β·Q[b,t] + p_Θ²/(4·K[b]) ,  ω* = |p_Θ/(2K)|.

OPERATING PRINCIPLE (inherited from C1, measured there): the SURROGATE
GUIDES, the DIRECT quadrature DECIDES. Searches run in smooth parametric
families (3-Gauss b + 2-ring t — no zigzag channel); every claimed minimum is
direct-audited. Note: the t-canary is NOT machine-exact (a twisted field has
real spatial variation ⇒ the decomposition-stencil delta applies, ~1-3%);
the machine canary is the t=0 one (≡ C1's).

GATES:
  G0  table regression: the s-only channels ≡ the C1 tables (machine);
      t=0 canary machine-exact; t-canary within the stencil envelope
  G1  THE TWIST-FUEL MAP: per-voxel v₀₂(θ, r) floor — onset θ, locus r,
      and the extractability analytics (free t ⇒ any v₀₂<0 bin counts)
  G2  direct extraction dial: localized t-ring amplitude at the G1 locus —
      V_direct(a) decreasing ⇒ the fuel is real in the exact convention
  G3  THE QUESTION: β > 0 (anchored), (β × p_Θ) × multi-start over the
      12-param (b, t) family — interior minimum BELOW the t≡0 control
      (= C1's dressed ground state ≈ 2.9)? Winners: Hessian (12×12), virial,
      direct 7-point audit (center vs b-amp, t-amp, width ±5%).
  G4  the t≡0 control reproduces C1's landscape (consistency)
  G5  ω(p_Θ) family at the winner (the C4 seed / M5.8.3 doorstep)

PREREQUISITE: _m5_8_2f_tables.npz (auto-built via the imported 2f module if
absent — ~45 s). Pure numpy/scipy, ti-FREE.

RESULTS (2026-06-06, C2-B EXECUTED — hard gates PASS; NEGATIVE-CLEAN):
  G0  s-only channels ≡ C1 tables at the ψ=0 slice (3.5e-16); t=0 canary
      machine. THE PHASE-SAMPLING FIX: 2b/C1's 4-pt secular phase set is NOT
      shift-invariant — the twisted field exposed it (t-canary 7-21% error);
      with the EXACT 9-pt average the t-canary drops to 0.01-0.02% — the
      twist channel is essentially exact in the surrogate.
  G1  the twist-fuel map: a real but TINY pocket — v₀₂ = −0.0031/voxel at
      (θ=0.35, r=2.1), only in the GEM-dip window θ ∈ [0.2, 0.5]; zero or
      positive elsewhere. Five orders weaker than C1's slope-channel pockets.
  G2  the extraction dial at the locus: V_direct monotone-INCREASING with
      twist amplitude — net cost (determinate).
  G3  NO twisted state below the t≡0 control in any (β, p_Θ) cell. The one
      marginal candidate (ΔH = −0.002, max|t| = 0.46 at β=6.5, p=0.5) was
      REFUTED by the direct twist-differential dial (surrogate dip −0.006 vs
      direct rise +0.04/+0.18/+0.41) — a NEAR-CANCELLING 3-GAUSS STACK
      (+1.55 −0.82 −0.82, max θ = 0.074) riding the s-channel stencil delta:
      the parametric reincarnation of the zigzag channel. Its "V = −0.74"
      was also artifact (direct V = +0.88 > 0 — no §10c violation).
  ⇒ THE C2-B CONCLUSION: the static clock-phase twist is PURE COST — the
    m5_6_2b KG twist-mass, confirmed variationally. Both static-profile
    reductions (C1 global clock, C2-B twisted frame) now closed decisive ⇒
    the 2d/2e breathing state is IRREDUCIBLY TIME-DEPENDENT in the profile
    sector — ḃ ≠ 0 IS the bounce. C3 REDEFINED: a PERIODIC-ORBIT search in
    the reduced CC dynamics, L = K_bb[b]ḃ² + 2K_bΘḃΘ̇ + K_ΘΘ[b]Θ̇² − V − βQ
    over the parametric profile family (harmonic balance / shooting, ω free
    — the M6 eigenvalue transfer, now for a limit cycle; kbb/kbΘ inertia
    channels already tabulated in C1, t-extension needed if twist included).
  METHOD LESSONS: (i) exact 9-pt phase averaging supersedes the 2b secular
    convention wherever profiles twist; (ii) the direct DIFFERENTIAL dial
    (fixed shared part, scale the candidate channel) resolves wells 100×
    below the absolute direct-audit floor; (iii) mixed-sign parametric
    families can reconstruct the zigzag channel via near-cancelling stacks —
    single-sign bounds or differential audits mandatory.

USAGE:  python m5_8_2f2_localized_clock.py [tab|gates|search|all]
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import NonlinearConstraint, minimize
from scipy.special import erf

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8 import (  # noqa: E402
    m5_8_2f_breathing_bvp as f2f,
)
from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8.m5_8_2f_breathing_bvp import (  # noqa: E402
    K_GUARD, NBIN, PHASES, TH_CAP, TH_MESH, build_bg, clock_mats, psgn, sigma4,
)
from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8.m5_8_2a_4d_hamiltonian import (  # noqa: E402
    D4, boost_field, conj, gen4, rot4,
)
from openwave.xperiments.m5_liquid_crystal.research.sandbox_v6.m5_6_2a_biaxial_hedgehog import (  # noqa: E402
    central, commf, matmul,
)

PLANE = (2, 3)                            # the (δ,0) clock plane (2a/2f), index-0 (was (1,2))
A_BOOST = 2                               # boost axis (2f article-combo), index-0 (was 1)
# EXACT phase averaging (an upgrade over 2b/C1's 4-pt secular convention,
# REQUIRED here): the twisted field shifts each voxel's local phase by ψ(r),
# so a half-period sample set is not shift-invariant (measured: 7-21% t-canary
# error). Densities have bounded harmonics — V,K ≤ e^{i8ψ}, Q = u² ≤ e^{i16ψ}
# — so 9 uniform points over the FULL π period average ALL channels exactly.
PHASES9 = tuple(k * np.pi / 9.0 for k in range(9))
TABLE_NPZ = HERE / "_m5_8_2f2_tables.npz"
PROF_NPZ = HERE / "_m5_8_2f2_profiles.npz"
E_C1_CONTROL = 2.93                      # C1's direct dressed ground state (t≡0)
# monomial bases over (s, t): v deg ≤ 2 (6), q deg ≤ 4 (15), k s-only (3)
MON_V = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
MON_Q = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1),
         (1, 2), (0, 3), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]
Q_IDX = {ab: i for i, ab in enumerate(MON_Q)}
CH_V, CH_Q, CH_K = slice(0, 6), slice(6, 21), slice(21, 24)
NCH = 24


# ---------------------------------------------------------------------------- tabulation
def tabulate2(bg, verbose=True):
    """The (s,t)-bilinear coefficient tables over (phase, θ, bin, channel).

    Per (ψp, θ) constant config: A_i = central(M), Mb = ∂M/∂θ, M_T = ∂M/∂phase.
    F_ij = P_ij + s·Qb_ij + t·Qt_ij (exactly linear — all slope-slope
    commutators vanish), Qb/Qt = x̂_j[A_i,·] − x̂_i[A_j,·].
    Units as C1: v,k ×2h³ (≡ 2a/2b A and C), q ×h³ (Q = h³Σu², u = 2v).
    """
    h, act, ridx, xh = bg["h"], bg["active"], bg["ridx"], bg["xh"]
    O4 = bg["O4"]
    Ba = sigma4(A_BOOST)
    xh_a = [x[act] for x in xh]
    rr2 = xh_a[0] ** 2 + xh_a[1] ** 2 + xh_a[2] ** 2
    tab = np.zeros((len(PHASES9), len(TH_MESH), NBIN, NCH))
    t0 = time.time()
    for ip, psi in enumerate(PHASES9):
        S, Gd = clock_mats(psi)
        for it, th in enumerate(TH_MESH):
            Bc = np.eye(4)
            Bc[A_BOOST, A_BOOST] = Bc[0, 0] = np.cosh(th)   # time axis = index 0
            Bc[A_BOOST, 0] = Bc[0, A_BOOST] = np.sinh(th)
            W = np.einsum("...ac,cb->...ab", O4, Bc)
            M = conj(W, S)
            Y = Bc @ S @ Bc.T
            Mb = conj(O4, Ba @ Y + Y @ Ba)
            MT = conj(W, Gd)
            A = [central(M, k, h) for k in range(3)]
            Cm = [commf(A[k], Mb)[act] for k in range(3)]
            Ct = [commf(A[k], MT)[act] for k in range(3)]
            T1 = commf(MT, Mb)[act]
            Aa = [A[k][act] for k in range(3)]
            v = [np.zeros(rr2.shape) for _ in range(6)]
            for i, j in ((0, 1), (0, 2), (1, 2)):   # GRADIENT pairs (∂_x,∂_y,∂_z) — stay {0,1,2} (index-0 de-conflation; matrix-eigen pairs live in psgn)
                P = commf(Aa[i], Aa[j])
                Qb = (xh_a[j][..., None, None] * Cm[i]
                      - xh_a[i][..., None, None] * Cm[j])
                Qt = (xh_a[j][..., None, None] * Ct[i]
                      - xh_a[i][..., None, None] * Ct[j])
                v[0] += psgn(P, P)
                v[1] += 2.0 * psgn(P, Qb)
                v[2] += 2.0 * psgn(P, Qt)
                v[3] += psgn(Qb, Qb)
                v[4] += 2.0 * psgn(Qb, Qt)
                v[5] += psgn(Qt, Qt)
            u = [2.0 * vi for vi in v]
            q = [np.zeros(rr2.shape) for _ in range(15)]
            for i, (a1, b1) in enumerate(MON_V):
                for j, (a2, b2) in enumerate(MON_V):
                    q[Q_IDX[(a1 + a2, b1 + b2)]] += u[i] * u[j]
            # kd: [M_T, D_i] = −Ct_i + s·x̂_i·T1  (t drops out — K ≡ C1's K)
            k0 = sum(psgn(Ct[i], Ct[i]) for i in range(3))
            k1 = -sum(2.0 * xh_a[i] * psgn(Ct[i], T1) for i in range(3))
            k2 = rr2 * psgn(T1, T1)
            fields = v + q + [k0, k1, k2]
            facs = [2 * h**3] * 6 + [h**3] * 15 + [2 * h**3] * 3
            for ch, (f, fac) in enumerate(zip(fields, facs)):
                tab[ip, it, :, ch] = fac * np.bincount(ridx, weights=f, minlength=NBIN)
        if verbose:
            print(f"      ψ={psi:.3f} tabulated  [{time.time() - t0:.0f}s]")
    return tab


def load_tables2(bg):
    if TABLE_NPZ.exists():
        z = np.load(TABLE_NPZ)
        if (np.array_equal(z["TH_MESH"], TH_MESH) and z["tab"].shape[2] == NBIN
                and z["tab"].shape[0] == len(PHASES9)):
            print("  [--] C2 tables loaded from cache "
                  "(delete _m5_8_2f2_tables.npz to recompute)")
            return z["tab"]
        TABLE_NPZ.unlink()
    print(f"  [--] tabulating {len(PHASES9)}ψ × {len(TH_MESH)}θ × {NBIN} bins"
          " × 24 ch (exact phase average) ...")
    tab = tabulate2(bg)
    np.savez(TABLE_NPZ, tab=tab, TH_MESH=TH_MESH, PHASES=np.array(PHASES9),
             r_rep=bg["r_rep"], counts=bg["counts"])
    print(f"  [--] tables cached → {TABLE_NPZ.name}")
    return tab


# ---------------------------------------------------------------------------- surrogate
class Surrogate2:
    """H_eq[b(r), t(r)] from the (s,t)-bilinear tables (θ PCHIP × monomials)."""

    def __init__(self, tab, bg):
        self.r_rep = bg["r_rep"]
        self.cs = PchipInterpolator(TH_MESH, tab.mean(axis=0), axis=0)
        self.c = self.cs.c
        self.bins = np.arange(NBIN)

    def channels(self, th):
        thc = np.clip(th, 0.0, TH_CAP)
        seg = np.clip(np.searchsorted(TH_MESH, thc, side="right") - 1, 0, len(TH_MESH) - 2)
        dx = (thc - TH_MESH[seg])[:, None]
        c = self.c[:, seg, self.bins, :]
        return ((c[0] * dx + c[1]) * dx + c[2]) * dx + c[3]

    def vkq(self, th, s, t):
        ch = self.channels(th)
        mv = [s**a * t**b for a, b in MON_V]
        mq = [s**a * t**b for a, b in MON_Q]
        V = float(sum(ch[:, i] * mv[i] for i in range(6)).sum())
        Q = float(sum(ch[:, 6 + i] * mq[i] for i in range(15)).sum())
        s2 = s * s
        K = float((ch[:, 21] + ch[:, 22] * s + ch[:, 23] * s2).sum())
        return V, K, Q


# ---------------------------------------------------------------------------- mapper
class DualMapper:
    """x = [b: (a,w)×3 | t: (a,c,w)×2] — 3-Gauss dressing + 2-ring twist slope.

    t(r) = ψ′(r) is parameterized DIRECTLY (only ψ′ enters phase-averaged);
    ψ(r) for direct audits is the analytic erf integral, ψ(0)=0."""

    name = "3-Gauss b + 2-ring t"
    nfree = 12

    def __init__(self, r_rep):
        self.r_rep = r_rep

    def bounds(self):
        return ([(-1.55, 1.55), (0.8, 4.5)] * 3
                + [(-4.0, 4.0), (0.0, 7.0), (0.6, 4.0)] * 2)

    def theta_s_t(self, x, lam=1.0):
        r = self.r_rep * lam
        th = np.zeros_like(r)
        s = np.zeros_like(r)
        for m in range(3):
            a, w = x[2 * m], x[2 * m + 1]
            g = a * np.exp(-((r / w) ** 2))
            th += g
            s += g * (-2.0 * r / w**2) * lam
        t = np.zeros_like(r)
        for m in range(2):
            a, c, w = x[6 + 3 * m], x[7 + 3 * m], x[8 + 3 * m]
            t += a * np.exp(-(((r - c) / w) ** 2)) * lam
        return th, s, t

    def theta_fn(self, x, lam=1.0, amp=1.0):
        def f(r):
            th = np.zeros_like(r)
            for m in range(3):
                a, w = x[2 * m], x[2 * m + 1]
                th = th + a * np.exp(-((r * lam / w) ** 2))
            return np.maximum(amp * th, 0.0)
        return f

    def psi_fn(self, x, lam=1.0, amp=1.0):
        def f(r):
            rr = r * lam
            ps = np.zeros_like(rr)
            for m in range(2):
                a, c, w = x[6 + 3 * m], x[7 + 3 * m], x[8 + 3 * m]
                ps = ps + a * w * np.sqrt(np.pi) / 2.0 * (
                    erf((rr - c) / w) + erf(c / w))
            return amp * ps
        return f

    def scale(self, x, amp_b=1.0, amp_t=1.0):
        y = np.array(x, float)
        y[0:6:2] *= amp_b
        y[6:12:3] *= amp_t
        return y


def make_objective2(sur, mp, pth, beta):
    def fun(x):
        th, s, t = mp.theta_s_t(x)
        pen = 1e3 * float((np.minimum(th, 0.0) ** 2).sum()
                          + (np.maximum(th - TH_CAP, 0.0) ** 2).sum())
        V, K, Q = sur.vkq(th, s, t)
        return V + beta * Q + pth**2 / (4.0 * K) + pen
    return fun


def grad_inf(fun, x, eps=1e-4):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = eps
        g[i] = (fun(x + e) - fun(x - e)) / (2 * eps)
    return float(np.abs(g).max()), g


def hess_eigs(fun, x, eps=1e-3):
    n = len(x)
    H = np.zeros((n, n))
    f0 = fun(x)
    fp = np.zeros(n)
    fm = np.zeros(n)
    for i in range(n):
        e = np.zeros(n)
        e[i] = eps
        fp[i] = fun(x + e)
        fm[i] = fun(x - e)
        H[i, i] = (fp[i] - 2 * f0 + fm[i]) / eps**2
    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = eps
            ej[j] = eps
            H[i, j] = H[j, i] = (fun(x + ei + ej) - fp[i] - fp[j] + 2 * f0
                                 - fm[i] - fm[j] + fun(x - ei - ej)) / (2 * eps**2)
    return np.linalg.eigvalsh(H)


# ---------------------------------------------------------------------------- direct (truth)
def direct_eval_twisted(bg, theta_fn, psi_fn, phases=PHASES9):
    """Ground-truth quadrature for the TWISTED frame: W = O₄·B(b(r))·R(Θ+ψ(r)).

    The rotation is now a per-voxel FIELD; central differences of the full
    twisted field — the same exact convention C1 anchored to 2a/2b."""
    h, act, O4 = bg["h"], bg["active"], bg["O4"]
    thv = theta_fn(bg["r"])
    psv = psi_fn(bg["r"])
    Bx = boost_field(thv, A_BOOST)
    WB = matmul(O4, Bx)
    G = gen4(PLANE)
    GD = G @ D4 - D4 @ G
    out = dict(V=0.0, K=0.0, Q=0.0, umin=np.inf)
    for ph in phases:
        ang = psv + ph
        ca, sa = np.cos(ang), np.sin(ang)
        Rx = np.zeros(ang.shape + (4, 4))
        for d in range(4):
            Rx[..., d, d] = 1.0
        p_, q_ = PLANE
        Rx[..., p_, p_] = ca
        Rx[..., q_, q_] = ca
        Rx[..., p_, q_] = -sa
        Rx[..., q_, p_] = sa
        W = matmul(WB, Rx)
        M = conj(W, D4)
        MT = conj(W, GD)
        Mi = [central(M, k, h) for k in range(3)]
        v = np.zeros(M.shape[:-2])
        for i, j in ((0, 1), (0, 2), (1, 2)):   # GRADIENT pairs (∂_x,∂_y,∂_z) — stay {0,1,2} (index-0 de-conflation; matrix-eigen pairs live in psgn)
            F = commf(Mi[i], Mi[j])
            v += psgn(F, F)
        kd = sum(psgn(commf(MT, Mi[k]), commf(MT, Mi[k])) for k in range(3))
        u = 2.0 * v[act]
        out["V"] += 2 * h**3 * float(v[act].sum()) / len(phases)
        out["K"] += 2 * h**3 * float(kd[act].sum()) / len(phases)
        out["Q"] += h**3 * float((u * u).sum()) / len(phases)
        out["umin"] = min(out["umin"], float(u.min()))
    return out


def direct_audit2(bg, mp, x, pth, beta):
    """7-point DIRECT local-min check: center vs b-amp ±5%, t-amp ±5%, width ±5%."""
    vals = {}
    pts = (("center", (1.0, 1.0, 1.0)), ("b+", (1.0, 1.05, 1.0)),
           ("b-", (1.0, 0.95, 1.0)), ("t+", (1.0, 1.0, 1.05)),
           ("t-", (1.0, 1.0, 0.95)), ("w+", (1 / 1.05, 1.0, 1.0)),
           ("w-", (1.05, 1.0, 1.0)))
    for tag, (lam, ab, at) in pts:
        y = mp.scale(x, amp_b=ab, amp_t=at)
        d = direct_eval_twisted(bg, mp.theta_fn(y, lam=lam), mp.psi_fn(y, lam=lam))
        vals[tag] = d["V"] + beta * d["Q"] + pth**2 / (4.0 * d["K"])
    is_min = all(vals["center"] < vals[k] for k in vals if k != "center")
    print("      direct 7-pt audit: " + "  ".join(f"{k}={v:.3f}" for k, v in vals.items()))
    print(f"      → center lowest: {is_min}")
    return is_min, vals["center"]


# ---------------------------------------------------------------------------- gate stages
def stage_gates(bg, tab):
    """G0 + G1 + G2."""
    sur = Surrogate2(tab, bg)
    mp = DualMapper(bg["r_rep"])
    ok = {}

    # G0 — regression + canaries ---------------------------------------------------
    print("\n[G0] regression + canaries:")
    tab1 = f2f.load_tables(bg)                       # C1 tables (auto-built)
    pairs = [(0, 0), (1, 1), (3, 2),                 # v s-only ↔ C1 V coeffs
             (6, 3), (7, 4), (9, 5), (12, 6), (16, 7),   # q s-only ↔ C1 Q
             (21, 8), (22, 9), (23, 10)]             # k ↔ C1 K
    emax = 0.0
    for c2, c1 in pairs:
        # compare at the shared ψ=0 phase slice (C1 uses the 4-pt secular set,
        # C2 the exact 9-pt set — only ψ=0 is common, and it is phase-resolved)
        a, b = tab[0, :, :, c2], tab1[0, :, :, c1]
        sc = np.abs(b).max() + 1e-30
        emax = max(emax, float(np.abs(a - b).max() / sc))
    reg_ok = emax < 1e-12
    print(f"      s-only channels vs the C1 tables (ψ=0 slice): max rel dev ="
          f" {emax:.2e} (≤1e-12) → {reg_ok}")
    d0 = direct_eval_twisted(bg, lambda r: np.full_like(r, 0.5),
                             lambda r: np.zeros_like(r))
    th0 = np.full(NBIN, 0.5)
    z0 = np.zeros(NBIN)
    V0, K0, Q0 = sur.vkq(th0, z0, z0)
    e0 = max(abs(V0 - d0["V"]) / abs(d0["V"]), abs(K0 - d0["K"]) / abs(d0["K"]),
             abs(Q0 - d0["Q"]) / abs(d0["Q"]))
    print(f"      t=0 canary (θ=0.5 const): max err {100 * e0:.2e}% (machine)"
          f" → {e0 < 1e-9}")
    d1 = direct_eval_twisted(bg, lambda r: np.full_like(r, 0.5),
                             lambda r: 0.4 * r)
    V1, K1, Q1 = sur.vkq(th0, z0, np.full(NBIN, 0.4))
    eV1 = abs(V1 - d1["V"]) / abs(d1["V"])
    eK1 = abs(K1 - d1["K"]) / max(abs(d1["K"]), 1.0)
    eQ1 = abs(Q1 - d1["Q"]) / max(abs(d1["Q"]), 10.0)
    print(f"      t-canary (θ=0.5, ψ=0.4r): V {100 * eV1:.2f}%  K {100 * eK1:.2f}%"
          f"  Q {100 * eQ1:.2f}%  (stencil envelope ≤3/3/10%)")
    ok["G0"] = reg_ok and e0 < 1e-9 and eV1 < 0.03 and eK1 < 0.03 and eQ1 < 0.10

    # G1 — THE TWIST-FUEL MAP -------------------------------------------------------
    print("\n[G1] the twist-fuel map — per-voxel v₀₂ floor by θ (t is FREE ⇒ any"
          " negative bin is extractable):")
    t_avg = tab.mean(axis=0)
    v02 = t_avg[:, :, 5] / np.maximum(bg["counts"], 1)
    print("        θ      min v₀₂/voxel   r(min)")
    for th_p in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6):
        it_ = int(np.argmin(np.abs(TH_MESH - th_p)))
        jb = int(np.argmin(v02[it_]))
        print(f"        {TH_MESH[it_]:4.2f}  {v02[it_, jb]:14.4f}   "
              f"{bg['r_rep'][jb]:5.2f}")
    it_g, jb_g = np.unravel_index(np.argmin(v02), v02.shape)
    th_g, r_g = float(TH_MESH[it_g]), float(bg["r_rep"][jb_g])
    has_fuel = v02[it_g, jb_g] < 0
    print(f"      global floor: v₀₂ = {v02[it_g, jb_g]:.4f}/voxel at θ={th_g:.2f},"
          f" r={r_g:.2f}  → twist-fuel exists: {has_fuel}")
    ok["G1"] = True                                  # a map, not a pass/fail
    locus = dict(th=th_g, r=r_g, fuel=has_fuel)

    # G2 — direct extraction dial ----------------------------------------------------
    if has_fuel:
        print("\n[G2] direct extraction dial at the locus (t-ring amplitude):")
        thb = min(th_g, 1.45)
        x0 = np.array([thb, max(r_g, 1.5), 0.0, 1.5, 0.0, 1.5,
                       0.0, r_g, 1.0, 0.0, 3.0, 1.0])
        Vs = []
        for a in (0.0, 0.5, 1.0, 1.5):
            y = x0.copy()
            y[6] = a
            d = direct_eval_twisted(bg, mp.theta_fn(y), mp.psi_fn(y))
            th, s, t = mp.theta_s_t(y)
            Vsur, _, _ = sur.vkq(th, s, t)
            Vs.append(d["V"])
            print(f"        t-amp={a:.1f}: V_direct = {d['V']:12.2f}   "
                  f"(surrogate {Vsur:12.2f})   u_min = {d['umin']:.3f}")
        extract = all(np.diff(Vs) < 0)
        print(f"      → V_direct monotone-decreasing with twist amplitude: {extract}")
        ok["G2"] = extract or all(np.diff(Vs) > 0)   # determinacy gate
        locus["extract"] = extract
    else:
        print("\n[G2] skipped — no negative v₀₂ anywhere (twist is pure cost in"
              " this family: the m5_6_2b KG mass, variationally; C2-B closes"
              " NEGATIVE and option A's diagnostic is next).")
        ok["G2"] = True
        locus["extract"] = False
    return ok, sur, mp, locus


def stage_search(bg, tab, sur, mp, locus):
    """G3 + G4 — the question on the C2 manifold."""
    print("\n[G3] the search — β anchored, (β × p_Θ) × multi-start, 12 params:")
    th_g, r_g = locus["th"], locus["r"]
    seed = np.array([min(th_g, 1.45), max(r_g, 1.5), 0.0, 1.5, 0.0, 1.5,
                     1.0, r_g, 1.0, 0.0, 3.0, 1.0])
    d = direct_eval_twisted(bg, mp.theta_fn(seed), mp.psi_fn(seed))
    umin = d["umin"]
    beta_a = 3.0 / (2.0 * abs(umin)) if umin < 0 else 1.0
    print(f"      fuel scale at the twisted seed: u_min = {umin:.3f}  ⇒  "
          f"β_anchor = {beta_a:.4f}")
    starts = []
    for b0, w0 in ((0.3, 2.5), (0.6, 2.5), (1.0, 2.5)):
        for ta in (1.0, -1.0):
            starts.append((f"b{b0}/t{ta:+.0f}",
                           np.array([b0, w0, 0.0, 1.5, 0.0, 1.5,
                                     ta, r_g, 1.0, 0.0, 3.0, 1.0])))
        starts.append((f"b{b0}/t0", np.array([b0, w0, 0.0, 1.5, 0.0, 1.5,
                                              0.0, r_g, 1.0, 0.0, 3.0, 1.0])))
    winners = []
    for beta in (0.3 * beta_a, beta_a, 3.0 * beta_a):
        for pth in (0.5, 4.0, 16.0):
            print(f"    β={beta:.4f}  p_Θ={pth:.1f}:")
            fun = make_objective2(sur, mp, pth, beta)
            # t≡0 control on the SAME (β, p): the C1-landscape baseline
            xc = np.array([0.13, 3.5, 0.0, 1.5, 0.0, 1.5,
                           0.0, 2.0, 1.0, 0.0, 3.0, 1.0])
            bounds_c = mp.bounds()
            bc = [(lo, hi) if i < 6 else (0.0, 0.0) if (i - 6) % 3 == 0
                  else (lo, hi) for i, (lo, hi) in enumerate(bounds_c)]
            rc = minimize(fun, xc, method="L-BFGS-B", bounds=bc,
                          options=dict(maxiter=2000, ftol=1e-14))
            print(f"      t≡0 control: H = {rc.fun:9.3f}  (C1 direct ≈ "
                  f"{E_C1_CONTROL})")
            best_cell = None
            for label, x0 in starts:
                res = minimize(fun, x0, method="L-BFGS-B", bounds=mp.bounds(),
                               options=dict(maxiter=3000, ftol=1e-14))
                th, s, t = mp.theta_s_t(res.x)
                V, K, Q = sur.vkq(th, s, t)
                H = V + beta * Q + pth**2 / (4.0 * K)
                g, _ = grad_inf(fun, res.x)
                tmax = float(np.abs(t).max())
                stat = "stationary" if g < 1e-2 * max(1.0, abs(H)) else f"‖∇‖={g:.1e}"
                if K < K_GUARD:           # stepped across the K=0 wall — the
                    stat = "ONSET DIVE (rejected)"  # onset layer, not a state
                print(f"      {label:12s} H={H:10.3f}  K={K:8.2f}  V={V:9.3f}"
                      f"  max|t|={tmax:5.2f}  {stat}")
                if K >= K_GUARD and (best_cell is None or H < best_cell["H"]):
                    best_cell = dict(x=np.array(res.x), H=H, K=K, V=V, Q=Q,
                                     pth=pth, beta=beta, g=g, tmax=tmax,
                                     H_ctrl=rc.fun, label=label)
            if best_cell is None:
                continue
            dH = best_cell["H"] - best_cell["H_ctrl"]
            tw = best_cell["tmax"] > 0.05
            print(f"      cell best: H={best_cell['H']:.3f} vs control "
                  f"{best_cell['H_ctrl']:.3f} (ΔH = {dH:+.3f}; twisted: {tw})")
            if tw and dH < -1e-3:
                # THE DECISIVE TOOL (the artifact-killer, 2026-06-06): the
                # direct twist-differential dial — scale the t-amplitude at
                # FIXED b; the shared-b stencil error cancels in the
                # difference, so even a 0.002-deep twist well is resolvable.
                # (It refuted the first marginal candidate: surrogate dip
                # −0.006 vs direct rise +0.04/+0.18/+0.41 — a near-cancelling
                # 3-Gauss stack riding the s-channel stencil delta.)
                print("      twist-differential dial (direct, fixed b):")
                Hds = []
                for cc in (0.0, 0.5, 1.0, 1.5):
                    y = mp.scale(best_cell["x"], amp_t=cc)
                    dd = direct_eval_twisted(bg, mp.theta_fn(y), mp.psi_fn(y))
                    Hds.append(dd["V"] + beta * dd["Q"] + pth**2 / (4 * dd["K"]))
                    print(f"        c={cc:.1f}: H_direct = {Hds[-1]:9.5f}"
                          f"  (ΔH = {Hds[-1] - Hds[0]:+.5f})")
                if Hds[2] < Hds[0] - 1e-4:
                    winners.append(best_cell)
                else:
                    print("      → twist gain NOT direct-confirmed — surrogate"
                          " artifact (cancelling-stack class); rejected")
    if not winners:
        print("\n    → NO twisted state below the t≡0 control in any (β, p_Θ)"
              " cell — on this manifold the static twist is pure cost"
              " (the KG mass, variationally) and C2-B closes NEGATIVE;"
              " the dynamics' breathing state must use the TIME-DEPENDENT"
              " twist channel (the 2b-2 oscillating mode) — the C3 seeding"
              " question changes accordingly.")
        return dict(beta_anchor=beta_a), []
    print("\n    twisted winners — G4 audit per cell winner:")
    audited = []
    for w in winners:
        print(f"    [{w['label']}  β={w['beta']:.4f}  p={w['pth']:.1f}]  "
              f"H*={w['H']:.3f} (ΔH vs t≡0: {w['H'] - w['H_ctrl']:+.3f})  "
              f"K*={w['K']:.2f}  ω*={abs(w['pth'] / (2 * w['K'])):.4f}")
        fun = make_objective2(sur, mp, w["pth"], w["beta"])
        eigs = hess_eigs(fun, w["x"])
        nneg = int((eigs < -1e-6 * max(1, abs(eigs).max())).sum())
        print(f"      Hessian eigs: min={eigs.min():.3e} max={eigs.max():.3e}"
              f"  n_neg={nneg}  → {'MIN' if nneg == 0 else f'SADDLE({nneg})'}")
        is_min, H_dir = direct_audit2(bg, mp, w["x"], w["pth"], w["beta"])
        print(f"      H_direct = {H_dir:.3f} (surrogate {w['H']:.3f}, Δ="
              f"{100 * abs(H_dir - w['H']) / max(abs(H_dir), 1.0):.2f}%)")
        w.update(nneg=nneg, H_direct=H_dir, direct_min=is_min)
        audited.append(w)
    return dict(beta_anchor=beta_a), audited


def stage_family(bg, sur, mp, audited):
    """G5 — ω(p_Θ) at the winning (β)."""
    good = [w for w in audited if w.get("nneg") == 0 and w.get("direct_min")]
    if not good:
        print("\n[G5] skipped — no direct-confirmed twisted minimum")
        return
    best = min(good, key=lambda w: w["H"])
    beta = best["beta"]
    print(f"\n[G5] the ω(p_Θ) family at β={beta:.4f}:")
    print("      p_Θ      H*        K*       ω*     max|t|")
    x = best["x"].copy()
    fam = []
    for pth in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        fun = make_objective2(sur, mp, pth, beta)
        res = minimize(fun, x, method="L-BFGS-B", bounds=mp.bounds(),
                       options=dict(maxiter=3000, ftol=1e-14))
        th, s, t = mp.theta_s_t(res.x)
        V, K, Q = sur.vkq(th, s, t)
        H = V + beta * Q + pth**2 / (4 * K)
        om = abs(pth / (2 * K))
        print(f"      {pth:5.2f}  {H:8.3f}  {K:8.2f}  {om:7.4f}  "
              f"{float(np.abs(t).max()):5.2f}")
        fam.append((pth, H, K, om))
        x = np.array(res.x)
    th_w, s_w, t_w = mp.theta_s_t(best["x"])
    np.savez(PROF_NPZ, x_winner=best["x"], beta_winner=beta,
             pth_winner=best["pth"], H_winner=best["H"], K_winner=best["K"],
             theta_winner=th_w, t_winner=t_w, r_rep=bg["r_rep"],
             family=np.array(fam))
    print(f"      → winner + family saved → {PROF_NPZ.name} (C3 seed)")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("=" * 78)
    print("M5.8.2f2 — Track C C2 (option B): the phase-twisted eigenframe clock")
    print("  W = O_hh·B(b(r))·R(Θ + ψ(r)) — strictly periodic, single ω;"
          " t(r)=ψ′ FREE")
    print(f"  tables: {len(PHASES9)}ψ × {len(TH_MESH)}θ × {NBIN} bins × 24 ch"
          f" ((s,t)-bilinear exact; 9-pt EXACT phase average); K ≡ C1's K")
    print("=" * 78)
    t0 = time.time()
    bg = build_bg()
    tab = load_tables2(bg)
    if mode == "tab":
        return 0
    ok, sur, mp, locus = stage_gates(bg, tab)
    hard = all(ok.values())
    print("\n  hard gates: " + "  ".join(f"{k}={v}" for k, v in ok.items()))
    if mode == "gates":
        print("PASS" if hard else "PARTIAL — inspect the failing gate above")
        return 0 if hard else 1
    meta, audited = stage_search(bg, tab, sur, mp, locus)
    stage_family(bg, sur, mp, audited)
    print("\n" + "=" * 78)
    n_ok = sum(1 for w in audited if w.get("nneg") == 0 and w.get("direct_min"))
    print(f"C2-B verdict: hard gates {'PASS' if hard else 'PARTIAL'};  twisted"
          f" minima below the t≡0 control (Hessian-clean + direct-confirmed):"
          f" {n_ok}")
    print(f"[{time.time() - t0:.0f}s total]")
    print("=" * 78)
    return 0 if hard else 1


if __name__ == "__main__":
    raise SystemExit(main())
