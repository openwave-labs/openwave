"""
M5.8.2f — Track C (C1): the breathing/clock state as a VARIATIONAL OBJECT —
radial dressing profile b(r) beyond the Gaussian family, reduced functional,
interior-minimum hunt with the 2d saturating quartic.

THE M6 v9 TRANSFER (the cross-pollination centerpiece): forward-IVP does not
find soliton ground states — pose the state directly and let the conserved
charge/frequency be the unknown. Here: the 2b CC machinery promoted from the
2-parameter Gaussian family b·exp(−(r/r_w)²) to a multi-Gaussian profile
family (primary, 6 params — smooth by construction) + a free-node spline
space (exploratory, curvature-penalized), with the M5.8.2d quartic
V_u = u + β·u² in the potential.

ANSATZ (2a/2b conventions): M(x,t) = W D₄ Wᵀ, W = O_hh(x)·B(b(r))·R_(δ,0)(Θ),
boost axis a=2, clock plane (2,3), D₄ = diag(8, 1, δ, 0). Θ cyclic ⇒
p_Θ = 2K_ΘΘ·Θ̇ conserved. Steady states (ḃ=0) = critical points of

    H_eq[b(r)] = V[b] + β·Q[b] + p_Θ²/(4·K_ΘΘ[b]) ,  Q = h³Σ u² ,  u = 2v

With β>0 the configuration sector V+βQ is bounded below ANALYTICALLY
(u+βu² ≥ −1/(4β) per voxel — the 2d floor); the only remaining −∞ channel is
p_Θ²/(4K) at the K→0⁻ edge. THE C1 QUESTION: does an interior ghost-branch
(K<0) local minimum exist — the breathing/clock state as a variational
object — where 2b's quadratic 2-parameter family had only a saddle?

REDUCTION (what makes profile families affordable): with
∂_iM = A_i + b'(r)·x̂_i·∂M/∂θ and [∂M/∂θ, ∂M/∂θ] = 0, every density is an
EXACT polynomial in s = b'(r): V,K,K_bb,K_bΘ deg≤2, Q deg-4, with
coefficients local in (x, θ). Tabulate bin-summed coefficients over
(ψ-phase, θ-mesh, r-bin) ONCE (~1 min, cached to _m5_8_2f_tables.npz); any
profile then evaluates in ~0.1 ms and scipy minimize is interactive.

SURROGATE HONESTY MODEL (measured, F0): the constant-θ canary is EXACT to
machine (0.00% V/K/Q — tabulation + contraction + binning verified); varying
profiles carry a DECOMPOSITION-STENCIL delta vs the direct convention
(frozen-θ A_i + analytic s·x̂·Mb vs central differences of the full profile
field — both O(h²)-consistent, different constants at fixed h): ≲1% gentle
(small-b Gaussians), ~2-3% moderate, ~8% V / ~22% Q on steep shells. The
NBIN 48→128 null showed bin-sharing is NOT the driver. Consequence — the
operating principle of this script: THE SURROGATE GUIDES, THE DIRECT
QUADRATURE (cc_eval's exact method, free-profile generalized) DECIDES. Every
claimed minimum is direct-audited; free-node descents are NOT trusted raw
(the optimizer exploits stencil error via node-scale zigzags — measured:
a "descent" to E_sur=−662 was direct +3319).

GATES:
  F0   surrogate honesty: canary machine-exact; method-delta envelope within
       the measured bands (regression guard, not a claim gate)
  F1a  anchor: direct K_ΘΘ(Gauss 0.6, 2.5, ψ=0) = −48.8408 ± 0.05 (2a's C)
  F1b  regression vs the 2b cache (gentle region b ≤ 0.6 within 3%) + the
       fine-dip audit: surrogate Gaussian-restricted minimum within 3% of the
       DIRECT value at the same b*. (2b-1's spline anchor E*=2.606 is its own
       Δb=0.1 mesh artifact — the direct dip is ≈ 2.93, documented.)
  F2   the β=0 slope-fuel channel, theory-guided: the table's c₂V(θ,bin)
       channel locates v₂<0 (V unbounded in s at β=0 ⟺ the variational face
       of the 2c quadratic runaway); a SMOOTH oscillation family at that
       locus is then DIRECT-evaluated — d²V/da² < 0 confirms the channel in
       the exact convention.
  F3   THE QUESTION: β>0 (anchored 2β|u_min|=3, scan ×{0.3,1,3}) × p_Θ scan,
       multi-start ghost search (K ≤ −K_GUARD) in the multi-Gaussian family —
       interior critical point vs edge-pinned, per start (judge the table).
       Winners get the F4 audit + a DIRECT 5-point local-minimum check
       (center vs amplitude±5% and width±5% perturbations).
  F4   stationarity audit at winners: ‖∇H‖∞, Hessian eigs, profile-scaling
       virial dH/dλ|₁ (b_λ(r)=b(λr); partial virial — hedgehog bg fixed)
  F5   the ω(p_Θ) family at the winning β — the C4 seed + the M5.8.3
       calibration handle. ω* = |p_Θ/(2K*)|.

PREREQUISITE: none (pure numpy/scipy, ti-FREE). _m5_8_2b_cache.npz enables
the F1b mesh regression (skipped with a note if absent).

RESULTS (2026-06-05/06 night, C1 EXECUTED — hard gates PASS, the question
answered NEGATIVE-DECISIVE for this ansatz):
  F0  canary 0.00% (machine); envelope: gentle ≤0.6%, moderate V 2.2-2.3% /
      Q 5.5-8.7%, steep (bump) V 8.0% / Q 22.4% — the decomposition-stencil
      delta, NOT bin-sharing (NBIN 48→128 null) nor interp (post-PCHIP).
  F1a −48.8408 vs −48.8408 — machine-exact (the direct path IS 2a/2b).
  F1b gentle ≤2.84%, onset zone |ΔK| ≤ 0.33; fine-dip audit: surrogate
      E*=2.900 vs direct 2.926 at b*=0.126 — and 2b-1's spline anchor
      E*=2.606 is its own Δb=0.1 mesh artifact (direct truth ≈ 2.93,
      the GEM-dip energy refined by +0.32).
  F2  the slope-fuel channel opens ONLY at extreme dressing (per-voxel c₂V:
      −42 at θ=1.2 → −6287 at θ=1.6, r≈2; θ ≤ 1.0 clean) and is NOT
      net-extractable by realizable radial profiles (k-dial at fixed
      θ-range: V_direct +1.67M from λ4→λ2) ⇒ V[b] effectively bounded at
      β=0 in this reduced sector — the 2c FIELD runaway lives outside it.
  F3  NO interior ghost-branch minimum at any (β, p_Θ): every converged
      start EDGE-pins at the K-guard with amplitude pulled DOWN to
      b≈0.13-0.20; the amplitude-ray slice shows V+βQ rising MONOTONE
      8.4 → 1.5e10 on the ghost branch (static u positive-dominated,
      u_min = −0.034 ⇒ Q taxes amplitude; the fuel term peaks at −6.2 only
      IN the onset layer). The 2b net-cancellation verdict SURVIVES the
      quartic, variationally.
  ⇒ THE C1 CONCLUSION: the rigid-global-clock + radial-boost-dressing
    ansatz is the WRONG reduced manifold for the 2d/2e breathing state —
    confirmed variationally with the quartic on. The M6 BVP-transfer's own
    lesson applies (the ansatz/BC matters as much as the eigenvalue): C2 =
    the CORE-LOCALIZED clock ansatz (the 2b-2/m5_6_2b twist mode,
    R(Θ·φ(r)) with its own radial profile — K-fuel localizes on-core
    without the global cancellation or the V+βQ amplitude tax; the twist
    cost (∂φ)² is the new balance term = the KG-mass mechanism).

USAGE:  python m5_8_2f_breathing_bvp.py [tab|gates|quartic|family|all]
        env: F2F_NODES (default 25 — exploratory spline nodes, 2 pinned 0)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import NonlinearConstraint, minimize

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8.m5_8_2a_4d_hamiltonian import (  # noqa: E402
    D4, boost_field, conj, gen4, rot4,
)
from openwave.xperiments.m5_liquid_crystal.research.sandbox_v6.m5_6_2a_biaxial_hedgehog import (  # noqa: E402
    RC, RHOC, build_frame, central, commf, matmul,
)

PLANE = (2, 3)                            # the article's (δ,0) clock (2a/2b), index-0
A_BOOST = 2                               # boost axis (2a article-combo), index-0 (was 1)
PHASES = (0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8)
# θ-mesh refined near 0 (the GEM dip lives at θ ≲ 0.2 — a uniform Δθ=0.1 mesh
# missed it: 97% V error at b=0.13, overshooting q-channels gave NEGATIVE Q);
# interpolation is monotone PCHIP, not CubicSpline, for the same reason.
TH_MESH = np.concatenate([np.linspace(0.0, 0.2, 9)[:-1],
                          np.linspace(0.2, 0.5, 7)[:-1],
                          np.linspace(0.5, 1.6, 12)])
NBIN = 128                                # radial bins over the active region
NODES = int(os.environ.get("F2F_NODES", "25"))
R_PROF = 8.0                              # profile support [0, R_PROF]; 0 beyond
K_GUARD = 1.0                             # ghost search: K ≤ −K_GUARD
W_MIN, W_MAX = 0.8, 4.5                   # Gaussian widths the grid can represent
TH_CAP = float(TH_MESH[-1]) - 1e-9
TABLE_NPZ = HERE / "_m5_8_2f_tables.npz"
PROF_NPZ = HERE / "_m5_8_2f_profiles.npz"
ANCHOR_K = -48.8408                       # 2a's C(0.6, 2.5, ψ=0) — F1a
ANCHOR_E, ANCHOR_B = 2.606, 0.130         # 2b-1's spline value (its mesh artifact)
A0_2B = 6.139                             # 2b bare static A(0)
# table channel layout: V deg0..2 | Q deg0..4 | K deg0..2 | Kbb | KbΘ deg0..1
CH_V, CH_Q, CH_K, CH_KBB, CH_KBT = slice(0, 3), slice(3, 8), slice(8, 11), 11, slice(12, 14)
NCH = 14


def sigma4(a):
    S = np.zeros((4, 4))
    S[a, 0] = S[0, a] = 1.0           # boost generator: spatial eigen-axis a (∈{1,2,3}) ↔ time (index 0)
    return S


def psgn(F, G):
    """Signed pair-sum ⟨F,G⟩ = Σ_sp F·G − Σ_tm F·G (the ℋ matrix-block split).

    index-0: spatial-eigen MATRIX pairs (1,2),(1,3),(2,3) → +; time-mixing
    MATRIX pairs (0,1),(0,2),(0,3) (time axis 0) → −.
    """
    out = (F[..., 1, 2] * G[..., 1, 2] + F[..., 1, 3] * G[..., 1, 3]
           + F[..., 2, 3] * G[..., 2, 3])
    out -= (F[..., 0, 1] * G[..., 0, 1] + F[..., 0, 2] * G[..., 0, 2]
            + F[..., 0, 3] * G[..., 0, 3])
    return out


def build_bg():
    """2a's background + the radial unit vector + bin indexing."""
    fr = build_frame()
    O3, h, r, rho = fr["O"], fr["h"], fr["r"], fr["rho"]
    O4 = np.zeros(O3.shape[:-2] + (4, 4))
    O4[..., 1:4, 1:4] = O3                # spatial-eigen block = axes 1,2,3 (index-0)
    O4[..., 0, 0] = 1.0                   # time axis = index 0
    interior = np.zeros(r.shape, bool)
    interior[2:-2, 2:-2, 2:-2] = True
    active = (r > 2 * RC) & (rho > RHOC) & interior
    rs = np.maximum(r, 1e-12)
    xh = [fr["X"] / rs, fr["Y"] / rs, fr["Z"] / rs]
    edges = np.linspace(2 * RC, r[active].max() + 1e-9, NBIN + 1)
    ridx = np.clip(np.searchsorted(edges, r[active]) - 1, 0, NBIN - 1)
    r_rep = np.bincount(ridx, weights=r[active], minlength=NBIN)
    counts = np.bincount(ridx, minlength=NBIN)
    r_rep = np.where(counts > 0, r_rep / np.maximum(counts, 1), 0.5 * (edges[:-1] + edges[1:]))
    return dict(h=h, r=r, O4=O4, active=active, xh=xh, edges=edges, ridx=ridx,
                r_rep=r_rep, counts=counts)


def clock_mats(psi):
    R = rot4(PLANE, psi)
    G = gen4(PLANE)
    S = R @ D4 @ R.T
    Gd = G @ S - S @ G
    return S, Gd


# ---------------------------------------------------------------------------- tabulation
def tabulate(bg, verbose=True):
    """Bin-summed s-polynomial coefficient tables over (phase, θ, bin, channel).

    Per (ψ, θ): the constant-θ field gives A_i = central(M) (frozen-θ spatial
    derivative, exact for uniform θ), Mb = ∂M/∂θ (analytic), M_T = ∂M/∂Θ.
    With D_i = A_i + s·x̂_i·Mb and [Mb, Mb] = 0:
      F_ij = [A_i,A_j] + s·(x̂_j[A_i,Mb] − x̂_i[A_j,Mb])      (linear in s)
      v = Σ_{i<j}⟨F_ij,F_ij⟩_sgn  deg-2 ;  u = 2v ;  q = u²   deg-4
      kd = Σ_i⟨[M_T,D_i],[M_T,D_i]⟩_sgn                       deg-2
      kbb = Σ_i⟨[Mb,A_i],[Mb,A_i]⟩_sgn (deg-0) ; kbΘ deg-1    (C2 spectrum prep)
    Units: V,K,Kbb,KbΘ ×2h³ (≡ 2a/2b A and C); Q ×h³ (Q = h³Σu², the 2d form).
    """
    h, act, ridx, xh = bg["h"], bg["active"], bg["ridx"], bg["xh"]
    O4 = bg["O4"]
    Ba = sigma4(A_BOOST)
    xh_a = [x[act] for x in xh]
    rr2 = xh_a[0] ** 2 + xh_a[1] ** 2 + xh_a[2] ** 2
    tab = np.zeros((len(PHASES), len(TH_MESH), NBIN, NCH))
    t0 = time.time()
    for ip, psi in enumerate(PHASES):
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
            Rr = [commf(MT, A[k])[act] for k in range(3)]
            T1 = commf(MT, Mb)[act]
            Aa = [A[k][act] for k in range(3)]
            v0 = np.zeros(rr2.shape)
            v1 = np.zeros(rr2.shape)
            v2 = np.zeros(rr2.shape)
            for i, j in ((0, 1), (0, 2), (1, 2)):   # GRADIENT pairs (∂_x,∂_y,∂_z) — stay {0,1,2} (index-0 de-conflation; matrix-eigen pairs live in psgn)
                P = commf(Aa[i], Aa[j])
                Qm = (xh_a[j][..., None, None] * Cm[i]
                      - xh_a[i][..., None, None] * Cm[j])
                v0 += psgn(P, P)
                v1 += 2.0 * psgn(P, Qm)
                v2 += psgn(Qm, Qm)
            u0, u1, u2 = 2 * v0, 2 * v1, 2 * v2
            q = [u0 * u0, 2 * u0 * u1, u1 * u1 + 2 * u0 * u2, 2 * u1 * u2, u2 * u2]
            k0 = sum(psgn(Rr[i], Rr[i]) for i in range(3))
            k1 = sum(2.0 * xh_a[i] * psgn(Rr[i], T1) for i in range(3))
            k2 = rr2 * psgn(T1, T1)
            kbb = sum(psgn(Cm[i], Cm[i]) for i in range(3))
            kbt0 = -sum(psgn(Cm[i], Rr[i]) for i in range(3))
            kbt1 = -sum(xh_a[i] * psgn(Cm[i], T1) for i in range(3))
            fields = [v0, v1, v2] + q + [k0, k1, k2, kbb, kbt0, kbt1]
            facs = [2 * h**3] * 3 + [h**3] * 5 + [2 * h**3] * 6
            for ch, (f, fac) in enumerate(zip(fields, facs)):
                tab[ip, it, :, ch] = fac * np.bincount(ridx, weights=f, minlength=NBIN)
        if verbose:
            print(f"      ψ={psi:.3f} tabulated  [{time.time() - t0:.0f}s]")
    return tab


def load_tables(bg):
    if TABLE_NPZ.exists():
        z = np.load(TABLE_NPZ)
        if (np.array_equal(z["TH_MESH"], TH_MESH) and z["tab"].shape[2] == NBIN
                and np.array_equal(z["PHASES"], np.array(PHASES))):
            print("  [--] coefficient tables loaded from cache "
                  "(delete _m5_8_2f_tables.npz to recompute)")
            return z["tab"]
        TABLE_NPZ.unlink()
    print(f"  [--] tabulating {len(PHASES)}ψ × {len(TH_MESH)}θ × {NBIN} bins ...")
    tab = tabulate(bg)
    np.savez(TABLE_NPZ, tab=tab, TH_MESH=TH_MESH, PHASES=np.array(PHASES),
             edges=bg["edges"], r_rep=bg["r_rep"], counts=bg["counts"])
    print(f"  [--] tables cached → {TABLE_NPZ.name}")
    return tab


# ---------------------------------------------------------------------------- surrogate
class Surrogate:
    """Fast H_eq[b(r)] from the coefficient tables (θ PCHIP-interp × s-poly)."""

    def __init__(self, tab, bg, phase_avg=True):
        self.r_rep = bg["r_rep"]
        t = tab.mean(axis=0) if phase_avg else tab[0]
        self.cs = PchipInterpolator(TH_MESH, t, axis=0)   # monotone — no overshoot
        self.c = self.cs.c                       # (4, nθ−1, NBIN, NCH)
        self.bins = np.arange(NBIN)

    def channels(self, th):
        thc = np.clip(th, 0.0, TH_CAP)
        seg = np.clip(np.searchsorted(TH_MESH, thc, side="right") - 1, 0, len(TH_MESH) - 2)
        dx = (thc - TH_MESH[seg])[:, None]
        c = self.c[:, seg, self.bins, :]         # (4, NBIN, NCH)
        return ((c[0] * dx + c[1]) * dx + c[2]) * dx + c[3]

    def vkq(self, th, s):
        ch = self.channels(th)
        s2 = s * s
        V = float((ch[:, 0] + ch[:, 1] * s + ch[:, 2] * s2).sum())
        Q = float((ch[:, 3] + ch[:, 4] * s + ch[:, 5] * s2 + ch[:, 6] * s2 * s
                   + ch[:, 7] * s2 * s2).sum())
        K = float((ch[:, 8] + ch[:, 9] * s + ch[:, 10] * s2).sum())
        return V, K, Q


# ---------------------------------------------------------------------------- profile mappers
class GaussMapper:
    """PRIMARY family: b(r) = Σ_{m≤3} a_m·exp(−(r/w_m)²) — smooth by
    construction (no zigzag channel for the optimizer to exploit)."""

    name = "multi-Gauss"

    def __init__(self, r_rep):
        self.r_rep = r_rep
        self.nfree = 6

    def bounds(self):
        return [(-1.55, 1.55), (W_MIN, W_MAX)] * 3

    def theta_s(self, x, lam=1.0):
        r = self.r_rep * lam
        th = np.zeros_like(r)
        s = np.zeros_like(r)
        for m in range(3):
            a, w = x[2 * m], x[2 * m + 1]
            g = a * np.exp(-((r / w) ** 2))
            th += g
            s += g * (-2.0 * r / w**2) * lam
        return th, s

    def extra_penalty(self, x):
        return 0.0

    def callable(self, x, lam=1.0, amp=1.0):
        def f(r):
            th = np.zeros_like(r)
            for m in range(3):
                a, w = x[2 * m], x[2 * m + 1]
                th = th + a * np.exp(-((r * lam / w) ** 2))
            return np.maximum(amp * th, 0.0)
        return f

    def scale_amp(self, x, c):
        y = np.array(x, float)
        y[0::2] *= c
        return y


class NodeMapper:
    """EXPLORATORY space: spline through NODES nodes on [0, R_PROF] (last two
    pinned 0), curvature-penalized — node-scale zigzags sit OUTSIDE the
    surrogate's validity (measured: fake descents), so |b″| > CAP2 is taxed."""

    name = f"{NODES}-node spline"
    CAP2 = 2.0

    def __init__(self, r_rep):
        self.r_nodes = np.linspace(0.0, R_PROF, NODES)
        self.dr = self.r_nodes[1] - self.r_nodes[0]
        self.r_rep = r_rep
        self.inside = r_rep < R_PROF
        self.nfree = NODES - 2

    def bounds(self):
        return [(0.0, TH_CAP)] * self.nfree

    def _spline(self, x):
        nodes = np.concatenate([x, [0.0, 0.0]])
        return CubicSpline(self.r_nodes, nodes, bc_type="natural")

    def theta_s(self, x, lam=1.0):
        cs = self._spline(x)
        rr = self.r_rep * lam
        ins = rr < R_PROF
        th = np.where(ins, cs(np.minimum(rr, R_PROF)), 0.0)
        s = np.where(ins, cs(np.minimum(rr, R_PROF), 1) * lam, 0.0)
        return th, s

    def extra_penalty(self, x):
        nodes = np.concatenate([x, [0.0, 0.0]])
        d2 = np.abs(np.diff(nodes, 2)) / self.dr**2
        return 50.0 * float((np.maximum(d2 - self.CAP2, 0.0) ** 2).sum())

    def callable(self, x, lam=1.0, amp=1.0):
        cs = self._spline(x)

        def f(r):
            rr = r * lam
            return np.where(rr < R_PROF,
                            np.maximum(amp * cs(np.minimum(rr, R_PROF)), 0.0), 0.0)
        return f

    def scale_amp(self, x, c):
        return np.array(x, float) * c

    def gauss_nodes(self, b0, rw):
        return b0 * np.exp(-((self.r_nodes[:self.nfree] / rw) ** 2))


def make_objective(sur, mapper, pth, beta):
    def fun(x):
        th, s = mapper.theta_s(x)
        pen = 1e3 * float((np.minimum(th, 0.0) ** 2).sum()
                          + (np.maximum(th - TH_CAP, 0.0) ** 2).sum())
        pen += mapper.extra_penalty(x)
        V, K, Q = sur.vkq(th, s)
        return V + beta * Q + pth**2 / (4.0 * K) + pen
    return fun


def make_kfun(sur, mapper):
    def kf(x):
        th, s = mapper.theta_s(x)
        return sur.vkq(th, s)[1]
    return kf


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
def direct_eval(bg, theta_of_r, phases=PHASES):
    """Ground-truth full-grid quadrature for an arbitrary radial profile.

    Same algorithm as 2b's cc_eval (central differences of the FULL profile
    field) — the surrogate must agree with THIS. Returns phase-averaged
    V, K, Q, u_min (+ the ψ=0 K for the F1a anchor).
    """
    h, act, O4 = bg["h"], bg["active"], bg["O4"]
    thv = theta_of_r(bg["r"])
    Bx = boost_field(thv, A_BOOST)
    W = matmul(O4, Bx)
    out = dict(V=0.0, K=0.0, Q=0.0, umin=np.inf, K0=None)
    for psi in phases:
        S, Gd = clock_mats(psi)
        M = conj(W, S)
        MT = conj(W, Gd)
        Mi = [central(M, k, h) for k in range(3)]
        v = np.zeros(M.shape[:-2])
        for i, j in ((0, 1), (0, 2), (1, 2)):   # GRADIENT pairs (∂_x,∂_y,∂_z) — stay {0,1,2} (index-0 de-conflation; matrix-eigen pairs live in psgn)
            F = commf(Mi[i], Mi[j])
            v += psgn(F, F)
        kd = sum(psgn(commf(MT, Mi[k]), commf(MT, Mi[k])) for k in range(3))
        u = 2.0 * v[act]
        V = 2 * h**3 * float(v[act].sum())
        K = 2 * h**3 * float(kd[act].sum())
        out["V"] += V / len(phases)
        out["K"] += K / len(phases)
        out["Q"] += h**3 * float((u * u).sum()) / len(phases)
        out["umin"] = min(out["umin"], float(u.min()))
        if psi == 0.0:
            out["K0"] = K
    return out


def direct_audit(bg, mapper, x, pth, beta):
    """5-point DIRECT local-minimum check: center vs amplitude ±5%, width ±5%.

    The surrogate found it; the direct quadrature must agree the center is
    lowest along these two physical directions. Returns (is_min, H_center)."""
    vals = {}
    for tag, (lam, amp) in (("center", (1.0, 1.0)), ("amp+", (1.0, 1.05)),
                            ("amp-", (1.0, 0.95)), ("wid+", (1 / 1.05, 1.0)),
                            ("wid-", (1.05, 1.0))):
        d = direct_eval(bg, mapper.callable(x, lam=lam, amp=amp))
        vals[tag] = d["V"] + beta * d["Q"] + pth**2 / (4.0 * d["K"])
    is_min = all(vals["center"] < vals[t] for t in ("amp+", "amp-", "wid+", "wid-"))
    print("      direct 5-pt audit: " + "  ".join(f"{t}={v:.3f}" for t, v in vals.items())
          + f"  → center lowest: {is_min}")
    return is_min, vals["center"]


# ---------------------------------------------------------------------------- search drivers
def classify(sur, mapper, fun, res, pth, beta, label):
    th, s = mapper.theta_s(res.x)
    V, K, Q = sur.vkq(th, s)
    H = V + beta * Q + pth**2 / (4.0 * K)
    om = abs(pth / (2.0 * K))
    if K > -1.25 * K_GUARD:
        kind = "EDGE (K pinned at the guard — no interior min this start)"
    else:
        g, _ = grad_inf(fun, res.x)
        kind = (f"INTERIOR ‖∇H‖∞={g:.2e}" if g < 1e-2 * max(1.0, abs(H))
                else f"NON-STATIONARY ‖∇H‖∞={g:.2e} (optimizer stall)")
    print(f"      {label:24s} H*={H:9.3f}  K*={K:9.2f}  ω*={om:7.4f}  "
          f"max b={float(th.max()):5.3f}  {kind}")
    return dict(x=np.array(res.x, float), H=H, K=K, V=V, Q=Q, om=om, pth=pth,
                beta=beta, interior="INTERIOR" in kind, label=label)


def ghost_search(sur, mapper, pth, beta, starts):
    """Multi-start constrained minimize on the ghost branch (K ≤ −K_GUARD)."""
    kf = make_kfun(sur, mapper)
    fun = make_objective(sur, mapper, pth, beta)
    con = NonlinearConstraint(kf, -np.inf, -K_GUARD)
    found = []
    skipped = 0
    for label, x0 in starts:
        if kf(np.asarray(x0, float)) > -2.0 * K_GUARD:
            skipped += 1
            continue
        res = minimize(fun, x0, method="trust-constr", constraints=[con],
                       bounds=mapper.bounds(),
                       options=dict(maxiter=600, gtol=1e-8, xtol=1e-10))
        found.append(classify(sur, mapper, fun, res, pth, beta, label))
    if skipped:
        print(f"      ({skipped} start(s) skipped — not ghost-side, "
              f"K(x₀) > −{2 * K_GUARD})")
    return found


def virial_audit(sur, mapper, x, pth, beta):
    """Profile-scaling stationarity: H(λ) for b_λ(r)=b(λr) (partial virial —
    the hedgehog background held fixed)."""
    def H_of(lam):
        th, s = mapper.theta_s(x, lam=lam)
        V, K, Q = sur.vkq(th, s)
        return V + beta * Q + pth**2 / (4.0 * K), V, beta * Q, pth**2 / (4.0 * K)

    eps = 0.02
    Hp = H_of(1 + eps)
    Hm = H_of(1 - eps)
    H1 = H_of(1.0)
    dH = (Hp[0] - Hm[0]) / (2 * eps)
    parts = [(Hp[i] - Hm[i]) / (2 * eps) for i in (1, 2, 3)]
    print(f"      virial dH/dλ|₁ = {dH:+.4f}  (V′={parts[0]:+.3f}, "
          f"βQ′={parts[1]:+.3f}, (p²/4K)′={parts[2]:+.3f};  H={H1[0]:.3f})")
    return dH, H1[0]


# ---------------------------------------------------------------------------- gate stages
def stage_gates(bg, tab):
    """F0 + F1a + F1b + F2."""
    sur = Surrogate(tab, bg)
    gm = GaussMapper(bg["r_rep"])
    ok = {}

    # F0 — surrogate honesty -----------------------------------------------------
    print("\n[F0] surrogate honesty — table vs direct full-grid quadrature:")
    tests = [("const(0.5)", lambda r: np.full_like(r, 0.5), 1e-9, 1e-9),
             ("Gauss(0.13, 3.5)", lambda r: 0.13 * np.exp(-((r / 3.5) ** 2)), 0.01, 0.01),
             ("Gauss(0.6, 2.5)", lambda r: 0.6 * np.exp(-((r / 2.5) ** 2)), 0.03, 0.10),
             ("Gauss(1.0, 3.0)", lambda r: 1.0 * np.exp(-((r / 3.0) ** 2)), 0.03, 0.10),
             ("bump(1.0, 2.0)", lambda r: 1.0 * np.e * ((r / 2.0) ** 2)
              * np.exp(-((r / 2.0) ** 2)), 0.10, 0.25)]
    ok["F0"] = True
    for name, f, tolVK, tolQ in tests:
        d = direct_eval(bg, f)
        eps_fd = 1e-5
        th = f(bg["r_rep"])
        s = (f(bg["r_rep"] + eps_fd) - f(bg["r_rep"] - eps_fd)) / (2 * eps_fd)
        V, K, Q = sur.vkq(th, s)
        eV = abs(V - d["V"]) / max(abs(d["V"]), 1.0)
        eK = abs(K - d["K"]) / max(abs(d["K"]), 1.0)
        eQ = abs(Q - d["Q"]) / max(abs(d["Q"]), 10.0)
        band = eV < tolVK and eK < tolVK and eQ < tolQ
        ok["F0"] = ok["F0"] and band
        print(f"      {name:18s} V {100 * eV:6.2f}%  K {100 * eK:6.2f}%  "
              f"Q {100 * eQ:6.2f}%   (bands {100 * tolVK:.0f}/{100 * tolQ:.0f}%)"
              f"  {'ok' if band else 'OUT OF BAND'}")
    print(f"    → canary machine-exact + method-delta within the measured "
          f"envelope: {ok['F0']}")

    # F1a — the 2a/2b anchor (direct, ψ=0) ----------------------------------------
    d = direct_eval(bg, lambda r: 0.6 * np.exp(-((r / 2.5) ** 2)), phases=(0.0,))
    ok["F1a"] = abs(d["K0"] - ANCHOR_K) < 0.05
    print(f"\n[F1a] direct K_ΘΘ(0.6, 2.5, ψ=0) = {d['K0']:+.4f}  (2a: {ANCHOR_K})"
          f"  → {ok['F1a']}")

    # F1b — regression vs the 2b cache + the fine-dip audit ------------------------
    print("\n[F1b] regression vs the 2b landscape:")
    cache = HERE / "_m5_8_2b_cache.npz"
    mesh_ok = True
    if cache.exists():
        z = np.load(cache)
        Bm, RWm = z["B_MESH"], z["RW_MESH"]
        K2b = z["KTT_sp"] - z["KTT_tm"]
        V2b = z["V_sp"] - z["V_tm"]
        eg = es = onset_abs = 0.0
        for i, rw in enumerate(RWm):
            for j, b in enumerate(Bm):
                th = b * np.exp(-((bg["r_rep"] / rw) ** 2))
                s = th * (-2 * bg["r_rep"] / rw**2)
                V, K, _ = sur.vkq(th, s)
                # dip-scale V (~2.5-6) gated with a floor — the GEM-dip accuracy
                # is independently DIRECT-audited below (the fine-dip audit)
                eV = abs(V - V2b[i, j]) / max(abs(V2b[i, j]), 5.0)
                # K near the ghost-onset zero-crossing: relative error explodes
                # by construction (2b's own ONSET_GUARD excluded it) — gate the
                # onset zone on ABSOLUTE error (scale: |C_ref| ≈ 49)
                if abs(K2b[i, j]) >= 10.0:
                    eK = abs(K - K2b[i, j]) / abs(K2b[i, j])
                else:
                    onset_abs = max(onset_abs, abs(K - K2b[i, j]))
                    eK = 0.0
                if 0.858 * b / rw <= 0.21:        # peak |b'| — the stencil-delta driver
                    eg = max(eg, eV, eK)
                else:
                    es = max(es, eV, eK)
        mesh_ok = eg < 0.03 and onset_abs < 3.0
        print(f"      vs cache: gentle region (peak |b'| ≤ 0.21) max err "
              f"{100 * eg:.2f}% (≤3%); onset zone (|K|<10) max |ΔK| = "
              f"{onset_abs:.2f} (≤3)  → {mesh_ok}")
        print(f"      steep region {100 * es:.2f}% (envelope, documented — "
              "claims ride on direct audits)")
    else:
        print("      (2b cache absent — mesh regression skipped)")
    bf = np.linspace(0.02, 1.4, 600)
    Hb = []
    for b in bf:
        th = b * np.exp(-((bg["r_rep"] / 3.5) ** 2))
        s = th * (-2 * bg["r_rep"] / 3.5**2)
        V, K, _ = sur.vkq(th, s)
        Hb.append(V + 0.5**2 / (4 * K) if K > K_GUARD else np.nan)
    Hb = np.array(Hb)
    jmin = np.nanargmin(Hb)
    bstar, Estar = bf[jmin], Hb[jmin]
    dd = direct_eval(bg, lambda r, b=bstar: b * np.exp(-((r / 3.5) ** 2)))
    E_dir = dd["V"] + 0.5**2 / (4 * dd["K"])
    sub_ok = abs(Estar - E_dir) / max(abs(E_dir), 1.0) < 0.03
    ok["F1b"] = mesh_ok and sub_ok
    print(f"      Gaussian-restricted (r_w=3.5, p_Θ=0.5): b*={bstar:.3f} "
          f"E*={Estar:.3f}  | direct at b*: E={E_dir:.3f}  → {sub_ok}")
    print(f"      (2b-1's spline anchor b*={ANCHOR_B}, E*={ANCHOR_E} — its Δb=0.1"
          f" mesh interpolates across the dip; direct refines it by "
          f"{E_dir - ANCHOR_E:+.3f})")
    th0 = np.zeros(NBIN)
    V0, K0c, _ = sur.vkq(th0, np.zeros(NBIN))
    print(f"      bare checks: A(0)={V0:.3f} (2b: {A0_2B})   K(0)={K0c:.2f} (>0)")

    # F2 — the β=0 slope-fuel channel, theory-guided -------------------------------
    # v₂ < 0 somewhere ⟺ V drops ~ −s² at β=0 (the variational face of the 2c
    # quadratic runaway). The table SAYS where it lives (measured: ONLY at
    # extreme dressing θ ≳ 1.4 — moderate θ ≤ 1.2 is s²-stable, which is why
    # the 2b family never saw it). Direct confirmation = a WAVELENGTH dial at
    # fixed amplitude: θ-range fixed, s² ∝ k² — the cosh-wall confound (which
    # sank the amplitude-dial version) is held constant. (Free-node descents
    # are not used — the optimizer exploits stencil error there.)
    print("\n[F2] the β=0 slope-fuel channel (table-guided, direct-confirmed):")
    t_avg = tab.mean(axis=0)                                  # (nθ, NBIN, NCH)
    c2V = t_avg[:, :, 2]
    dens = c2V / np.maximum(bg["counts"], 1)
    print("      per-voxel c₂V floor by θ (the channel's θ-onset):")
    for th_p in (1.0, 1.2, 1.3, 1.4, 1.5, 1.6):
        it_ = int(np.argmin(np.abs(TH_MESH - th_p)))
        jb = int(np.argmin(dens[it_]))
        print(f"        θ={TH_MESH[it_]:4.2f}:  min c₂V/voxel = {dens[it_, jb]:10.3f}"
              f"  at r = {bg['r_rep'][jb]:.2f}")
    has_channel = dens.min() < -1.0
    if has_channel:
        th0, r0, a_osc = 1.42, 2.1, 0.15          # the measured locus; 1.42+0.15<1.6

        def osc(lam_w, a=a_osc, rs=r0, t0=th0):
            return (lambda r: np.maximum(
                t0 * np.exp(-(((r - rs) / 2.0) ** 2))
                + a * np.sin(2 * np.pi * (r - rs) / lam_w)
                * np.exp(-(((r - rs) / 1.5) ** 2)), 0.0))
        Vs = []
        for lam_w in (0.0, 4.0, 8.0 / 3.0, 2.0):  # 0 ⇒ no oscillation (reference)
            f = osc(lam_w) if lam_w > 0 else osc(4.0, a=0.0)
            dv = direct_eval(bg, f)
            Vs.append(dv["V"])
            tag = f"λ={lam_w:.2f} (s²×{(4.0 / lam_w) ** 2:.1f})" if lam_w else "a=0  (reference)"
            print(f"        {tag:22s} V_direct = {dv['V']:12.2f}")
        extractable = Vs[3] < Vs[2] < Vs[1]
        stable = Vs[3] > Vs[2] > Vs[1]
        ok["F2"] = extractable or stable      # gate = DETERMINACY of the dial;
        #                                       the measured branch is the finding
        if extractable:
            print(f"      → V decreasing with k (ΔV(λ4→λ2) = {Vs[3] - Vs[1]:+.2f})"
                  " ⇒ s²-fuel NET-EXTRACTABLE at extreme dressing — the"
                  " variational face of the 2c runaway; β>0 floors it.")
        elif stable:
            print(f"      → V INCREASING with k (ΔV(λ4→λ2) = {Vs[3] - Vs[1]:+.2f})"
                  " ⇒ the c₂V<0 pocket is NOT net-extractable by realizable"
                  " radial profiles (its neighbors' positive s² dominates):")
            print("        V[b] is effectively bounded below at β=0 in the"
                  " radial-profile + global-clock sector — the 2c FIELD runaway"
                  " lives outside this reduction (non-radial/incoherent channels;"
                  " consistent with 2b-2's field-level finding).")
        else:
            print("      → non-monotone dial — INCONCLUSIVE, inspect.")
    else:
        print("      → c₂V ≥ −1/voxel everywhere: no meaningful slope channel at"
              " β=0 in this family — the 2c runaway is not visible in this"
              " reduction (document).")
        ok["F2"] = True                       # a clean finding either way
    np.savez(PROF_NPZ, r_rep=bg["r_rep"], bstar_gauss=bstar, Estar_gauss=Estar,
             E_dir_gauss=E_dir)
    return ok, sur, gm


def stage_quartic(bg, tab, sur, gm):
    """F3 + F4 — the question: interior ghost minimum with the quartic on."""
    print("\n[F3] the quartic question — β anchored, p_Θ × β scan, ghost search")
    print(f"      primary family: {gm.name} (smooth — surrogate-valid); every"
          " winner is direct-audited")
    d = direct_eval(bg, lambda r: 1.0 * np.exp(-((r / 2.5) ** 2)))
    umin = d["umin"]
    beta_a = 3.0 / (2.0 * abs(umin)) if umin < 0 else np.nan
    print(f"      fuel scale at ghost seed G(1.0, 2.5): u_min = {umin:.3f}  ⇒  "
          f"β_anchor = 3/(2|u_min|) = {beta_a:.4f}")
    starts = [(f"G({a},{w})", [a, w, 0.0, 1.5, 0.0, 1.5])
              for a in (1.0, 1.2, 1.4) for w in (2.0, 2.5, 3.0)]
    starts += [("G−G(1.3,3,−.6,1.2)", [1.3, 3.0, -0.6, 1.2, 0.0, 1.5])]
    winners = []
    for beta in (0.3 * beta_a, beta_a, 3.0 * beta_a):
        for pth in (1.0, 4.0, 16.0):
            print(f"    β={beta:.4f}  p_Θ={pth:.1f}:")
            found = ghost_search(sur, gm, pth, beta, starts)
            ints = [f for f in found if f["interior"]]
            if ints:
                winners.append(min(ints, key=lambda f: f["H"]))
    if not winners:
        print("\n    → NO interior ghost-branch minimum at any (β, p_Θ) scanned in"
              " the multi-Gauss family. The amplitude-ray slice (the evidence —"
              " β=β_anchor, p_Θ=4, a·Gauss(2.5)):")
        print("        a       V        β·Q       p²/4K       K       H_eq")
        for a in (0.13, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.4):
            th, s = gm.theta_s(np.array([a, 2.5, 0.0, 1.5, 0.0, 1.5]))
            V, K, Q = sur.vkq(th, s)
            pk = 16.0 / (4 * K)
            print(f"       {a:.2f} {V:9.1f} {beta_a * Q:11.3g} {pk:8.3f} {K:9.2f}"
                  f" {V + beta_a * Q + pk:11.3g}")
        print("      V+βQ rises MONOTONICALLY along the ghost branch (the static"
              " u is positive-dominated — Q taxes the amplitude the ghost needs)"
              " while p²/4K only diverges IN the onset layer ⇒ saddle-type only:")
        print("      the 2b net-cancellation verdict survives the quartic,"
              " variationally. The 2d/2e breathing state lives OUTSIDE this"
              " reduced manifold ⇒ C2 pivot: the CORE-LOCALIZED clock ansatz"
              " (the 2b-2/m5_6_2b twist mode, R(Θ·φ(r)) with its own profile).")
        return dict(beta_anchor=beta_a), []
    print("\n    interior minima — F4 audit + DIRECT 5-point check per winner:")
    audited = []
    for w in winners:
        print(f"    [{w['label']}  β={w['beta']:.4f}  p={w['pth']:.1f}]  "
              f"H*={w['H']:.3f}  K*={w['K']:.2f}  ω*={w['om']:.4f}")
        fun = make_objective(sur, gm, w["pth"], w["beta"])
        dH, _ = virial_audit(sur, gm, w["x"], w["pth"], w["beta"])
        eigs = hess_eigs(fun, w["x"])
        nneg = int((eigs < -1e-6 * max(1, abs(eigs).max())).sum())
        print(f"      Hessian eigs: min={eigs.min():.3e} max={eigs.max():.3e} "
              f" n_neg={nneg}  → {'MIN' if nneg == 0 else f'SADDLE({nneg})'}")
        is_min_dir, H_dir = direct_audit(bg, gm, w["x"], w["pth"], w["beta"])
        dd = direct_eval(bg, gm.callable(w["x"]))
        print(f"      direct: V={dd['V']:.3f} K={dd['K']:.2f} Q={dd['Q']:.2f} "
              f"u_min={dd['umin']:.3f}  → H_direct={H_dir:.3f} (surrogate "
              f"{w['H']:.3f}, Δ={100 * abs(H_dir - w['H']) / max(abs(H_dir), 1.0):.2f}%)")
        w.update(nneg=nneg, dH=dH, H_direct=H_dir, K_direct=dd["K"],
                 direct_min=is_min_dir)
        audited.append(w)
    return dict(beta_anchor=beta_a), audited


def stage_family(bg, sur, gm, audited, beta_a):
    """F5 — ω(p_Θ) at the winning β (the C4 seed)."""
    good = [w for w in audited if w.get("nneg") == 0 and w.get("direct_min")]
    if not good:
        print("\n[F5] skipped — no direct-confirmed interior minimum to continue")
        return
    best = min(good, key=lambda w: w["H"])
    beta = best["beta"]
    print(f"\n[F5] the ω(p_Θ) family at β={beta:.4f} (warm-started from the winner):")
    print("      p_Θ      H*        K*       ω*      interior")
    x = best["x"].copy()
    fam = []
    kf = make_kfun(sur, gm)
    for pth in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        fun = make_objective(sur, gm, pth, beta)
        con = NonlinearConstraint(kf, -np.inf, -K_GUARD)
        res = minimize(fun, x, method="trust-constr", constraints=[con],
                       bounds=gm.bounds(),
                       options=dict(maxiter=600, gtol=1e-8, xtol=1e-10))
        th, s = gm.theta_s(res.x)
        V, K, Q = sur.vkq(th, s)
        H = V + beta * Q + pth**2 / (4 * K)
        om = abs(pth / (2 * K))
        interior = K < -1.25 * K_GUARD
        print(f"      {pth:5.2f}  {H:8.3f}  {K:8.2f}  {om:7.4f}   "
              f"{'yes' if interior else 'EDGE'}")
        fam.append((pth, H, K, om, float(interior)))
        if interior:
            x = np.array(res.x, float)
    z = dict(np.load(PROF_NPZ)) if PROF_NPZ.exists() else {}
    th_w, s_w = gm.theta_s(best["x"])
    z.update(x_winner=best["x"], beta_winner=beta, pth_winner=best["pth"],
             H_winner=best["H"], K_winner=best["K"], om_winner=best["om"],
             theta_winner=th_w, family=np.array(fam))
    np.savez(PROF_NPZ, **z)
    print(f"      → winner profile + family saved → {PROF_NPZ.name} "
          "(C2 warm-start / C3 seed)")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("=" * 78)
    print("M5.8.2f — Track C (C1): the breathing/clock state as a variational object")
    print("  primary: 3-Gauss profile family;  exploratory: "
          f"{NODES}-node spline (curvature-penalized)")
    print(f"  tables: {len(PHASES)}ψ × {len(TH_MESH)}θ × {NBIN} bins; quartic "
          f"Q = h³Σu², u = 2v (the 2d form); ghost guard K ≤ −{K_GUARD}")
    print("=" * 78)
    t0 = time.time()
    bg = build_bg()
    print(f"  active voxels: {100 * bg['active'].mean():.1f}%   "
          f"r ∈ [{bg['edges'][0]:.2f}, {bg['edges'][-1]:.2f}]   [{time.time() - t0:.0f}s]")
    tab = load_tables(bg)
    if mode == "tab":
        return 0
    ok, sur, gm = stage_gates(bg, tab)
    hard = all(ok.values())
    print("\n  hard gates: " + "  ".join(f"{k}={v}" for k, v in ok.items()))
    if mode == "gates":
        print("PASS" if hard else "PARTIAL — inspect the failing gate above")
        return 0 if hard else 1
    meta, audited = stage_quartic(bg, tab, sur, gm)
    if mode != "quartic":
        stage_family(bg, sur, gm, audited, meta.get("beta_anchor"))
    print("\n" + "=" * 78)
    n_int = sum(1 for w in audited if w.get("nneg") == 0 and w.get("direct_min"))
    print(f"C1 verdict: hard gates {'PASS' if hard else 'PARTIAL'};  interior "
          f"ghost-branch minima (Hessian-clean + direct-confirmed): {n_int}")
    print(f"[{time.time() - t0:.0f}s total]")
    print("=" * 78)
    return 0 if hard else 1


if __name__ == "__main__":
    raise SystemExit(main())
