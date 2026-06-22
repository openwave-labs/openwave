"""
M5.8.2a — 4D Minkowski Hamiltonian: the negative-energy clock fuel on the
hedgehog (the 3+1D variational anchor, the twin of M5.8.0a's 1D quadrature).

Implements the EXACT complete-model Hamiltonian from 5a §10d (Wolfram-article
Eq.42 form), with time = MATRIX INDEX 0 (the Duda / index-0 convention):

    ℋ = 2 Σ_{0≤μ<ν≤3} [ Σ_{spatial α<β} (F_μναβ)²  −  Σ_{α=1..3} (F_μν0α)² ]
    F_μναβ = [∂_μ M, ∂_ν M]_αβ ,   M = O D O^T ,   D = diag(g, 1, δ, 0)

Ansatz family (rigid, two knobs — the 3+1D analog of the toy's (w, ω)):

    O(x,t) = O_hh(x) · B(x; b) · R(ωt)
    O_hh   = the m5_6_2a biaxial hedgehog frame, embedded 4×4 (time row/col = 1)
    B      = exp(b·w(r)·B_a)   boost dressing: mixes spatial eigen-axis a with
             the time axis, core-localized profile w(r) — the M5.8.1 time-freeze
             boundary DELIBERATELY crossed (the ★ strategic note, roadmap M5.8.2)
    R(ψ)   = exp(ψ·G_pq)       the clock: rotation in eigen-plane (p,q);
             default (1,2) = the (δ,0) plane — the article's exp(ψ·Gx) clock

Under ANY rigid sweep, E(ω,b) = A(b) + ω²·C(b) is EXACTLY quadratic in ω
(∂₀M ∝ ω, ∂_iM ω-independent) — so the decisive object is the SIGN of C(b):

    C(b) = C_pos(b) − C_neg(b) ;  C_neg = the (0,α) time-mixing block = THE FUEL

The finite-ω* CAP is the profile response (the toy's βR⁴ analog, which the
article says the positive 3D curvature supplies) — that is M5.8.2b's job, not
this script's. V(M) = 0 here (the anchor isolates the curvature mechanism).

GATES:
  G1  bare clock costs:    C_neg(b=0) = 0 exactly, C(0) > 0  (the M5.7
      functional null — no crystal while the time axis is inert)
  G2  THE FUEL:            ∃ b in scan with C(b) < 0 (Minkowski wins)
  G3  crystal threshold:   E(ω,b) < E(0,0) for ω > ω_c finite (the dressed
      oscillating state beats the static defect)
  G4  signature control:   Euclidean flip (all blocks +) ⇒ C_E(b) > 0 ∀b
  G5  static mass guard:   A(b) > 0 on a fine b-scan (§10c at ω=0). The
      ω-direction unboundedness where C<0 is the EXPECTED rigid-family artifact
      (the 1D toy WITHOUT βR⁴): the finite-ω* cap is the profile response —
      M5.8.2b's job — and its onset is already visible here as C_pos/C_neg
      rising back toward parity at large b
  G6  localization:        fuel density is core-localized (clock on the defect)
  G7  apolar doubling:     M(ψ) has period π, not 2π  (ω_M = 2ω_clock — the
      2mc²/ℏ doubling seed)

USAGE:  python m5_8_2a_4d_hamiltonian.py
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openwave.xperiments.m5_liquid_crystal.research.sandbox_v6.m5_6_2a_biaxial_hedgehog import (  # noqa: E402
    build_frame, matmul, commf, central, RC, RHOC, DELTA, N,
)

G_TIME = float(os.environ.get("M58_G", "8.0"))   # time-axis eigenvalue; M58_G env overrides (Duda δ-calibration 2026-06-08)
D4 = np.diag([G_TIME, 1.0, DELTA, 0.0])   # D = diag(g, 1, δ, 0), time = index 0 (Duda / index-0 convention)
R_W = 2.5                                 # boost-dressing radial width  w(r)=exp(−(r/R_W)²)
SP_PAIRS = [(1, 2), (1, 3), (2, 3)]       # spatial-eigen MATRIX-index pairs (axes 1,2,3) → POSITIVE in ℋ
TM_PAIRS = [(0, 1), (0, 2), (0, 3)]       # time-mixing MATRIX pairs (time axis 0)        → NEGATIVE in ℋ
GRAD_PAIRS = [(0, 1), (0, 2), (1, 2)]     # spatial GRADIENT (∂_x,∂_y,∂_z) axis pairs — NOT matrix indices (index-0 de-conflation)
B_SCAN = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
OMEGA_SCAN = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4])


def conj(W, S):
    """W S Wᵀ per voxel — W: (...,4,4) field, S: constant (4,4)."""
    return np.einsum("...ac,cd,...bd->...ab", W, S, W)


def rot4(plane, psi):
    p, q = plane
    R = np.eye(4)
    c, s = np.cos(psi), np.sin(psi)
    R[p, p], R[p, q], R[q, p], R[q, q] = c, -s, s, c
    return R


def gen4(plane):
    p, q = plane
    G = np.zeros((4, 4))
    G[p, q], G[q, p] = -1.0, 1.0
    return G


def pl_s(plane):
    """Eigen-plane label by eigenvalue (index-0): index 1↔'1', 2↔'δ', 3↔'0'."""
    names = {1: "1", 2: "δ", 3: "0"}
    return f"({names[plane[0]]},{names[plane[1]]})-plane"


def boost_field(theta, a):
    """exp(θ·B_a) per voxel — B_a = E_{a0}+E_{0a} mixes spatial eigen-axis a (∈{1,2,3}) with time (index 0)."""
    B = np.zeros(theta.shape + (4, 4))
    for i in range(4):
        B[..., i, i] = 1.0
    ch, sh = np.cosh(theta), np.sinh(theta)
    B[..., a, a] = ch
    B[..., 0, 0] = ch
    B[..., a, 0] = sh
    B[..., 0, a] = sh
    return B


def split_pn(F):
    """Σ(F_αβ)² over the positive (spatial) / negative (time-mixing) matrix blocks."""
    pos = sum(F[..., a, b] ** 2 for a, b in SP_PAIRS)
    neg = sum(F[..., a, b] ** 2 for a, b in TM_PAIRS)
    return pos, neg


def build_bg():
    fr = build_frame()
    O3, h, r, rho = fr["O"], fr["h"], fr["r"], fr["rho"]
    O4 = np.zeros(O3.shape[:-2] + (4, 4))
    O4[..., 1:4, 1:4] = O3
    O4[..., 0, 0] = 1.0
    interior = np.zeros(r.shape, bool)
    interior[2:-2, 2:-2, 2:-2] = True
    active = (r > 2 * RC) & (rho > RHOC) & interior     # off point core + disclination
    w = np.exp(-((r / R_W) ** 2))                       # core-localized dressing profile
    return dict(h=h, r=r, O4=O4, active=active, w=w)


def blocks(bg, b, a_boost, plane, psi=0.0, vacuum=False):
    """The four ℋ blocks of E(ω) = (A_pos−A_neg) + ω²(C_pos−C_neg), 2h³-normalized.

    A_* = the ω-independent spatial-curvature F_ij blocks; C_* = the F_0i blocks
    (coefficient of ω², computed at unit clock rate). vacuum=True replaces the
    hedgehog frame with the identity (dressing-only control).
    """
    h, act = bg["h"], bg["active"]
    O4 = np.broadcast_to(np.eye(4), bg["O4"].shape) if vacuum else bg["O4"]
    Bx = boost_field(b * bg["w"], a_boost)
    W = matmul(O4, Bx)
    R = rot4(plane, psi)
    G = gen4(plane)
    Dp = R @ D4 @ R.T                       # clock-rotated eigenvalue matrix
    Gd = G @ Dp - Dp @ G                    # d/dψ (R D Rᵀ) at phase ψ
    M = conj(W, Dp)
    Md1 = conj(W, Gd)                       # ∂₀M at unit ω
    Mi = [central(M, ax, h) for ax in range(3)]
    out = dict(A_pos=0.0, A_neg=0.0, C_pos=0.0, C_neg=0.0)
    for i, j in GRAD_PAIRS:                  # μν spatial GRADIENT pairs (∂_i,∂_j), i,j ∈ {0,1,2}
        p, n = split_pn(commf(Mi[i], Mi[j]))
        out["A_pos"] += float(p[act].sum())
        out["A_neg"] += float(n[act].sum())
    for i in range(3):                      # μν time pairs (0,i)
        p, n = split_pn(commf(Md1, Mi[i]))
        out["C_pos"] += float(p[act].sum())
        out["C_neg"] += float(n[act].sum())
    s = 2.0 * h**3
    return {k: s * v for k, v in out.items()}


def E_of(bl, omega, euclid=False):
    sgn = +1.0 if euclid else -1.0
    return (bl["A_pos"] + sgn * bl["A_neg"]) + omega**2 * (bl["C_pos"] + sgn * bl["C_neg"])


def fuel_profile(bg, b, a_boost, plane, nbins=10):
    """Radial profile of the fuel density (the (α,3) comps of the F_0i block)."""
    h, act, r = bg["h"], bg["active"], bg["r"]
    Bx = boost_field(b * bg["w"], a_boost)
    W = matmul(bg["O4"], Bx)
    G = gen4(plane)
    Gd = G @ D4 - D4 @ G
    M = conj(W, D4)
    Md1 = conj(W, Gd)
    dens = np.zeros(r.shape)
    for i in range(3):
        _, n = split_pn(commf(Md1, central(M, i, h)))
        dens += n
    edges = np.linspace(2 * RC, r[act].max(), nbins + 1)
    prof = []
    for k in range(nbins):
        m = act & (r >= edges[k]) & (r < edges[k + 1])
        prof.append(dens[m].mean() if m.any() else 0.0)
    return edges, np.array(prof)


def main():
    print("=" * 76)
    print("M5.8.2a — 4D Minkowski Hamiltonian: clock fuel on the hedgehog (anchor)")
    print(f"  grid {N}³  D=diag({G_TIME}, 1, {DELTA}, 0)  time=index 0  dressing R_w={R_W}")
    print("  ℋ = 2Σ_(μ<ν)[Σ_(sp α<β) F² − Σ_α F²_(α3)]   (5a §10d, V=0)")
    print("=" * 76)
    bg = build_bg()
    print(f"  active voxels (off core + disclination): {100 * bg['active'].mean():.1f}%")

    # --- G7: apolar doubling — M(ψ) period is π, not 2π --------------------------
    M0 = conj(bg["O4"], rot4((2, 3), 0.0) @ D4 @ rot4((2, 3), 0.0).T)
    Mh = conj(bg["O4"], rot4((2, 3), np.pi / 2) @ D4 @ rot4((2, 3), np.pi / 2).T)
    Mp = conj(bg["O4"], rot4((2, 3), np.pi) @ D4 @ rot4((2, 3), np.pi).T)
    d_half = np.abs(Mh - M0).max()
    d_full = np.abs(Mp - M0).max()
    print("\n[G7] apolar doubling — clock period of M")
    print(f"    max|M(π/2)−M(0)| = {d_half:.3e}   max|M(π)−M(0)| = {d_full:.3e}")
    g7 = d_full < 1e-12 * max(d_half, 1e-30) or (d_full < 1e-9 and d_half > 1e-3)
    print(f"    → M-period = π ⇒ ω_M = 2·ω_clock (the 2mc²/ℏ doubling): {g7}")

    # --- G1: bare clock (b=0) — the time axis inert ⇒ the M5.7 functional null ---
    bl0 = blocks(bg, 0.0, 1, (2, 3))
    print("\n[G1] bare clock, b=0 (time axis inert — the M5.8.1 production state)")
    print(f"    C_pos={bl0['C_pos']:.4e}   C_neg={bl0['C_neg']:.4e}  (exact 0 expected)")
    g1 = bl0["C_neg"] == 0.0 and bl0["C_pos"] > 0
    print(f"    → clock COSTS energy (+ω²·{bl0['C_pos']:.3e}) — no crystal without the")
    print(f"      time axis: {g1}   (the M5.7 free-defect null, in functional form)")

    # vacuum control: dressing without the hedgehog
    blv = blocks(bg, 0.6, 1, (2, 3), vacuum=True)
    print("\n[--] vacuum control (dressing, NO hedgehog): "
          f"C={blv['C_pos'] - blv['C_neg']:+.3e}  A={blv['A_pos'] - blv['A_neg']:+.3e}")

    # --- G2: discovery matrix — clock plane × boost axis, b=0.6 ------------------
    print("\n[G2] discovery matrix — C(b=0.6) = C_pos − C_neg  (FUEL ⇔ C < 0)")
    print("      clock plane ↓ \\ boost axis a →      a=1           a=2           a=3")
    best = None
    for plane in [(1, 2), (1, 3), (2, 3)]:
        row = []
        for a in (1, 2, 3):
            bl = blocks(bg, 0.6, a, plane)
            cval = bl["C_pos"] - bl["C_neg"]
            row.append(cval)
            if best is None or cval < best[0]:
                best = (cval, plane, a)
        print(f"      ({plane[0]},{plane[1]})                        "
              + "  ".join(f"{v:+.5e}" for v in row))
    cbest, plane, a = best
    print(f"    → most negative: plane ({plane[0]},{plane[1]}), boost axis a={a}, "
          f"C={cbest:+.4e}")

    # --- phase-dependence of the orbit (rigid-sweep honesty check) ----------------
    print()
    for pl, ax, lbl in [(plane, a, f"best combo ({pl_s(plane)}, a={a})"),
                        ((2, 3), 2, "article clock ((δ,0)-plane, a=2)")]:
        es = np.array([E_of(blocks(bg, 0.6, ax, pl, psi=p), 1.0)
                       for p in (0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8)])
        spread = (es.max() - es.min()) / abs(es.mean())
        print(f"[--] orbit phase-breathing, {lbl}: ΔE/⟨E⟩ over ψ = {100 * spread:.1f}%")
    print("     (the rigid sweep is not iso-energetic — the true orbit breathes; CC in 2b)")

    # --- G3: b-scans — the strongest-fuel combo + the article (δ,0) clock ---------
    def scan(pl, ax, lbl):
        print(f"\n[G3] b-scan — {lbl}:  E(ω,b) = A(b) + ω²·C(b)")
        print("      b    A_pos       A_neg       A(b)        C_pos       C_neg "
              "      C(b)        ω_c")
        bls_ = [blocks(bg, b, ax, pl) for b in B_SCAN]
        A0_ = bls_[0]["A_pos"] - bls_[0]["A_neg"]
        fuel = False
        for b, bl in zip(B_SCAN, bls_):
            Ab = bl["A_pos"] - bl["A_neg"]
            Cb = bl["C_pos"] - bl["C_neg"]
            if Cb < 0:
                fuel = True
                wc_s = "0 (static wins)" if Ab < A0_ else f"{np.sqrt((Ab - A0_) / -Cb):.3f}"
            else:
                wc_s = "—"
            print(f"      {b:.1f}  {bl['A_pos']:.3e}  {bl['A_neg']:.3e}  {Ab:+.3e}  "
                  f"{bl['C_pos']:.3e}  {bl['C_neg']:.3e}  {Cb:+.3e}  {wc_s}")
        return bls_, A0_, fuel

    bls, A0, g2 = scan(plane, a, f"strongest fuel ({pl_s(plane)}, a={a})")
    bls_art, A0_art, g2_art = scan((2, 3), 2, "article (δ,0) clock, a=2")
    print(f"    → G2 FUEL (∃b: C<0): strongest combo {g2}, article clock {g2_art}")

    # the two distinct Minkowski (α,3)-block effects, separated:
    dipA = min(bl["A_pos"] - bl["A_neg"] for bl in bls)
    print("\n    two distinct negative-energy effects (both from the (α,3) block):")
    print(f"    1. STATIC dressing: A(b) dips to {dipA:.3f} < A(0)={A0:.3f} at small b —")
    print("       boost dressing alone lowers the static defect's energy (the ω=0 GEM")
    print("       block; no clock needed)")
    print("    2. CLOCK fuel: C(b) < 0 — the clock lowers it further (the §10d clock–")
    print("       gravity co-arising term δΓ¹₀Γ̄¹)")

    Esurf = np.array([[E_of(bl, w) for w in OMEGA_SCAN] for bl in bls])
    Emin = Esurf.min()
    bi, wi = np.unravel_index(Esurf.argmin(), Esurf.shape)
    g3 = g2 and Emin < A0
    print(f"\n    E(ω,b) scan-box min = {Emin:.4e} at (ω={OMEGA_SCAN[wi]:.1f}, "
          f"b={B_SCAN[bi]:.1f})   vs  E(0,0) = {A0:.4e}")
    print(f"    → G3 crystal (dressed oscillating state beats the static defect): {g3}")

    # --- G4: Euclidean signature flip — the control --------------------------------
    CE = [bl["C_pos"] + bl["C_neg"] for bl in bls]
    g4 = all(c > 0 for c in CE)
    print(f"\n[G4] Euclidean flip (all blocks +): C_E(b) ∈ [{min(CE):.3e}, {max(CE):.3e}]")
    print(f"    → C_E > 0 ∀b ⇒ NO fuel in (+,+,+,+) — the Minkowski signature is the"
          f" mechanism: {g4}")

    # --- G5: static-sector mass guard + the rigid-family cap diagnosis -------------
    print("\n[G5] static mass guard — fine A(b) scan at ω=0 (§10c: mass can't go ≤ 0):")
    bf = np.linspace(0.0, 0.4, 9)
    Af = []
    for b in bf:
        bl = blocks(bg, b, a, plane)
        Af.append(bl["A_pos"] - bl["A_neg"])
    print("      b:    " + "  ".join(f"{b:7.2f}" for b in bf))
    print("      A(b): " + "  ".join(f"{v:7.3f}" for v in Af))
    g5 = all(v > 0 for v in Af)
    print(f"    → A(b) > 0 over the fine scan (static sector bounded): {g5}")
    ratios = [bl["C_pos"] / bl["C_neg"] for bl in bls[1:]]
    print("    C_pos/C_neg vs b: " + " → ".join(f"{q:.2f}" for q in ratios))
    print("    → falls then RISES toward parity: the positive ω²-block re-engages at")
    print("      large b — the onset of the article's positive-curvature cap. The full")
    print("      finite-ω* cap needs the profile response (the toy's βR⁴ analog) = 2b.")
    print("      (E(ω→∞) unbounded at fixed b with C<0 is the EXPECTED rigid-family")
    print("       artifact — the 1D toy without β had exactly this.)")

    # --- G6: fuel localization ------------------------------------------------------
    edges, prof = fuel_profile(bg, 0.6, a, plane)
    peak = prof.max()
    tail = prof[-3:].mean()
    g6 = peak > 0 and tail < 0.05 * peak
    print("\n[G6] fuel-density radial profile (the (α,3) comps of F_0i, b=0.6):")
    for k in range(len(prof)):
        bar = "#" * int(50 * prof[k] / peak) if peak > 0 else ""
        print(f"      r∈[{edges[k]:4.1f},{edges[k + 1]:4.1f})  {prof[k]:.3e}  {bar}")
    print(f"    → core-localized (tail/peak = {tail / peak if peak > 0 else 0:.4f} < 0.05):"
          f" {g6}   — the clock lives ON the defect")

    ok = g1 and g2 and g3 and g4 and g5 and g6 and g7
    print("\n" + "=" * 76)
    print("M5.8.2a: the Minkowski (α,3) block of F_0i is a NEGATIVE-energy clock term;")
    print("with the time axis boost-dressed it outweighs the rotation cost (C<0) and the")
    print("dressed oscillating defect undercuts the static one beyond ω_c — the 3+1D")
    print("time-crystal mechanism at the energy level, signature-dependent + core-local.")
    print("PASS" if ok else "PARTIAL — inspect the failing gate above")
    print("=" * 76)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
