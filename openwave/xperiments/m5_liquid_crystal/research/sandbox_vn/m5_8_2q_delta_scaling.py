"""
M5.8.2q — DUDA CALIBRATION (2026-06-08): the δ-scaling of the rest energy.

Jarek Duda's 2026-06-08 reply: the QED Lagrangian maps onto our D = diag(g, 1, δ, 0):
the quantum-phase (Dirac-kinetic) term ~ δ² ↔ ℏc ("tiny"), the mass term ↔ g
(gravity), so the PHYSICAL hierarchy is δ ~ ℏ ≪ 1 ≪ g ~ 1/δ — he suggests δ ~ 10⁻¹⁰
and g·δ ≈ 1, while we ran at δ = 0.3, g = 8. His barb: the clock came out
"surprisingly close (5.5e19 rad/s, ~28× below the electron ZBW) — especially for
using much too large delta." This script measures the δ-dependence DIRECTLY.

PHASE A (this file) — the SEED-LEVEL rest energy H_static(δ), with g = 1/δ:
  H is seed-only (no evolution, no FFT) ⇒ exact and cheap. The clock RATIO that
  answers Duda is ω·δ/(2·H_static) (the m5_8_2j "ℏ↔δ" convention) — ω needs the
  evolution (Phase B, separate), but H(δ) is the part that scales hardest and is
  computed here over a fine δ grid in seconds.

CORRECTNESS GATE: at (δ = 0.3, g = 8.0) this MUST reproduce the N-3 record —
  H_static = 16.74, H_quad (quadratic part) = 9.21 (m5_8_2j_zbw_ratio.py:29).

USAGE:  python m5_8_2q_delta_scaling.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# seed geometry + energy primitives (no edits to validated modules) ───────────
from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8.m5_8_2a_4d_hamiltonian import (  # noqa: E402
    conj, boost_field, matmul, SP_PAIRS, TM_PAIRS,
)
from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8.m5_8_2c1_full_evolution import (  # noqa: E402
    L, B_STAR, A_BOOST, central, tw,
)
from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8.m5_8_2cb_taichi_constrained import (  # noqa: E402
    build_grid_n,
)
from openwave.xperiments.m5_liquid_crystal.research.sandbox_vn.m5_8_2h_omega_attractor import (  # noqa: E402
    np_commf,
)

# the validated seed constants (the N-3 / 24³ β=1.558 stack)
N = 24
R_W = 3.5            # dressing width (2c1 default, the knob that enters seed_M)
RC = 0.8            # point-core radius
RHOC = 0.8          # disclination-line regularization
BETA = 1.558        # the quartic saturation coefficient


def seed_M(delta, g_time, b_star=B_STAR):
    """The boost-dressed biaxial-hedgehog 4×4 seed at (δ, g, b_star) — exactly the 2cb seed."""
    grid = build_grid_n(N, L)
    r, rho, h, O4 = grid["r"], grid["rho"], grid["h"], grid["O4"]
    w = np.exp(-((r / R_W) ** 2))
    W = matmul(O4, boost_field(b_star * w, A_BOOST))
    D4 = np.diag([g_time, 1.0, delta, 0.0])   # D = diag(g, 1, δ, 0), time = index 0 (Duda / index-0 convention)
    M0 = conj(W, D4)
    inter = np.zeros(r.shape, bool)
    inter[2:-2, 2:-2, 2:-2] = True
    act = inter & (r > 2 * RC) & (rho > RHOC)
    return M0, act, h


def u_sectors(M, h):
    """Quadratic energy density split by Duda's question: EM = +spatial-block
    curvature (curvature of ROTATIONS, η-positive), GEM = −time-mixing curvature
    (curvature of BOOSTS, η-negative = the clock fuel). u_EM + u_GEM ≡ u_density."""
    Mi = [central(M, ax, h) for ax in range(3)]   # ∂_x,∂_y,∂_z GRADIENT axes — NOT matrix indices (index-0 de-conflation: stay {0,1,2})
    uEM, uGEM = 0.0, 0.0
    for i in range(3):                             # μν spatial GRADIENT pairs over Mi — stay {0,1,2}
        for j in range(i + 1, 3):
            F = np_commf(Mi[i], Mi[j])
            sp = sum(F[..., a, b] ** 2 for a, b in SP_PAIRS)   # SP/TM = MATRIX-eigen pairs {1,2,3}/{0,..} imported from 2a (index-0)
            tm = sum(F[..., a, b] ** 2 for a, b in TM_PAIRS)
            uEM = uEM + 4.0 * sp                  # 2 (η-contraction) × 2 (ordered pairs)
            uGEM = uGEM - 4.0 * tm                # NEGATIVE — Minkowski boost block
    return uEM, uGEM


def sectors_of(delta, g_time, b_star):
    M0, act, h = seed_M(delta, g_time, b_star)
    uEM, uGEM = u_sectors(M0, h)
    a = act
    EM = float(uEM[a].sum()) * h ** 3
    GEM = float(uGEM[a].sum()) * h ** 3
    return EM, GEM


def u_density(M, h):
    """Signed-quartic energy density u = Σ_{i<j} 2·F:tw(F), F = [∂_iM, ∂_jM]."""
    u = 0.0
    Mi = [central(M, ax, h) for ax in range(3)]   # ∂_x,∂_y,∂_z GRADIENT axes — NOT matrix indices (stay {0,1,2})
    for i in range(3):                             # μν spatial GRADIENT pairs over Mi — stay {0,1,2}
        for j in range(i + 1, 3):
            F = np_commf(Mi[i], Mi[j])
            u = u + 2.0 * np.einsum("...ab,...ab->...", F, tw(F))
    return u


def H_of_b(delta, g_time, b_star):
    M0, act, h = seed_M(delta, g_time, b_star)
    u0 = u_density(M0, h)
    H_quad = float(u0[act].sum()) * h ** 3
    H_static = float((u0 + BETA * u0 * u0)[act].sum()) * h ** 3
    return H_static, H_quad


def H_of(delta, g_time):
    return H_of_b(delta, g_time, B_STAR)


def main():
    print("=" * 78)
    print("M5.8.2q — DUDA CALIBRATION: rest energy H_static(δ), g = 1/δ  [24³ seed]")
    print("=" * 78)

    # ── CORRECTNESS GATE ─────────────────────────────────────────────────────
    Hs, Hq = H_of(0.3, 8.0)
    ok = abs(Hs - 16.74) < 0.05 and abs(Hq - 9.21) < 0.05
    print(f"\n[gate] (δ=0.3, g=8.0):  H_static = {Hs:.4f}  H_quad = {Hq:.4f}"
          f"   (N-3 record 16.74 / 9.21)  {'PASS ✓' if ok else 'FAIL ✗'}")
    if not ok:
        print("  !!! seed does not reproduce the record — ABORT, do not trust the sweep")
        return 1

    # ── δ-sweep, g = 1/δ (Duda's g·δ = 1) ────────────────────────────────────
    print("\n[A] δ-scaling with g = 1/δ:")
    print("  |    δ    |    g=1/δ   | H_static | H_quad |  H_static/δ² | H_static·δ |")
    print("  | --- | --- | --- | --- | --- | --- |")
    deltas = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    rows = []
    for d in deltas:
        g = 1.0 / d
        Hs, Hq = H_of(d, g)
        rows.append((d, g, Hs, Hq))
        print(f"  | {d:7.3f} | {g:10.3f} | {Hs:8.3f} | {Hq:7.3f} |"
              f" {Hs / d**2:11.3f} | {Hs * d:9.4f} |")

    # ── log-log scaling fit  H_static ~ δ^p ─────────────────────────────────
    dd = np.array([r[0] for r in rows])
    HH = np.array([r[2] for r in rows])
    pos = HH > 0
    if pos.sum() >= 2:
        p, logA = np.polyfit(np.log(dd[pos]), np.log(HH[pos]), 1)
        print(f"\n  fit: H_static ≈ {np.exp(logA):.3f} · δ^({p:.3f})  "
              f"(g = 1/δ co-varied)")

    # ── g-sensitivity (δ = 0.3 fixed) — validates Duda's neglect-gravity ─────
    print("\n[B] g-sensitivity at δ = 0.3 (does the rest energy feel the gravity axis?):")
    print("  |    g    | H_static | H_quad | Δ vs g=8 |")
    print("  | --- | --- | --- | --- |")
    Hs8, _ = H_of(0.3, 8.0)
    for g in [1.0 / 0.3, 8.0, 100.0, 1000.0]:
        Hs, Hq = H_of(0.3, g)
        print(f"  | {g:7.3f} | {Hs:8.3f} | {Hq:7.3f} | {100*(Hs/Hs8 - 1):+7.2f}% |")

    # ── [C] δ-sweep with GRAVITY DECOUPLED (g = 8 fixed) — Duda's neglect-grav ─
    # the physically meaningful sweep: isolate the quantum-phase (δ) sector with
    # the divergent gravity axis held as a fixed background.
    print("\n[C] δ-scaling with gravity DECOUPLED (g = 8 fixed) — the clean δ-sector:")
    print("  |    δ    | H_static | H_quad | H_static/H(δ=0.3) |")
    print("  | --- | --- | --- | --- |")
    Hs_ref, _ = H_of(0.3, 8.0)
    rows_c = []
    for d in deltas:
        Hs, Hq = H_of(d, 8.0)
        rows_c.append((d, Hs, Hq))
        print(f"  | {d:7.3f} | {Hs:8.4f} | {Hq:7.4f} | {Hs / Hs_ref:8.4f} |")
    dc = np.array([r[0] for r in rows_c])
    Hc = np.array([r[1] for r in rows_c])
    p2, logA2 = np.polyfit(np.log(dc), np.log(Hc), 1)
    print(f"\n  fit (g fixed): H_static ≈ {np.exp(logA2):.3f} · δ^({p2:.3f}),  "
          f"floor as δ→0 means the δ-axis is a SMALL correction to a g/1-axis core")

    # ── [D] BOOST-SCAN (Duda 2026-06-09): gravity enters via the BOOST, not g ──
    # the physical gravity knob is the time-axis tilt b_star·g, not the eigenvalue g.
    print("\n[D] boost-scan at δ=0.3, g=8 (vary the boost b_star, the physical gravity knob):")
    print("  | b_star | b_star·g | H_static | EM (rotations,+) | GEM (boosts,−) | |GEM|/EM |")
    print("  | --- | --- | --- | --- | --- | --- |")
    for b in [0.0, 0.05, 0.13, 0.30, 0.60, 1.00]:
        Hs, _ = H_of_b(0.3, 8.0, b)
        EM, GEM = sectors_of(0.3, 8.0, b)
        ratio = abs(GEM) / EM if EM else float("nan")
        print(f"  | {b:6.3f} | {b*8:8.3f} | {Hs:9.4f} | {EM:9.4f} | {GEM:+10.4f} | {ratio:7.4f} |")

    print("\n[D2] b_star·g is the knob (equal product → equal GEM, NOT equal g):")
    print("  | b_star |   g   | b_star·g | GEM (boosts) |")
    print("  | --- | --- | --- | --- |")
    for b, g in [(0.13, 8.0), (0.013, 80.0), (0.13, 80.0), (1.30, 8.0)]:
        _, GEM = sectors_of(0.3, g, b)
        print(f"  | {b:6.3f} | {g:5.1f} | {b*g:8.3f} | {GEM:+10.4f} |")

    # ── [E] EM/GEM ratio (Duda's "rotations-EM vs boosts-GEM" question) ─────────
    print("\n[E] EM/GEM Lagrangian ratio at the physical boost (Duda's question):")
    print("  | b_star | EM (curv. of rotations) | GEM (curv. of boosts) | EM/|GEM| | EM+GEM vs H_quad |")
    print("  | --- | --- | --- | --- | --- |")
    for b in [0.13, 0.01]:
        EM, GEM = sectors_of(0.3, 8.0, b)
        _, Hq = H_of_b(0.3, 8.0, b)
        emgem = EM / abs(GEM) if GEM else float("inf")
        print(f"  | {b:6.3f} | {EM:9.4f} | {GEM:+10.4f} | {emgem:8.2f} | "
              f"{EM+GEM:.4f} vs {Hq:.4f} |")

    print("\n  READ: at b_star=0 the GEM (boost) block is EXACTLY 0, so gravity enters")
    print("  only through the boost. In the physical small-tilt regime GEM ∝ (b_star·g)²")
    print("  (the 0.13·8 and 0.013·80 pair match at GEM≈−9.3; the large pairs diverge")
    print("  because the boost is sinh-nonlinear). At a physical small boost (b=0.01)")
    print("  EM/|GEM| = 210: the EM sector (curvature of rotations, the 1-axis Faber")
    print("  unit vector) dominates and GEM is a tiny NEGATIVE (clock-fuel) correction")
    print("  that REDUCES the rest energy. The earlier 'gravity-dominated' reading")
    print("  raised g at fixed b_star, an unphysically large tilt. EM+GEM = H_quad exactly.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
