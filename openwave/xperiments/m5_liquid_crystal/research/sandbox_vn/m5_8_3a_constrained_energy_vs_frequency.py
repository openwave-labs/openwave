"""
M5.8.3a — CONSTRAINED E(ω): seed-level energy with the clock phase IMPOSED at
frequency ω. The direct analog of Duda's toymodel E(ω) (arXiv:2501.04036, eq 3):
he imposes ψ = ωt on a static kink profile and computes E = ∫H dx, finding a
MINIMUM at the de Broglie ω because his curvature coupling gives a FREQUENCY-coupled
NEGATIVE term (the (1−αω²) factor, the −αR² coupling).

The M5.8 question this answers: does imposing a clock velocity Mdot = ω·Mth give the
kinetic energy a NEGATIVE (boost / GEM, Minkowski-signature) contribution that grows
with ω, reproducing Duda's minimum — or is M5.8's energy monotonic in the imposed
frequency (its negative term living only in the static boost dressing, 2u/2z)?

Method (seed-level, numpy, fast): at fixed dressing b, for each imposed ω set
Mdot = ω·Mth and compute H = T(Mdot) + u(M0) + β·u², splitting T into the rotation
(EM, +) and boost (GEM, −) sectors. Sweep ω. VALIDATION: H(ω=0) must equal the
static H_of_b (no clock velocity).
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("M58_DELTA", "0.3")
os.environ.setdefault("M58_G", "8.0")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8.m5_8_2cb_taichi_constrained import (  # noqa: E402
    build_grid, seed_M, make_masks, B_STAR,
)
from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8.m5_8_2c1_full_evolution import (  # noqa: E402
    central, tw,
)
from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8.m5_8_2a_4d_hamiltonian import (  # noqa: E402
    SP_PAIRS, TM_PAIRS,
)
from openwave.xperiments.m5_liquid_crystal.research.sandbox_vn.m5_8_2h_omega_attractor import np_commf  # noqa: E402
from openwave.xperiments.m5_liquid_crystal.research.sandbox_vn.m5_8_2q_delta_scaling import H_of_b  # noqa: E402

DELTA, G = float(os.environ["M58_DELTA"]), float(os.environ["M58_G"])
BETA = 1.558
B_FIXED = B_STAR                              # the production clock dressing
OUT_NPZ = HERE / "data" / "_m5_8_3a_constrained_energy_vs_frequency.npz"
OUT_PNG = HERE / "plots" / "_m5_8_3a_constrained_energy_vs_frequency.png"
OMEGAS = np.linspace(0.0, 2.0, 21)


def kinetic(Md, M0, h):
    """T density + its EM(rotation,+) / GEM(boost,−) split, mirroring u_sectors for F0=[Md,∂iM]."""
    Mi = [central(M0, ax, h) for ax in range(3)]   # ∂_x,∂_y,∂_z GRADIENT axes — NOT matrix indices (index-0 de-conflation: stay {0,1,2})
    T = 0.0; T_em = 0.0; T_gem = 0.0
    for i in range(3):                             # μν spatial GRADIENT loop over Mi — stays {0,1,2}
        F0 = np_commf(Md, Mi[i])
        T = T + 2.0 * np.einsum("...ab,...ab->...", F0, tw(F0))
        T_em = T_em + 4.0 * sum(F0[..., a, b] ** 2 for a, b in SP_PAIRS)   # SP/TM = MATRIX-eigen pairs {1,2,3}/{0,..} imported from 2a (index-0), indexing F0
        T_gem = T_gem - 4.0 * sum(F0[..., a, b] ** 2 for a, b in TM_PAIRS)
    return T, T_em, T_gem


def u_density(M0, h):
    u = 0.0
    Mi = [central(M0, ax, h) for ax in range(3)]   # ∂_x,∂_y,∂_z GRADIENT axes — NOT matrix indices (stay {0,1,2})
    for i in range(3):                             # μν spatial GRADIENT pairs over Mi — stay {0,1,2}
        for j in range(i + 1, 3):
            F = np_commf(Mi[i], Mi[j])
            u = u + 2.0 * np.einsum("...ab,...ab->...", F, tw(F))
    return u


def main():
    print("=" * 78)
    print(f"M5.8.3a — CONSTRAINED E(ω) (Duda-faithful): b={B_FIXED}, δ={DELTA}, g={G}")
    print("=" * 78)
    g = build_grid()
    act, _ = make_masks(g)
    h = g["h"]
    a = act > 0.5
    M0, Mth = seed_M(g, B_FIXED)
    u = u_density(M0, h)
    H_pot = float((u + BETA * u * u)[a].sum()) * h ** 3   # ω=0 energy

    Hs, Tems, Tgems = [], [], []
    for w in OMEGAS:
        Md = w * Mth
        T, T_em, T_gem = kinetic(Md, M0, h)
        H = float((T + u + BETA * u * u)[a].sum()) * h ** 3
        Hs.append(H)
        Tems.append(float(T_em[a].sum()) * h ** 3)
        Tgems.append(float(T_gem[a].sum()) * h ** 3)
    Hs = np.array(Hs); Tems = np.array(Tems); Tgems = np.array(Tgems)

    # validation: H(ω=0) vs the static H_of_b
    H_static_ref, _ = H_of_b(DELTA, G, B_FIXED)
    print(f"\n  VALIDATION: H(ω=0)={Hs[0]:.4f} vs static H_of_b={H_static_ref:.4f} "
          f"(Δ={abs(Hs[0]-H_static_ref):.2e})  {'PASS' if abs(Hs[0]-H_static_ref)<1e-6 else 'CHECK'}")

    i_min = int(np.argmin(Hs))
    print("\n  | ω | H total | T_EM (rot,+) | T_GEM (boost,−) |")
    for w, H, te, tg in zip(OMEGAS, Hs, Tems, Tgems):
        mark = "  <- MIN" if w == OMEGAS[i_min] else ""
        if abs(w * 10 % 2) < 1e-9 or mark:
            print(f"  | {w:4.2f} | {H:8.3f} | {te:9.3f} | {tg:+10.3f} |{mark}")
    print(f"\n  RESULT: H(ω) {'HAS A MINIMUM at ω=%.2f' % OMEGAS[i_min] if 0 < i_min < len(OMEGAS)-1 else 'is MONOTONIC'}")
    print(f"  T_GEM (boost kinetic) at ω=2: {Tgems[-1]:+.3f} (negative ⇒ Minkowski boost-sector lowers kinetic)")

    np.savez(OUT_NPZ, omega=OMEGAS, H=Hs, T_em=Tems, T_gem=Tgems, b=B_FIXED, delta=DELTA, g=G)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(OMEGAS, Hs, "o-", color="#1f77b4", label="H total")
    ax.plot(OMEGAS, Tems + Hs[0], "--", color="#2ca02c", lw=1, label="T_EM (rotation, +) + H₀")
    ax.plot(OMEGAS, Tgems + Hs[0], "--", color="#d62728", lw=1, label="T_GEM (boost, −) + H₀")
    if 0 < i_min < len(OMEGAS) - 1:
        ax.plot([OMEGAS[i_min]], [Hs[i_min]], "*", color="crimson", ms=16,
                label=f"min at ω={OMEGAS[i_min]:.2f}")
    ax.set_xlabel("imposed clock frequency ω  (Mdot = ω·Mth)")
    ax.set_ylabel("seed energy H")
    ax.set_title("M5.8.3a constrained E(ω): energy with the clock phase imposed at ω")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    print(f"  plot -> {OUT_PNG.name}\n  data -> {OUT_NPZ.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
