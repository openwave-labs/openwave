"""
M5.8.2l — N-5: COMPLETING THE INVARIANT MATRIX (capped).

The 2d/2e invariant matrix measured: signed `u + β·u²` (the winner — bounded
+ clock grows), Skyrme `β_E·u_E²` (bounded but clock damped 10×), Euclid
twin (kill-control). Two candidates remained un-measured; this script caps
them per the roadmap N-5 row:

(i) A4 — THE M6-STYLE AMPLITUDE QUARTIC (mode `a4`, GPU):
    V_A = β_A·(⟨M,M⟩_s − c₀)², the matrix analog of M6's λ(φ†φ)² — a LOCAL
    amplitude potential (no gradients; force = −4β_A(s₂−c₀)·M_tw added to
    the curvature force; the flux/kinetic sector is the 2d stack verbatim,
    β_u = 0 so A4 is measured ALONE against the quadratic runaway).
    c₀ = the seed's act-median ⟨M,M⟩_s (the vacuum amplitude); β_A scanned
    at 3 anchors like the 2d β ladder. Question: does an amplitude quartic
    saturate the runaway, and does the clock survive?

(ii) THE COVARIANT `𝒮 + β𝒮²` (mode `classify`, CPU sympy — runs now):
    𝒮 = T − U (the covariant scalar). 𝒮² = T² − 2TU + U²:
      · the STATIC/Derrick sector (T = 0) gives H = U − βU² — the
        stabilizing branch is β < 0, which reproduces the already-validated
        2d quartic U + |β|U² exactly (nothing new to test statically);
      · the NEW content is the kinetic mixing (T², TU): P = ∂L/∂Ṁ becomes
        CUBIC in Ṁ (sympy-verified below on the exact 1-mode reduction),
        i.e. the Legendre inversion is non-quadratic AND the inertia
        operator the constrained integrator spectrally projects becomes
        state-dependent + rank-augmented (A_eff = (1+2β(T−U))·A
        + 2β·(∂T/∂Ṁ)⊗(∂T/∂Ṁ)).
    PER THE CAP: the dynamic test is DEFERRED — "requires non-quadratic
    momentum inversion" is the reported finding, machine-checked here.

RESULTS (2026-06-07):
  classify — CONFIRMED by sympy on the exact 1-mode reduction
    (L = a·q̇² − u + β(a·q̇² − u)²): p = 2a·q̇·(1 + 2β(a·q̇² − u)) — degree 3
    in q̇ ⇒ NON-quadratic Legendre. Static sector: H(q̇=0) = u − β·u² — the
    stabilizing branch is β < 0, which matches the validated u + |β|u²
    EXACTLY (same static family; 2d already scanned it). ⇒ the covariant
    candidate adds NOTHING testable statically; its only new content is the
    kinetic mixing (T², TU), which needs a cubic per-mode inversion + a
    state-dependent rank-augmented inertia projection — DEFERRED, the
    honest finding per the cap.
  a4 — THE AMPLITUDE CHANNEL IS GEOMETRICALLY PINNED (degenerate, no
    lever arm): the seed's signed amplitude ⟨M,M⟩_s is CONSTANT across the
    field (c₀ = 65.09, dev p95 = 0.0000 machine-zero — orthogonal
    conjugation leaves it invariant; the boost dressing's O(b²w²)
    correction sits below f32 noise in the act region) ⇒ V_A = β_A(s₂−c₀)²
    has nothing to grab: the anchoring rule exploded to β_A ~ 3e8–3e9,
    amplifying f32 noise (the H prints are the noise term, not dynamics;
    |s| stays ~1e-3 — clock unaffected). VERDICT: the M6-style amplitude
    quartic CANNOT regulate this state class — the runaway fuel lives in
    the GRADIENT/curvature sector while the amplitude is constraint-pinned
    — which is precisely WHY the signed-u quartic is the correct invariant
    class. The invariant matrix is COMPLETE: signed-u (winner) / Skyrme
    u_E (saturates, damps clock 10×) / Euclid (kill-control) / A4
    (degenerate-pinned, measured) / covariant 𝒮+β𝒮² (deferred:
    non-quadratic Legendre; static sector ≡ the validated quartic).

USAGE:  python m5_8_2l_invariant_completion.py classify     (CPU, instant)
        python m5_8_2l_invariant_completion.py a4 [steps]   (GPU, ~6 min)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V8 = HERE.parent / "sandbox_v8"
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8.m5_8_2c1_full_evolution import (  # noqa: E402
    DT, central, tw,
)
from openwave.xperiments.m5_liquid_crystal.research.sandbox_vn.m5_8_2h_omega_attractor import (  # noqa: E402
    np_commf,
)


def classify_main():
    import sympy as sp
    print("=" * 78)
    print("[N-5 ii] the covariant 𝒮 + β𝒮² — variational classification"
          " (sympy-checked)")
    print("=" * 78)
    qd, a, u, b = sp.symbols("qdot a u beta", real=True, positive=True)
    T = a * qd ** 2
    L = (T - u) + b * (T - u) ** 2
    p = sp.expand(sp.diff(L, qd))
    deg = sp.degree(sp.Poly(p, qd))
    Lstat = sp.expand(L.subs(qd, 0))
    print(f"  1-mode exact reduction  L = a·q̇² − u + β(a·q̇² − u)²")
    print(f"  p = ∂L/∂q̇ = {p}")
    print(f"  degree of p in q̇: {deg}   (quadratic kinetic ⇒ 1; here ⇒"
          " NON-quadratic Legendre)")
    print(f"  static sector L(q̇=0) = {Lstat}  ⇒ static H = u − β·u²:")
    print("  the STABILIZING branch needs β < 0 in 𝒮+β𝒮², which then matches the validated")
    print("  u + |β|·u² EXACTLY — same one-parameter static family, no new static physics")
    print("\n  field-level corollaries (by the same algebra):")
    print("  · P(Ṁ) is cubic per voxel-mode ⇒ the P → Ṁ inversion needs a"
          " cubic solve")
    print("  · the inertia operator becomes A_eff = (1+2β(T−U))·A +"
          " 2β·(∂T/∂Ṁ)⊗(∂T/∂Ṁ)")
    print("    — state-dependent + rank-augmented: the constrained spectral"
          " projection must be")
    print("    rebuilt per step on A_eff, not A. ⇒ DYNAMIC TEST DEFERRED"
          " (the cap's honest finding);")
    print("  · the Derrick/static content of 𝒮+β𝒮² (β<0 branch) coincides"
          " with u+βu² — already measured (2d).")
    print("=" * 78)
    return 0


def a4_main():
    from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8 import (
        m5_8_2d_quartic_saturation as d2,
    )
    ti = d2.ti

    @ti.kernel
    def k_amp_force(beta_a: ti.f32, c0: ti.f32, dt: ti.f32):
        """P += dt·(−dV_A/dM), V_A = β_A(⟨M,M⟩_s − c₀)²; d⟨M,M⟩_s/dM = 2·tw(M)."""
        for i, j, k in d2.Pf:
            m = d2.Mf[i, j, k]
            mt = m
            for r_ in ti.static(range(1, 4)):          # tw = ηmη: flip (0,α)
                mt[0, r_] = -mt[0, r_]                 # and (α,0); [0,0] stays
                mt[r_, 0] = -mt[r_, 0]
            s2 = (m * mt).sum()
            d2.Pf[i, j, k] += dt * (-4.0 * beta_a * (s2 - c0) * mt)

    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    seed = d2.load_seed()
    M0, Mth, act, core, _, h = seed
    actb = act > 0.5
    n_act = int(act.sum())
    Mt0 = tw(M0)
    s2_seed = np.einsum("...ab,...ab->...", M0, Mt0)
    c0 = float(np.median(s2_seed[actb]))
    dev = np.abs(s2_seed[actb] - c0)
    p95 = float(np.percentile(dev, 95))
    betas = [0.3 / max(p95, 1e-9), 1.0 / max(p95, 1e-9), 3.0 / max(p95, 1e-9)]
    print("=" * 78)
    print(f"[N-5 i] A4 amplitude quartic — V_A = β_A(⟨M,M⟩_s − c₀)², ALONE"
          f" (β_u = 0), {steps} steps, kicked clock 0.05")
    print(f"  c₀ = {c0:.4f} (act-median seed amplitude); dev p95 = {p95:.4f};"
          f"  β_A scan: " + ", ".join(f"{b:.4g}" for b in betas))
    print("=" * 78)
    import time as _t
    inv2h = 1.0 / (2.0 * h)
    for ba in betas:
        d2.Mf.from_numpy(M0.astype(np.float32))
        d2.actf.from_numpy(act.astype(np.int32))
        d2.Bas.from_numpy(d2.SYM_BASIS.astype(np.float32))
        Md0 = (0.05 * Mth) * act[..., None, None]
        d2.Mdf.from_numpy(Md0.astype(np.float32))
        Mi = [central(M0, ax, h) for ax in range(3)]
        P0 = np.zeros_like(M0)
        for i_ in range(3):
            P0 += np_commf(tw(np_commf(Md0, Mi[i_])), Mi[i_])
        d2.Pf.from_numpy((4.0 * P0).astype(np.float32))
        tr = dict(H=[], s=[], onset=None)
        t0 = _t.time()
        for n_ in range(steps):
            d2.k_flux(inv2h, 0.0)                       # β_u = 0: A4 alone
            d2.k_force(inv2h, DT)
            k_amp_force(ba, c0, DT)
            d2.red.fill(0.0)
            d2.k_clamp_sum()
            r = d2.red.to_numpy()
            d2.k_clamp_apply(float(r[0]) / n_act, float(r[1]) / n_act,
                             float(r[2]) / n_act)
            d2.k_solve(inv2h)
            d2.k_update(DT)
            if n_ % 100 == 99:
                M = d2.Mf.to_numpy().astype(np.float64)
                Md = d2.Mdf.to_numpy().astype(np.float64)
                u = d2.uf.to_numpy().astype(np.float64)
                Mi = [central(M, ax, h) for ax in range(3)]
                T = 0.0
                for i_ in range(3):
                    F0 = np_commf(Md, Mi[i_])
                    T = T + 2.0 * np.einsum("...ab,...ab->...", F0, tw(F0))
                s2 = np.einsum("...ab,...ab->...", M, tw(M))
                H = float((T + u)[actb].sum() * h ** 3
                          + ba * ((s2 - c0) ** 2)[actb].sum() * h ** 3)
                s = float(np.einsum("...ab,...ab->...", M - M0, Mth)[core].mean())
                tr["H"].append(H)
                tr["s"].append(abs(s))
                if tr["onset"] is None and len(tr["H"]) > 3 and H < 0:
                    tr["onset"] = n_ + 1
                if n_ % 2000 == 1999:
                    print(f"   [β_A={ba:.3g}] step {n_ + 1:5d}"
                          f" [{_t.time() - t0:3.0f}s]  H={H:12.4f}"
                          f"  |s|={abs(s):.2e}")
                if not np.isfinite(H):
                    print(f"   [β_A={ba:.3g}] NON-FINITE at {n_ + 1}")
                    break
        Hs, ss = np.array(tr["H"]), np.array(tr["s"])
        print(f"  [β_A={ba:.3g}] H {Hs[0]:.2f} → {Hs[-1]:.2f}"
              f" (min {Hs.min():.2f});  onset(H<0) = {tr['onset']};"
              f"  |s| {ss[0]:.2e} → {ss[-1]:.2e}")
    print("\n  read vs the 2d ladder: quadratic control runaway onset ~1150;"
          " the signed-u quartic")
    print("  saturates with the clock GROWING. A4 verdict = the trend table"
          " above (bounded? clock?).")
    print("=" * 78)
    return 0


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "classify"
    if mode == "classify":
        return classify_main()
    if mode == "a4":
        return a4_main()
    print(__doc__.split("USAGE:")[1])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
