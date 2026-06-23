"""
M5.8.2cB-2 — headless repro of the _topo_dressed4d_signed GUI explosion
(63³, the exact launcher constrained path, Metal f32). Rodrigo's GUI smoke
scrambled by ~step 31 and blew up around ~1000+ steps; the pre-handoff Metal
sanity ran exactly 1003 steps and saw ‖Ṁ‖ DECREASING — stopped just under the
horizon (the 2c-2 short-check lesson, one octave up).

THE HONEST GAP: the B-1 spike validated the constrained kernel at V = 0 (the
2c-1 reference is V=0); production runs V-ON (dressed-t* well) + the
plane-sampled momentum clamp + the production seed geometry at 63³. Variants
isolate which delta is the pump:

    (a) GUI config exact:  V-on,  plane clamp        — expected to reproduce
    (b) V-off twin:        V=0,   plane clamp        — is V the pump?
    (c) no-clamp twin:     V-on,  clamp disabled     — is the plane clamp the pump?
    (d) V-off + no-clamp:  the bare constrained kernel at production scale

Probes every 250 steps: max|M|, max‖Ṁ‖, mean|M − M_seed| (act), and the
(0,α) coherent-drift magnitude. 6000-step horizon (~90 s/variant on Metal).

USAGE:  python m5_8_2cb2_signed_gui_repro.py
"""
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import taichi as ti

ti.init(arch=ti.metal, default_fp=ti.f32)

from openwave.xperiments.m5_liquid_crystal import medium
from openwave.xperiments.m5_liquid_crystal import engine1_seeds as seeds
from openwave.xperiments.m5_liquid_crystal import engine2_pde as pde

EDGE = 1e-15
N_TARGET = 64**3
STEPS = 6000
PROBE = 250


def build():
    tf = medium.TensorField([EDGE] * 3, N_TARGET, [0.5, 0.5, 0.5], viz_stride=2)
    n, c = tf.nx, tf.nx // 2
    seeds.seed_dressed_hedgehog_M(tf, c, c, c, 0.06 * n, 3.0, 0.30, 0.13, 0.29 * n, 0.05)
    pde.compute_tstar(tf)
    idx = np.indices((n, n, n)).astype(np.float32)
    r_vox = np.sqrt(((idx - c) ** 2).sum(0))
    rho_vox = np.sqrt((idx[0] - c) ** 2 + (idx[1] - c) ** 2)
    act = np.zeros((n, n, n), np.float32)
    act[2:-2, 2:-2, 2:-2] = 1.0
    act *= (r_vox > 6.0) * (rho_vox > 3.0)
    tf.act4d.from_numpy(act)
    basis = np.zeros((10, 4, 4), np.float32)
    for a in range(4):
        basis[a, a, a] = 1.0
    bi = 4
    for a in range(4):
        for c_ in range(a + 1, 4):
            basis[bi, a, c_] = basis[bi, c_, a] = 1.0 / np.sqrt(2.0)
            bi += 1
    tf.sym_basis.from_numpy(basis)
    n_act_pl = float(act[n // 2].sum() + act[:, n // 2].sum() + act[:, :, n // 2].sum())
    return tf, act, n_act_pl


def run_variant(tag, ldg_k, clamp_on, steps=STEPS):
    tf, act, n_act_pl = build()
    c_amrs = 0.2998
    dt_rs = 0.007 * tf.dx_am * 0.95 / (c_amrs * 3**0.5)
    dt_eff = c_amrs * dt_rs
    cc4d = 0.5 * ldg_k / tf.dx_am**4
    pde.init_P_4d(tf, 0.05)        # the 2c-1 velocity-kick semantics (CLOCK_KICK)
    M_seed = tf.M_am.to_numpy().astype(np.float64)
    actb = act > 0.5
    # seed director (principal eigenvector of M_sp) — the orientation-coherence
    # reference: align(t) = mean |⟨n̂(t), n̂(0)⟩| over act. THE glyph-scramble
    # metric ("bounded" alone gates amplitude only — GUI chaos is ORIENTATION;
    # this probe sees it, max|M| does not).
    _, v0 = np.linalg.eigh(M_seed[..., 1:4, 1:4][actb])
    n0 = v0[..., -1]
    print(f"\n[{tag}] N={tf.nx} dt_eff={dt_eff:.4f} cc4d={cc4d:.2e} "
          f"clamp={'plane' if clamp_on else 'OFF'}")
    t0 = time.time()
    for n_ in range(steps):
        pde.flux_4d_constrained(tf)
        pde.update_P_4d(tf, dt_eff, cc4d)
        if clamp_on:
            pde.sample_p03_drift(tf)
            pde.apply_p03_clamp(tf, tf.v03_sums[0] / n_act_pl,
                                tf.v03_sums[1] / n_act_pl, tf.v03_sums[2] / n_act_pl)
        pde.solve_constrained_4d(tf)
        pde.update_M_4d_constrained(tf, dt_eff)
        tf.swap_matrix_buffers()
        if n_ % PROBE == PROBE - 1 or n_ == 30:
            M = tf.M_am.to_numpy().astype(np.float64)
            Md = tf.Md_am.to_numpy().astype(np.float64)
            mx = float(np.abs(M).max())
            vmax = float(np.sqrt(np.einsum("...ab,...ab->...", Md, Md)).max())
            dM = float(np.abs(M - M_seed).max(axis=(-1, -2))[actb].mean())
            d03 = float(np.abs(M[..., 1:4, 0][actb]).mean())
            _, vt = np.linalg.eigh(M[..., 1:4, 1:4][actb])
            align = float(np.abs(np.einsum("...i,...i->...", vt[..., -1], n0)).mean())
            print(f"   step {n_ + 1:5d} [{time.time() - t0:4.0f}s] max|M|={mx:10.3e} "
                  f"max‖Ṁ‖={vmax:8.3f} ⟨|ΔM|⟩={dM:9.3e} ⟨|M_a0|⟩={d03:9.3e} "
                  f"align={align:.3f}")
            if not np.isfinite(mx) or mx > 1e4:
                print(f"   → EXPLODED by step {n_ + 1}")
                return False, n_ + 1
    print("   → bounded through the run")
    return True, steps


def main():
    print("=" * 78)
    print("M5.8.2cB-2 — signed-GUI explosion repro (63³ Metal f32, constrained path)")
    print("=" * 78)
    res = {}
    res["a"] = run_variant("a: GUI exact (V-on, plane clamp)", 1.0, True)
    res["b"] = run_variant("b: V-OFF twin (plane clamp)", 0.0, True)
    res["c"] = run_variant("c: V-on, clamp OFF", 1.0, False)
    res["d"] = run_variant("d: V-OFF, clamp OFF (bare kernel)", 0.0, False)
    print("\n" + "=" * 78)
    for k, (ok, n_) in res.items():
        print(f"  ({k}) {'bounded' if ok else f'EXPLODED @ {n_}'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
