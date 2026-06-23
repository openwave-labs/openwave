# ✅ MIGRATED to the 4×4 substrate (2026-06-05) — RUNNABLE key-finding reproduction.
# Uses the PRODUCTION seeder/kernels (4×4-native since M5.8.1); the Tr(M²) amplitude
# analysis now reads the SPATIAL 3×3 block (V_M acts spatial-only — the time-g axis is
# decoupled), so the confinement check is UNCHANGED. (TensorField was WaveField pre-M5.8.)
# Re-validated on 4×4 (2026-06-05): confines ~3.3× across k∈[0.5,25], no blow-up — consistent.
"""M5.6.5c — find PRODUCTION-unit LdG coefficients with the real evolve_M kernel.

The sandbox proved the b=0 amplitude well confines (dimensionless). But production
evolve_M uses physical dx_am/c_amrs and dt_rs²≈dx²·3.34 — raw sandbox coeffs blow up.
This sweeps the well stiffness in production units (scaled by the cubic-balance factor
c_amrs²/dx_am⁴) and finds the magnitude that confines without blowing up. Result: the
b=0 well confines ~3.3× across K∈[0.5,25] with no blow-up (dt²-stable) ⇒ the launcher
sets ldg_c = K·c²/dx⁴, ldg_a = −2·ldg_c·(1+δ²), default K=1 (xparameter LDG_STIFFNESS_K).

USAGE:
    python -m openwave.xperiments.m5_liquid_crystal.research.sandbox_v6.m5_6_5c_prod_scale
"""
import numpy as np, taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f32)
from openwave.xperiments.m5_liquid_crystal import medium
from openwave.xperiments.m5_liquid_crystal import engine1_seeds as seeds
from openwave.xperiments.m5_liquid_crystal import engine2_pde as pde

EDGE_AM = 1e-15 / 1e-18                       # 1e-15 m in am = 1000 am (matches _biaxial1)
N = 56
DELTA = 0.30
S2_STAR = 1.0 + DELTA**2
N_STEPS = 400

tf = medium.TensorField([EDGE_AM * 1e-18] * 3, target_voxels=N**3)
dx = tf.dx_am
c_amrs = 0.3
dt_rs = dx * 0.95 / (c_amrs * np.sqrt(3.0))   # launcher CFL formula at SIM_SPEED=1
cfl = (c_amrs * dt_rs / dx) ** 2
nn = tf.nx
c = nn // 2
r0_vox = 0.06 * nn
rhoc_vox = 3.0
print(f"grid {nn}³  dx_am={dx:.2f}  c_amrs={c_amrs}  dt_rs={dt_rs:.2f}  cfl={cfl:.3f}  dt²={dt_rs**2:.0f}")

# natural cubic-balance unit: curvature kick ~ c²·M³/dx⁴ ⇒ coeff scale ~ c²/dx⁴
base = c_amrs**2 / dx**4
print(f"cubic-balance unit c²/dx⁴ = {base:.3e}\n")


def amp_dev_after(ldg_c, n_steps):
    """Seed fresh, run n_steps of evolve_M with b=0 well (a=−2c·s2*), return (max_dev, finite)."""
    seeds.seed_biaxial_hedgehog_M(tf, c, c, c, r0_vox, rhoc_vox, DELTA)
    a = -2.0 * ldg_c * S2_STAR
    devs = []
    for step in range(n_steps):
        pde.compute_curvature_flux(tf)
        pde.evolve_M(tf, c_amrs, dt_rs, a, 0.0, ldg_c)
        tf.swap_matrix_buffers()
        if step % 40 == 0:
            M = tf.M_am.to_numpy()
            # M5.8.1: spatial 3×3 block only — V_M pins the SPATIAL Tr(M²); the
            # constant-g time axis (index 0) is decoupled and excluded.
            Msp = M[..., 1:4, 1:4]
            s2 = np.einsum("...ab,...ab->...", Msp, Msp)   # Tr(M_sp²)=‖M_sp‖_F²
            # interior, off the disclination z-axis + point core
            xs = np.arange(nn) - c
            X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
            r = np.sqrt(X**2 + Y**2 + Z**2); rho = np.sqrt(X**2 + Y**2)
            m = np.zeros(r.shape, bool); m[3:-3, 3:-3, 3:-3] = True
            m &= (r > 2 * r0_vox) & (rho > rhoc_vox)
            devs.append(float(np.mean(np.abs(s2 - S2_STAR)[m])))
            if not np.isfinite(M).all() or np.abs(M).max() > 1e3:
                return max(devs), False
    return max(devs), True


print(f"{'coeff (×c²/dx⁴)':>16} {'ldg_c':>12} {'max amp-dev':>12} {'finite':>8}")
dev_off, fin_off = amp_dev_after(0.0, N_STEPS)
print(f"{'V OFF':>16} {0.0:>12.3e} {dev_off:>12.4f} {str(fin_off):>8}")
results = []
for k in (0.5, 1.0, 2.0, 5.0, 10.0, 25.0):
    ldg_c = k * base
    dev, fin = amp_dev_after(ldg_c, N_STEPS)
    flag = "✓ confines" if (fin and dev < 0.7 * dev_off) else ("blow-up" if not fin else "weak")
    print(f"{k:>16.1f} {ldg_c:>12.3e} {dev:>12.4f} {str(fin):>8}  {flag}")
    if fin and dev < 0.7 * dev_off:
        results.append((k, ldg_c, dev))

print()
if results:
    k, ldg_c, dev = min(results, key=lambda t: t[2])
    print(f"BEST: k={k} → ldg_c={ldg_c:.3e}, ldg_a={-2*ldg_c*S2_STAR:.3e}, ldg_b=0  "
          f"(amp-dev {dev:.4f} vs V-off {dev_off:.4f})")
    print(f"  production rule: ldg_c = {k}·c_amrs²/dx_am⁴, ldg_a = −2·ldg_c·(1+δ²)")
else:
    print("no confining coeff found in sweep — widen range")
