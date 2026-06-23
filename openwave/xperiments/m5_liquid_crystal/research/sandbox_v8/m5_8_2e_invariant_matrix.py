"""
M5.8.2e — the INVARIANT MATRIX + closeouts (Track B of the in-house program).

Builds on m5_8_2d (imports its fields/kernels — that module ti.inits Metal f32
at import; this script must NOT re-init). Adds a GENERALIZED flux kernel with
two saturation knobs + an Euclidean solve twin, and four modes:

    skyrme  — (A3) the literature-canonical candidate: the Euclid-norm quartic
              β_E·u_E² (u_E = 2Σ‖F_ij‖²_F — Faddeev–Skyrme class, the Hopf
              bound's home). β_E scan anchored to the seed's u_E scale,
              6000 steps each, the 2d gate ladder.
    euclid  — the KILL-CONTROL in the saturated regime: full-Euclidean twin
              (tw ≡ identity in flux AND inertia; positive-definite — bounded
              by construction, no quartic needed) of the winning config, same
              seed + kick, 24000 steps — measure the clock spectrum: the 2b-2
              signature-cohesion discriminator at FULL nonlinearity (Minkowski
              single-mode vs Euclid multi-mode is the claim under test).
    omega   — the breathing-ω measurement: the winner (β_s=1.558, dt/2) for
              48000 steps with DENSE s(t)/H(t) capture (every 20 steps) → FFT
              of the post-bounce window → the first measured ω of a bounded
              3+1D state (the M5.8.3 doorstep). Saves _m5_8_2e_omega.npz.
    f64     — the f64 numpy confirmation of the winning quartic config
              (the 2c-1 loop + the quartic flux, 3000 steps, ~20 min): the
              f32 Metal trajectory class must reproduce (plateau → relax,
              NO runaway) at full precision.

USAGE:  python m5_8_2e_invariant_matrix.py <skyrme|euclid|omega|f64>

PREREQUISITE: `_m5_8_2cb_ref.npz` (regenerate: `CB_STEPS=900 python
m5_8_2cb_taichi_constrained.py ref`).

RESULTS (2026-06-05 late night — full log in 0b_M5_roadmap.md § M5.8.2e):
    skyrme — ALL THREE β_E (1.37/4.57/13.7) saturate (bounded, align
        0.936–0.948) but the clock is 10× weaker (|s| ~3×10⁻³): sign-blind,
        damps fuel AND clock. The SIGNED quartic is the time-crystal-preferred
        invariant.
    euclid — THE KILL-CONTROL HEADLINE: same seed+kick, the Euclid twin's
        clock DECAYS (1.4e-2 → 6.7e-3) while the Minkowski+quartic clock
        GROWS (7.6e-3 → 2.4e-2). The signature IS the engine, at full
        nonlinearity in the saturated regime.
    omega — the saturated state is STRICTLY PERIODIC: ω₀ = 0.262 with an
        EXACT 2–5× harmonic comb (breathing period τ = 24, matching the
        bounce). The fast 2b-2 clock (5.86) is not dominant in s(t) — a
        phase-sensitive probe is the refinement item. This realization
        re-showed late-time stepper heating (onset varies with FP ordering).
    f64 — H 66.6 → 57.9 over 3000 steps, plateau→relax, NO runaway: the f32
        results stand at full precision (~1% per-probe agreement).
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# importing m5_8_2d runs its module-level ti.init(metal, f32) + field allocs
import openwave.xperiments.m5_liquid_crystal.research.sandbox_v8.m5_8_2d_quartic_saturation as d2  # noqa: E402
import taichi as ti  # noqa: E402
from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8.m5_8_2c1_full_evolution import (  # noqa: E402
    DT, EPS_EIG, VCAP, SYM_BASIS, central, tw, to_coeff, from_coeff,
    build_A_matrix, solve_constrained, A_apply,
)

n = d2.n
Mf, Mdf, Pf, Gf, uf = d2.Mf, d2.Mdf, d2.Pf, d2.Gf, d2.uf
actf, Bas, red = d2.actf, d2.Bas, d2.red
twm, comm, sdot = d2.twm, d2.comm, d2.sdot
gx, gy, gz = d2.gx, d2.gy, d2.gz


@ti.kernel
def k_flux_gen(inv2h: ti.f32, beta_s: ti.f32, beta_e: ti.f32, euclid: ti.i32):
    """G_i = (1+2β_s·u_s)·dU_signed + (2β_e·u_e)·dU_euclid + dT.

    euclid=1 ⇒ the FULL Euclidean theory (tw ≡ identity in dU and dT; u_s
    then equals u_e). uf stores the saturation-relevant density for probes:
    u_s (signed runs) or u_e (Skyrme/Euclid runs use the β_e channel)."""
    for i, j, k in Mf:
        mx = gx(i, j, k, inv2h)
        my = gy(i, j, k, inv2h)
        mz = gz(i, j, k, inv2h)
        md = Mdf[i, j, k]
        fxy = comm(mx, my)
        fxz = comm(mx, mz)
        fyz = comm(my, mz)
        u_e = 2.0 * ((fxy * fxy).sum() + (fxz * fxz).sum() + (fyz * fyz).sum())
        u_s = u_e
        if euclid == 0:
            u_s = 2.0 * (sdot(fxy, fxy) + sdot(fxz, fxz) + sdot(fyz, fyz))
        uf[i, j, k] = u_s if beta_s > 0.0 else u_e
        pref_s = 1.0 + 2.0 * beta_s * u_s
        pref_e = 2.0 * beta_e * u_e
        if euclid == 1:
            # full-Euclid: quadratic flux + optional Euclid quartic, no signs
            Gf[0, i, j, k] = (4.0 * (pref_s + pref_e) * (comm(fxy, my) + comm(fxz, mz))
                              + 4.0 * comm(comm(md, mx), md))
            Gf[1, i, j, k] = (4.0 * (pref_s + pref_e) * (comm(-fxy, mx) + comm(fyz, mz))
                              + 4.0 * comm(comm(md, my), md))
            Gf[2, i, j, k] = (4.0 * (pref_s + pref_e) * (comm(-fxz, mx) + comm(-fyz, my))
                              + 4.0 * comm(comm(md, mz), md))
        else:
            Gf[0, i, j, k] = (4.0 * pref_s * (comm(twm(fxy), my) + comm(twm(fxz), mz))
                              + 4.0 * pref_e * (comm(fxy, my) + comm(fxz, mz))
                              + 4.0 * comm(twm(comm(md, mx)), md))
            Gf[1, i, j, k] = (4.0 * pref_s * (comm(twm(-fxy), mx) + comm(twm(fyz), mz))
                              + 4.0 * pref_e * (comm(-fxy, mx) + comm(fyz, mz))
                              + 4.0 * comm(twm(comm(md, my)), md))
            Gf[2, i, j, k] = (4.0 * pref_s * (comm(twm(-fxz), mx) + comm(twm(-fyz), my))
                              + 4.0 * pref_e * (comm(-fxz, mx) + comm(-fyz, my))
                              + 4.0 * comm(twm(comm(md, mz)), md))


@ti.kernel
def k_solve_e(inv2h: ti.f32):
    """The Euclidean inertia solve: A_E(Ṁ) = 4Σ[[Ṁ,M_i],M_i] is PSD by
    construction (⟨V,A_E V⟩ = 2Σ‖[V,M_i]‖² ≥ 0) — same spectral machinery,
    no sign structure (the kill-control twin of d2.k_solve)."""
    for i, j, k in Mf:
        mx = gx(i, j, k, inv2h)
        my = gy(i, j, k, inv2h)
        mz = gz(i, j, k, inv2h)
        aloc = ti.Matrix.zero(ti.f32, 10, 10)
        for li in range(10):
            el = Bas[li]
            ae = 4.0 * (comm(comm(el, mx), mx) + comm(comm(el, my), my)
                        + comm(comm(el, mz), mz))
            for kk in range(10):
                aloc[kk, li] = (ae * Bas[kk]).sum()
        for p_ in range(10):
            for q_ in range(p_ + 1, 10):
                v = 0.5 * (aloc[p_, q_] + aloc[q_, p_])
                aloc[p_, q_] = v
                aloc[q_, p_] = v
        qloc = ti.Matrix.identity(ti.f32, 10)
        normf2 = (aloc * aloc).sum()
        off2 = normf2
        sweep = 0
        while sweep < d2.MAX_SWEEPS and off2 > d2.JTOL2 * normf2:
            for p_ in range(9):
                for q_ in range(p_ + 1, 10):
                    apq = aloc[p_, q_]
                    if ti.abs(apq) > 0:
                        tau = (aloc[q_, q_] - aloc[p_, p_]) / (2.0 * apq)
                        t = 1.0 / (ti.abs(tau) + ti.sqrt(1.0 + tau * tau))
                        if tau < 0:
                            t = -t
                        c = 1.0 / ti.sqrt(1.0 + t * t)
                        s = t * c
                        for r_ in range(10):
                            arp = aloc[r_, p_]
                            arq = aloc[r_, q_]
                            aloc[r_, p_] = c * arp - s * arq
                            aloc[r_, q_] = s * arp + c * arq
                        for r_ in range(10):
                            apr = aloc[p_, r_]
                            aqr = aloc[q_, r_]
                            aloc[p_, r_] = c * apr - s * aqr
                            aloc[q_, r_] = s * apr + c * aqr
                        for r_ in range(10):
                            qrp = qloc[r_, p_]
                            qrq = qloc[r_, q_]
                            qloc[r_, p_] = c * qrp - s * qrq
                            qloc[r_, q_] = s * qrp + c * qrq
            off2 = 0.0
            for p_ in range(9):
                for q_ in range(p_ + 1, 10):
                    off2 += 2.0 * aloc[p_, q_] * aloc[p_, q_]
            sweep += 1
        lmax = 0.0
        for a_ in range(10):
            lmax = ti.max(lmax, ti.abs(aloc[a_, a_]))
        pm = Pf[i, j, k]
        pc = ti.Vector.zero(ti.f32, 10)
        for a_ in range(10):
            pc[a_] = (pm * Bas[a_]).sum()
        pproj = ti.Vector.zero(ti.f32, 10)
        cdot = ti.Vector.zero(ti.f32, 10)
        for kk in range(10):
            lamk = aloc[kk, kk]
            qp = 0.0
            for a_ in range(10):
                qp += qloc[a_, kk] * pc[a_]
            if lamk > EPS_EIG * (lmax + 1e-30):
                for a_ in range(10):
                    pproj[a_] += qloc[a_, kk] * qp
                    cdot[a_] += qloc[a_, kk] * qp / lamk
        pnew = ti.Matrix.zero(ti.f32, 4, 4)
        mdnew = ti.Matrix.zero(ti.f32, 4, 4)
        for a_ in range(10):
            pnew += pproj[a_] * Bas[a_]
            mdnew += cdot[a_] * Bas[a_]
        if actf[i, j, k] == 0:
            mdnew = ti.Matrix.zero(ti.f32, 4, 4)
        vn = ti.sqrt((mdnew * mdnew).sum())
        if vn > VCAP:
            mdnew = mdnew * (VCAP / (vn + 1e-30))
        Pf[i, j, k] = pnew
        Mdf[i, j, k] = mdnew


def run_gen(tag, beta_s, beta_e, euclid, seed, dt, steps, s_every=0):
    """Generalized evolution; probes the 2d set + dense s(t) when s_every>0.
    H probe: T(signed or euclid per mode) + u_s + β_s u_s² + β_e u_e²
    (u channels recomputed in numpy at probe time for exactness)."""
    M0, Mth, act, core, rhat, h = seed
    n_act = int(act.sum())
    actb = act > 0.5
    Mf.from_numpy(M0.astype(np.float32))
    Pf.fill(0.0)
    actf.from_numpy(act.astype(np.int32))
    Bas.from_numpy(SYM_BASIS.astype(np.float32))
    inv2h = 1.0 / (2.0 * h)
    Md0 = 0.05 * Mth
    Mdf.from_numpy((Md0 * act[..., None, None]).astype(np.float32))
    Mi = [central(M0, ax, h) for ax in range(3)]
    tw_eff = (lambda A: A) if euclid else tw
    P0 = np.zeros_like(M0)
    for i_ in range(3):
        P0 += d2.np_commf(tw_eff(d2.np_commf(Md0, Mi[i_])), Mi[i_])
    Pf.from_numpy((4.0 * P0).astype(np.float32))
    _, v0 = np.linalg.eigh(M0[..., 1:4, 1:4][actb])
    n0 = v0[..., -1]
    Hs, aligns, ss, s_dense = [], [], [], []
    t0 = time.time()
    onset = None
    for n_ in range(steps):
        k_flux_gen(inv2h, beta_s, beta_e, 1 if euclid else 0)
        d2.k_force(inv2h, dt)
        red.fill(0.0)
        d2.k_clamp_sum()
        r = red.to_numpy()
        d2.k_clamp_apply(float(r[0]) / n_act, float(r[1]) / n_act, float(r[2]) / n_act)
        if euclid:
            k_solve_e(inv2h)
        else:
            d2.k_solve(inv2h)
        d2.k_update(dt)
        if s_every and n_ % s_every == s_every - 1:
            M = Mf.to_numpy().astype(np.float64)
            s_dense.append(float(np.einsum("...ab,...ab->...", M - M0, Mth)[core].mean()))
        if n_ % d2.PROBE == d2.PROBE - 1:
            M = Mf.to_numpy().astype(np.float64)
            Md = Mdf.to_numpy().astype(np.float64)
            Mi = [central(M, ax, h) for ax in range(3)]
            T = 0.0
            u_s = 0.0
            u_e = 0.0
            for i_ in range(3):
                F0 = d2.np_commf(Md, Mi[i_])
                T = T + 2.0 * np.einsum("...ab,...ab->...", F0, tw_eff(F0))
                for j_ in range(i_ + 1, 3):
                    F = d2.np_commf(Mi[i_], Mi[j_])
                    u_e = u_e + 2.0 * np.einsum("...ab,...ab->...", F, F)
                    u_s = u_s + 2.0 * np.einsum("...ab,...ab->...", F, tw_eff(F))
            H = float((T + u_s + beta_s * u_s * u_s + beta_e * u_e * u_e)[actb].sum()) * h**3
            _, vt = np.linalg.eigh(M[..., 1:4, 1:4][actb])
            al = float(np.abs(np.einsum("...i,...i->...", vt[..., -1], n0)).mean())
            s = float(np.einsum("...ab,...ab->...", M - M0, Mth)[core].mean())
            Hs.append(H)
            aligns.append(al)
            ss.append(s)
            if onset is None and len(Hs) > 3 and H < 0:
                onset = n_ + 1
            if n_ % 2000 == 1999:
                print(f"   [{tag}] step {n_ + 1:5d} [{time.time() - t0:4.0f}s] "
                      f"H={H:12.4f} align={al:.3f} |s|={abs(s):.2e}")
            if not np.isfinite(H):
                print(f"   [{tag}] NON-FINITE at step {n_ + 1}")
                break
    return dict(Hs=np.array(Hs), align=np.array(aligns), s=np.array(ss),
                s_dense=np.array(s_dense), onset=onset)


def mode_skyrme():
    print("=" * 78)
    print("[A3/E2] the SKYRME candidate: β_E·u_E² (Euclid-norm quartic) on the")
    print("        SIGNED dynamics — the Faddeev–Skyrme-class invariant, 6000 steps/run")
    print("=" * 78)
    seed = d2.load_seed()
    M0, _, act, _, _, h = seed
    Mi = [central(M0, ax, h) for ax in range(3)]
    u_e = 0.0
    for i_ in range(3):
        for j_ in range(i_ + 1, 3):
            F = d2.np_commf(Mi[i_], Mi[j_])
            u_e = u_e + 2.0 * np.einsum("...ab,...ab->...", F, F)
    ue_p95 = float(np.percentile(u_e[act > 0.5], 95))
    print(f"  seed u_E scale: p95 = {ue_p95:.3f}")
    betas = [0.15 / ue_p95, 0.5 / ue_p95, 1.5 / ue_p95]
    print("  β_E scan: " + ", ".join(f"{b:.4g}" for b in betas))
    ok_any = False
    for b in betas:
        print(f"\n  β_E = {b:.4g}:")
        R = run_gen(f"skyrme β_E={b:.3g}", 0.0, b, False, seed, DT, 6000)
        bounded = d2.trend_report(R, f"β_E={b:.3g}")
        ok_any = ok_any or (bounded and R["align"][-1] > 0.8)
    print(f"\n[A3 verdict] Skyrme Euclid-quartic saturates the signed dynamics: "
          f"{'YES (≥1 β_E bounded+coherent)' if ok_any else 'NO in this scan'}")
    return 0 if ok_any else 1


def mode_euclid():
    print("=" * 78)
    print("[E-control] FULL-Euclidean twin (positive-definite — bounded by")
    print("            construction), same seed+kick, 24000 steps: the signature-")
    print("            cohesion kill-control in the nonlinear regime")
    print("=" * 78)
    seed = d2.load_seed()
    R = run_gen("euclid", 0.0, 0.0, True, seed, DT, 24000, s_every=20)
    d2.trend_report(R, "euclid")
    np.savez(HERE / "_m5_8_2e_euclid.npz", **{k: v for k, v in R.items() if k != "onset"})
    print(f"  s(t) dense series saved ({len(R['s_dense'])} samples) → "
          f"_m5_8_2e_euclid.npz (spectrum vs the Minkowski winner in `omega`)")
    return 0


def mode_omega():
    print("=" * 78)
    print("[ω] the breathing-ω measurement: winner (β_s=1.558), dt/2, 48000 steps,")
    print("    dense s(t)/H(t) — FFT of the post-bounce window (the M5.8.3 doorstep)")
    print("=" * 78)
    seed = d2.load_seed()
    R = run_gen("omega β_s=1.558", 1.558, 0.0, False, seed, DT / 2, 48000, s_every=20)
    d2.trend_report(R, "omega")
    np.savez(HERE / "_m5_8_2e_omega.npz", **{k: v for k, v in R.items() if k != "onset"})
    s = R["s_dense"]
    if len(s) > 1000:
        late = s[len(s) // 2:]                       # post-bounce window
        late = late - late.mean()
        f = np.fft.rfftfreq(len(late), d=20 * DT / 2)  # cycles per τ
        amp = np.abs(np.fft.rfft(late))
        pk = np.argmax(amp[1:]) + 1
        omega = 2 * np.pi * f[pk]
        print(f"  [ω] dominant clock/breathing mode: ω = {omega:.3f} "
              f"(2b-2 linear reference: 5.86; apolar-doubled field rate)")
        top = np.argsort(amp[1:])[::-1][:5] + 1
        print("  [ω] top-5 peaks: " + ", ".join(
            f"ω={2 * np.pi * f[i]:.3f}(A={amp[i]:.2e})" for i in top))
    return 0


def mode_f64(beta=1.558, steps=3000):
    print("=" * 78)
    print(f"[f64] numpy confirmation of the winning quartic config (β={beta}, "
          f"{steps} steps — plateau→relax, NO runaway expected)")
    print("=" * 78)
    seed = d2.load_seed()
    M0, Mth, act, core, rhat, h = seed
    actb = act > 0.5
    M = M0.copy()
    Md = 0.05 * Mth * act[..., None, None]
    Mi = [central(M, ax, h) for ax in range(3)]
    P = A_apply(Md, Mi)
    t0 = time.time()
    Hs = []
    for n_ in range(steps):
        # force: quartic-scaled dU + dT (the 2c-1 rhs with the d2 quartic flux)
        Mi = [central(M, ax, h) for ax in range(3)]
        dU, u = d2.np_quartic_dU(Mi, beta)
        dT = [-4.0 * d2.np_commf(tw(d2.np_commf(Md, Mi[i_])), Md) for i_ in range(3)]
        force = (sum(central(dU[ax], ax, h) for ax in range(3))
                 - sum(central(dT[ax], ax, h) for ax in range(3)))
        P = P + DT * force
        for a_ in range(1, 4):                       # spatial-eigen boost axes {1,2,3} (index-0)
            P[..., a_, 0] -= P[..., a_, 0][actb].mean() * actb
            P[..., 0, a_] = P[..., a_, 0]
        Amat = build_A_matrix(Mi)
        cdot, keep, Pc_proj = solve_constrained(Amat, to_coeff(P))
        P = from_coeff(Pc_proj)
        Md = from_coeff(cdot) * act[..., None, None]
        vn = np.sqrt(np.einsum("...ab,...ab->...", Md, Md))
        if (vn > VCAP).any():
            Md = Md * np.where(vn > VCAP, VCAP / (vn + 1e-30), 1.0)[..., None, None]
        M = M + DT * Md
        if n_ % 100 == 99:
            T = 0.0
            for i_ in range(3):
                F0 = d2.np_commf(Md, Mi[i_])
                T = T + 2.0 * np.einsum("...ab,...ab->...", F0, tw(F0))
            H = float((T + u + beta * u * u)[actb].sum()) * h**3
            Hs.append(H)
            if n_ % 500 == 499:
                print(f"   [f64] step {n_ + 1:5d} [{time.time() - t0:4.0f}s] H={H:10.4f}")
            if not np.isfinite(H):
                print("   [f64] NON-FINITE")
                return 1
    Hs = np.array(Hs)
    ok = bool(np.isfinite(Hs).all() and (Hs > 0).all() and Hs[0] > Hs[-1] > 0)
    print(f"  [f64] H: {Hs[0]:.1f} → {Hs[-1]:.1f} over {steps} steps — "
          f"{'CONFIRMS the f32 trajectory class (plateau→relax, no runaway)' if ok else 'MISMATCH — inspect'}")
    return 0 if ok else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    mode = a[0] if a else ""
    if mode == "skyrme":
        raise SystemExit(mode_skyrme())
    if mode == "euclid":
        raise SystemExit(mode_euclid())
    if mode == "omega":
        raise SystemExit(mode_omega())
    if mode == "f64":
        raise SystemExit(mode_f64())
    print(__doc__)
    raise SystemExit(2)
