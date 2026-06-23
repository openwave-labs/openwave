"""
M5.8.2p — N-6e: THE FIRST SPIN READOUT — the rotation Noether charge J on
the breather, the spontaneous state, and the cold settled state.

THE DERIVATION (the load-bearing part, machine-gated below):
  The action L = ∫d³x (T − U) is built entirely from commutators and
  signed-Frobenius contractions of M and its derivatives ⇒ it is invariant
  under the SIMULTANEOUS spatial rotation + internal frame rotation
      M(x) → R M(R⁻¹x) Rᵀ,   R = exp(θ G) embedded 4×4 (time axis inert).
  The Noether charge for axis a:
      J_a = Σ_x h³ ⟨P, δ_aM⟩_F,
      δ_aM = −(x × ∇)_a M  +  [G_a⁴, M]
             └─ ORBITAL ─┘     └─ INTERNAL ("spin") ─┘
  with P the code's own canonical momentum (the constrained integrator's
  Pf — the object that generates Ṁ). Pairing convention gated:
    G1  ⟨P, Ṁ⟩ = 2T exactly at a kicked init (Euler relation for the
        quadratic kinetic — validates plain-Frobenius pairing for J);
    G2  conservation: J_a(t) along an evolution — the lattice + box +
        momentum clamp break exact rotation symmetry, so the DRIFT is
        measured and reported, not assumed zero.

THE MEASUREMENT (first reading, no pass/fail):
  Arms: (i) the KICKED clock state (kick 0.05 — the clock kick injects
  internal rotation; J ≠ 0 expected), (ii) the SPONTANEOUS breather
  (P₀ = 0 exact ⇒ J(0) = 0 exactly; conservation ⇒ does the selected state
  STAY spin-silent, or does symmetry breaking grow a net J?), (iii) the
  COLD settled restart (the regular clock post-N-6c — same question at
  low excitation). All three axes (J_x, J_y, J_z), internal + orbital
  split, h³-weighted, probed every 200 steps over 6000 steps.
  ℏ/2 comparison under the N-6b anchors (P2: lattice action ↔ ℏ):
  spin-½ ⟺ |J_int| ≈ 0.5 in lattice units — the COMPARISON CLASS, not a
  claimed result: the kicked J scales with the kick (not intrinsic), and
  the spontaneous/settled states choose their own J. First measurement
  framing throughout.

RESULTS (2026-06-07 — N-6e COMPLETE: the first spin readout is a clean
NULL + an instrument bound):
  G1 pairing: ⟨P,Ṁ⟩/(2T) = 1.000000 EXACT — the Noether machinery is
    validated against the code's own canonical momentum.
  ★ THE CLOCK KICK IS J-NEUTRAL: the kicked clock state reads
    J_int = J_orb = 0.0000 (< 1e-4) at t → 0 despite finite P — the
    Θ-clock twist summed over the hedgehog sphere CANCELS: the ZBW
    breathing/clock channel carries NO net frame angular momentum.
    The spontaneous and cold arms start at J = 0 exactly (P₀ = 0).
  ★ G2 (conservation) FAILED INFORMATIVELY — BOX TORQUE: J(t) grows
    secularly in every arm (O(1–10) by t = 6; the wall-contacting settled
    arm reads J ~ 10 immediately) — the 24³/L = 6 boundary breaks rotation
    invariance (clamped-edge gradients exert torque once the field
    reaches the walls; act-domain vs interior-domain comparison confirms
    region flux + walls, not the clamp, dominate). The lattice-anisotropy
    contribution is subdominant at early t.
  ⇒ VERDICT (first measurement, honestly bounded): the canonical M5.8
    states carry NO intrinsic angular momentum above the box-torque noise
    floor — spin-½, if this model carries it, does NOT live in the
    breathing/clock channel as net frame J at sandbox scale. Candidate
    carriers for the resumed track (NG-1/M5.9): the polarized seed class
    (the Wolfram-slide ellipsoid spinning about its director), the
    hopfion sector, or far-field tilt J outside the box. An ℏ/2-class
    measurement needs torque-free boundaries (bigger box / absorbing
    far field — production-scale, the engine-port territory).
  The instrument (J decomposition, 3 axes, internal + orbital, gated) is
  the durable deliverable — it ships with the repo.

USAGE:  python m5_8_2p_spin_readout.py [steps]      (default 6000)
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V8 = HERE.parent / "sandbox_v8"
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8.m5_8_2c1_full_evolution import (  # noqa: E402
    N, L, DT, central, tw,
)
from openwave.xperiments.m5_liquid_crystal.research.sandbox_vn.m5_8_2h_omega_attractor import (  # noqa: E402
    np_commf,
)

BETA = 1.558
PROBE = 200

# 4×4 embeddings of the spatial rotation generators (time = matrix index 0, inert;
# spatial-eigen matrix axes {1,2,3} under the index-0 convention)
G4 = np.zeros((3, 4, 4))
G4[0, 2, 3], G4[0, 3, 2] = -1.0, 1.0          # G_x: rotation about x (mixes eigen-axes 2,3)
G4[1, 3, 1], G4[1, 1, 3] = -1.0, 1.0          # G_y (mixes eigen-axes 3,1)
G4[2, 1, 2], G4[2, 2, 1] = -1.0, 1.0          # G_z (mixes eigen-axes 1,2)


def J_measure(P, M, h, XYZ, actb):
    """(J_int[3], J_orb[3]) = Σ h³⟨P, δM⟩ over the act region."""
    X, Y, Z = XYZ
    Mi = [central(M, ax, h) for ax in range(3)]
    Jint, Jorb = np.zeros(3), np.zeros(3)
    Lops = ((Y, Mi[2], Z, Mi[1]),              # (x×∇)_x = y∂_z − z∂_y
            (Z, Mi[0], X, Mi[2]),              # (x×∇)_y = z∂_x − x∂_z
            (X, Mi[1], Y, Mi[0]))              # (x×∇)_z = x∂_y − y∂_x
    for a in range(3):
        dint = np_commf(np.broadcast_to(G4[a], M.shape), M)
        Jint[a] = float(np.einsum("...ab,...ab->...", P, dint)[actb].sum()) * h ** 3
        c1, m1, c2, m2 = Lops[a]
        dorb = -(c1[..., None, None] * m1 - c2[..., None, None] * m2)
        Jorb[a] = float(np.einsum("...ab,...ab->...", P, dorb)[actb].sum()) * h ** 3
    return Jint, Jorb


def run_arm(d2, tag, M_start, act, h, XYZ, dt, steps, kick, Mth, M0):
    actb = act > 0.5
    inter = np.zeros(act.shape, bool)           # J domain: FULL interior —
    inter[2:-2, 2:-2, 2:-2] = True              # the act boundary leaks J
    jdom = inter
    n_act = int(act.sum())
    inv2h = 1.0 / (2.0 * h)
    d2.Mf.from_numpy(M_start.astype(np.float32))
    d2.Mdf.fill(0.0)
    d2.Pf.fill(0.0)
    d2.actf.from_numpy(act.astype(np.int32))
    d2.Bas.from_numpy(d2.SYM_BASIS.astype(np.float32))
    if kick > 0.0:
        Md0 = (kick * Mth) * act[..., None, None]
        d2.Mdf.from_numpy(Md0.astype(np.float32))
        Mi = [central(M_start, ax, h) for ax in range(3)]
        P0 = np.zeros_like(M_start)
        T0 = 0.0
        for i_ in range(3):
            F0 = np_commf(Md0, Mi[i_])
            P0 += np_commf(tw(F0), Mi[i_])
            T0 = T0 + 2.0 * np.einsum("...ab,...ab->...", F0, tw(F0))
        P0 *= 4.0
        d2.Pf.from_numpy(P0.astype(np.float32))
        # G1 — the pairing gate: ⟨P, Ṁ⟩ must equal 2T (Euler relation)
        pdot = float(np.einsum("...ab,...ab->...", P0, Md0)[actb].sum()) * h ** 3
        twoT = 2.0 * float(T0[actb].sum()) * h ** 3
        print(f"   [G1 pairing] ⟨P,Ṁ⟩/(2T) = {pdot / twoT:.6f}  (gate: = 1)")
    rows = []
    t0 = time.time()
    for n_ in range(steps):
        d2.k_flux(inv2h, BETA)
        d2.k_force(inv2h, dt)
        d2.red.fill(0.0)
        d2.k_clamp_sum()
        r = d2.red.to_numpy()
        d2.k_clamp_apply(float(r[0]) / n_act, float(r[1]) / n_act,
                         float(r[2]) / n_act)
        d2.k_solve(inv2h)
        d2.k_update(dt)
        if n_ % PROBE == PROBE - 1:
            M = d2.Mf.to_numpy().astype(np.float64)
            P = d2.Pf.to_numpy().astype(np.float64)
            Ji, Jo = J_measure(P, M, h, XYZ, jdom)
            rows.append((n_ + 1, *Ji, *Jo))
    rows = np.array(rows)
    print(f"   [{tag}] {steps} steps [{time.time() - t0:.0f}s] — J probes:")
    print("   | step | J_int x | y | z | J_orb x | y | z | |J_int| |")
    print("   | --- | --- | --- | --- | --- | --- | --- | --- |")
    for k in (0, len(rows) // 2, len(rows) - 1):
        r_ = rows[k]
        jmag = np.linalg.norm(r_[1:4])
        print(f"   | {int(r_[0])} | {r_[1]:+.4f} | {r_[2]:+.4f} |"
              f" {r_[3]:+.4f} | {r_[4]:+.4f} | {r_[5]:+.4f} | {r_[6]:+.4f} |"
              f" {jmag:.4f} |")
    jmags = np.linalg.norm(rows[:, 1:4], axis=1)
    drift = (jmags.max() - jmags.min()) / max(jmags.mean(), 1e-12)
    print(f"   [G2 conservation] |J_int| mean = {jmags.mean():.4f},"
          f" relative drift over the run = {100 * drift:.1f}%")
    return rows


def main():
    from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8 import (
        m5_8_2d_quartic_saturation as d2,
    )
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    print("=" * 78)
    print("M5.8.2p — N-6e: the first spin readout (Noether J, internal +"
          " orbital, 3 axes)")
    print("=" * 78)
    M0, Mth, act, core, _, h = d2.load_seed()
    xs = np.linspace(-L, L, N)
    XYZ = np.meshgrid(xs, xs, xs, indexing="ij")
    dt = DT / 2
    out = {}
    print("\n[arm 1 — the KICKED clock state (kick 0.05)]")
    out["kicked"] = run_arm(d2, "kicked", M0, act, h, XYZ, dt, steps,
                            0.05, Mth, M0)
    print("\n[arm 2 — the SPONTANEOUS breather (P₀ = 0 exact)]")
    out["spont"] = run_arm(d2, "spont", M0, act, h, XYZ, dt, steps,
                           0.0, Mth, M0)
    zs = np.load(HERE / "data" / "_m5_8_2o_settled.npz")
    print("\n[arm 3 — the COLD settled restart (the regular clock)]")
    out["cold"] = run_arm(d2, "cold", zs["M_settled"], act, h, XYZ, dt,
                          steps, 0.0, Mth, M0)
    np.savez(HERE / "data" / "_m5_8_2p_spin.npz",
             **{k: v for k, v in out.items()}, dt=dt, probe=PROBE)
    print("\n" + "=" * 78)
    print("[N-6e reading] under the N-6b anchors (lattice action ↔ ℏ):"
          " spin-½ class ⟺ |J_int| ≈ 0.5")
    print("  (the z-rod breaks x/y rotation symmetry EXPLICITLY — J_z is"
          " the protected axis)")
    for k, v in out.items():
        jz = v[:, 3]
        jm = float(np.abs(jz).mean())
        print(f"  {k:8s} J_z(int): start {jz[0]:+.4f} → end {jz[-1]:+.4f},"
              f" |mean| {jm:.4f}   →  {jm / 0.5:.3f} × (ℏ/2)")
    print("  FIRST MEASUREMENT framing: the kicked J scales with the kick"
          " (not intrinsic); the")
    print("  spontaneous/cold states choose their own J — their values are"
          " the physical statement.")
    print("  saved → _m5_8_2p_spin.npz")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
