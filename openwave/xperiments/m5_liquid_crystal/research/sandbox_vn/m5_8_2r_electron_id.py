"""
M5.8.2r, ELECTRON-ID (EID): the 3-way EM/QM/GEM split + the 3x3 fixed-clock
electron (mu and J). Spec: 0b_M5_roadmap.md § ELECTRON-ID PROJECT (frozen
2026-06-09). Source: Duda round-3 ("try getting such electron with 3x3 field
and fixed clock, to get proper magnetic dipole moment and angular momentum").

EID-B  3-way sector split. The 2q split partitions lab-frame index pairs
  (spatial vs time-mixing) and cannot separate tilt from twist. Here F is
  conjugated into the analytic hedgehog frame O4 (orthogonal, block-diag,
  commutes with eta, so the signed contraction is preserved) and the spatial
  block splits by eigen-axis pairs (1=major/1, 2=minor/delta, 3=zero, 0=time):
    TILT  (EM) = pairs (1,2), (1,3)   - they move the major axis
    TWIST (QM) = pair  (2,3)          - rotation about the major axis (Duda's Gamma^1)
    BOOST (GEM)= pairs (0,a)          - eta-negative time-mixing
  Gates: TILT+TWIST+GEM = H_quad to machine precision; the 2q gate (16.7379
  at delta=0.3, g=8, b=0.13) keeps passing; TWIST ~ delta^2 at b=0.

EID-C  the 3x3 fixed-clock electron. Seed at b=0: M = conj(O4, D4) =
  block-diag(3x3 biaxial hedgehog, inert g). Clock PINNED at phase phi:
  W(phi) = O4 . rot4(plane, phi), Mdot = omega * Mth(phi) with
  Mth = conj(W, G D4 - D4 G) analytic (no evolution, no time stepping).
    J: momentum density p_i = <P, d_i M>, P = A_apply(Mdot, Mi)  (the 2c1
       conjugate momentum); J = sum_vox r x p h^3.
    mu (director / Mermin-Ho route): F^EM_ij = n.(d_i n x d_j n),
       B_k = 1/2 eps_kij F_ij; time part E_i = n.(d_t n x d_i n) with
       d_t n = omega * dn/dphi (analytic clock); j = curl B - omega dE/dphi;
       mu = 1/2 sum_vox r x j h^3.
  BOTH clock planes are scanned: twist (2,3) (the validated PLANE: the
  director does NOT move, so the abelian current may vanish by symmetry)
  and tilt (1,2) (the director precesses: a real circulating current).
  A zero is reported as a finding, not hidden.

OMEGA CONVENTION (G-EID-3, stated once): omega = omega_clock = 1 in lattice
units. The apolar doubling (G7: omega_M = 2 omega_clock) applies to M-field
FFT observables; J and mu below are computed from the analytic Mdot at the
M-field level and scale linearly with the chosen omega, so the RATIO mu/J is
omega-independent and the g-factor uses matched conventions throughout.

RESULTS (2026-06-10, EID COMPLETE at the static-seed level):

  EID-B (all gates PASS; sum exact to 1e-11; the 2q 16.7379 gate holds):
  | b    | EM (R^1) | QM (R^2,R^3) | GEM (boost) |
  | 0.13 | 16.34    | 2.23         | -9.37       |
  | 0.01 | 10.83    | 1.38         | -0.06       |
  | 0.00 | 10.82    | 1.39         |  0.00       |
  The SECTOR MAP note below was CORRECTED BY THE FIRST RUN: tilt x tilt
  curvature points ALONG the major generator (Faber R = Gamma x Gamma), so
  EM = component pair (2,3). With the right labels the measurement lands
  exactly on Duda's hierarchy: EM dominant, QM small (delta-weighted over a
  ~0.2 geometric floor from the regularized disclination frame), GEM small
  and negative. EM(delta->0) -> the 2q delta-flat hedgehog floor (~19).

  EID-C findings (static pinned clock; omega = 1 lattice):
  1. ORBITAL J = 0 STRUCTURALLY. r x p, localized-clock, and Poynting
     L_EM = r x (ExB) ALL vanish to 1e-14 while max|p| ~ O(1): the centered
     hedgehog is dyon-like (emergent E and B both radial from one center,
     E || B kills the Thomson term) and the rigid clock is equivariant.
     The static face of the 2p J-neutrality.
  2. SPIN lives in the Noether charge of the clock rotation, L_int =
     sum <P, Mth> h^3 (the L/Q = omega identity family): finite, and
     phi-independent for the twist clock (61.61 +/- 0.03% at 24^3).
  3. mu = 0 for the TWIST clock at every grid/dressing - structural: the
     abelian (director-projected) current is blind to twist; Duda's Gamma^1
     clock carries no EM dipole in this construction. mu != 0 ONLY for the
     TILT (precession) channel: mu = 0.221 (24^3, phi=0 linear response,
     b-independent). The electron's dipole requires the tilt component of
     the Zitterbewegung, not the twist phase.
  4. Tilt clock at FINITE phi destroys the hedgehog (phi = pi/2 turns the
     director into the disclination texture) - the tilt channel is only
     meaningful infinitesimally; its phi = 0 value is the linear-response
     moment.
  5. RESIDUALS (-> NG-1/NG-3): (a) box convergence - mu 0.221/0.248/0.277
     and L_int 61.6/65.1/68.3 across 24/32/48 (~+11%/step, tail-dominated
     integrals; need bigger boxes or radial windowing); (b) the g-factor
     is NOT computable without the cross-sector normalization: mu is in
     director-curvature units, L_int in action units, and their ratio is
     set exactly by the Coulomb e_scale calibration (the spec's "unit-free
     g" claim was WRONG - corrected here). Duda's "fix units by comparing
     with Coulomb" is REQUIRED for g, not optional.

USAGE:
  python m5_8_2r_electron_id.py            # EID-B + EID-C at 24^3, phi=0
  python m5_8_2r_electron_id.py gates      # + phi sweep + box ladder 24/32/48
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8.m5_8_2a_4d_hamiltonian import (  # noqa: E402
    conj, boost_field, matmul, rot4, gen4,
)
from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8.m5_8_2c1_full_evolution import (  # noqa: E402
    L, B_STAR, A_BOOST, central, tw, A_apply,
)
from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8.m5_8_2cb_taichi_constrained import (  # noqa: E402
    build_grid_n,
)
from openwave.xperiments.m5_liquid_crystal.research.sandbox_vn.m5_8_2h_omega_attractor import (  # noqa: E402
    np_commf,
)

DELTA = 0.3
G_TIME = 8.0
RC, RHOC = 0.8, 0.8
BETA = 1.558
OMEGA = 1.0                       # lattice clock rate (see OMEGA CONVENTION)
# SECTOR MAP (corrected by the first run, 2026-06-10): curvature components
# label by the GENERATOR axis, not by the moved axes. R = Gamma x Gamma of two
# TILTS points ALONG the major axis, whose generator acts in plane (2,3):
#   EM  (tilt-tilt, Duda R^1)      = component pair (2,3)
#   QM  (tilt-twist, Duda R^2,R^3) = component pairs (1,2), (1,3)  [delta weight]
#   GEM (boosts)                   = pairs (0,a), eta-negative
# First-run magnitudes with these labels: EM 16.3 dominant, QM 2.2 small,
# GEM -9.4 at clock dressing / -0.06 at physical boost = Duda's hierarchy.
EM_PAIR = [(2, 3)]
QM_PAIRS = [(1, 2), (1, 3)]
GEM_PAIRS = [(0, 1), (0, 2), (0, 3)]


def D4_of(delta=DELTA, g=G_TIME):
    return np.diag([g, 1.0, delta, 0.0])


def grid_and_seed(n, delta=DELTA, g=G_TIME, b=B_STAR, plane=(2, 3), phi=0.0):
    """Grid + pinned-clock seed W(phi) = O4 . boost . rot4(plane, phi)."""
    gr = build_grid_n(n, L)
    w = np.exp(-((gr["r"] / 3.5) ** 2))
    W = matmul(gr["O4"], boost_field(b * w, A_BOOST))
    if phi != 0.0:
        W = W @ rot4(plane, phi)
    D4 = D4_of(delta, g)
    M = conj(W, D4)
    G = gen4(plane)
    Mth = conj(W, G @ D4 - D4 @ G)            # the analytic clock tangent
    inter = np.zeros(gr["r"].shape, bool)
    inter[2:-2, 2:-2, 2:-2] = True
    act = inter & (gr["r"] > 2 * RC) & (gr["rho"] > RHOC)
    return gr, W, M, Mth, act


# ── EID-B: the 3-way split ───────────────────────────────────────────────────

def split3(M, O4, act, h):
    """(EM, QM, GEM, H_quad) energies; F conjugated to the hedgehog frame."""
    Mi = [central(M, ax, h) for ax in range(3)]
    em = qm = gem = quad = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            F = np_commf(Mi[i], Mi[j])
            Fh = np.einsum("...ca,...cd,...db->...ab", O4, F, O4)   # O4^T F O4
            t1 = sum(Fh[..., a, b] ** 2 for a, b in EM_PAIR)
            t2 = sum(Fh[..., a, b] ** 2 for a, b in QM_PAIRS)
            t3 = sum(Fh[..., a, b] ** 2 for a, b in GEM_PAIRS)
            em = em + 4.0 * t1
            qm = qm + 4.0 * t2
            gem = gem - 4.0 * t3
            quad = quad + 2.0 * np.einsum("...ab,...ab->...", F, tw(F))
    s = lambda u: float(u[act].sum()) * h ** 3
    return s(em), s(qm), s(gem), s(quad)


def run_eid_b():
    print("=" * 78)
    print("EID-B: the 3-way EM/QM/GEM split  [hedgehog frame]")
    print("=" * 78)
    n = 24
    print("\n  |    b   | EM (tilt-tilt R¹) | QM (tilt-twist R²R³) | GEM (boost) | sum-vs-H_quad |")
    print("  | --- | --- | --- | --- | --- |")
    for b in (0.13, 0.01, 0.0):
        gr, W, M, _, act = grid_and_seed(n, b=b)
        # NOTE: the frame used is the analytic hedgehog O4 (exact eigenframe at
        # b=0; at b>0 the boost is not undone - documented approximation).
        em, qm, gm, q = split3(M, gr["O4"], act, gr["h"])
        ok = "PASS ✓" if abs(em + qm + gm - q) < 1e-8 * max(1, abs(q)) else "FAIL ✗"
        print(f"  | {b:5.2f} | {em:10.4f} | {qm:10.4f} | {gm:+11.4f} |"
              f" {em + qm + gm - q:+.2e} {ok} |")

    # the 2q gate must keep passing (H_static at delta=.3, g=8, b=.13)
    gr, W, M, _, act = grid_and_seed(n, b=0.13)
    Mi = [central(M, ax, gr["h"]) for ax in range(3)]
    u = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            F = np_commf(Mi[i], Mi[j])
            u = u + 2.0 * np.einsum("...ab,...ab->...", F, tw(F))
    Hs = float((u + BETA * u * u)[act].sum()) * gr["h"] ** 3
    print(f"\n  [gate] H_static(δ=0.3, g=8, b=0.13) = {Hs:.4f}  (record 16.7379)"
          f"  {'PASS ✓' if abs(Hs - 16.7379) < 0.01 else 'FAIL ✗'}")

    # delta-scaling of the two spatial sectors at b=0: EM should approach the
    # delta-flat hedgehog floor (the 2q [C] result); QM carries the delta weight
    print("\n  δ-scaling at b=0 (EM → δ-flat floor; QM → carries the δ weight):")
    print("  |    δ    | EM (R¹) | QM (R²R³) | QM/δ | QM/δ² |")
    print("  | --- | --- | --- | --- | --- |")
    for d in (0.3, 0.1, 0.03, 0.01):
        gr, W, M, _, act = grid_and_seed(n, delta=d, b=0.0)
        em, qm, _, _ = split3(M, gr["O4"], act, gr["h"])
        print(f"  | {d:7.3f} | {em:9.4f} | {qm:9.5f} | {qm / d:8.4f} | {qm / d**2:9.3f} |")
    return 0


# ── EID-C: the fixed-clock electron ─────────────────────────────────────────

def director_of(W):
    """n = the major (eigenvalue-1) axis in space frame = first column of W's
    spatial block (exact at b=0; at b=0 W is orthogonal block-diag)."""
    return W[..., 1:4, 1]


def J_of(gr, M, Mth, act):
    """Field angular momentum J = sum r x p h^3, p_i = <P, d_i M> (signed)."""
    h = gr["h"]
    Mi = [central(M, ax, h) for ax in range(3)]
    P = A_apply(OMEGA * Mth, Mi)
    p = np.stack([np.einsum("...ab,...ab->...", P, tw(central(M, ax, h)))
                  for ax in range(3)], axis=-1)
    n = M.shape[0]
    c = (n - 1) / 2.0
    idx = (np.indices((n, n, n)).transpose(1, 2, 3, 0) - c) * h
    Jv = np.cross(idx, p)[act].sum(axis=0) * h ** 3
    return Jv


def EB_of(n_hat, dn_dphi, h):
    """Emergent (E, B) from the director: B = Mermin-Ho curvature,
    E_i = n.(d_t n x d_i n) with d_t n = omega dn/dphi (the pinned clock)."""
    dn = [central(n_hat, ax, h) for ax in range(3)]
    F12 = np.einsum("...a,...a->...", n_hat, np.cross(dn[0], dn[1]))
    F13 = np.einsum("...a,...a->...", n_hat, np.cross(dn[0], dn[2]))
    F23 = np.einsum("...a,...a->...", n_hat, np.cross(dn[1], dn[2]))
    B = np.stack([F23, -F13, F12], axis=-1)
    dtn = OMEGA * dn_dphi
    E = np.stack([np.einsum("...a,...a->...", n_hat, np.cross(dtn, dn[ax]))
                  for ax in range(3)], axis=-1)
    return E, B


def curl(V, h):
    return np.stack([
        central(V[..., 2], 1, h) - central(V[..., 1], 2, h),
        central(V[..., 0], 2, h) - central(V[..., 2], 0, h),
        central(V[..., 1], 0, h) - central(V[..., 0], 1, h)], axis=-1)


def mu_of(gr, EB_now, EB_next, dphi2, act):
    """mu = 1/2 sum r x j h^3 with the FULL Maxwell current
    j = curl B - omega dE/dphi (displacement term from the steady rotation)."""
    h = gr["h"]
    (E0, B0), (E1, _) = EB_now, EB_next
    j = curl(B0, h) - OMEGA * (E1 - E0) / dphi2
    n = E0.shape[0]
    c = (n - 1) / 2.0
    idx = (np.indices((n, n, n)).transpose(1, 2, 3, 0) - c) * h
    mu = 0.5 * np.cross(idx, j)[act].sum(axis=0) * h ** 3
    jmax = float(np.linalg.norm(j, axis=-1)[act].max())
    return mu, jmax, float(np.abs(E0[act]).max())


def run_eid_c(n=24, phis=(0.0,), planes=((2, 3), (1, 2)), bs=(0.0, B_STAR)):
    print("\n" + "=" * 78)
    print(f"EID-C: the fixed-clock electron  [{n}³, ω={OMEGA}]")
    print("=" * 78)
    dphi, dphi2 = 1e-4, 1e-3
    print("\n  | plane | b | φ | |J|(r×p) | |J|(loc-clock) | max|p| | |μ| | |L_EM| | L_int (Noether) |")
    print("  | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    out = {}
    for plane in planes:
        tag = "twist" if plane == (2, 3) else "tilt"
        for b in bs:
            for phi in phis:
                gr, W, M, Mth, act = grid_and_seed(n, b=b, plane=plane, phi=phi)
                h = gr["h"]
                Mi = [central(M, ax, h) for ax in range(3)]
                P = A_apply(OMEGA * Mth, Mi)
                p = np.stack([np.einsum("...ab,...ab->...", P, tw(Mi[ax]))
                              for ax in range(3)], axis=-1)
                nn = M.shape[0]
                c = (nn - 1) / 2.0
                idx = (np.indices((nn, nn, nn)).transpose(1, 2, 3, 0) - c) * h
                Jv = np.cross(idx, p)[act].sum(axis=0) * h ** 3
                pmax = float(np.linalg.norm(p, axis=-1)[act].max())
                # localized-clock variant: the dressing envelope w(r) breaks
                # the rigid-rotation symmetry that cancels the uniform-clock J
                w_env = np.exp(-((gr["r"] / 3.5) ** 2))
                P_loc = A_apply(OMEGA * w_env[..., None, None] * Mth, Mi)
                p_loc = np.stack([np.einsum("...ab,...ab->...", P_loc, tw(Mi[ax]))
                                  for ax in range(3)], axis=-1)
                Jloc = np.cross(idx, p_loc)[act].sum(axis=0) * h ** 3
                # director route mu, with displacement current
                def eb(ph):
                    _, Wx, _, _, _ = grid_and_seed(n, b=b, plane=plane, phi=ph)
                    n0 = director_of(Wx)
                    _, Wy, _, _, _ = grid_and_seed(n, b=b, plane=plane,
                                                   phi=ph + dphi)
                    return EB_of(n0, (director_of(Wy) - n0) / dphi, h)
                E0, B0 = eb(phi)
                mu, jmax, Emax = mu_of(gr, (E0, B0), eb(phi + dphi2), dphi2, act)
                # EM-sector angular momentum (Poynting): the Thomson pairing
                LEM = np.cross(idx, np.cross(E0, B0))[act].sum(axis=0) * h ** 3
                # L_int: the Noether charge of the clock rotation (spin as
                # internal rotation; the M6 L/Q = omega identity family).
                # L_int = sum <P, Mth> h^3 = (2/omega) T_clock.
                Lint = float(np.einsum("...ab,...ab->...", P,
                                       tw(Mth))[act].sum()) * h ** 3
                Jm, mum, Lm = (np.linalg.norm(Jv), np.linalg.norm(mu),
                               np.linalg.norm(LEM))
                print(f"  | {tag} | {b:4.2f} | {phi:4.2f} | {Jm:.3e} |"
                      f" {np.linalg.norm(Jloc):.3e} | {pmax:.2e} | {mum:.3e} |"
                      f" {Lm:.3e} | {Lint:+.4f} |")
                out[(tag, b, phi)] = (Jm, mum, Lint, LEM, mu)
    return out


def main():
    run_eid_b()
    out = run_eid_c()
    if len(sys.argv) > 1 and sys.argv[1] == "gates":
        print("\n" + "=" * 78)
        print("GATES: φ-sweep (G-EID-2) + box ladder (G-EID-1)")
        print("=" * 78)
        run_eid_c(24, phis=(0.0, 0.79, 1.57, 2.36))
        for n in (32, 48):
            run_eid_c(n, phis=(0.0,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
