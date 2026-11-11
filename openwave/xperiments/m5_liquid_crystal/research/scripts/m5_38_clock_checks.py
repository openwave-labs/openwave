"""M5.38 R2: the two clock checks on the electron-ID instrument (tasks/m5_38_task_details.md § 3).

The instrument is m5_8_2r_electron_id.py, loaded UNMODIFIED through the import shim of m5_34_kit_checks.py
(its imports name a retired package layout). The object is its fixed-clock electron: the analytic 3x3 biaxial
hedgehog with the clock pinned at phase phi in the tilt plane (1, 2), W(phi) = O4 . boost . rot4((1, 2), phi).

    n(x; phi)   = the instrument's director (the major axis of W)
    B           = the Mermin-Ho curvature of n (the instrument's EB_of), the radial 2-form
    rho(x; phi) = div B                               e(phi) = sum rho h^3
    p(phi)      = sum x rho h^3                       c(phi) = p / e,  c = c_par z + c_perp  (z = the moment axis)
    Q_ij(phi)   = sum (3 x_i x_j - r^2 delta_ij) rho h^3
    <d^2>_orbit = < |c_perp|^2 >_phi                  <d^2>_axial = < (c_par - <c_par>)^2 >_phi

Check (a): the centroid's second moments over one clock cycle against (3/4) lambda_C^2 (the Casimir radius) and
(1/4) lambda_C^2 (the projection), lambda_C = c / omega (rest-phase reading) or 2 c / omega (beat reading).
Check (b): the moving part of the monopole, the dipole and the traceless quadrupole over the cycle, as the cycle
variation over e0 r_rms^l, with the harmonic of phi that carries it.

Every number is read on the FULL INTERIOR (core and string included) and, beside it, on the instrument's active
mask (core and string excluded). Controls run first: a planted orbiting Gaussian charge must come back at its
planted orbit radius and rotating quadrupole within 1 percent.

Run from this directory: python3 m5_38_clock_checks.py [--write]   ->  ../data/m5_38_clock_checks.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def _shim(pkg_path, modname, filename):
    parts = pkg_path.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            m = types.ModuleType(name); m.__path__ = []; sys.modules[name] = m
    full = pkg_path + "." + modname
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec); sys.modules[full] = m; spec.loader.exec_module(m)
    return m


BASE = "openwave.xperiments.m5_liquid_crystal.research"
_shim(BASE + ".sandbox_v6", "m5_6_2a_biaxial_hedgehog", "m5_6_2a_biaxial_hedgehog.py")
_shim(BASE + ".sandbox_v8", "m5_8_2a_4d_hamiltonian", "m5_8_2a_4d_hamiltonian.py")
_shim(BASE + ".sandbox_v8", "m5_8_2c1_full_evolution", "m5_8_2c1_full_evolution.py")
_shim(BASE + ".sandbox_v8", "m5_8_2cb_taichi_constrained", "m5_8_2cb_taichi_constrained.py")
_shim(BASE + ".sandbox_vn", "m5_8_2h_omega_attractor", "m5_8_2h_omega_attractor.py")
_spec = importlib.util.spec_from_file_location("eid", os.path.join(HERE, "m5_8_2r_electron_id.py"))
eid = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(eid)

NPHI = 16
PLANES = {"tilt": (1, 2), "twist": (2, 3)}   # tilt = the plan's read; twist = a reference row added at the run
AT_REST_BAR = 1e-9            # orbit radius over lambda_C
AUTHOR_BAR = 2.6e-9           # the author's bar for check (b), listed and not applied (its unit is author-gated)
E_MIN_FRAC = 0.05             # the centroid is defined where |e| exceeds this fraction of 4 pi


def div(V, h):
    return eid.central(V[..., 0], 0, h) + eid.central(V[..., 1], 1, h) + eid.central(V[..., 2], 2, h)


def coords(n, h):
    c = (n - 1) / 2.0
    return (np.indices((n, n, n)).transpose(1, 2, 3, 0) - c) * h


def moments(rho, X, mask, h):
    w = rho[mask] * h ** 3
    x = X[mask]
    e = float(w.sum())
    p = (x * w[:, None]).sum(axis=0)
    r2 = (x ** 2).sum(axis=1)
    Q = 3.0 * np.einsum("ni,nj,n->ij", x, x, w) - np.eye(3) * float((r2 * w).sum())
    return e, p, Q


def harmonics(series):
    """amplitudes of the harmonics 0..4 of a real series sampled at NPHI phases (any trailing shape)."""
    F = np.fft.rfft(np.asarray(series, float), axis=0) / len(series)
    amp = np.abs(F); amp[1:] *= 2.0
    return [float(np.linalg.norm(amp[k])) for k in range(5)]


def cycle_stats(e, p, Q, e_ref, r_rms, lam_c=1.0):
    e, p, Q = np.asarray(e), np.asarray(p), np.asarray(Q)
    ok = np.abs(e) > E_MIN_FRAC * 4 * np.pi
    c = np.full_like(p, np.nan)
    c[ok] = p[ok] / e[ok, None]
    c_par, c_perp = c[:, 2], c[:, :2]
    out = {"e_by_phase": e.tolist(), "p_by_phase": p.tolist(), "phases_with_centroid": int(ok.sum()),
           "centroid_by_phase": [x if np.all(np.isfinite(x)) else None for x in c.tolist()]}
    if ok.any():
        orbit2 = float(np.mean(np.sum(c_perp[ok] ** 2, axis=1)))
        axial2 = float(np.mean((c_par[ok] - c_par[ok].mean()) ** 2))
        out.update({"orbit_second_moment": orbit2, "axial_second_moment": axial2,
                    "orbit_radius_rms": orbit2 ** 0.5, "axial_rms": axial2 ** 0.5,
                    "orbit_over_lambda2_rest_phase_reading": orbit2 / lam_c ** 2,
                    "orbit_over_lambda2_beat_reading": orbit2 / (2 * lam_c) ** 2,
                    "max_abs_centroid": float(np.nanmax(np.linalg.norm(c[ok], axis=1))),
                    "axial_note": "c_par = p_z / e is read only where |e| > 5 percent of 4 pi; where e crosses zero the centroid is undefined and the axial numbers are set by that cut"})
    out["abs_e_min_over_max"] = float(np.min(np.abs(e)) / np.max(np.abs(e)))
    out["p_z_amplitude"] = float(np.max(np.abs(p[:, 2])))
    out["p_perp_max"] = float(np.max(np.abs(p[:, :2])))
    def moving(series, l):
        s = np.asarray(series, float)
        dev = s - s.mean(axis=0)
        return float(np.sqrt(np.mean(np.sum(dev.reshape(len(s), -1) ** 2, axis=1))) / (abs(e_ref) * r_rms ** l))
    out["moving_fraction"] = {"monopole_l0": moving(e, 0), "dipole_l1": moving(p, 1), "quadrupole_l2": moving(Q, 2)}
    out["harmonic_amplitudes_0_to_4"] = {"monopole": harmonics(e), "dipole": harmonics(p), "quadrupole": harmonics(Q)}
    out["dipole_axis_fraction_z"] = float(np.sum(p[:, 2] ** 2) / max(np.sum(p ** 2), 1e-300))
    Qm = Q - Q.mean(axis=0)
    m0 = Qm[:, 2, 2] ** 2 * 1.5                       # the m = 0 part about z of a traceless symmetric tensor
    tot = np.sum(Qm ** 2, axis=(1, 2))
    out["quadrupole_moving_part_m0_fraction_about_z"] = float(m0.sum() / max(tot.sum(), 1e-300))
    return out


def read(n, b, plane=PLANES["tilt"]):
    phis = 2 * np.pi * np.arange(NPHI) / NPHI
    rows = {"full_interior": ([], [], []), "active_mask": ([], [], [])}
    flux = []
    rho0 = None
    for ph in phis:
        gr, W, _, _, act = eid.grid_and_seed(n, b=b, plane=plane, phi=float(ph))
        h = gr["h"]; X = coords(n, h)
        n_hat = eid.director_of(W)
        _, B = eid.EB_of(n_hat, np.zeros_like(n_hat), h)
        rho = div(B, h)
        inter = np.zeros(rho.shape, bool); inter[2:-2, 2:-2, 2:-2] = True
        if rho0 is None:
            rho0 = rho.copy()
            w0 = np.abs(rho0[inter])
            r_rms = float(np.sqrt(((X[inter] ** 2).sum(axis=1) * w0).sum() / w0.sum()))
            e_ref = float(rho0[inter].sum() * h ** 3)
        for key, mask in (("full_interior", inter), ("active_mask", act)):
            e, p, Q = moments(rho, X, mask, h)
            rows[key][0].append(e); rows[key][1].append(p); rows[key][2].append(Q)
        k = 2
        flux.append(float((B[-1 - k, k:-k, k:-k, 0].sum() - B[k, k:-k, k:-k, 0].sum() + B[k:-k, -1 - k, k:-k, 1].sum() - B[k:-k, k, k:-k, 1].sum()
                           + B[k:-k, k:-k, -1 - k, 2].sum() - B[k:-k, k:-k, k, 2].sum()) * h ** 2))
    out = {"n": n, "h": float(h), "b": float(b), "L_box": float(eid.L), "omega": float(eid.OMEGA), "phases": phis.tolist(),
           "e_ref_full_interior_phi0": e_ref, "four_pi": float(4 * np.pi), "r_rms_of_abs_rho_phi0": r_rms,
           "flux_of_B_through_the_interior_faces_by_phase": flux}
    for key in rows:
        out[key] = cycle_stats(*rows[key], e_ref=e_ref, r_rms=r_rms)
    return out, (rho0, h, e_ref, r_rms)


def _tri_omega(a, b, c):
    """signed solid angle of the spherical triangle (a, b, c) of unit vectors (van Oosterom and Strackee)."""
    num = np.einsum("...i,...i->...", a, np.cross(b, c))
    den = 1.0 + np.einsum("...i,...i->...", a, b) + np.einsum("...i,...i->...", b, c) + np.einsum("...i,...i->...", c, a)
    return 2.0 * np.arctan2(num, den)


def degree_on_cube(nh, k):
    """the degree of the unit field nh on the closed surface of the sub-cube [k, n-1-k]^3 (vertices = cell centers)."""
    n = nh.shape[0]; sl = slice(k, n - k); tot = 0.0
    for ax in range(3):
        for side, sgn in ((k, -1), (n - 1 - k, +1)):
            idx = [sl, sl, sl]; idx[ax] = side
            F = nh[tuple(idx)]
            a, b, c, d = F[:-1, :-1], F[1:, :-1], F[1:, 1:], F[:-1, 1:]
            orient = sgn * (1 if ax != 1 else -1)          # the remaining axes (u, v): u x v = +x, -y, +z
            tot += orient * float((_tri_omega(a, b, c) + _tri_omega(a, c, d)).sum())
    return tot / (4 * np.pi)


TOPO_PHASES = (0.0, 0.125, 0.25, 0.375, 0.45, 0.49, 0.5, 0.51, 0.75, 1.0)     # in units of pi


def topology(n):
    """added on the audit: is the cycle variation of the lattice charge a property of the director's topology?
    Per phase: the norm of the instrument's director (it is not a unit field near the axis at phi != 0), the
    degree of the normalized director on two closed surfaces, the lattice charge from the raw and the normalized
    director, and the flux of the normalized director's curvature through the faces of the interior cube, split
    into the part off the axis (cylindrical radius above 3 h), against 4 pi cos(phi), and the part near the axis."""
    rows = []
    for f in TOPO_PHASES:
        gr, W, _, _, _ = eid.grid_and_seed(n, b=0.0, plane=PLANES["tilt"], phi=f * np.pi)
        h = gr["h"]; X = coords(n, h)
        raw = eid.director_of(W); nrm = np.linalg.norm(raw, axis=-1); nh = raw / nrm[..., None]
        rho_cyl = np.sqrt(X[..., 0] ** 2 + X[..., 1] ** 2)
        _, Bn = eid.EB_of(nh, np.zeros_like(nh), h)
        _, Br = eid.EB_of(raw, np.zeros_like(raw), h)
        inter = np.zeros(nrm.shape, bool); inter[2:-2, 2:-2, 2:-2] = True
        k = 2; off = near = 0.0
        for a in range(3):
            for side, sgn in ((k, -1), (n - 1 - k, +1)):
                idx = [slice(k, n - k)] * 3; idx[a] = side; idx = tuple(idx)
                fl = sgn * Bn[idx][..., a] * h ** 2; m = rho_cyl[idx] > 3 * h
                off += float(fl[m].sum()); near += float(fl[~m].sum())
        rows.append({"phi_over_pi": f, "director_norm_min": float(nrm.min()),
                     "degree_cube_k2": degree_on_cube(nh, 2), "degree_cube_k6": degree_on_cube(nh, 6),
                     "charge_raw_director": float(div(Br, h)[inter].sum() * h ** 3),
                     "charge_normalized_director": float(div(Bn, h)[inter].sum() * h ** 3),
                     "flux_off_axis": off, "four_pi_cos_phi": float(4 * np.pi * np.cos(f * np.pi)), "flux_near_axis": near})
    return rows


def control_planted(n, base):
    """a Gaussian charge on a circle of radius R about z, rotating with phi, added to the static phi = 0 density."""
    rho0, h, e_ref, r_rms = base
    X = coords(n, h)
    inter = np.zeros(rho0.shape, bool); inter[2:-2, 2:-2, 2:-2] = True
    R, sig, q = 0.5, 0.4, 0.1 * e_ref
    es, ps, Qs = [], [], []
    for ph in 2 * np.pi * np.arange(NPHI) / NPHI:
        x0 = np.array([R * np.cos(ph), R * np.sin(ph), 0.0])
        g = np.exp(-((X - x0) ** 2).sum(axis=-1) / (2 * sig ** 2)); g *= q / (g.sum() * h ** 3)
        e, p, Q = moments(rho0 + g, X, inter, h)
        es.append(e); ps.append(p); Qs.append(Q)
    st = cycle_stats(es, ps, Qs, e_ref=e_ref, r_rms=r_rms)
    expect_orbit = q * R / (e_ref + q)
    Qs = np.asarray(Qs)
    q2 = harmonics(Qs[:, 0, 0] - Qs[:, 1, 1])[2]            # the amplitude of cos 2 phi in Q_xx - Q_yy
    expect_q2 = 3.0 * q * R ** 2
    ok = abs(st["orbit_radius_rms"] / expect_orbit - 1) < 0.01 and abs(q2 / expect_q2 - 1) < 0.01
    return {"n": n, "planted": {"R": R, "sigma": sig, "q_over_e": 0.1},
            "orbit_radius_measured": st["orbit_radius_rms"], "orbit_radius_expected": expect_orbit,
            "Qxx_minus_Qyy_second_harmonic_measured": q2, "expected_3qR2": expect_q2,
            "moving_fraction": st["moving_fraction"], "pass": bool(ok),
            "fails_if": "the reader does not return the planted orbit radius q R / (e + q) and the rotating quadrupole 3 q R^2 within 1 percent"}


def main():
    out = {"versions": {"python": sys.version.split()[0], "numpy": np.__version__},
           "bars": {"at_rest_orbit_radius_over_lambda": AT_REST_BAR, "author_bar_listed_not_applied": AUTHOR_BAR,
                    "casimir": 0.75, "projection": 0.25}, "rows": {}, "controls": {}}
    ns = [24, 32, 48]
    bases = {}
    for n in ns:
        for b in (0.0, eid.B_STAR):
            tag = f"tilt_n{n}_b{b:.2f}"
            res, base = read(n, b)
            out["rows"][tag] = res
            if b == 0.0:
                bases[n] = base
            f, a = res["full_interior"], res["active_mask"]
            print(f"{tag}: h {res['h']:.3f} e(phi) full min {min(f['e_by_phase']):+.3f} max {max(f['e_by_phase']):+.3f}  "
                  f"flux min {min(res['flux_of_B_through_the_interior_faces_by_phase']):+.3f} max {max(res['flux_of_B_through_the_interior_faces_by_phase']):+.3f}  "
                  f"orbit rms {f.get('orbit_radius_rms', float('nan')):.3e} axial rms {f.get('axial_rms', float('nan')):.3e}  "
                  f"moving l0/l1/l2 full {f['moving_fraction']['monopole_l0']:.3e} {f['moving_fraction']['dipole_l1']:.3e} {f['moving_fraction']['quadrupole_l2']:.3e}  "
                  f"act {a['moving_fraction']['monopole_l0']:.3e} {a['moving_fraction']['dipole_l1']:.3e} {a['moving_fraction']['quadrupole_l2']:.3e}", flush=True)
    # the reference row added at the run: the twist plane (2, 3), in which the director does not move
    for n in ns:
        res, _ = read(n, 0.0, plane=PLANES["twist"])
        out["rows"][f"twist_n{n}_b0.00"] = res
        f = res["full_interior"]
        print(f"twist_n{n}_b0.00: e(phi) full min {min(f['e_by_phase']):+.3f} max {max(f['e_by_phase']):+.3f}  "
              f"moving l0/l1/l2 full {f['moving_fraction']['monopole_l0']:.3e} {f['moving_fraction']['dipole_l1']:.3e} {f['moving_fraction']['quadrupole_l2']:.3e}", flush=True)
    out["topology_added_on_the_audit"] = {}
    for n in ns:
        out["topology_added_on_the_audit"][f"n{n}"] = topology(n)
        for r_ in out["topology_added_on_the_audit"][f"n{n}"]:
            print(f"topology n{n} phi {r_['phi_over_pi']:5.3f} pi: |n| min {r_['director_norm_min']:.3f}  degree {r_['degree_cube_k2']:+.3f} / {r_['degree_cube_k6']:+.3f}  "
                  f"charge raw {r_['charge_raw_director']:+.3f} normalized {r_['charge_normalized_director']:+.3f}  "
                  f"flux off-axis {r_['flux_off_axis']:+.3f} (4 pi cos {r_['four_pi_cos_phi']:+.3f}) near-axis {r_['flux_near_axis']:+.3f}", flush=True)
    for n in ns:
        out["controls"][f"planted_orbit_n{n}"] = control_planted(n, bases[n])
        c = out["controls"][f"planted_orbit_n{n}"]
        print(f"control planted_orbit_n{n}: {'PASS' if c['pass'] else 'FAIL'}  orbit {c['orbit_radius_measured']:.6f} vs {c['orbit_radius_expected']:.6f}  "
              f"Q2 {c['Qxx_minus_Qyy_second_harmonic_measured']:.6f} vs {c['expected_3qR2']:.6f}")
    # the lattice floor: the static phi = 0 hedgehog's traceless quadrupole over e r_rms^2 (zero by cubic symmetry)
    for n in ns:
        rho0, h, e_ref, r_rms = bases[n]
        X = coords(n, h); inter = np.zeros(rho0.shape, bool); inter[2:-2, 2:-2, 2:-2] = True
        e, p, Q = moments(rho0, X, inter, h)
        out["controls"][f"floor_n{n}"] = {"dipole_over_e_rrms": float(np.linalg.norm(p) / (abs(e) * r_rms)),
                                          "quadrupole_over_e_rrms2": float(np.linalg.norm(Q) / (abs(e) * r_rms ** 2))}
    out["controls_pass"] = bool(all(v.get("pass", True) for v in out["controls"].values()))
    out["fails_if"] = ("the centered-charge reading of M5.34 fails if the full-interior centroid orbits at a radius converged across "
                       "the three resolutions")
    if not out["controls_pass"]:
        print("controls FAILED: the reads are not labeled")
    if "--write" in sys.argv:
        with open(os.path.join(DATA, "m5_38_clock_checks.json"), "w") as f:
            json.dump(out, f, indent=1)
        print("wrote ../data/m5_38_clock_checks.json")
    if not out["controls_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
