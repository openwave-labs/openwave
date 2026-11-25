"""M5.42 R1: the controls of the evaluator (m5_42_eq42.py), the band term, the chart and gates 1 and 2.

  a  the forward kernel against a plain NumPy reference written from the docstring of the author's script,
     on the author's two deposited fields: the stored inertia density cell by cell, the terms, the gradient
  b  the printed numbers of the author's box: I = 511 (22 inside r < 2) and 2 J Omega / E = 0.83
  c  the mirror-averaged kernel against the plain mean of the reference over the eight reflections
  d  the gradient of E_J against central differences (three random directions; J = 0 and J_e/10; band on)
  e  the band term: zero on the author's two fields; its gradient against differences on a field past both edges
  f  the chart: the minor axis of the frame near the +z axis from eight azimuths, and at the axis node
  g  gate 1, the rank-one regression: on M = R(theta) D R(theta)^T, theta = a x + b y, the forward quartic
     density is 8 kappa (1 - delta)^4 a^2 b^2 (b - a)^2 h^2 at leading order
  h  gate 2, the mirror test of the two stencils on a field and its mirror images
  t  the time per evaluation of the two stencils on four grids

Each control prints the input that would make it fail and writes its numbers to the JSON given by --out.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_42_eq42 as Q  # noqa: E402

PAIRS = ((0, 1), (0, 2), (1, 2))
SL = np.s_[:-1, :-1, :-1]


def comm(A, B):
    return A @ B - B @ A


def reference_forward(M, G, h, L1, kap, v, ck):
    """The forward functional in plain NumPy. Returns the terms, the inertia, the inertia density per cell
    and the gradients of U and of I over the nine entries of every node."""
    d = [((np.roll(M, -1, ax) - M) / h)[SL] for ax in range(3)]
    Mc, Gc = M[SL], G[SL]
    e2 = sum((di * di).sum((-1, -2)) for di in d)
    C = {p: comm(d[p[0]], d[p[1]]) for p in PAIRS}
    e4 = sum((C[p] * C[p]).sum((-1, -2)) for p in PAIRS)
    M2 = Mc @ Mc
    M3 = M2 @ Mc
    tr = lambda A: np.trace(A, axis1=-2, axis2=-1)  # noqa: E731
    t = [tr(Mc) - ck[0], tr(M2) - ck[1], tr(M3) - ck[2]]
    eV = t[0] ** 2 + t[1] ** 2 + t[2] ** 2
    K = comm(Gc, Mc)
    Qs = [comm(K, di) for di in d]
    idn = 2 * (L1 * (K * K).sum((-1, -2)) + kap * sum((q * q).sum((-1, -2)) for q in Qs))
    h3 = h ** 3
    eye = np.eye(3)
    gMU = h3 * v * (2 * t[0][..., None, None] * eye + 4 * t[1][..., None, None] * Mc + 6 * t[2][..., None, None] * M2)
    PK = 4 * L1 * K + 4 * kap * sum(comm(Qs[i], d[i]) for i in range(3))
    gMI = h3 * comm(PK, Gc)
    gU = np.zeros_like(M)
    gI = np.zeros_like(M)
    gU[SL] += gMU
    gI[SL] += gMI
    for i in range(3):
        gu = h3 * 2 * L1 * d[i]
        for (a, b) in PAIRS:
            if a == i:
                gu = gu + h3 * kap * 2 * comm(C[(a, b)], d[b])
            elif b == i:
                gu = gu + h3 * kap * 2 * comm(d[a], C[(a, b)])
        gi = h3 * 4 * kap * comm(K, Qs[i])
        idx = [slice(0, -1)] * 3
        idx[i] = slice(1, None)
        for g, gd in ((gU, gu), (gI, gi)):
            g[SL] -= gd / h
            g[tuple(idx)] += gd / h
    return dict(U2=h3 * L1 * e2.sum(), U4=h3 * kap * e4.sum(), UV=h3 * v * eV.sum(), I=h3 * idn.sum(), idn=idn, gU=gU, gI=gI)


def reference_averaged(M, G, h, L1, kap, v, ck):
    """The plain mean of the forward reference over the eight reflections of the grid axes."""
    acc = None
    for sx in (0, 1):
        for sy in (0, 1):
            for sz in (0, 1):
                axes = tuple(ax for ax, s in enumerate((sx, sy, sz)) if s == 0)
                f = (lambda A: np.flip(A, axes)) if axes else (lambda A: A)
                p = reference_forward(np.ascontiguousarray(f(M)), np.ascontiguousarray(f(G)), h, L1, kap, v, ck)
                p["gU"], p["gI"] = f(p["gU"]), f(p["gI"])
                p.pop("idn")
                if acc is None:
                    acc = {k: val / 8 for k, val in p.items()}
                else:
                    for k, val in p.items():
                        acc[k] = acc[k] + val / 8
    return acc


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(np.max(np.abs(np.asarray(b))), 1e-300))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--author-dir", type=Path, required=True, help="the folder holding the author's two deposited fields")
    ap.add_argument("--out", type=Path, default=HERE.parent / "data" / "m5_42_r1_controls.json")
    ap.add_argument("--timing-grids", type=int, nargs="*", default=[33, 49, 65, 97])
    a = ap.parse_args()
    out, checks = {}, []

    def check(name, value, bound, fails_if):
        met = bool(value <= bound)
        checks.append(dict(check=name, value=float(value), bound=bound, met=met, fails_if=fails_if))
        print(f"{'MET   ' if met else 'FAILED'} {name}: {value:.3e} (bound {bound:.0e})", flush=True)

    delta, L1 = 0.3, 0.3
    Je = Q.j_electron(delta)
    dep = Q.Model(33, 6.0, delta, L1, stencil="forward", fixed_chart=False)
    fields = {}
    for tag, J in (("J0", 0.0), ("J168", Je / 10)):
        z = np.load(a.author_dir / f"field_L1_0.3_N33_{tag}.npz")
        fields[tag] = (z["m6"].astype(float), z["idens"].astype(float), J)

    # a: the forward kernel against the reference and the stored density
    for tag, (m6, idens, J) in fields.items():
        M = np.where(dep.mask[..., None, None], Q.from6(m6), dep.seed)
        m6f = Q.to6(M)
        ref = reference_forward(M, dep.G, dep.h, L1, 1.0, 1.0, dep.ck)
        p = dep.evaluate(m6f, J)
        check(f"a {tag}: stored inertia density, cell by cell", np.max(np.abs(dep.idn[SL] - idens)) / np.max(np.abs(idens)), 1e-12,
              "a kernel term or weight differing from the author's script")
        for k in ("U2", "U4", "UV", "I"):
            check(f"a {tag}: {k} against the reference", abs(p[k] - ref[k]) / abs(ref[k]), 1e-12, "a term of the kernel off")
        # the field is stationary, so its gradient is a cancellation: the difference is measured against
        # the size of the terms that cancel (the gradients of U and of I taken alone)
        g = ref["gU"] if J == 0 else ref["gU"] - J * J / (2 * ref["I"] ** 2) * ref["gI"]
        scale = np.abs(Q.grad_to6(ref["gU"])).max()
        check(f"a {tag}: gradient against the reference, over the size of its terms",
              np.abs(p["grad"] - Q.grad_to6(g) * dep.mask[..., None]).max() / scale, 1e-12, "a gradient term of the kernel off")
        check(f"a {tag}: dI/dM against the reference", rel(Q.grad_to6(dep.gI), Q.grad_to6(ref["gI"])), 1e-12, "an inertia gradient term off")
        r_in = float(dep.h ** 3 * dep.idn[dep.r < 2].sum())
        ratio = 2 * J * J / p["I"] / p["E"] if J else None
        out[f"author_{tag}"] = dict(U2=p["U2"], U4=p["U4"], UV=p["UV"], U=p["U"], I=p["I"], E=p["E"], inertia_inside_r2=r_in, two_J_Omega_over_E=ratio,
                                    gmax_over_h3=float(np.abs(p["grad"]).max() / dep.h ** 3))
    # b: the printed numbers
    check("b: I of the J = 0 field prints as 511", abs(round(out["author_J0"]["I"]) - 511), 0, "another inertia")
    check("b: inertia inside r < 2 prints as 22", abs(round(out["author_J0"]["inertia_inside_r2"]) - 22), 0, "another core share")
    check("b: 2 J Omega / E at J_e/10 prints as 0.83", abs(round(out["author_J168"]["two_J_Omega_over_E"], 2) - 0.83), 1e-9, "another ratio")

    # c: the averaged kernel against the mean over the eight reflections
    rng = np.random.default_rng(42)
    avg = Q.Model(33, 6.0, delta, L1, stencil="averaged", fixed_chart=True)
    small = Q.Model(13, 6.0, delta, L1, stencil="averaged", fixed_chart=True, free_faces=True)
    cases = [("random 13^3", small, Q.to6(small.seed) + 0.3 * rng.standard_normal(small.shape6))]
    for tag, (m6, _, _) in fields.items():
        cases.append((f"author {tag}", avg, Q.to6(np.where(avg.mask[..., None, None], Q.from6(m6), avg.seed))))
    for name, mod, m6 in cases:
        J = Je / 10
        M = Q.from6(m6)
        ref = reference_averaged(M, mod.G, mod.h, L1, 1.0, 1.0, mod.ck)
        p = mod.evaluate(m6, J)
        for k in ("U2", "U4", "UV", "I"):
            check(f"c {name}: {k} against the eight reflections", abs(p[k] - ref[k]) / abs(ref[k]), 1e-12, "a weight of the averaged stencil off")
        g = Q.grad_to6(ref["gU"] - J * J / (2 * ref["I"] ** 2) * ref["gI"]) * mod.mask[..., None]
        check(f"c {name}: gradient against the eight reflections", rel(p["grad"], g), 1e-11, "a gradient weight of the averaged stencil off")
    # the node weights of the nodal terms
    ones = Q.Model(5, 1.0, delta, 0.0, kappa=0.0, v=1.0, stencil="averaged", free_faces=True)
    m6 = np.zeros(ones.shape6)  # M = 0: the potential density is c1^2 + c2^2 + c3^2 at every node
    p = ones.evaluate(m6, 0.0, need_grad=False)
    dens = sum(c * c for c in ones.ck)
    expect = ones.h ** 3 * dens * (3 * 3 * 3 + 6 * 9 * 0.5 + 12 * 3 * 0.25 + 8 * 0.125)
    check("c: nodal weights (interior 1, face 1/2, edge 1/4, corner 1/8) on a uniform field", abs(p["UV"] - expect) / expect, 1e-13, "other node weights")

    # d: the gradient against central differences
    band = dict(w=1e6, lo=-0.5, hi=1.5)
    for name, mod in (("averaged", avg), ("forward", dep), ("averaged with band", Q.Model(33, 6.0, delta, L1, band=band))):
        base = Q.to6(np.where(mod.mask[..., None, None], Q.from6(fields["J168"][0]), mod.seed))
        base = base + 0.05 * rng.standard_normal(base.shape) * mod.mask[..., None]
        if "band" in name:
            base[16, 16, 20] += np.array([2.5, 0.3, 0.0, -2.0, 0.2, 0.1])  # eigenvalues past both edges
        for J in (0.0, Je / 10):
            g = mod.evaluate(base, J)["grad"]
            worst = 0.0
            for _ in range(3):
                u = rng.standard_normal(base.shape) * mod.mask[..., None]
                u /= np.linalg.norm(u)
                eps = 1e-3  # fourth-order central differences
                f = lambda s: mod.evaluate(base + s * eps * u, J, need_grad=False)["E"]  # noqa: E731
                fd = (8 * (f(1) - f(-1)) - (f(2) - f(-2))) / (12 * eps)
                worst = max(worst, abs(fd - (g * u).sum()) / abs(fd))
            check(f"d {name}, J = {J:.1f}: gradient against central differences", worst, 1e-6, "an analytic gradient term off")

    # e: the band term is zero on the author's fields
    bm = Q.Model(33, 6.0, delta, L1, stencil="forward", fixed_chart=False, band=band)
    for tag, (m6, _, J) in fields.items():
        p = bm.evaluate(Q.to6(np.where(bm.mask[..., None, None], Q.from6(m6), bm.seed)), J, need_grad=False)
        check(f"e {tag}: band energy on the author's field", p["UB"], 0.0, "an eigenvalue of the field outside [-0.5, 1.5]")
    lam = np.linalg.eigvalsh(Q.from6(fields["J168"][0]))
    out["author_eigenvalue_range"] = [float(lam.min()), float(lam.max())]

    # f: the chart near the +z axis
    worst, node = 0.0, None
    for rho in (1e-2, 1e-3, 1e-4):
        for az in np.arange(8) * np.pi / 4 + 0.1:
            X, Y, Z = np.array([rho * np.cos(az)]), np.array([rho * np.sin(az)]), np.array([3.0])
            Mv, n, _ = Q.vacuum_frame(X, Y, Z, 1.0, False)
            minor = Mv[0] - np.outer(n[0], n[0])
            if rho == 1e-4:
                worst = max(worst, float(np.abs(minor - np.diag([1.0, 0, 0])).max()))
    for fixed in (False, True):
        Mv, n, _ = Q.vacuum_frame(np.array([0.0]), np.array([0.0]), np.array([3.0]), 1.0, fixed)
        minor = Mv[0] - np.outer(n[0], n[0])
        out[f"minor_axis_at_plus_z_node_fixed_{fixed}"] = np.round(minor, 12).tolist()
        if fixed:
            node = float(np.abs(minor - np.diag([1.0, 0, 0])).max())
    check("f: the frame's minor axis at rho = 1e-4 from the +z axis is the x direction, from eight azimuths", worst, 1e-3, "a frame not regular at the north pole, or another limit")
    check("f: the fixed chart sets the axis node to that limit", node, 1e-12, "the fix not applied")
    dM = np.abs(Q.Model(33, 6.0, delta, L1, fixed_chart=True).frame - Q.Model(33, 6.0, delta, L1, fixed_chart=False).frame).max((-1, -2))
    out["nodes_changed_by_the_fixed_chart"] = int((dM > 1e-12).sum())

    # g: gate 1, the rank-one regression on the forward stencil (quartic alone)
    a_, b_ = 0.7, 0.4
    coeff = 8 * (1 - delta) ** 4 * a_ ** 2 * b_ ** 2 * (b_ - a_) ** 2
    rows = []
    for N in (17, 33, 65, 129):
        mod = Q.Model(N, 2.0, delta, 0.0, kappa=1.0, v=0.0, stencil="forward", free_faces=True)
        X, Y, _ = np.meshgrid(mod.x, mod.x, mod.x, indexing="ij")
        th = a_ * X + b_ * Y
        R = np.zeros(X.shape + (3, 3))
        R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1], R[..., 2, 2] = np.cos(th), -np.sin(th), np.sin(th), np.cos(th), 1.0
        M = np.einsum("...ak,k,...bk->...ab", R, np.array([1.0, delta, 0.0]), R)
        p = mod.evaluate(Q.to6(M), 0.0, need_grad=False, need_I=False)
        vol = (2 * 2.0) ** 3 * ((N - 1) / (N - 1))
        rows.append(dict(N=N, h=mod.h, density=p["U4"] / vol, density_over_h2=p["U4"] / vol / mod.h ** 2))
    order = float(np.log(rows[-2]["density"] / rows[-1]["density"]) / np.log(rows[-2]["h"] / rows[-1]["h"]))
    out["gate1"] = dict(a=a_, b=b_, derived_coefficient=coeff, rows=rows, order_two_finest=order)
    check("g: gate 1, the h^2 coefficient of the forward quartic on a rank-one field (finest spacing)", abs(rows[-1]["density_over_h2"] - coeff) / coeff, 0.02, "a quartic that sees a rank-one gradient at another order or size")
    check("g: gate 1, the order between the two finest spacings is 2", abs(order - 2.0), 0.1, "another order")

    # h: gate 2, the mirror test of the stencils
    base = Q.to6(avg.seed) + 0.2 * rng.standard_normal(avg.shape6) * avg.mask[..., None]
    out["gate2"] = {}
    for name in ("forward", "averaged"):
        worst = dict(U4=0.0, I=0.0, U=0.0)
        mod = Q.Model(33, 6.0, delta, L1, stencil=name, fixed_chart=True, free_faces=True)
        p0 = mod.evaluate(base, 0.0, need_grad=False)
        for ax in range(3):
            # the mirror image of the field in the plane normal to ax: M'(x) = R M(R x) R, G'(x) = -R G(R x) R
            Rm = np.eye(3)
            Rm[ax, ax] = -1.0
            Mm = np.einsum("ab,...bc,cd->...ad", Rm, np.flip(Q.from6(base), ax), Rm)
            mir = Q.Model(33, 6.0, delta, L1, stencil=name, fixed_chart=True, free_faces=True)
            mir.G = np.ascontiguousarray(-np.einsum("ab,...bc,cd->...ad", Rm, np.flip(mod.G, ax), Rm))
            p1 = mir.evaluate(Q.to6(Mm), 0.0, need_grad=False)
            for k in worst:
                worst[k] = max(worst[k], abs(p1[k] - p0[k]) / abs(p0[k]))
        out["gate2"][name] = worst
    check("h: gate 2, the averaged stencil under the three mirrors (energy, quartic, inertia)", max(out["gate2"]["averaged"].values()), 1e-12, "a stencil that is not mirror symmetric")
    out["gate2"]["note"] = "the forward stencil is expected to fail this test; its defect is the number printed here"
    print("forward stencil mirror defect:", out["gate2"]["forward"], flush=True)

    # t: time per evaluation
    out["timing"] = []
    for N in a.timing_grids:
        for name in ("forward", "averaged"):
            mod = Q.Model(N, 6.0 * (N - 1) / 32, delta, L1, stencil=name)
            m6 = mod.seed6()
            mod.evaluate(m6, Je / 10)
            t0 = time.time()
            reps = 3 if N <= 65 else 1
            for _ in range(reps):
                mod.evaluate(m6, Je / 10)
            dt = (time.time() - t0) / reps
            out["timing"].append(dict(N=N, stencil=name, seconds_per_evaluation=dt, microseconds_per_node=1e6 * dt / N ** 3))
            print(f"timing N {N} {name}: {dt:.3f} s per evaluation, {1e6 * dt / N ** 3:.2f} us per node", flush=True)

    out["checks"] = checks
    out["all_met"] = bool(all(c["met"] for c in checks))
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print("ALL MET" if out["all_met"] else "SOME FAILED", "->", a.out)
    sys.exit(0 if out["all_met"] else 1)


if __name__ == "__main__":
    main()
