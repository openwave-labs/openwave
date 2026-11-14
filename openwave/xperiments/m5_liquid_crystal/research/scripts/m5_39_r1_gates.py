"""M5.39 R1: the gates of the independent Card C evaluator (m5_39_cardc.py).

(a) every energy term and the gradient against the archived evaluation, on the archived seeds and on
    perturbed states, at 33^3 and 45^3: relative difference at most 1e-10;
(b) a central-difference check of our gradient in three random directions;
(c) the reading check: the energy the text prints (v3.4, the integration notes of the hadronic
    section), coefficient by coefficient, against our terms: the two node sums in their printed
    factorized form to round-off, the five derivative terms against central-difference estimates of
    the printed integrals on a smooth state at two spacings (a wrong coefficient is a ratio off one);
(d) the mirror check: the three coordinate reflections applied to states (m5_39_cardc.mirror), each
    term scored; rough states for round-off, a smooth state on three grids for the order of a defect;
(s) our seeds, refinement and frame reset against the archived ones (same arrays);
(t) the cost of one evaluation, ours against the archived one.

The archived code is untrusted input and is never imported from the filed folder: pass a hashed copy
outside the repository, and run with the interpreter isolated from the environment:

    python3 -E -s m5_39_r1_gates.py --archive <copy of frozen_packets/STRING_v0.6/sector_b_action>

Each check states the input that makes it fail: a term or gradient entry differing beyond the bound,
a digest that is not the payload manifest's.
"""

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m5_39_cardc as C  # noqa: E402

MANIFEST = {  # Card C payload manifest, the three code files the archive ships
    "bq_core.py": "80ee71bca244292cab74c383fcaa774df9e4fde34d6ccb57db67ab114328de83",
    "dn87_relax.py": "fd701f27c016c0087956f1c4c6d09cb00a2c2ca9d6fb4ed5c8fead1d4f480a12",
    "fast_quartic.py": "f529b63ccdc21c8be2d16bfb65ebac3126f5a357e13c0f4bfa618d2fbb8bb22a",
}
BOUND = 1e-10
GRIDS = [(33, 3.6), (45, 6.075)]
PARAMS = dict(delta=0.2, kx=0.25, ko=0.02)


def archived_args(grid, box, kind, **over):
    base = dict(card="C", delta=0.2, ko=0.02, kx=0.25, grid=grid, box=box, seed=kind, tetra="symmetric", stencil="edge",
                resume=None, align_start=False, reset_frame=False)
    base.update(over)
    return types.SimpleNamespace(**base)


def perturbed(state, rng, size):
    out = state.copy()
    C.interior(out)[...] += size * rng.normal(size=C.interior(out).shape)
    return out


def comm(a, b):
    return a @ b - b @ a


def reading_check():
    """(c) The printed energy against our terms. Fails if a node sum misses round-off, or if the ratio
    of a derivative term to its printed integral does not approach one along 33^3, 65^3, 129^3 (both
    sides are second-order rules, so the ratio is extrapolated from the two finest grids as such): the
    extrapolated ratio within 2 percent of one. A coefficient off by a factor would sit at that factor."""
    delta, kx, ko, box = PARAMS["delta"], PARAMS["kx"], PARAMS["ko"], 3.6
    rows, ok = [], True
    for grid in (33, 65, 129):
        state, coords, h = C.smooth_test_state(grid, box)
        card = C.CardC(grid, box, **PARAMS)
        terms, _ = card(state)
        q = state[..., :4] / np.linalg.norm(state[..., :4], axis=-1, keepdims=True)
        f = state[..., 4:].reshape(state.shape[:3] + (3, 3))
        s = np.swapaxes(f, -1, -2) @ f
        eye = np.eye(3)
        c = 0.5 * (1 - q[..., 0])[..., None, None]
        # the printed factorized forms
        x1 = s @ (s - eye) @ (s - delta * c * eye)
        x2 = s @ x1
        pot = s @ (s - c * eye) @ (s - delta * c * eye)
        r = np.trace(s, axis1=-2, axis2=-1) - (1 + delta) * c[..., 0, 0]
        e4 = 0.25 * h ** 3 * (np.sum(r * r) + np.sum(pot * pot))
        e5 = (138.0 / (4.259 * 92.4)) ** 2 * h ** 3 * np.sum(1 - q[..., 0])
        # E(q): the columns are q times i, j, k (right multiplication)
        q0, q1, q2, q3 = np.moveaxis(q, -1, 0)
        E = np.stack((np.stack((-q1, q0, q3, -q2), -1), np.stack((-q2, -q3, q0, q1), -1), np.stack((-q3, q2, -q1, q0), -1)), -1)
        a = E @ f
        N = -q[..., :, None] * q[..., None, :] + a @ np.swapaxes(a, -1, -2) / 230.0
        u = (s - s @ s) / (delta * (1 - delta))

        def grad(field):
            return np.gradient(field, h, axis=(0, 1, 2), edge_order=2)

        def norm2(field):
            return h ** 3 * sum(np.sum(d * d) for d in grad(field))

        dq, ds, du = grad(q), grad(s), grad(u)
        w4 = h ** 3 * sum(np.sum(np.sum(dq[i] ** 2, -1) * np.sum(dq[j] ** 2, -1) - np.sum(dq[i] * dq[j], -1) ** 2)
                          for i, j in ((0, 1), (0, 2), (1, 2)))
        c4 = h ** 3 * sum(np.sum(comm(ds[i], ds[j]) ** 2) for i, j in ((0, 1), (0, 2), (1, 2)))
        orient = 0.5 * h ** 3 * sum(np.sum(comm(u, d) ** 2) for d in du)
        printed = [0.25 * norm2(N), 0.5 * w4, 0.125 * c4, 0.5 * kx * (norm2(x1) + norm2(x2)), e4, e5, ko * orient]
        ratio = [float(t / p) for t, p in zip(terms, printed)]
        rows.append(dict(grid=grid, spacing=h, terms=terms.tolist(), printed_form=[float(p) for p in printed], ratio=ratio))
        print(f"{grid} reading check, ours over the printed form: " + " ".join(f"{x:.6f}" for x in ratio), flush=True)
    for k in (4, 5):
        ok &= all(abs(row["ratio"][k] - 1) <= 1e-12 for row in rows)
    extrapolated = [float(rows[2]["ratio"][k] + (rows[2]["ratio"][k] - rows[1]["ratio"][k]) / 3) for k in range(7)]
    for k in (0, 1, 2, 3, 6):
        dev = [abs(row["ratio"][k] - 1) for row in rows]
        ok &= abs(extrapolated[k] - 1) <= 0.02 and dev[2] < dev[1] < dev[0]
    print("reading check, ratios extrapolated at second order: " + " ".join(f"{x:.5f}" for x in extrapolated), flush=True)
    coefficients = [
        dict(printed="1/4 |dN|^2, N = -q q^T + a a^T / G_time, a = E(q) f, G_time = 230", term=0),
        dict(printed="1/2 W4[q], the quaternion quartic", term=1),
        dict(printed="1/8 C4[s], the record commutator quartic, s = f^T f", term=2),
        dict(printed="k_x/2 (|d x1|^2 + |d x2|^2), x1 = s (s - 1)(s - delta c), x2 = s x1, k_x = 0.25", term=3),
        dict(printed="1/4 (|tr s - (1 + delta) c|^2 + |s (s - c)(s - delta c)|^2), c = (1 - q0)/2, delta = 0.2", term=4),
        dict(printed="beta^2 (1 - q0), beta = m_pi / (e f_pi) = 138 / (4.259 x 92.4)", term=5),
        dict(printed="K_orient O[s], K_orient = 0.02; O is named and not written out in the printed line: the card's "
                     "O is 1/2 sum_i |[u, d_i u]|^2 with u = (s - s^2) / (delta (1 - delta)), read on the lattice as "
                     "(h/2) sum over edges |[u_lo, u_hi]|^2. The ratio of this term therefore tests the lattice rule against the "
                     "card's own continuum form, not against the text. The text also names W4 and C4 without their sums: the "
                     "coefficients 1/2 and 1/8 are those of sums over the pairs i < j", term=6),
    ]
    return dict(rows=rows, extrapolated_ratio=extrapolated, coefficients=coefficients, passed=bool(ok)), bool(ok)


def mirror_check(rng):
    """(d) Each term under the three reflections. Round-off is 1e-11 relative on rough states."""
    box, out, ok = 3.6, dict(rough=[], smooth=[]), True
    card = C.CardC(17, box, **PARAMS)
    for trial in range(3):
        st = C.seed(17, box, "linked")[0]
        C.interior(st)[...] += 0.05 * rng.normal(size=C.interior(st).shape)
        t0 = card(st)[0].copy()
        for axis in range(3):
            t1 = card(C.mirror(st, axis))[0]
            rel = (np.abs(t1 - t0) / np.abs(t0)).tolist()
            out["rough"].append(dict(trial=trial, axis=axis, relative_change=rel))
    for grid in (17, 33, 65, 129):
        st, _, h = C.smooth_test_state(grid, box)
        card = C.CardC(grid, box, **PARAMS)
        t0 = card(st)[0].copy()
        for axis in range(3):
            t1 = card(C.mirror(st, axis))[0]
            rel = (np.abs(t1 - t0) / np.abs(t0)).tolist()
            out["smooth"].append(dict(grid=grid, spacing=h, axis=axis, relative_change=rel, total_relative_change=float(abs(t1.sum() - t0.sum()) / t0.sum())))
    worst = np.max([r["relative_change"] for r in out["rough"]], axis=0)
    symmetric = [bool(w <= 1e-11) for w in worst]
    out["worst_rough_relative_change"] = worst.tolist()
    out["term_returns_to_round_off"] = symmetric
    orders = {}
    for k, sym in enumerate(symmetric):
        if not sym:
            d = {g: max(r["relative_change"][k] for r in out["smooth"] if r["grid"] == g) for g in (17, 33, 65, 129)}
            share = max(r["total_relative_change"] for r in out["smooth"] if r["grid"] == 129)
            orders[C.TERM_NAMES[k]] = dict(defect=d, order_17_33=float(np.log2(d[17] / d[33])), order_33_65=float(np.log2(d[33] / d[65])),
                                           order_65_129=float(np.log2(d[65] / d[129])), change_of_the_total_energy_at_129=share)
    out["defect_orders"] = orders
    out["label"] = "MIRROR SYMMETRIC" if all(symmetric) else "MIRROR DEFECT IN: " + ", ".join(orders)
    print("mirror check, worst relative change per term on rough states: " + " ".join(f"{w:.1e}" for w in worst), flush=True)
    for name, o in orders.items():
        print(f"  {name}: defect {o['defect']}, orders {o['order_17_33']:.2f} {o['order_33_65']:.2f} {o['order_65_129']:.2f}", flush=True)
    # the check fails if the map is wrong: every term but the record quartic must return to round-off
    ok &= all(symmetric[k] for k in (0, 1, 3, 4, 5, 6))
    return out, bool(ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "m5_39_r1_gates.json")
    ap.add_argument("--tmp", type=Path, required=True, help="a scratch folder for the refinement check")
    a = ap.parse_args()

    digests = {name: hashlib.sha256((a.archive / name).read_bytes()).hexdigest() for name in MANIFEST}
    digest_ok = digests == MANIFEST
    sys.path.insert(0, str(a.archive))
    import bq_core as B  # noqa: E402

    rng = np.random.default_rng(3939)
    rows, seeds, fd_rows, cost = [], [], [], []
    ok = digest_ok
    for grid, box in GRIDS:
        card = C.CardC(grid, box, **PARAMS)
        for kind in ("linked", "smooth"):
            base, h, coords = B.seed(archived_args(grid, box, kind))
            ours, coords_o, h_o = C.seed(grid, box, kind)
            d = float(np.max(np.abs(ours - base)))
            seeds.append(dict(grid=grid, kind=kind, max_abs_difference=d, spacing_difference=abs(h - h_o)))
            ok &= d <= 1e-14
            states = [("seed", base)]
            if kind == "linked":
                states += [(f"perturbed {s:g}", perturbed(base, rng, s)) for s in (1e-3, 1e-2, 5e-2)]
            for label, st in states:
                x = C.interior(st).ravel()
                t0 = time.perf_counter()
                terms_a, grad_a = B.evaluate(x, base, h, 0.2, 0.25, 0.02, "edge", "symmetric", "C")
                t_a = time.perf_counter() - t0
                t0 = time.perf_counter()
                terms_o, grad_full = card(st)
                t_o = time.perf_counter() - t0
                grad_o = C.interior(grad_full).ravel()
                term_rel = (np.abs(terms_o - terms_a) / np.maximum(np.abs(terms_a), 1e-300)).tolist()  # 0 against 0 passes
                grad_rel = float(np.max(np.abs(grad_o - grad_a)) / np.max(np.abs(grad_a)))
                # the two blocks of the gradient separately, each against its own scale
                ga, go = grad_a.reshape(-1, 13), grad_o.reshape(-1, 13)
                grad_q = float(np.max(np.abs(go[:, :4] - ga[:, :4])) / np.max(np.abs(ga[:, :4])))
                grad_f = float(np.max(np.abs(go[:, 4:] - ga[:, 4:])) / np.max(np.abs(ga[:, 4:])))
                passed = max(term_rel) <= BOUND and max(grad_rel, grad_q, grad_f) <= BOUND
                ok &= passed
                rows.append(dict(grid=grid, seed=kind, state=label, terms_archived=terms_a.tolist(), terms_ours=terms_o.tolist(),
                                 term_relative_difference=term_rel, gradient_relative_difference=grad_rel,
                                 gradient_relative_difference_winding=grad_q, gradient_relative_difference_factor=grad_f,
                                 passed=bool(passed)))
                cost.append(dict(grid=grid, state=label, archived_s=t_a, ours_s=t_o))
                print(f"{grid} {kind:7s} {label:15s} terms {max(term_rel):.1e} grad {grad_rel:.1e} (q {grad_q:.1e}, f {grad_f:.1e}) "
                      f"archived {t_a:.2f}s ours {t_o:.2f}s {'PASS' if passed else 'FAIL'}", flush=True)
        # (b) central differences on a perturbed state, where no term vanishes
        st = perturbed(C.seed(grid, box, "linked")[0], rng, 1e-2)
        _, g = card(st)
        g = C.interior(g).copy()
        for _ in range(3):
            d = rng.normal(size=g.shape)
            d /= np.linalg.norm(d)
            eps = 2e-5
            sp, sm = st.copy(), st.copy()
            C.interior(sp)[...] += eps * d
            C.interior(sm)[...] -= eps * d
            fd = (card(sp)[0].sum() - card(sm)[0].sum()) / (2 * eps)
            an = float(np.sum(d * g))
            err = abs(fd - an)
            fd_rows.append(dict(grid=grid, finite_difference=float(fd), analytic=an, difference=float(err)))
            ok &= err <= 3e-7  # the archived driver's own bound on the same check
            print(f"{grid} central difference {fd:.10f} analytic {an:.10f} diff {err:.1e}", flush=True)
        # (s) refinement and frame reset against the archived resume path
        coarse = grid // 2 + 1 if (grid // 2) % 2 == 0 else grid // 2
        cst = perturbed(C.seed(coarse, box, "linked")[0], rng, 1e-2)
        ccoords, _ = C.coordinates(coarse, box)
        a.tmp.mkdir(parents=True, exist_ok=True)
        np.savez(a.tmp / "coarse.npz", state=cst, coords=ccoords)
        for reset in (False, True):
            base, h, coords = B.seed(archived_args(grid, box, "linked", resume=a.tmp / "coarse.npz", reset_frame=reset))
            ours, _, _ = C.refine(cst, ccoords, grid, box)
            if reset:
                ours = C.reset_frame(ours)
            d = float(np.max(np.abs(ours - base)))
            seeds.append(dict(grid=grid, kind=f"refined from {coarse}" + (" with the frame reset" if reset else ""), max_abs_difference=d))
            ok &= d <= 1e-14
            print(f"{grid} refine from {coarse} reset {reset}: {d:.1e}", flush=True)

    reading, reading_ok = reading_check()
    mirror, mirror_ok = mirror_check(rng)
    ok &= reading_ok and mirror_ok
    out = dict(task="M5.39 R1", bound=BOUND, digests=digests, digests_match_manifest=digest_ok, comparisons=rows,
               seeds=seeds, central_differences=fd_rows, reading_check=reading, mirror_check=mirror, cost=cost,
               all_passed=bool(ok))
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print("ALL PASSED" if ok else "FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
