"""M5.42: gates 3, 4 and 5 of the author's five-part gate, on the states of half-box 6 at L1 = 0.3.

  gate 2  on states: the change of U, of the quartic and of I under the three mirrors, on the stencil of the state
  gate 3  two discretizations: U, I and 2 J Omega / E of the forward chains (g3_*) beside the mirror-averaged
          ones (r3_*_L0.3) at J = 0, J_e/10 and J_e on spacings 0.375, 0.25 and 0.1875, with their labels
  gate 4  the trace potential summed at the nodes (as the stencil of the state sums it) against the potential
          of the field interpolated linearly inside each cell, by 2 x 2 x 2 Gauss quadrature; the author's
          bound is a difference of at most 0.01 E
  gate 5  the lowest eigenvalue of the Hessian of E_J on the free unknowns, matrix-free on central
          differences of the gradient (Lanczos, smallest algebraic), for the states at spacing 0.375 and for
          the states of the author's box (r2_g33)

Gate 1 and gate 2 on a random field are in m5_42_controls.py. The states are read from the chain records and the saved end fields.
"""

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh, lobpcg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_42_eq42 as Q  # noqa: E402
from m5_42_relax import chains  # noqa: E402

GP = 0.5 + np.array([-1.0, 1.0]) / (2 * np.sqrt(3.0))  # Gauss points on [0, 1]


def model_of(spec):
    return Q.Model(spec["N"], spec["half_box"], spec["delta"], spec["L1"], stencil=spec["stencil"], fixed_chart=spec["fixed_chart"],
                   free_faces=spec["free_faces"], band=spec["band"])


def potential_by_quadrature(mod, m6):
    """h^3 sum over cells of the mean over eight Gauss points of v sum_k (tr M^k - c_k)^2, M trilinear."""
    M = Q.from6(m6.reshape(mod.shape6))
    N = mod.N
    total = 0.0
    for a in GP:
        for b in GP:
            for c in GP:
                acc = 0.0
                for da, wa in ((0, 1 - a), (1, a)):
                    for db, wb in ((0, 1 - b), (1, b)):
                        for dc, wc in ((0, 1 - c), (1, c)):
                            acc = acc + wa * wb * wc * M[da:N - 1 + da, db:N - 1 + db, dc:N - 1 + dc]
                M2 = acc @ acc
                tr = lambda A: np.trace(A, axis1=-2, axis2=-1)  # noqa: E731
                t = [tr(acc) - mod.ck[0], tr(M2) - mod.ck[1], tr(M2 @ acc) - mod.ck[2]]
                total += (t[0] ** 2 + t[1] ** 2 + t[2] ** 2).sum() / 8
    return float(mod.h ** 3 * mod.v * total)


def potential_at_cell_means(mod, m6):
    """The same sum with one point per cell, the mean of its eight corners (the measure of the plan's pilot)."""
    M = Q.from6(m6.reshape(mod.shape6))
    N = mod.N
    acc = sum(M[da:N - 1 + da, db:N - 1 + db, dc:N - 1 + dc] for da in (0, 1) for db in (0, 1) for dc in (0, 1)) / 8
    M2 = acc @ acc
    tr = lambda A: np.trace(A, axis1=-2, axis2=-1)  # noqa: E731
    t = [tr(acc) - mod.ck[0], tr(M2) - mod.ck[1], tr(M2 @ acc) - mod.ck[2]]
    return float(mod.h ** 3 * mod.v * (t[0] ** 2 + t[1] ** 2 + t[2] ** 2).sum())


def gate4(job):
    chain, path, spec, f = job
    z = np.load(f)
    m6, J = z["m6"], float(z["J"])
    mod = model_of(spec)
    p = mod.evaluate(m6, J, need_grad=False)
    quad = potential_by_quadrature(mod, m6)
    diff = abs(quad - p["UV"])
    mean = potential_at_cell_means(mod, m6)
    return dict(potential_at_cell_means=mean, difference_at_cell_means=abs(mean - p["UV"]), met_at_cell_means=bool(abs(mean - p["UV"]) <= 0.01 * p["E"]),
                chain=chain, path=path, N=spec["N"], stencil=spec["stencil"], J=J, E=p["E"], potential_at_nodes=p["UV"], potential_by_quadrature=quad,
                difference=diff, bound=0.01 * p["E"], met=bool(diff <= 0.01 * p["E"]))


def gate2(job):
    """The mirror test on a state: the relative change of U, of the quartic and of I under the three mirrors,
    M'(x) = R M(R x) R, on the stencil of the state."""
    chain, path, spec, f, label = job
    z = np.load(f)
    m6, J = z["m6"], float(z["J"])
    mod = model_of(spec)
    p0 = mod.evaluate(m6, J, need_grad=False)
    worst = dict(U=0.0, U4=0.0, I=0.0)
    by_mirror = {}
    for ax, name in enumerate("xyz"):
        Rm = np.eye(3)
        Rm[ax, ax] = -1.0
        Mm = np.einsum("ab,...bc,cd->...ad", Rm, np.flip(Q.from6(m6), ax), Rm)
        p1 = mod.evaluate(Q.to6(Mm), J, need_grad=False)
        by_mirror[name] = {k: abs(p1[k] - p0[k]) / abs(p0[k]) for k in worst}
        for k in worst:
            worst[k] = max(worst[k], by_mirror[name][k])
    return dict(chain=chain, path=path, N=spec["N"], stencil=spec["stencil"], J=J, label=label, largest_relative_change=worst, by_mirror=by_mirror,
                met=bool(max(worst.values()) <= 1e-10))


def gate5(job):
    chain, path, spec, f, label = job
    z = np.load(f)
    x, J = z["m6"].reshape(-1).astype(float), float(z["J"])
    mod = model_of(spec)
    free = np.repeat(mod.mask.reshape(-1), 6)
    n = int(free.sum())
    eps = 1e-3
    count = dict(n=0)

    def matvec(v):
        count["n"] += 1
        u = np.zeros_like(x)
        u[free] = v
        gp = mod.evaluate(x + eps * u, J)["grad"].reshape(-1)
        gm = mod.evaluate(x - eps * u, J)["grad"].reshape(-1)
        return ((gp - gm) / (2 * eps))[free]

    op = LinearOperator((n, n), matvec=matvec, dtype=float)
    converged = True
    residual = None
    try:
        lam, vec = eigsh(op, k=1, which="SA", tol=1e-3, ncv=40, maxiter=150)
        lam, vec = float(lam[0]), vec[:, 0]
    except ArpackNoConvergence as e:
        converged = False
        if len(e.eigenvalues):
            lam, vec = float(e.eigenvalues[0]), e.eigenvectors[:, 0]
        else:
            # no Ritz value converged: a block descent from a random vector gives a Rayleigh quotient, which is an
            # upper bound of the lowest eigenvalue (a negative value shows a descent direction; a positive one shows nothing)
            warnings.simplefilter("ignore")
            rng = np.random.default_rng(5)
            X0 = rng.standard_normal((n, 1))
            vals, vecs, hist = lobpcg(op, X0, largest=False, maxiter=200, tol=1e-6, retResidualNormsHistory=True)
            lam, vec = float(vals[0]), vecs[:, 0]
            residual = float(np.atleast_1d(hist[-1])[0])
    out = dict(chain=chain, path=path, N=spec["N"], stencil=spec["stencil"], J=J, label=label, lowest_eigenvalue=lam, lanczos_converged=converged,
               matvecs=count["n"], unknowns=n, upper_bound_only=bool(not converged), residual_norm=residual, lowest_over_h3=(lam / mod.h ** 3 if lam is not None else None))
    if vec is not None:
        full = np.zeros_like(x)
        full[free] = vec
        w = (full.reshape(mod.shape6) ** 2).sum(-1)
        at = np.unravel_index(int(np.argmax(w)), w.shape)
        out.update(mode_largest_node_share=float(w[at] / w.sum()), mode_largest_position=[float(mod.x[q]) for q in at])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=HERE.parent / "data")
    ap.add_argument("--chains-dir", type=Path, default=HERE.parent / "checkpoints" / "m5_42" / "chains")
    ap.add_argument("--out", type=Path, default=HERE.parent / "data" / "m5_42_r1_gates.json")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    C = chains()

    def paths(name):
        f = a.chains_dir / f"{name}.json"
        return [p for p in json.loads(f.read_text())["paths"] if not p.get("skipped")] if f.exists() else []

    g3 = []
    j4, j5 = [], []
    for N in (33, 49, 65):
        for name in (f"g3_g{N}", f"r3_g{N}_L0.3"):
            for p in paths(name):
                if p["name"] not in ("j0", "j10", "j1"):
                    continue
                last = p["stops"][-1]
                g3.append(dict(chain=name, path=p["name"], N=N, stencil=C[name]["model"]["stencil"], U=last["U"], I=last["I"],
                               two_J_Omega_over_E=last["two_J_Omega_over_E"], label=p["label"], converged=p["converged"], iterations=p["iterations"]))
                j4.append((name, p["name"], C[name]["model"], str(a.data_dir / p["end_file"])))
                if N == 33:
                    j5.append((name, p["name"], C[name]["model"], str(a.data_dir / p["end_file"]), p["label"]))
    for p in paths("r2_g33"):
        j4.append(("r2_g33", p["name"], C["r2_g33"]["model"], str(a.data_dir / p["end_file"])))
        j5.append(("r2_g33", p["name"], C["r2_g33"]["model"], str(a.data_dir / p["end_file"]), p["label"]))
    with ProcessPoolExecutor(a.workers) as ex:
        r4 = list(ex.map(gate4, j4))
        r2 = list(ex.map(gate2, j5))
        r5 = list(ex.map(gate5, j5))
    a.out.write_text(json.dumps(dict(gate2_on_states=r2, gate3=g3, gate4=r4, gate5=r5), indent=1) + "\n")
    for r in r2:
        print("gate 2", r["chain"], r["path"], r["stencil"], {k: f"{v:.2e}" for k, v in r["largest_relative_change"].items()}, "MET" if r["met"] else "FAILED")
    for r in r4:
        print("gate 4", r["chain"], r["path"], f"quadrature {r['difference']:.4f}, cell means {r['difference_at_cell_means']:.4f}, bound {r['bound']:.4f}",
              "MET" if r["met"] else "FAILED", "/", "MET" if r["met_at_cell_means"] else "FAILED")
    for r in r5:
        print("gate 5", r["chain"], r["path"], r["label"], r["lowest_eigenvalue"], "converged" if r["lanczos_converged"] else "NOT CONVERGED", r["matvecs"])
    print("->", a.out)


if __name__ == "__main__":
    main()
