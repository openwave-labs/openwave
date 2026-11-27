"""M5.57: one chain (or one K5 re-relaxation) per process, single thread: the rows of rungs A, B and C, the controls
C5 to C8 that are chains, the releases, and the status reads (CONVERGED, the second step, K5, in hold, usable) that
the local launcher and m5_57_reads.py use.

Each chain is a descent of E (U, or E_mu = U + mu/2 (D' - d)^2 under a hold) over the six components of the inner
nodes, the faces held at the far field (plan section 1):

  L-BFGS-B, maxcor 10, ftol 0, gtol 1e-7 on the gradient with its factor h^3, maxfun 1.25 times the cap (a total)
  the gate G = max |dE/dm| / h^3 over the free nodes; free chains G <= 1e-3; held chains
      G <= max(1e-5, min(1e-3, 0.1 F)), F = max |f dD'/dm| / h^3; every chain the minor torque <= 1e-5
  CONVERGED  the gate met and (a) the end by gtol, or (b) two major stops >= 5000 iterations apart within 1e-8 relative
             in U, 0.002 in s (every R_L on one ring; not used on a ONE OBJECT end), 1e-4 in D' and mu_free <= 0.003,
             or (c) an end by the relative-reduction test that repeats after its restarts, G <= 1e-5, second step met
  restarts   an abnormal line-search end, an end by the relative-reduction test: from the last iterate with a fresh
             memory, three at most
  caps       30000 iterations (references, rung A), 10000 (a rung, a release, a K5); wall 24 h on 121^3, 12 h on 97^3,
             6 h on 73^3 and 49^3, counted across resumes
  stops      route 0, 10, 20, 50, 100, 200, 300, 500, 700, 1000, then every 500, and the end; major 1000, 3000, 10000,
             20000, 30000 and the end (the two-ring reader, D' with its regime, D with its regime; on 49^3 M5.51's
             read_state)
  noise      1e-4 normal draws on each free component, default_rng(57000 + n), n the row's index below; continuation
             rungs, refinement starts and releases none; K5 seeds 57100 + n

Usage:  python m5_57_relax.py --row N | --k5 N | --list | --spec N
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_51_action as A  # noqa: E402
import m5_51_reads as RD  # noqa: E402
import m5_56_turn as T  # noqa: E402
import m5_57_hold as H  # noqa: E402

CK = H.CK
WORK = Path(os.environ.get("M557_WORK", str(CK / "chains")))
DECISIONS = CK / "decisions.json"
MAJOR = (1000, 3000, 10000, 20000, 30000)
WALL = dict(H1=6.0, P=6.0, P375=12.0, A24=12.0, A30=24.0)
INWARD = [24, 22, 20, 18, 16, 15, 14, 13, 12, 10, 8, 6, 4, 2]
OUTWARD = [10, 12, 14, 16, 20, 24]
C7_TARGETS = [20, 16, 12]
MU = 20.0


_CACHE = {}


def _dfree():
    if "dfree" not in _CACHE:
        m30, x, _ = H.load56(30)
        _CACHE["dfree"] = H.dprime(m30, x)["D"]
    return _CACHE["dfree"]


def build_rows():
    R = {}

    def add(n, name, grid, kind, start, cap, noise=False, rung=None, **kw):
        R[n] = dict(n=n, name=name, grid=grid, kind=kind, start=start, cap=cap, noise=noise, rung=rung, **kw)
    add(1, "A1", "A24", "free", ("rungA", 25.525, 14.0), 30000, True, "A", box=24)
    add(2, "A2", "A24", "free", ("rungA", 37.526, 14.0), 30000, True, "A", box=24)
    add(3, "A3", "A24", "free", ("rungA", 25.525, 0.0), 30000, True, "A", box=24, far=0.0)
    add(4, "A4", "A30", "free", ("rungA", 25.525, 14.0), 30000, True, "A", box=30)
    add(5, "A5", "A30", "free", ("rungA", 49.526, 14.0), 30000, True, "A", box=30)
    add(6, "F375", "P375", "free", ("interp", "S30"), 30000, True, "ref")
    add(7, "M375", "P375", "free", ("interp", "S33"), 30000, True, "ref")
    add(8, "C5h1", "H1", "held", ("end", "S22"), 10000, False, "C5", target=1.0, mu=MU, form="one")
    add(9, "C5h2", "H1", "held", ("end", "S22"), 10000, False, "C5", target=2.0, mu=MU, form="one")
    add(10, "C5r1", "H1", "release", ("end", 8), 10000, False, "C5", form="one")
    add(11, "C5r2", "H1", "release", ("end", 9), 10000, False, "C5", form="one")
    for k in range(1, 5):
        add(11 + k, f"C6k{k}", "P", "held", ("end", "S30" if k == 1 else 11 + k - 1), 10000, False, "C6", target=("Dfree", k), mu=MU, form="pair")
    for i, t in enumerate(INWARD):
        add(16 + i, f"I05_{t}", "P", "held", ("end", "S30" if i == 0 else 16 + i - 1), 10000, False, "B", target=float(t), mu=MU, form="pair",
            ladder="in", spacing=0.5, pos=i)
    for i, t in enumerate(OUTWARD):
        add(30 + i, f"O05_{t}", "P", "held", ("end", "S33" if i == 0 else 30 + i - 1), 10000, False, "B", target=float(t), mu=MU, form="pair",
            ladder="out", spacing=0.5, pos=i)
    for i, t in enumerate(INWARD):
        add(36 + i, f"I375_{t}", "P375", "held", ("interp", 16 + i), 10000, False, "B", target=float(t), mu=MU, form="pair",
            ladder="in", spacing=0.375, pos=i, parent=16 + i)
    for i, t in enumerate(OUTWARD):
        add(50 + i, f"O375_{t}", "P375", "held", ("interp", 30 + i), 10000, False, "B", target=float(t), mu=MU, form="pair",
            ladder="out", spacing=0.375, pos=i, parent=30 + i)
    for j, t in enumerate(C7_TARGETS):
        i = INWARD.index(t)
        add(56 + j, f"C7_05_{t}", "P", "held", ("end", 16 + i - 1), 10000, False, "C7", target=float(t), mu=1.0, form="pair", pair_with=16 + i)
        add(59 + j, f"C7_375_{t}", "P375", "held", ("interp", 16 + i), 10000, False, "C7", target=float(t), mu=1.0, form="pair", pair_with=36 + i,
            parent=16 + i)
    add(62, "C8_05", "P", "release", ("end", 16), 10000, False, "C8")
    add(63, "C8_375", "P375", "release", ("end", 36), 10000, False, "C8")
    add(64, "Rel_I05", "P", "release", ("decided",), 10000, False, "release")
    add(65, "Rel_O05", "P", "release", ("decided",), 10000, False, "release")
    add(66, "Rel_I375", "P375", "release", ("decided",), 10000, False, "release")
    add(67, "Rel_O375", "P375", "release", ("decided",), 10000, False, "release")
    for j in range(3):
        add(68 + j, f"CT{j + 1}", "P", "held", ("decided",), 10000, True, "C", mu=MU, form="pair")
    return R


ROWS = build_rows()


def cname(n):
    return f"r{n:02d}_{ROWS[n]['name']}"


def decisions():
    return json.loads(DECISIONS.read_text()) if DECISIONS.exists() else {}


def load_rec(name):
    f = WORK / f"{name}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return d if d.get("finished") else None


def rec(n, k5=False):
    return load_rec(("k5_" if k5 else "") + cname(n))


def end_field(n, k5=False):
    name = ("k5_" if k5 else "") + cname(n)
    r = load_rec(name)
    z = np.load(WORK / name / r["end_file"], allow_pickle=False)
    return np.asarray(z["m6"], float)


def far_of(spec, x):
    if spec["grid"] == "H1":
        return T.seed_bi(x)
    return H.far_field(x, spec.get("far", H.DFAR))


def target_of(spec):
    t = spec.get("target")
    if isinstance(t, tuple) or isinstance(t, list):
        return _dfree() - t[1]
    return t


# ----------------------------------------------------------------------------- the job

class Job:
    pass


def source_field(src):
    """A start's source: a stored end of M5.56 ('S30') or the end of a row of this task."""
    if isinstance(src, str) and src.startswith("S"):
        m6, xs, r = H.load56(int(src[1:]))
        return m6, xs
    r = rec(src)
    if r is None or r.get("end_state") == "NOT RUN":
        raise RuntimeError(f"not run: no source ({src})")
    return end_field(src), H.axis_of(ROWS[src]["grid"])


def build(n, k5=False):
    spec = dict(ROWS[n])
    dec = decisions().get(str(n), {})
    j = Job()
    j.n, j.spec, j.k5 = n, spec, k5
    j.name = ("k5_" if k5 else "") + cname(n)
    j.grid = spec["grid"]
    j.x = H.axis_of(j.grid)
    j.h = j.x[1] - j.x[0]
    j.N = len(j.x)
    j.info = {}
    hold = far_of(spec, j.x)
    face = T.faces_mask(j.N)
    st = spec["start"]
    j.ref = None
    if st[0] == "rungA":
        m6, hold2, lad = H.rungA_start(j.x, st[1], st[2])
        assert np.array_equal(hold2, hold)
        j.info.update(d0=st[1], far_separation=st[2], ladder=lad)
    elif st[0] == "interp":
        src, xs = source_field(st[1])
        m6 = H.interp(src, xs, j.x)
        j.info.update(source=st[1] if isinstance(st[1], str) else cname(st[1]), face_deviation_before_reset=float(np.abs(m6 - hold)[face].max()))
    elif st[0] == "end":
        src, xs = source_field(st[1])
        assert np.allclose(xs, j.x)
        m6 = src
        j.info.update(source=st[1] if isinstance(st[1], str) else cname(st[1]))
    elif st[0] == "decided":
        if "source" in dec:
            src, xs = source_field(dec["source"])
            m6 = src
            j.info.update(source=cname(dec["source"]))
        elif "ref" in dec:
            ref = dec["ref"]
            rr = rec(ref)
            j.ref = dict(n=ref, name=cname(ref), rec=rr)
            m6 = end_field(ref)
            spec["target"] = ROWS[ref]["target"]
            spec["mu"] = ROWS[ref]["mu"]
            j.info.update(reference=cname(ref))
        else:
            raise RuntimeError("not run: no decision")
    else:
        raise ValueError(st)
    j.mod = H.make_model(j.x, hold)
    m6 = j.mod.hold_faces(m6)
    j.hold6 = hold
    j.free = j.mod.mask.copy()
    j.info["faces_vs_far_field"] = float(np.abs(m6 - hold)[face].max())
    j.held = spec["kind"] == "held"
    j.target = target_of(spec) if j.held else None
    j.mu = spec.get("mu") if j.held else None
    j.form = spec.get("form", "pair")
    j.E = H.Held(j.mod, j.x, j.target, j.mu, j.form)
    j.start_clean = m6.copy()
    j.noise_seed = (57100 + n) if k5 else ((57000 + n) if spec["noise"] else None)
    j.cap = 10000 if k5 else spec["cap"]
    j.wall = WALL[j.grid]
    j.collar = None
    if j.ref is not None:
        prepare_turn(j)
    return j


def prepare_turn(j):
    """Rung C: the collar about ring 1 of the reference, the turn T[2pi] about its fitted normal, the reads' masks."""
    e = j.ref["rec"]["end"]
    r1, r2 = e["pair"]["rings"][0], e["pair"]["rings"][1]
    c1, c2 = np.asarray(r1["c"]), np.asarray(r2["c"])
    L = j.x[-1]
    R1 = r1["RL"] + 1.5
    R2 = min(float(np.linalg.norm(c1 - c2)) - r2["RL"] - 1.5, min(L - abs(c1[a]) for a in range(3)) - 2 * j.h)
    X = T.mesh(j.x)
    excl = np.linalg.norm(X - c2, axis=-1) <= r2["RL"] + 2.0
    nu = T.unit(r1["nu"])
    j.rings = [dict(c=np.asarray(q["c"]), nu=T.unit(q["nu"]), RL=q["RL"]) for q in (r1, r2)]
    j.collar = dict(c=c1, R1=R1, R2=R2, outer=R2 + j.h, exclude=excl, axis=nu, Theta=2 * np.pi)
    j.ref6 = j.start_clean.copy()
    j.tube = T.tubes(j.x, j.rings)
    j.mod.evaluate(j.ref6, need_grad=False)
    j.ref_dens = j.mod.edens.copy()
    j.U_ref = float(j.mod.evaluate(j.ref6, need_grad=False)["U"])
    j.info.update(collar=dict(c=c1.tolist(), R1=R1, R2=R2, width=R2 - R1, axis=nu.tolist(), outer=R2 + j.h))


def turned_start(j):
    cl = j.collar
    return T.T_turn(j.ref6, j.x, cl["Theta"], cl["axis"], cl["c"], cl["R1"], cl["R2"])


def add_noise(j, m6):
    if j.noise_seed is None:
        return m6
    rng = np.random.default_rng(j.noise_seed)
    out = m6.copy()
    out[j.free] += 1e-4 * rng.standard_normal((int(j.free.sum()), 6))
    return out


# ----------------------------------------------------------------------------- reads at the stops

def route_read(j, m6, it, major):
    m6 = np.asarray(m6, float).reshape(j.N, j.N, j.N, 6)
    p = j.E.evaluate(m6, need_grad=True)
    G = float(np.abs(p["grad"]).max() / j.h ** 3)
    out = dict(iteration=it, major=bool(major), U=p["U"], U4=p["U4"], UV=p["UV"], ULV=p["ULV"], E=p["E"], gate=G)
    out["torque"] = H.torque(j.mod, None, m6, j.mod.gU)
    eig = T.eigs(m6)
    dp = H.dprime(m6, j.x, one=j.form == "one", eig=eig)
    if j.form == "one":
        out.update(Dprime=dp["D1"])
    else:
        out.update(Dprime=dp["D"], Dprime_plus=dp["Dp"], Dprime_minus=dp["Dm"], offset=dp["Dp"] - dp["Dm"])
    if j.held:
        out.update(f=p["f"], F=p["F"], hold_energy=p["hold_energy"], gate_threshold=j.E.gate_threshold(p["F"]))
    else:
        out["gate_threshold"] = 1e-3
    if j.collar is not None:
        cl = j.collar
        j.mod.evaluate(m6, need_grad=False)
        dens = j.mod.edens.copy()
        tr = T.turn_reader(m6, j.ref6, j.x, cl["c"], cl["R1"], outer=cl["outer"], exclude=cl["exclude"], axis=cl["axis"], dens=(dens, j.ref_dens))
        out["dU"] = out["U"] - j.U_ref
        out["turn"] = dict(sigma=tr["sigma"], q0_sign_shell=tr["q0_sign_shell"], sign_shell=tr["sign_shell"], chi_max=tr["chi_max"],
                           rho_t_cross=tr["rho_t_cross"], rho_least_q0=tr["rho_least_q0"], r_max=tr["r_max"], step_max=tr["step_max"],
                           q0_outer=tr["q0_outer"], regime=tr["regime"], in_regime=tr["in_regime"],
                           shells=[dict(rho=r["rho"], q=r["q"], r=r["r"], chi=r["chi"], angle=r.get("angle"), dU=r.get("dU_shell"), n=r["n"]) for r in tr["rows"]])
        out["tear"] = T.tear_reads(m6, j.x, H.DELTA, j.tube, j.free)
        out["mu_free"] = T.mu(m6, j.ref6, j.free & ~j.tube)
    if major:
        out.update(major_read(j, m6, eig))
    return out


def major_read(j, m6, eig):
    out = {}
    if j.grid == "H1":
        r = RD.read_state(m6, j.x, 12.0, "mid", with_grad=False)
        keep = ("RL", "center", "normal", "line_regime", "line_gmax", "halfturn_all", "cls", "one_ring", "all_crossings", "far_nodes", "degree")
        out.update({k: r.get(k) for k in keep})
        out["dprime_regime"] = H.dprime_regime(m6, j.x, eig=eig)
        return out
    pr = T.two_ring_reader(m6, j.x)
    out["pair"] = pr
    out["dprime_regime"] = H.dprime_regime(m6, j.x, eig=eig)
    dc = pr.get("separation") if pr["word"] != "ONE OBJECT" and pr.get("separation") is not None else 0.0
    out["dmoment"] = H.dmoment(m6, j.x, dc, eig=eig)
    if pr.get("separation") is not None:
        out["s"] = pr["separation"]
        out["g_faces"] = [float(j.x[-1] - abs(q["c"][0])) for q in pr["rings"]]
    return out


def own_tube(j, e):
    """The tube of the second step's mu: about the end's own fitted line(s) (M5.56 deviation 7 (k))."""
    if j.collar is not None:
        return j.tube
    if j.grid == "H1":
        if e.get("RL") is None:
            return np.zeros(j.free.shape, bool)
        return T.tubes(j.x, [dict(c=np.asarray(e["center"]), nu=np.asarray(e["normal"]), RL=e["RL"])])
    rings = (e.get("pair") or {}).get("rings") or []
    return T.tubes(j.x, [dict(c=np.asarray(q["c"]), nu=np.asarray(q["nu"]), RL=q["RL"]) for q in rings]) if rings else np.zeros(j.free.shape, bool)


def word_of(e):
    if "pair" in e:
        return e["pair"]["word"]
    return "one ring" if e.get("one_ring") else "not one ring"


def ring_cmp(e1, e2):
    """|ds| on two TWO RINGS reads; the largest |dR_L| on one ring (H1); None when the clause is not used (ONE
    OBJECT or neither on both); inf when the words differ."""
    if "pair" in e1:
        w1, w2 = e1["pair"]["word"], e2["pair"]["word"]
        if w1 != w2:
            return float("inf")
        if w1 == "TWO RINGS":
            return abs(e1["pair"]["separation"] - e2["pair"]["separation"])
        return None
    if e1.get("RL") is None or e2.get("RL") is None:
        return float("inf")
    return abs(e1["RL"] - e2["RL"])


def compare(j, e, er, fe, fr, dU_bar, ring_bar, dprime_bar, mu_bar):
    """Two reads of one chain: the checks of the second step (or of rule (b))."""
    ch = {}
    vals = dict(dU_rel=abs(e["U"] / er["U"] - 1), dDprime=abs(e["Dprime"] - er["Dprime"]))
    ch["U"] = bool(vals["dU_rel"] <= dU_bar)
    rc = ring_cmp(e, er)
    vals["ds_or_dRL"] = rc
    ch["s_or_RL"] = bool(rc is None or rc <= ring_bar)
    ch["Dprime"] = bool(vals["dDprime"] <= dprime_bar)
    m = T.mu(fe, fr, j.free & ~own_tube(j, e))
    vals["mu_free"] = m
    ch["mu_free"] = bool(m["in_regime"] and m["mu"] <= mu_bar)
    ch["word"] = bool(word_of(e) == word_of(er))
    return dict(values=vals, checks=ch, ok=bool(all(ch.values())))


# ----------------------------------------------------------------------------- the descent

def is_rel_reduction(message):
    """scipy's L-BFGS-B ends by the relative-reduction test with the message 'CONVERGENCE: RELATIVE REDUCTION OF F <=
    FACTR*EPSMCH' (older releases: 'REL_REDUCTION_OF_F_<=_FACTR*EPSMCH')."""
    m = str(message).upper()
    return "RELATIVE REDUCTION" in m or "REL_REDUCTION" in m


def run(j, m6_0, work, rec_, log, guard, deadline):
    state_f = work / "state.json"
    st = json.loads(state_f.read_text()) if state_f.exists() else dict(iteration=0, nfev=0, stops=[], restarts=[], resumes=[], saved=[])
    x = m6_0.reshape(-1).copy()
    if st["iteration"] > 0:
        last = [s for s in st["saved"] if not s.get("deleted")][-1]
        x = np.load(work / last["file"], allow_pickle=False)["m6"].reshape(-1)
        st["resumes"].append(st["iteration"])
        log(f"resumes at it {st['iteration']} from {last['file']} with a fresh optimizer memory")
    box = dict(it=st["iteration"], nfev=0, cut=None)
    cap = j.cap
    majors = tuple(m for m in MAJOR if m <= cap)

    def is_route(it):
        return it in (10, 20, 50, 100, 200, 300, 500, 700, 1000) or (it > 1000 and it % 500 == 0)

    def save_state():
        state_f.write_text(json.dumps(dict(st, nfev=st["nfev"] + box["nfev"]), default=T._jsonable))

    def stop(z, it, kind):
        major = it in majors or kind == "end"
        f = f"s{it:05d}.npz" if kind != "end" else "end.npz"
        np.savez(work / f, m6=np.asarray(z).reshape(j.N, j.N, j.N, 6), x=j.x)
        r = route_read(j, z, it, major)
        r.update(kind=kind, file=f)
        st["stops"].append(r)
        st["saved"].append(dict(iteration=it, file=f, major=bool(major), keep=bool(it <= 1000 or major or kind == "end")))
        route_only = [s for s in st["saved"] if not s["keep"]]
        for s in route_only[:-3]:
            (work / s["file"]).unlink(missing_ok=True)
            s["deleted"] = True
            s["keep"] = True
        st["iteration"] = it
        save_state()
        line = f"{kind} it {it} U {r['U']:.10f} gate {r['gate']:.3e} thr {r['gate_threshold']:.1e} tq {r['torque']:.1e} D' {r['Dprime']:.6f}"
        if j.held:
            line += f" f {r['f']:.5f}"
        if "pair" in r:
            line += f" pair {r['pair']['word']} s {r.get('s')}"
        if "turn" in r:
            line += f" sigma {r['turn']['sigma']} chi {r['turn']['chi_max']:.3f} reg {r['turn']['in_regime']} tear {r['tear']['tear']}"
        if "RL" in r and major:
            line += f" RL {r['RL']} c {r.get('center')}"
        log(line)
        return r

    if st["iteration"] == 0 and not st["stops"]:
        stop(x, 0, "start")

    def fun(z):
        p = j.E.evaluate(z.reshape(j.N, j.N, j.N, 6))
        box["nfev"] += 1
        return p["E"], p["grad"].reshape(-1)

    def cb(z, *args):
        box["it"] += 1
        it = box["it"]
        if is_route(it) or it in majors:
            stop(np.asarray(z).copy(), it, "stop")
            (work / "progress.json").write_text(json.dumps(dict(chain=j.name, iteration=it, time=time.strftime("%Y-%m-%d %H:%M:%S"))))
            if guard is not None:
                try:
                    if guard.wrap_up():
                        box["cut"] = "power"
                        raise StopIteration
                except StopIteration:
                    raise
                except Exception:  # noqa: BLE001
                    pass
        if it % 25 == 0 and time.time() > deadline:
            box["cut"] = "wall"
            box["x"] = np.asarray(z).copy()
            raise StopIteration
        if it >= cap:
            box["cut"] = "cap"
            box["x"] = np.asarray(z).copy()
            raise StopIteration

    message = ""
    by_tol = False
    rel_red = False
    while True:
        left = cap - box["it"]
        if left <= 0:
            message = "CAP"
            break
        maxfun = int(1.25 * cap) - (st["nfev"] + box["nfev"])
        if maxfun <= 0:
            message = "MAXFUN"
            break
        res = minimize(fun, x, jac=True, method="L-BFGS-B", callback=cb, options=dict(maxiter=left, maxfun=maxfun, ftol=0.0, gtol=1e-7, maxcor=10))
        x = res.x
        message = str(res.message)
        if box["cut"] == "power":
            np.savez(work / f"s{box['it']:05d}.npz", m6=x.reshape(j.N, j.N, j.N, 6), x=j.x)
            st["saved"].append(dict(iteration=box["it"], file=f"s{box['it']:05d}.npz", major=False, keep=True))
            st["iteration"] = box["it"]
            save_state()
            log("power warning: state saved, exit 75")
            sys.exit(75)
        if box["cut"] == "wall":
            x = box["x"]
            message = "CUT BY THE WALL-TIME CAP"
            break
        if box["cut"] == "cap":
            x = box["x"]
            message = "CUT BY THE ITERATION CAP (the callback count)"
            break
        mu_ = message.upper()
        by_tol = "PROJECTED GRADIENT" in mu_
        if by_tol:
            break
        abnormal = "ABNORMAL" in mu_
        rel_red = is_rel_reduction(mu_)
        stalled = bool(st["restarts"]) and st["restarts"][-1]["iteration"] == box["it"] and not rel_red
        if (abnormal or rel_red) and len(st["restarts"]) < 3 and not stalled:
            st["restarts"].append(dict(iteration=box["it"], message=message))
            log(f"restart {len(st['restarts'])} at it {box['it']}: {message}")
            continue
        break
    it = box["it"]
    st["nfev"] += box["nfev"]
    box["nfev"] = 0
    if st["stops"][-1]["iteration"] == it and st["stops"][-1]["kind"] != "start":
        st["stops"].pop()
        st["saved"] = [s for s in st["saved"] if s["iteration"] != it]
    end = stop(x, it, "end")
    rec_.update(iterations=it, evaluations=st["nfev"], message=message, ended_by_gtol=bool(by_tol), cut_by_wall=bool(box["cut"] == "wall"),
                ended_at_cap=bool(box["cut"] == "cap"), restarts=st["restarts"], resumes=st["resumes"], end=end, end_file="end.npz", stops=st["stops"],
                saved=st["saved"], ended_by_rel_reduction=bool(is_rel_reduction(message)))
    fe = x.reshape(j.N, j.N, j.N, 6)
    # the second step: the last saved stop >= 500 iterations before the end, else >= 100
    cands = [s for s in st["saved"] if not s.get("deleted") and s["file"] != "end.npz" and (work / s["file"]).exists()]
    earlier = None
    for gap in (500, 100):
        c = [s for s in cands if s["iteration"] <= it - gap]
        if c:
            earlier = max(c, key=lambda q: q["iteration"])
            break
    if earlier is not None:
        fr = np.load(work / earlier["file"], allow_pickle=False)["m6"]
        er = route_read(j, fr, earlier["iteration"], True)
        er.update(kind=f"saved stop {earlier['iteration']}", file=earlier["file"])
        rec_["earlier"] = er
        ss = compare(j, end, er, fe, fr, 1e-6, 0.002, 1e-4, 0.003)
        ss.update(source=f"saved stop {earlier['iteration']}")
        rec_["second_step"] = ss
    else:
        rec_["earlier"] = None
        rec_["second_step"] = dict(ok=None, source="needs K5 (no saved stop 100 or more before the end)")
    rec_["converged_check"] = converged(j, rec_, work)
    rec_["end_state"] = "CONVERGED" if rec_["converged_check"]["converged"] else "CAPPED"
    return rec_


def converged(j, r, work):
    e = r["end"]
    out = dict(gate=e["gate"], gate_threshold=e["gate_threshold"], gate_ok=bool(e["gate"] <= e["gate_threshold"]), torque=e["torque"],
               torque_ok=bool(e["torque"] <= 1e-5), by_gtol=r["ended_by_gtol"])
    maj = [s for s in r["stops"] if s.get("major") and s["kind"] in ("stop", "end")]
    pair = None
    q = maj[-1]
    cands = [p for p in maj[:-1] if q["iteration"] - p["iteration"] >= 5000]
    if cands:
        p = cands[-1]
        fp = work / p["file"]
        if fp.exists():
            cp = compare(j, q, p, np.load(work / q["file"], allow_pickle=False)["m6"], np.load(fp, allow_pickle=False)["m6"], 1e-8, 0.002, 1e-4, 0.003)
            pair = dict(iterations=[p["iteration"], q["iteration"]], **cp)
    out["stops_pair"] = pair
    rule_b = bool(pair and pair["ok"])
    ss = r.get("second_step") or {}
    rule_c = bool(r["ended_by_rel_reduction"] and len(r["restarts"]) >= 1 and is_rel_reduction(r["restarts"][-1]["message"])
                  and e["gate"] <= 1e-5 and ss.get("ok") is True)
    out.update(rule_a=bool(r["ended_by_gtol"]), rule_b=rule_b, rule_c=rule_c)
    out["converged"] = bool(out["gate_ok"] and out["torque_ok"] and (out["rule_a"] or rule_b or rule_c))
    return out


def k5_compare(j, r, src, work_src, work_k5):
    """K5 met: its own relaxation CONVERGED, U within 1e-6 relative, the pair word unchanged, s (or R_L) within 0.002,
    D' within 1e-4, mu_free <= 0.01; on a rung C chain also sigma unchanged and chi_max on the same side of 0.05 and
    pi/2 (M5.56's turn clauses, the stricter reading)."""
    e, s = r["end"], src["end"]
    fe = np.load(work_k5 / "end.npz", allow_pickle=False)["m6"]
    fs = np.load(work_src / "end.npz", allow_pickle=False)["m6"]
    c = compare(j, e, s, fe, fs, 1e-6, 0.002, 1e-4, 0.01)
    ch = dict(c["checks"])
    ch["own_relaxation_converged"] = bool(r["end_state"] == "CONVERGED")
    if "turn" in e:
        ch["sigma"] = bool(e["turn"]["sigma"] == s["turn"]["sigma"])
        ch["chi_side"] = bool(all((e["turn"]["chi_max"] <= b) == (s["turn"]["chi_max"] <= b) for b in (0.05, np.pi / 2)))
    return dict(values=c["values"], checks=ch, passed=bool(all(ch.values())))


# ----------------------------------------------------------------------------- status reads (launcher and reads)

def k5_state(n):
    k = rec(n, k5=True)
    if k is None:
        return "not run"
    return "met" if k["k5"]["passed"] else ("not met (capped)" if k.get("end_state") != "CONVERGED" else "not met")


def second_step_ok(n):
    """True, False, or None (pending: needs a K5 not yet finished)."""
    r = rec(n)
    if r is None or r.get("end_state") == "NOT RUN":
        return False
    ss = r.get("second_step") or {}
    if ss.get("ok") is not None:
        return bool(ss["ok"])
    k = k5_state(n)
    if k == "not run":
        return None
    return k == "met"


def needs_k5_for_second_step(n):
    r = rec(n)
    return bool(r is not None and r.get("end_state") == "CONVERGED" and (r.get("second_step") or {}).get("ok") is None)


def in_hold(n):
    r = rec(n)
    if r is None or r.get("end_state") != "CONVERGED":
        return False
    ss = second_step_ok(n)
    if ss is None:
        return None
    return bool(ss and r["end"]["dprime_regime"]["in_regime"])


def pair_word(n):
    r = rec(n)
    if r is None or r.get("end_state") == "NOT RUN":
        return None
    return r["end"]["pair"]["word"]


def evaded(n):
    r = rec(n)
    e = r["end"]
    return bool(e["pair"]["word"] == "TWO RINGS" and abs(e["Dprime"] - e["pair"]["separation"]) > 0.5)


def usable(n):
    h = in_hold(n)
    return bool(h and not evaded(n))


# ----------------------------------------------------------------------------- the controls that are chains (C5 to C8)

U22 = None


def _u22():
    global U22
    if U22 is None:
        m22, x, r = H.load56(22)
        U22 = (float(r["end"]["U"]), m22, x)
    return U22


def finished(n):
    return rec(n) is not None


def not_run(n):
    r = rec(n)
    return bool(r is not None and r.get("end_state") == "NOT RUN")


def good_end(n):
    """CONVERGED with the second step met (True / False / None pending)."""
    r = rec(n)
    if r is None:
        return None
    if r.get("end_state") != "CONVERGED":
        return False
    return second_step_ok(n)


def c5_result():
    """C5: rows 8 and 9 (held at a = 1.0 and 2.0), rows 10 and 11 (their releases). Every chain CONVERGED with its
    second step met (this task's stricter reading), and the listed checks. Returns dict with met True/False/None."""
    U0, m22, x = _u22()
    mod = _CACHE.setdefault("mod22", H.make_model(x, T.seed_bi(x)))
    out = dict(rows={})
    met = True
    for held, rel, a in ((8, 10, 1.0), (9, 11, 2.0)):
        k = int(round(a / 0.5))
        if ("cost", k) not in _CACHE:
            _CACHE[("cost", k)] = float(mod.evaluate(mod.hold_faces(H.moved(m22, k, 0, T.seed_bi(x))), need_grad=False)["U"] - U0)
        cost = _CACHE[("cost", k)]
        o = dict(a=a, fixed_field_cost_of_moved_copy=cost)
        for n in (held, rel):
            g = good_end(n)
            o[f"row{n}_converged_second_step"] = g
            if g is None:
                met = None if met is not False else False
            elif not g:
                met = False
        rh, rr = rec(held), rec(rel)
        if rh is not None and rh.get("end_state") != "NOT RUN":
            e = rh["end"]
            c = e.get("center")
            o.update(center=c, RL=e.get("RL"), share=e["dprime_regime"]["share"], dU=e["U"] - U0, Dprime=e["Dprime"])
            ok = bool(c is not None and abs(c[0] - a) <= 0.1 and abs(c[1]) <= 0.1 and abs(c[2]) <= 0.1 and e.get("RL") is not None
                      and abs(e["RL"] - 2.9391) <= 0.02 and e["dprime_regime"]["share"] >= 0.99 and 0 <= e["U"] - U0 <= cost)
            o["held_checks"] = ok
            met = False if not ok else met
        if rr is not None and rr.get("end_state") != "NOT RUN":
            e = rr["end"]
            c = e.get("center")
            o.update(release_U_rel=abs(e["U"] / U0 - 1), release_center=c)
            ok = bool(abs(e["U"] / U0 - 1) <= 1e-6 and c is not None and np.linalg.norm(c) <= 0.05)
            o["release_checks"] = ok
            met = False if not ok else met
        if rh is None or rr is None:
            met = None if met is not False else False
        out["rows"][a] = o
    out["met"] = met
    return out


def step_identity(Ua, Ub, fa, fb, Da, Db, rel, floor):
    dU = Ub - Ua
    Tk = 0.5 * (fa + fb) * (Db - Da)
    return dict(dU=dU, T=Tk, err=abs(dU - Tk), bound=max(rel * abs(dU), floor), passes=bool(abs(dU - Tk) <= max(rel * abs(dU), floor)))


def c6_result():
    rows = [12, 13, 14, 15]
    out = dict(steps=[])
    if any(rec(n) is not None and rec(n).get("end_state") == "NOT RUN" for n in rows):
        out["met"] = False
        out["why"] = [rec(n).get("reason") for n in rows if rec(n) is not None and rec(n).get("end_state") == "NOT RUN"][0]
        return out
    if any(rec(n) is None for n in rows):
        out["met"] = None
        return out
    counted = 0
    ok = True
    pend = False
    r30 = _CACHE.setdefault("r30", H.load56(30)[2])
    e30 = dict(U=r30["end"]["U"], f=0.0, Dprime=_dfree())
    prev = None
    for n in rows:
        e = rec(n)["end"]
        hold = in_hold(n)
        if hold is None:
            pend = True
        cur = dict(n=n, U=e["U"], f=e["f"], Dprime=e["Dprime"], in_hold=hold, word=e["pair"]["word"])
        if prev is not None:
            si = step_identity(prev["U"], cur["U"], prev["f"], cur["f"], prev["Dprime"], cur["Dprime"], 0.05, 1e-4)
            si["rows"] = [prev["n"], n]
            si["counted"] = bool(prev["in_hold"] and cur["in_hold"] and abs(si["dU"]) >= 2e-4)
            if si["counted"]:
                counted += 1
                ok &= si["passes"]
            out["steps"].append(si)
        else:
            si = step_identity(e30["U"], cur["U"], 0.0, cur["f"], e30["Dprime"], cur["Dprime"], 0.05, 1e-4)
            si["rows"] = ["row 30", n]
            si["descriptor_only"] = True
            out["from_the_free_end"] = si
        prev = cur
    out["counted"] = counted
    out["met"] = None if pend else bool(ok and counted >= 2)
    return out


def c7_result(spacing):
    js = [56, 57, 58] if spacing == 0.5 else [59, 60, 61]
    out = dict(targets=[])
    read = 0
    ok = True
    pend = False
    for jn in js:
        pw = ROWS[jn]["pair_with"]
        r1, r20 = rec(jn), rec(pw)
        t = dict(target=ROWS[jn]["target"], rows=[pw, jn])
        if r1 is None or r20 is None:
            pend = True
            t["read"] = "pending"
        elif r1.get("end_state") == "NOT RUN" or r20.get("end_state") == "NOT RUN":
            t["read"] = "not read (not run)"
        else:
            h1, h20 = in_hold(jn), in_hold(pw)
            if h1 is None or h20 is None:
                pend = True
            e1, e20 = r1["end"], r20["end"]
            si = step_identity(e20["U"], e1["U"], e20["f"], e1["f"], e20["Dprime"], e1["Dprime"], 0.05, 1e-4)
            t.update(si)
            t.update(words=[e20["pair"]["word"], e1["pair"]["word"]], in_hold=[h20, h1])
            if h1 and h20 and e1["pair"]["word"] == "TWO RINGS" and e20["pair"]["word"] == "TWO RINGS" and abs(si["dU"]) >= 2e-4:
                t["read"] = "read"
                read += 1
                ok &= si["passes"]
            else:
                t["read"] = "not read"
        out["targets"].append(t)
    out["read"] = read
    out["met"] = None if pend else bool(ok and read >= 2)
    return out


def free_end(spacing):
    """(U, s, word, D') of the free reference of a spacing: row 30 of M5.56, or F375."""
    if spacing == 0.5:
        r = _CACHE.setdefault("r30", H.load56(30)[2])["end"]
        return dict(U=r["U"], s=r["pair"]["separation"], word=r["pair"]["word"], Dprime=_dfree(), name="M5.56 row 30")
    r = rec(6)
    if r is None or r.get("end_state") == "NOT RUN":
        return None
    e = r["end"]
    return dict(U=e["U"], s=e.get("s"), word=e["pair"]["word"], Dprime=e["Dprime"], name=cname(6))


def merged_end(spacing):
    if spacing == 0.5:
        if "merged05" not in _CACHE:
            m33, x, r = H.load56(33)
            _CACHE["merged05"] = dict(U=r["end"]["U"], word=r["end"]["pair"]["word"], Dprime=H.dprime(m33, x)["D"], name="M5.56 row 33")
        return dict(_CACHE["merged05"])
    r = rec(7)
    if r is None or r.get("end_state") == "NOT RUN":
        return None
    e = r["end"]
    return dict(U=e["U"], word=e["pair"]["word"], Dprime=e["Dprime"], name=cname(7))


def c8_result(spacing):
    n = 62 if spacing == 0.5 else 63
    r = rec(n)
    fe = free_end(spacing)
    if r is None or fe is None:
        return dict(met=None if (r is None or (spacing == 0.375 and rec(6) is None)) else False)
    if r.get("end_state") == "NOT RUN":
        return dict(met=False, reason=r.get("reason"))
    g = good_end(n)
    e = r["end"]
    out = dict(converged_second_step=g, U=e["U"], U_free=fe["U"], U_rel=abs(e["U"] / fe["U"] - 1), word=e["pair"]["word"], s=e.get("s"), s_free=fe["s"])
    if g is None:
        out["met"] = None
        return out
    out["met"] = bool(g and out["U_rel"] <= 1e-6 and e["pair"]["word"] == "TWO RINGS" and e.get("s") is not None and fe["s"] is not None
                      and abs(e["s"] - fe["s"]) <= 0.01)
    return out


def ladder_rows(spacing, which):
    if spacing == 0.5:
        return list(range(16, 30)) if which == "in" else list(range(30, 36))
    return list(range(36, 50)) if which == "in" else list(range(50, 56))


def levels(spacing):
    """The level rule of section 4 on what has ended: per target, the usable rungs of both ladders and E_low."""
    fe, me = free_end(spacing), merged_end(spacing)
    tg = {}
    for which in ("in", "out"):
        for n in ladder_rows(spacing, which):
            r = rec(n)
            t = ROWS[n]["target"]
            d = tg.setdefault(t, dict(target=t, rungs=[]))
            if r is None or r.get("end_state") == "NOT RUN":
                d["rungs"].append(dict(n=n, ladder=which, state="pending" if r is None else r.get("reason")))
                continue
            e = r["end"]
            u = usable(n)
            d["rungs"].append(dict(n=n, ladder=which, U=e["U"], word=e["pair"]["word"], usable=u, in_hold=in_hold(n), evaded=evaded(n)))
    for t, d in tg.items():
        us = [q for q in d["rungs"] if q.get("usable")]
        if us:
            b = min(us, key=lambda q: q["U"])
            d.update(E_low=b["U"], rung=b["n"], ladder=b["ladder"], word=b["word"])
        else:
            d["E_low"] = None
    return dict(free=fe, merged=me, targets=tg)

# ----------------------------------------------------------------------------- main

def write_not_run(n, reason, extra=None):
    WORK.mkdir(parents=True, exist_ok=True)
    d = dict(name=cname(n), row=n, finished=True, end_state="NOT RUN", reason=reason)
    if extra:
        d.update(extra)
    (WORK / f"{cname(n)}.json").write_text(json.dumps(d, indent=1, default=T._jsonable) + "\n")


def rungC_prep(j):
    """Rung C before its chain: C10 on the reference, the path tables, the start rule (every clause on the relaxed
    reference and the start built from it)."""
    cl = j.collar
    out = dict(collar=j.info["collar"])
    fits = bool(cl["R2"] - cl["R1"] >= 4.0)
    out["collar_fits"] = fits
    if not fits:
        return out, "not run: the collar does not fit"
    h = j.h
    st = turned_start(j)
    # C10: the turn reader about ring 1 with the other ring's ball and tube left out
    tr = T.turn_reader(st, j.ref6, j.x, cl["c"], cl["R1"], outer=cl["outer"], exclude=cl["exclude"], axis=cl["axis"])
    errs = [abs(r["angle"] - 2 * np.pi * T.fprof(r["rho"], cl["R1"], cl["R2"])) for r in tr["rows"]]
    c10 = dict(sigma=tr["sigma"], q0_sign_shell=tr["q0_sign_shell"], angle_err_max=float(max(errs)), r_max=tr["r_max"], in_regime=tr["in_regime"],
               regime=tr["regime"], shells=len(tr["rows"]))
    c10["passed"] = bool(tr["sigma"] == -1 and max(errs) <= 0.15 and tr["r_max"] <= 0.15)
    L = j.x[-1]
    trf = T.turn_reader(st, j.ref6, j.x, cl["c"], cl["R1"], outer=L - 2 * h, exclude=None, axis=cl["axis"])
    c10["failing_input"] = dict(input="the other ring's nodes left in and the shells read through the other ring", in_regime=trf["in_regime"],
                                regime=trf["regime"], fails_as_planned=bool(not trf["in_regime"]))
    out["C10"] = c10
    # the path tables
    tabs = {}
    for an, a in (("normal", cl["axis"]), ("inplane", T.inplane_axis(cl["axis"]))):
        tabs[an] = T.path_table(j.mod, j.ref6, j.x, a, cl["c"], cl["R1"], cl["R2"], top=2.0)
    out["paths"] = tabs
    # the start rule
    th = 2 * np.pi
    nominal = th * h / (cl["R2"] - cl["R1"])
    out_t = ~j.tube
    jl_s = T.jump_links(st, j.x, H.DELTA, out_t)
    jl_r = T.jump_links(j.ref6, j.x, H.DELTA, out_t)
    new_j = link_diff(st, j.ref6, j.x, out_t, "long")
    new_m = link_diff(st, j.ref6, j.x, out_t, "minor")
    rule = dict(nominal_deg=float(np.degrees(nominal)), clause_i=bool(nominal <= np.pi / 4 + 1e-12), jump_links_start=jl_s["jumps"],
                jump_links_reference=jl_r["jumps"], largest_vacuum_deg=jl_s["largest_deg"], new_jump_links=new_j, clause_ii=bool(new_j == 0),
                new_minor_jump_links=new_m, clause_iv=bool(new_m == 0))
    rule["passes"] = bool(rule["clause_i"] and rule["clause_ii"] and rule["clause_iv"])
    out["start_rule"] = rule
    if not c10["passed"]:
        return out, "not run: C10 not met on this reference"
    if not rule["passes"]:
        return out, "not run: start rule"
    return out, None


def link_diff(st, ref, x, region, kind):
    """Jump links of the start (long axis between near-vacuum nodes, or minor) that the reference does not hold, node
    pair by node pair."""
    def links(f6):
        w, V = T.eigs(f6)
        S = set()
        if kind == "long":
            s = A.sfun(A.from6(f6), T.ck(H.DELTA))
            vec = V[..., :, 0]
        else:
            g23 = w[..., 1] - w[..., 2]
            vec = V[..., :, 1]
        for ax, (s0, s1) in enumerate(T._pairs(len(x))):
            if kind == "long":
                ok = (s[s0] < 1e-4) & (s[s1] < 1e-4)
            else:
                ok = (g23[s0] >= H.DELTA / 2) & (g23[s1] >= H.DELTA / 2)
            ok &= region[s0] & region[s1]
            ang = np.degrees(np.arccos(np.clip(np.abs((vec[s0] * vec[s1]).sum(-1)), 0, 1)))
            for idx in np.argwhere(ok & (ang > 45.0)):
                S.add((ax,) + tuple(idx))
        return S
    return len(links(st) - links(ref))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", type=int)
    ap.add_argument("--k5", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--spec", type=int)
    ap.add_argument("--cap", type=int, default=None, help="smoke tests only")
    a = ap.parse_args()
    if a.list:
        for n in ROWS:
            print(n, cname(n), ROWS[n]["grid"], ROWS[n]["kind"], ROWS[n]["start"])
        return
    WORK.mkdir(parents=True, exist_ok=True)
    if a.spec:
        j = build(a.spec)
        print(json.dumps(j.info, default=T._jsonable, indent=1))
        return
    n = a.row or a.k5
    k5 = a.k5 is not None
    name = ("k5_" if k5 else "") + cname(n)
    work = WORK / name
    jf = WORK / f"{name}.json"
    if jf.exists() and json.loads(jf.read_text()).get("finished"):
        print("already finished")
        return
    work.mkdir(parents=True, exist_ok=True)
    logf = open(work / "log.txt", "a")

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {name} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    try:
        j = build(n, k5=k5)
    except RuntimeError as err:
        d = dict(name=name, row=n, finished=True, end_state="NOT RUN", reason=str(err))
        jf.write_text(json.dumps(d, indent=1) + "\n")
        log(f"NOT RUN: {err}")
        return
    guard = None
    try:
        sys.path.insert(0, os.path.expanduser("~/Library/Application Support/UPSSentinel"))
        import ups_guard
        ups_guard.register()
        guard = ups_guard
    except Exception as e:  # noqa: BLE001
        log(f"no power guard: {e}")
    rec_ = dict(name=name, row=n, chain=ROWS[n]["name"], grid=j.grid, kind=ROWS[n]["kind"], rung=ROWS[n]["rung"], target=j.target, mu=j.mu,
                form=j.form, noise_seed=j.noise_seed, info=j.info, cap=j.cap, finished=False)
    if j.collar is not None:
        rec_.update(reference=j.ref["name"], U_ref=j.U_ref, rings=[dict(c=r["c"].tolist(), nu=r["nu"].tolist(), RL=r["RL"]) for r in j.rings])
    if k5:
        src = rec(n)
        m6e = end_field(n)
        m6_0 = add_noise(j, m6e)
        rec_["k5_of"] = cname(n)
    else:
        if j.collar is not None and not (work / "state.json").exists():
            prep, why = rungC_prep(j)
            rec_["prep"] = prep
            (work / "prep.json").write_text(json.dumps(prep, indent=1, default=T._jsonable))
            log(f"rung C prep: {json.dumps(dict(C10=prep.get('C10', {}).get('passed'), rule=prep.get('start_rule'), why=why), default=T._jsonable)}")
            if why:
                rec_.update(finished=True, end_state="NOT RUN", reason=why)
                jf.write_text(json.dumps(rec_, indent=1, default=T._jsonable) + "\n")
                return
        elif j.collar is not None:
            rec_["prep"] = json.loads((work / "prep.json").read_text())
        if ROWS[n]["start"][0] == "rungA":
            rule = H.rungA_rule(j.start_clean, j.x)
            rec_["start_rule"] = rule
            if not rule["passes"]:
                rec_.update(finished=True, end_state="NOT RUN", reason="not run: start rule")
                jf.write_text(json.dumps(rec_, indent=1, default=T._jsonable) + "\n")
                log(f"NOT RUN: start rule {rule}")
                return
        base = turned_start(j) if j.collar is not None else j.start_clean
        m6_0 = add_noise(j, base)
    if a.cap:
        j.cap = a.cap
        rec_["cap"] = a.cap
    log(f"starts: {json.dumps({k: v for k, v in rec_.items() if k not in ('finished', 'prep')}, default=T._jsonable)}")
    t0 = time.time()
    prev = json.loads((work / "wall.json").read_text())["seconds"] if (work / "wall.json").exists() else 0.0
    deadline = t0 + j.wall * 3600 - prev
    try:
        rec_ = run(j, m6_0, work, rec_, log, guard, deadline)
    finally:
        (work / "wall.json").write_text(json.dumps(dict(seconds=prev + time.time() - t0)))
    if k5:
        rec_["k5"] = k5_compare(j, rec_, src, WORK / cname(n), work)
        log(f"K5 {'MET' if rec_['k5']['passed'] else 'NOT MET'} {json.dumps(rec_['k5'], default=T._jsonable)}")
    rec_["finished"] = True
    jf.write_text(json.dumps(rec_, indent=1, default=T._jsonable) + "\n")
    log(f"END it {rec_['iterations']} {rec_['message']} {rec_['end_state']} gate {rec_['end']['gate']:.3e} U {rec_['end']['U']:.10f}")


if __name__ == "__main__":
    main()
