"""M5.56: the path tables at run time, the control C4, the words of sections 5, 11 and 12, the statements R1, R2 and
R4, the scoring of OW56-1, OW56-3 and of the expectations, the barrier lines, the summary JSON and the plots.

Every word is given by the frozen rule of the plan as written; this script reads the chain records of
m5_56_relax.py (checkpoints/m5_56/chains/) and the stored fields, and computes nothing that a rule does not name.

Usage:
  python m5_56_reads.py --paths TAG       TAG = H1, H2, H3, H1d, H2d, P (fixed fields; written to checkpoints/m5_56/paths_TAG.json)
  python m5_56_reads.py --c4              the word rule of C4 on fixed fields and the null chains
  python m5_56_reads.py --summary         the words, the statements, data/m5_56_summary.json and the plots
"""

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_51_action as A  # noqa: E402
import m5_51_reads as RD  # noqa: E402
import m5_56_turn as T  # noqa: E402
import m5_56_relax as RX  # noqa: E402

CK = T.CK
WORK = RX.WORK
DATA = HERE.parent / "data"
PLOTS = HERE.parent / "plots"
PI = np.pi


def rec(name):
    return RX.load_rec(name)


def srec(name):
    """A record of the supplement outside the frozen plan (checkpoints/m5_56/supp/)."""
    f = RX.SUPP / f"{name}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return d if d.get("finished") else None


def sname(row):
    return f"s{row:02d}_{RX.ROWS[row]['name']}"


def _dir(r):
    return RX.SUPP if r.get("supplement") else WORK


def field(name, file="end.npz"):
    z = np.load(WORK / name / file, allow_pickle=False)
    return np.asarray(z["m6"], float)


# ----------------------------------------------------------------------------- the path tables at run time

def paths(tag):
    out = _paths(tag)
    (CK / f"paths_{tag}.json").write_text(json.dumps(out, indent=1, default=T._jsonable) + "\n")
    return out


def _paths(tag):
    out = dict(tag=tag)
    if tag == "H1":
        m6, x, ring, e = T.H1_reference()
        mod = T.make_model("H1")
        out["reference"] = "G6_mid_a"
        tabs = [("normal", ring["nu"], 11.0, 2.0), ("inplane", T.inplane_axis(ring["nu"]), 11.0, 2.0)]
    elif tag in ("H2", "H3"):
        row = 2 if tag == "H2" else 3
        if not RX.qualifies(row):
            return dict(tag=tag, not_run="no reference")
        m6 = field(RX.cname(row))
        x = T.axis_of(tag)
        ring = T.ring_of(rec(RX.cname(row))["end"])
        mod = T.make_model(tag)
        out["reference"] = RX.cname(row)
        ip = T.inplane_axis(ring["nu"])
        if tag == "H2":
            tabs = [("normal", ring["nu"], 11.0, 4.0), ("inplane", ip, 11.0, 4.0)]
        else:
            tabs = [("normal", ring["nu"], 11.0, 2.0), ("inplane", ip, 11.0, 2.0), ("normal_wide", ring["nu"], 14.0, 4.0), ("inplane_wide", ip, 14.0, 4.0)]
    elif tag in ("H1d", "H2d"):
        row = RX.h1d_ref_row() if tag == "H1d" else (24 if RX.qualifies(24) else None)
        if row is None:
            return dict(tag=tag, not_run="no reference")
        g = "H1" if tag == "H1d" else "H2"
        m6 = field(RX.cname(row))
        x = T.axis_of(g)
        ring = T.ring_of(rec(RX.cname(row))["end"])
        mod = T.make_model(g, T.DELTA, hold6=T.seed_bi(x))
        out["reference"] = RX.cname(row)
        tabs = [("normal", ring["nu"], 11.0, 2.0), ("inplane", T.inplane_axis(ring["nu"]), 11.0, 2.0)]
        R1 = ring["RL"] + 1.5
        U0 = mod.evaluate(m6, False)["U"]
        out["W"] = {lab: mod.evaluate(T.W_turn(m6, m6, x, Th, ring["c"], R1, 11.0), False)["U"] - U0 for lab, Th in (("W2pi", 2 * PI), ("Wpi", PI))}
        out["W_regime"] = T.W_regime(m6, x, ring["c"], R1, 11.0)
    elif tag == "P":
        wide, narrow = RX.pair_refs()
        out["ends"] = {}
        x = T.axis_of("P")
        mod = T.make_model("P", T.DELTA, hold6=T.pair_bi(x, 14.0))
        for lab, row in (("wide", wide), ("narrow", narrow)):
            if row is None:
                continue
            m6 = field(RX.cname(row))
            e = rec(RX.cname(row))["end"]["pair"]
            r1, r2 = e["rings"][0], e["rings"][1]
            c1 = np.asarray(r1["c"])
            sep = float(np.linalg.norm(c1 - np.asarray(r2["c"])))
            R1 = r1["RL"] + 1.5
            R2 = min(sep - r2["RL"] - 1.5, min(18.0 - abs(c1[a]) for a in range(3)) - 1.0)
            o = dict(row=row, separation=sep, R1=R1, R2=R2)
            if R2 - R1 >= 4.0:
                nu = T.unit(r1["nu"])
                for an, a in (("normal", nu), ("inplane", T.inplane_axis(nu))):
                    o[an] = T.path_table(mod, m6, x, a, c1, R1, R2, top=2.0, sign=-1.0)
            else:
                o["not_read"] = "R2 - R1 under 8h (start rule (i))"
            out["ends"][lab] = o
        if not out["ends"]:
            out["not_run"] = "no two-ring state"
        return out
    else:
        raise ValueError(tag)
    R1 = ring["RL"] + 1.5
    out["R1"] = R1
    out["tables"] = {}
    for an, a, R2, top in tabs:
        pt = T.path_table(mod, m6, x, a, ring["c"], R1, R2, top=top)
        if top == 4.0:
            sub = [r["dU"] for r in pt["rows"] if r["psi_over_pi"] <= 2.0 + 1e-12]
            pt["word_from_2pi"] = "MONOTONE" if all(sub[i + 1] < sub[i] for i in range(len(sub) - 1)) else "BARRIER"
        pt["R2"] = R2
        out["tables"][an] = pt
    return out


# ----------------------------------------------------------------------------- shared reads

def masks(r):
    """free nodes (faces and the held ball out) and the tube of the row's reference rings."""
    g = r["grid"]
    x = T.axis_of(g)
    N = len(x)
    free = ~T.faces_mask(N)
    X = T.mesh(x)
    if r.get("held") == "ball":
        cl = r["info"]["collar"]
        free &= np.linalg.norm(X - np.asarray(cl["c"]), axis=-1) > cl["R1"]
    rings = [dict(c=np.asarray(q["c"]), nu=np.asarray(q["nu"]), RL=q["RL"]) for q in r.get("rings", [])]
    if not rings:
        e = r["end"]
        if (e.get("pair") or {}).get("rings"):
            # the fitted line(s) of the end's two-ring reads, TWO RINGS or ONE OBJECT (the tube about the object's line)
            rings = [dict(c=np.asarray(q["c"]), nu=np.asarray(q["nu"]), RL=q["RL"]) for q in e["pair"]["rings"]]
        elif e.get("RL") is not None and e.get("center") is not None:
            rings = [dict(c=np.asarray(e["center"]), nu=np.asarray(e["normal"]), RL=e["RL"])]
    tube = T.tubes(x, rings) if rings else np.zeros((N, N, N), bool)
    return x, free, tube


def ref_reads(r):
    """The reads of the reference the row is compared with."""
    if r.get("reference") == "G6_mid_a":
        return T.load51("G6_mid_a")[2]
    return rec(r["reference"])["end"]


def rings_of(e):
    if e.get("pair") is not None:
        p = e["pair"]
        return p["rings"] if p.get("word") == "TWO RINGS" else None
    if e.get("RL") is None:
        return None
    return [dict(RL=e["RL"], c=e["center"], nu=e["normal"], cls=e.get("cls"))]


def readers_agree(r, e, ref, delta, mu_val=None):
    """Section 5, with the minor clause of section 10 at delta > 0. Returns (agree, clauses)."""
    cl = {}
    cl["U"] = bool(abs(e["U"] / r["U_ref"] - 1) <= 1e-5)
    if r["grid"] == "P":
        cl["two_rings"] = bool(e.get("pair", {}).get("word") == "TWO RINGS")
        if cl["two_rings"]:
            a, b = e["pair"]["rings"], ref["pair"]["rings"]
            cl["RL"] = bool(all(abs(p["RL"] - q["RL"]) <= 0.01 for p, q in zip(a, b)))
            cl["center"] = bool(all(np.linalg.norm(np.asarray(p["c"]) - np.asarray(q["c"])) <= 0.05 for p, q in zip(a, b)))
            cl["class"] = bool(all(p["cls"] == q["cls"] for p, q in zip(a, b)))
            cl["degree"] = bool(all(p["cube_q"].get("abs_q") == q["cube_q"].get("abs_q") for p, q in zip(a, b))
                                and e["pair"]["box_q"].get("abs_q") == ref["pair"]["box_q"].get("abs_q"))
        else:
            for k in ("RL", "center", "class", "degree"):
                cl[k] = False
    else:
        ok = e.get("RL") is not None and ref.get("RL") is not None
        cl["RL"] = bool(ok and abs(e["RL"] - ref["RL"]) <= 0.01)
        cl["center"] = bool(ok and np.linalg.norm(np.asarray(e["center"]) - np.asarray(ref["center"])) <= 0.05)
        cl["class"] = bool(e.get("cls") == ref.get("cls"))
        cl["degree"] = bool(e.get("degree", {}).get("values") == ref.get("degree", {}).get("values"))
        cl["one_ring_in_regime"] = bool(e.get("one_ring") and e.get("line_regime"))
    if delta > 0:
        m = e.get("mu_free") if mu_val is None else mu_val
        cl["mu_free"] = bool(m is not None and m["in_regime"] and m["mu"] <= 0.03)
    return bool(all(cl.values())), cl


def second_step(r, k5r):
    """Section 4 (and section 10 at delta > 0): the reads at the end against the earlier state."""
    e = r["end"]
    er = r.get("earlier")
    src = "earlier" if er is not None else None
    efile = _dir(r) / r["name"] / "end.npz"
    if er is None:
        if r["row"] in (8, 9):
            return dict(ok=False, reason="no saved stop 100 or more before the end, and no K5 (rows 8, 9): no word")
        if k5r is None or "end" not in k5r:
            return dict(ok=False, reason="no earlier state and no K5 end")
        er = k5r["end"]
        src = "the end of the chain's K5"
        efile2 = _dir(r) / k5r["name"] / "end.npz"
    else:
        efile2 = _dir(r) / r["name"] / er["file"]
    out = dict(source=src, earlier_iteration=er.get("iteration"))
    ch = {}
    ch["U"] = bool(abs(e["U"] / er["U"] - 1) <= 1e-4)
    ra, rb = rings_of(e), rings_of(er)
    if ra is None and rb is None and r["grid"] == "P":
        ch["RL"] = True
    else:
        ch["RL"] = bool(ra is not None and rb is not None and len(ra) == len(rb) and all(abs(p["RL"] - q["RL"]) <= 0.001 for p, q in zip(ra, rb)))
    if "turn" in e:
        ch["sigma"] = bool(e["turn"]["sigma"] == er["turn"]["sigma"])
        ch["chi_side"] = bool(all((e["turn"]["chi_max"] <= b) == (er["turn"]["chi_max"] <= b) for b in (0.05, PI / 2)))
    if r["grid"] == "P":
        ch["pair_word"] = bool(e["pair"]["word"] == er["pair"]["word"])
    else:
        ch["class"] = bool(e.get("cls") == er.get("cls"))
        ch["degree"] = bool(e.get("degree", {}).get("values") == er.get("degree", {}).get("values"))
    if r["delta"] > 0:
        x, free, tube = masks(r)
        m = T.mu(np.load(efile, allow_pickle=False)["m6"], np.load(efile2, allow_pickle=False)["m6"], free & ~tube)
        out["mu"] = m
        ch["mu"] = bool(m["in_regime"] and m["mu"] <= 0.01)
        if "minor" in e:
            a0, a1 = e["minor"]["alpha_s"], er["minor"]["alpha_s"]
            ch["alpha_s"] = bool(abs(a0 - a1) <= 0.1 and all((abs(a0) <= b) == (abs(a1) <= b) for b in (0.1, PI)))
    out["checks"] = ch
    out["ok"] = bool(all(ch.values()))
    out["earlier"] = er
    return out


def tear_read(s, rung):
    t = s.get("tear")
    if t is None:
        return False
    return bool(t["tear"])


def release_stop(r):
    """The first route stop with sigma = +1 in regime (2pi starts) or chi_max <= pi/2 in regime (4pi starts)."""
    Th = abs(r["info"]["collar"]["Theta"])
    for s in r["stops"]:
        if "turn" not in s or not s["turn"]["in_regime"]:
            continue
        if (Th < 3 * PI and s["turn"]["sigma"] == 1) or (Th > 3 * PI and s["turn"]["chi_max"] <= PI / 2):
            return s["iteration"]
    return None


def route_words(r):
    """Section 5, the route words, over the route stops from the start to the release stop (or the end)."""
    rel = release_stop(r)
    stops = [s for s in r["stops"] if rel is None or s["iteration"] <= rel]
    stops = sorted({s["iteration"]: s for s in stops}.values(), key=lambda q: q["iteration"])
    vac = [s["iteration"] for s in stops if s.get("tear", {}).get("other_components_outside", 0) > 0]
    left = [s["iteration"] for s in stops if s.get("tear", {}).get("ring_nodes_outside", 0) > 0]
    link = [s["iteration"] for s in stops if s.get("tear", {}).get("jump_links", 0) > 0 and s["tear"]["z_nodes_outside"] == 0]
    tears = []
    if vac:
        tears.append("TORN IN THE VACUUM")
    if left:
        tears.append("RING LEFT ITS TUBE")
    if link:
        tears.append("TORN ON A LINK")
    out = dict(release_stop=rel, stops_read=[s["iteration"] for s in stops], vacuum_stops=vac, ring_left_stops=left, link_stops=link)
    if tears:
        out["word"] = " and ".join(tears)
        return out
    if r["rung"] == 1 and rel is not None:
        cl = r["info"]["collar"]
        Rs = stops[-1]["turn"]["sign_shell"]
        f_s = float(T.fprof(Rs, cl["R1"], cl["R2"]))
        ints = [s.get("interior") for s in stops]
        all_reg = all(q is not None and q["in_regime"] for q in ints)
        psi0, psir = ints[0]["psi_int"] if ints[0] else None, ints[-1]["psi_int"] if ints[-1] else None
        thr = PI / f_s
        out.update(threshold=thr, psi_start=psi0, psi_release=psir, interior_all_in_regime=all_reg,
                   psi_monotone=bool(all(ints[i + 1]["psi_int"] <= ints[i]["psi_int"] + 1e-12 for i in range(len(ints) - 1))) if all(ints) else None)
        if all_reg and psir is not None and psir < psi0 and psir < thr:
            out["word"] = "THE CORE TURNED BACK"
            return out
        before = [s for s in stops if s["iteration"] < rel]
        if not all_reg and before and before[-1]["turn"].get("rho_t_cross") is not None and before[-1]["turn"]["rho_t_cross"] <= Rs + 1.0:
            out["word"] = "AT THE CORE"
            return out
    out["word"] = "route not read"
    return out


def any_tear(r, upto=None):
    return [s["iteration"] for s in r["stops"] if (upto is None or s["iteration"] <= upto) and s.get("tear", {}).get("tear")]


LENIENT_K5 = False


def k5_state(row, supp=False):
    """K5 is met only when its own relaxation ends CONVERGED by the rule of its row AND its listed clauses hold (the
    stricter reading, ruled at the fold). With LENIENT_K5, the listed clauses alone (kept as a separate summary key)."""
    k = srec("k5_" + sname(row)) if supp else rec("k5_" + RX.cname(row))
    if k is None:
        return None, "not run"
    if not k["k5"]["passed"]:
        return k, "not met"
    if k.get("end_state") != "CONVERGED" and not LENIENT_K5:
        return k, f"not met (its relaxation ended {k.get('end_state')}, gate {k['end']['gate']:.2e}; every listed clause held)"
    return k, "met"


def chain_word(row, c4=None, h1_ball=None, supp=False):
    """The word of one chain end by the frozen rules (section 5; section 11 for W rows; section 12 on the pair). With
    supp, the rules READ on a supplement row (outside the frozen plan; no registered score)."""
    name = sname(row) if supp else RX.cname(row)
    r = srec(name) if supp else rec(name)
    if r is None:
        return dict(row=row, name=name, word="not finished")
    if r.get("end_state") == "NOT RUN":
        return dict(row=row, name=name, word=r["reason"], start_rule=r.get("start_rule"))
    k5r, k5s = k5_state(row, supp)
    e = r["end"]
    out = dict(row=row, name=name, chain=r["chain"], grid=r["grid"], delta=r["delta"], rung=r["rung"], end_state=r["end_state"], iterations=r["iterations"],
               message=r["message"], k5=k5s, U=e["U"], gate=e["gate"], torque=e.get("torque"), restarts=len(r["restarts"]), resumes=len(r["resumes"]),
               start_rule=r.get("start_rule"))
    if r.get("U_ref") is not None:
        out.update(U_ref=r["U_ref"], dU=e["U"] - r["U_ref"], dU_start=r["stops"][0]["U"] - r["U_ref"])
    if r["rung"] in ("reference", "null"):
        out["word"] = None
    ss = second_step(r, k5r)
    out["second_step"] = {k: v for k, v in ss.items() if k != "earlier"}
    why = []
    if r["end_state"] != "CONVERGED":
        why.append("CAPPED")
    if not ss["ok"]:
        why.append("the second step: " + (ss.get("reason") or ", ".join(k for k, v in ss["checks"].items() if not v)))
    if row not in (8, 9) and k5s != "met":
        why.append(f"K5 {k5s}")
    kind = r["info"].get("collar", {}).get("kind")
    if "turn" in e and kind == "T" and not e["turn"]["in_regime"]:
        why.append("the turn reader out of regime at the end")
    if kind == "W":
        if not e.get("minor", {}).get("in_regime"):
            why.append("the minor turn profile out of regime at the end")
        if not (e.get("mu_free") or {}).get("in_regime"):
            why.append("mu out of regime at the end")
        if r["grid"] != "P" and not (e.get("one_ring") and e.get("line_regime")):
            why.append("the ring's reads out of regime at the end")
        if r["grid"] == "P" and not (e.get("pair", {}).get("word") == "TWO RINGS"):
            why.append("the ring's reads out of regime at the end (not TWO RINGS)")
    out["why_no_label"] = why
    if r["rung"] == "pair":
        out["pair"] = e.get("pair")
        out["word"] = "no label" if why else e["pair"]["word"]
        return out
    if r["rung"] == "reference":
        out["qualifies"] = bool(not why and (e.get("one_ring") and e.get("line_regime") and e.get("degree", {}).get("reads_one")))
        out["word"] = None
        return out
    ref = ref_reads(r)
    agree, cl = readers_agree(r, e, ref, r["delta"])
    out["readers_agree"] = agree
    out["readers_clauses"] = cl
    if e.get("normal") is not None and ref.get("normal") is not None:
        out["normal_angle_to_reference_deg"] = RD.angle(e["normal"], ref["normal"])
        out["center_offset"] = float(np.linalg.norm(np.asarray(e["center"]) - np.asarray(ref["center"])))
    t = e.get("turn", {})
    out.update(sigma=t.get("sigma"), chi_max=t.get("chi_max"), rho_t_cross=t.get("rho_t_cross"), rho_least_q0=t.get("rho_least_q0"),
               q0_sign_shell=t.get("q0_sign_shell"), turn_in_regime=t.get("in_regime"))
    if "minor" in e:
        out.update(alpha_s=e["minor"]["alpha_s"], minor_in_regime=e["minor"]["in_regime"])
    if "mu_free" in e:
        out.update(mu_free=e["mu_free"]["mu"], mu_col=e.get("mu_col", {}).get("mu"))
    if r["rung"] == "null":
        out["word"] = "no label" if why else ("RELEASED" if (agree and t.get("sigma") == 1 and t.get("chi_max") <= 0.05) else "MOVED")
        out["null_C4"] = dict(U_rel=abs(e["U"] / r["U_ref"] - 1), within_1e6=bool(abs(e["U"] / r["U_ref"] - 1) <= 1e-6), chi_max=t.get("chi_max"))
        return out
    Th = r["info"]["collar"]["Theta"]
    four = abs(Th) > 3 * PI
    er = ss.get("earlier")
    rel = release_stop(r)
    tears = any_tear(r, rel)
    out["tear_stops"] = tears
    out["release_stop"] = rel
    if kind == "W":
        a_s = e["minor"]["alpha_s"]
        if why:
            word = "no label"
        elif agree and abs(a_s) <= 0.1:
            word = "RELEASED"
        else:
            keep_cl = {k: v for k, v in cl.items() if k not in ("U", "mu_free")}
            kept = (abs(a_s) >= PI and er is not None and "minor" in er and abs(er["minor"]["alpha_s"]) >= PI and e["minor"]["in_regime"]
                    and er["minor"]["in_regime"] and all(keep_cl.values()))
            word = "KEPT" if kept else "MOVED"
        out["word"] = word
        out["kept_clauses_but_U_mu"] = {k: v for k, v in cl.items() if k not in ("U", "mu_free")}
        return out
    if why:
        out["word"] = "no label"
    elif agree and t["sigma"] == 1 and t["chi_max"] <= 0.05:
        out["word"] = "RELEASED"
    else:
        held = False
        if er is not None and "turn" in er:
            want = 1 if four else -1
            tr_ok = all(q["turn"]["chi_max"] >= PI / 2 and q["turn"]["sigma"] == want for q in (e, er))
            reg = "free" if r["held"] == "ball" else "outside the tube"
            no_tear = all(q["tear"]["z_nodes_outside"] == 0 and q["tear"]["jump_links"] == 0 for q in (e, er))
            held = bool(tr_ok and no_tear)
            out["held_check"] = dict(chi_sigma=tr_ok, no_tear=no_tear, region=reg)
        if held:
            out["word"] = "HELD"
        elif tears:
            out["word"] = "TORN"
        else:
            out["word"] = "MOVED"
            if four and t.get("sigma") == -1:
                out["note"] = "a 2pi remnant"
    # rung 2 renames
    if r["rung"] == 2 and out["word"] == "RELEASED":
        if tears:
            out["word"] = "TORN"
            out["note"] = "readers agree with the reference, but a tear was read before the release (TORN, the stricter reading)"
        else:
            out["word"] = "UNWOUND" if four else "RELEASED WITH NO TEAR READ"
    if out["word"] == "TORN" or r["rung"] in (1, 4, 5) or out["word"] in ("RELEASED WITH NO TEAR READ", "UNWOUND"):
        out["route"] = route_words(r)
        if r["rung"] == 2 and out["word"] == "TORN":
            rw = out["route"]["word"]
            out["tear_word"] = rw if rw in ("TORN IN THE VACUUM", "TORN ON A LINK", "TORN IN THE VACUUM and TORN ON A LINK") else rw
    return out


# ----------------------------------------------------------------------------- C4

def c4():
    res = dict(control="C4 the null chain and the word rule")
    m6, x, ring, e = T.H1_reference()
    me, _, ee = T.load51("G6_end_a")
    mod = T.make_model("H1")
    U_ref = mod.evaluate(m6, False)["U"]
    R1 = ring["RL"] + 1.5
    out = []
    for nm, f6 in (("G6_end_a", me), ("T[2pi] start", T.T_turn(m6, x, 2 * PI, ring["nu"], ring["c"], R1, 11.0))):
        rd = RD.read_state(f6, x, 12.0, "mid", with_grad=False)
        tr = T.turn_reader(f6, m6, x, ring["c"], R1, axis=ring["nu"])
        fake = dict(U_ref=U_ref, grid="H1")
        rd["U"] = mod.evaluate(f6, False)["U"]
        agree, cl = readers_agree(fake, rd, e, 0.0)
        if agree and tr["sigma"] == 1 and tr["chi_max"] <= 0.05:
            w = "RELEASED"
        elif tr["chi_max"] >= PI / 2 and tr["sigma"] == -1:
            w = "HELD (no tear read on a fixed field)"
        else:
            w = "MOVED"
        ring_alone = bool(cl["RL"] and cl["center"] and cl["class"] and cl["degree"] and cl["one_ring_in_regime"])
        out.append(dict(input=nm, word=w, readers_clauses=cl, sigma=tr["sigma"], chi_max=tr["chi_max"], U_rel=rd["U"] / U_ref - 1,
                        ring_alone_rule="RELEASED" if ring_alone else "not RELEASED"))
    res["fixed_fields"] = out
    res["word_rule_ok"] = bool(out[0]["word"] == "MOVED" and out[1]["word"] != "RELEASED")
    res["failing_rule_ok"] = bool(out[1]["ring_alone_rule"] == "RELEASED")
    # the null chains: row 1 (H1), the K5 of B2 (H2) and of B3 (H3)
    nulls = {}
    r1 = rec(RX.cname(1))
    if r1 is not None and r1.get("end_state") != "NOT RUN":
        w = chain_word(1)
        nulls["H1"] = dict(source=RX.cname(1), end_state=r1["end_state"], word=w["word"], U_rel=w["null_C4"]["U_rel"], chi_max=w["null_C4"]["chi_max"],
                           met=bool(r1["end_state"] == "CONVERGED" and w["null_C4"]["within_1e6"] and w["word"] == "RELEASED" and w["null_C4"]["chi_max"] <= 0.05),
                           why_no_label=w["why_no_label"])
    for g, row in (("H2", 2), ("H3", 3)):
        k = rec("k5_" + RX.cname(row))
        b = rec(RX.cname(row))
        if k is None or b is None:
            continue
        xg = T.axis_of(g)
        mb = field(RX.cname(row))
        mk = field(k["name"])
        ringb = T.ring_of(b["end"])
        tr = T.turn_reader(mk, mb, xg, ringb["c"], ringb["RL"] + 1.5, axis=ringb["nu"])
        fake = dict(U_ref=b["end"]["U"], grid=g)
        agree, cl = readers_agree(fake, k["end"], b["end"], 0.0)
        word = "RELEASED" if (agree and tr["sigma"] == 1 and tr["chi_max"] <= 0.05) else "MOVED"
        urel = abs(k["end"]["U"] / b["end"]["U"] - 1)
        nulls[g] = dict(source=k["name"], end_state=k["end_state"], word=word, U_rel=urel, chi_max=tr["chi_max"], sigma=tr["sigma"], turn_in_regime=tr["in_regime"],
                        met=bool(urel <= 1e-6 and word == "RELEASED" and tr["chi_max"] <= 0.05), readers_clauses=cl)
    res["null_chains"] = nulls
    res["passed"] = bool(res["word_rule_ok"] and all(v["met"] for v in nulls.values()))
    (CK / "control_C4.json").write_text(json.dumps(res, indent=1, default=T._jsonable) + "\n")
    return res


# ----------------------------------------------------------------------------- statements, scores, the summary

def path_word(tag, key="word"):
    f = CK / f"paths_{tag}.json"
    if not f.exists():
        return None
    p = json.loads(f.read_text())
    if p.get("not_run"):
        return dict(not_run=p["not_run"])
    return {an: t[key] if key in t else t["word"] for an, t in p.get("tables", {}).items()}


def controls_summary():
    out = {}
    for f in sorted(CK.glob("control_*.json")):
        d = json.loads(f.read_text())
        nm = f.stem.replace("control_", "")
        base, _, tr = nm.rpartition("_try")
        if not _:
            base, tr = nm, "1"
        out.setdefault(base, {})[f"try{tr}"] = bool(d.get("passed"))
    return out


def statements(W):
    """R1, R2, R4; OW56-1, OW56-3; the expectations; the barrier lines."""
    S = {}
    pw = {t: path_word(t) for t in ("H1", "H2", "H3")}
    mono = {t: (v is not None and not v.get("not_run") and all(w == "MONOTONE" for w in v.values())) for t, v in pw.items()}
    r67 = [W.get(6, {}).get("word"), W.get(7, {}).get("word")]
    S["R1"] = dict(word="SHED" if (all(mono.values()) and r67 == ["RELEASED", "RELEASED"]) else "tabled", path_tables=pw, monotone=mono, rows_6_7=r67,
                   routes={r: W.get(r, {}).get("route_final") for r in (6, 7, 8, 9, 10)})
    # R2
    w2 = {r: W.get(r, {}).get("word") for r in (11, 13, 16, 17, 18)}
    grid_of = {11: "H1", 13: "H2", 16: "H3"}
    with_word = {r: w for r, w in w2.items() if r in (11, 13, 16) and w not in (None, "no label") and not str(w).startswith("not run")}
    stays = bool(with_word and all(w == "HELD" for w in with_word.values()) and {grid_of[r] for r in with_word} == {"H1", "H2", "H3"})
    if stays:
        word = "STAYS"
        if any(w2.get(r) == "HELD" for r in (17, 18)):
            word += " (the descent also holds a trivial turn)"
    elif w2.get(11) == "TORN" and w2.get(13) not in (None, "no label") and w2.get(13) != "TORN" and not str(w2.get(13)).startswith("not run"):
        word = "TORN AT 0.5 ONLY"
    elif w2.get(11) == "TORN" and w2.get(13) == "no label":
        word = "tabled, no statement (row 13 carries no label)"
    else:
        word = "tabled"
    S["R2"] = dict(word=word, words=w2)
    # OW56-1
    want = {11: "HELD", 13: "HELD", 16: "HELD", 17: "UNWOUND", 18: "UNWOUND"}
    nolab = [r for r in want if w2.get(r) in (None, "no label") or str(w2.get(r)).startswith("not run")]
    fail = [r for r in want if r not in nolab and w2.get(r) != want[r]]
    S["OW56-1"] = dict(score="not met" if fail else ("met" if not nolab else "not scored"), failing_rows=fail, rows_with_no_label=nolab, words=w2)
    # R4
    w4 = {r: W.get(r, {}).get("word") for r in (25, 26, 27, 28, 29)}
    pd = path_word("H1d")
    mono4 = bool(pd is not None and not pd.get("not_run") and all(v == "MONOTONE" for v in pd.values()))
    if w4[28] == "KEPT" and w4[29] == "KEPT":
        r4 = "A TWIST KEPT AT delta = 0.05"
    elif w4[28] == "KEPT" and w4[29] not in (None, "no label", "KEPT") and not str(w4[29]).startswith("not run"):
        r4 = "KEPT AT 0.5 ONLY"
    elif w4[28] == "KEPT" and (w4[29] in (None, "no label") or str(w4[29]).startswith("not run")):
        r4 = "tabled, no statement (row 28 KEPT, row 29 no label)"
    elif mono4 and w4[25] == "RELEASED" and w4[26] == "RELEASED" and w4[28] == "RELEASED" and w4[29] != "KEPT":
        r4 = "SHED AT delta = 0.05"
    else:
        r4 = "tabled"
    S["R4"] = dict(word=r4, words=w4, H1d_path_tables=pd, H2d_path_tables=path_word("H2d"))
    first = bool(w4[25] == "RELEASED" and w4[26] == "RELEASED" and w4[28] == "RELEASED" and w4[29] != "KEPT")
    if first:
        ow3 = "met (first clause)"
    elif w4[28] == "KEPT" and w4[29] == "KEPT":
        ow3 = "KEPT WITH THE LINE ENDS HELD ON THE FACES: not scored, the second clause excepts a held minor gap"
    else:
        ow3 = "not scored"
    S["OW56-3"] = dict(score=ow3, words=w4)
    S["OW56-2"] = dict(score="not scored: rung 3 is not run")
    return S


def pair_summary(W, E_core, supp=False):
    rows = {}
    ends = []
    for r in (30, 31, 32, 33):
        w = W.get(r, {})
        rr = rec(RX.cname(r))
        if rr is None or rr.get("end_state") == "NOT RUN":
            continue
        p = rr["end"].get("pair", {})
        k5r, k5s = k5_state(r)
        rows[r] = dict(word=w.get("word"), end_state=rr["end_state"], k5=k5s, U=rr["end"]["U"], separation=p.get("separation"),
                       radii=[q["RL"] for q in p.get("rings", [])], normals=[q["nu"] for q in p.get("rings", [])], D=p.get("D"),
                       components=p.get("components"), sizes=p.get("sizes"), circle_distance=p.get("circle_distance"),
                       record=[dict(iteration=s["iteration"], U=s["U"], separation=s.get("pair", {}).get("separation"), word=s.get("pair", {}).get("word"))
                               for s in rr["stops"] if s.get("major")],
                       lines=rr["end"].get("lines"))
        if rr["end_state"] == "CONVERGED" and k5s == "met":
            ends.append(rr["end"]["U"])
    E_min = min(ends) if ends else None
    for r, q in rows.items():
        q["U_minus_Emin"] = None if E_min is None else q["U"] - E_min
        q["U_minus_Emin_in_Ecore"] = None if (E_min is None or E_core is None) else (q["U"] - E_min) / E_core
    # barrier lines
    lines = []
    for tr, wr in ((34, 36), (35, 37)):
        a, b = W.get(tr, {}), W.get(wr, {})
        ra = (srec(sname(tr)) or rec(RX.cname(tr))) if supp else rec(RX.cname(tr))
        if ra is None or ra.get("end_state") == "NOT RUN" and str(ra.get("reason", "")).startswith("not run: no two-ring") or a.get("word") is None:
            continue
        ref = ra.get("reference")
        refrow = int(ref[1:3]) if ref else None
        pp = json.loads((CK / "paths_P.json").read_text()) if (CK / "paths_P.json").exists() else {}
        lab = "wide" if tr == 34 else "narrow"
        pt = pp.get("ends", {}).get(lab, {})
        mono = bool(pt.get("normal", {}).get("word") == "MONOTONE" and pt.get("inplane", {}).get("word") == "MONOTONE")
        line = dict(T_row=tr, W_row=wr, reference=ref, separation=None if refrow is None else rows.get(refrow, {}).get("separation"),
                    T_word=a.get("word"), W_word=b.get("word"), T_path_monotone=mono)
        if mono and a.get("word") == "RELEASED" and b.get("word") == "RELEASED":
            line["line"] = "the relative turn is not a state at this separation: no barrier is read; the start's dU is a lattice cost"
        elif b.get("word") == "KEPT":
            rb = (srec(sname(wr)) or rec(RX.cname(wr))) if supp else rec(RX.cname(wr))
            Eu = rows.get(refrow, {}).get("U")
            line["E_turned_minus_Emin"] = rb["end"]["U"] - E_min if E_min is not None else None
            line["E_unturned_minus_Emin"] = Eu - E_min if (E_min is not None and Eu is not None) else None
            if E_core:
                line["in_Ecore"] = [None if v is None else v / E_core for v in (line["E_turned_minus_Emin"], line["E_unturned_minus_Emin"])]
            line["line"] = "W KEPT with free cores: the two energies are printed; a ladder with a hold on d is the next task"
        else:
            line["line"] = "tabled with no line"
        lines.append(line)
    return dict(rows=rows, E_min=E_min, E_core=E_core, barrier_lines=lines)


def expectations(W, S, P):
    def w(r):
        return W.get(r, {}).get("word")
    out = []
    pw = S["R1"]["path_tables"]
    x1 = [pw.get("H2"), pw.get("H3")]
    out.append(dict(id="X1", text="the path tables of H2, H3 (and the pair end, not run) read MONOTONE", confidence=85,
                    outcome="right" if all(S["R1"]["monotone"].get(t) for t in ("H2", "H3")) else "wrong", detail=x1))
    x2 = (w(6) == "RELEASED" and w(7) == "RELEASED" and W.get(6, {}).get("route_final") == "THE CORE TURNED BACK"
          and W.get(7, {}).get("route_final") == "THE CORE TURNED BACK")
    out.append(dict(id="X2", text="rows 6, 7 read RELEASED, route THE CORE TURNED BACK, matched by their CG twins", confidence=55,
                    outcome="right" if x2 else "wrong", detail=[w(6), w(7), W.get(6, {}).get("route_final"), W.get(7, {}).get("route_final")]))
    torn_h2 = any(W.get(r, {}).get("tear_stops") for r in (13, 17))
    x3 = S["R2"]["word"].startswith("STAYS") and not torn_h2
    out.append(dict(id="X3", text="statement R2 reads STAYS, with no tear read on H2", confidence=60, outcome="right" if x3 else "wrong",
                    detail=[S["R2"]["word"], torn_h2]))
    out.append(dict(id="X4", text="the two 4pi probes read UNWOUND", confidence=50, outcome="right" if (w(17) == "UNWOUND" and w(18) == "UNWOUND") else "wrong",
                    detail=[w(17), w(18)]))
    out.append(dict(id="X5", text="Pa and Pb both read TWO RINGS", confidence=50, outcome="not scored (rows 4, 5 not run)"))
    out.append(dict(id="X6", text="rows 19 and 20 both read HELD", confidence=40, outcome="not scored (rows 19, 20 not run)"))
    pd, p2 = S["R4"]["H1d_path_tables"], S["R4"]["H2d_path_tables"]
    x7 = all(p is not None and not p.get("not_run") and all(v == "MONOTONE" for v in p.values()) for p in (pd, p2))
    out.append(dict(id="X7", text="the T path tables at delta = 0.05 read MONOTONE on H1 and H2, on both axes", confidence=95, outcome="right" if x7 else "wrong",
                    detail=[pd, p2]))
    out.append(dict(id="X8", text="rows 25 and 26 read RELEASED", confidence=80, outcome="right" if (w(25) == "RELEASED" and w(26) == "RELEASED") else "wrong",
                    detail=[w(25), w(26)]))
    out.append(dict(id="X9", text="rows 28 and 29 read RELEASED", confidence=60, outcome="right" if (w(28) == "RELEASED" and w(29) == "RELEASED") else "wrong",
                    detail=[w(28), w(29)]))
    out.append(dict(id="X10", text="row 30 ends TWO RINGS", confidence=40, outcome="right" if w(30) == "TWO RINGS" else "wrong", detail=w(30)))
    out.append(dict(id="X11", text="row 33 ends ONE OBJECT", confidence=60, outcome="right" if w(33) == "ONE OBJECT" else "wrong", detail=w(33)))
    for xid, rows_, text, conf in (("X12", (34, 35), "a turned chain on the pair by T, if run, reads RELEASED", 70),
                                   ("X13", (36, 37), "a turned chain on the pair by W, if run, reads RELEASED", 55)):
        ran = [r for r in rows_ if w(r) is not None and not str(w(r)).startswith("not run") and w(r) != "not finished"]
        if not ran:
            out.append(dict(id=xid, text=text, confidence=conf, outcome="not scored (not run)", detail=[w(r) for r in rows_]))
        else:
            out.append(dict(id=xid, text=text, confidence=conf, outcome="right" if all(w(r) == "RELEASED" for r in ran) else "wrong", detail=[w(r) for r in rows_]))
    return out


def all_words():
    W = {}
    for r in RX.ROWS:
        W[r] = chain_word(r)
    # rows 6, 7: the route word stands when the CG twin gives the same chain word and route word
    for a, b in ((6, 8), (7, 9)):
        ra, rb = W[a].get("route", {}).get("word"), W[b].get("route", {}).get("word")
        if ra is None:
            W[a]["route_final"] = None
        elif W[a].get("word") == W[b].get("word") and ra == rb:
            W[a]["route_final"] = ra
        else:
            W[a]["route_final"] = "route not read (two descents differ)"
    W[10]["route_final"] = W[10].get("route", {}).get("word")
    # rung 2: a 2pi RELEASED WITH NO TEAR READ withholds every other word and route word on its grid
    for r in (11, 13, 16):
        if W[r].get("word") == "RELEASED WITH NO TEAR READ":
            g = W[r]["grid"]
            for q, wq in W.items():
                if q != r and wq.get("grid") == g and wq.get("delta") == 0.0 and wq.get("word") not in (None,):
                    wq["withheld"] = f"no word on {g}: row {r} read RELEASED WITH NO TEAR READ (section 5)"
                    wq["word_before_withholding"] = wq["word"]
                    wq["word"] = "no label"
    # C4 on a grid not met: no RELEASED word there
    c4f = CK / "control_C4.json"
    if c4f.exists():
        c4d = json.loads(c4f.read_text())
        for g, v in c4d.get("null_chains", {}).items():
            if not v["met"]:
                for q, wq in W.items():
                    if wq.get("grid") == g and wq.get("delta") == 0.0 and wq.get("word") == "RELEASED":
                        wq["word_before_withholding"] = wq["word"]
                        wq["word"] = "no label"
                        wq["withheld"] = f"C4 not met on {g}: no RELEASED word there"
    return W


def summary():
    W = all_words()
    S = statements(W)
    href = RX.h1d_ref_row()
    E_core = rec(RX.cname(href))["end"]["U"] if href is not None else None
    P = pair_summary(W, E_core)
    X = expectations(W, S, P)
    two = None
    r22, r23 = rec(RX.cname(22)), rec(RX.cname(23))
    if r22 and r23 and r22.get("end") and r23.get("end"):
        e1, e2 = r22["end"], r23["end"]
        x, free, tube = masks(r22)
        m = T.mu(field(RX.cname(22)), field(RX.cname(23)), free & ~tube)
        same = bool(abs(e1["U"] / e2["U"] - 1) <= 1e-5 and e1.get("RL") is not None and e2.get("RL") is not None and abs(e1["RL"] - e2["RL"]) <= 0.01
                    and e1.get("cls") == e2.get("cls") and e1["degree"]["values"] == e2["degree"]["values"] and m["mu"] <= 0.03)
        two = dict(same_state=same, U=[e1["U"], e2["U"]], RL=[e1.get("RL"), e2.get("RL")], cls=[e1.get("cls"), e2.get("cls")], mu_free=m,
                   lines=[e1.get("lines"), e2.get("lines")])
    sup = {}
    for r_ in range(25, 38):
        if srec(sname(r_)) is not None:
            w_ = chain_word(r_, supp=True)
            w_["clause_iv_split"] = srec(sname(r_)).get("supplement_clause_iv")
            sup[r_] = w_
    if sup:
        Wsup = {k: dict(v) for k, v in W.items()}
        for r_, w_ in sup.items():
            Wsup[r_] = w_
        Ssup = statements(Wsup)
        supplement = dict(note="outside the frozen plan: start-rule clause (iv) set aside, for the rows it alone stopped, after the go, while chains of "
                               "rungs 1, 2 and 5 were running or had ended and after the rung-1 chains' ends were on record, and before any supplement "
                               "chain was relaxed; the words are the frozen word rules READ on chains the frozen plan did not admit; no registered score",
                          rows={str(k): _trim_word(v) for k, v in sup.items()},
                          R4_under_the_supplement=Ssup["R4"], OW56_3_under_the_supplement=Ssup["OW56-3"])
        if 34 in sup or 36 in sup:
            supplement["barrier_lines_under_the_supplement"] = pair_summary(Wsup, E_core, supp=True)["barrier_lines"]
    else:
        supplement = None
    # the lenient reading of K5 (its listed clauses alone), kept apart from the statements
    global LENIENT_K5
    LENIENT_K5 = True
    try:
        Wl = all_words()
        Sl = statements(Wl)
        lenient = dict(note="under the reading that K5 is judged on its listed clauses alone (not the statement)",
                       changed_words={str(r): dict(strict=W[r].get("word"), lenient=Wl[r].get("word")) for r in W if W[r].get("word") != Wl[r].get("word")},
                       R1=Sl["R1"]["word"], R2=Sl["R2"]["word"], R4=Sl["R4"]["word"], OW56_1=Sl["OW56-1"]["score"], OW56_3=Sl["OW56-3"]["score"])
        lsup = {}
        for r_ in range(25, 38):
            if srec(sname(r_)) is not None:
                wl = chain_word(r_, supp=True)
                if sup.get(r_, {}).get("word") != wl.get("word"):
                    lsup[str(r_)] = dict(strict=sup[r_].get("word"), lenient=wl.get("word"))
        lenient["changed_supplement_words"] = lsup
    finally:
        LENIENT_K5 = False
    paths_all = {}
    for t in ("H1", "H2", "H3", "H1d", "H2d", "P"):
        f = CK / f"paths_{t}.json"
        if f.exists():
            paths_all[t] = json.loads(f.read_text())
    c4 = json.loads((CK / "control_C4.json").read_text()) if (CK / "control_C4.json").exists() else None
    out = dict(task="M5.56", controls=controls_summary(), C4=c4, path_tables=paths_all,
               chains={str(r): _trim_word(w) for r, w in W.items()}, statements=S, OW56={k: S[k] for k in ("OW56-1", "OW56-2", "OW56-3")},
               R4=S["R4"], rung4=dict(two_states_rows_22_23=two, reference_row=href), rung5=P, E_core=E_core, expectations=X,
               supplement=supplement, lenient_K5_reading=lenient, minor_controls={k: v for k, v in controls_summary().items() if k in ("C3a_d5", "C8_d5", "C9", "C10", "C11", "C12", "C13", "C14", "C15",
                                                                                     "C16", "C17")})
    DATA.mkdir(exist_ok=True)
    f = DATA / "m5_56_summary.json"
    f.write_text(json.dumps(out, indent=1, default=T._jsonable) + "\n")
    plots(W)
    if sup:
        plots({**W, **sup}, supp=True)
    return out


def _trim_word(w):
    w = dict(w)
    if "second_step" in w:
        w["second_step"] = {k: v for k, v in w["second_step"].items() if k != "earlier"}
    return w


# ----------------------------------------------------------------------------- plots

def start_lines(rr, d):
    """The lines read on a chain's start field (the start is a route stop, not a major stop, so the chain did not read them)."""
    f = d / "s00000.npz"
    if not f.exists() or not rr.get("rings"):
        return None
    x = T.axis_of(rr["grid"])
    rg = rr["rings"][0]
    ring_t = T.tubes(x, [dict(c=np.asarray(rg["c"]), nu=np.asarray(rg["nu"]), RL=rg["RL"])])
    radii = (12.0, 14.0, 16.0) if rr["grid"] == "P" else (6.0, 8.0, 10.0)
    c = np.zeros(3) if rr["grid"] == "P" else np.asarray(rg["c"])
    nu = np.array([0, 0, 1.0]) if rr["grid"] == "P" else np.asarray(rg["nu"])
    return T.lines_read(np.load(f, allow_pickle=False)["m6"], x, c, nu, radii, rr["delta"], tube_mask=ring_t)


def plots(W, supp=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                         "grid.linewidth": 0.6})
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#c94f7c", "#7a7a75"]
    # 1. the path tables
    tags = [] if supp else [t for t in ("H1", "H2", "H3", "H1d", "H2d") if (CK / f"paths_{t}.json").exists()]
    if tags:
        fig, axs = plt.subplots(1, len(tags) + 1, figsize=(3.0 * (len(tags) + 1), 2.8))
        for ax, t in zip(axs, tags):
            p = json.loads((CK / f"paths_{t}.json").read_text())
            for i, (an, tb) in enumerate(p.get("tables", {}).items()):
                ax.plot([r["psi_over_pi"] for r in tb["rows"]], [r["dU"] for r in tb["rows"]], marker="o", ms=2.5, color=cols[i], label=f"{an} ({tb['word']})")
            ax.set_title(f"{t}: {p.get('reference', '')}", fontsize=8)
            ax.set_xlabel("psi / pi")
            ax.set_ylabel("dU along T[psi]")
            ax.set_yscale("symlog", linthresh=1.0)
            ax.legend(fontsize=6, frameon=False)
        ax = axs[-1]
        if (CK / "paths_P.json").exists():
            p = json.loads((CK / "paths_P.json").read_text())
            i = 0
            for lab, o in p.get("ends", {}).items():
                for an in ("normal", "inplane"):
                    if an in o:
                        ax.plot([r["psi_over_pi"] for r in o[an]["rows"]], [r["dU"] for r in o[an]["rows"]], marker="o", ms=2.5, color=cols[i],
                                label=f"{lab}, {an} ({o[an]['word']})")
                        i += 1
            if i == 0:
                ax.text(0.5, 0.5, "P: not read\n(no collar about ring 1)", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title("P: T[psi] about ring 1", fontsize=8)
            ax.set_xlabel("psi / pi")
            ax.set_yscale("symlog", linthresh=1.0)
            ax.legend(fontsize=6, frameon=False)
        else:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(PLOTS / "m5_56_path_tables.png", dpi=110)
        plt.close(fig)
    # 2. per chain: the turn profile (q0 against rho) at the stops; dU and the least gap outside the tube against the iteration
    getr = (lambda r: srec(sname(r))) if supp else (lambda r: rec(RX.cname(r)))
    gdir = (lambda r: RX.SUPP / sname(r)) if supp else (lambda r: WORK / RX.cname(r))
    pre = "m5_56_supp_" if supp else "m5_56_"
    trows = [r for r in ((25, 26, 27, 34, 35) if supp else (6, 7, 8, 9, 10, 11, 13, 16, 17, 18, 25, 26, 27, 34, 35))
             if getr(r) and getr(r).get("end_state") != "NOT RUN"]
    if trows:
        n = len(trows)
        fig, axs = plt.subplots(n, 3, figsize=(10, 2.0 * n))
        axs = np.atleast_2d(axs)
        for i, r in enumerate(trows):
            rr = getr(r)
            st = rr["stops"]
            sel = [s for s in st if s["iteration"] in (0, 10, 50, 100, 200, 500, 1000) or s["kind"] == "end" or s.get("major")]
            cmap = plt.get_cmap("viridis")
            for k, s in enumerate(sel):
                if "turn" in s:
                    axs[i, 0].plot([q["rho"] for q in s["turn"]["shells"]], [q["q"][0] for q in s["turn"]["shells"]], color=cmap(k / max(1, len(sel) - 1)),
                                   lw=1, label=str(s["iteration"]) if k in (0, len(sel) - 1) else None)
            axs[i, 0].axhline(0, color="#999", lw=0.5)
            axs[i, 0].set_ylabel("q0", fontsize=7)
            axs[i, 0].set_title(f"S{r} (supplement, no registered score)" if supp else f"row {r} {rr['chain']}: {W[r].get('word')}", fontsize=7)
            axs[i, 0].legend(fontsize=6, frameon=False)
            its = [max(s["iteration"], 1) for s in st]
            axs[i, 1].plot(its, [s.get("dU", np.nan) for s in st], marker=".", ms=3, color=cols[0])
            axs[i, 1].set_xscale("log")
            axs[i, 1].set_yscale("symlog", linthresh=0.01)
            axs[i, 1].set_ylabel("dU", fontsize=7)
            axs[i, 2].plot(its, [s.get("tear", {}).get("least_g12_outside", np.nan) for s in st], marker=".", ms=3, color=cols[1])
            axs[i, 2].axhline(0.5, color="#999", lw=0.5, ls="--")
            axs[i, 2].set_xscale("log")
            axs[i, 2].set_ylabel("least g12 (tear region)", fontsize=7)
        axs[-1, 0].set_xlabel("shell radius")
        axs[-1, 1].set_xlabel("iteration (0 drawn at 1)")
        axs[-1, 2].set_xlabel("iteration (0 drawn at 1)")
        fig.tight_layout()
        fig.savefig(PLOTS / f"{pre}turn_profiles.png", dpi=90)
        plt.close(fig)
        # 3. the gap on a meridian plane (the plane holding the collar axis... the box xz plane) at the start and the end
        fig, axs = plt.subplots(n, 2, figsize=(5.0, 2.3 * n))
        axs = np.atleast_2d(axs)
        for i, r in enumerate(trows):
            rr = getr(r)
            x = T.axis_of(rr["grid"])
            N = len(x)
            for j, fn in enumerate(("s00000.npz", "end.npz")):
                pth = gdir(r) / fn
                if not pth.exists():
                    axs[i, j].axis("off")
                    continue
                m6 = np.load(pth, allow_pickle=False)["m6"]
                w, _ = T.eigs(m6[:, N // 2, :, :])
                im = axs[i, j].imshow((w[..., 0] - w[..., 1]).T, origin="lower", extent=[x[0], x[-1], x[0], x[-1]], vmin=0, vmax=1, cmap="magma")
                axs[i, j].set_title(f"{'S' if supp else 'row '}{r} {'start' if j == 0 else 'end'}: g12, y = 0", fontsize=7)
                axs[i, j].tick_params(labelsize=6)
        fig.colorbar(im, ax=axs, shrink=0.3)
        fig.savefig(PLOTS / f"{pre}gap_meridian.png", dpi=80)
        plt.close(fig)
    # 4. the W rows: the minor turn profile at the stops; the lines at the start and the end
    wrows = [r for r in (28, 29, 36, 37) if getr(r) and getr(r).get("end_state") != "NOT RUN"]
    if wrows:
        fig, axs = plt.subplots(len(wrows), 2, figsize=(8, 2.4 * len(wrows)))
        axs = np.atleast_2d(axs)
        for i, r in enumerate(wrows):
            rr = getr(r)
            st = rr["stops"]
            sel = [s for s in st if s["iteration"] in (0, 10, 50, 100, 200, 500, 1000) or s["kind"] == "end" or s.get("major")]
            cmap = plt.get_cmap("viridis")
            for k, s in enumerate(sel):
                if "minor" in s:
                    axs[i, 0].plot([q[0] for q in s["minor"]["shells"]], [q[1] for q in s["minor"]["shells"]], color=cmap(k / max(1, len(sel) - 1)), lw=1,
                                   label=str(s["iteration"]) if k in (0, len(sel) - 1) else None)
            axs[i, 0].set_title(f"S{r} (supplement, no registered score)" if supp else f"row {r} {rr['chain']}: {W[r].get('word')}", fontsize=7)
            axs[i, 0].set_ylabel("alpha_k", fontsize=7)
            axs[i, 0].set_xlabel("shell radius", fontsize=7)
            axs[i, 0].legend(fontsize=6, frameon=False)
            for lab, s in (("start", st[0]), ("end", rr["end"])):
                L = s.get("lines")
                if L is None and lab == "start":
                    L = start_lines(rr, gdir(r))
                if L is None:
                    continue
                for q in L:
                    axs[i, 1].plot([q["R"]] * len(q["comps"]), [c["polar_deg"] for c in q["comps"]], "o" if lab == "start" else "x", ms=4,
                                   color=cols[0] if lab == "start" else cols[1], label=lab if q is L[0] else None)
            axs[i, 1].set_ylabel("polar angle of a minor line (deg)", fontsize=7)
            axs[i, 1].set_xlabel("shell radius", fontsize=7)
            axs[i, 1].legend(fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(PLOTS / f"{pre}minor_profiles.png", dpi=90)
        plt.close(fig)
    # 5. the pair ladder: U and the separation at the major stops
    prs = [] if supp else [r for r in (30, 31, 32, 33) if rec(RX.cname(r))]
    if prs:
        fig, axs = plt.subplots(1, 2, figsize=(8, 2.8))
        for i, r in enumerate(prs):
            rr = rec(RX.cname(r))
            st = [s for s in rr["stops"] if s.get("major")]
            axs[0].plot([s["iteration"] for s in rr["stops"] if s["iteration"] > 0], [s["U"] for s in rr["stops"] if s["iteration"] > 0], color=cols[i],
                        label=f"row {r} {rr['chain']}: {W[r].get('word')}")
            seps = [(s["iteration"], s["pair"].get("separation")) for s in st if s.get("pair") and s["pair"].get("separation") is not None]
            if seps:
                axs[1].plot([q[0] for q in seps], [q[1] for q in seps], marker="o", color=cols[i], label=f"row {r}")
        axs[0].set_xscale("log")
        axs[0].set_xlabel("iteration")
        axs[0].set_ylabel("U")
        axs[0].legend(fontsize=6, frameon=False)
        axs[1].set_xlabel("iteration")
        axs[1].set_ylabel("separation (two-ring reads)")
        axs[1].legend(fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(PLOTS / "m5_56_pair.png", dpi=100)
        plt.close(fig)


# ----------------------------------------------------------------------------- markdown tables of the record (printed, pasted into the findings)

def _f(v, d=6):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            m, e = f"{v:.2e}".split("e")
            return f"${m} \\times 10^{{{int(e)}}}$"
        return f"{v:.{d}f}".rstrip("0").rstrip(".") if d else f"{v:.0f}"
    return str(v)


def tables():
    S = json.loads((DATA / "m5_56_summary.json").read_text())
    ch = S["chains"]
    print("| # | Chain | Grid | End | Iterations | U | dU | Gate | K5 | sigma | chi_max | rho_t | Word | Route | Why no label |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted(ch, key=int):
        w = ch[r]
        if w.get("word") and str(w["word"]).startswith("not run"):
            print(f"| {r} | {w['name']} | | not run | | | | | | | | | {w['word']} | | |")
            continue
        if "end_state" not in w:
            print(f"| {r} | {w['name']} | | {w.get('word')} | | | | | | | | | | | |")
            continue
        route = w.get("route_final") or (w.get("route") or {}).get("word")
        print(f"| {r} | `{w['chain']}` | {w['grid']} | {w['end_state']} | {w['iterations']} | {_f(w['U'])} | {_f(w.get('dU'), 4)} | {_f(w['gate'])} | {w['k5']} | "
              f"{_f(w.get('sigma'))} | {_f(w.get('chi_max'), 3)} | {_f(w.get('rho_t_cross'), 2)} | {w.get('word')} | {route or ''} | {'; '.join(w.get('why_no_label', []))} |")
    print()
    print(json.dumps(S["statements"], indent=1, default=str)[:6000])
    print(json.dumps(S["expectations"], indent=1, default=str)[:6000])
    print(json.dumps(S["rung5"], indent=1, default=str)[:6000])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths")
    ap.add_argument("--c4", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--tables", action="store_true", help="print the markdown tables of the record from the summary")
    a = ap.parse_args()
    if a.paths:
        o = paths(a.paths)
        print(json.dumps({k: (v if k != "tables" else {an: (t["word"], t.get("word_from_2pi"), round(t["largest_rise"], 3)) for an, t in v.items()})
                          for k, v in o.items() if k != "ends"}, default=T._jsonable))
        if "ends" in o:
            print({lab: {an: (q[an]["word"] if isinstance(q.get(an), dict) else q.get(an)) for an in ("normal", "inplane", "separation", "not_read")}
                   for lab, q in o["ends"].items()})
    elif a.c4:
        r = c4()
        print(json.dumps({k: v for k, v in r.items() if k != "fixed_fields"}, default=T._jsonable)[:3000])
        print([(q["input"], q["word"], q["ring_alone_rule"]) for q in r["fixed_fields"]])
    elif a.tables:
        tables()
    elif a.summary:
        o = summary()
        print(json.dumps(o["statements"], default=T._jsonable)[:4000])
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
