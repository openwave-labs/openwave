"""M5.36: compile the reproduction record of the scorecard v1.7 scripts V58 to V84 (V78 excluded) and the
Release 2.3 gate into data/m5_36_reproduction.json.

Inputs (paths as arguments): the pool logs folder (one <name>.log per job with START/END lines and the
script's own prints, plus _pool.json from the launcher), the shipped tree (the archive copy, reference
JSONs), the run tree (the isolated copy the scripts ran in), the gate log. A JSON in the run tree counts
as regenerated when its mtime is newer than the launch time (the earliest log's mtime minus its job's wall,
with a 5 s margin); each regenerated JSON is compared leaf by leaf with
the shipped copy (max relative difference and its leaf; strings and booleans must be equal).
Output: per job exit code and wall seconds; per regenerated JSON the leaf count, the max relative
difference and its leaf, the count of leaves above 1e-6 and above 1e-3; the gate's passed / failed line.
No absolute path is written to the record (basenames only).
Usage: python3 m5_36_reproduction_record.py <logs_dir> <shipped_dir> <run_dir> <gate_log> [--write]
"""
import json, math, os, re, sys


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append((path + "/" + str(k), "missing", float("inf")))
                continue
            walk(a[k], b[k], path + "/" + str(k), out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, "len %d vs %d" % (len(a), len(b)), float("inf")))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, path + "[%d]" % i, out)
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        if a != b:
            out.append((path, "%r vs %r" % (a, b), float("inf")))
    else:
        try:
            a = float(a)
            b = float(b)
            if math.isnan(a) and math.isnan(b):
                return
            d = abs(a - b)
            r = d / max(abs(a), abs(b), 1e-300)
            out.append((path, "%.10g vs %.10g" % (a, b), r if d > 0 else 0.0))
        except Exception:
            out.append((path, "uncomparable", float("inf")))


def compare(shipped, run, t0):
    rows = {}
    for fn in sorted(os.listdir(run)):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(run, fn)
        if os.path.getmtime(p) < t0:
            continue
        q = os.path.join(shipped, fn)
        if not os.path.exists(q):
            rows[fn] = {"status": "new (no shipped copy)"}
            continue
        out = []
        walk(json.load(open(q)), json.load(open(p)), "", out)
        finite = [o for o in out if math.isfinite(o[2])]
        worst = max(out, key=lambda o: o[2]) if out else ("", "", 0.0)
        rows[fn] = {
            "leaves": len(out),
            "max_rel_diff": worst[2] if math.isfinite(worst[2]) else "inf",
            "at": worst[0],
            "values": worst[1],
            "above_1e-6": sum(1 for o in finite if o[2] > 1e-6) + (len(out) - len(finite)),
            "above_1e-3": sum(1 for o in finite if o[2] > 1e-3) + (len(out) - len(finite)),
        }
    return rows


def parse_logs(d):
    pool = json.load(open(os.path.join(d, "_pool.json")))
    jobs = {}
    for j in pool["jobs"]:
        log = os.path.join(d, j["name"] + ".log")
        txt = open(log, errors="replace").read() if os.path.exists(log) else ""
        flags = [w for w in ("Traceback", "Error", "nan", "inf", "Warning") if re.search(r"\b%s\b" % w, txt)]
        jobs[j["name"]] = {"script": j["script"], "pass": j.get("pass", 1), "flags_run": j.get("flags_note", "python3 -I -E -s"), "rc": j["rc"], "wall_s": j["wall_s"], "log_lines": txt.count("\n"), "flags": flags}
    return pool, jobs


def parse_gate(path):
    txt = open(path, errors="replace").read() if os.path.exists(path) else ""
    m = re.search(r"(\d+)\s+passed,?\s+(\d+)\s+failed", txt)
    fails = [l.strip() for l in txt.splitlines() if l.strip().startswith("FAIL")]
    return {"passed": int(m.group(1)) if m else None, "failed": int(m.group(2)) if m else None, "fail_lines": fails, "lines": txt.count("\n")}


def main():
    logs, shipped, run, gate = sys.argv[1:5]
    write = "--write" in sys.argv
    pool, jobs = parse_logs(logs)
    # launch time: the earliest log start (each log is created at its job's START), minus a 5 s margin
    t0 = min(os.path.getmtime(os.path.join(logs, f)) - j["wall_s"] for f, j in ((n + ".log", j) for n, j in jobs.items())) - 5
    rec = {
        "task": "M5.36",
        "what": "scorecard v1.7 scripts V58 to V84 as shipped (V78 excluded), 6 single-threaded workers; the Release 2.3 gate",
        "pool": {"workers": pool["workers"], "pool_wall_s": pool["pool_wall_s"], "jobs": len(jobs), "exit_0": sum(1 for j in jobs.values() if j["rc"] == 0), "passes": pool.get("passes", {})},
        "jobs": jobs,
        "regenerated_json": compare(shipped, run, t0),
        "gate": parse_gate(gate),
    }
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "data", "m5_36_reproduction.json")
    if write:
        json.dump(rec, open(out, "w"), indent=1)
        print("written", os.path.basename(out))
    print("jobs", rec["pool"]["jobs"], "exit 0:", rec["pool"]["exit_0"], "pool wall", rec["pool"]["pool_wall_s"], "s")
    for k, v in rec["regenerated_json"].items():
        print("%-48s %s" % (k, v.get("max_rel_diff", v.get("status")), ), v.get("at", ""))
    print("gate:", rec["gate"]["passed"], "passed", rec["gate"]["failed"], "failed")


if __name__ == "__main__":
    main()
