"""Add the refutation-phase outputs to audit_results.json under the key 'refutation' (other keys untouched)."""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
allres = json.loads((HERE / "audit_results.json").read_text())
before = set(allres)
ref = {}
for f in ("audit_ref_compare.json", "audit_ref_newchecks.json", "audit_ref_item13_relations.json"):
    ref[f.replace(".json", "")] = json.loads((HERE / f).read_text())
logs = {}
for d in sorted((HERE / "ref_runs").glob("mut_*")):
    for lg in sorted(d.glob("log_*.txt")):
        lines = lg.read_text().splitlines()
        logs[f"{d.name}/{lg.name}"] = lines[-12:]
ref["mutation_logs_tail"] = logs
allres["refutation"] = ref
assert before <= set(allres)
(HERE / "audit_results.json").write_text(json.dumps(allres, indent=1, default=str))
print(sorted(allres), len(logs))
