"""Add the stage-2 results to audit_results.json (stage-1 keys are left untouched)."""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
allres = json.loads((HERE / "audit_results.json").read_text())
stage1_keys = set(allres)
allres["stage2_item19"] = json.loads((HERE / "audit_res_stage2_item19.json").read_text())
allres["stage2_item20"] = json.loads((HERE / "audit_res_stage2_item20.json").read_text())
assert stage1_keys <= set(allres)
(HERE / "audit_results.json").write_text(json.dumps(allres, indent=1))
print(sorted(allres))
