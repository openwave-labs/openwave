"""Merge out/*.json into results.json (exact values as strings)."""
import os, json, glob

HERE = os.path.dirname(os.path.abspath(__file__))
res = {}
for f in sorted(glob.glob(os.path.join(HERE, "out", "*.json"))):
    res[os.path.splitext(os.path.basename(f))[0]] = json.load(open(f))
json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1, sort_keys=False)
print("results.json written with sections:", sorted(res))
