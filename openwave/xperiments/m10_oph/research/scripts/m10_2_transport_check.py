#!/usr/bin/env python3
"""M10.2: require the package integrity and numeric comparison gates to fail."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

from m10_2_replay import PACKAGE, verify_checkout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_dir.resolve()
    verify_checkout(root)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("static_replay", root / PACKAGE / "replay.py")
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    replay.verify()
    passed = []

    def rejected(name, operation):
        try:
            operation()
        except (ValueError, FileNotFoundError, KeyError):
            passed.append(name)
        else:
            raise AssertionError("Guard accepted mutation: " + name)

    for case in ("missing_input", "changed_input", "extra_input", "changed_input_and_manifest"):
        with tempfile.TemporaryDirectory(prefix="m10-transport-") as temporary:
            package = Path(temporary) / "package"
            shutil.copytree(root / PACKAGE, package)
            replay.HERE = package
            replay.EVIDENCE = package / "evidence"
            target = replay.EVIDENCE / "openwave/shared/static_candidate_join/source.py"
            if case == "missing_input":
                target.unlink()
            elif case == "extra_input":
                (replay.EVIDENCE / "extra.txt").write_text("unexpected input\n")
            else:
                target.write_bytes(target.read_bytes() + b"\n# mutation\n")
                if case == "changed_input_and_manifest":
                    path = package / "MANIFEST.json"
                    manifest = json.loads(path.read_text())
                    manifest["files"][str(target.relative_to(replay.EVIDENCE))] = hashlib.sha256(
                        target.read_bytes()
                    ).hexdigest()
                    path.write_text(json.dumps(manifest))
            rejected(case, replay.verify)
    for label, value in (
        ("nan", float("nan")),
        ("positive_infinity", float("inf")),
        ("negative_infinity", -float("inf")),
    ):
        rejected(label, lambda v=value: replay.compare({"value": v}, {"value": 1.0}))
    rejected(
        "extra_result_field", lambda: replay.compare({"value": 1.0, "extra": True}, {"value": 1.0})
    )
    print(json.dumps({"task": "M10.2", "all_rejected": True, "mutations": passed}, indent=2))


if __name__ == "__main__":
    main()
