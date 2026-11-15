#!/usr/bin/env python3
"""Exercise the pinned package guard on valid, missing and corrupted inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
import m10_3_replay as replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay.require(sys.flags.optimize == 0, "Optimized Python is unsupported")
    replay.require(not args.output.exists(), "Choose a fresh output file")
    source = args.source.resolve()
    replay.require(not args.output.resolve().is_relative_to(source), "Output is inside source")
    count = replay.verify_source(source)
    checks = {"unchanged_package": count == 614}
    with tempfile.TemporaryDirectory(prefix="m10_3_transport_") as directory:
        root = Path(directory) / "package"
        shutil.copytree(source, root)
        manifest = root / "PACKAGE.sha256"
        script = root / "reproduce.py"
        manifest_bytes, script_bytes = manifest.read_bytes(), script.read_bytes()

        def rejected(name):
            try:
                replay.verify_source(root)
            except (ValueError, OSError):
                checks[name] = True
            else:
                raise ValueError("Mutation survived: " + name)

        manifest.unlink()
        rejected("missing_manifest")
        manifest.write_bytes(manifest_bytes + b"wrong\n")
        rejected("changed_manifest")
        manifest.write_bytes(manifest_bytes)
        script.unlink()
        rejected("missing_source")
        script.write_bytes(script_bytes + b"# altered\n")
        rejected("changed_source")
        replacement = hashlib.sha256(script.read_bytes()).hexdigest()
        original = hashlib.sha256(script_bytes).hexdigest()
        manifest.write_bytes(manifest_bytes.replace(original.encode(), replacement.encode()))
        rejected("changed_source_with_rehashed_manifest")
        manifest.write_bytes(manifest_bytes)
        script.write_bytes(script_bytes)
        extra = root / "unsealed.py"
        extra.write_bytes(b"raise RuntimeError('unsealed')\n")
        rejected("extra_source")
        extra.unlink()
        replay.verify_source(root)
        result = subprocess.run(
            [
                sys.executable,
                "-O",
                "-B",
                str(Path(replay.__file__)),
                "--source",
                str(root),
                "--output",
                str(Path(directory) / "unused"),
            ],
            capture_output=True,
        )
        checks["optimized_entry"] = result.returncode != 0 and b"Optimized Python" in result.stderr
    replay.require(all(checks.values()) and len(checks) == 8, "Incomplete transport checks")
    replay.require(replay.verify_source(source) == count, "Source was changed by checks")
    report = {
        "status": "pass",
        "source_commit": replay.SOURCE_COMMIT,
        "manifest_sha256": replay.PACKAGE_MANIFEST,
        "files_verified": count,
        "checks": checks,
        "mutations_rejected": 7,
        "scope": "Developer corruption controls; no independent adversarial audit",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Passed unchanged-package check and rejected seven transport mutations")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
