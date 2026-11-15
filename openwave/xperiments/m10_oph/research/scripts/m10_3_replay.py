#!/usr/bin/env python3
"""Check the pinned wave package, then reproduce its finite headline or full replay.

No download occurs here. --source names code/openwave_wave_evidence in a checkout
of the author commit below. All package files are hashed before executing code.
The headline verifies saved native records with the independent review program;
full mode regenerates the four source leaves using the declared amendment.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

SOURCE_COMMIT = "2c000fd4ff9dbdd0303328d7dfa028a68a6f4766"
PACKAGE_MANIFEST = "a906a237f87604e7aff23123865821375aade8e6a27eaa0539b7cbd63fd2a4bd"
SOURCE_REPOSITORY = "https://github.com/FloatingPragma/observer-patch-holography"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(root):
    manifest = root / "PACKAGE.sha256"
    require(sha(manifest) == PACKAGE_MANIFEST, "Package manifest mismatch")
    names = {"PACKAGE.sha256"}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        require(name not in names and "\\" not in name and ":" not in name,
                "Invalid or duplicate package path")
        target = root / name
        require(target.resolve().is_relative_to(root) and target.resolve() != root,
                "Package path escape")
        require(sha(target) == expected, "Package bytes changed: " + name)
        names.add(name)
    files = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), "Package contains symlink")
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    require(files == names and len(files) == 614, "Package inventory mismatch")
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("headline", "full"), default="headline")
    args = parser.parse_args()
    require(sys.flags.optimize == 0, "Optimized Python is unsupported")
    source, output = args.source.resolve(), args.output.resolve()
    count = verify_source(source)
    require(not output.is_relative_to(source), "Output must be outside the package")
    output.mkdir(parents=True, exist_ok=False)
    command = "independent" if args.mode == "headline" else "replay"
    artifact = output / ("independent.json" if args.mode == "headline" else "replay")
    start = time.monotonic()
    process = subprocess.run(
        [sys.executable, "-B", str(source / "reproduce.py"), command, "--output", str(artifact)],
        capture_output=True,
        check=False,
    )
    elapsed = time.monotonic() - start
    for stream in ("stdout", "stderr"):
        (output / (stream + ".txt")).write_bytes(getattr(process, stream))
    receipt = {
        "task": "M10.3",
        "written": datetime.now(timezone.utc).isoformat(),
        "package": {"repository": SOURCE_REPOSITORY, "commit": SOURCE_COMMIT,
                    "manifest_sha256": PACKAGE_MANIFEST, "files_verified": count},
        "mode": args.mode,
        "exit_code": process.returncode,
        "wall_seconds": elapsed,
        "logs": {name: {"file": name + ".txt", "sha256": sha(output / (name + ".txt"))}
                 for name in ("stdout", "stderr")},
        "status": "pass" if process.returncode == 0 else "fail",
        "scope": "Developer reproduction; no platform preregistration, row award or candidate adoption",
    }
    try:
        require(process.returncode == 0, "Source replay failed; inspect retained logs")
        require(verify_source(source) == count, "Source changed during execution")
        record_path = artifact if args.mode == "headline" else artifact / "REPLAY.json"
        record = json.loads(record_path.read_bytes())
        if args.mode == "headline":
            require(record["all_review_checks_pass"] is True, "Independent checks failed")
            require(record["free_tapes"]["actual_current_records"] == 131072,
                    "Wrong current-record count")
            require(record["free_tapes"]["branches"] == 37908, "Wrong branch count")
            receipt["headline"] = {"current_records": 131072, "branches": 37908}
        else:
            require(record["status"] == "pass", "Full replay failed")
            require(record["total_native_calls_in_overlapping_lineage"] == 1009734,
                    "Wrong lineage census")
            receipt["headline"] = {"overlapping_lineage_calls": 1009734}
    except Exception as error:
        receipt.update(status="fail", error=str(error))
        raise
    finally:
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "headline": receipt["headline"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
