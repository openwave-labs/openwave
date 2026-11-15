#!/usr/bin/env python3
"""Reproduce the pinned M10.2 static-source package and independent audit.

Network access is limited to an explicit Git fetch from the author repository.
The package runs in disposable copies. --source-dir accepts a local checkout
at exactly the pinned commit and verifies its package bytes against Git first.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

AUTHOR_REPOSITORY = "https://github.com/FloatingPragma/observer-patch-holography.git"
AUTHOR_COMMIT = "c330c6e7f23506b5f87c339cd73906928c1a7cd9"
PACKAGE = "code/openwave_static_signed_source"


def git(root: Path, *arguments: str) -> str:
    """Run a Git read/fetch with no shell interpolation or checkout hooks."""
    return subprocess.check_output(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(root), *arguments],
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


def verify_checkout(root: Path) -> dict:
    """Bind every executed package byte to the selected commit, including extras."""
    if git(root, "rev-parse", "HEAD") != AUTHOR_COMMIT:
        raise ValueError("Author checkout is not at the pinned commit")
    tracked = git(root, "ls-tree", "-r", AUTHOR_COMMIT, "--", PACKAGE).splitlines()
    names = set()
    for entry in tracked:
        meta, name = entry.split("\t", 1)
        mode, kind, wanted = meta.split()
        path = root / name
        if kind != "blob" or mode not in ("100644", "100755") or path.is_symlink():
            raise ValueError("Unsupported package entry: " + name)
        if git(root, "hash-object", "--", name) != wanted:
            raise ValueError("Package byte mismatch: " + name)
        names.add(name)
    actual = {p.relative_to(root).as_posix() for p in (root / PACKAGE).rglob("*") if p.is_file()}
    if actual != names or not names:
        raise ValueError("Package file set differs from the commit")
    return {"files": len(names), "commit": AUTHOR_COMMIT}


def reproduce(root: Path) -> dict:
    """Run original evidence checks and the separate implementation audit."""
    identity = verify_checkout(root)
    env = os.environ.copy()
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        PYTHONHASHSEED="0",
    )
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONOPTIMIZE", None)
    commands = [
        ["-B", PACKAGE + "/replay.py"],
        ["-B", PACKAGE + "/audit/audit.py", "--evidence-root", PACKAGE + "/evidence", "--check"],
    ]
    jobs = []
    for arguments in commands:
        start = time.perf_counter()
        run = subprocess.run(
            [sys.executable, *arguments],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if run.returncode:
            sys.stderr.write(run.stderr)
            raise RuntimeError("Pinned replay failed: " + arguments[1])
        result = json.loads(run.stdout)
        jobs.append(
            {
                "command": "python3 " + " ".join(arguments),
                "exit": run.returncode,
                "wall_seconds": time.perf_counter() - start,
                "result": result,
            }
        )
    verify_checkout(root)
    return {"identity": identity, "jobs": jobs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, help="Local author checkout at the pinned commit"
    )
    parser.add_argument("--output", type=Path, help="Write the replay record here")
    args = parser.parse_args()
    if args.source_dir:
        result = reproduce(args.source_dir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="m10-static-source-") as directory:
            root = Path(directory)
            git(root, "init", "--quiet")
            git(root, "remote", "add", "origin", AUTHOR_REPOSITORY)
            git(
                root,
                "fetch",
                "--quiet",
                "--filter=blob:none",
                "--depth=1",
                "origin",
                AUTHOR_COMMIT,
            )
            git(root, "sparse-checkout", "set", "--no-cone", "/" + PACKAGE + "/")
            git(root, "checkout", "--quiet", "--detach", "FETCH_HEAD")
            result = reproduce(root)
    result.update(
        task="M10.2",
        written=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        package=AUTHOR_REPOSITORY.removesuffix(".git") + "/tree/" + AUTHOR_COMMIT + "/" + PACKAGE,
        versions={
            "python": sys.version.split()[0],
            **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "sympy")},
        },
        verdict="REPRODUCED WITH DISCLOSED SOURCE PREMISES",
        official_rows_solved=[],
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
