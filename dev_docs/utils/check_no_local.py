#!/usr/bin/env python3
"""Refuse machine-local absolute paths in committed content.

Paths like `/Users/someone/src/openwave/...` reach a repository the same way
every time: a script gets a hardcoded default, a run log captures the working
directory, an exception traceback is saved as test data, a generated file
records the command that produced it. None of it is deliberate, all of it is
useless to anyone else, and it is permanent once pushed, because history is not
editable in practice on a repository that has clones and forks.

It also publishes the committer's username and directory layout, which is
nobody's business.

Usage:
    python3 dev_docs/utils/check_no_local.py --staged     # pre-commit hook
    python3 dev_docs/utils/check_no_local.py --tracked    # whole-repo audit
    python3 dev_docs/utils/check_no_local.py --changed    # vs origin/main
    python3 dev_docs/utils/check_no_local.py FILE [FILE ...]

An empty selection is not a pass. The success line states how many files were
read, and `--require-files` turns "read nothing" into a failure, which is what
a scripted handover wants: run the check after the commit rather than before
and the staged set is empty, so a bare exit code reports success having opened
nothing. The pre-commit hook leaves the flag off, because a commit of only
binary files legitimately reads nothing.

Fixes, in order of preference: a repo-relative path; an environment variable
with a sensible default (`os.environ.get("SCRATCH", ".")`); a placeholder such
as `<repo>/` in captured output.

A path that genuinely has to stay is waived per line with `allow-local-path` on
that line, which stays visible in review.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

WAIVER = "allow-local-path"

# Case-sensitive on purpose: `/users/project/` in a documentation URL is not a
# home directory, and flagging it would train people to ignore this check.
PATTERNS: list[tuple[str, str]] = [
    ("unix home directory", r"/(?:Users|home)/[A-Za-z0-9._-]+/"),
    ("flattened home directory", r"-(?:Users|home)-[A-Za-z0-9._-]+-"),
    ("windows home directory", r"[A-Za-z]:\\\\?Users\\\\?[A-Za-z0-9._-]+"),
    ("machine scratch directory", r"/private/(?:tmp|var)/|/tmp/[A-Za-z0-9._-]+-\d+/"),
]

SELF_EXEMPT = {
    "dev_docs/utils/check_no_local.py",
    ".githooks/pre-commit",
}

TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".txt",
    ".json",
    ".csv",
    ".toml",
    ".cfg",
    ".ini",
    ".yml",
    ".yaml",
    ".rst",
    ".sh",
    ".gitignore",
    ".gitattributes",
    "",
}


def is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name.startswith(".")


def git(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], capture_output=True, text=True)
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def collect(mode: str, explicit: list[str]) -> list[str]:
    if explicit:
        return explicit
    if mode == "staged":
        return git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    if mode == "tracked":
        return git("ls-files")
    if mode == "changed":
        return git("diff", "origin/main...HEAD", "--name-only", "--diff-filter=ACMR") + git(
            "diff", "--name-only", "--diff-filter=ACMR"
        )
    return []


def scan(paths: list[str]) -> tuple[list[str], int]:
    """Return the hits, and how many files were actually read.

    The second number is the one a caller can trust. A path is selected by git
    and then skipped here for being a directory, a binary, or unreadable, so
    the size of the selection says nothing about how much was checked.
    """
    hits: list[str] = []
    seen: set[str] = set()
    read = 0
    for rel in paths:
        if rel in seen or rel in SELF_EXEMPT:
            continue
        seen.add(rel)
        p = Path(rel)
        if not p.is_file() or not is_text(p):
            continue
        try:
            lines = p.read_text(errors="replace").splitlines()
        except OSError:
            continue
        read += 1
        for n, line in enumerate(lines, 1):
            if WAIVER in line:
                continue
            for label, pat in PATTERNS:
                if re.search(pat, line):
                    hits.append(f"{rel}:{n}: [{label}] {line.strip()[:110]}")
                    break
    return hits, read


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--staged", action="store_true")
    g.add_argument("--tracked", action="store_true")
    g.add_argument("--changed", action="store_true")
    ap.add_argument(
        "--require-files",
        action="store_true",
        help="fail when nothing was read, instead of passing an empty selection",
    )
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()

    mode = (
        "staged"
        if args.staged
        else "tracked" if args.tracked else "changed" if args.changed else "staged"
    )
    paths = collect(mode, args.files)

    hits, read = scan(paths)
    if hits:
        print(f"\n❌ path check failed: {len(hits)} line(s) carry a machine-local path.\n")
        for h in hits:
            print(f"    {h}")
        print(
            "\nUse a repo-relative path, an environment variable, or a `<repo>/` "
            f"placeholder in captured output.\nIf a path must stay, append "
            f"`{WAIVER}` to that line.\n"
        )
        return 1

    if not read:
        why = "nothing selected" if not paths else f"{len(paths)} selected, none readable text"
        print(f"⚠️ path check READ NOTHING ({why}): this result checks nothing")
        return 1 if args.require_files else 0

    print(f"✅ path check clean ({read} file(s) read of {len(paths)} selected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
