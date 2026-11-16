#!/usr/bin/env python3
"""Refuse files over the size cap in committed content, unless allowed by name.

This is a public repository, and every byte committed is downloaded by every
clone and fork forever: history is not editable in practice once pushed. The
local-only rule for heavy arrays (`.gitignore`, the `data/` block) goes by file
type; this check goes by size, so a large JSON, log, transcript or image is
caught too.

The cap is 1 MB (1,000,000 bytes) per file. A file over it is refused unless
its path is listed in `dev_docs/utils/large_files_allowed.txt`, one path per
line, its reason after whitespace and `#`. Prefer, in order: keep the file
local and gitignored, with its regeneration command in the task record and the
folder's `_DATASETS.md`; commit a distilled summary instead; only then allow it
by name, with the reason a reviewer can read.

Usage:
    python3 dev_docs/utils/check_file_size.py --staged           # pre-commit hook
    python3 dev_docs/utils/check_file_size.py --range main..HEAD # a branch or a PR
    python3 dev_docs/utils/check_file_size.py --tracked          # whole-repo audit
    python3 dev_docs/utils/check_file_size.py --selftest         # the check can fail

What each mode measures, and which allowlist it reads:
    --staged   every blob staged for the commit; the allowlist as staged
    --range    every blob that enters history between the two ends, including
               files added and removed again inside the range; the allowlist at
               the range's tip
    --tracked  every blob in HEAD's tree; the allowlist at HEAD
Submodules (gitlinks) carry no blob and are skipped. Run from anywhere in the
repository: paths are repo-relative.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

LIMIT = 1_000_000
ALLOWLIST = "dev_docs/utils/large_files_allowed.txt"


class GitError(Exception):
    pass


def git(*args: str, data: str | None = None) -> str:
    r = subprocess.run(["git", *args], input=data, capture_output=True, text=True)
    if r.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {r.stderr.strip() or 'exit ' + str(r.returncode)}")
    return r.stdout


def parse_allowlist(text: str) -> set[str]:
    out = set()
    for line in text.splitlines():
        path = re.split(r"\s+#", line, maxsplit=1)[0].strip()
        if path and not path.startswith("#"):
            out.add(path)
    return out


def allowlist_at(spec: str) -> set[str]:
    """spec is ':' for the index, or a revision; a missing allowlist there is an empty set"""
    try:
        return parse_allowlist(git("show", f"{spec}{ALLOWLIST}" if spec == ":" else f"{spec}:{ALLOWLIST}"))
    except GitError:
        return set()


def blob_sizes(pairs: list[tuple[str, str]]) -> list[tuple[str, int]]:
    """pairs of (object id, path) -> (path, size) for the blobs among them"""
    if not pairs:
        return []
    out = git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)", data="".join(o + "\n" for o, _ in pairs))
    sizes = []
    for (oid, path), line in zip(pairs, out.splitlines()):
        parts = line.split()
        if len(parts) == 3 and parts[1] == "blob":
            sizes.append((path, int(parts[2])))
    return sizes


def staged_sizes() -> list[tuple[str, int]]:
    # raw records: ":<old mode> <new mode> <old id> <new id> <status>\0<path>\0" (two paths for renames and copies)
    raw = git("diff", "--cached", "--raw", "-z", "--no-abbrev", "--diff-filter=ACMR").split("\0")
    pairs, i = [], 0
    while i < len(raw) - 1:
        meta = raw[i].lstrip(":").split()
        if len(meta) < 5:
            i += 1
            continue
        new_mode, new_id, status = meta[1], meta[3], meta[4]
        step = 3 if status[0] in "RC" else 2
        path = raw[i + step - 1]
        if new_mode != "160000":
            pairs.append((new_id, path))
        i += step
    return blob_sizes(pairs)


def range_sizes(rev_range: str) -> list[tuple[str, int]]:
    if ".." not in rev_range:
        raise GitError(f"--range needs two ends, A..B or A...B, got {rev_range!r}")
    pairs = []
    for line in git("rev-list", "--objects", rev_range).splitlines():
        oid, _, path = line.partition(" ")
        if path:
            pairs.append((oid, path))
    return blob_sizes(pairs)


def tracked_sizes() -> list[tuple[str, int]]:
    out = []
    for line in git("ls-tree", "-r", "-l", "-z", "--full-tree", "HEAD").split("\0"):
        if not line:
            continue
        meta, name = line.split("\t", 1)
        size = meta.split()[3]
        if size != "-":
            out.append((name, int(size)))
    return out


def oversize(entries: list[tuple[str, int]], allowed: set[str], limit: int = LIMIT) -> list[tuple[str, int]]:
    seen, bad = set(), []
    for n, s in entries:
        if s > limit and n not in allowed and (n, s) not in seen:
            seen.add((n, s))
            bad.append((n, s))
    return bad


def selftest() -> int:
    entries = [("small.json", 10_000), ("big.json", 2_500_000), ("big_allowed.gif", 3_000_000), ("edge.bin", LIMIT), ("a#b.bin", 2_000_000)]
    allowed = parse_allowlist("# header\n\nbig_allowed.gif   # reason\na#b.bin\t# a path with a hash\n")
    got = oversize(entries, allowed)
    ok = got == [("big.json", 2_500_000)] and allowed == {"big_allowed.gif", "a#b.bin"}
    print("selftest:", "PASS (an unlisted 2.5 MB file is refused; listed ones, a path with '#' among them, and one at the cap pass)" if ok else f"FAIL {got} {allowed}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--staged", action="store_true")
    g.add_argument("--range")
    g.add_argument("--tracked", action="store_true")
    g.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        os.chdir(git("rev-parse", "--show-toplevel").strip())
        if a.staged:
            entries, allowed = staged_sizes(), allowlist_at(":")
        elif a.range:
            tip = (a.range.split("...")[-1] if "..." in a.range else a.range.split("..")[-1]) or "HEAD"
            entries, allowed = range_sizes(a.range), allowlist_at(tip)
        else:
            entries, allowed = tracked_sizes(), allowlist_at("HEAD")
    except GitError as e:
        print(f"check_file_size: {e}", file=sys.stderr)
        return 2
    bad = oversize(entries, allowed)
    total = sum(s for _, s in entries)
    if bad:
        for name, size in sorted(bad, key=lambda x: -x[1]):
            print(f"❌ {name}: {size / 1e6:.2f} MB, over the {LIMIT / 1e6:.0f} MB cap", file=sys.stderr)
        print(f"Keep the file local and gitignored, commit a summary instead, or allow it by name with its reason in {ALLOWLIST}.", file=sys.stderr)
        return 1
    largest = max((s for _, s in entries), default=0)
    print(f"✅ size check clean ({len(entries)} file(s), {total / 1e6:.2f} MB in all, largest {largest / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
