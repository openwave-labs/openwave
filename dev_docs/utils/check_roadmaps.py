"""Lint the roadmaps: the word budgets defined in dev_docs/ROADMAP_STANDARDS.md.

Checks, per roadmap file:
1. Budget: every `Description` cell, row `Title` cell, blockquote and change-log
   entry stays inside its cap (CAPS below). Words are counted after markdown is
   removed, so formatting never buys room: link labels count and link targets do
   not, "<br>" is a space, and ` * _ # are stripped.
2. Column: every live/DONE table declares a column named exactly `Description`,
   which is what the budget attaches to. A table without one is reported rather
   than silently skipped.
3. Shape: every data row carries as many cells as its header. This is what an
   unescaped "|" inside a cell looks like from here, and such a row otherwise
   parses by position and reports the wrong column's contents.

ARCHIVE, LEGACY and PRE-REGISTERED sections are frozen (ROADMAP_STANDARDS.md
section 7) and are skipped entirely: they preserve how the work read, or what was
filed, at the time.

Usage: python3 dev_docs/utils/check_roadmaps.py [path ...]   (default: this repo's)
Any roadmap path may be passed explicitly, including one outside this repository.
Exit 0 = clean, 1 = violations (listed on stdout).
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAPS = {
    "description": 65,
    "title": 15,
    "other": 35,
    "blockquote": 50,
    "intro": 80,
    "changelog": 200,
}
FROZEN = ("ARCHIVE", "LEGACY", "PRE-REGISTERED")

LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
BREAK = re.compile(r"<br\s*/?>")
MARKUP = re.compile(r"[`*_#>]")
RULE = re.compile(r"^\|[\s\-:|]+\|$")
SPLIT = re.compile(r"(?<!\\)\|")  # a literal pipe in a cell is written \|


REQUIRED = ("dev_docs/platform_roadmap.md",)


def roadmaps():
    """Every roadmap this repository binds, existing or not.

    A required file that is absent is returned anyway, so that main() can
    report it. Filtering the list through an existence test instead would mean
    deleting a roadmap makes this checker pass, which is the failure mode where
    the check reports on nothing and the exit code reads as clean.
    """
    found = [ROOT / r for r in REQUIRED]
    found += sorted((ROOT / "openwave" / "xperiments").glob("*/research/m*_roadmap.md"))
    return found


def words(cell):
    """Prose word count: link labels in, link targets out, markup stripped."""
    return len(MARKUP.sub("", BREAK.sub(" ", LINK.sub(r"\1", cell))).split())


def cells_of(line):
    return [c.strip() for c in SPLIT.split(line.strip())[1:-1]]


def check(path):
    errors = []
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    section = None  # nearest "## ..." heading
    frozen = False
    changelog = False
    idx = None  # column index of `Description` in the current table
    width = None  # cell count of the current table's header
    header_line = None
    tables = 0  # tables seen in this file, for the "no Description column" report

    for i, line in enumerate(path.read_text().splitlines(), 1):
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            title = line.strip("# ").strip()
            if level == 1:
                section, frozen, changelog = None, False, False
            elif level == 2:  # only a top-level section can open or close a frozen region
                section = title
                frozen = any(w in title.upper() for w in FROZEN)
                changelog = "CHANGE-LOG" in title.upper()
            else:  # a sub-heading inherits its parent section's frozen / change-log state
                section = title
            idx = width = None
            continue

        if frozen:
            continue

        if not line.startswith("|"):
            idx = width = None  # a blank or prose line ends the table above it

        if line.startswith(">"):
            cap = CAPS["blockquote"] if section else CAPS["intro"]
            n = words(line)
            if n > cap:
                what = f"{section} blockquote" if section else "intro blockquote"
                errors.append(f"{rel}:{i}: {what} is {n} words, cap {cap}")
            continue

        if changelog and line.strip().startswith("**"):
            n = words(line)
            if n > CAPS["changelog"]:
                errors.append(f"{rel}:{i}: change-log entry is {n} words, cap {CAPS['changelog']}")
            continue

        if not line.startswith("|") or RULE.match(line):
            continue

        cells = cells_of(line)
        if "Description" in cells:  # a header row: this table is in scope
            idx, width, header_line = cells.index("Description"), len(cells), i
            header = cells
            tables += 1
            continue
        if idx is None:
            if "TaskID" in cells or "Task ID" in cells:  # a table, but not a scoped one
                tables += 1
                errors.append(
                    f"{rel}:{i}: task table has no column named 'Description'"
                    " (ROADMAP_STANDARDS.md section 2)"
                )
                width = len(cells)
            continue

        if len(cells) != width:
            errors.append(
                f"{rel}:{i}: row '{cells[0][:30]}' has {len(cells)} cells,"
                f" header (L{header_line}) has {width}"
                r" (an unescaped '|' inside a cell does this; write it as '\|')"
            )
            continue  # parsing by column index would report the wrong cells

        n = words(cells[idx])
        if n > CAPS["description"]:
            errors.append(
                f"{rel}:{i}: {cells[0][:30]} description is {n} words,"
                f" cap {CAPS['description']} (over by {n - CAPS['description']})"
            )
        if len(cells) > 1:
            t = words(cells[1])
            if t > CAPS["title"]:
                errors.append(
                    f"{rel}:{i}: {cells[0][:30]} title is {t} words, cap {CAPS['title']}"
                )
        for j, cell in enumerate(cells):  # gate / owner / completed: pointers, not records
            if j in (0, 1, idx):
                continue
            n = words(cell)
            if n > CAPS["other"]:
                errors.append(
                    f"{rel}:{i}: {cells[0][:30]} '{header[j][:18]}' cell is {n} words,"
                    f" cap {CAPS['other']}"
                )

    return errors, tables


def main():
    paths = [Path(a).resolve() for a in sys.argv[1:]] or roadmaps()

    missing = [p for p in paths if not p.exists()]
    if missing or not paths:
        for p in missing:
            print(f"❌ roadmap not found: {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
        if not paths:
            print("❌ no roadmap found to check")
        print(f"\nnothing checked | {len(missing)} missing roadmap(s)")
        return 1

    errors, tables = [], 0
    for p in paths:
        e, t = check(p)
        errors += e
        tables += t
    for e in errors:
        print(e)
    print(f"\n{len(paths)} roadmap(s) | {tables} scoped table(s) | caps {CAPS}")
    print("clean" if not errors else f"{len(errors)} violation(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
