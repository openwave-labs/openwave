"""Lint the roadmaps: the word budgets defined in dev_docs/ROADMAP_STANDARDS.md.

Checks, per roadmap file:
1. Budget: every `Description` cell, row `Title` cell, blockquote and change-log
   entry stays inside its cap (CAPS below). Words are counted after markdown is
   removed, so formatting never buys room: link labels count and link targets do
   not, "<br>" is a space, and ` * _ # are stripped. A blockquote paragraph and a
   change-log entry are each counted whole, across every line they wrap onto: a
   paragraph ends at a quoted blank line (">"), an entry runs from its bold opening
   line to the next blank line.
2. Column: every live/DONE table declares a column named exactly `Description`,
   which is what the budget attaches to. A table without one is reported rather
   than silently skipped.
3. Shape: every data row carries as many cells as its header. This is what an
   unescaped "|" inside a cell looks like from here, and such a row otherwise
   parses by position and reports the wrong column's contents.
4. Status icon: in a task table (first column `TaskID`), every row's first cell
   opens with the icon of the region the table sits in, then a space, then the ID:
   🔶 under IN PROGRESS (🔶🔶 for a task in progress over several days, waiting for
   a later day), 🚧 under BACKLOG or LATER, ✅ under DONE. A section that is
   none of these (a phase table, say) inherits the region of the last one above it.
   The Description does not open with one of these three icons: the status is said
   once, beside the ID. A `(none)` placeholder row carries no icon.

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

# The status icon of a task row, by the region its table sits in (docstring, check 4).
STATUS = {"progress": "\U0001f536", "backlog": "\U0001f6a7", "done": "\u2705"}
REGION_NAME = {"progress": "IN PROGRESS", "backlog": "BACKLOG", "done": "DONE"}
LIFECYCLE = re.compile("^(?:\u2705|\U0001f536|\U0001f6a7)")


def region_of(title, current):
    """The region a level-2 heading opens; any other heading keeps the current one."""
    t = title.upper()
    if t.startswith("IN PROGRESS"):
        return "progress"
    if t.startswith(("BACKLOG", "LATER")):
        return "backlog"
    if t.startswith("DONE"):
        return "done"
    return current


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
    region = None  # progress / backlog / done: sets the status icon of a task row
    task_table = False  # the current table's first column is the task ID
    frozen = False
    changelog = False
    idx = None  # column index of `Description` in the current table
    width = None  # cell count of the current table's header
    header_line = None
    tables = 0  # tables seen in this file, for the "no Description column" report
    # An open blockquote paragraph or change-log entry. Each is budgeted whole, so it
    # is collected across every line it wraps onto and counted when it ends:
    # counting line by line let a hard-wrapped entry pass at any length.
    block = None

    def close():
        nonlocal block
        if block is not None:
            n = words(" ".join(block["lines"]))
            if n > block["cap"]:
                errors.append(
                    f"{rel}:{block['start']}: {block['what']} is {n} words, cap {block['cap']}"
                )
        block = None

    for i, line in enumerate(path.read_text().splitlines(), 1):
        if block is not None:
            if block["kind"] == "quote" and line.startswith(">") and line.strip() != ">":
                block["lines"].append(line)
                continue
            if block["kind"] == "entry" and line.strip() and not line.startswith(("#", "|", ">")):
                block["lines"].append(line)  # an entry is one paragraph: it runs to a blank line
                continue
            close()

        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            title = line.strip("# ").strip()
            if level == 1:
                section, frozen, changelog, region = None, False, False, None
            elif level == 2:  # only a top-level section can open or close a frozen region
                section = title
                region = region_of(title, region)
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
            if line.strip() == ">":
                continue  # a quoted blank line separates paragraphs, each budgeted alone
            cap = CAPS["blockquote"] if section else CAPS["intro"]
            what = f"{section} blockquote" if section else "intro blockquote"
            block = {"kind": "quote", "start": i, "lines": [line], "cap": cap, "what": what}
            continue

        if changelog and line.strip().startswith("**"):
            block = {
                "kind": "entry",
                "start": i,
                "lines": [line],
                "cap": CAPS["changelog"],
                "what": "change-log entry",
            }
            continue

        if not line.startswith("|") or RULE.match(line):
            continue

        cells = cells_of(line)
        if "Description" in cells:  # a header row: this table is in scope
            idx, width, header_line = cells.index("Description"), len(cells), i
            header = cells
            task_table = cells[0] in ("TaskID", "Task ID")
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

        if task_table and region:
            want = STATUS[region]
            bare = LIFECYCLE.sub("", cells[0]).lstrip("\ufe0f").strip()
            if bare == "(none)":
                if LIFECYCLE.match(cells[0]):
                    errors.append(f"{rel}:{i}: a '(none)' placeholder row carries no status icon")
            elif not (cells[0].startswith(want + " ") or (region == "progress" and cells[0].startswith(want * 2 + " "))):
                errors.append(
                    f"{rel}:{i}: {cells[0][:30]} must open with '{want} ' before the task ID"
                    f" (a row under {REGION_NAME[region]}; ROADMAP_STANDARDS.md section 2)"
                )
            if LIFECYCLE.match(cells[idx]):
                errors.append(
                    f"{rel}:{i}: {cells[0][:30]} description opens with a status icon;"
                    " the status is said once, beside the task ID"
                )

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

    close()  # a block that runs to the end of the file
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
