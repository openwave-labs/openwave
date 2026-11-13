"""Measure what MODELS.md cells and roadmap Description cells actually cost a reader.

The cell budgets in MODELS.md and ROADMAP_STANDARDS.md are visualization limits:
how much a table cell may carry before the table stops being scannable. This
script supplies the evidence a budget change should rest on, so the number can be
re-derived after a structural change instead of inherited across one (T3).

Three measures per cell, because the two live checkers do NOT count the same thing:

  counted-words   what that file's own linter counts, verbatim:
                  - MODELS.md (check_models_md.prose_words): drops the leading
                    "<icon> [status]" tag, drops WHOLE links (label included),
                    drops "<br>-> ..." pointer tails
                  - roadmaps (check_roadmaps.words): keeps link LABELS, drops
                    only the targets, keeps everything else
  prose+links     the roadmap rule applied to BOTH files: prose plus link
                  labels, status tag and pointer tails still excluded. This
                  isolates the one substantive difference between the linters
                  and is the number to compare across files
  read-words      every visible word a reader sees, tags and pointer tails
                  included: the full reading load of the cell
  read-chars      visible characters under the same rule, the closest proxy for
                  how tall the rendered cell grows in its column

Because MODELS.md excludes freight that roadmaps count, equal counted-word caps
do NOT mean equal cells. The ratios printed per file are how much bigger a cell
renders than its budget claims.

The second half of the derivation is COLUMN GEOMETRY. A budget is a limit on how
many rendered LINES a cell occupies, and lines = characters / column width. A
browser sizes each column once for the whole table, roughly in proportion to its
max-content, so the two-column per-model tables give their summary column a much
larger share than a four-column roadmap gives its Description. Reading load and
column geometry pull in opposite directions and the honest answer is a bracket,
not a point; the report prints both ends.

Usage: python3 dev_docs/utils/models_cell_stats.py [--csv]
Reads MODELS.md and every roadmap check_roadmaps.py knows about. No writes.
"""

import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_roadmaps as CR  # noqa: E402  (path set above)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "MODELS.md"

LINK_KEEP_LABEL = re.compile(r"\[([^\]]*)\]\([^)]*\)")
LINK_WHOLE = re.compile(r"\[[^\]]*\]\([^)]*\)")
TAG = re.compile(r"^\s*[^\[]*\[[^\]]*\]")
BREAK = re.compile(r"<br\s*/?>")
MARKUP = re.compile(r"[`*_#]")
RULE = re.compile(r"^\|[\s\-:|]+\|$")
SPLIT = re.compile(r"(?<!\\)\|")
MODEL_HEAD = re.compile(r"^#{2,4} .*\((M\d+)\)\s*$")
MODEL_COL = re.compile(r"^\[(M\d+)\]\(#")


def cells_of(line):
    return [c.strip() for c in SPLIT.split(line.strip())[1:-1]]


def read_text(cell):
    """What a reader sees: link labels kept, targets gone, markup stripped."""
    return MARKUP.sub("", BREAK.sub(" ", LINK_KEEP_LABEL.sub(r"\1", cell))).strip()


def models_body(cell):
    """The cell minus the status tag and minus the '<br>-> ...' pointer tails."""
    segs = [s for s in cell.split("<br>") if not s.strip().startswith("→")]
    return TAG.sub("", " ".join(segs), count=1)


def models_counted(cell):
    """check_models_md.prose_words, reproduced so the two stay comparable here."""
    body = LINK_WHOLE.sub(" ", models_body(cell))
    body = re.sub(r"[`*]", "", body)
    return [w for w in body.split() if re.search(r"[0-9A-Za-zα-ωΑ-Ω]", w)]


def prose_plus_links(cell, strip_tag):
    """The roadmap counting rule: prose plus link LABELS. Tag/tails excluded."""
    body = models_body(cell) if strip_tag else cell
    return CR.words(body)


def n_links(cell):
    return len(LINK_WHOLE.findall(cell))


def models_cells():
    """Every summary cell of the per-model RESULTS BY MODEL tables."""
    out = []  # (criterion, model, counted, prose+links, read_words, read_chars, links)
    mode = None
    model = None
    seen = None
    for line in MODELS.read_text().splitlines():
        head = MODEL_HEAD.match(line)
        if head:
            seen = head.group(1)
        if not line.startswith("|"):
            if line.startswith("#"):
                mode = None
            continue
        cells = cells_of(line)
        if RULE.match(line) or not cells:
            continue
        if cells[0] == "Criteria":
            n_model_cols = sum(1 for j, c in enumerate(cells) if j and MODEL_COL.match(c))
            if len(cells) == 2 and n_model_cols == 0:
                mode, model = "model", seen
            else:
                mode = None
            continue
        if mode != "model" or model is None or len(cells) != 2:
            continue
        if cells[0].startswith("**") and not cells[1].strip():
            continue  # group divider
        vis = read_text(cells[1])
        out.append((
            cells[0], model, len(models_counted(cells[1])),
            prose_plus_links(cells[1], strip_tag=True),
            len(vis.split()), len(vis), n_links(cells[1]),
        ))
    return out


def roadmap_cells():
    """Every Description cell of every roadmap, frozen sections skipped."""
    out = []  # (taskid, file, counted, prose+links, read_words, read_chars, links)
    for path in CR.roadmaps():
        idx = width = None
        frozen = False
        for line in path.read_text().splitlines():
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                if level == 1:
                    frozen = False
                elif level == 2:
                    frozen = any(w in line.strip("# ").upper() for w in CR.FROZEN)
                idx = width = None
                continue
            if frozen or not line.startswith("|") or RULE.match(line):
                if not line.startswith("|"):
                    idx = width = None
                continue
            cells = cells_of(line)
            if "Description" in cells:
                idx, width = cells.index("Description"), len(cells)
                continue
            if idx is None or len(cells) != width:
                continue
            vis = read_text(cells[idx])
            out.append((
                cells[0], path.name, CR.words(cells[idx]),
                prose_plus_links(cells[idx], strip_tag=False),
                len(vis.split()), len(vis), n_links(cells[idx]),
            ))
    return out


def pct(values, p):
    if not values:
        return 0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def report(label, rows, cap):
    counted = [r[2] for r in rows]
    plinks = [r[3] for r in rows]
    rwords = [r[4] for r in rows]
    rchars = [r[5] for r in rows]
    links = [r[6] for r in rows]
    print(f"\n{label}  (n = {len(rows)}, stated cap = {cap} counted-words)")
    print(f"{'measure':<14}{'median':>8}{'mean':>8}{'p90':>8}{'p95':>8}{'max':>8}")
    for name, v in (
        ("counted-words", counted),
        ("prose+links", plinks),
        ("read-words", rwords),
        ("read-chars", rchars),
        ("links/cell", links),
    ):
        print(
            f"{name:<14}{statistics.median(v):>8.0f}{statistics.mean(v):>8.1f}"
            f"{pct(v, 90):>8}{pct(v, 95):>8}{max(v):>8}"
        )
    pl = [p / c for c, p in zip(counted, plinks) if c]
    rd = [w / c for c, w in zip(counted, rwords) if c]
    print(
        f"vs counted     prose+links median {statistics.median(pl):.2f} max {max(pl):.2f}"
        f"  |  read median {statistics.median(rd):.2f} max {max(rd):.2f}"
    )
    print(
        f"a cell AT the cap carries ~{cap * statistics.median(pl):.0f} prose+link words"
        f" and reads as ~{cap * statistics.median(rd):.0f} words"
    )
    over = [r for r in rows if r[2] > cap]
    print(f"over cap: {len(over)}" + (f"  {[r[0] for r in over]}" if over else ""))
    return counted, plinks, rwords, rchars


def histogram(counted, edges):
    lo = 0
    for hi in edges:
        n = sum(1 for c in counted if lo <= c <= hi)
        print(f"  {lo:>3}-{hi:<3} {'#' * min(n, 60)} {n}")
        lo = hi + 1


def column_widths():
    """Per-TABLE column max-content, which is what sizes a column in auto layout.

    Returns (models_share, roadmap_share): the median fraction of its table's
    width taken by the MODELS.md summary column and by the roadmap Description
    column. Measured per table, never per row, because a browser sizes a column
    once for the whole table.
    """
    mt = []
    cur = None
    mode = None
    seen = None
    for line in MODELS.read_text().splitlines():
        head = MODEL_HEAD.match(line)
        if head:
            seen = head.group(1)
        if not line.startswith("|"):
            if line.startswith("#"):
                if cur:
                    mt.append(cur)
                cur, mode = None, None
            continue
        cells = cells_of(line)
        if not cells or RULE.match(line):
            continue
        if cells[0] == "Criteria":
            if cur:
                mt.append(cur)
            cur = None
            two_col = len(cells) == 2 and cells[1].strip("* `").lower() != "simplest test"
            mode, cur = ("model", [0, 0]) if (two_col and seen) else (None, None)
            continue
        if mode == "model" and cur and len(cells) == 2:
            cur = [max(cur[0], len(read_text(cells[0]))), max(cur[1], len(read_text(cells[1])))]
    if cur:
        mt.append(cur)

    rt = []
    for path in CR.roadmaps():
        cur = idx = width = None
        frozen = False
        for line in path.read_text().splitlines():
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                if level == 1:
                    frozen = False
                elif level == 2:
                    frozen = any(w in line.strip("# ").upper() for w in CR.FROZEN)
                if cur and idx is not None:
                    rt.append((cur, idx))
                cur = idx = None
                continue
            if frozen or not line.startswith("|") or RULE.match(line):
                if not line.startswith("|"):
                    if cur and idx is not None:
                        rt.append((cur, idx))
                    cur = idx = None
                continue
            cells = cells_of(line)
            if "Description" in cells:
                if cur and idx is not None:
                    rt.append((cur, idx))
                idx, width = cells.index("Description"), len(cells)
                cur = [len(read_text(c)) for c in cells]
                continue
            if idx is None or len(cells) != width:
                continue
            cur = [max(a, len(read_text(c))) for a, c in zip(cur, cells)]
        if cur and idx is not None:
            rt.append((cur, idx))

    ms = [t[1] / sum(t) for t in mt if sum(t)]
    rs = [t[i] / sum(t) for t, i in rt if sum(t)]
    return statistics.median(ms), statistics.median(rs), len(mt), len(rt)


def companion_cells():
    """The simplest-test table: a scope check, it carries no budget by design."""
    out = []
    mode = None
    for line in MODELS.read_text().splitlines():
        if line.startswith("#"):
            mode = None
            continue
        if not line.startswith("|") or RULE.match(line):
            continue
        cells = cells_of(line)
        if not cells:
            continue
        if cells[0] == "Criteria":
            mode = (
                "tests"
                if len(cells) == 2 and cells[1].strip("* `").lower() == "simplest test"
                else None
            )
            continue
        if mode == "tests" and len(cells) == 2 and cells[1].strip():
            out.append(len(read_text(cells[1]).split()))
    return out


def main():
    m = models_cells()
    r = roadmap_cells()
    mc, mpl, mw, mch = report("MODELS.md per-model summary cells", m, 65)
    rc, rpl, rw, rch = report("Roadmap Description cells", r, 65)

    print("\nMODELS.md counted-word histogram")
    histogram(mc, [20, 35, 45, 50, 55, 65, 80, 200])
    print("\nRoadmap counted-word histogram")
    histogram(rc, [20, 35, 45, 50, 55, 65, 80, 200])

    tests = companion_cells()
    print(
        f"\nSimplest-test companion table (no budget by design): n = {len(tests)},"
        f" median {statistics.median(tests):.0f}, max {max(tests)} visible words"
    )

    print("\nHOW BIG IS A CELL AT ITS CAP, in each file?")
    near_chars = {}
    for label, rows, lo, hi in (
        ("MODELS.md 51-55", m, 51, 55),
        ("roadmap  61-65", r, 61, 65),
    ):
        near = [x for x in rows if lo <= x[2] <= hi]
        rw_ = [x[4] for x in near]
        rch_ = [x[5] for x in near]
        near_chars[label] = statistics.median(rch_)
        print(
            f"  {label}: n = {len(near)},"
            f" read-words median {statistics.median(rw_):.0f} max {max(rw_)},"
            f" read-chars median {statistics.median(rch_):.0f} max {max(rch_)}"
        )

    m_share, r_share, n_mt, n_rt = column_widths()
    ratio = m_share / r_share
    road_cap_chars = near_chars["roadmap  61-65"]
    chars_per_word = near_chars["MODELS.md 51-55"] / 53  # measured at the old 55 cap
    load_cap = 65 * statistics.median([w / c for c, w in zip(rc, rw) if c]) / statistics.median(
        [w / c for c, w in zip(mc, mw) if c]
    )
    geom_cap = road_cap_chars * ratio / chars_per_word

    print("\nCOLUMN GEOMETRY (per-table max-content share, n ="
          f" {n_mt} model tables, {n_rt} roadmap tables)")
    print(f"  MODELS.md summary column takes  {m_share:.3f} of its table's width")
    print(f"  roadmap Description column takes {r_share:.3f} of its table's width")
    print(f"  width ratio {ratio:.2f}x: the two-column layout is the room the split bought")

    print("\nEQUIVALENT-CAP BRACKET for MODELS.md, against the roadmap's 65")
    print(f"  reading load alone (geometry ignored):  {load_cap:.0f} counted words")
    print(f"  column geometry alone (equal lines):    {geom_cap:.0f} counted words")
    print("  The two pull opposite ways, so the answer is a bracket. The standard sits"
          " at 65, inside it. Re-run this before moving the number again.")

    if "--csv" in sys.argv:
        print("\nkey,file_or_model,counted,prose_links,read_words,read_chars,links")
        for row in m + r:
            print(",".join(str(x).replace(",", ";") for x in row))


if __name__ == "__main__":
    main()
