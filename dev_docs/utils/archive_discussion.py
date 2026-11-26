#!/usr/bin/env python3
"""Keep a verbatim, append-only copy of a GitHub discussion thread inside this repository.

A thread that carries registered predictions, plans posted before their runs and the answers to
them lives in a repository its participants do not own. If that repository, the discussion or one
comment is deleted, the record goes with it, and so does every link that points at it. This script
copies the thread into plain-text files, one per version of each post, and never rewrites a file it
has written.

What it writes into --out:

  <posted>_<id>_<login>.txt                    a post as first posted
  <posted>_<id>_<login>_edited-<when>.txt      the same post after an edit (one file per edit)
  _INDEX.md                                    a table of every file, rebuilt on each run

Each file is a short header, a separator line, then the text of the post. The text is what the
GitHub API returns, with CRLF line ends turned into LF and one LF added at the end when the text has
none. `sha256` in the header is the hash of every byte of the file after the separator line (UTF-8).
Earlier versions of an edited post come from the edit history the API keeps.

Rules it holds itself to:

  - a file that exists is never changed; a post whose text differs from its archived file with no
    edit on record stops the run with exit code 1 and writes nothing for that post;
  - a post that is gone from the live thread keeps its files, and the index names it;
  - a reply list or an edit history longer than one page stops the run instead of being cut short.

Usage:
  archive_discussion.py --repo OWNER/NAME --number N --out DIR [--raw DIR]
  archive_discussion.py --repo OWNER/NAME --number N --out DIR --from-raw DIR     (no network)
  archive_discussion.py --selftest

--raw keeps the API pages as JSON (a local folder outside the tracked tree). The network read uses
the `gh` command line tool with the reader's own login; nothing is written to GitHub.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SEP = "==== verbatim text below this line ===="
EDITS = "userContentEdits(first:50){ totalCount nodes{ editedAt diff } }"
POST = "url createdAt includesCreatedEdit author{ login } body " + EDITS
QUERY = (
    "query($owner:String!,$name:String!,$number:Int!,$c:String){ repository(owner:$owner,name:$name){"
    " discussion(number:$number){ number title "
    + POST
    + " comments(first:40, after:$c){ totalCount"
    " pageInfo{ hasNextPage endCursor } nodes{ databaseId "
    + POST
    + " replies(first:100){ totalCount"
    " nodes{ databaseId " + POST + " } } } } } } }"
)


def fetch(repo, number, raw):
    owner, name = repo.split("/")
    pages, cursor = [], None
    while True:
        cmd = ["gh", "api", "graphql", "-f", "query=" + QUERY, "-f", "owner=" + owner]
        cmd += ["-f", "name=" + name, "-F", f"number={number}"]
        if cursor:
            cmd += ["-f", "c=" + cursor]
        page = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
        pages.append(page)
        if raw:
            Path(raw).mkdir(parents=True, exist_ok=True)
            (Path(raw) / f"page_{len(pages)}.json").write_text(json.dumps(page), encoding="utf-8")
        info = page["data"]["repository"]["discussion"]["comments"]["pageInfo"]
        if not info["hasNextPage"]:
            return pages
        cursor = info["endCursor"]


def load_raw(raw):
    files = sorted(Path(raw).glob("page_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


def posts(pages, repo):
    """Every post of the thread: the opening post, the comments, their replies."""
    d = pages[0]["data"]["repository"]["discussion"]
    thread = f"{repo} discussion #{d['number']}"
    yield dict(node=d, ident=f"discussion-{d['number']}", kind="opening post", thread=thread)
    total = seen = 0
    for page in pages:
        c = page["data"]["repository"]["discussion"]["comments"]
        total = c["totalCount"]
        for n in c["nodes"]:
            seen += 1
            yield dict(node=n, ident=str(n["databaseId"]), kind="comment", thread=thread)
            r = n["replies"]
            if r["totalCount"] > len(r["nodes"]):
                sys.exit(f"STOP: comment {n['databaseId']} has more replies than one page holds")
            for q in r["nodes"]:
                kind = f"reply to discussioncomment-{n['databaseId']}"
                yield dict(node=q, ident=str(q["databaseId"]), kind=kind, thread=thread)
    if seen != total:
        sys.exit(f"STOP: the thread declares {total} comments and the pages hold {seen}")


def norm(text):
    text = text.replace("\r\n", "\n")
    return text if text.endswith("\n") else text + "\n"


def stamp(iso):
    return iso.replace(":", "")


def versions(node):
    """(label, when, text) per known version, oldest first; the label is None for the first post."""
    e = node["userContentEdits"]
    if e["totalCount"] > len(e["nodes"]):
        sys.exit(f"STOP: {node['url']} has more edits than one page holds")
    body = norm(node["body"])
    hist = sorted((x for x in e["nodes"] if x["diff"] is not None), key=lambda x: x["editedAt"])
    out = []
    for k, x in enumerate(hist):
        first = k == 0 and node.get("includesCreatedEdit") and len(hist) == e["totalCount"]
        out.append((None if first else x["editedAt"], x["editedAt"], norm(x["diff"])))
    if not out:
        if e["totalCount"] > (1 if node.get("includesCreatedEdit") else 0):
            last = max(x["editedAt"] for x in e["nodes"])
            return [(last, last, body)]
        return [(None, node["createdAt"], body)]
    if out[-1][2] != body:
        sys.exit(f"STOP: {node['url']}: the newest edit on record is not the current text")
    return out


def header_of(path):
    head = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == SEP:
                return head
            key, _, val = line.partition(": ")
            head[key] = val
    sys.exit(f"STOP: {path} has no separator line")


def archive(pages, repo, out, today):
    """Write what is new. Returns (files written, conflicts, identifiers seen in the live thread)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written, conflicts, live = [], [], set()
    for p in posts(pages, repo):
        n = p["node"]
        login = (n["author"] or {}).get("login", "ghost")
        base = f"{stamp(n['createdAt'])}_{p['ident']}_{login}"
        live.add(p["ident"])
        for label, when, text in versions(n):
            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            name = base + (".txt" if label is None else f"_edited-{stamp(label)}.txt")
            path = out / name
            if path.exists():
                if header_of(path).get("sha256") != sha:
                    conflicts.append(name)
                continue
            head = [
                f"source: {n['url']}",
                f"thread: {p['thread']}",
                f"kind: {p['kind']}",
                f"id: {p['ident']}",
                f"author: {login}",
                f"posted: {n['createdAt']}",
                "version: as first posted" if label is None else f"version: edited {when}",
                f"sha256: {sha}",
                f"archived: {today}",
                SEP,
            ]
            path.write_bytes(("\n".join(head) + "\n" + text).encode("utf-8"))
            written.append(name)
    return written, conflicts, live


def write_index(out, repo, number, live):
    out = Path(out)
    rows, groups = [], {}
    for path in sorted(out.glob("*.txt")):
        h = header_of(path)
        groups.setdefault((h["posted"], h["id"]), []).append((path.name, h))
    gone = []
    for (posted, ident), files in sorted(groups.items()):
        h = files[0][1]
        links = []
        for name, hh in files:
            word = "as posted" if hh["version"] == "as first posted" else hh["version"]
            links.append(f"[{word}]({name})")
        if live is not None and ident not in live:
            gone.append(ident)
        rows.append(
            f"| {posted} | {h['author']} | {h['kind']} | {', '.join(links)} | [live]({h['source']}) |"
        )
    script = os.path.relpath(os.path.abspath(__file__), out.resolve()).replace(os.sep, "/")
    text = [
        f"# Verbatim archive of discussion #{number} of {repo}: index",
        "",
        f"> Rebuilt by [`archive_discussion.py`]({script})"
        " on each run. One row per post, oldest first; an edited post lists one file per version."
        " What the files are and how they are made: [`README.md`](README.md).",
        "",
        f"Posts: {len(groups)}. Files: {sum(len(v) for v in groups.values())}."
        f" Latest post: {max(k[0] for k in groups) if groups else 'none'}.",
        "",
        "No longer in the live thread at the last run: "
        + (", ".join(gone) if gone else "none")
        + ".",
        "",
        "| Posted (UTC) | Author | Kind | Files | Live |",
        "| --- | --- | --- | --- | --- |",
    ]
    (out / "_INDEX.md").write_text("\n".join(text + rows) + "\n", encoding="utf-8")
    return gone


def run(pages, repo, number, out, today):
    written, conflicts, live = archive(pages, repo, out, today)
    gone = write_index(out, repo, number, live)
    print(
        f"archive: {len(written)} file(s) written, {len(conflicts)} conflict(s), {len(gone)} gone"
    )
    for name in conflicts:
        print(f"CONFLICT: {name} exists with another text and no edit on record; left as it is")
    return 1 if conflicts else 0


def selftest():
    def post(ident, text, edits=()):
        nodes = [dict(editedAt=t, diff=x) for t, x in edits]
        e = dict(totalCount=len(nodes), nodes=nodes)
        d = dict(url=f"https://example.invalid/d#discussioncomment-{ident}", body=text)
        d.update(
            createdAt="2026-01-01T00:00:00Z",
            includesCreatedEdit=bool(nodes),
            author=dict(login="a"),
        )
        d.update(userContentEdits=e, databaseId=ident)
        return d

    def thread(comments):
        for c in comments:
            c.setdefault("replies", dict(totalCount=0, nodes=[]))
        cs = dict(totalCount=len(comments), nodes=comments, pageInfo=dict(hasNextPage=False))
        d = post(0, "opening")
        d.update(number=1, title="t", comments=cs)
        return [dict(data=dict(repository=dict(discussion=d)))]

    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        one = thread([post(1, "first\r\nline"), post(2, "second")])
        w, c, _ = archive(one, "o/n", out, "2026-01-02")
        write_index(out, "o/n", 1, None)
        index = (out / "_INDEX.md").read_text()
        check("first run writes one file per post", len(w) == 3 and not c)
        w, c, _ = archive(one, "o/n", out, "2026-01-03")
        write_index(out, "o/n", 1, None)
        check(
            "second run writes nothing and the index is unchanged",
            not w and index == (out / "_INDEX.md").read_text(),
        )
        w0 = "2026-01-01T000000Z_2_a.txt"
        kept = (out / w0).read_bytes()
        hist = [("2026-01-01T00:00:00Z", "second"), ("2026-01-05T00:00:00Z", "second, edited")]
        two = thread([post(1, "first\r\nline"), post(2, "second, edited", hist)])
        w, c, _ = archive(two, "o/n", out, "2026-01-06")
        check(
            "an edit adds one file and leaves the first untouched",
            w == ["2026-01-01T000000Z_2_a_edited-2026-01-05T000000Z.txt"]
            and kept == (out / w0).read_bytes(),
        )
        bad = thread(
            [post(1, "first, changed with no edit on record"), post(2, "second, edited", hist)]
        )
        w, c, _ = archive(bad, "o/n", out, "2026-01-07")
        check(
            "a changed text with no edit on record is a conflict and writes nothing",
            c == ["2026-01-01T000000Z_1_a.txt"] and not w,
        )
        w, c, live = archive(thread([post(2, "second, edited", hist)]), "o/n", out, "2026-01-08")
        gone = write_index(out, "o/n", 1, live)
        check(
            "a post gone from the thread keeps its file and is named in the index",
            gone == ["1"] and (out / "2026-01-01T000000Z_1_a.txt").exists(),
        )
        body = (out / "2026-01-01T000000Z_1_a.txt").read_bytes().split((SEP + "\n").encode())[1]
        sha = hashlib.sha256(body).hexdigest()
        check(
            "the hash in the header is that of the stored text",
            header_of(out / "2026-01-01T000000Z_1_a.txt")["sha256"] == sha and b"\r" not in body,
        )
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", help="OWNER/NAME of the repository that hosts the discussion")
    ap.add_argument("--number", type=int, help="the discussion number")
    ap.add_argument("--out", help="the folder of the archive")
    ap.add_argument("--raw", help="keep the API pages as JSON in this folder")
    ap.add_argument("--from-raw", help="read the API pages from this folder, no network")
    ap.add_argument("--date", help="the date written as `archived` (default: today, UTC)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.repo and a.number and a.out):
        ap.error("--repo, --number and --out are required")
    import datetime

    today = a.date or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    pages = load_raw(a.from_raw) if a.from_raw else fetch(a.repo, a.number, a.raw)
    return run(pages, a.repo, a.number, a.out, today)


if __name__ == "__main__":
    sys.exit(main())
