# T12: A verbatim archive of an external discussion thread

> Platform task **T12**. Status: ✅ **CLOSED 2026-11-25** (filed and closed the same day, on the repo owner's ruling) · Roadmap: [`../platform_roadmap.md`](../platform_roadmap.md) · Deliverables: the tool [`../utils/archive_discussion.py`](../utils/archive_discussion.py) and the first archive, [`archive/d186/`](../../openwave/xperiments/m5_liquid_crystal/research/archive/d186/README.md) of the M5 research folder.

## PLANNING

**Scope.** The M5 work is coordinated with the model's author on a public discussion hosted in a repository that neither party owns. Registered predictions, plans posted before their runs and the answers to them exist only there, and the thread record of M5 holds a reading of each post with a link, not its words. A deleted repository, discussion or comment would take the record and every link with it. This task keeps a copy inside OpenWave: every post, as posted and after each edit, in files that are added and never rewritten.

**The repo owner's rulings (2026-11-25).** The whole thread is copied, every participant included. Names appear in the archive as posted. The copies are plain text.

**Definition of done.** A tool that reads a discussion through the GitHub API and writes one file per version of each post, with a self-test that includes an input it must refuse; the first archive of the thread; an index; the rule that the tool runs at every landing that follows a post or a reading of a new post; pointers from the thread record and the M5 roadmap.

**Gating.** None beyond the rulings above.

**Blindspot pass.** (i) An edit can remove what a post said when a run was planned on it: the edit history is copied, one file per version. (ii) A copy that can be rewritten is no record: a file that exists is never changed, and a text that differs with no edit on record stops the tool. (iii) A hash is only checkable with its encoding: the header states which bytes it covers, and a shell line in the archive's README recomputes it. (iv) Verbatim posts break this repository's writing rules, so they are plain text, which the doc checker and the linter do not read. (v) A private string could ride in through an old version of a post: every file of the first archive, edit histories included, was searched for machine paths, mail addresses and private names before it was staged. (vi) A copy kept by one party is that party's record, not a third party's timestamp: the archive's README says so.

## FINDINGS (2026-11-25)

| Deliverable | Where | State |
| --- | --- | --- |
| The tool | [`archive_discussion.py`](../utils/archive_discussion.py): the thread read through `gh api graphql` with the reader's own login, nothing written to GitHub; one file per version; an index rebuilt on each run; `--raw` keeps the API pages in a local folder, `--from-raw` reruns without the network | ✅ |
| The self-test | `--selftest`, six checks on a made-up thread: one file per post; a second run writes nothing and leaves the index unchanged; an edit adds one file and leaves the first untouched; a changed text with no edit on record is refused and writes nothing (the input it must fail on); a deleted post keeps its file and is named in the index; the hash in the header is that of the stored text | ✅ |
| The first archive | [`archive/d186/`](../../openwave/xperiments/m5_liquid_crystal/research/archive/d186/_INDEX.md): 168 posts by six participants in 190 files (22 of them earlier or later versions of 18 edited posts), from the opening post of 2026-08-29 to the post of 2026-11-25 22:19 UTC; a second run on the saved pages wrote nothing | ✅ |
| The search before staging | no machine path, no mail address and no private name in any file, edit histories included; the path checker clean on all of them | ✅ |
| The hash, checked outside the tool | one file's header hash recomputed with `tail` and `shasum` | ✅ |
| The rule | the archive's [`README.md`](../../openwave/xperiments/m5_liquid_crystal/research/archive/d186/README.md) and the conventions of the [M5 roadmap](../../openwave/xperiments/m5_liquid_crystal/research/m5_roadmap.md): the tool runs at every landing that follows a post from OpenWave or a reading of a new post | ✅ |

**Not done.** The 52 images and files attached to posts are links to GitHub's servers and are not copied. No copy is lodged with a third party. The discussions hosted in OpenWave's own repository are not archived by this task.

**Scope**: a copy of the words of one thread as the API returned them on the day of each run, with the edit history GitHub kept; it is the maintainers' own record and proves a date only through the commit that carries it.
