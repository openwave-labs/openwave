# Verbatim archive of discussion #186

> A copy, post by post, of the public discussion on which the M5 Lagrangian hunt is coordinated with the model's author: [vantasnerdan/substrate-framework discussion #186](https://github.com/vantasnerdan/substrate-framework/discussions/186). The index of every file: [`_INDEX.md`](_INDEX.md). The reading of each post, with what it changed on OpenWave: the thread record [`../../tasks/m5_32_convo.md`](../../tasks/m5_32_convo.md). Platform task: [T12](../../../../../../dev_docs/tasks/t12_task_details.md).

**Why it exists.** The thread carries predictions registered before their runs, plans posted before they were run, and the answers to both. It lives in a repository that neither OpenWave nor the model's author owns. If that repository, the discussion or one comment is deleted, the record and every link to it go with it. This folder keeps the words.

**What a file is.** One plain-text file per version of a post. A post that was edited has one file as first posted and one per edit, from the edit history the GitHub API keeps.

| Header Line | What It Holds |
| --- | --- |
| `source` | the link of the post on GitHub |
| `thread` | the repository and the number of the discussion |
| `kind` | `opening post`, `comment`, or `reply to` the comment it answers |
| `id` | the number GitHub gives the comment |
| `author` | the login that posted it |
| `posted` | when it was posted (UTC) |
| `version` | `as first posted`, or `edited` with the time of that edit |
| `sha256` | the hash of every byte of the file after the separator line |
| `archived` | the day the file was written |

After the header comes one separator line, then the text as the API returns it, with CRLF line ends turned into LF and one LF added at the end when the text has none. To check a file:

```text
tail -n +11 <file> | shasum -a 256
```

**The rules of this folder.**

| Rule | What It Means |
| --- | --- |
| Written by the tool only | [`archive_discussion.py`](../../../../../../dev_docs/utils/archive_discussion.py) writes the files and rebuilds the index. Nothing here is edited by hand |
| Append-only | A file that exists is never changed. A post whose text differs from its file with no edit on record stops the tool |
| A deleted post stays | A post that is gone from the live thread keeps its files, and the index names it |
| Verbatim | The files are records, not documents of this repository: its writing rules, its link rules and its naming of people do not apply inside them. They are plain text for that reason |
| Each post is its author's | The posts are public and are copied with their author's login and a link to the original. Copying them here claims nothing about them |
| Run at every landing | The tool runs whenever a post is made on the thread from OpenWave or a new post is read, and its new files ride that pull request |

**What it does not hold.** Images and files attached to posts are links to GitHub's servers and are not copied. The edit history is what the API returned on the day of the copy. A copy kept by one party of a thread is that party's record: the date of a file is the date of its commit in this repository, not a third party's timestamp.

**How to run it.**

```text
python3 dev_docs/utils/archive_discussion.py --repo vantasnerdan/substrate-framework --number 186 --out openwave/xperiments/m5_liquid_crystal/research/archive/d186
```
