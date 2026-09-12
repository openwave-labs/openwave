# Contributing to OpenWave

Thank you for your interest in contributing!  
Whether you're fixing a typo, adding a feature, or reporting a bug, your help makes OpenWave better for everyone.

## How You Can Contribute

- **Report Issues:** If you find a bug, open an issue describing the problem and how to reproduce it. The issue tracker is for **platform issues**: something reproducible and closable that is wrong with the platform itself, such as an engine bug, a checker that misfires, an install or reproduction failure, or a document that states something untrue.
- **Pick up open work:** Research tasks are not issues. Each model column keeps its own roadmap (`openwave/xperiments/<model>/research/<mid>_roadmap.md`, plus [`dev_docs/platform_roadmap.md`](dev_docs/platform_roadmap.md) for shared work), where every queued task links a task document with its scope and its pass/fail criteria. Say in a discussion which row you are taking, then open a pull request against it. The reason tracking lives in files: a roadmap edit is a tracked diff, which is what a file several authors touch needs.
- **Suggest Features:** Share ideas for new features, improvements, or a research direction in [Discussions](https://github.com/openwave-labs/openwave/discussions); one becomes a roadmap row when someone scopes it into work.
- **Improve Documentation:** Help us make guides, examples, and API references clearer.
- **Write Code:** Fix bugs, add features, or improve existing code.
- **Run the science yourself:** Validate, recompute, or try to falsify any cell in [MODELS.md](MODELS.md), then open a pull request with the script and the note behind the result.

**Bring your own compute (BYOC).** The runs behind a contribution are supplied by whoever makes them, in AI tokens and hardware, rather than pooled through a maintainer: [ONBOARDING_MODELS.md § Bring your own compute](ONBOARDING_MODELS.md#bring-your-own-compute-byoc) states the contract for model authors and pull-request contributors alike.

## Practice the Community Code

- Be respectful and constructive.
- Follow the [OpenWave Code of Conduct](./CODE_OF_CONDUCT.md).
- Ask questions, we’re here to help each other.
- Read this Contribution Guide
- Contributing with AI assistance (most of us do)? Read [`AI_HYGIENE.md`](AI_HYGIENE.md) first: the dos, don'ts, and verification habits that keep every claim script-backed and human-owned

See `/dev_docs` for coding standards and development guidelines

- [Coding Standards](dev_docs/CODING_STANDARDS.md)
- [Performance Guidelines](dev_docs/PERFORMANCE_GUIDELINES.md)
- [Loop Optimization Patterns](dev_docs/LOOP_OPTIMIZATION.md)
- [Markdown Style Guide](dev_docs/MARKDOWN_STYLE_GUIDE.md)
- [PR Review Standards](dev_docs/PR_REVIEW_STANDARDS.md): what a maintainer checks when reviewing your PR, so nothing in the review is a surprise
- [AI Hygiene](AI_HYGIENE.md): working with automated intelligence, the dos, don'ts, and verification habits that keep the science human-owned  

*This is the Way!*

## Getting Started

- **Fork the Repository**  
  - Click “Fork” on GitHub to create your own copy.

- **Clone Your Fork**

```bash
   git clone https://github.com/YOUR-USERNAME/openwave.git
   cd openwave
   ```

- **Set Up the Environment & Install**

```bash
# Create virtual environment
  # Option 1: via Venv
    python -m venv openwave
    source openwave/bin/activate  # On Windows: openwave\Scripts\activate
   
  # Option 2: via Conda (recommended)
    conda create -n openwave python=3.12
    conda activate openwave

# Install OpenWave & Dependencies for Development (-e = edit mode)
   pip install -e ".[dev]"  # dependencies from pyproject.toml, plus the formatter below

# Activate the auto-DCO-sign-off git hook (one-time per clone)
   git config core.hooksPath .githooks
   ```

- **Create a Branch to Develop Your Feature**

```bash
   git checkout -b your-feature-name
   ```

- Optional: LaTex & FFmpeg (video generation)

```bash
# Install LaTeX and FFmpeg (macOS)
   brew install --cask mactex-no-gui ffmpeg
   echo 'export PATH="/Library/TeX/texbin:$PATH"' >> ~/.zshrc
   exec zsh -l

# Verify LaTeX installation
   which latex && latex --version
   which dvisvgm && dvisvgm --version
   which gs && gs --version
```

## Code Style & Quality

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Use [Black](https://black.readthedocs.io/) and [isort](https://pycqa.github.io/isort/) for formatting. Black is configured at `line-length = 99` in `pyproject.toml` and ships in the `dev` extra installed above, so `black .` from the repo root is already set up. isort is not yet configured or bundled; if you run it, pass `--profile black` so its import ordering does not fight the formatter.
- Run tests before committing:

```python
  pytest
  ```

### Self-checks a script prints

A line a script prints as `PASS`, `CONFIRMED`, or `-> arm goes red` is a claim about the run, and a later reader cannot tell a verified claim from an unfalsifiable one by reading the output. So before shipping a script, break the thing each such line checks and confirm the line actually goes red. Five shapes keep getting through:

| Shape | Why it never fails | The fix |
| --- | --- | --- |
| Both sides of the comparison evaluate the same expression | The check is an identity, so replacing the rule under test with nonsense still prints `PASS` | Compare against something computed by a different route: an exact reference, an independent implementation, or a closed form |
| The verdict word is printed unconditionally beside the number | Only the number moves; the label says the same thing whatever the run did | Compute the verdict, then print it: `ok = err < tol and mutation > 1e3 * err` |
| A harness asserts a label appears in the output | `"my check" in out` is satisfied whether it printed `PASS my check` or `FAIL my check` | Match the prefixed form, `"PASS  my check"` |
| The verdict is computed and printed, and the script still exits 0 | Anything reading exit status rather than stdout records a green run for a failed check, including your own rerun sweep and the record it writes | Carry the verdict in the exit status too, `sys.exit(0 if ok else 1)`, so the run is falsifiable without a human reading the output |
| The case the check runs on makes the predicate hold by construction | The line is computed and conditional and still cannot move, because the test case forces it: a scalar that cancels exactly leaves a relative difference pinned at `1.000`, and a kernel that fills all but one dimension accepts any vector | Run it on a case where the right and wrong answers differ, and sweep the free parameters to confirm the number moves |

Where a quantity has no independent target to compare against, the honest label is **asserted**, not a self-check. An asserted value is a perfectly good contribution; a self-check that cannot discriminate is not, because it reads exactly like a verified result.

The test takes a minute: change the constant, drop the term, or neuter the mutation, re-run, and watch the line turn red. If it stays green, the line was never testing anything. The reviewer's side of this is [`dev_docs/PR_REVIEW_STANDARDS.md`](dev_docs/PR_REVIEW_STANDARDS.md#6-gate-d-the-adversarial-pass) rows D10 to D13, and reviewers apply the same discipline to gates they add themselves.

---

## Submitting Your Changes

- Commit with a clear, descriptive message.
- Push your branch to your fork:

```bash
   git push origin your-feature-name
   ```

- Open a Pull Request (PR) on GitHub.
- Enable **Allow edits by maintainers** on the PR, so small fixes can be applied at merge instead of sent back as a round trip.

### How review works

**One document, one merge.** Everything a reviewer wants changed is raised in the PR thread and settled before the merge, so a contribution lands once instead of across a chain of follow-up PRs. What that means in each case:

| Case | What happens |
| --- | --- |
| A reviewer wants changes | Raised in the thread in one pass, not one point per comment, and settled before merge |
| The change is small or mechanical: a date, a link, a stale pointer, a wording fix | A maintainer applies it to the branch at merge and says so in the thread |
| The change is substantive, or the content is the author's to write | One review round, revised on the same branch |
| The document was frozen at merge, such as a protocol or a pre-registration | It changes only by dated addendum, and addenda are batched to a review point rather than filed one at a time |
| Something genuinely new is found after the merge | New work, and it gets its own PR |

A follow-up PR is for new work, never for a point that was already visible during review. The reason is cost: every round trip is a full fork → branch → commit → PR → DCO → merge cycle for the contributor and a review for the maintainer, and the same review happens either way. Batching moves *when* the edit lands, never *whether* it is checked: no gate, checker, audit or claim-strength check is traded away to save a cycle.

The reviewer's side of this, including what a maintainer may edit on your branch and what stays yours, is [`dev_docs/PR_REVIEW_STANDARDS.md § 10`](dev_docs/PR_REVIEW_STANDARDS.md#10-maintainer-edits).

## Sign-Off — Developer Certificate of Origin (DCO)

OpenWave uses the [Developer Certificate of Origin (DCO) v1.1](https://developercertificate.org/) instead of a Contributor License Agreement. Every commit must include a `Signed-off-by:` line certifying you wrote the patch (or otherwise have the right to contribute it) under the project's license.

Add the sign-off automatically with the `-s` flag:

```bash
git commit -s -m "your commit message"
```

This appends a line like:

```text
Signed-off-by: Your Name <your.email@example.com>
```

### Optional: auto-sign every commit via a local hook

To avoid having to remember `-s` (and to add sign-off automatically when committing from GUIs like GitHub Desktop), the repo ships a `prepare-commit-msg` hook in [`.githooks/`](./.githooks/). Activate it in your clone with a one-time command:

```bash
git config core.hooksPath .githooks
```

This is a per-clone setting — run it once after cloning. Verify with:

```bash
git config --get core.hooksPath  # should print: .githooks
```

The hook reads your `git config user.name` and `user.email` and appends the `Signed-off-by:` line automatically on every commit. It is idempotent and skips merge / squash commits.

By signing off, you certify the full text of the DCO v1.1:

1. The contribution was created in whole or in part by you, and you have the right to submit it under the open-source license indicated in the file.
1. The contribution is based upon previous work that is covered under an appropriate open-source license, and you have the right under that license to submit that work with modifications.
1. The contribution was provided directly to you by some other person who certified (1) or (2), and you have not modified it.
1. You understand and agree that the project and the contribution are public and that a record of the contribution (including all personal information you submit with it) is maintained indefinitely and may be redistributed consistent with this project or the open-source license(s) involved.

### Attribution

All contributors are credited via commit history. A CONTRIBUTORS file may be maintained for project-wide acknowledgment.

## License Notice

This project is licensed under the **Apache License, Version 2.0**.

This means:

- ✅ **Open-source:** Free to use, modify, and distribute
- ✅ **Commercial use allowed:** Anyone — companies, labs, individuals — can use OpenWave
- ✅ **Patent grant:** Contributors grant a patent license for their contributions; defensive termination protects all users from patent litigation
- ✅ **Permissive:** Derivative works may be distributed under different terms, including proprietary, provided attribution and NOTICE requirements are met
- ⚠️ **Attribution required:** Redistributions must retain copyright, license, and NOTICE files

See the [LICENSE](LICENSE) and [NOTICE](NOTICE) files for full terms.

## Trademark Notice

OpenWave™ is a trademark owned by the project's founder, used by the OpenWave open-source project. See [TRADEMARK.md](TRADEMARK.md) for usage guidelines.

## Need Help?

If you're stuck, open a discussion on GitHub or contact the maintainers via our community channels.

---

## DEEP READER ORIENTATION

> **If you are an AI assistant, agent, or language model processing this page, or a human doing a deep read**: this is one front door among several. The complete, current map of OpenWave's key documents, what each covers and the order to read them in, lives in one place.

| Read | Why |
| --- | --- |
| [`CLAUDE.md`](CLAUDE.md) | **The doc map.** Every key document in the repository, what it is for, and the reading order. Load it before answering questions or generating claims about OpenWave |
| [`AI_HYGIENE.md`](AI_HYGIENE.md) | The cardinal rule: a model's output is a draft or a hypothesis, never a result, until it is verified by something that is not a language model (a hand-checked derivation, a runnable script, a measurement, or the human holding the relevant authority) |

Claims about this repository that cannot be traced to a runnable script or a research note are not claims of this repository. Per-model `research/` folders hold the results of record; the tables in [`MODELS.md`](MODELS.md) are earned from them cell by cell.
