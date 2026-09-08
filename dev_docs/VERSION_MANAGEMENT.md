# Version Management

## Overview

OpenWave uses a **single source of truth** approach for version management that works correctly with both regular installations and editable (development) installations.

## How It Works

### Version Definition

The version is defined in two places:

1. **Source code**: `openwave/__init__.py` contains `__version__ = "X.Y.Z"`
2. **Build metadata**: `pyproject.toml` contains `version = "X.Y.Z"`

Both files should always have the **same version number**.

### Version Access

The codebase accesses the version from the source code directly:

```python
from openwave import __version__
```

This approach ensures:

- Developers with editable installs (`pip install -e .`) see the current version immediately
- No need to reinstall after version bumps
- Works correctly with both development and production installations
- Fallback to metadata if `__version__` is not available

## For Developers

### Updating the Version

When bumping the version, update **both** files:

1. Edit `openwave/__init__.py`:

   ```python
   __version__ = "26.9.7"  # Update this to today's date
   ```

1. Edit `pyproject.toml`:

   ```toml
   version = "26.9.7"  # Update this to match
   ```

### Why This Works with Editable Installs

- **Editable install** (`pip install -e .`): Python imports directly from your source directory, so changes to `__init__.py` are immediately visible
- **Regular install** (`pip install .`): The version in `__init__.py` is copied during installation, and the metadata is generated from `pyproject.toml`

Both installation methods will see the same version number.

## Version Numbering

OpenWave uses **calendar versioning (CalVer)** in the form `YY.M.D`: the date the release was
cut. `26.9.7` is 7 September 2026. Adopted 2026-09-07, replacing SemVer at `1.6.10`.

### Why not SemVer

A SemVer number is a backward-compatibility promise addressed to a dependency resolver. This
project has no resolver to make one to: it is not published to PyPI, nothing pins a version
range against it, and with no CI there is nothing that could check such a promise even in
principle. `MINOR` versus `PATCH` was a judgment call made on every release for no reader.

A date is a fact the project can keep. It also carries information the old scheme did not: a
reader of a dated finding, method note or run record can tell at a glance whether the engine
predates or postdates it, which matters in a repository where results cite the version they
were produced with.

The tradeoff, stated plainly: CalVer says nothing about compatibility, and it makes a release
gap visible on the front page. Both are accepted. The compatibility signal was never verified,
and "last released September 2026" is useful to anyone deciding whether to build on the engine.

### Rules

| Rule | Reason |
| --- | --- |
| No zero padding: `26.9.7`, never `26.09.07` | PEP 440 strips leading zeros, so the padded form installs as `26.9.7` and stops matching its own tag |
| No `v` in `__init__.py` or `pyproject.toml` | PEP 440 strips it too. The `v` prefix belongs on the git tag alone, which is where the existing convention already puts it |
| A second release on a day already used adds a fourth component: `26.9.7.1` | ⚠️ NOT a letter suffix. PEP 440 reads `26.9.7b` as a **beta** of `26.9.7`, which sorts BEFORE it and which `pip install` skips by default, so an emergency fix numbered that way would be both invisible and considered older than the release it fixes |
| One release per day is the norm | Four days in the pre-CalVer history carried two releases (2026-01-20, 2026-07-02, 2026-07-20, 2026-07-29). The fourth component exists for that case and is expected to stay rare |

### Ordering

`YY.M.D` sorts correctly under PEP 440 because each component is compared as an integer, not as
text: `26.9.7 < 26.10.1 < 27.1.3`. The switch also moved forward, never back, since
`1.6.10 < 26.9.7`. That makes it a one-way change: returning to SemVer would require a version
decrease, which no installer would select.

## Implementation Details

### Files Using Version

The following files display the version to users:

- `openwave/i_o/cli.py`: CLI menu headers (lines 159-164, 223-229)
- `openwave/i_o/render.py`: Window title (lines 20-29)

All files use the same pattern:

```python
try:
    from openwave import __version__
    pkg_version = __version__
except ImportError:
    # Fallback to metadata if __version__ not available
    from importlib.metadata import version
    pkg_version = version("OPENWAVE")
```

### Why Not Use `importlib.metadata.version()` Only?

The `importlib.metadata.version()` function reads from package metadata installed by pip. This metadata is only updated when you reinstall the package:

- With editable installs, metadata is created once during `pip install -e .`
- Subsequent code changes (including version bumps) don't update the metadata
- Developers would need to run `pip install -e .` after every version bump

By reading from `__version__` in the source code, we avoid this issue entirely.

## Alternative Approaches

### setuptools-scm (Not Used)

An alternative approach is to use `setuptools-scm` to derive versions from git tags. We chose not to use this because:

- Adds complexity and dependencies
- Requires proper git tagging discipline
- Can be confusing when working with uncommitted changes
- Simple dual-file approach is more transparent and explicit

## When to Bump Version

### Recommended Workflow: Bump BEFORE Creating Tag/Release

The best practice is to bump the version **before** creating git tags and GitHub releases:

1. **Update the version** in your source code (`__init__.py` and `pyproject.toml`)
1. **Commit the version bump** with a clear message
1. **Create a git tag** matching that version
1. **Create a GitHub release** from that tag
1. **Publish to PyPI** (if applicable) using that tagged version

#### Why This Order?

This ensures:

- The tag points to code that actually contains that version number
- Users installing from that tag get the correct version
- Clear history: `git log` shows when each version was created
- The git tag and the package version are synchronized

#### Typical Workflow Example

```bash
# 1. Make your changes and test them
git add .
git commit -m "Add new feature X"

# 2. Bump the version to today's date, YY.M.D with no zero padding
date +%y.%-m.%-d          # prints the version to use, e.g. 26.9.7
# Edit: openwave/__init__.py → __version__ = "26.9.7"
# Edit: pyproject.toml → version = "26.9.7"

# 3. Commit the version bump
git add openwave/__init__.py pyproject.toml
git commit -m "Bump version to 26.9.7"

# 4. Create a git tag
git tag -a v26.9.7 -m "Release version 26.9.7"

# 5. Push everything
git push origin main
git push origin v26.9.7

# 6. Create GitHub release (via UI or gh cli)
gh release create v26.9.7 --title "v26.9.7" --notes "Release notes here"

# 7. (Optional) Publish to PyPI
python -m build
python -m twine upload dist/*
```

#### Alternative: Separate Version Bump Commit

Some teams prefer a dedicated "version bump" commit at the end of a release cycle:

```bash
# After all feature work is done:
git commit -m "Implement feature X"
git commit -m "Fix bug Y"
git commit -m "Update docs"

# Then bump version as last commit before tag
# Edit version files to today's date...
git commit -m "Bump version to 26.9.7"
git tag -a v26.9.7 -m "Release v26.9.7"
```

### What NOT to Do

- **Don't bump version AFTER creating the tag**: The tag would point to old version number
- **Don't commit version bumps on every commit**: Creates noise in git history
- **Don't leave version bumps uncommitted**: Other developers won't see the new version

### Which Number to Use

Read a calendar. There is no judgment call to make and no need to look up the previous
version:

```bash
date +%y.%-m.%-d          # 26.9.7
```

The only decision left is the rare one: if that exact version was already released today, add
a fourth component (`26.9.7.1`, then `26.9.7.2`).

### Pre-release Versions

⚠️ SemVer-style suffixes such as `0.2.0-dev` or `0.2.0-alpha.1` are NOT canonical PEP 440 and
silently mutate on install. Python rewrites them:

| Written | What pip records |
| --- | --- |
| `26.9.7-dev` | `26.9.7.dev0` |
| `26.9.7-alpha.1` | `26.9.7a1` |
| `26.9.7-beta.1` | `26.9.7b1` |
| `26.9.7-rc.1` | `26.9.7rc1` |

Write the canonical form directly if a pre-release is ever needed:

```python
__version__ = "26.9.7.dev0"   # Development (ongoing work)
__version__ = "26.9.7a1"      # Alpha (early testing)
__version__ = "26.9.7b1"      # Beta (feature complete)
__version__ = "26.9.7rc1"     # Release candidate (final testing)
__version__ = "26.9.7"        # Final release
```

⚠️ All four sort BEFORE `26.9.7`, and `pip install` skips them unless asked for with
`--pre`. That is what they are for. It is also why a same-day hotfix must never be numbered
`26.9.7b`: it would be treated as a beta of a release that already shipped.

In practice a date-based scheme has little use for these. The working version between releases
is simply the last released version until the day a new one is cut.

### Automation Options

Version bumping needs no tool now that the number is a date, and `bumpver update --patch`
style commands no longer map onto anything. If any automation is added later, the useful
target is checking that the two files agree and that the tag matches, not computing the number:

```bash
# Verify the two version strings are in sync before tagging
python -c "import tomllib,re,sys; \
t=tomllib.load(open('pyproject.toml','rb'))['project']['version']; \
i=re.search(r'__version__ = \"([^\"]+)\"',open('openwave/__init__.py').read()).group(1); \
sys.exit(0 if t==i else f'version mismatch: pyproject {t} vs __init__ {i}')"
```

## Best Practices Summary

1. Always update both `__init__.py` and `pyproject.toml` together
1. Bump version BEFORE creating git tags and releases
1. Use the pattern shown above when accessing version in code
1. Keep version numbers synchronized between source and build config
1. Use today's date, `YY.M.D`, with no zero padding and no `v` prefix in the files
1. Create dedicated version bump commits
1. Document version changes in commit messages and release notes
1. Reserve a fourth component (`26.9.7.1`) for a second release on the same day
