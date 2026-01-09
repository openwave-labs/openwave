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
   __version__ = "0.2.0"  # Update this
   ```

1. Edit `pyproject.toml`:

   ```toml
   version = "0.2.0"  # Update this to match
   ```

### Why This Works with Editable Installs

- **Editable install** (`pip install -e .`): Python imports directly from your source directory, so changes to `__init__.py` are immediately visible
- **Regular install** (`pip install .`): The version in `__init__.py` is copied during installation, and the metadata is generated from `pyproject.toml`

Both installation methods will see the same version number.

## Version Numbering

OpenWave follows [Semantic Versioning (SemVer)](https://semver.org/):

- **MAJOR**: Breaking changes (incompatible API changes)
- **MINOR**: New features (backward-compatible)
- **PATCH**: Bug fixes (backward-compatible)
- **0.x**: Pre-release versions (API still evolving)

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

# 2. Bump the version
# Edit: openwave/__init__.py → __version__ = "0.2.0"
# Edit: pyproject.toml → version = "0.2.0"

# 3. Commit the version bump
git add openwave/__init__.py pyproject.toml
git commit -m "Bump version to 0.2.0"

# 4. Create a git tag
git tag -a v0.2.0 -m "Release version 0.2.0"

# 5. Push everything
git push origin main
git push origin v0.2.0

# 6. Create GitHub release (via UI or gh cli)
gh release create v0.2.0 --title "v0.2.0" --notes "Release notes here"

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
# Edit version files...
git commit -m "Bump version to 0.2.0"
git tag -a v0.2.0 -m "Release v0.2.0"
```

### What NOT to Do

- **Don't bump version AFTER creating the tag**: The tag would point to old version number
- **Don't commit version bumps on every commit**: Creates noise in git history
- **Don't leave version bumps uncommitted**: Other developers won't see the new version

### When to Bump Which Number?

Following Semantic Versioning (SemVer):

#### MAJOR version (X.0.0) - Breaking Changes

Increment when making incompatible API changes:

- Remove or rename public functions
- Change function signatures
- Remove deprecated features
- Restructure modules/packages

#### MINOR version (0.X.0) - New Features

Increment when adding functionality in a backward-compatible manner:

- Add new functions/classes
- Add new optional parameters
- New capabilities that don't break existing code
- New experimental features

#### PATCH version (0.0.X) - Bug Fixes

Increment when making backward-compatible bug fixes:

- Fix broken functionality
- Performance improvements
- Documentation updates
- Internal refactoring

### Pre-release Versions

For development versions between releases, use pre-release suffixes:

```python
__version__ = "0.2.0-dev"       # Development (ongoing work)
__version__ = "0.2.0-alpha.1"   # Alpha release (early testing)
__version__ = "0.2.0-beta.1"    # Beta release (feature complete)
__version__ = "0.2.0-rc.1"      # Release candidate (final testing)
__version__ = "0.2.0"           # Final release
```

### Automation Options

Consider automating version bumping with tools:

- **bump2version** / **bumpver**: CLI tools for version management
- **semantic-release**: Automatically determines version from commit messages
- **GitHub Actions**: Automate bumping on merge to main

Example using `bumpver`:

```bash
# Install
pip install bumpver

# Bump patch version (0.1.1 → 0.1.2)
bumpver update --patch

# Bump minor version (0.1.2 → 0.2.0)
bumpver update --minor

# Bump major version (0.2.0 → 1.0.0)
bumpver update --major
```

## Best Practices Summary

1. Always update both `__init__.py` and `pyproject.toml` together
1. Bump version BEFORE creating git tags and releases
1. Use the pattern shown above when accessing version in code
1. Keep version numbers synchronized between source and build config
1. Follow semantic versioning guidelines strictly
1. Create dedicated version bump commits
1. Document version changes in commit messages and release notes
1. Use pre-release suffixes for development versions
