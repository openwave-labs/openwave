# OpenWave Documentation

This directory contains the source files for OpenWave's documentation, built using [Sphinx](https://www.sphinx-doc.org/).

## Quick Start

### Build Documentation (Recommended Method)

```bash
cd docs
make docs
```

This command will:

1. Clean old API documentation files (`.rst`)
2. Regenerate API docs from current source code
3. Build HTML documentation
4. Output location: `_build/html/index.html`

### View Documentation

```bash
# Option 1: Open directly in browser
open _build/html/index.html

# Option 2: Start local server
python -m http.server 8000 --directory _build/html
# Then visit: http://localhost:8000
```

## What are RST Files?

**RST (reStructuredText)** files are the source files for documentation:

- **Source format**: Like `.py` for code, `.rst` files are for documentation
- **Auto-generated**: RST files in `api/` are automatically generated from Python docstrings
- **Compiled to HTML**: Sphinx reads `.rst` → generates `.html`

**Important**: Don't manually edit `api/*.rst` files - they get regenerated from source code!

## Available Make Commands

```bash
# Recommended: Full rebuild with API regeneration
make docs

# Alternative commands
make apidoc       # Regenerate API RST files only
make html         # Build HTML (without regenerating API docs)
make clean        # Clean build directory
make clean-all    # Clean everything (build + generated files)
make watch        # Auto-rebuild on file changes
```

## Documentation Structure

```text
docs/
├── api/                    # Auto-generated API docs (RST files)
├── _build/                 # Generated HTML output
├── _static/                # Custom CSS, images
├── _templates/             # Custom templates
├── index.rst               # Main documentation page
├── getting_started.rst     # Installation & quick start
├── architecture.rst        # System architecture
├── physics_guide.rst       # Physics concepts
├── contributing.rst        # Contribution guidelines
├── conf.py                 # Sphinx configuration
├── Makefile               # Build commands
└── build_docs.sh          # Alternative build script

## Why Rebuild API Docs?

API documentation needs regeneration when:

- ✅ Modules are renamed
- ✅ New modules/classes are added
- ✅ Docstrings are updated
- ✅ Package structure changes

**Solution**: Always use `make docs` - it handles this automatically!

## Troubleshooting

### "Module not found" errors

```bash
# Install dependencies
pip install -r requirements.txt
```

### Outdated API docs

```bash
# Force regeneration
make clean-all
make docs
```

### Import errors during build

Some modules may fail to import during documentation build (e.g., due to Taichi initialization). This is normal - the build will continue and generate docs for importable modules.

## Configuration

- **Sphinx config**: `conf.py`
- **Mock imports**: Configured in `conf.py` to avoid import errors
- **Theme**: Read the Docs theme
- **Extensions**: autodoc, napoleon, intersphinx, etc.

## Contributing to Documentation

1. Update docstrings in source code (`.py` files)
2. Update content RST files (`getting_started.rst`, etc.)
3. Run `make docs` to rebuild
4. Check output in `_build/html/`
5. Commit changes (don't commit `_build/` or `api/*.rst`)
