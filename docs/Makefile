# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile deps clean-all watch apidoc docs

# Regenerate API documentation from source code
apidoc:
	@echo "Regenerating API documentation from source..."
	@rm -rf api/*.rst
	@python -m sphinx.ext.apidoc -f -e -M -o api ../openwave
	@echo "Creating openwave._io.rst..."
	@printf "openwave.\\\\_io\n=============\n\n.. automodule:: openwave._io\n\n\n.. rubric:: Modules\n\n.. autosummary::\n   :toctree:\n   :recursive:\n\n   cli\n   render\n" > api/openwave._io.rst
	@echo "Fixing openwave.rst to include _io module..."
	@grep -q "_io" api/openwave.rst || perl -i.bak -pe 's/^(   common)$$/   _io\n$$1/' api/openwave.rst && rm -f api/openwave.rst.bak
	@echo "API documentation regenerated"

# Generate dependency graphs
deps:
	@echo "Generating dependency graphs..."
	@python generate_deps.py

# Complete documentation build (recommended)
docs: apidoc
	@echo "Re-adding _io to openwave.rst (in case it was removed)..."
	@grep -q "^   _io$$" api/openwave.rst || perl -i.bak -pe 's/^(   common)$$/   _io\n$$1/' api/openwave.rst && rm -f api/openwave.rst.bak
	@echo "Building complete documentation..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Documentation built at: $(BUILDDIR)/html/index.html"

# Clean everything including generated API docs
clean-all:
	@echo "Cleaning all generated files..."
	@rm -rf $(BUILDDIR)
	@rm -rf api/*.rst
	@rm -rf api/generated
	@rm -rf _static/deps
	@echo "Clean complete"

# Build HTML with dependency graphs
html-full: deps
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Full documentation built at: $(BUILDDIR)/html/index.html"

# Watch for changes and rebuild
watch:
	@echo "Watching for changes (requires sphinx-autobuild)..."
	@command -v sphinx-autobuild >/dev/null 2>&1 || pip install sphinx-autobuild
	@sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" \
		--ignore "*.pyc" \
		--ignore "*/__pycache__/*" \
		--ignore "*/_build/*" \
		--port 8000

# Quick build without dependency regeneration
quick:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Documentation built at: $(BUILDDIR)/html/index.html"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
