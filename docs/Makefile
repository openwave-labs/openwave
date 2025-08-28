# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    = -W --keep-going
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile deps clean-all watch

# Generate dependency graphs
deps:
	@echo "Generating dependency graphs..."
	@python generate_deps.py

# Clean everything including generated API docs
clean-all:
	@echo "Cleaning all generated files..."
	@rm -rf $(BUILDDIR)
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
