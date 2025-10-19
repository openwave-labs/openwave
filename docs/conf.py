"""
Configuration file for Sphinx documentation builder.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(".."))

# Read version from pyproject.toml
def get_version():
    """Extract version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "r") as f:
        for line in f:
            if line.startswith("version"):
                # Extract version string from: version = "0.1.2"
                return line.split("=")[1].strip().strip('"').split("#")[0].strip()
    return "0.1.0"  # fallback

# Project information
project = "OpenWave"
copyright = f"{datetime.now().year}, OpenWave Team"
author = "OpenWave Team"
release = get_version()
version = release  # Short version (for display)

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
]

# Add support for Markdown files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Mock imports for libraries that may not be available during doc build
autodoc_mock_imports = [
    "taichi",
    "matplotlib",
    "numpy",
    "scipy",
]

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
    "inherited-members": True,
    "private-members": False,
}

autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# HTML output configuration
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
    "prev_next_buttons_location": "both",
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Intersphinx mapping for cross-references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Graphviz configuration for dependency diagrams
graphviz_output_format = "svg"
inheritance_graph_attrs = dict(
    rankdir="TB",
    size='""',
    fontsize=14,
    ratio="compress",
)

# Enable TODO directives
todo_include_todos = True

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\.\. |\$ "
copybutton_prompt_is_regexp = True
