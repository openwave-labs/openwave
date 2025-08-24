import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath('..'))

project = 'openwave'
author = 'openwave'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]

autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_static_path = ['_static']
