# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SeqControl'
copyright = '2026, Ilyes Ait Aissa'
author = 'Ilyes Ait Aissa'
release = '1.0.0'

import sys
import os

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []

# 1. Add your Python project source path (relative to conf.py)
sys.path.insert(0, os.path.abspath('../../')) 

# 2. Enable autodoc and napoleon (for Google/NumPy style docstrings)
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode', # Adds links to highlighted source code
]
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


