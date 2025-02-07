# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SIDISH'
copyright = '2025, Yasmin Jolasun'
author = 'Yasmin Jolasun'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Automatically extract documentation from docstrings
    'sphinx.ext.napoleon',  # Support for Google-style docstrings
    'sphinx.ext.viewcode',  # Include links to source code
    'sphinx.ext.todo',  # Support for TODOs in docs
    'sphinx.ext.mathjax',  # Support for mathematical equations
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Autodoc settings --------------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'special-members': False,
    'show-inheritance': True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Ensure Read the Docs can build the project ------------------------------
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if on_rtd:
    html_static_path = []