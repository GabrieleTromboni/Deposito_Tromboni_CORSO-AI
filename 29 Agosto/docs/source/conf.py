# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Prove_Sphinx'
copyright = '2025, Gabriele'
author = 'Gabriele'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# Add path to find project root (where prove.py lives)
import os, sys
sys.path.insert(0, os.path.abspath("../../"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # per Google/NumPy style
    "sphinx_autodoc_typehints",
]
napoleon_google_docstring = True
napoleon_numpy_docstring = False