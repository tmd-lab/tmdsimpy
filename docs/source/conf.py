# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TMDSimPy'
copyright = '2024, MIT License'
author = 'Justin H. Porter'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',     # Pulls docstrings from Python code
              'sphinx.ext.autosummary', # Allows generation for all functions
              'numpydoc'                # docstring format used
              ]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autosummary_imported_members = False # Default
autosummary_ignore_module_all = False

# autosummary / methods / stubfiles issues
# https://exchangetuts.com/sphinx-warning-autosummary-stub-file-not-found-for-the-methods-of-the-class-check-your-autosummary-generate-settings-1640709483669579
numpydoc_show_class_members = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'bizstyle' # 'alabaster'
html_static_path = ['_static']

import sys
import os
import pathlib

pathlib.Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          '_static/')).mkdir(exist_ok=True)

# -- Path to Module -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/quickstart.html#autodoc

# import sys
# sys.path.append('..')

# Alternative documentation: 
# https://www.sphinx-doc.org/en/master/tutorial/describing-code.html#including-doctests-in-your-documentation
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

# numpydoc_class_members_toctree = False
