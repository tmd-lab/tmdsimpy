# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TMDSimPy'
copyright = '2024, Justin H. Porter'
author = 'Justin H. Porter'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',     # Pulls docstrings from Python code
              'sphinx.ext.autosummary', # Allows generation for all functions
              'numpydoc'                # docstring format used
              ]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autosummary_imported_members = False # Default
autosummary_ignore_module_all = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'bizstyle' # 'alabaster'
html_static_path = ['_static']

# -- Path to Module -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/quickstart.html#autodoc

# import sys
# sys.path.append('..')

# Alternative documentation: 
# https://www.sphinx-doc.org/en/master/tutorial/describing-code.html#including-doctests-in-your-documentation
import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

# numpydoc_class_members_toctree = False
