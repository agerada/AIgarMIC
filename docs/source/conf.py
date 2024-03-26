# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# import os
# import sys

#sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'src', 'aigarmic')))

project = 'AIgarMIC'
copyright = '2024, Alessandro Gerada'
author = 'Alessandro Gerada'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "autoapi.extension"
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autoapi_dirs = ['../../src']


def skip_submodules(app, what, name, obj, skip, options):
    if name == "aigarmic.main":
        skip = True
    if name == "aigarmic._img_utils":
        skip = True
    if name == "aigarmic._nn_design":
        skip = True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_submodules)