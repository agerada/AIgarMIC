# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'src')))

project = 'AIgarMIC'
copyright = '2024, Alessandro Gerada'
author = 'Alessandro Gerada'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "sphinxarg.ext",
    "sphinx.ext.doctest"
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autoapi_dirs = ['../../src']


def skip_submodules(app, what, name, obj, skip, options):
    skip_these = {
        "aigarmic._img_utils": True,
        "aigarmic._nn_design": True,
        "aigarmic.clean_up_annotations": True,
        "aigarmic.main": True,
        "aigarmic.manual_annotator": True,
        "aigarmic.model_performance": True,
        "aigarmic.rename_images": True,
        "aigarmic.train_modular": True
    }
    return skip_these.get(name, skip)


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_submodules)
