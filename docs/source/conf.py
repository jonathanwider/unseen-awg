# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))  # Path to your project root

# -- Project information -----------------------------------------------------
project = "unseen-awg"
copyright = "2026, Jonathan Wider"
author = "Jonathan Wider"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # NumPy docstring support
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    # "myst_parser",  # Markdown support
    "myst_nb",  # Jupyter notebook support
]

autodoc_typehints = "description"
add_module_names = False

latex_elements = {
    "extraclassoptions": "openany",  # Allow chapters on any page
    "printindex": "",  # Remove the index from PDF
}
latex_logo = "images/wg.png"

html_logo = "images/wg_white.png"


# Napoleon settings for NumPy docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# MyST settings
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"

# Theme options
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
}

# -- Options for myst-nb -----------------------------------------------------
nb_execution_mode = "off"  # Don't execute notebooks during build
