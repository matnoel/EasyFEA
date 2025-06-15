# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import re
import sys

import pyvista
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "EasyFEA"
copyright = "2025, Matthieu Noel"
author = "Matthieu Noel"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",  # Measure durations of Sphinx processing
    # DOC
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    # plot
    "jupyter_sphinx",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
    "pyvista.ext.plot_directive",
    "pyvista.ext.viewer_directive",
]
autodoc_typehints = "signature"


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Gallery configuration ---------------------------------------------------

# Manage errors
pyvista.set_error_output_file("errors.txt")
# Ensure that offscreen rendering is used for docs generation
pyvista.OFF_SCREEN = True  # disables showing figures in a GUI window
# Preferred plotting style for documentation
pyvista.set_plot_theme("document")

# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True
os.environ["PYVISTA_BUILDING_GALLERY"] = "true"


def natural_sort_key(s):
    """Split string into integers and letters for natural sorting."""
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def FileNameSortKey(filename):
    base = os.path.basename(filename)
    return natural_sort_key(base)


sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "examples",
    "image_scrapers": (DynamicScraper(), "matplotlib"),
    "download_all_examples": False,
    "remove_config_comments": True,
    "filename_pattern": r".*\.py",
    "ignore_pattern": r"__init__\.py",
    "within_subsection_order": FileNameSortKey,
    "line_numbers": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"  # "furo", "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/matnoel/EasyFEA",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/easyfea/",
            "icon": "fa-solid fa-box",
            "type": "fontawesome",
        },
        {
            "name": "Issues",
            "url": "https://github.com/matnoel/EasyFEA/issues",
            "icon": "fa-solid fa-issues",
            "type": "fontawesome",
        },
        {
            "name": "Read the Docs",
            "url": "https://app.readthedocs.org/projects/easyfea/",
            "icon": "fa-solid fa-book",
            "type": "fontawesome",
        },
    ],
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "matnoel",
    "github_repo": "EasyFEA",
    "github_version": "main",
    "doc_path": "docs/",
}

html_title = "EasyFEA"

html_static_path = ["_static"]
