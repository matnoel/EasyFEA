# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import string
import inspect
import sys

import pyvista
import EasyFEA
from EasyFEA.Geoms import _Init_Geoms_NInstance
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "EasyFEA"
copyright = "2025, Matthieu Noel"
author = "Matthieu Noel"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # markdown
    # DOC
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    # plots and gallery
    "jupyter_sphinx",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
    "pyvista.ext.plot_directive",
    "pyvista.ext.viewer_directive",
    "sphinx.ext.duration",  # Measure durations of Sphinx processing
]

suppress_warnings = [
    "toc.not_included",
    "toc.not_readable",
]

# -- Auto doc ---------------------------------------------------

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#automatically-document-modules
autodoc_default_options = {
    "members": True,
    "private-members": True,
    "imported-members": False,
    "undoc-members": True,
    "show-inheritance": False,
    "member-order": "groupwise",
}
maximum_signature_line_length = 88


def linkcode_resolve(
    domain: str, info: dict[str, str], edit: bool = False
) -> str | None:
    """Determine the URL corresponding to a Python object.

    Parameters
    ----------
    domain : str
        Only useful when 'py'.

    info : dict
        With keys "module" and "fullname".

    edit : bool, default=False
        Jump right to the edit page.

    Returns
    -------
    str
        The code URL. Empty string if there is no valid link.

    Notes
    -----
    This function is used by the `sphinx.ext.linkcode` extension to create the "[Source]"
    button whose link is edited in this function.

    Adapted from PyVista (pyvista/pyvista/core/utilities/docs.py).
    """

    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    # Little clean up to avoid pyvista.pyvista
    if fullname.startswith(modname):
        fullname = fullname[len(modname) + 1 :]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    # deal with our decorators properly
    while hasattr(obj, "fget"):
        obj = obj.fget

    # deal with wrapped object
    while hasattr(obj, "__wrapped__"):
        obj = obj.__wrapped__
    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None

    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            return None
        return None

    fn = os.path.relpath(fn, start=os.path.dirname(EasyFEA.__file__))
    fn = "/".join(os.path.normpath(fn).split(os.sep))

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    linespec = f"#L{lineno}-L{lineno + len(source) - 1}" if lineno and not edit else ""

    blob_or_edit = "edit" if edit else "blob"
    github = "https://github.com/matnoel/EasyFEA"
    link = f"{github}/{blob_or_edit}/main/EasyFEA/{fn}{linespec}"
    return link


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

EasyFEA.BUILDING_GALLERY = True


def natural_sort_key(s):
    """Split string into integers and letters for natural sorting."""
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def FileNameSortKey(filename):
    base = os.path.basename(filename)
    return natural_sort_key(base)


class ResetEasyFEA:
    """Reset EasyFEA module to default settings.\n
    Adapted from PyVista (doc/source/conf.py)."""

    def __call__(self, gallery_conf, fname):
        _Init_Geoms_NInstance()
        EasyFEA.Tic.Clear()

    def __repr__(self):
        return "ResetEasyFEA"


resetEasyFEA = ResetEasyFEA()

# https://sphinx-gallery.github.io/stable/configuration.html
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "examples",
    "abort_on_example_error": True,
    "image_scrapers": (DynamicScraper(), "matplotlib"),
    "download_all_examples": False,
    "remove_config_comments": True,
    "filename_pattern": r".*\.py",
    "ignore_pattern": r"__init__\.py",
    "within_subsection_order": FileNameSortKey,
    "reset_modules": (resetEasyFEA,),
    "line_numbers": True,
    "parallel": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# https://fontawesome.com/search

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
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
        {
            "name": "Discussions",
            "url": "https://github.com/matnoel/EasyFEA/discussions",
            "icon": "fa-solid fa-comment",
            "type": "fontawesome",
        },
        {
            "name": "Issues",
            "url": "https://github.com/matnoel/EasyFEA/issues",
            "icon": "fa-solid fa-issues",
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
html_css_files = ["custom.css"]

# -- Latex -------------------------------------------------


def get_function(function: str, arg: str):
    return chr(92) + function + "{" + arg + "}"


def get_dict_formats(
    function: str, format: str, letters: str = string.ascii_letters
) -> dict[str, str]:
    # {letter}{format}: \function{letter}
    return {f"{l}{format}": get_function(function, l) for l in letters}


mathjax3_config = {
    "loader": {"load": ["input/tex"]},
    "tex": {
        "macros": {
            **get_dict_formats("mathrm", "rm"),
            **get_dict_formats("mathcal", "c"),
            **get_dict_formats("mathbf", "bf"),
            **get_dict_formats("mathbb", "bb"),
            **get_dict_formats("boldsymbol", "b"),
            # operators
            "grad": r"\boldsymbol{\nabla}",
            "lap": r"\boldsymbol{\Delta}",
            "diver": r"\boldsymbol{\nabla} \cdot",
            "dpartial": [r"\dfrac{\partial #1}{\partial #2}", 2],
            "dNpartial": [r"\dfrac{\partial^#1 #2}{\partial #3 ^#1}", 3],
            # others
            "dt": r"\Delta t",
            # meca
            "Eps": r"\boldsymbol{\varepsilon}",
            "Sig": r"\boldsymbol{\sigma}",
            # differentitation
            "dO": r"\drm \Omega",
            "dS": r"\drm S",
        }
    },
}

# -- Setup -------------------------------------------------


def skip_member_handler(app, obj_type, name, obj, skip, options):
    # https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#event-autodoc-skip-member
    # app: the Sphinx application object
    # obj_type: the type of the object which the docstring belongs to (one of 'module', 'class', 'exception', 'function', 'decorator', 'method', 'property', 'attribute', 'data', or 'type')
    # name: the fully qualified name of the object
    # obj: the object itself
    # skip: a boolean indicating if autodoc will skip this member if the user handler does not override the decision
    # options: the options given to the directive: an object with attributes corresponding to the options used in the auto directive, e.g. inherited_members, undoc_members, or show_inheritance.

    # remove private methods
    if "__" in name and name.startswith("_"):
        return True

    # ignore numpy, scipy, ... imports
    module = getattr(obj, "__module__", "")
    if module and not module.startswith("EasyFEA"):
        return True

    return skip  # default behavior


def setup(app):
    app.connect("autodoc-skip-member", skip_member_handler)
