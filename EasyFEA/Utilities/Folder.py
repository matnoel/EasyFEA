# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing functions used to facilitate folder and file creation using (os)."""

import os
import inspect
from pathlib import Path


def Dir(path: str, depth: int = 1) -> str:
    """Returns the directory of the specified path."""

    assert isinstance(path, str), "filename must be str"
    assert isinstance(depth, int) and depth > 0, "depth must be a positive integer"

    normPath = os.path.normpath(path)

    dir = os.path.dirname(normPath)
    for _ in range(depth - 1):
        dir = os.path.dirname(dir)

    return dir


EASYFEA_DIR = Dir(__file__, 3)


def Join(*args: str, mkdir=False) -> str:
    """Joins two or more pathname components and create (or not) the path."""

    path = os.path.join(*args)

    if not Exists(path) and mkdir:
        if "." in path:
            dir = Dir(path)
            os.makedirs(dir, exist_ok=True)
        else:
            os.makedirs(path)

    return path


def Results_Dir() -> str:
    """Provides the directory path where results should be stored, relative to the calling Python script: `<script_directory>/results/<script_name>`.

    WARNING
    -------
    This function does not work in a Jupyter notebook!
    """
    from .. import BUILDING_GALLERY

    stack = inspect.stack()
    if BUILDING_GALLERY:
        # In Sphinx Gallery, Python scripts are parsed with `py_source_parser.py`
        # See: https://github.com/sphinx-gallery/sphinx-gallery/blob/master/sphinx_gallery/py_source_parser.py
        # The parsed code is executed by the `execute_code_block` function
        # See: https://github.com/sphinx-gallery/sphinx-gallery/blob/fc649a70dbbc23e0229adeaa5a60422ec592333b/sphinx_gallery/gen_rst.py#L977C5-L1097
        # This function is itself called in the `execute_script` function.
        # See: https://github.com/sphinx-gallery/sphinx-gallery/blob/fc649a70dbbc23e0229adeaa5a60422ec592333b/sphinx_gallery/gen_rst.py#L1127-L1222

        # Look for the `execute_code_block` function in the stack
        pythonScript = None
        for frame in stack:
            function = getattr(frame, "function")
            if function == "execute_code_block":
                # get local variables
                f_locals = frame[0].f_locals
                try:
                    pythonScript = f_locals["script_vars"]["src_file"]
                except KeyError:
                    raise Exception(
                        "sphinx_gallery may change over time; look out for `execute_script` or `execute_code_block` functions."
                    )
        # make sure that the file is detected
        assert (
            pythonScript is not None
        ), "`execute_code_block` function was not detected"

    else:
        pythonScript = stack[1].filename

    # get pythonScript as a path
    path = Path(pythonScript)
    assert path.exists(), f"{pythonScript} does not exist."
    assert path.is_file(), f"{pythonScript} must be a file."
    assert (
        Join(EASYFEA_DIR, "EasyFEA") not in pythonScript
    ), "You cannot write results in the src directory; therefore, you should not use this function in the src files."

    return Join(Dir(pythonScript), "results", path.stem)


def Exists(path: str) -> bool:
    """Test whether a path exists. Returns False for broken symbolic links"""
    return os.path.exists(path)
