# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from functools import wraps


def Create_requires_decorator(module: str):
    """Creates a decorator that checks if a module is available before executing a function.

    Parameters
    ----------
    module : str
        The name of the module to check (e.g., “meshio”, "pyvista").

    Returns
    -------
    function
        A decorator that raises an ImportError if the module is not available.
    """

    try:
        __import__(module)
        can_use_module = True
    except ImportError:
        can_use_module = False

    install = f"Please install it with: pip install {module}"

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not can_use_module:
                raise ImportError(
                    f"{module} is required for {func.__name__}.\n{install}"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
