# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from functools import wraps


def Create_requires_decorator(*modules: str):
    """Creates a decorator that checks if modules are available before executing a function.

    Returns
    -------
    function
        A decorator that raises an ImportError if the modules are not available.
    """

    try:
        [__import__(module) for module in modules]
        can_use_module = True
    except ImportError:
        can_use_module = False

    def __get_error(func) -> str:
        if len(modules) > 1:
            modules_str = ", ".join(modules[:-1])
            modules_str += f" and {modules[-1]}"
            error = f"{modules_str} are required for {func.__name__} function.\nPlease install them with: pip install {', '.join(modules)} command."
        else:
            module = modules[0]
            error = f"{module} is required for {func.__name__} function.\nPlease install it with: pip install {module} command."
        return error

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not can_use_module:
                raise ImportError(__get_error(func))
            return func(*args, **kwargs)

        return wrapper

    return decorator
