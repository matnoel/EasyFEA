# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from functools import wraps
from importlib.util import find_spec
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def _Is_available(module: str) -> bool:
    """Whether `module` can be imported, without importing it."""
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def Create_requires_decorator(
    *modules: str, libraries: list[str] | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Creates a decorator that checks if modules are available before executing a function.

    Returns
    -------
    function
        A decorator that raises an ImportError if the modules are not available.
    """

    can_use_modules = all(_Is_available(module) for module in modules)

    install = " ".join(modules if libraries is None else libraries)

    def __get_error(func) -> str:
        if len(modules) > 1:
            modules_str = ", ".join(modules[:-1])
            modules_str += f" and {modules[-1]}"
            error = f"{modules_str} are required for {func.__name__} function."
            error += f"\nPlease install them with: pip install {install} command."
        else:
            module = modules[0]
            error = f"{module} is required for {func.__name__} function."
            error += f"\nPlease install it with: pip install {install} command."
        return error

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not can_use_modules:
                raise ImportError(__get_error(func))
            return func(*args, **kwargs)

        return wrapper

    return decorator
