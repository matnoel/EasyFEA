# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Terminal/console output helpers: colored text, section headers and screen clearing."""

import os
import platform
from enum import Enum

from ._mpi import rank0_only, MPI_COMM


class __Colors(str, Enum):
    blue = "\033[34m"
    cyan = "\033[36m"
    white = "\033[37m"
    green = "\033[32m"
    black = "\033[30m"
    red = "\033[31m"
    yellow = "\033[33m"
    magenta = "\033[35m"


class __Sytles(str, Enum):
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    RESET = "\33[0m"


@rank0_only
def MyPrint(
    text: str,
    color="cyan",
    bold=False,
    italic=False,
    underLine=False,
    end: str = "",
) -> str:
    dct = dict(map(lambda item: (item.name, item.value), __Colors))

    if color not in dct:
        return MyPrint(f"Color must be in {dct.keys()}", "red")

    else:
        formatedText = ""

        if bold:
            formatedText += __Sytles.BOLD
        if italic:
            formatedText += __Sytles.ITALIC
        if underLine:
            formatedText += __Sytles.UNDERLINE

        formatedText += dct[color] + str(text)

        formatedText += __Sytles.RESET

        if end == "\r" and MPI_COMM is not None:
            end = "\n"

        print(formatedText, end=end)
        return formatedText


def MyPrintError(text: str) -> str:
    return MyPrint(text, "red")


def Section(text: str, verbosity=True) -> str:
    """Creates a new section in the terminal."""
    edges = "======================="

    lengthText = len(text)

    lengthTot = 45

    edges = "=" * int((lengthTot - lengthText) / 2)

    section = f"\n\n{edges} {text} {edges}\n"

    if verbosity:
        MyPrint(section)

    return section


def Clear() -> None:
    """Clears the terminal."""
    from .. import BUILDING_GALLERY

    if not BUILDING_GALLERY:
        syst = platform.system()
        if syst in ["Linux", "Darwin"]:
            os.system("clear")
        elif syst == "Windows":
            os.system("cls")
