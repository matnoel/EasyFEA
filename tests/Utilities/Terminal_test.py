# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import os
import platform
from typing import Any

import pytest

from EasyFEA import Terminal
from EasyFEA.Utilities import MPI_RANK

RED, CYAN, RESET = "\033[31m", "\033[36m", "\033[0m"
BOLD, ITALIC, UNDERLINE = "\033[1m", "\033[3m", "\033[4m"


def test_MyPrint_prints_and_returns_the_colored_text(capsys):
    text = Terminal.MyPrint("hello", "red")
    assert text == RED + "hello" + RESET
    assert capsys.readouterr().out == text


def test_MyPrint_defaults_to_cyan():
    assert Terminal.MyPrint("hello") == CYAN + "hello" + RESET


def test_MyPrint_prefixes_the_style(subtests):
    for style, code in [("bold", BOLD), ("italic", ITALIC), ("underLine", UNDERLINE)]:
        with subtests.test(style=style):
            kwargs: dict[str, Any] = {style: True}
            assert Terminal.MyPrint("hello", **kwargs) == code + CYAN + "hello" + RESET


def test_MyPrint_reports_an_unknown_color_in_red():
    text = Terminal.MyPrint("hello", "pink")
    assert text.startswith(RED + "Color must be in")


def test_MyPrint_appends_the_end_argument(capsys):
    Terminal.MyPrint("hello", end="\n")
    assert capsys.readouterr().out.endswith(RESET + "\n")


@pytest.mark.mpi
def test_MyPrint_prints_on_rank_0_only(capsys):
    text = Terminal.MyPrint("hello")
    assert (text is not None) == (MPI_RANK == 0)
    assert (capsys.readouterr().out != "") == (MPI_RANK == 0)


@pytest.mark.mpi
def test_MyPrint_ends_a_carriage_return_line_under_mpirun(capsys):
    Terminal.MyPrint("50 %", end="\r")
    if MPI_RANK == 0:
        assert capsys.readouterr().out.endswith(RESET + "\n\033[F")


def test_MyPrintError_prints_in_red():
    assert Terminal.MyPrintError("error") == RED + "error" + RESET


def test_Section_centers_the_text_between_edges(capsys):
    section = Terminal.Section("Solve")
    assert section == f"\n\n{'=' * 20} Solve {'=' * 20}\n"
    assert section in capsys.readouterr().out


def test_Section_prints_nothing_without_verbosity(capsys):
    Terminal.Section("Solve", verbosity=False)
    assert capsys.readouterr().out == ""


@pytest.fixture
def commands(monkeypatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(os, "system", calls.append)
    monkeypatch.delenv("EASYFEA_BUILDING_GALLERY", raising=False)
    return calls


def test_Clear_runs_the_system_clear_command(monkeypatch, commands, subtests):
    for system, command in [
        ("Linux", "clear"),
        ("Darwin", "clear"),
        ("Windows", "cls"),
    ]:
        with subtests.test(system=system):
            commands.clear()
            monkeypatch.setattr(platform, "system", lambda: system)
            Terminal.Clear()
            assert commands == [command]


def test_Clear_does_nothing_while_building_the_gallery(monkeypatch, commands):
    monkeypatch.setenv("EASYFEA_BUILDING_GALLERY", "1")
    Terminal.Clear()
    assert commands == []
