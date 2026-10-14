# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import time

import pytest

from EasyFEA import Tic


@pytest.fixture(autouse=True)
def empty_history():
    Tic.Clear()
    yield
    Tic.Clear()


def test_Tic_times_from_its_construction():
    tic = Tic()
    time.sleep(0.01)
    assert tic.Tac() >= 0.01


def test_Tic_Get_time_unity_at_unit_boundaries(subtests):
    for seconds, (valueRef, unityRef) in [
        (5e-4, (500.0, "µs")),
        (1e-3, (1.0, "ms")),
        (1.0, (1.0, "s")),
        (60.0, (1.0, "m")),
        (3600.0, (1.0, "h")),
        (86400.0, (1.0, "j")),
    ]:
        with subtests.test(seconds=seconds):
            value, unity = Tic.Get_time_unity(seconds)
            assert unity == unityRef
            assert value == pytest.approx(valueRef)


def test_Tic_Get_Remaining_Time_is_empty_at_the_first_step():
    assert Tic.Get_Remaining_Time(0, 10, 1.0) == ""


def test_Tic_Get_Remaining_Time_extrapolates_the_step_time():
    assert Tic.Get_Remaining_Time(5, 10, 2.0) == "(50.00 %) 10.00 s"


def test_Tic_Tac_accumulates_the_history_per_category_and_text():
    tic = Tic()
    first = tic.Tac("Solve", "K")
    second = tic.Tac("Solve", "K")
    tic.Tac("Mesh", "gmsh")
    history = Tic.Get_History()
    assert history["Solve"]["K"] == [pytest.approx(first + second), 2]
    assert history["Mesh"]["gmsh"][1] == 1


def test_Tic_Tac_restarts_the_timer():
    tic = Tic()
    time.sleep(0.02)
    first = tic.Tac()
    assert tic.Tac() < first


def test_Tic_Tac_prints_the_text_with_the_time_when_verbose(capsys):
    Tic().Tac("Solve", "K", verbosity=True)
    assert capsys.readouterr().out.startswith("K (")


def test_Tic_Clear_empties_the_history():
    Tic().Tac("Solve", "K")
    Tic.Clear()
    assert Tic.Get_History() == {}


def test_Tic_nTic_counts_the_categories():
    tic = Tic()
    tic.Tac("Solve", "K")
    tic.Tac("Solve", "F")
    tic.Tac("Mesh", "gmsh")
    assert Tic.nTic() == 2


def test_Tic_Resume_is_empty_without_history():
    assert Tic.Resume() == ""


def test_Tic_Resume_lists_one_line_per_category(capsys):
    tic = Tic()
    tic.Tac("Solve", "K")
    tic.Tac("Mesh", "gmsh")
    resume = Tic.Resume(verbosity=True)
    lines = resume.split("\n")[1:]
    assert [line.split(":")[0] for line in lines] == ["Solve", "Mesh"]
    assert capsys.readouterr().out.split("\n")[:-1] == lines


def test_Tic_Get_History_maps_category_and_text_to_total_time_and_count():
    elapsed = Tic().Tac("Solve", "K")
    assert Tic.Get_History() == {"Solve": {"K": [pytest.approx(elapsed), 1]}}
