# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import os

import pytest

from EasyFEA import Folder
from EasyFEA.Utilities import MPI_RANK

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def gallery(monkeypatch):
    monkeypatch.setenv("EASYFEA_BUILDING_GALLERY", "1")


def execute_code_block(script_vars: dict) -> str:
    """Stands for sphinx-gallery's frame, where `Folder` reads the running script."""
    return Folder.Dir()


def test_Dir_returns_the_nth_parent(tmp_path, subtests):
    file = tmp_path / "a" / "b" / "file.py"
    for n in [1, 2]:
        with subtests.test(n=n):
            assert Folder.Dir(str(file), n=n) == str(file.parents[n - 1])


def test_Dir_ignores_a_trailing_separator(tmp_path):
    assert Folder.Dir(str(tmp_path / "sub") + os.sep) == str(tmp_path)


def test_Dir_defaults_to_the_calling_script():
    assert Folder.Dir() == HERE


def test_Dir_defaults_to_the_gallery_script_while_building_the_gallery(
    gallery, tmp_path
):
    assert execute_code_block({"src_file": str(tmp_path / "plot.py")}) == str(tmp_path)


def test_Dir_fails_when_the_gallery_frame_has_no_script(gallery):
    with pytest.raises(Exception, match="sphinx_gallery may change"):
        execute_code_block({})


def test_Dir_fails_outside_a_gallery_frame_while_building_the_gallery(gallery):
    with pytest.raises(AssertionError):
        Folder.Dir()


def test_Dir_rejects_invalid_arguments(subtests):
    for path, n in [(123, 1), ("dir", 0), ("dir", -1), ("dir", 1.5)]:
        with subtests.test(path=path, n=n):
            with pytest.raises(AssertionError):
                Folder.Dir(path, n=n)


def test_Join_joins_without_creating_by_default(tmp_path):
    path = Folder.Join(str(tmp_path), "a", "b")
    assert path == os.path.join(str(tmp_path), "a", "b")
    assert not os.path.exists(path)


def test_Join_mkdir_creates_nested_directories(tmp_path):
    path = Folder.Join(str(tmp_path), "a", "b", "c", mkdir=True)
    assert os.path.isdir(path)


def test_Join_mkdir_creates_the_parent_of_a_file(tmp_path):
    path = Folder.Join(str(tmp_path), "subdir", "file.txt", mkdir=True)
    assert os.path.isdir(tmp_path / "subdir")
    assert not os.path.exists(path)


def test_Join_mkdir_accepts_an_existing_directory(tmp_path):
    assert Folder.Join(str(tmp_path), mkdir=True) == str(tmp_path)


def test_Is_building_gallery_reads_the_environment_at_call_time(monkeypatch, subtests):
    monkeypatch.delenv("EASYFEA_BUILDING_GALLERY", raising=False)
    assert not Folder._Is_building_gallery()
    for value, expected in [("1", True), ("TRUE", True), ("0", False)]:
        with subtests.test(value=value):
            monkeypatch.setenv("EASYFEA_BUILDING_GALLERY", value)
            assert Folder._Is_building_gallery() is expected


def test_Results_Dir_is_the_script_results_folder():
    assert Folder.Results_Dir() == os.path.join(HERE, "results", "Folder_test")


def test_Exists_finds_files_and_directories(tmp_path):
    file = tmp_path / "file.txt"
    file.touch()
    assert Folder.Exists(str(file)) is True
    assert Folder.Exists(str(tmp_path)) is True
    assert Folder.Exists(str(tmp_path / "ghost.txt")) is False


def test_Exists_is_false_for_a_broken_symlink(tmp_path):
    link = tmp_path / "link.txt"
    link.symlink_to(tmp_path / "target.txt")
    assert Folder.Exists(str(link)) is False


def test_Rank_Dir_is_the_path_in_serial():
    assert Folder.Rank_Dir("/some/path") == "/some/path"


@pytest.mark.mpi
def test_Rank_Dir_adds_the_rank_folder():
    assert Folder.Rank_Dir("/some/path") == os.path.join(
        "/some/path", f"Rank{MPI_RANK}"
    )
