# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import os
import pytest

from EasyFEA import Folder


class TestDir:

    def test_n1_returns_parent(self, tmp_path):
        file = tmp_path / "sub" / "file.py"
        file.parent.mkdir()
        file.touch()
        result = Folder.Dir(str(file), n=1)
        assert result == str(tmp_path / "sub")

    def test_n2_returns_grandparent(self, tmp_path):
        file = tmp_path / "a" / "b" / "file.py"
        file.parent.mkdir(parents=True)
        file.touch()
        result = Folder.Dir(str(file), n=2)
        assert result == str(tmp_path / "a")

    def test_on_directory_path(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        result = Folder.Dir(str(subdir), n=1)
        assert result == str(tmp_path)

    def test_invalid_path_type_raises(self):
        with pytest.raises(AssertionError):
            Folder.Dir(123)

    def test_invalid_n_zero_raises(self, tmp_path):
        with pytest.raises(AssertionError):
            Folder.Dir(str(tmp_path), n=0)

    def test_invalid_n_negative_raises(self, tmp_path):
        with pytest.raises(AssertionError):
            Folder.Dir(str(tmp_path), n=-1)

    def test_invalid_n_float_raises(self, tmp_path):
        with pytest.raises(AssertionError):
            Folder.Dir(str(tmp_path), n=1.5)

    def test_normalizes_trailing_sep(self, tmp_path):
        path_with_sep = str(tmp_path / "sub") + os.sep
        sub = tmp_path / "sub"
        sub.mkdir()
        result = Folder.Dir(path_with_sep, n=1)
        assert result == str(tmp_path)


class TestJoin:

    def test_joins_components(self, tmp_path):
        result = Folder.Join(str(tmp_path), "a", "b")
        assert result == os.path.join(str(tmp_path), "a", "b")

    def test_no_mkdir_by_default(self, tmp_path):
        path = Folder.Join(str(tmp_path), "new_dir")
        assert not os.path.exists(path)

    def test_mkdir_creates_directory(self, tmp_path):
        path = Folder.Join(str(tmp_path), "new_dir", mkdir=True)
        assert os.path.isdir(path)

    def test_mkdir_creates_nested_directories(self, tmp_path):
        path = Folder.Join(str(tmp_path), "a", "b", "c", mkdir=True)
        assert os.path.isdir(path)

    def test_mkdir_with_file_path_creates_parent_dir(self, tmp_path):
        path = Folder.Join(str(tmp_path), "subdir", "file.txt", mkdir=True)
        assert os.path.isdir(os.path.join(str(tmp_path), "subdir"))
        assert not os.path.exists(path)

    def test_mkdir_existing_path_no_error(self, tmp_path):
        path = Folder.Join(str(tmp_path), mkdir=True)
        assert os.path.isdir(path)

    def test_returns_string(self, tmp_path):
        result = Folder.Join(str(tmp_path), "x")
        assert isinstance(result, str)


class TestExists:

    def test_existing_file_returns_true(self, tmp_path):
        f = tmp_path / "file.txt"
        f.touch()
        assert Folder.Exists(str(f)) is True

    def test_existing_directory_returns_true(self, tmp_path):
        assert Folder.Exists(str(tmp_path)) is True

    def test_nonexistent_path_returns_false(self, tmp_path):
        assert Folder.Exists(str(tmp_path / "ghost.txt")) is False

    def test_broken_symlink_returns_false(self, tmp_path):
        target = tmp_path / "target.txt"
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        # target does not exist → broken symlink
        assert Folder.Exists(str(link)) is False


class TestRankDir:

    def test_serial_returns_same_path(self, tmp_path):
        # MPI_SIZE == 1 in normal test runs
        result = Folder.Rank_Dir(str(tmp_path))
        assert result == str(tmp_path)

    def test_serial_string_unchanged(self):
        result = Folder.Rank_Dir("/some/path")
        assert result == "/some/path"
