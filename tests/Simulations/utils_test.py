# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import os

import numpy as np
import pytest

from EasyFEA.Simulations import Load_pickle, Save_pickle


def test_Save_pickle_writes_folder_filename_pickle(tmp_path):
    folder = str(tmp_path / "new")

    Save_pickle([1, 2], folder, "obj")

    assert os.path.isfile(os.path.join(folder, "obj.pickle"))


def test_Load_pickle_restores_what_Save_pickle_wrote(tmp_path):
    obj = {"u": np.arange(4.0), "name": "bar"}
    Save_pickle(obj, str(tmp_path), "obj")

    loaded = Load_pickle(str(tmp_path), "obj")

    assert loaded.keys() == obj.keys()
    assert np.array_equal(loaded["u"], obj["u"])
    assert loaded["name"] == "bar"


def test_Load_pickle_rejects_a_missing_file(tmp_path):
    with pytest.raises(AssertionError, match="does not exist"):
        Load_pickle(str(tmp_path), "missing")
