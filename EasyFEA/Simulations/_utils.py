# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Callable

from ..Utilities import Folder, _types
from ..FEM._linalg import FeArray
from ..FEM import _kelvin_mandel

import numpy as np
import pickle

# ----------------------------------------------
# Save obj in pickle file
# ----------------------------------------------


def Save_pickle(obj, folder: str, filename: str) -> None:
    """Saves the object in folder/filename.pickle."""

    file = Folder.Join(folder, f"{filename}.pickle")

    Folder.os.makedirs(folder, exist_ok=True)

    with open(file, "wb") as f:
        pickle.dump(obj, f)


def Load_pickle(folder: str, filename: str):
    """Returns folder/filename.pickle object."""

    file = Folder.Join(folder, f"{filename}.pickle")

    shortName = file.replace(Folder.EASYFEA_DIR, "")
    error = f"{shortName} does not exist"
    assert Folder.Exists(file), error

    with open(file, "rb") as f:
        obj = pickle.load(f)

    return obj


# ----------------------------------------------
# Strain and stress results
# ----------------------------------------------


def _Field_per_groupElem(
    field_e_pg: Callable[..., FeArray.FeArrayALike], list_groupElem: list
) -> FeArray.FeArrayALike | dict:
    """``field_e_pg(groupElem)`` as an ``FeArray`` on one group, ``{groupElem: FeArray}`` on several."""
    if len(list_groupElem) == 1:
        return field_e_pg(list_groupElem[0])
    return {groupElem: field_e_pg(groupElem) for groupElem in list_groupElem}


def __Result_e_pg(field_e_pg: FeArray.FeArrayALike, result: str) -> _types.FloatArray:
    """(Ne, nPg, …) ``result`` of a Kelvin–Mandel strain or stress field: a component, ``vm``, or the whole field."""
    assert (
        isinstance(field_e_pg, FeArray) and field_e_pg._ndim == 1
    ), "must be a vector FeArray"
    values = _kelvin_mandel.Components(np.asarray(field_e_pg))
    if result in values:
        return values[result]
    if result == "vm":
        xx, yy, zz, yz, xz, xy = [
            values.get(name, 0.0) for name in _kelvin_mandel.ORDER
        ]
        return np.sqrt(
            0.5
            * (
                (xx - yy) ** 2
                + (yy - zz) ** 2
                + (zz - xx) ** 2
                + 6 * (xy**2 + yz**2 + xz**2)
            )
        )
    if result in ("Strain", "Stress", "Green-Lagrange", "Piola-Kirchhoff"):
        return np.stack(list(values.values()), -1)
    raise ValueError(
        f"result must be in [{', '.join(values)}, vm, Strain, Stress, Green-Lagrange, Piola-Kirchhoff]"
    )


def Result_strain_or_stress_field_e(
    field: FeArray.FeArrayALike | dict,
    result: str,
) -> _types.FloatArray:
    """Per-element ``result`` of a strain/stress ``field`` — one ``FeArray`` or ``{groupElem: FeArray}`` concatenated in its order — Gauss points averaged."""
    fields = field.values() if isinstance(field, dict) else [field]
    return np.concatenate(
        [__Result_e_pg(field_e_pg, result).mean(1) for field_e_pg in fields]
    )
