# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np

from EasyFEA import Mesh

from _numerics import rel_error

L = 2
H = 1


def check_mesh(mesh1: Mesh, mesh2: Mesh):
    """Same element and node counts, same length, area and volume."""

    assert mesh1.Ne == mesh2.Ne
    assert mesh1.Nn == mesh2.Nn

    if mesh1.dim >= 1:
        assert rel_error(mesh1.length, mesh2.length) < 1e-12
    if mesh1.dim >= 2:
        assert rel_error(mesh1.area, mesh2.area) < 1e-12
    if mesh1.dim >= 3:
        assert rel_error(mesh1.volume, mesh2.volume) < 1e-12


def get_tags(mesh: Mesh) -> list[str]:
    return sorted(
        {
            tag
            for groupElem in mesh.dict_groupElem.values()
            for tag in groupElem.nodeTags
        }
    )


def check_tags(mesh1: Mesh, mesh2: Mesh):
    """Every tag comes back holding the same nodes, and the main group's tags the same elements."""

    tags = get_tags(mesh1)
    assert tags == get_tags(mesh2), "tag names differ"

    for tag in tags:
        assert np.array_equal(
            np.sort(mesh1.Nodes_Tags(tag)), np.sort(mesh2.Nodes_Tags(tag))
        ), f"nodes of {tag} differ"

    for tag in mesh1.groupElem.elementTags:
        assert np.array_equal(
            np.sort(mesh1.Elements_Tags(tag)), np.sort(mesh2.Elements_Tags(tag))
        ), f"elements of {tag} differ"


def frames_of(mesh: Mesh, nFrames: int = 3) -> list[np.ndarray]:
    """Rigid translations along x, one per frame."""
    return [np.tile([0.1 * i, 0, 0], (mesh.Nn, 1)) for i in range(nFrames)]


def main_group_only(mesh: Mesh) -> Mesh:
    """The mesh's main group alone, without its boundary groups."""
    return Mesh({mesh.elemType: mesh.groupElem})
