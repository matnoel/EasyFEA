# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA import ElemType, Mesh
from EasyFEA.Geoms import Domain

L = 2
H = 1


def far_meshes() -> tuple[Mesh, Mesh]:
    """A 10-wide cube at the origin and a 1-wide one at 20."""
    big = Domain((0, 0), (10, 10), 2.0)
    small = Domain((20, 20), (21, 21), 1.0)
    meshBig = big.Mesh_Extrude([], [0, 0, 10], [2], ElemType.TETRA4)
    meshSmall = small.Mesh_Extrude([], [0, 0, 1], [1], ElemType.TETRA4)
    return meshBig, meshSmall


def element_tags(mesh: Mesh) -> list[str]:
    """Element tags holding at least one element, over every group."""
    return [
        tag
        for groupElem in mesh.dict_groupElem.values()
        for tag in groupElem.elementTags
        if len(groupElem.Get_Elements_Tag(tag)) > 0
    ]
