# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.


import pytest
import numpy as np

from EasyFEA import Mesh, IO

L = 2
H = 1


class TestVizir:

    def test_barycentric_coordinates(self, all_elemType_meshes: list[Mesh]):

        for mesh in all_elemType_meshes:

            groupElem = mesh.groupElem
            elemType = groupElem.elemType
            local_coords = groupElem.Get_Local_Coords()
            vertices = local_coords[: groupElem.Nvertex]

            if elemType.startswith(("SEG", "TETRA", "TRI")):

                barycentric_coords = IO.Vizir._Get_BaryCentric_Coordinates(groupElem)

                verif_coords = np.zeros_like(groupElem.Get_Local_Coords())
                for i, coefs in enumerate(barycentric_coords):
                    weighted_coords = coefs[:, None] * vertices
                    verif_coords[i] = np.sum(weighted_coords, axis=0)

            else:
                continue

            test = np.linalg.norm(verif_coords - local_coords) / np.linalg.norm(
                local_coords
            )
            assert test < 1e-12


def test_save_simu_needs_a_type_per_result(tmp_path):
    from EasyFEA import Models, Simulations
    from EasyFEA.Geoms import Domain

    contour = Domain((0, 0), (1, 1), 0.5)
    mesh = contour.Mesh_2D()
    simu = Simulations.Elastic(mesh, Models.Elastic.Isotropic(2))

    with pytest.raises(ValueError, match="types"):
        IO.Vizir.Save_simu(simu, str(tmp_path), results=["ux"])
