# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import matplotlib.pyplot as plt
import numpy as np
import pytest

from EasyFEA import ElemType, Folder, Mesh, Mesher, Matplotlib
from EasyFEA.Geoms import Circle, CircleArc, Domain, Line, Point, Points


class TestMatplotlib:

    def test_Plot_2D(self, cracked_meshes_2D: list[Mesh]):
        """Builds all 2D meshes"""
        list_mesh2D = cracked_meshes_2D
        nbMesh = len(list_mesh2D)
        nrows = 5
        ncols = 10
        assert nbMesh < nrows * ncols, "Not enough space"
        fig, axs = plt.subplots(nrows, ncols)
        axs: list[Matplotlib.plt.Axes] = np.ravel(axs)

        for m, mesh2D in enumerate(list_mesh2D):
            ax = axs[m]
            ax.axis("off")
            Matplotlib.Plot_Mesh(mesh2D, ax=ax)
            Matplotlib.Plot_Nodes(mesh2D, showId=False, ax=ax, color="black")
            ax.set_title("")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # plt.pause(1e-12)

            Matplotlib.Plot_Tags(mesh2D)
            # plt.pause(1e-12)
            plt.close()

        # plt.show()

    def test_Plot_3D(self, cracked_meshes_3D: list[Mesh]):
        """Builds all 3D meshes"""
        list_mesh3D = list(cracked_meshes_3D)

        partPath = Folder.Join(Folder.EASYFEA_DIR, "examples", "_parts", "beam.stp")
        if Folder.Exists(partPath):
            mesher = Mesher()
            for elemType in [ElemType.TETRA4, ElemType.TETRA10]:
                list_mesh3D.append(
                    mesher.Mesh_Import_part(
                        partPath, 3, meshSize=7.5, elemType=elemType
                    )
                )

        for mesh3D in list_mesh3D:
            ax = Matplotlib.Plot_Mesh(mesh3D)
            Matplotlib.Plot_Nodes(mesh3D, showId=False, ax=ax, color="black")
            # plt.pause(1e-12)
            ax.axis("off")
            plt.close()

            Matplotlib.Plot_Tags(mesh3D)
            # plt.pause(1e-12)
            plt.close()

        # plt.show()


def _rounded_corners() -> Points:
    return Points([Point(), Point(5, 0), Point(5, 5, r=2), Point(0, 5, r=-3)])


@pytest.mark.parametrize(
    "makeGeom",
    [
        lambda: Line(Point(), Point(5, 1)),
        lambda: Domain(Point(), Point(1, 1, 1)),
        lambda: Circle(Point(), 5, n=(1, 1, 1)),
        lambda: CircleArc(Point(3, 1, 3), Point(-3, 1, 3), center=Point(0, 1)),
        _rounded_corners,
        lambda: _rounded_corners().Get_Contour(),
    ],
    ids=["Line", "Domain", "Circle", "CircleArc", "Points", "Contour"],
)
def test_Plot_Geoms_draws_on_the_given_axes(makeGeom):
    geom = makeGeom()
    ax = Matplotlib.Plot_Geoms(geom)
    nLines = len(ax.lines)

    moved = geom.Translate(10, copy=True)
    assert Matplotlib.Plot_Geoms(moved, ax=ax) is ax
    assert len(ax.lines) > nLines
    plt.close("all")
