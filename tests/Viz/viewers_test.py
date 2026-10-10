# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Behaviour both viewers share: the view covers everything drawn, and backend spellings are refused."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import pyvista as pv

from EasyFEA import ElemType, Matplotlib, PyVista
from EasyFEA.Geoms import Domain

pv.OFF_SCREEN = True


@pytest.fixture
def far_meshes():
    """A 10-wide cube at the origin and a 1-wide one at 20."""
    big = Domain((0, 0), (10, 10), 2.0)
    small = Domain((20, 20), (21, 21), 1.0)
    meshBig = big.Mesh_Extrude([], [0, 0, 10], [2], ElemType.TETRA4)
    meshSmall = small.Mesh_Extrude([], [0, 0, 1], [1], ElemType.TETRA4)
    return meshBig, meshSmall


class TestFraming:

    @pytest.mark.parametrize("second", ["Plot_Mesh", "Plot_Nodes", "Plot_Elements"])
    def test_matplotlib_3D_view_covers_both(self, far_meshes, second):
        meshBig, meshSmall = far_meshes
        ax = Matplotlib.Plot_Mesh(meshBig)
        getattr(Matplotlib, second)(meshSmall, ax=ax)
        xmin, xmax = ax.get_xlim()
        assert xmin <= 0 and xmax >= 21
        plt.close("all")

    def test_matplotlib_bounds_fix_the_view(self, far_meshes):
        meshBig, _ = far_meshes
        ax = Matplotlib.Plot_Mesh(meshBig, bounds=(0, 5, 0, 5, 0, 5))
        assert ax.get_xlim() == (0, 5)
        plt.close("all")

    def test_2D_bounds_take_four_values(self):
        contour = Domain((0, 0), (10, 10), 2.0)
        mesh = contour.Mesh_2D()
        ax = Matplotlib.Plot_Mesh(mesh, bounds=(0, 5, 0, 5))
        assert ax.get_xlim() == (0, 5)
        plt.close("all")
        PyVista.Plot_Mesh(mesh, bounds=(0, 5, 0, 5)).close()

    def test_pyvista_view_covers_both(self, far_meshes):
        meshBig, meshSmall = far_meshes
        plotter = PyVista.Plot_Mesh(meshBig)
        PyVista.Plot_Mesh(meshSmall, plotter=plotter)
        focal = np.asarray(plotter.camera.focal_point)
        assert np.allclose(focal[:2], 10.5, atol=0.5)
        plotter.close()


class TestKwargs:

    def test_matplotlib_refuses_lw(self, far_meshes):
        with pytest.raises(ValueError, match="linewidth"):
            Matplotlib.Plot(far_meshes[0], lw=2)
        plt.close("all")

    def test_pyvista_refuses_line_width(self, far_meshes):
        with pytest.raises(ValueError, match="linewidth"):
            PyVista.Plot(far_meshes[0], line_width=2)

    def test_matplotlib_forwards_other_kwargs(self, far_meshes):
        ax = Matplotlib.Plot(far_meshes[0], color="c", hatch="/")
        assert ax.collections[0].get_hatch() == "/"
        plt.close("all")
