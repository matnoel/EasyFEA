# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest
import matplotlib.pyplot as plt
import numpy as np

from EasyFEA import Mesher, Display


class TestDisplay:

    def test_Plot_2D(self):
        """Builds all 2D meshes"""
        list_mesh2D = Mesher._Construct_2D_meshes()
        nbMesh = len(list_mesh2D)
        nrows = 5
        ncols = 10
        assert nbMesh < nrows * ncols, "Not enough space"
        fig, axs = plt.subplots(nrows, ncols)
        axs: list[Display.plt.Axes] = np.ravel(axs)

        for m, mesh2D in enumerate(list_mesh2D):
            ax = axs[m]
            ax.axis("off")
            Display.Plot_Mesh(mesh2D, ax=ax)
            Display.Plot_Nodes(mesh2D, showId=False, ax=ax, color="black")
            ax.set_title("")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # plt.pause(1e-12)

            Display.Plot_Tags(mesh2D)
            # plt.pause(1e-12)
            plt.close()

        # plt.show()

    def test_Plot_3D(self):
        """Builds all 3D meshes"""
        list_mesh3D = Mesher._Construct_3D_meshes(useImport3D=True)
        for mesh3D in list_mesh3D:
            ax = Display.Plot_Mesh(mesh3D)
            Display.Plot_Nodes(mesh3D, showId=False, ax=ax, color="black")
            # plt.pause(1e-12)
            ax.axis("off")
            plt.close()

            Display.Plot_Tags(mesh3D)
            # plt.pause(1e-12)
            plt.close()

        # plt.show()
