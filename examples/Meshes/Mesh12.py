# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Meshing of a part designed by cad software."""

from EasyFEA import Display, Folder, Mesher, ElemType

if __name__ == "__main__":

    folder = Folder.Dir(__file__)  # Meshes folder
    examples_folder = Folder.Dir(folder)

    stp = Folder.Join(examples_folder, "_parts", "beam.stp")
    mesh_stp = Mesher().Mesh_Import_part(stp, 3, 13 / 5, ElemType.TETRA4)

    igs = Folder.Join(examples_folder, "_parts", "beam.igs")
    mesh_igs = Mesher().Mesh_Import_part(igs, 3, 13 / 5, ElemType.TETRA10)
    # An igs file will provide a contour mesh only.

    Display.Plot_Mesh(mesh_stp)
    Display.Plot_Mesh(mesh_igs)

    Display.plt.show()
