# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Mesh12
======

Meshing of a part designed by cad software.
"""

from EasyFEA import Display, Folder, Mesher, ElemType, PyVista

if __name__ == "__main__":
    Display.Clear()

    parts_dir = Folder.os.path.abspath("../_parts")

    stp = Folder.Join(parts_dir, "beam.stp")
    mesh_stp = Mesher().Mesh_Import_part(stp, 3, 13 / 5, ElemType.TETRA4)

    igs = Folder.Join(parts_dir, "beam.igs")
    mesh_igs = Mesher().Mesh_Import_part(igs, 3, 13 / 5, ElemType.TETRA10)
    # An igs file will provide a contour mesh only.

    pltr1 = PyVista.Plot_Mesh(mesh_stp)
    pltr1.add_title("beam.stp")
    pltr1.show()

    pltr2 = PyVista.Plot_Mesh(mesh_igs)
    pltr2.add_title("beam.igs")
    pltr2.show()
