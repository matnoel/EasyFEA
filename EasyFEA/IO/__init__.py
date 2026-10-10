# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Mesh and simulation files, one module per format family."""

from . import Gmsh, Medit, Ensight, PyVista, Paraview, Vizir, USD, GLTF

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._protocols import MeshSaver, MeshLoader, SimuSaver

    _gmsh_saver: MeshSaver = Gmsh
    _medit_saver: MeshSaver = Medit
    _ensight_saver: MeshSaver = Ensight
    _usd_saver: MeshSaver = USD
    _gltf_saver: MeshSaver = GLTF

    _gmsh_loader: MeshLoader = Gmsh
    _medit_loader: MeshLoader = Medit
    _ensight_loader: MeshLoader = Ensight

    _gmsh_simu: SimuSaver = Gmsh
    _paraview_simu: SimuSaver = Paraview
    _vizir_simu: SimuSaver = Vizir
    _usd_simu: SimuSaver = USD
    _gltf_simu: SimuSaver = GLTF
