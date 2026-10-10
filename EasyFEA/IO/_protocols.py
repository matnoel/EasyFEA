# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""What a format module can do; mypy checks the claims at the bottom."""

from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    from ..FEM._mesh import Mesh
    from ..Simulations._simu import _Simu


class MeshSaver(Protocol):
    def Save_mesh(
        self, mesh: Mesh, folder: str, name: str, *, useBinary: bool | None = None
    ) -> str | None:
        """Writes the mesh, `useBinary=None` meaning the format's default; returns the path."""
        ...


class MeshLoader(Protocol):
    def Load_mesh(self, path: str) -> Mesh:
        """Reads a mesh file."""
        ...


class SimuSaver(Protocol):
    def Save_simu(
        self, simu: _Simu, folder: str, N: int = ..., *, results: Sequence[str] = ()
    ) -> str | None:
        """Writes the saved iterations (at most `N`); returns what was written."""
        ...


if TYPE_CHECKING:
    # here, outside the IO import cycle, where mypy knows the decorated functions
    from . import Gmsh, Medit, Ensight, Paraview, Vizir, USD, GLTF

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
