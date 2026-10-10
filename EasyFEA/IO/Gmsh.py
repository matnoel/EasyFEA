# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Gmsh meshes (.msh)."""

from __future__ import annotations
from typing import TYPE_CHECKING

from ..Utilities import Folder, Terminal
from ..FEM._mesh import Mesh
from ._meshio import (
    requires_meshio,
    _EasyFEA_to_Meshio,
    _Meshio_to_EasyFEA,
    _Get_dict_tags_converter,
)

if TYPE_CHECKING:
    import meshio


@requires_meshio
def Save_mesh(mesh: Mesh, folder: str, name: str, useBinary=False) -> str:
    """Converts EasyFEA mesh to Gmsh format.

    Parameters
    ----------
    mesh : Mesh
        EasyFEA mesh object.
    folder : str
        Directory to save the Gmsh file.
    name : str
        The name of the Gmsh file, without the extension.
    useBinary : bool, optional
        Whether to save as binary (default is False).

    Returns
    -------
    str
        Path to the saved Gmsh file.

    Examples
    --------
    >>> from EasyFEA import IO
    >>> IO.Gmsh.Save_mesh(mesh, folder="results", name="my_mesh")
    """

    import meshio

    assert isinstance(mesh, Mesh), "mesh must be a EasyFEA mesh!"

    dict_tags_converter = _Get_dict_tags_converter(mesh)

    # gmsh:physical is the only key meshio writes as an element reference; anything else lands
    # in $ElementData, where the references are lost and the zeroed gmsh:physical wins on read
    meshioMesh = _EasyFEA_to_Meshio(mesh, dict_tags_converter, "gmsh:physical")

    # gmsh resolves a physical group through the geometrical entities it holds, so leaving
    # gmsh:geometrical unset puts every element in entity 0 and each group then covers the whole
    # mesh. Giving each tag its own entity keeps the groups apart, which is what
    # Mesher.Mesh_Import_mesh reads back through getNodesForPhysicalGroup.
    meshioMesh.cell_data["gmsh:geometrical"] = [
        data.copy() for data in meshioMesh.cell_data["gmsh:physical"]
    ]

    filename = Folder.Join(folder, f"{name}.msh", mkdir=True)

    Terminal.MyPrint(f"\nCreation of: {filename}", "green")

    meshio.gmsh.write(filename, meshioMesh, "2.2", useBinary)
    # Error with 4.1

    return filename


@requires_meshio
def Load_mesh(gmshMesh: str) -> Mesh:
    """Converts Gmsh mesh to EasyFEA format.

    Args:
        gmshMesh (str): Path to the Gmsh mesh file.

    Returns:
        Mesh: Converted EasyFEA mesh object.

    Examples
    --------
    >>> from EasyFEA import IO
    >>> mesh = IO.Gmsh.Load_mesh("mesh.msh")
    """

    import meshio

    meshioMesh: meshio.Mesh = meshio.gmsh.read(gmshMesh)

    if len(meshioMesh.cells) == 0:
        Terminal.MyPrintError(
            f"The gmsh mesh:\n {gmshMesh}\n does not contain any elements!"
        )
        return None  # type: ignore [return-value]

    mesh = _Meshio_to_EasyFEA(meshioMesh)

    return mesh
