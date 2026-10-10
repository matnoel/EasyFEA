# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Medit meshes (.mesh, .meshb)."""

from ..Utilities import Folder, Terminal
from ..FEM._mesh import Mesh
from ._meshio import requires_meshio, _EasyFEA_to_Meshio, _Meshio_to_EasyFEA


@requires_meshio
def Save_mesh(
    mesh: Mesh,
    folder: str,
    name: str,
    dict_tags_converter: dict[str, int] = {},
    useBinary=False,
) -> str:
    """Converts EasyFEA mesh to Medit format.

    Parameters
    ----------
    mesh : Mesh
        EasyFEA mesh object.
    folder : str
        Directory to save the Medit file.
    name : str
        The name of the Medit file, without the extension.
    dict_tags_converter : dict[str, int], optional
        Dictionary converting string tags to integers (default is {}).
    useBinary : bool, optional
        Whether to save as binary (default is False).

    Returns
    -------
    str
        Path to the saved Medit file.

    Examples
    --------
    >>> from EasyFEA import IO
    >>> IO.Medit.Save_mesh(mesh, folder="results", name="my_mesh")
    """

    import meshio

    assert isinstance(mesh, Mesh), "mesh must be a EasyFEA mesh!"

    meshioMesh = _EasyFEA_to_Meshio(mesh, dict_tags_converter)

    extension = "meshb" if useBinary else "mesh"
    filename = Folder.Join(folder, f"{name}.{extension}", mkdir=True)

    Terminal.MyPrint(f"\nCreation of: {filename}\n", "green")
    meshio.medit.write(filename, meshioMesh)

    return filename


@requires_meshio
def Load_mesh(meditMesh: str) -> Mesh:
    """Converts Medit mesh to EasyFEA format.

    Parameters
    ----------
    meditMesh : str
        Path to the Medit mesh file.

    Returns
    -------
    Mesh
        Converted EasyFEA mesh object.

    Examples
    --------
    >>> from EasyFEA import IO
    >>> mesh = IO.Medit.Load_mesh("mesh.mesh")
    """

    import meshio

    meshioMesh = meshio.medit.read(meditMesh)
    # Please note that your python's meshio must come from https://github.com/matnoel/meshio/tree/medit_higher_order_elements

    if len(meshioMesh.cells) == 0:
        Terminal.MyPrintError(
            f"The medit mesh:\n {meditMesh}\n does not contain any elements!"
        )
        return None  # type: ignore [return-value]

    mesh = _Meshio_to_EasyFEA(meshioMesh)

    return mesh
