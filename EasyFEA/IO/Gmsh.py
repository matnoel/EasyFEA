# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Gmsh meshes (.msh)."""

from __future__ import annotations
import sys
from typing import TYPE_CHECKING, Sequence
import gmsh
import numpy as np

from ..Utilities import Folder, Terminal, _types
from ..Utilities._requires import Create_requires_decorator
from ..FEM._mesh import Mesh
from ._meshio import (
    requires_meshio,
    _EasyFEA_to_Meshio,
    _Meshio_to_EasyFEA,
    _Get_dict_tags_converter,
)

if TYPE_CHECKING:
    import meshio
    from ..Simulations._simu import _Simu

requires_matplotlib = Create_requires_decorator("matplotlib")


@requires_meshio
def Save_mesh(
    mesh: Mesh, folder: str, name: str, *, useBinary: bool | None = None
) -> str:
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
        Whether to save as binary, by default None (text).

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

    meshio.gmsh.write(filename, meshioMesh, "2.2", bool(useBinary))
    # Error with 4.1

    return filename


@requires_meshio
def Load_mesh(path: str) -> Mesh:
    """Converts Gmsh mesh to EasyFEA format.

    Args:
        path (str): Path to the Gmsh mesh file.

    Returns:
        Mesh: Converted EasyFEA mesh object.

    Examples
    --------
    >>> from EasyFEA import IO
    >>> mesh = IO.Gmsh.Load_mesh("mesh.msh")
    """

    import meshio

    meshioMesh: meshio.Mesh = meshio.gmsh.read(path)

    if len(meshioMesh.cells) == 0:
        Terminal.MyPrintError(
            f"The gmsh mesh:\n {path}\n does not contain any elements!"
        )
        return None  # type: ignore [return-value]

    mesh = _Meshio_to_EasyFEA(meshioMesh)

    return mesh


@requires_matplotlib
def Save_simu(
    simu: _Simu,
    folder: str,
    N: int = 200,
    *,
    results: Sequence[str] = (),
    details: bool = False,
    edgecolor: str = "black",
    plotMesh: bool = True,
    showAxes: bool = False,
    openGmsh: bool = False,
) -> str:
    """Save the simulation in gmsh.pos format using gmsh.view

    Parameters
    ----------
    simu : _Simu
        simulation
    folder : str
        folder used to save the simu.pos file
    N : int, optional
        Maximal number of iterations saved, by default 200
    results : Sequence[str], optional
        results saved on top of the default ones, by default ()
    details : bool, optional
        get default result values with details or not see `simu.Results_nodesField_elementsField(details)`, by default False
    edgecolor : str, optional
        color used to plot the edges, by default 'black'
    plotMesh : bool, optional
        plot the mesh, by default True
    showAxes : bool, optional
        show the axes, by default False
    openGmsh : bool, optional
        opens the gmsh window, by default False

    Returns
    -------
    str
        Path to the simu.pos file.
    """

    results = list(results)
    path = Folder.Join(folder, "simu.pos", mkdir=True)
    if Folder.os.path.exists(path):
        # each view is appended to the file
        Folder.os.remove(path)

    # get mesh informations
    mesh = simu.mesh

    from matplotlib.colors import to_rgb

    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("model")

    def getColor(c: str):
        """transform matplotlib color to rgb"""
        rgb = np.asarray(to_rgb(c)) * 255
        rgb = np.asarray(rgb, dtype=int)
        return rgb

    def reshape(values: _types.FloatArray, connect_e: _types.IntArray):
        """reshape nodal values to get them at the corners of the elements"""
        values_n: _types.FloatArray = np.reshape(values, (mesh.Nn, -1))
        values_e = values_n[connect_e]
        if len(values_e.shape) == 3:
            values_e = np.transpose(values_e, (0, 2, 1))
        return values_e.reshape((connect_e.shape[0], -1))

    gmshTopo = {
        "POINT": "P",
        "SEG": "L",
        "TRI": "T",
        "QUAD": "Q",
        "TETRA": "S",
        "HEXA": "H",
        "PRISM": "I",
        "PYRA": "Y",
    }

    colorElems = getColor(edgecolor)

    # one static block per element group of the main dimension; this lets meshes that mix element types (e.g. QUAD4 + TRI3) be exported into a single gmsh view through several addListData calls.
    group_blocks = []
    for groupElem in mesh.Get_list_groupElem(mesh.dim):
        # quadratic elements are exported with their corner nodes only
        nbCorners = groupElem.Nvertex
        connect_e = groupElem.connect[:, :nbCorners]
        group_blocks.append(
            (
                groupElem.Ne,
                connect_e,
                gmshTopo[groupElem.elemType.topology],
                reshape(mesh.coord, connect_e),  # corner coordinates
            )
        )

    # get nodes and elements field to plot
    nodesField, elementsField = simu.Results_nodeFields_elementFields(details)
    [
        results.append(result)  # type: ignore [func-returns-value]
        for result in (nodesField + elementsField)
        if result not in results
    ]

    dict_results: dict[str, list[_types.FloatArray]] = {
        result: [] for result in results
    }

    Niter = simu.Niter
    iterations = np.linspace(
        0,
        Niter - 1,
        min(Niter, N),
        endpoint=True,
        dtype=int,
    )

    # activates the first iteration
    simu.Set_Iter(0, resetAll=True)

    for i in iterations:
        simu.Set_Iter(i)
        for result in results:
            dict_results[result].append(
                np.asarray(simu.Result(result))
            )  # raw nodal field

    def AddView(name: str, list_values: list[_types.FloatArray]):
        """Add a view; list_values holds one nodal field per iteration."""

        if name == "displacement_matrix_0":
            name = "ux"
        elif name == "displacement_matrix_1":
            name = "uy"
        elif name == "displacement_matrix_2":
            name = "uz"

        view = gmsh.view.add(name)

        gmsh.view.option.setNumber(view, "IntervalsType", 3)
        # (1: iso, 2: continuous, 3: discrete, 4: numeric)
        gmsh.view.option.setNumber(view, "NbIso", 10)

        if plotMesh:
            gmsh.view.option.setNumber(view, "ShowElement", 1)

        if showAxes:
            gmsh.view.option.setNumber(view, "Axes", 1)
            # (0: none, 1: simple axes, 2: box, 3: full grid, 4: open grid, 5: ruler)

        gmsh.view.option.setColor(view, "Lines", *colorElems)
        gmsh.view.option.setColor(view, "Triangles", *colorElems)
        gmsh.view.option.setColor(view, "Quadrangles", *colorElems)
        gmsh.view.option.setColor(view, "Tetrahedra", *colorElems)
        gmsh.view.option.setColor(view, "Hexahedra", *colorElems)
        gmsh.view.option.setColor(view, "Pyramids", *colorElems)
        gmsh.view.option.setColor(view, "Prisms", *colorElems)

        # one scalar data block per element group (time steps stacked along axis 1)
        for Ne, connect_e, gmshType, elements_e in group_blocks:
            values_e = np.concatenate(
                [reshape(values, connect_e) for values in list_values], axis=1
            )
            res = np.concatenate((elements_e, values_e), axis=1)
            gmsh.view.addListData(view, "S" + gmshType, Ne, res.ravel())

        gmsh.view.write(view, path, True)

        return view

    for result, list_values in dict_results.items():
        nIter = len(list_values)

        if nIter == 0:
            continue

        dof_n = np.reshape(list_values[0], (mesh.Nn, -1)).shape[1]

        if dof_n == 1:
            AddView(result, list_values)
        else:
            [
                AddView(
                    result + f"_{n}",
                    [np.reshape(v, (mesh.Nn, -1))[:, n] for v in list_values],
                )
                for n in range(dof_n)
            ]

    # Launch the GUI to see the results:
    if "-nopopup" not in sys.argv and openGmsh:
        gmsh.fltk.run()

    gmsh.finalize()

    return path
