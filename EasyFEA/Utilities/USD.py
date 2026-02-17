# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module providing an interface with Graphics Library Transmission Format (GLTF) using pygltflib (https://pypi.org/project/pygltflib/)."""
# https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#geometry

from __future__ import annotations
import os
from typing import TYPE_CHECKING

import numpy as np

from ._requires import Create_requires_decorator
from ..Simulations._simu import _Init_obj

from . import Folder, Display

if TYPE_CHECKING:
    from ..FEM._mesh import Mesh
    from ..Simulations import _Simu

try:
    from pxr import Usd, UsdGeom, Gf, UsdUtils
except ImportError:
    pass
requires_pxr = Create_requires_decorator("pxr", libraries=["usd-core"])

from .GLTF import _get_list_nodesValues


@requires_pxr
def Save_simu(
    simu: "_Simu",
    results: list[str],
    folder: str,
    filename: str = None,
    N: int = 50,
    deformFactor=1.0,
    plotMesh=False,
    fps: int = 30,
    unit: float = 1.0,
) -> None:
    """Saves the simulation as usda file.

    Parameters
    ----------
    simu : _Simu
        simulation
    results : list[str]
        results that you want to plot
    folder : str
        folder where you want to save the video
    N : int, optional
        Maximal number of iterations displayed, by default 200
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    plotMesh : bool, optional
        displays mesh, by default False
    fps : int, optional
        Frames per second, by default 30
    unit: float, optional
        Meters per unit, by default 1.0

    Returns
    -------
    str
        The path to the created usda file.
    """

    simu, mesh, _, _ = _Init_obj(simu)  # type: ignore [assignment]

    if simu is None:
        Display.MyPrintError("Must give a simulation.")
        return

    Niter = len(simu.results)
    N = np.min([Niter, N])
    iterations = np.linspace(0, Niter - 1, N, endpoint=True, dtype=int)

    for result in results:

        # init list
        list_displacementMatrix: list[np.ndarray] = [None] * N
        list_nodesValues_n: list[np.ndarray] = [None] * N
        list_mesh: list[Mesh] = [None] * N

        # activates the first iteration
        simu.Set_Iter(0, resetAll=True)

        # get values
        for i, iter in enumerate(iterations):
            simu.Set_Iter(iter)
            list_mesh[i] = simu.mesh
            list_displacementMatrix[i] = (
                deformFactor * simu.Results_displacement_matrix()
            )
            list_nodesValues_n[i] = simu.Result(result).reshape(simu.mesh.Nn, -1)

        dof_n = list_nodesValues_n[0].shape[1]

        if dof_n == 1:
            unknowns = [""]
        else:
            unknowns = [f"_{d}" for d in range(dof_n)]

        # save each dofs
        for d in range(dof_n):
            Save_mesh(
                mesh=list_mesh,
                folder=folder,
                filename=f"{result}{unknowns[d]}",
                list_displacementMatrix=list_displacementMatrix,
                list_nodesValues_n=[
                    nodesValues_n[:, d] for nodesValues_n in list_nodesValues_n
                ],
                plotMesh=plotMesh,
                fps=fps,
                unit=unit,
            )

        if dof_n > 1:
            Save_mesh(
                mesh=list_mesh,
                folder=folder,
                filename=f"{result}_norm",
                list_displacementMatrix=list_displacementMatrix,
                list_nodesValues_n=list_nodesValues_n,
                plotMesh=plotMesh,
                fps=fps,
                unit=unit,
            )


@requires_pxr
def Save_mesh(
    mesh: "Mesh",
    folder: str,
    filename: str = "mesh",
    list_displacementMatrix: list[np.ndarray] = [],
    list_nodesValues_n: list[np.ndarray] = [],
    plotMesh=False,
    cmap: str = "jet",
    fps: int = 30,
    unit: float = 1.0,
) -> str:
    """Saves the mesh as glb file.

    Parameters
    ----------
    mesh : Mesh
        The mesh
    folder : str
        The directory where the results will be saved.
    filename : str, optional
        The name of the solution file, by default "mesh"
    list_displacementMatrix : list[np.ndarray], optional
        List of displacement matrix, by default []
    plotMesh : bool, optional
        displays mesh, by default False
    cmap: str, optional
        the color map used near the figure, by default "jet" \n
        ["jet", "seismic", "binary", "viridis"] -> https://matplotlib.org/stable/tutorials/colors/colormaps.html
    fps : int, optional
        Frames per second, by default 30
    unit: float, optional
        Meters per unit, by default 1.0

    Returns
    -------
    str
        The path to the created glb file.
    """

    updatedMesh = isinstance(mesh, list)
    if updatedMesh:
        list_mesh = mesh
        mesh = list_mesh[0]
    else:
        list_mesh = [mesh]

    assert mesh.dim >= 2

    # check list length
    Ndisplacement = len(list_displacementMatrix)
    Nvalues = len(list_nodesValues_n)
    if Ndisplacement > 0 and Nvalues > 0:
        assert Ndisplacement == Nvalues, (
            f"list_displacementMatrix and list_nodesValues_n must have the same length. "
            f"Got {Ndisplacement} and {Nvalues}"
        )
        if updatedMesh:
            assert Ndisplacement == len(list_mesh)
    Niter = int(np.max([Ndisplacement, Nvalues, 1]))

    # get list of nodes values
    if Nvalues > 0:
        list_nodesValues = _get_list_nodesValues(list_nodesValues_n)
        if updatedMesh:
            vMin = np.min([np.min(nodeValues) for nodeValues in list_nodesValues])
            vMax = np.max([np.max(nodeValues) for nodeValues in list_nodesValues])
        else:
            vMin, vMax = np.min(list_nodesValues), np.max(list_nodesValues)
        Display._Save_colorbar(
            vMin=vMin,
            vMax=vMax,
            folder=folder,
            filename=f"colorbar_{filename}",
            cmap=cmap,
        )

    # init stage
    usdaFile = Folder.Join(folder, f"{filename}.usda")
    stage = Usd.Stage.CreateNew(usdaFile)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(Ndisplacement - 1 if Ndisplacement > 0 else 0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, unit)

    for i in range(Niter):

        if updatedMesh or i == 0:
            # get triangles connectivity
            triangles = np.concatenate(
                [
                    groupElem.connect[:, groupElem.triangles].reshape(-1, 3)
                    for groupElem in mesh.Get_list_groupElem(2)
                ],
                axis=0,
            )

            # get default colors
            defaultColors = [Gf.Vec3f(0.5, 0.5, 0.5)] * mesh.Nn

            if plotMesh:
                # get list_lines
                list_lines: list[np.ndarray] = []
                for groupElem in mesh.Get_list_groupElem(2):
                    segments = groupElem.segments
                    if segments.shape[1] > 2:
                        repeats = [2] * segments.shape[1]
                        repeats[0] = 1
                        repeats[-1] = 1
                        segments = np.repeat(segments, repeats, axis=1)
                    lines = groupElem.connect[:, segments].reshape(-1, 2)
                    list_lines.append(lines)

                # get lines
                lines = np.concatenate(list_lines, axis=0)

                # get default colors
                blackColors = [Gf.Vec3f(0.0, 0.0, 0.0)] * lines.shape[0]

        # create mesh
        xform = UsdGeom.Xform.Define(stage, f"/Frame_{i:03d}")
        mesh_prim = UsdGeom.Mesh.Define(stage, f"/Frame_{i:03d}/Mesh")
        mesh_prim.CreateFaceVertexCountsAttr([3] * triangles.shape[0])
        mesh_prim.CreateFaceVertexIndicesAttr(triangles.ravel().tolist())
        mesh_prim.CreateDoubleSidedAttr(True)
        if plotMesh:
            line_prim = UsdGeom.BasisCurves.Define(stage, f"/Frame_{i:03d}/Line")
            line_prim.CreateTypeAttr(UsdGeom.Tokens.linear)
            line_prim.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
            line_prim.CreateCurveVertexCountsAttr([2] * lines.shape[0])

        # set mesh coordinates
        coords = mesh.coord
        if Ndisplacement > 0:
            coords += list_displacementMatrix[i]
        list_point = [Gf.Vec3f(x, y, z) for x, y, z in zip(*coords.T)]
        mesh_prim.CreatePointsAttr().Set(list_point)
        if plotMesh:
            line_prim.CreatePointsAttr().Set(list_point)

        # set colors
        if Nvalues > 0:
            colors = Display._Get_colors_for_values(
                list_nodesValues[i], vMax=vMax, vMin=vMin, cmap=cmap
            )
            list_color = [Gf.Vec3f(*color) for color in colors]
        else:
            list_color = defaultColors
        mesh_prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(list_color)
        if plotMesh:
            lines_coords = coords[lines].reshape(-1, 3)
            list_point = [Gf.Vec3f(*coords) for coords in lines_coords]
            line_prim.CreatePointsAttr().Set(list_point)
            line_prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(blackColors)

        if Niter > 1:
            # Tips for forcing stepwise interpolation and avoiding clipping!
            scale_op = xform.AddScaleOp()
            # For each frame of the animation
            for f in range(Ndisplacement):
                if f == i:  # visible
                    scale_op.Set(Gf.Vec3f(1, 1, 1), f)
                    scale_op.Set(Gf.Vec3f(1, 1, 1), f + 0.99999)
                else:  # invisible
                    scale_op.Set(Gf.Vec3f(0, 0, 0), f)
                    if f < Ndisplacement - 1:
                        scale_op.Set(Gf.Vec3f(0, 0, 0), f + 0.99999)

    stage.GetRootLayer().Save()

    # define usdz file
    usdzFile = Folder.Join(folder, f"{filename}.usdz")
    UsdUtils.CreateNewUsdzPackage(usdaFile, usdzFile)

    return usdaFile
