# Copyright (C) 2021-2025 Université Gustave Eiffel.
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
    from pxr import Usd, UsdGeom, Gf
except ImportError:
    pass
requires_pxr = Create_requires_decorator("pxr", libraries=["usd-core"])

from .GLTF import _get_list_nodesValues, _get_colors_for_values


@requires_pxr
def Save_simu(
    simu: "_Simu",
    result: str,
    folder: str,
    filename: str = None,
    N: int = 50,
    deformFactor=1.0,
    clim=None,
) -> None:
    """Saves the simulation as usda file.

    Parameters
    ----------
    simu : _Simu
        simulation
    result : str
        result that you want to plot
    folder : str
        folder where you want to save the video
    filename : str, optional
        filename of the usda file, by default 'result.usda'
    N : int, optional
        Maximal number of iterations displayed, by default 50
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0

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

    # activates the first iteration
    simu.Set_Iter(0, resetAll=True)

    # init list
    list_displacementMatrix: list[np.ndarray] = [None] * N
    list_nodesValues_n: list[np.ndarray] = [None] * N

    # get values
    for i, iter in enumerate(iterations):
        simu.Set_Iter(iter)
        list_displacementMatrix[i] = deformFactor * simu.Results_displacement_matrix()
        nodesValues_n = simu.Result(result, nodeValues=True)
        list_nodesValues_n[i] = (
            nodesValues_n if clim is None else np.clip(nodesValues_n, *clim)
        )

    if filename is None:
        filename = result

    return Save_mesh(
        mesh=mesh,
        folder=folder,
        filename=filename,
        list_displacementMatrix=list_displacementMatrix,
        list_nodesValues_n=list_nodesValues_n,
    )


@requires_pxr
def Save_mesh(
    mesh: "Mesh",
    folder: str,
    filename: str = "mesh",
    list_displacementMatrix: list[np.ndarray] = [],
    list_nodesValues_n: list[np.ndarray] = [],
    fps: int = 30,
) -> str:
    assert mesh.dim >= 2

    coord0 = mesh.coord
    triangles = np.concatenate(
        [
            groupElem.connect[:, groupElem.triangles].reshape(-1, 3)
            for groupElem in mesh.Get_list_groupElem(2)
        ],
        axis=0,
    )

    # get list of nodes values
    Nvalues = len(list_nodesValues_n)
    if Nvalues > 0:
        list_nodesValues = _get_list_nodesValues(list_nodesValues_n)
        vMax, vMin = np.max(list_nodesValues), np.min(list_nodesValues)

    Ndisplacement = len(list_displacementMatrix)

    # init stage
    usdaFile = Folder.Join(folder, f"{filename}.usda")
    stage = Usd.Stage.CreateNew(usdaFile)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(Ndisplacement - 1 if Ndisplacement > 0 else 0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    if Ndisplacement == 0:
        mesh_prim = UsdGeom.Mesh.Define(stage, "/Mesh")
        mesh_prim.CreateFaceVertexCountsAttr([3] * triangles.shape[0])
        mesh_prim.CreateFaceVertexIndicesAttr(triangles.ravel().tolist())
        mesh_prim.CreateDoubleSidedAttr(True)
        list_point = [Gf.Vec3f(x, y, z) for x, y, z in zip(*coord0.T)]
        mesh_prim.CreatePointsAttr().Set(list_point)
        list_color = [Gf.Vec3f(0.5, 0.5, 0.5)] * mesh.Nn
        mesh_prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(list_color)
    else:
        for i, displacement in enumerate(list_displacementMatrix):
            xform = UsdGeom.Xform.Define(stage, f"/Frame_{i:03d}")
            mesh_prim = UsdGeom.Mesh.Define(stage, f"/Frame_{i:03d}/Mesh")

            mesh_prim.CreateFaceVertexCountsAttr([3] * triangles.shape[0])
            mesh_prim.CreateFaceVertexIndicesAttr(triangles.ravel().tolist())
            mesh_prim.CreateDoubleSidedAttr(True)

            coords = coord0 + displacement
            list_point = [Gf.Vec3f(x, y, z) for x, y, z in zip(*coords.T)]
            mesh_prim.CreatePointsAttr().Set(list_point)

            if Nvalues > 0:
                nodesValues = list_nodesValues_n[i]
                colors = _get_colors_for_values(nodesValues, vMax, vMin)
                list_color = [Gf.Vec3f(*color) for color in colors]
            else:
                list_color = [Gf.Vec3f(0.5, 0.5, 0.5)] * mesh.Nn
            mesh_prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(list_color)

            scale_op = xform.AddScaleOp()

            # Tips for forcing stepwise interpolation and avoiding clipping!
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
    os.system(f"usdzip {usdzFile} {usdaFile}")

    return usdzFile
