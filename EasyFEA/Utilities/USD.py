# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module providing an interface with Universal Scene Description Format (USD) using usd-core (https://pypi.org/project/usd-core/)."""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ._requires import Create_requires_decorator
from ._mpi import rank0_only
from ..Simulations._simu import _Init_obj
from ..Utilities.MeshIO import Surface_reconstruction

from . import Folder, Display

if TYPE_CHECKING:
    from ..FEM._mesh import Mesh
    from ..Simulations import _Simu

try:
    from pxr import Usd, UsdGeom, Gf, UsdUtils, Vt
except ImportError:
    pass
requires_pxr = Create_requires_decorator("pxr", libraries=["usd-core"])

from .GLTF import _get_list_nodesValues


@rank0_only
@requires_pxr
def Save_simu(
    simu: "_Simu",
    results: list[str],
    folder: str,
    N: int = 50,
    deformFactor=1.0,
    plotMesh=False,
    cmap: str = "jet",
    fps: int = 30,
    unit: float = 1.0,
    smoothAnimation: bool = True,
) -> None:
    """Saves the simulation's results as usdz files.

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
        If True, wrong camera zoom in Keynote.
        If False, good camera zoom in Keynote.
        Default False
    cmap: str, optional
        the color map used near the figure, by default "jet" \n
        ["jet", "seismic", "binary", "viridis"] -> https://matplotlib.org/stable/tutorials/colors/colormaps.html
    fps : int, optional
        Frames per second, by default 30
    unit: float, optional
        Meters per unit, by default 1.0
    smoothAnimation: bool, optional
        If True, smooth interpolation on Preview but no animation on Keynote.
        If False, frame-by-frame animation on Previewer and Keynote.
        Default True.
    """

    simu, mesh, _, _ = _Init_obj(simu)  # type: ignore [assignment]

    updatedMesh = simu.Nmesh > 1
    reconstructSurface = len(mesh.Get_list_groupElem(2)) == 0
    if not updatedMesh:
        list_mesh = Surface_reconstruction(mesh) if reconstructSurface else mesh

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
        if updatedMesh:
            list_mesh: list[Mesh] = [None] * N

        # activates the first iteration
        simu.Set_Iter(0, resetAll=True)

        # get values
        for i, iter in enumerate(iterations):
            simu.Set_Iter(iter)
            if updatedMesh:
                list_mesh[i] = (
                    Surface_reconstruction(simu.mesh)
                    if reconstructSurface
                    else simu.mesh
                )
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
                cmap=cmap,
                fps=fps,
                unit=unit,
                smoothAnimation=smoothAnimation,
            )

        if dof_n > 1:
            Save_mesh(
                mesh=list_mesh,
                folder=folder,
                filename=f"{result}_norm",
                list_displacementMatrix=list_displacementMatrix,
                list_nodesValues_n=list_nodesValues_n,
                plotMesh=plotMesh,
                cmap=cmap,
                fps=fps,
                unit=unit,
                smoothAnimation=smoothAnimation,
            )


def _get_triangles(mesh: "Mesh") -> np.ndarray:
    return np.concatenate(
        [
            groupElem.connect[:, groupElem.triangles].reshape(-1, 3)
            for groupElem in mesh.Get_list_groupElem(2)
        ],
        axis=0,
    )


def _get_lines(mesh: "Mesh") -> np.ndarray:
    list_lines = []
    for groupElem in mesh.Get_list_groupElem(2):
        segments = groupElem.segments
        if segments.shape[1] > 2:
            repeats = [2] * segments.shape[1]
            repeats[0] = 1
            repeats[-1] = 1
            segments = np.repeat(segments, repeats, axis=1)
        list_lines.append(groupElem.connect[:, segments].reshape(-1, 2))
    return np.concatenate(list_lines, axis=0)


@rank0_only
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
    smoothAnimation: bool = True,
) -> str:
    """Saves the mesh as usdz file.

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
    list_nodesValues_n : list[np.ndarray], optional
        List of node values for colors, by default []
    plotMesh : bool, optional
        If True, wrong camera zoom in Keynote.
        If False, good camera zoom in Keynote.
        Default False
    cmap: str, optional
        the color map used near the figure, by default "jet" \n
        ["jet", "seismic", "binary", "viridis"] -> https://matplotlib.org/stable/tutorials/colors/colormaps.htmlcmap: str, optional
    fps : int, optional
        Frames per second, by default 30
    unit: float, optional
        Meters per unit, by default 1.0
    smoothAnimation: bool, optional
        If True, smooth interpolation in Preview but no animation in Keynote.
        If False, frame-by-frame animation in Preview and Keynote.
        Default True.

    Returns
    -------
    str
        The path to the created usdz file.
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
            vMin = np.min([np.min(v) for v in list_nodesValues])
            vMax = np.max([np.max(v) for v in list_nodesValues])
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
    usdcFile = Folder.Join(folder, f"{filename}.usdc")
    stage = Usd.Stage.CreateNew(usdcFile)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(Niter - 1 if Niter > 1 else 0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, unit)

    # get initial topology
    triangles = _get_triangles(mesh)
    defaultColors = np.full((mesh.Nn, 3), 0.5, dtype=np.float32)
    if plotMesh:
        lines = _get_lines(mesh)

    # ========== SMOOTH ANIMATION MODE ==========
    if smoothAnimation:

        meshPrim = UsdGeom.Mesh.Define(stage, "/Mesh")
        meshPrim.CreateDoubleSidedAttr(True)

        # create time-sampled attributes for triangles
        faceCount_attr = meshPrim.CreateFaceVertexCountsAttr()
        faceIndex_attr = meshPrim.CreateFaceVertexIndicesAttr()
        points_attr = meshPrim.CreatePointsAttr()
        color_primvar = meshPrim.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)

        # and for lines
        if plotMesh:
            linePrim = UsdGeom.BasisCurves.Define(stage, "/Lines")
            linePrim.CreateTypeAttr(UsdGeom.Tokens.linear)
            linePrim.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
            # updatable attributes
            curveCount_attr = linePrim.CreateCurveVertexCountsAttr()
            linePoints_attr = linePrim.CreatePointsAttr()
            lineColor_primvar = linePrim.CreateDisplayColorPrimvar(
                UsdGeom.Tokens.uniform
            )
        else:
            meshPrim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

        # set static topology if not updated
        if not updatedMesh:
            faceCount_attr.Set(
                Vt.IntArray.FromNumpy(np.full(triangles.shape[0], 3, dtype=np.int32))
            )
            faceIndex_attr.Set(
                Vt.IntArray.FromNumpy(triangles.ravel().astype(np.int32))
            )
            if plotMesh:
                curveCount_attr.Set(
                    Vt.IntArray.FromNumpy(np.full(lines.shape[0], 2, dtype=np.int32))
                )
                lineColor_primvar.Set(
                    Vt.Vec3fArray.FromNumpy(
                        np.zeros((lines.shape[0], 3), dtype=np.float32)
                    )
                )

        for i in range(Niter):
            timecode = i if Niter > 1 else Usd.TimeCode.Default()

            if updatedMesh:
                mesh = list_mesh[i]
                # update topology
                triangles = _get_triangles(mesh)
                faceCount_attr.Set(
                    Vt.IntArray.FromNumpy(
                        np.full(triangles.shape[0], 3, dtype=np.int32)
                    ),
                    timecode,
                )
                faceIndex_attr.Set(
                    Vt.IntArray.FromNumpy(triangles.ravel().astype(np.int32)), timecode
                )
                if plotMesh:
                    lines = _get_lines(mesh)
                    curveCount_attr.Set(
                        Vt.IntArray.FromNumpy(
                            np.full(lines.shape[0], 2, dtype=np.int32)
                        ),
                        timecode,
                    )
            else:
                mesh = mesh

            # update coordinates
            coords = mesh.coord.astype(np.float32).copy()
            if Ndisplacement > 0:
                coords += list_displacementMatrix[i].astype(np.float32)
            points_attr.Set(Vt.Vec3fArray.FromNumpy(coords), timecode)

            if plotMesh:
                linePoints_attr.Set(
                    Vt.Vec3fArray.FromNumpy(coords[lines].reshape(-1, 3)), timecode
                )
                if updatedMesh:
                    lineColor_primvar.Set(
                        Vt.Vec3fArray.FromNumpy(
                            np.zeros((lines.shape[0], 3), dtype=np.float32)
                        ),
                        timecode,
                    )

            # colors
            if Nvalues > 0:
                colors = Display._Get_colors_for_values(
                    list_nodesValues[i], vMax=vMax, vMin=vMin, cmap=cmap
                ).astype(np.float32)
            else:
                colors = np.full((mesh.Nn, 3), 0.5, dtype=np.float32)

            color_primvar.Set(Vt.Vec3fArray.FromNumpy(colors), timecode)

    # ========== SLIDESHOW ANIMATION MODE ==========
    else:

        for i in range(Niter):

            if updatedMesh:
                mesh = list_mesh[i]
                triangles = _get_triangles(mesh)
                defaultColors = np.full((mesh.Nn, 3), 0.5, dtype=np.float32)
                if plotMesh:
                    lines = _get_lines(mesh)
            else:
                mesh = mesh

            # coordinates
            coords = mesh.coord.astype(np.float32)
            if Ndisplacement > 0:
                coords += list_displacementMatrix[i].astype(np.float32)

            # create frame
            xform = UsdGeom.Xform.Define(stage, f"/Frame_{i:03d}")

            meshPrim = UsdGeom.Mesh.Define(stage, f"/Frame_{i:03d}/Mesh")
            meshPrim.CreateDoubleSidedAttr(True)
            meshPrim.CreateFaceVertexCountsAttr(
                Vt.IntArray.FromNumpy(np.full(triangles.shape[0], 3, dtype=np.int32))
            )
            meshPrim.CreateFaceVertexIndicesAttr(
                Vt.IntArray.FromNumpy(triangles.ravel().astype(np.int32))
            )
            meshPrim.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(coords))

            if plotMesh:
                linePrim = UsdGeom.BasisCurves.Define(stage, f"/Frame_{i:03d}/Lines")
                linePrim.CreateTypeAttr(UsdGeom.Tokens.linear)
                linePrim.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
                linePrim.CreateCurveVertexCountsAttr(
                    Vt.IntArray.FromNumpy(np.full(lines.shape[0], 2, dtype=np.int32))
                )
                line_coords = coords[lines].reshape(-1, 3)
                linePrim.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(line_coords))
                linePrim.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(
                    Vt.Vec3fArray.FromNumpy(
                        np.zeros((lines.shape[0], 3), dtype=np.float32)
                    )
                )
            else:
                meshPrim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

            # colors
            if Nvalues > 0:
                colors = Display._Get_colors_for_values(
                    list_nodesValues[i], vMax=vMax, vMin=vMin, cmap=cmap
                ).astype(np.float32)
            else:
                colors = defaultColors

            meshPrim.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(
                Vt.Vec3fArray.FromNumpy(colors)
            )

            # animate visibility via scale
            if Niter > 1:
                scale_op = xform.AddScaleOp()
                for f in range(Niter):
                    if f == i:
                        scale_op.Set(Gf.Vec3f(1, 1, 1), f)
                        scale_op.Set(Gf.Vec3f(1, 1, 1), f + 0.99999)
                    else:
                        scale_op.Set(Gf.Vec3f(0, 0, 0), f)
                        if f < Niter - 1:
                            scale_op.Set(Gf.Vec3f(0, 0, 0), f + 0.99999)

    stage.GetRootLayer().Save()

    usdzFile = Folder.Join(folder, f"{filename}.usdz")
    UsdUtils.CreateNewUsdzPackage(usdcFile, usdzFile)

    return usdzFile
