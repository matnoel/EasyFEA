# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Helpers shared by the viewers."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from scipy.sparse import csr_matrix

from ..FEM._utils import MatrixType

if TYPE_CHECKING:
    from ..FEM._boundary_conditions import BoundaryCondition
    from ..FEM._group_elem import _GroupElem
    from ..FEM._mesh import Mesh
    from ..Geoms._geom import _Geom
    from ..Simulations._simu import _Simu
    from ..Utilities import _types

# fmt: off
# [colors.rgb2hex(color) for color in plt.get_cmap("tab10").colors]
tab10_colors = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
]
# [colors.rgb2hex(color) for color in plt.get_cmap("tab20").colors]
tab20_colors = [
    "#1f77b4","#aec7e8","#ff7f0e","#ffbb78","#2ca02c",
    "#98df8a","#d62728","#ff9896","#9467bd","#c5b0d5",
    "#8c564b","#c49c94","#e377c2","#f7b6d2","#7f7f7f",
    "#c7c7c7","#bcbd22","#dbdb8d","#17becf","#9edae5",
]
# fmt: on


class PlotOptions(TypedDict, total=False):
    """The options a `Movie_simu` forwards to `Plot`."""

    cmap: str
    nColors: int
    clim: tuple[float, float] | None
    colorbarTitle: str | None
    plotColorbar: bool
    verticalColorbar: bool
    color: str | None
    edgecolor: str
    linewidth: float | None
    alpha: float
    plotMesh: bool
    plotNodes: bool
    nodeSize: float | None
    label: str | None
    showGrid: bool
    bounds: _types.Numbers | None


def _Get_bc_vectors(
    simu: _Simu, bc: BoundaryCondition, nodes: _types.IntArray
) -> tuple[_types.FloatArray, _types.FloatArray]:
    """Translation and rotation (len(nodes), 3) of ``bc`` on ``nodes``, its values summed per DOF, each scaled by its largest norm."""
    problemType = bc.problemType
    availableUnknowns = simu.Get_unknowns(problemType)
    nDof = simu._Get_FunctionSpace(problemType).Nn * simu.Get_dof_n(problemType)
    summedValues = csr_matrix(
        (bc.dofsValues, (bc.dofs, np.zeros_like(bc.dofs))), (nDof, 1)
    ).toarray()

    unknowns_rot = ["rx", "ry", "rz"]
    vector = np.zeros((len(nodes), 3))
    vectorRot = np.zeros((len(nodes), 3))
    for direction in bc.unknowns:
        lines = simu.Bc_dofs_nodes(nodes, [direction], problemType)
        values = np.ravel(summedValues[lines])
        if direction in unknowns_rot:
            vectorRot[:, unknowns_rot.index(direction)] = values
        else:
            vector[:, availableUnknowns.index(direction)] = values

    for array in (vector, vectorRot):
        norm = np.linalg.norm(array, axis=1).max()
        if norm > 0:
            array /= norm
    return vector, vectorRot


def _Get_outward_face_normals(
    faces: _GroupElem, mainGroups: list[_GroupElem], coord: _types.FloatArray
) -> _types.FloatArray:
    """Normals (Ne, 3) of ``faces``, whose sign is ambiguous, turned away from the element holding each, ``coord`` indexed by node id; zero on a face two elements hold."""
    normal_e = faces.Get_normals_e_pg(MatrixType.mass).mean(1)
    center_e = coord[faces.connect].mean(1)
    nVertex, Ne = faces.Nvertex, faces.Ne

    sign_e = np.zeros(Ne)
    nHolder_e = np.zeros(Ne, dtype=int)
    for group in mainGroups:
        # number of the face's vertices each element holds (Ne, group.Ne)
        rows = np.repeat(np.arange(Ne), nVertex)
        columns = faces.connect[:, :nVertex].ravel()
        faces_n = csr_matrix((np.ones(rows.size), (rows, columns)), (Ne, group.Ncoords))
        shared = (faces_n @ group.Get_connect_n_e()).tocoo()
        isHolder = shared.data == nVertex
        rows, elements = shared.row[isHolder], shared.col[isHolder]

        inside = center_e[rows] - coord[group.connect[elements]].mean(1)
        sign_e[rows] = np.sign(np.sum(normal_e[rows] * inside, axis=1))
        nHolder_e += np.bincount(rows, minlength=Ne)
    sign_e[nHolder_e != 1] = 0

    return normal_e * sign_e[:, None]


def _Get_arrow_tails(
    simu: _Simu,
    bc: BoundaryCondition,
    nodes: _types.IntArray,
    points: _types.FloatArray,
    arrows: _types.FloatArray,
) -> _types.FloatArray:
    """Tails of ``arrows`` (len(nodes), dim) on ``points``: on the surface for an arrow pointing out of the material, so that it stays outside, else so that it ends there."""
    tails = points - arrows
    dim = simu.mesh.dim
    if dim < 2:
        return tails

    space = simu._Get_FunctionSpace(bc.problemType)
    coord = space.coord
    isSelected = np.zeros(coord.shape[0], dtype=bool)
    isSelected[nodes] = True
    # normals by node: from the faces the selection holds, so that a corner of a side keeps the side's; from all the faces around for a point load
    selected, around = np.zeros((2, isSelected.size, 3))
    for faces in space.Get_list_groupElem(dim - 1):
        normal_e = _Get_outward_face_normals(
            faces, space.Get_list_groupElem(dim), coord
        )
        isHeld = isSelected[faces.connect[:, : faces.Nvertex]].all(axis=1)
        np.add.at(around, faces.connect, normal_e[:, None])
        np.add.at(selected, faces.connect[isHeld], normal_e[isHeld, None])

    normals = selected[nodes]
    hasNoFace = ~normals.any(axis=1)
    normals[hasNoFace] = around[nodes][hasNoFace]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms > 0)

    lengths = np.linalg.norm(arrows, axis=1)
    cosines = np.sum(normals[:, : arrows.shape[1]] * arrows, axis=1)
    isOutward = cosines > 1e-6 * lengths
    tails[isOutward] = points[isOutward]
    return tails


def _Movie_fps(filename: str, fps: int | None) -> int:
    """`fps`, or the default: 10 for a gif, 24 for a movie."""
    if fps is not None:
        return fps
    return 10 if filename.endswith(".gif") else 24


def _Flatten_geoms(geoms: tuple) -> list[_Geom]:
    """Unpacks the lists among `geoms`."""
    return [g for geom in geoms for g in (geom if isinstance(geom, list) else [geom])]


def _Check_kwargs(kwargs: dict[str, Any], aliases: dict[str, tuple[str, ...]]) -> None:
    """Raises ValueError when `kwargs` holds a backend spelling of one of our options."""
    for ours, theirs in aliases.items():
        for name in theirs:
            if name in kwargs:
                raise ValueError(f"'{name}' is set through '{ours}' in EasyFEA.")


def _Union_bounds(
    drawn: _types.FloatArray | None, coord: _types.FloatArray
) -> _types.FloatArray:
    """(2, 3) box holding `drawn` and the points `coord`."""
    box = np.array([coord.min(0), coord.max(0)], dtype=float)
    if drawn is not None:
        box = np.array([np.minimum(drawn[0], box[0]), np.maximum(drawn[1], box[1])])
    return box


def _View_box(bounds: _types.Numbers) -> _types.FloatArray:
    """`bounds` as (3, 2) limits; a 2D box (xmin, xmax, ymin, ymax) gets z in (0, 0)."""
    lims = np.asarray(bounds, dtype=float).ravel()
    if lims.size == 4:
        lims = np.append(lims, [0.0, 0.0])
    if lims.size != 6:
        raise ValueError("bounds must be (xmin, xmax, ymin, ymax[, zmin, zmax]).")
    return lims.reshape(3, 2)


def _Gauss_points_averaged(
    mesh: Mesh, result: str | _types.AnyArray | dict | None
) -> str | _types.AnyArray | None:
    """An ``(Ne, nPg)`` array, or ``{_GroupElem: scalar | (Ne_g,) | (Ne_g, nPg)}``, keyed by the mesh's groups or a field's, flattened in ``Get_list_groupElem(dim)`` order, as ``(Ne,)`` with the Gauss points averaged; any other ``result`` as is."""
    if isinstance(result, np.ndarray) and result.ndim == 2 and len(result) == mesh.Ne:
        return result.mean(1)
    if not isinstance(result, dict):
        return result
    values = []
    for groups in zip(*mesh._Get_lists_groupElem(mesh.dim)):
        groupElem = groups[0]
        key = next((group for group in groups if group in result), None)
        if key is None:
            raise KeyError(f"no value given for the {groupElem.elemType} group")
        value = np.asarray(result[key], dtype=float)
        if value.ndim == 2:
            value = value.mean(1)
        values.append(np.broadcast_to(value, (groupElem.Ne,)))
    return np.concatenate(values)
