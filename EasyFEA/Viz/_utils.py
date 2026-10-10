# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Helpers shared by the viewers."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

if TYPE_CHECKING:
    from ..FEM._mesh import Mesh
    from ..Geoms._geom import _Geom
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
    """An ``(Ne, nPg)`` array, or ``{_GroupElem: scalar | (Ne_g,) | (Ne_g, nPg)}`` flattened in ``Get_list_groupElem(dim)`` order, as ``(Ne,)`` with the Gauss points averaged; any other ``result`` as is."""
    if isinstance(result, np.ndarray) and result.ndim == 2 and len(result) == mesh.Ne:
        return result.mean(1)
    if not isinstance(result, dict):
        return result
    values = []
    for groupElem in mesh.Get_list_groupElem(mesh.dim):
        if groupElem not in result:
            raise KeyError(f"no value given for the {groupElem.elemType} group")
        value = np.asarray(result[groupElem], dtype=float)
        if value.ndim == 2:
            value = value.mean(1)
        values.append(np.broadcast_to(value, (groupElem.Ne,)))
    return np.concatenate(values)
