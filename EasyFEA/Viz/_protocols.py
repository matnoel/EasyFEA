# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""What a viewer module can do, with each backend's defaults (`...`); mypy checks the claims at the bottom."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from typing import Unpack
    from ..Utilities import _types
    from ._utils import PlotOptions


class Viewer(Protocol):
    def Plot(
        self,
        obj: Any,
        result: str | _types.AnyArray | dict | None = ...,
        deformFactor: float = ...,
        coef: float = ...,
        nodeValues: bool = ...,
        *,
        cmap: str = ...,
        nColors: int = ...,
        clim: tuple[float, float] | None = ...,
        colorbarTitle: str | None = ...,
        plotColorbar: bool = ...,
        verticalColorbar: bool = ...,
        color: str | None = ...,
        edgecolor: str = ...,
        linewidth: float | None = ...,
        alpha: float = ...,
        plotMesh: bool = ...,
        plotNodes: bool = ...,
        nodeSize: float | None = ...,
        title: str = ...,
        label: str | None = ...,
        showGrid: bool = ...,
        bounds: _types.Numbers | None = ...,
        **kwargs: Any,
    ) -> Any:
        """Draws `obj`, colored by `result` when given; `kwargs` go to the backend's draw call."""
        ...

    def Plot_Mesh(
        self,
        obj: Any,
        deformFactor: float = ...,
        *,
        color: str | None = ...,
        edgecolor: str = ...,
        linewidth: float | None = ...,
        alpha: float = ...,
        plotMesh: bool = ...,
        plotNodes: bool = ...,
        nodeSize: float | None = ...,
        title: str = ...,
        label: str | None = ...,
        showGrid: bool = ...,
        bounds: _types.Numbers | None = ...,
    ) -> Any:
        """Draws the mesh."""
        ...

    def Plot_Nodes(
        self,
        obj: Any,
        nodes: _types.IntArray | None = ...,
        showId: bool = ...,
        *,
        deformFactor: float = ...,
        color: str | None = ...,
        alpha: float = ...,
        nodeSize: float | None = ...,
        title: str = ...,
        label: str | None = ...,
        showGrid: bool = ...,
        bounds: _types.Numbers | None = ...,
    ) -> Any:
        """Draws the nodes, all of them by default."""
        ...

    def Plot_Elements(
        self,
        obj: Any,
        nodes: _types.IntArray | None = ...,
        dimElem: int | None = ...,
        showId: bool = ...,
        *,
        deformFactor: float = ...,
        color: str | None = ...,
        edgecolor: str = ...,
        linewidth: float | None = ...,
        alpha: float = ...,
        plotMesh: bool = ...,
        plotNodes: bool = ...,
        nodeSize: float | None = ...,
        title: str = ...,
        label: str | None = ...,
        showGrid: bool = ...,
        bounds: _types.Numbers | None = ...,
    ) -> Any:
        """Draws the elements using `nodes`, all of them by default."""
        ...

    def Plot_BoundaryConditions(
        self,
        simu: Any,
        *,
        deformFactor: float = ...,
        alpha: float = ...,
        nodeSize: float | None = ...,
        title: str = ...,
        showGrid: bool = ...,
        plotLegend: bool = ...,
        bounds: _types.Numbers | None = ...,
    ) -> Any:
        """Draws the boundary conditions over the mesh, `alpha` being the mesh's."""
        ...

    def Plot_Tags(
        self,
        obj: Any,
        *,
        deformFactor: float = ...,
        alpha: float = ...,
        linewidth: float | None = ...,
        title: str = ...,
        showId: bool = ...,
        showGrid: bool = ...,
        plotLegend: bool = ...,
        useColorCycler: bool = ...,
        bounds: _types.Numbers | None = ...,
    ) -> Any:
        """Draws the tagged elements (up to 2D), `showId` writing the tags."""
        ...

    def Plot_Geoms(
        self,
        *geoms: Any,
        color: str | None = ...,
        linewidth: float | None = ...,
        alpha: float = ...,
        title: str = ...,
        label: str | None = ...,
        showGrid: bool = ...,
        plotLegend: bool = ...,
        bounds: _types.Numbers | None = ...,
    ) -> Any:
        """Draws geoms, or lists of them; `label` replaces each geom's name."""
        ...

    def Movie_simu(
        self,
        simu: Any,
        result: str,
        folder: str,
        filename: str = ...,
        N: int = ...,
        deformFactor: float = ...,
        coef: float = ...,
        nodeValues: bool = ...,
        *,
        fps: int = ...,
        **kwargs: Unpack[PlotOptions],
    ) -> None:
        """Movie of `result` over at most `N` saved iterations."""
        ...

    def Movie_func(
        self,
        func: Callable[[Any, int], None],
        N: int,
        folder: str,
        filename: str = ...,
        *,
        fps: int = ...,
    ) -> None:
        """Movie of `func(scene, i)` for `i` in `range(N)`."""
        ...


if TYPE_CHECKING:
    # here, outside the Viz import cycle, where mypy knows the decorated functions
    from . import Matplotlib, PyVista

    _matplotlib_viewer: Viewer = Matplotlib
    _pyvista_viewer: Viewer = PyVista
