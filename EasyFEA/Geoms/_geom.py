# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the _Geom class used to construct geometry. Geometry inherits from _Geom."""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import copy

from ._utils import Point, Rotate, Symmetry

from ..FEM._utils import ElemType
from typing import Union, Optional, Iterable, TYPE_CHECKING
from ..Utilities import _types, _params

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

if TYPE_CHECKING:
    from ..Geoms import Point, Line, Circle, CircleArc, Points, Contour, Domain

    GeomCompatible = Union["_Geom", Domain, Circle, Points, Contour]
    ContourCompatible = Union[Line, CircleArc, Points]
    CrackCompatible = Union[Line, Points, Contour, CircleArc]
    RefineCompatible = Union[Point, Circle, str]


class _Geom(ABC):
    """Geometric class."""

    meshSize: float = _params.PositiveScalarParameter()
    """Element size used for meshing."""

    name: str = _params.StringParameter()
    """Name of the geometric object."""

    isHollow: bool = _params.BoolParameter()
    """Indicates whether the formed geometry is hollow."""

    isOpen: bool = _params.BoolParameter()
    """Indicates whether the geometry is open, typically to model cracks."""

    def __init__(
        self,
        points: list[Point],
        meshSize: _types.Number,
        name: str,
        isHollow: bool,
        isOpen: bool,
    ):
        """Creates a geometric object.

        Parameters
        ----------
        points : list[Point]
            list of points to build the geometric object
        meshSize : _types.Number
            mesh size that will be used to create the mesh >= 0
        name : str
            object name
        isHollow : bool
            Indicates whether the formed geometry is hollow (i.e., contains voids).
        isOpen : bool
            Indicates whether the geometry is open, for instance to represent a crack.
        """

        assert isinstance(points, Iterable) and isinstance(
            points[0], Point
        ), "points must be a list of points."
        self.__points: list[Point] = points

        self.meshSize = meshSize
        self.name = name
        self.isHollow = isHollow
        self.isOpen = isOpen

    @staticmethod
    @abstractmethod
    def _Init_Ninstance():
        """Initializes the instance number."""
        ...

    # points doesn't have a public setter for safety
    @property
    def points(self) -> list[Point]:
        """The list of points used to build the geometric object."""
        return self.__points

    @property
    def coord(self) -> _types.FloatArray:
        """Returns the coordinates of all points as a NumPy array."""
        return np.asarray([p.coord for p in self.points], dtype=float)

    @abstractmethod
    def Get_coord_for_plot(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        """Returns lines and points coordinates for plotting.

        Returns
        -------
        tuple of ndarray
            Lines and points coordinates as NumPy arrays.
        """

        lines = self.coord
        points = lines[[0, -1]]
        return lines, points

    def copy(self):
        new = copy.deepcopy(self)
        new.name = new.name + "_copy"
        return new

    def Translate(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        """Translates the geometry in 3D space.

        Parameters
        ----------
        dx : float, optional
            Translation along the x-axis, by default 0.0.
        dy : float, optional
            Translation along the y-axis, by default 0.0.
        dz : float, optional
            Translation along the z-axis, by default 0.0.
        """

        # to translate an object, all you have to do is move these points
        [p.Translate(dx, dy, dz) for p in self.__points]  # type: ignore [func-returns-value]

    def Rotate(
        self, theta: float, center: tuple = (0, 0, 0), direction: tuple = (0, 0, 1)
    ) -> None:
        """Rotates the object coordinates around an axis.

        Parameters
        ----------
        theta : float
            rotation angle in [deg]
        center : tuple, optional
            rotation center, by default (0,0,0)
        direction : tuple, optional
            rotation direction, by default (0,0,1)
        """
        oldCoord = self.coord
        newCoord = Rotate(oldCoord, theta, center, direction)

        dec = newCoord - oldCoord
        [point.Translate(*dec[p]) for p, point in enumerate(self.points)]  # type: ignore [func-returns-value]

    def Symmetry(
        self, point: _types.Coords = (0, 0, 0), n: _types.Coords = (1, 0, 0)
    ) -> None:
        """Reflects the geometry with respect to a plane.

        Parameters
        ----------
        point : Coords, optional
            A point on the reflection plane, by default (0, 0, 0).
        n : Coords, optional
            Normal vector of the plane, by default (1, 0, 0).
        """

        oldCoord = self.coord
        newCoord = Symmetry(oldCoord, point, n)

        dec = newCoord - oldCoord
        [point.Translate(*dec[p]) for p, point in enumerate(self.points)]  # type: ignore [func-returns-value]

    def Mesh_2D(
        self,
        inclusions: list[GeomCompatible] = [],
        elemType: ElemType = ElemType.TRI3,
        cracks: list["CrackCompatible"] = [],
        refineGeoms: list["RefineCompatible"] = [],
        isOrganised=False,
        additionalSurfaces: list[GeomCompatible] = [],
        additionalLines: list[Union["Line", "CircleArc"]] = [],
        additionalPoints: list["Point"] = [],
        folder="",
    ):
        """Creates a 2D mesh from a contour and inclusions that must form a closed plane surface.

        Parameters
        ----------
        inclusions : list[Domain, Circle, Points, Contour], optional
            list of hollow and filled geom objects inside the domain
        elemType : ElemType, optional
            element type, by default "TRI3" ["TRI3", "TRI6", "TRI10", "TRI15", "QUAD4", "QUAD8", "QUAD9"]
        cracks : list[Line | Points | Contour | CircleArc]
            list of geom object used to create open or closed cracks
        refineGeoms : list[Domain|Circle|str], optional
            list of geom object for mesh refinement, by default []
        isOrganised : bool, optional
            mesh is organized, by default False
        additionalSurfaces : list[Domain, Circle, Points, Contour]
            additional surfaces that will be added to or removed from the surfaces created by the contour and the inclusions. (e.g Domain, Circle, Contour, Points). Tip: if the mesh is not well generated, you can also give the inclusions.
        additionalLines : list[Union[Line,CircleArc]]
            additional lines that will be added to the surfaces created by the contour and the inclusions. (e.g Domain, Circle, Contour, Points). WARNING: lines must be within the domain.
        additionalPoints : list[Point]
            additional points that will be added to the surfaces created by the contour and the inclusions. WARNING: points must be within the domain.
        folder : str, optional
            default mesh.msh folder, by default "" does not save the mesh

        Returns
        -------
        Mesh
            Created mesh
        """
        from ..FEM._mesher import Mesher

        mesher = Mesher()
        mesh = mesher.Mesh_2D(
            self,
            inclusions=inclusions,
            elemType=elemType,
            cracks=cracks,
            refineGeoms=refineGeoms,
            isOrganised=isOrganised,
            additionalSurfaces=additionalSurfaces,
            additionalLines=additionalLines,
            additionalPoints=additionalPoints,
            folder=folder,
        )
        return mesh

    def Mesh_Extrude(
        self,
        inclusions: list[GeomCompatible] = [],
        extrude: _types.Coords = (0, 0, 1),
        layers: list[int] = [],
        elemType: ElemType = ElemType.TETRA4,
        cracks: list["CrackCompatible"] = [],
        refineGeoms: list["RefineCompatible"] = [],
        isOrganised=False,
        additionalSurfaces: list[GeomCompatible] = [],
        additionalLines: list[Union["Line", "CircleArc"]] = [],
        additionalPoints: list["Point"] = [],
        folder="",
    ):
        """Creates a 3D mesh by extruding a surface constructed from a contour and inclusions.

        Parameters
        ----------
        inclusions : list[Domain, Circle, Points, Contour], optional
            list of hollow and filled geom objects inside the domain
        extrude : Coords, optional
            extrusion vector, by default [0,0,1]
        layers: list[int], optional
            layers in the extrusion, by default []
        elemType : ElemType, optional
            element type, by default "TETRA4" ["TETRA4", "TETRA10", "HEXA8", "HEXA20", "HEXA27", "PRISM6", "PRISM15", "PRISM18"]
        cracks : list[Line | Points | Contour | CircleArc]
            list of geom object used to create open or closed cracks
        refineGeoms : list[Domain|Circle|str], optional
            list of geom object for mesh refinement, by default []
        isOrganised : bool, optional
            mesh is organized, by default False
        additionalSurfaces : list[Domain, Circle, Points, Contour]
            additional surfaces that will be added to or removed from the surfaces created by the contour and the inclusions. (e.g Domain, Circle, Contour, Points). Tip: if the mesh is not well generated, you can also give the inclusions.
        additionalLines : list[Union[Line,CircleArc]]
            additional lines that will be added to the surfaces created by the contour and the inclusions. (e.g Domain, Circle, Contour, Points). WARNING: lines must be within the domain.
        additionalPoints : list[Point]
            additional points that will be added to the surfaces created by the contour and the inclusions. WARNING: points must be within the domain.
        folder : str, optional
            default mesh.msh folder, by default "" does not save the mesh

        Returns
        -------
        Mesh
            Created mesh
        """
        from ..FEM._mesher import Mesher

        mesher = Mesher()
        mesh = mesher.Mesh_Extrude(
            self,
            inclusions=inclusions,
            extrude=extrude,
            layers=layers,
            elemType=elemType,
            cracks=cracks,
            refineGeoms=refineGeoms,
            isOrganised=isOrganised,
            additionalSurfaces=additionalSurfaces,
            additionalLines=additionalLines,
            additionalPoints=additionalPoints,
            folder=folder,
        )
        return mesh

    def Mesh_Revolve(
        self,
        inclusions: list[GeomCompatible] = [],
        axis: Optional["Line"] = None,
        angle=360,
        layers: list[int] = [30],
        elemType: ElemType = ElemType.TETRA4,
        cracks: list["CrackCompatible"] = [],
        refineGeoms: list["RefineCompatible"] = [],
        isOrganised=False,
        additionalSurfaces: list[GeomCompatible] = [],
        additionalLines: list[Union["Line", "CircleArc"]] = [],
        additionalPoints: list["Point"] = [],
        folder="",
    ):
        """Creates a 3D mesh by rotating a surface along an axis.

        Parameters
        ----------
        inclusions : list[Domain, Circle, Points, Contour], optional
            list of hollow and filled geom objects inside the domain
        axis : Line, optional
            revolution axis, by default Line((0, 0), (0, 1))
        angle : float|int, optional
            revolution angle in [deg], by default 360
        layers: list[int], optional
            layers in extrusion, by default [30]
        elemType : ElemType, optional
            element type, by default "TETRA4" ["TETRA4", "TETRA10", "HEXA8", "HEXA20", "HEXA27", "PRISM6", "PRISM15", "PRISM18"]
        cracks : list[Line | Points | Contour | CircleArc]
            list of geom object used to create open or closed cracks
        refineGeoms : list[Domain|Circle|str], optional
            list of geom object for mesh refinement, by default []
        isOrganised : bool, optional
            mesh is organized, by default False
        additionalSurfaces : list[Domain, Circle, Points, Contour]
            additional surfaces that will be added to or removed from the surfaces created by the contour and the inclusions. (e.g Domain, Circle, Contour, Points). Tip: if the mesh is not well generated, you can also give the inclusions.
        additionalLines : list[Union[Line,CircleArc]]
            additional lines that will be added to the surfaces created by the contour and the inclusions. (e.g Domain, Circle, Contour, Points). WARNING: lines must be within the domain.
        additionalPoints : list[Point]
            additional points that will be added to the surfaces created by the contour and the inclusions. WARNING: points must be within the domain.
        folder : str, optional
            default mesh.msh folder, by default "" does not save the mesh

        Returns
        -------
        Mesh
            Created mesh
        """
        from ..FEM._mesher import Mesher

        if axis is None:
            from ..Geoms import Line

            axis = Line((0, 0), (0, 1))

        mesher = Mesher()
        mesh = mesher.Mesh_Revolve(
            self,
            inclusions=inclusions,
            axis=axis,
            angle=angle,
            layers=layers,
            elemType=elemType,
            cracks=cracks,
            refineGeoms=refineGeoms,
            isOrganised=isOrganised,
            additionalSurfaces=additionalSurfaces,
            additionalLines=additionalLines,
            additionalPoints=additionalPoints,
            folder=folder,
        )
        return mesh

    def Plot(
        self,
        ax: Optional[plt.Axes] = None,
        color: str = "",
        name: str = "",
        lw: Optional[_types.Number] = None,
        ls: Optional[str] = None,
        plotPoints: bool = True,
    ) -> plt.Axes:
        """Plots the geometry using Matplotlib.

        Parameters
        ----------
        ax : matplotlib axis, optional
            Axis to plot on. If None, a new one is created.
        color : str, optional
            Line color.
        name : str, optional
            Label for the object.
        lw : float, optional
            Line width.
        ls : str, optional
            Line style.
        plotPoints : bool, optional
            If True, display the object's defining points.

        Returns
        -------
        Axes
            The axis with the plotted geometry.
        """

        from ..Utilities.Display import Init_Axes, _Axis_equal_3D

        lines, points = self.Get_coord_for_plot()

        if ax is None:
            dimMax = 2 if np.abs(lines[:, 2].max()) == 0 else 3
            ax = Init_Axes(dimMax)
            ax.grid()

        inDim = 3 if ax.name == "3d" else 2

        name = self.name if name == "" else name

        if color != "":
            ax.plot(*lines[:, :inDim].T, color=color, label=name, lw=lw, ls=ls)
        else:
            ax.plot(*lines[:, :inDim].T, label=name, lw=lw, ls=ls)
        if plotPoints:
            ax.plot(*points[:, :inDim].T, ls="", marker=".", c="black")

        if inDim == 3:
            xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()  # type: ignore [union-attr]
            oldBounds = np.array([xlim, ylim, zlim]).T
            lines = np.concatenate((lines, oldBounds), 0)
            _Axis_equal_3D(ax, lines)  # type: ignore
        else:
            ax.axis("equal")

        return ax

    @staticmethod
    def Plot_Geoms(
        geoms: list["_Geom"],
        ax: Optional[plt.Axes] = None,
        color: str = "",
        name: str = "",
        plotPoints: bool = True,
        plotLegend: bool = True,
    ) -> plt.Axes:
        """Plots a list of geometric objects on the same axis.

        Parameters
        ----------
        geoms : list of _Geom
            Geometries to plot.
        ax : matplotlib axis, optional
            Axis to use. If None, a new one is created.
        color : str, optional
            Line color.
        name : str, optional
            Label for the geometries.
        plotPoints : bool, optional
            Whether to plot defining points.
        plotLegend : bool, optional
            Whether to display the legend.

        Returns
        -------
        Axes
            The axis with the plotted geometries.
        """

        for g, geom in enumerate(geoms):
            if isinstance(geom, Point):
                continue
            if ax is None:
                ax = geom.Plot(color=color, name=name, plotPoints=plotPoints)
            else:
                geom.Plot(ax, color, name, plotPoints=plotPoints)

        if plotLegend:
            ax.legend()  # type: ignore [union-attr]

        return ax  # type: ignore
