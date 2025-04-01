# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the _Geom class used to construct geometry. Geometry inherits from _Geom."""

from abc import ABC, abstractmethod
import numpy as np
import copy
from collections.abc import Iterable
import matplotlib.pyplot as plt

from ._utils import Point, Rotate, Symmetry

class _Geom(ABC):

    def __init__(self, points: list[Point], meshSize: float, name: str, isHollow: bool, isOpen: bool):
        """Creates a geometric object.

        Parameters
        ----------
        points : list[Point]
            list of points to build the geometric object
        meshSize : float
            mesh size that will be used to create the mesh >= 0
        name : str
            object name
        isHollow : bool
            Indicates whether the the formed domain is hollow/empty
        isOpen : bool
            Indicates whether the object can open to represent an open crack (openCrack)
        """
        
        assert isinstance(points, Iterable) and isinstance(points[0], Point), "points must be a list of points."
        self.__points: list[Point] = points

        self.meshSize = meshSize
        self.name = name
        self.isHollow = isHollow
        self.isOpen = isOpen

    # TODO Add a To_Mesh_Function() ?

    @property
    def meshSize(self) -> float:
        """element size used for meshing"""
        return self.__meshSize
    
    @meshSize.setter
    def meshSize(self, value) -> None:        
        assert value >= 0, "meshSize must be >= 0"
        self.__meshSize = value

    # points doesn't have a public setter for safety
    @property
    def points(self) -> list[Point]:
        """Points used to build the object."""
        return self.__points
    
    @property
    def coord(self) -> np.ndarray:
        return np.asarray([p.coord for p in self.points])
    
    @abstractmethod
    def Get_coord_for_plot(self) -> tuple[np.ndarray,np.ndarray]:
        """Returns coordinates for constructing lines and points."""
        lines = self.coord
        points = lines[[0,-1]]
        return lines, points    
    
    def copy(self):
        new = copy.deepcopy(self)
        new.name = new.name +'_copy'        
        return new

    @property
    def name(self) -> str:
        """object name"""
        return self.__name
    
    @name.setter
    def name(self, val: str) -> None:
        assert isinstance(val, str), "must be a string"
        self.__name = val
    
    @property
    def isHollow(self) -> bool:
        """Indicates whether the the formed domain is hollow/filled."""
        return self.__isHollow
    
    @isHollow.setter
    def isHollow(self, value: bool) -> None:
        assert isinstance(value, bool), "must be a boolean"
        self.__isHollow = value
    
    @property
    def isOpen(self) -> bool:
        """Indicates whether the object can open to represent an open crack."""
        return self.__isOpen
    
    @isOpen.setter
    def isOpen(self, value: bool) -> None:
        assert isinstance(value, bool), "must be a boolean"
        self.__isOpen = value
    
    def Translate(self, dx: float=0.0, dy: float=0.0, dz: float=0.0) -> None:
        """Translates the object."""
        # to translate an object, all you have to do is move these points
        [p.Translate(dx,dy,dz) for p in self.__points]
    
    def Rotate(self, theta: float, center: tuple=(0,0,0), direction: tuple=(0,0,1)) -> None:        
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
        [point.Translate(*dec[p]) for p, point in enumerate(self.points)]

    def Symmetry(self, point=(0,0,0), n=(1,0,0)) -> None:
        """Symmetrizes the object coordinates with a plane.

        Parameters
        ----------
        point : tuple, optional
            a point belonging to the plane, by default (0,0,0)
        n : tuple, optional
            normal to the plane, by default (1,0,0)
        """

        oldCoord = self.coord
        newCoord = Symmetry(oldCoord, point, n)

        dec = newCoord - oldCoord
        [point.Translate(*dec[p]) for p, point in enumerate(self.points)]

    def Plot(self, ax: plt.Axes=None, color:str="", name:str="", lw=None, ls=None, plotPoints=True) -> plt.Axes:

        from ..utilities.Display import Init_Axes, _Axis_equal_3D

        if ax is None:
            ax = Init_Axes(3)
            
        inDim = 3 if ax.name == '3d' else 2

        name = self.name if name == "" else name

        lines, points = self.Get_coord_for_plot()
        if color != "":
            ax.plot(*lines[:,:inDim].T, color=color, label=name, lw=lw, ls=ls)
        else:
            ax.plot(*lines[:,:inDim].T, label=name, lw=lw, ls=ls)
        if plotPoints:
            ax.plot(*points[:,:inDim].T, ls='', marker='.',c='black')

        if inDim == 3:
            xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
            oldBounds = np.array([xlim, ylim, zlim]).T
            lines = np.concatenate((lines, oldBounds), 0)
            _Axis_equal_3D(ax, lines)
        else:
            ax.axis('equal')

        return ax
    
    @staticmethod
    def Plot_Geoms(geoms: list, ax: plt.Axes=None,
                   color:str="", name:str="", plotPoints=True, plotLegend=True) -> plt.Axes:
        geoms: list[_Geom] = geoms
        for g, geom in enumerate(geoms):
            if isinstance(geom, Point): continue
            if ax is None:
                ax = geom.Plot(color=color, name=name, plotPoints=plotPoints)
            else:
                geom.Plot(ax, color, name, plotPoints=plotPoints)

        if plotLegend:
            ax.legend()

        return ax