# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from abc import abstractmethod

# utilities
import numpy as np

# geom
from ...Geoms import Line, AsCoords, Normalize

# fem
from ...FEM import Mesh, _GroupElem, FeArray, MatrixType

# materials
from .._utils import _IModel, ModelType
from ...Utilities import _params, _types

# ----------------------------------------------
# Beam
# ----------------------------------------------


class _Beam(_IModel):
    """Beam class model."""

    # number of created beams
    __nBeam = -1

    @property
    def modelType(self) -> ModelType:
        return ModelType.beam

    @property
    def dim(self):
        return self.__dim

    @property
    def thickness(self) -> float:
        return 1.0

    @property
    def area(self) -> float:
        """cross-section area."""
        return self.__section.area

    @property
    def Iy(self) -> float:
        """squared moment of the cross-section around y-axis.\n
        int_S z^2 dS"""
        # here z is the x axis of the section thats why we use x
        Iy: float = np.sum(
            [
                grp.Integrate_e(lambda x, y, z: x**2).sum()
                for grp in self.__section.Get_list_groupElem(2)
            ]
        )
        return Iy

    @property
    def Iz(self) -> float:
        """squared moment of the cross-section around z-axis.\n
        int_S y^2 dS"""
        Iz: float = np.sum(
            [
                grp.Integrate_e(lambda x, y, z: y**2).sum()
                for grp in self.__section.Get_list_groupElem(2)
            ]
        )
        return Iz

    @property
    def J(self) -> float:
        """polar area moment of inertia (Iy + Iz)."""
        J: float = np.sum(
            [
                grp.Integrate_e(lambda x, y, z: x**2 + y**2).sum()
                for grp in self.__section.Get_list_groupElem(2)
            ]
        )
        return J

    def __init__(
        self,
        dim: int,
        line: Line,
        section: Mesh,
        yAxis: _types.Coords = (0, 1, 0),
    ):
        """Creates a beam.

        Parameters
        ----------
        dim : int
            ceam dimension (1D, 2D or 3D)
        line : Line
            line characterizing the neutral fiber.
        section : Mesh
            beam cross-section.\n
            Must be a 2D mesh belonging to the 2D space.\n
            The cross-section will automatically be centered on its center of gravity.
        yAxis: _types.Coords, optional
            vertical cross-beam axis, by default (0,1,0)
        """

        _Beam.__nBeam += 1
        self.__name = f"beam{_Beam.__nBeam}"

        self.__dim: int = dim
        self.__line: Line = line

        self.section = section

        self.yAxis = yAxis  # type: ignore [assignment]

    @property
    def line(self) -> Line:
        """average fiber line of the beam."""
        return self.__line

    @property
    def section(self) -> Mesh:
        """beam cross-section in (x,y) plane"""
        return self.__section

    @section.setter
    def section(self, section: Mesh) -> None:
        assert (
            section.inDim == 2
        ), "The cross-beam section must be contained in the (x,y) plane."
        # make sure that the section is centered in (0,0)
        section.Translate(*-section.center)
        Iyz = section.groupElem.Integrate_e(lambda x, y, z: x * y).sum()
        assert np.abs(Iyz) <= 1e-9, "The section must have at least 1 symetry axis."
        self.Need_Update()
        self.__section: Mesh = section

    @property
    def xAxis(self) -> _types.FloatArray:
        """perpendicular cross-beam axis (fiber)"""
        return self.__line.unitVector

    @property
    def yAxis(self) -> _types.FloatArray:
        """vertical cross-beam axis"""
        return self.__yAxis.copy()

    @yAxis.setter
    def yAxis(self, value: _types.Coords):
        # set y axis
        xAxis = self.xAxis
        yAxis = Normalize(AsCoords(value))

        # check that the yaxis is not colinear to the fiber axis
        crossProd = np.cross(xAxis, yAxis)
        if np.linalg.norm(crossProd) <= 1e-12:
            # create a new y-axis
            yAxis = Normalize(np.cross([0, 0, 1], xAxis))
            print(
                f"The beam's vertical axis has been selected incorrectly (collinear with the beam x-axis).\nAxis {np.array_str(yAxis, precision=3)} has been assigned for {self.name}."
            )
        else:
            # get the horizontal direction of the beam
            zAxis = Normalize(np.cross(xAxis, yAxis))
            # make sure that x,y,z are orthogonal
            yAxis = Normalize(np.cross(zAxis, xAxis))

        self.Need_Update()

        self.__yAxis = yAxis

    @property
    def name(self) -> str:
        """beam name/tag"""
        return self.__name

    @property
    def dof_n(self) -> int:
        """Degrees of freedom per node
        1D -> [u1, . . . , un]\n
        2D -> [u1, v1, rz1, . . ., un, vn, rzn]\n
        3D -> [u1, v1, w1, rx1, ry1, rz1, . . ., u2, v2, w2, rx2, ry2, rz2]"""
        if self.__dim == 1:
            return 1  # u
        elif self.__dim == 2:
            return 3  # u v rz
        elif self.__dim == 3:
            return 6  # u v w rx ry rz
        return self.__dim

    @abstractmethod
    def Get_D(self) -> _types.FloatArray:
        """Returns a matrix characterizing the beam's stiffness behavior."""
        return None  # type: ignore [return-value]

    def __str__(self) -> str:
        text = ""
        text += f"\n{self.name}:"
        text += f"\n  area = {self.__section.area:.2},"
        text += f"\n  Iz = {self.Iz:.2},"
        text += f"\n  Iy = {self.Iy:.2},"
        text += f"\n  J = {self.J:.2}"

        return text

    def _Calc_P(self) -> _types.FloatArray:
        """P matrix use to transform beam coordinates to global coordinates.\n
        [ix, jx, kx\n
        iy, jy, ky\n
        iz, jz, kz]\n
        coord(x,y,z) = P • coord(i,j,k)
        """
        line = self.line

        i = line.unitVector
        j = self.yAxis
        k = Normalize(np.cross(i, j))

        J = np.array([i, j, k]).T
        return J


class Isotropic(_Beam):
    """Isotropic elastic beam."""

    def __init__(
        self,
        dim: int,
        line: Line,
        section: Mesh,
        E: float,
        v: float,
        yAxis: _types.Coords = (0, 1, 0),
    ):
        """Creates an isotropic elastic beam.

        Parameters
        ----------
        dim : int
            Beam dimension (1D, 2D or 3D)
        line : Line
            Line characterizing the neutral fiber.
        section : Mesh
            Beam cross-section
        E : float
            Young's module
        v : float
            Poisson's ratio
        yAxis: _types.Coords, optional
            vertical cross-beam axis, by default (0,1,0)
        """

        _Beam.__init__(self, dim, line, section, yAxis)

        self.E = E
        self.v = v

    E: float = _params.PositiveParameter()
    """Young's modulus"""

    v: float = _params.IntervalccParameter(inf=-1, sup=0.5)
    """Poisson's ratio"""

    @property
    def mu(self) -> float:
        """shear modulus (G)"""
        return self.E / (2 * (1 + self.v))

    def Get_D(self) -> _types.FloatArray:
        dim = self.dim
        section = self.section
        A = section.area
        Iy = self.Iy
        Iz = self.Iz
        J = self.J

        E = self.E

        if dim == 1:
            # u = [u1, . . . , un]
            D = np.diag([E * A])
        elif dim == 2:
            # u = [u1, v1, rz1, . . . , un, vn, rzn]
            D = np.diag([E * A, E * Iz])
        elif dim == 3:
            # u = [u1, v1, w1, rx1, ry1 rz1, . . . , un, vn, wn, rxn, ryn rzn]
            mu = self.mu
            D = np.diag([E * A, mu * J, E * Iy, E * Iz])

        return D


class BeamStructure(_IModel):
    """Beam structure class."""

    @property
    def modelType(self) -> ModelType:
        return ModelType.beam

    @property
    def dim(self) -> int:
        """model dimensions  \n
        1D -> tension compression \n
        2D -> tension compression + bending + flexion \n
        3D -> all
        """
        return self.__dim

    @property
    def thickness(self) -> float:
        """The beam structure can have several beams and therefore different sections.\n
        You need to look at the section of the beam you are interested in."""
        return None  # type: ignore [return-value]

    @property
    def areas(self) -> list[float]:
        """beams areas"""
        return [beam.area for beam in self.__beams]

    def __init__(self, beams: list[_Beam]) -> None:
        """Creates a beam structure.

        Parameters
        ----------
        beams : list[_Beam_Model]
            Beam list
        """

        dims = [beam.dim for beam in beams]
        assert len(set(dims)) == 1, "The structure must use identical beams dimensions."

        self.__dim: int = dims[0]

        self.__beams: list[_Beam] = beams

        self.__dof_n = beams[0].dof_n

    @property
    def beams(self) -> list[_Beam]:
        return self.__beams

    @property
    def nBeam(self) -> int:
        """Number of beams in the structure"""
        return len(self.__beams)

    @property
    def dof_n(self) -> int:
        """Degrees of freedom per node.\n
        1D -> [u1, . . . , un]\n
        2D -> [u1, v1, rz1, . . ., un, vn, rzn]\n
        3D -> [u1, v1, w1, rx1, ry1, rz1, . . ., u2, v2, w2, rx2, ry2, rz2]
        """
        return self.__dof_n

    def Calc_D_e_pg(self, groupElem: _GroupElem) -> FeArray.FeArrayALike:
        """Returns a matrix characterizing the beams's stiffness behavior."""

        if groupElem.dim != 1:
            return None  # type: ignore [return-value]

        listBeam = self.__beams
        list_D = [beam.Get_D() for beam in listBeam]

        Ne = groupElem.Ne
        nPg = groupElem.Get_gauss(MatrixType.beam).nPg
        # Initialize D_e_pg :
        D_e_pg = FeArray.zeros(Ne, nPg, *list_D[0].shape)

        # For each beam, we will construct the law of behavior on the associated nodes.
        for beam, D in zip(listBeam, list_D):
            # recover elements
            elems = groupElem.Get_Elements_Tag(beam.name)
            D_e_pg[elems] = D

        return D_e_pg

    def Get_axis_e(
        self, groupElem: _GroupElem
    ) -> tuple[_types.FloatArray, _types.FloatArray]:
        """Returns the fiber and cross bar vertical axis on every elements.\n
        return xAxis_e, yAxis_e"""

        if groupElem.dim != 1:
            return None  # type: ignore [return-value]

        beams = self.__beams

        Ne = groupElem.Ne

        xAxis_e = np.zeros((Ne, 3), dtype=float)
        yAxis_e = np.zeros((Ne, 3), dtype=float)

        for beam in beams:
            elems = groupElem.Get_Elements_Tag(beam.name)
            xAxis_e[elems] = beam.xAxis
            yAxis_e[elems] = beam.yAxis

        return xAxis_e, yAxis_e
