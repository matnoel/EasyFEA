# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from abc import abstractmethod

# utilities
import numpy as np

# geom
from ...Geoms import Line, AsCoords, Normalize

# fem
from ...FEM import Mesh, _GroupElem, FeArray, MatrixType, ElemType
from ...FEM import Field, BiLinearForm, LinearForm
from ...FEM.Elems._beam import _Timoshenko, _EulerBernoulli

# simulations / models — used by _shear_kappa (Saint-Venant Poisson solve)
from ... import Models, Simulations

# materials
from .._utils import _IModel, ModelType
from ...Utilities import Display, _params, _types

# Linear-order 2-D element types. The Saint-Venant Poisson solution is cubic
# for typical sections, so these only achieve O(h²) — _shear_kappa warns once.
_LINEAR_2D_ELEMS = (ElemType.TRI3, ElemType.QUAD4)

# ----------------------------------------------
# Beam
# ----------------------------------------------


class _Beam(_IModel):
    """Beam class model."""

    # number of created beams
    __nBeam = -1

    _ky = _params.PositiveScalarParameter()
    """Shear correction factor k for the cross-section."""
    _kz = _params.PositiveScalarParameter()
    """Shear correction factor k for the cross-section."""

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

        self._ky = self._Get_shear_kappa("y")
        self._kz = self._Get_shear_kappa("z")

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
    def Get_D(self, useTimoshenko: bool = False) -> _types.FloatArray:
        """Returns a matrix characterizing the beam's stiffness behavior."""
        return None  # type: ignore [return-value]

    @abstractmethod
    def Get_M(self) -> _types.FloatArray:
        """Returns a matrix characterizing the beam's mass behavior."""
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

    def _Get_shear_kappa(self, axis: str = "y") -> float:
        """Cowper's (1966) shear correction factor k for the cross-section.

        Computes k by solving a single Saint-Venant Poisson problem on the
        2-D section mesh:

            ∇²φ = -s   in the section S
            ∂φ/∂n = 0  on the boundary ∂S

        where s is the slicing coordinate (s = y ν for ``axis="y"`` returns k_y
        paired with Iz; s = x ν for ``axis="z"`` returns k_z paired with Iy).
        The rigid (constant) mode is fixed by clamping one node to 0 — k
        below is translation-invariant.

        With ∂φ/∂n = 0 the energy identity gives uᵀ·f = ∫_S |∇φ|² = ∫_S s·φ

            k = I² / (A · uᵀ · f)

        with I = Iz (or Iy for axis="z").  Reference values:

            rectangle (any bxh)  →  5/6   ≈ 0.8333
            circle    (any d)    →  6/7   ≈ 0.8571
            I-beam, tube, etc.   →  whatever the geometry says

        Note: Pure Jouravski's 1-D Q/b formula gives 9/10 for a circle; this
        function returns 6/7 because the Poisson PDE captures the *2-D*
        shear-stress field, which is what the underlying elasticity problem
        actually has.  Both methods agree on rectangles.

        Cowper-with-ν ≠ 0 would need a different PDE (non-zero Neumann
        boundary term involving ν) and is not implemented here — if you need
        a specific k value (Cowper-ν, Pure Jouravski, a value from a steel
        section table, …), set ``beam._ky`` / ``beam._kz`` directly instead
        of using this helper.
        """
        if axis == "y":
            bending_inertia = self.Iz
        elif axis == "z":
            bending_inertia = self.Iy
        else:
            raise ValueError(f"axis must be 'y' or 'z', got {axis!r}")

        section = self.section

        # Warn once per beam if the section uses linear 2-D elements (TRI3 /
        # QUAD4): the Poisson solution is cubic, so we get only O(h²) accuracy.
        # TRI6 / QUAD8 capture cubic with O(h³), TRI10+ are exact.
        if section.groupElem.elemType in _LINEAR_2D_ELEMS and not getattr(
            self, "_warned_linear_section", False
        ):
            Display.MyPrint(
                f"Beam: section uses linear {section.groupElem.elemType.name} "
                "elements — _shear_kappa converges at O(h²). "
                "Use TRI6 / QUAD8 (or finer mesh) for accurate k.",
                color="yellow",
                end="\n",
            )
            self._warned_linear_section = True

        # Weak form of  ∇²φ = -s  with Neumann ∂φ/∂n = 0  is
        #   ∫_S ∇u · ∇v dS  =  ∫_S s · v dS.
        field = Field(section.groupElem, 1)

        @BiLinearForm
        def bilinear(u: Field, v: Field):
            return u.grad.dot(v.grad)

        @LinearForm
        def linear(v: Field):
            x, y, _ = v.Get_coords()
            return (y if axis == "y" else x) * v

        weakForms = Models.WeakForms(field, computeK=bilinear, computeF=linear)
        simu = Simulations.WeakForms(section, weakForms, verbosity=False)
        # Pin one DOF to remove the rigid mode (Neumann-only problem is singular).
        # Doesn't affect uᵀ·f because the source ∫_S s dS = 0 on a centered section.
        simu.add_dirichlet([0], [0.0], ["u"])
        simu.Solve()

        # uᵀ·f  ≡  ∫_S s · φ dS  ≡  ∫_S |∇φ|² dS  (energy identity)
        f = simu.Get_K_C_M_F()[3]
        u = simu._Get_u_n(simu.problemType, asCsrMatrix=True)
        integral = (u.T @ f)[0, 0]
        kappa = bending_inertia**2 / (section.area * integral)
        return kappa


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

    def Get_D(self, useTimoshenko: bool = False) -> _types.FloatArray:
        dim = self.dim
        section = self.section
        A = section.area
        Iy = self.Iy
        Iz = self.Iz
        J = self.J

        E = self.E

        # Shear correction factors are read straight from the beam's _ky / _kz
        # parameters (default 1.0 — "no correction").  Set them explicitly to
        # use a numerical value (e.g. ``beam._ky = beam._shear_kappa("y")`` for
        # Cowper-ν=0, or ``beam._ky = 5/6`` for the textbook rectangular value).
        if dim == 1:
            # u = [u1, . . . , un]
            D = np.diag([E * A])
        elif dim == 2:
            # u = [u1, v1, rz1, . . . , un, vn, rzn]
            if useTimoshenko:
                mu = self.mu
                D = np.diag([E * A, E * Iz, mu * self._ky * A])
            else:
                D = np.diag([E * A, E * Iz])
        elif dim == 3:
            # u = [u1, v1, w1, rx1, ry1 rz1, . . . , un, vn, wn, rxn, ryn rzn]
            mu = self.mu
            if useTimoshenko:
                # 6 rows: [axial, torsion, flex-y, flex-z, shear-y, shear-z]
                D = np.diag(
                    [
                        E * A,
                        mu * J,
                        E * Iy,
                        E * Iz,
                        self._ky * mu * A,
                        self._kz * mu * A,
                    ]
                )
            else:
                D = np.diag([E * A, mu * J, E * Iy, E * Iz])

        return D

    def Get_M(self) -> _types.FloatArray:
        dim = self.dim
        section = self.section
        A = section.area

        if dim == 1:
            # u = [u1, . . . , un]
            M = np.diag([A])
        elif dim == 2:
            # u = [u1, v1, rz1, . . . , un, vn, rzn]
            M = np.diag([A, A, 0])
        elif dim == 3:
            # u = [u1, v1, w1, rx1, ry1 rz1, . . . , un, vn, wn, rxn, ryn rzn]
            M = np.diag([A, A, A, 0, 0, 0])

        return M


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

    def Calc_D_e_pg(
        self,
        groupElem: _GroupElem,
        matrixType: MatrixType = MatrixType.beam,
    ) -> FeArray.FeArrayALike:
        """Returns a matrix characterizing the beams's stiffness behavior."""

        assert isinstance(groupElem, (_Timoshenko, _EulerBernoulli))
        useTimoshenko = isinstance(groupElem, _Timoshenko)

        listBeam = self.__beams

        list_D = [beam.Get_D(useTimoshenko) for beam in listBeam]

        Ne = groupElem.Ne
        nPg = groupElem.Get_gauss(matrixType).nPg
        # Initialize D_e_pg :
        D_e_pg = FeArray.zeros(Ne, nPg, *list_D[0].shape)

        # For each beam, we will construct the law of behavior on the associated nodes.
        for beam, D in zip(listBeam, list_D):
            # recover elements
            elems = groupElem.Get_Elements_Tag(beam.name)
            D_e_pg[elems] = D

        return D_e_pg

    def Calc_M_e_pg(self, groupElem: _GroupElem) -> FeArray.FeArrayALike:
        """Returns a matrix characterizing the beams's stiffness behavior."""

        assert isinstance(groupElem, (_Timoshenko, _EulerBernoulli))

        listBeam = self.__beams
        list_M = [beam.Get_M() for beam in listBeam]

        Ne = groupElem.Ne
        nPg = groupElem.Get_gauss(MatrixType.beam).nPg
        # Initialize D_e_pg :
        M_e_pg = FeArray.zeros(Ne, nPg, *list_M[0].shape)

        # For each beam, we will construct the law of behavior on the associated nodes.
        for beam, M in zip(listBeam, list_M):
            # recover elements
            elems = groupElem.Get_Elements_Tag(beam.name)

            M_e_pg[elems] = M

        return M_e_pg

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
