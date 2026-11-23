# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information


import numpy as np

from ..FEM import (
    ElemType,
    FeArray,
    Kinematics,
    Mesh,
    MatrixType,
    Operators,
    _GroupElem,
)
from ..FEM._kelvin_mandel import Tensor_to_Kelvin
from ..Models.HyperElastic._laws import _HyperElastic
from ..Utilities import _types
from ..Solvers import Newton
from .Problem._problem_type import ProblemType
from ._simu import _Simu, UnknownValues
from .Problem._terms import Term
from ._utils import (
    Result_strain_or_stress_field_e,
    Stress_names,
    Strain_names,
    _Field_per_groupElem,
)


class IncompressibleHyperElastic(_Simu):
    r"""Perfectly incompressible hyperelastic simulation: displacement and pressure solved by one Newton over ``[u | p]``, on the Taylor–Hood pair derived from u's space (TRI6/TRI3, QUAD9/QUAD4, TETRA10/TETRA4, HEXA27/HEXA8, PRISM18/PRISM6), static, with a direct solver.

    .. math::
        \Sb = \dpartial{W}{\eb} - p \, J \, \Cb^{-1}, \qquad \int_{\Omega_0} q \, (1 - J) \, \dO = 0 \quad \forall \, q

    :math:`p` is a Lagrange multiplier, positive in compression; it is the hydrostatic pressure only with a decoupled law. It is computed, never prescribed: when the whole boundary has prescribed normal displacement its constant mode is pinned and it has zero mean.

    A law with a non-zero bulk term is refused: set its ``K`` to 0. So is a viscosity or an active stress: this simulation is static and elastic. A law without a bulk term that is not decoupled (``SaintVenantKirchhoff``, ``HolzapfelOgden`` with ``K = 0``) is accepted, and :math:`p` is then not the hydrostatic pressure. ``AutoDiff`` is not checked: write its ``W`` without a bulk term.

    The simulation works on a copy of the given mesh that carries p's space, ``simu.mesh``: node ids are the same, moves of the given mesh do not reach it. For P2 u on a linear mesh, ``mesh._With_FunctionSpace(IncompressibleHyperElastic.U, order=2)`` first.
    """

    U = ProblemType("displacement")
    """the displacement problem"""

    P = ProblemType("pressure")
    """the pressure problem, on the vertices of every main-dimension element"""

    problemTypes = (U, P)
    dofFields = {U: ("displacement",), P: ("pressure",)}

    def __init__(
        self,
        mesh: Mesh,
        model: _HyperElastic,
        folder: str = "",
        absTol: float = 1e-6,
        relTol: float = 1e-10,
        incTol: float = 1e-11,
        maxIter: int = 20,
        verbosity: bool = False,
    ):
        """Creates an incompressible simulation; ``matrixType`` defaults to ``MatrixType.mass``, ``rigi`` integrating the coupling too low on curved tetrahedra to keep the constant p a mode."""
        if not isinstance(model, _HyperElastic):
            raise TypeError("model must be a hyperelastic law.")
        for name in model._volumetricParameters:
            if np.any(getattr(model, name) != 0):
                raise ValueError(
                    f"{type(model).__name__} has a bulk term, which locks against the pressure: set {name} to 0."
                )
        if model.eta != 0 or np.any(model.active_stress != 0):
            raise ValueError(
                "This simulation is static and elastic: set eta and active_stress to 0."
            )
        super().__init__(mesh, model, folder, verbosity)
        self.Solver_Set_Nonlinear_Strategy(
            Newton(absTol=absTol, relTol=relTol, incTol=incTol, maxIter=maxIter)
        )
        self.matrixType = MatrixType.mass

    @property
    def mesh(self) -> Mesh:
        """simulation's mesh, carrying p's space."""
        return _Simu.mesh.fget(self)  # type: ignore [attr-defined]

    @mesh.setter
    def mesh(self, mesh: Mesh) -> None:
        _Simu.mesh.fset(self, self.__With_pressure_space(mesh))  # type: ignore [attr-defined]

    def __With_pressure_space(self, mesh: Mesh) -> Mesh:
        """New mesh with p on the vertices of `mesh`'s main-dimension elements; raises when u has no stable pressure space or p already has another one."""
        dim = mesh.dim
        if dim == 1:
            raise ValueError("u/p needs a 2D or 3D mesh: J = 1 makes a 1D mesh rigid.")
        space_u = mesh._Get_FunctionSpace(self.U)
        space_p = mesh._Get_FunctionSpace(self.P)
        for groupElem, uGroup, pGroup in zip(
            mesh.Get_list_groupElem(dim),
            space_u.Get_list_groupElem(dim),
            space_p.Get_list_groupElem(dim),
        ):
            elemType = uGroup.elemType
            if elemType not in (
                ElemType.TRI6,
                ElemType.QUAD9,
                ElemType.TETRA10,
                ElemType.HEXA27,
                ElemType.PRISM18,
            ):
                raise ValueError(
                    f"no stable pressure space for u on {elemType}; use "
                    "mesh._With_FunctionSpace(Simulations.IncompressibleHyperElastic.U, order=2)."
                )
            if pGroup is not groupElem and pGroup.order != 1:
                raise ValueError(
                    f"The mesh carries another pressure space, {pGroup.elemType}: "
                    f"u on {elemType} takes p on its vertices."
                )
        return mesh._With_FunctionSpace(self.P, 1)

    # --------------------------------------------------------------------------
    # General
    # --------------------------------------------------------------------------

    def _Get_monolithic_problems(self) -> list[tuple[ProblemType, ...]]:
        return [(self.U, self.P)]

    def Get_unknowns(self, problemType: ProblemType | None = None) -> list[str]:
        if problemType == self.P:
            return ["p"]
        return ["x", "y", "z"][: self.dim]

    @property
    def matrixType(self) -> MatrixType:
        """Integration rule the tangent and internal force are built at."""
        return self.__matrixType

    @matrixType.setter
    def matrixType(self, value: MatrixType) -> None:
        self.__matrixType = value
        self.Need_Update()

    @property
    def material(self) -> _HyperElastic:
        """hyperelastic material"""
        return self.model  # type: ignore [return-value]

    @property
    def displacement(self) -> _types.FloatArray:
        """Displacement DOFs, [uxi, uyi, uzi, ...]."""
        return self._Get_u_n(self.U)

    @property
    def pressure(self) -> _types.FloatArray:
        """Pressure DOFs, one per vertex."""
        return self._Get_u_n(self.P)

    def add_dirichlet(
        self,
        nodes: _types.IntArray,
        values: list,
        unknowns: list[str],
        problemType: ProblemType | None = None,
        description: str = "",
    ) -> None:
        if problemType == self.P or "p" in unknowns:
            raise ValueError("p is computed: apply a load or a displacement.")
        super().add_dirichlet(
            nodes,
            values,
            unknowns,
            problemType,
            description,
        )

    # --------------------------------------------------------------------------
    # Solve
    # --------------------------------------------------------------------------

    def Get_terms(
        self, x: UnknownValues, problemType: ProblemType | None = None
    ) -> list[Term]:
        u, p = self.U, self.P
        # the constraint's term carries p's equation too
        if problemType == p:
            return []
        u_t = self.Get_u_v_a(u, x)[0]
        p_t = self.Get_u_v_a(p, x)[0]
        return [
            Term("KR", self.__Stress, u=u_t),
            Term(
                "KRKKR",
                self.__MixedIncompressibility,
                u=u_t,
                p=p_t,
                problemTypes=(
                    (u, u),
                    u,
                    (u, p),
                    (p, u),
                    p,
                ),
            ),
        ]

    def __Kinematics(
        self, groupElem: _GroupElem, u: _types.FloatArray, matrixType: MatrixType
    ) -> Kinematics:
        """Kinematics of u's group `groupElem` at node values `u`, at the explicit rule p is sampled at too."""
        rule = self._Get_FunctionSpace(self.U).Get_rule(groupElem, matrixType)
        return Kinematics(groupElem, u, rule)  # type: ignore [arg-type]

    def __Stress(
        self, groupElem: _GroupElem, u: _types.FloatArray
    ) -> tuple[np.ndarray, np.ndarray]:
        """``∂W/∂e``: tangent and internal force on u's group, guarded against an inverted element."""
        kinematics = Kinematics(groupElem, u, self.matrixType)
        assert kinematics.Compute_J().min() > 0, "det(F) < 0 - reduce load steps"
        return Operators.NonLinear.SecondPiolaKirchhoffStressTensor(
            self.material, kinematics
        )

    def __MixedIncompressibility(
        self,
        groupElem: _GroupElem,
        pGroup: _GroupElem,
        u: _types.FloatArray,
        p: _types.FloatArray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``∫ p (1 − J) dΩ`` at `u`, `p`: both equations' tangents and residuals."""
        return Operators.NonLinear.MixedIncompressibility(
            self.__Kinematics(groupElem, u, self.matrixType), pGroup, p
        )

    # --------------------------------------------------------------------------
    # Results
    # --------------------------------------------------------------------------

    def Results_Available(self) -> list[str]:
        return [
            *super().Results_Available(),
            *Stress_names(self.dim, "Piola-Kirchhoff"),
            *Strain_names(self.dim, "Green-Lagrange"),
            "W",
            "W_e",
            "J",
        ]

    def _Result(
        self, result: str, nodeValues: bool = True
    ) -> _types.FloatArray | float:
        if result in Stress_names(self.dim, "Piola-Kirchhoff"):
            field = self._Calc_SecondPiolaKirchhoff()
            values_e = Result_strain_or_stress_field_e(field, result)
        elif result in Strain_names(self.dim, "Green-Lagrange"):
            field = self._Calc_GreenLagrange()
            values_e = Result_strain_or_stress_field_e(field, result)
        elif result == "W":
            return self._Calc_W()
        elif result == "W_e":
            values_e = np.asarray(self._Calc_W(False))
        elif result == "J":
            values_e = self._Calc_J()
        else:
            return super()._Result(result, nodeValues)
        return self.Results_Reshape_values_e(values_e, nodeValues)

    def __Solved_kinematics(
        self, groupElem: _GroupElem, matrixType: MatrixType | None = None
    ) -> Kinematics:
        """:meth:`__Kinematics` at the solved displacement, at ``self.matrixType`` by default."""
        u = self._Get_FunctionSpace(self.U).Get_nodeValues(self.displacement, self.dim)
        return self.__Kinematics(groupElem, u, matrixType or self.matrixType)

    def _Calc_W(
        self, returnScalar: bool = True, matrixType: MatrixType = MatrixType.rigi
    ) -> float | np.ndarray:
        r"""Hyperelastic strain energy :math:`\int_{\Omega_0} W \, \dO`, a float, or per element (Ne,)."""
        list_W = []
        space_u = self._Get_FunctionSpace(self.U)
        for groupElem in space_u.Get_list_groupElem():
            kinematics = self.__Solved_kinematics(groupElem, matrixType)
            wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)
            W_e_pg: FeArray = wJ_e_pg * self.material.Compute_W(kinematics)
            list_W.append(self.thickness * W_e_pg.integrate())
        if not returnScalar:
            return np.concatenate(list_W)
        return float(self.mesh._Sum_on_ownedElements_e(list_W))

    def _Calc_J(self, matrixType: MatrixType | None = None) -> _types.FloatArray:
        """Mean of det(F) on each element (Ne,), at ``self.matrixType`` by default."""
        space_u = self._Get_FunctionSpace(self.U)
        return np.concatenate(
            [
                self.__Solved_kinematics(groupElem, matrixType).Compute_J().mean(1)
                for groupElem in space_u.Get_list_groupElem()
            ]
        )

    def _Calc_GreenLagrange(
        self, matrixType: MatrixType | None = None
    ) -> FeArray.FeArrayALike | dict:
        """Green-Lagrange strain in Kelvin-Mandel form, at ``self.matrixType`` by default: an ``FeArray`` on one group, ``{groupElem: FeArray}`` on several."""
        space_u = self._Get_FunctionSpace(self.U)
        return _Field_per_groupElem(
            lambda groupElem: Tensor_to_Kelvin(
                self.__Solved_kinematics(groupElem, matrixType).Compute_GreenLagrange(),
                2,
            ),
            space_u.Get_list_groupElem(),
        )

    def _Calc_SecondPiolaKirchhoff(
        self, matrixType: MatrixType | None = None
    ) -> FeArray.FeArrayALike | dict:
        """Total second Piola-Kirchhoff stress ``∂W/∂e − p J C⁻¹``, at ``self.matrixType`` by default: an ``FeArray`` on one group, ``{groupElem: FeArray}`` on several."""
        space_u = self._Get_FunctionSpace(self.U)
        space_p = self._Get_FunctionSpace(self.P)
        list_groupElem = space_u.Get_list_groupElem()
        pGroups = dict(zip(list_groupElem, space_p.Get_list_groupElem()))

        def S_e_pg(groupElem: _GroupElem) -> FeArray.FeArrayALike:
            kinematics = self.__Solved_kinematics(groupElem, matrixType)
            p_e_pg = pGroups[groupElem].Interpolate_e_pg(
                space_p.Get_nodeValues(self.pressure, 1), kinematics.matrixType
            )
            dJdE_e_pg = kinematics.Compute_dI3dC() / kinematics.Compute_J()
            return self.material.Compute_dWde(kinematics) - p_e_pg * dJdE_e_pg

        return _Field_per_groupElem(S_e_pg, list_groupElem)

    def Results_displacement_matrix(self) -> _types.FloatArray:
        space_u = self._Get_FunctionSpace(self.U)
        values = space_u.Get_meshValues(self.displacement, self.dim)
        matrix = np.zeros((self.mesh.Nn, 3))
        matrix[:, : self.dim] = values
        return matrix

    def Results_nodeFields_elementFields(
        self, details: bool = False
    ) -> tuple[list[str], list[str]]:
        elementsField = ["Green-Lagrange"] if details else []
        return ["displacement", "pressure"], [*elementsField, "Piola-Kirchhoff"]
