# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import TYPE_CHECKING

import numpy as np

from ..Utilities import Tic, _types

if TYPE_CHECKING:
    from ..FEM import Mesh
from ..FEM import MatrixType, FeArray, Kinematics, Operators, _GroupElem

from ._utils import (
    Result_strain_or_stress_field_e,
    Stress_names,
    Strain_names,
    _Field_per_groupElem,
)
from ..Models.InElastic._behavior import _Behavior

from ..Solvers import Newton
from ._simu import _Simu, UnknownValues
from .Problem._terms import Term
from .Problem._problem_type import ProblemType


class InElastic(_Simu):
    r"""Quasi-static mechanics :math:`\diver{\Sig} + \fb = 0` by Newton-Raphson, the stress coming from a :class:`~EasyFEA.Models.InElastic._Behavior`.

    A converged :meth:`Solve` is a step: it commits the internal variables, which the next step integrates from and :meth:`Result` reads; :meth:`Save_Iter` only records them.
    """

    U = ProblemType("displacement")
    """the displacement problem"""

    problemTypes = (U,)
    dofFields = {U: ("displacement",)}

    def __init__(
        self,
        mesh: "Mesh",
        model: _Behavior,
        folder: str = "",
        verbosity: bool = False,
        absTol: float = 1e-6,
        relTol: float = 1e-10,
        incTol: float = 1e-11,
        maxIter: int = 20,
    ):
        assert isinstance(model, _Behavior), "model must be a _Behavior"
        super().__init__(mesh, model, folder, verbosity)

        self.Solver_Set_Nonlinear_Strategy(
            Newton(absTol=absTol, relTol=relTol, incTol=incTol, maxIter=maxIter)
        )

        self.__dt = 0.0

    @property
    def dt(self) -> float:
        """Time increment passed to the behavior."""
        return self.__dt

    @dt.setter
    def dt(self, value: float) -> None:
        assert value >= 0.0, "dt must be >= 0"
        self.__dt = value

    @property
    def _externalNames(self) -> tuple[str, ...]:
        return self.material.externalNames

    @property
    def material(self) -> _Behavior:
        return self.model  # type: ignore[return-value]

    @property
    def displacement(self) -> _types.FloatArray:
        """Displacement vector field, [uxi, uyi, (uzi), ...]."""
        return self._Get_u_n(self.problemType)

    def Get_unknowns(self, problemType=None) -> list[str]:
        return {2: ["x", "y"], 3: ["x", "y", "z"]}[self.dim]

    def Results_nodeFields_elementFields(
        self, details=False
    ) -> tuple[list[str], list[str]]:
        elementsField = ["Svm", "Stress", "Strain"] if details else ["Svm", "Stress"]
        return ["displacement", *self.material.externalNames], elementsField

    # --------------------------------------------------------------------------
    # Integration
    # --------------------------------------------------------------------------

    def __Groups(self) -> list[_GroupElem]:
        """The groups of the displacement's space, which the integration hands `__Assemble` and the internal variables live on."""
        return self._Get_FunctionSpace().Get_list_groupElem(self.dim)

    def __Node_values(self, u: _types.FloatArray) -> _types.FloatArray:
        return self._Get_FunctionSpace().Get_nodeValues(u, self.dim)

    def __Internal_e_pg(self, groupElem: _GroupElem) -> dict[str, FeArray]:
        """The internal variables committed at the last converged solve, virgin before any."""
        internals = self._variables.internals
        if not internals:
            nPg = groupElem.Get_gauss(MatrixType.rigi).nPg
            return self.material.Virgin_internals_e_pg(groupElem.Ne, nPg)
        internal = internals.get(groupElem)
        # they could be projected onto the new mesh instead
        assert internal is not None, (
            "the internal variables cannot follow a mesh change"
        )
        return internal

    def __Solved_externals(self) -> dict[str, _types.FloatArray]:
        return self._variables.Solved_externals(self._externalNames)

    def __External_e_pg(
        self, groupElem: _GroupElem, external: dict[str, _types.FloatArray]
    ) -> dict[str, FeArray.FeArrayALike]:
        """The nodal ``external``, interpolated at the Gauss points."""
        # they could be interpolated onto the new mesh instead
        assert all(v.size == groupElem.Nn for v in external.values()), (
            "the external variables cannot follow a mesh change - set them again"
        )
        return {
            name: groupElem.Interpolate_e_pg(v, MatrixType.rigi)
            for name, v in external.items()
        }

    def __Integrate(
        self, u: _types.FloatArray, groupElem: _GroupElem
    ) -> tuple[FeArray, FeArray, dict[str, FeArray]]:
        return self.material.Integrate(
            Kinematics(groupElem, u),
            self.__Internal_e_pg(groupElem),
            self.__dt,
            **self.__External_e_pg(groupElem, self._variables.nextExternals),
        )

    def _Calc_Epsilon(
        self, matrixType: MatrixType = MatrixType.rigi
    ) -> FeArray.FeArrayALike | dict:
        """Strain (Ne, pg, 3 or 6): an ``FeArray`` on one group, ``{groupElem: FeArray}`` on several."""
        return _Field_per_groupElem(
            lambda groupElem: Kinematics(
                groupElem, self.__Node_values(self.displacement), matrixType
            ).Compute_Epsilon(),
            self.__Groups(),
        )

    def _Calc_Sigma(
        self, matrixType: MatrixType = MatrixType.rigi
    ) -> FeArray.FeArrayALike | dict:
        """Stress (Ne, pg, 3 or 6) under the committed internal state: an ``FeArray`` on one group, ``{groupElem: FeArray}`` on several."""
        assert matrixType == MatrixType.rigi, (
            "the internal state lives at the rigi points"
        )
        return _Field_per_groupElem(
            lambda groupElem: self.material.Compute_Sigma(
                Kinematics(
                    groupElem, self.__Node_values(self.displacement), matrixType
                ),
                self.__Internal_e_pg(groupElem),
                **self.__External_e_pg(groupElem, self.__Solved_externals()),
            ),
            self.__Groups(),
        )

    def _Trial_internals(self) -> dict[_GroupElem, dict[str, FeArray]]:
        u = self.__Node_values(self.displacement)
        return {g: self.__Integrate(u, g)[2] for g in self.__Groups()}

    def Get_terms(self, x: UnknownValues, problemType=None) -> list[Term]:
        """One term: the tangent ``∫BᵀC_alg B`` and the internal force ``∫Bᵀσ``."""
        return [Term("KR", self.__Assemble, u=self.Get_u_v_a(self.U, x)[0])]

    def __Assemble(
        self,
        groupElem: _GroupElem,
        u: _types.FloatArray,
        matrixType: MatrixType = MatrixType.rigi,
    ) -> tuple[np.ndarray, np.ndarray]:
        sigma_e_pg, C_e_pg, _ = self.__Integrate(u, groupElem)

        tic = Tic()
        K_e = Operators.Bilinear.LinearizedElasticity(
            groupElem, C_e_pg, matrixType=matrixType
        )
        R_e = Operators.Linear.InternalForce(
            groupElem, sigma_e_pg, matrixType=matrixType
        )
        tic.Tac("Matrix", f"Construct K_e and R_e ({groupElem.elemType})", False)
        return K_e, R_e

    # --------------------------------------------------------------------------
    # Results
    # --------------------------------------------------------------------------

    def __Scalar_states(self) -> list[str]:
        virgin = self.material.Virgin_internals_e_pg(1, 1)
        return [name for name, v in virgin.items() if v.shape == (1, 1)]

    def Results_Available(self) -> list[str]:
        results = [
            *super().Results_Available(),
            *Stress_names(self.dim),
            *Strain_names(self.dim),
        ]
        # scalar internal variables are plottable
        results.extend(self.__Scalar_states())
        results.extend(self.material.externalNames)
        return results

    def _Result(
        self, result: str, nodeValues: bool = True
    ) -> _types.FloatArray | float:
        if result in self.material.externalNames:
            values = self.__Solved_externals()[result]
            return self.Results_Reshape_values_n(values, nodeValues)

        elif result in self.__Scalar_states():
            values_e = np.concatenate(
                [
                    np.mean(self.__Internal_e_pg(g)[result], axis=1)
                    for g in self.__Groups()
                ]
            )
            return self.Results_Reshape_values_e(values_e, nodeValues)

        elif result in Stress_names(self.dim):
            values_e = Result_strain_or_stress_field_e(self._Calc_Sigma(), result)
            return self.Results_Reshape_values_e(values_e, nodeValues)

        elif result in Strain_names(self.dim):
            values_e = Result_strain_or_stress_field_e(self._Calc_Epsilon(), result)
            return self.Results_Reshape_values_e(values_e, nodeValues)

        else:
            return super()._Result(result, nodeValues)

    def Results_displacement_matrix(self) -> _types.FloatArray:
        Nn = self.mesh.Nn
        matrix = np.zeros((Nn, 3))
        space = self._Get_FunctionSpace()
        matrix[:, : self.dim] = space.Get_meshValues(self.displacement, self.dim)
        return matrix
