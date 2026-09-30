# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ..Utilities import Terminal, Tic, _types

if TYPE_CHECKING:
    from ..FEM import Mesh
    from ..FEM._utils import ElemType
from ..FEM import MatrixType, FeArray, Operators, _GroupElem

from ..Models import Result_strain_or_stress_field_e
from ..Models.InElastic.Contract import _Behavior

from ._simu import _Simu
from ._terms import Term
from ._problem_type import ProblemType


class _Committed(NamedTuple):
    stress: FeArray
    state: dict[str, FeArray]

    def Copy(self) -> "_Committed":
        return _Committed(
            self.stress.copy(), {name: v.copy() for name, v in self.state.items()}
        )


class InElasticContract(_Simu):
    r"""Quasi-static mechanics :math:`\diver{\Sig} + \fb = 0` by Newton-Raphson, the stress coming from a :class:`~EasyFEA.Models.InElastic.Contract._Behavior`.

    Every step integrates from the state committed at the last :meth:`Save_Iter`, which commits the state and the stress together.
    """

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
        assert isinstance(model, _Behavior), "model must be a Behavior"
        super().__init__(mesh, model, folder, verbosity)

        self._Solver_Set_Newton_Raphson_Algorithm(absTol, relTol, incTol, maxIter)

        self.__dt = 0.0
        # per group, at the last Save_Iter; empty before any
        self.__committed: dict["ElemType", _Committed] = {}
        self.__isSaved = False

    @property
    def dt(self) -> float:
        """Time increment passed to the behavior."""
        return self.__dt

    @dt.setter
    def dt(self, value: float) -> None:
        assert value >= 0.0, "dt must be >= 0"
        self.__dt = value

    @property
    def material(self) -> _Behavior:
        return self.model  # type: ignore[return-value]

    @property
    def displacement(self) -> _types.FloatArray:
        """Displacement vector field, [uxi, uyi, (uzi), ...]."""
        return self._Get_u_n(self.problemType)

    def Get_unknowns(self, problemType=None) -> list[str]:
        return {2: ["x", "y"], 3: ["x", "y", "z"]}[self.dim]

    def Get_problemTypes(self) -> list[ProblemType]:
        return [ProblemType("elastic")]

    def Get_dof_n(self, problemType=None) -> int:
        return self.dim

    def Get_x0(self, problemType=None):
        if self.displacement.size != self.mesh.Nn * self.dim:
            return np.zeros(self.mesh.Nn * self.dim)
        return self.displacement

    def Results_nodeFields_elementFields(
        self, details=False
    ) -> tuple[list[str], list[str]]:
        elementsField = ["Svm", "Stress", "Strain"] if details else ["Svm", "Stress"]
        return ["displacement"], elementsField

    # --------------------------------------------------------------------------
    # Integration
    # --------------------------------------------------------------------------

    def __Groups(self) -> list[_GroupElem]:
        return self.mesh.Get_list_groupElem(self.dim)

    def __Strain(self, u: _types.FloatArray, groupElem: _GroupElem) -> FeArray:
        u_e = groupElem.Locates_sol_e(u, asFeArray=True)
        return groupElem.Get_B_e_pg(MatrixType.rigi) @ u_e

    def __Start(self, groupElem: _GroupElem) -> dict[str, FeArray]:
        """The state the current step integrates from."""
        if groupElem.elemType in self.__committed:
            return self.__committed[groupElem.elemType].state
        nPg = groupElem.Get_gauss(MatrixType.rigi).nPg
        return self.material.Virgin_state_e_pg(groupElem.Ne, nPg)

    def __Integrate(
        self, u: _types.FloatArray, groupElem: _GroupElem
    ) -> tuple[FeArray, FeArray, dict[str, FeArray]]:
        eps_e_pg = self.__Strain(u, groupElem)
        return self.material.Integrate(eps_e_pg, self.__Start(groupElem), self.__dt)

    def __Current(self, groupElem: _GroupElem) -> _Committed:
        """Committed once saved; otherwise the trial step at the current displacement."""
        if self.__isSaved:
            return self.__committed[groupElem.elemType]
        sig, _, z = self.__Integrate(self.displacement, groupElem)
        return _Committed(sig, z)

    def Get_terms(self, problemType=None) -> list[Term]:
        """One term: the tangent ``∫BᵀC_alg B`` and the internal force ``∫Bᵀσ``."""
        return [Term("KR", self.__Assemble)]

    def __Assemble(
        self, groupElem: _GroupElem, matrixType: MatrixType = MatrixType.rigi
    ) -> tuple[np.ndarray, np.ndarray]:
        self.__isSaved = False
        u = self._Solver_Get_Newton_Raphson_current_solution()
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
    # Iterations
    # --------------------------------------------------------------------------

    def Save_Iter(self, iter=None):
        if iter is None:
            iter = {}
        iter["displacement"] = self.displacement

        self.__committed = {g.elemType: self.__Current(g) for g in self.__Groups()}
        self.__isSaved = True
        iter["committed"] = {et: c.Copy() for et, c in self.__committed.items()}

        return super().Save_Iter(iter)

    def Set_Iter(self, iter: int = -1, resetAll=False) -> dict:
        results = super().Set_Iter(iter)
        if results is None:
            return results

        u = results["displacement"]
        self._Set_solutions(self.problemType, u, np.zeros_like(u), np.zeros_like(u))
        self.__committed = {et: c.Copy() for et, c in results["committed"].items()}
        self.__isSaved = True
        return results

    # --------------------------------------------------------------------------
    # Results
    # --------------------------------------------------------------------------

    def __Scalar_states(self) -> list[str]:
        virgin = self.material.Virgin_state_e_pg(1, 1)
        return [name for name, v in virgin.items() if v.shape == (1, 1)]

    def Results_Available(self) -> list[str]:
        results = ["displacement", "displacement_norm", "displacement_matrix"]
        if self.dim == 2:
            results.extend(["ux", "uy", "Sxx", "Syy", "Sxy", "Exx", "Eyy", "Exy"])
        else:
            results.extend(["ux", "uy", "uz"])
            results.extend(["Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy"])
            results.extend(["Exx", "Eyy", "Ezz", "Eyz", "Exz", "Exy"])
        results.extend(["Svm", "Stress", "Evm", "Strain"])
        # scalar internal variables are plottable
        results.extend(self.__Scalar_states())
        return results

    def Result(
        self,
        result: str,
        nodeValues: bool = True,
        iter: int | None = None,
    ) -> _types.FloatArray | float:
        if iter is not None:
            self.Set_Iter(iter)

        if not self._Results_Check_Available(result):
            return None  # type: ignore[return-value]

        Nn = self.mesh.Nn
        u = self.displacement

        if result in ["ux", "uy", "uz"]:
            values = u.reshape(Nn, -1)[:, {"x": 0, "y": 1, "z": 2}[result[-1]]]

        elif result == "displacement":
            values = u

        elif result == "displacement_norm":
            values = np.linalg.norm(u.reshape(Nn, -1), axis=1)

        elif result == "displacement_matrix":
            values = self.Results_displacement_matrix()

        elif result in self.__Scalar_states():
            values = np.concatenate(
                [
                    np.mean(self.__Current(g).state[result], axis=1)
                    for g in self.__Groups()
                ]
            )

        elif ("S" in result or "E" in result) and "_norm" not in result:
            isStress = "S" in result and result != "Strain"
            res = result if result in ["Strain", "Stress"] else result[-2:]

            def field_e_pg(groupElem):
                if isStress:
                    return self.__Current(groupElem).stress
                return self.__Strain(u, groupElem)

            values = Result_strain_or_stress_field_e(
                field_e_pg=field_e_pg,
                list_groupElem=self.__Groups(),
                result=res,
                coef=self.material.coef,
            )

        else:
            Terminal.MyPrintError(f"The result '{result}' is not implemented yet.")
            return None  # type: ignore[return-value]

        return self.Results_Reshape_values(values, nodeValues)

    def Results_Iter_Summary(
        self,
    ) -> tuple[list[int], list[tuple[str, _types.FloatArray]]]:
        return super().Results_Iter_Summary()

    def Results_dict_Energy(self) -> dict[str, float]:
        # the contract declares no free energy
        return {}

    def Results_displacement_matrix(self) -> _types.FloatArray:
        Nn = self.mesh.Nn
        matrix = np.zeros((Nn, 3))
        matrix[:, : self.dim] = self.displacement.reshape((Nn, -1))
        return matrix
