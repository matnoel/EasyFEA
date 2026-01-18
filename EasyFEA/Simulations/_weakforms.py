# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Union, Optional, TYPE_CHECKING
import numpy as np

# utilities
from ..Utilities import Tic, _types

# fem
if TYPE_CHECKING:
    from ..FEM import Mesh

# models
from .. import Models
from ..Models import ModelType

# simu
from ._simu import _Simu
from .Solvers import AlgoType


class WeakForms(_Simu):
    def __init__(
        self,
        mesh: "Mesh",
        model: Models.WeakForms,
        isNonLinear=False,
        tolConv=1e-5,
        maxIter=20,
        verbosity=False,
    ):
        """Creates a thermal simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh used.
        model : WeakForms
            The model used.
        isNonLinear : bool, optional
            If True, the simulation is non linear. Defaults to False.
        tolConv : float, optional
            threshold used to check convergence, by default 1e-5
        maxIter : int, optional
            Maximum iterations for convergence, by default 20
        verbosity : bool, optional
            If True, the simulation can write in the terminal. Defaults to False.
        """

        assert isinstance(model, Models.WeakForms), "model must be a weakf form manager"
        super().__init__(mesh, model, verbosity)

        if isNonLinear:
            self._Solver_Set_Newton_Raphson_Algorithm(tolConv=tolConv, maxIter=maxIter)

    def _Check_dim_mesh_material(self) -> None:
        pass

    def Get_unknowns(self, problemType=None) -> list[str]:

        dof_n = self.weakForms.field.dof_n

        if dof_n == 1:
            return ["u"]
        elif 1 < dof_n <= 3:
            dofs = ["x", "y", "z"]
            return [dofs[d] for d in range(dof_n)]
        else:
            raise ValueError("Unknown dof_n configuration.")

    def Get_dof_n(self, problemType=None) -> int:
        return self.weakForms.field.dof_n

    def Results_nodeFields_elementFields(
        self, details=False
    ) -> tuple[list[str], list[str]]:
        nodesField = ["u", "v", "a"]
        elementsField: list[str] = []
        return nodesField, elementsField

    def Get_problemTypes(self) -> list[ModelType]:
        return [ModelType.weakForm]

    @property
    def weakForms(self) -> Models.WeakForms:
        """Weak form manager."""
        return self.model  # type: ignore [return-value]

    @property
    def u(self) -> _types.FloatArray:
        """node field u."""
        return self._Get_u_n(self.problemType)

    @property
    def v(self) -> _types.FloatArray:
        """node field v = dudt"""
        return self._Get_v_n(self.problemType)

    @property
    def a(self) -> _types.FloatArray:
        """node field a = d2udt2"""
        return self._Get_a_n(self.problemType)

    def Get_x0(self, problemType=None):
        return self.u

    def Construct_local_matrix_system(self, problemType):

        # Data
        weakForms = self.weakForms
        field = weakForms.field
        thickness = 1.0 if self.mesh.inDim == 3 else weakForms.thickness

        tic = Tic()

        computeK = weakForms.computeK
        if computeK is None:
            K_e = None
        else:
            K_e = computeK.Integrate_e(field) * thickness

        tic.Tac("Matrix", "Compute the local K matrix.", self._verbosity)

        computeC = weakForms.computeC
        if computeC is None:
            C_e = None
        else:
            C_e = computeC.Integrate_e(field) * thickness

        tic.Tac("Matrix", "Compute the local C matrix.", self._verbosity)

        computeM = weakForms.computeM
        if computeM is None:
            M_e = None
        else:
            M_e = computeM.Integrate_e(field) * thickness

        tic.Tac("Matrix", "Compute the local M matrix.", self._verbosity)

        computeF = weakForms.computeF
        if computeF is None:
            F_e = None
        else:
            F_e = computeF.Integrate_e(field) * thickness

        tic.Tac("Matrix", "Compute the local F vector.", self._verbosity)

        return K_e, C_e, M_e, F_e

    def Save_Iter(self):
        iter = super().Save_Iter()

        if self.algo == AlgoType.elliptic:
            iter["u"] = self.u

        elif self.algo == AlgoType.parabolic:
            iter["u"] = self.u
            iter["v"] = self.v

        elif self.algo in AlgoType.Get_Hyperbolic_Types():
            iter["u"] = self.u
            iter["v"] = self.v
            iter["a"] = self.a

        else:
            raise TypeError("Unknown algo type.")

        self._results.append(iter)

    def Set_Iter(self, iter: int = -1, resetAll=False) -> dict:
        results = super().Set_Iter(iter)

        if results is None:
            return

        if self.algo == AlgoType.elliptic:
            u = results["u"]
            self._Set_solutions(self.problemType, u)

        elif self.algo == AlgoType.parabolic:
            u = results["u"]
            v = results["v"]
            self._Set_solutions(self.problemType, u, v)

        elif self.algo in AlgoType.Get_Hyperbolic_Types():
            u = results["u"]
            v = results["v"]
            a = results["a"]
            self._Set_solutions(self.problemType, u, v, a)

        else:
            raise TypeError("Unknown algo type.")

        return results

    def Results_Available(self) -> list[str]:
        options = []
        options.extend(["u", "v", "a", "displacement_matrix"])

        dof_n = self.weakForms.field.dof_n

        if dof_n == 1:
            pass
        elif 1 < dof_n <= 3:
            sols = ["u", "v", "a"]
            dofs = ["x", "y", "z"]
            [
                options.append(f"{sols[s]}{dofs[d]}")  # type: ignore [func-returns-value]
                for s in range(3)
                for d in range(dof_n)
            ]
        else:
            raise ValueError("Unknown dof_n configuration.")

        return options

    def __indexResult(self, result: str) -> int:
        if len(result) <= 2:
            "Case were ui, vi or ai"
            if "x" in result:
                return 0
            elif "y" in result:
                return 1
            elif "z" in result:
                return 2
            else:
                raise ValueError("result error")
        else:
            raise ValueError("result error")

    def Result(
        self, result: str, nodeValues: bool = True, iter: Optional[int] = None
    ) -> Union[_types.FloatArray, float]:
        if iter is not None:
            self.Set_Iter(iter)

        if not self._Results_Check_Available(result):
            return None  # type: ignore [return-value]

        # begin cases ----------------------------------------------------

        Nn = self.mesh.Nn

        if result == "u":
            values = self.u

        elif result in ["ux", "uy", "uz"]:
            values_n = self.u.reshape(Nn, -1)
            values = values_n[:, self.__indexResult(result)]

        elif result == "v":
            values = self.v

        elif result in ["vx", "vy", "vz"]:
            values_n = self.u.reshape(Nn, -1)
            values = values_n[:, self.__indexResult(result)]

        elif result == "a":
            values = self.a

        elif result in ["ax", "ay", "az"]:
            values_n = self.u.reshape(Nn, -1)
            values = values_n[:, self.__indexResult(result)]

        elif result == "displacement_matrix":
            values = self.Results_displacement_matrix()

        # end cases ----------------------------------------------------

        return self.Results_Reshape_values(values, nodeValues)

    def Results_Iter_Summary(
        self,
    ) -> tuple[list[int], list[tuple[str, _types.FloatArray]]]:
        return super().Results_Iter_Summary()

    def Results_dict_Energy(self) -> dict[str, float]:
        return super().Results_dict_Energy()

    def Results_displacement_matrix(self) -> _types.FloatArray:

        dof_n = self.weakForms.field.dof_n
        Nn = self.mesh.Nn
        displacement_matrix = np.zeros((Nn, 3))

        if dof_n == 1:
            pass
        elif 1 < dof_n <= 3:
            coord = self.u.reshape((Nn, -1))
            dim = coord.shape[1]
            displacement_matrix[:, :dim] = coord
        else:
            raise ValueError("Unknown dof_n configuration.")

        return displacement_matrix
