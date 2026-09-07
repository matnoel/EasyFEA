# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Union, Optional, TYPE_CHECKING
import numpy as np

# utilities
from ..Utilities import Terminal, Tic, _types

# fem
if TYPE_CHECKING:
    from ..FEM import Mesh
from ..FEM import _GroupElem, _Form

# models
from .. import Models

# simu
from ._simu import _Simu
from ._terms import Term
from .Solvers import AlgoType
from ._problem_type import ProblemType


class WeakForms(_Simu):
    r"""Generic weak-form simulation.

    Assembles and solves a user-defined variational problem given directly as weak-form terms (bilinear and linear operators over the mesh), rather than a fixed physics. Supports static, transient and dynamic time schemes for linear or nonlinear (Newton-Raphson) problems — useful for custom PDEs and rapid prototyping.
    """

    def __init__(
        self,
        mesh: "Mesh",
        model: Models.WeakForms,
        folder: str = "",
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
        folder : str, optional
            save folder, by default "".
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
        super().__init__(mesh, model, folder, verbosity)

        if isNonLinear:
            self._Solver_Set_Newton_Raphson_Algorithm(absTol=tolConv, maxIter=maxIter)

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

    def Get_problemTypes(self) -> list[ProblemType]:
        return [ProblemType("weakForm")]

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

    @property
    def thickness(self) -> float:
        """The weak form's own thickness, keyed on the **ambient** dimension: a form written on a surface embedded in 3D integrates over that surface already, so nothing is left to scale."""
        return 1.0 if self.mesh.inDim == 3 else self.weakForms.thickness

    def Get_terms(self, problemType=None) -> list[Term]:
        """One term per form the user supplied; the fold applies :py:attr:`thickness` to each.

        The K/C/M forms fill a single slot, so in a nonlinear simulation the fold also contracts their residual ``-K·u_t`` / ``-C·v_t`` / ``-M·a_t``. ``computeF`` is a load, hence the ``F`` slot.
        """
        weakForms = self.weakForms

        forms = {
            "K": weakForms.computeK,
            "C": weakForms.computeC,
            "M": weakForms.computeM,
            "F": weakForms.computeF,
        }

        return [
            Term(slot, self.__Integrate, form=form)
            for slot, form in forms.items()
            if form is not None
        ]

    def __Integrate(self, groupElem: _GroupElem, form: _Form) -> Optional[np.ndarray]:
        """Integrates one weak form over a group.

        The form is written against ``weakForms.field``, which is bound to one element group, so any other group of the same dimension contributes nothing — returning None rather than repeating the same values, which is what a mesh carrying several groups of one dimension (PRISM18 + HEXA27) would otherwise get.
        """
        field = self.weakForms.field
        if groupElem is not field.groupElem:
            return None

        tic = Tic()
        values_e = form.Integrate_e(field)
        tic.Tac("Matrix", "Integrate the weak form.", self._verbosity)
        return values_e

    def Save_Iter(self, iter=None):

        if iter is None:
            iter = {}

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

        super().Save_Iter(iter)

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

        else:
            Terminal.MyPrintError(f"The result '{result}' is not implemented yet.")
            return None  # type: ignore [return-value]

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
