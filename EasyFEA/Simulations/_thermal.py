# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Union, Optional, TYPE_CHECKING
import numpy as np

# utilities
from ..Utilities import _types

# fem
if TYPE_CHECKING:
    from ..FEM import Mesh
from ..FEM import MatrixType

# models
from .. import Models
from ..Models import ModelType, Reshape_variable

# simu
from ._simu import _Simu
from .Solvers import AlgoType


class Thermal(_Simu):
    r"""Thermal simulation.
    
    Strong form:

    .. math::
        \diver q + \rho \, c \, \dot{t} &= r && \quad \text{in } \Omega, \\
        % 
        q &= -k \, \nabla t && \quad \text{in } \Omega, \\
        % 
        q \cdot \nb &= -q_n && \quad \text{on } \partial\Omega_q, \\
        % 
        t &= t_D && \quad \text{on } \partial\Omega_t,

    Weak form:

    .. math::
        \int_\Omega k \, \nabla t \cdot \nabla \delta t \, \dO +
        \int_\Omega \rho \, c \, \dot{t} \, \delta t \, \dO =
        \int_{\partial\Omega_q} q_n \, \delta t \, \dS +
        \int_{\partial\Omega_t} r \, \delta t \, \dO \quad \forall \, \delta t \in V

    """

    def __init__(
        self,
        mesh: "Mesh",
        model: Models.Thermal,
        verbosity=False,
    ):
        """Creates a thermal simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh used.
        model : IModel
            The model used.
        verbosity : bool, optional
            If True, the simulation can write in the terminal. Defaults to False.
        """

        assert isinstance(model, Models.Thermal), "model must be a thermal model"
        super().__init__(mesh, model, verbosity)

        # Set solver petsc4py options, even if petsc4py is unavailable.
        self._Solver_Set_PETSc4Py_Options(pcType="ilu")

    def Get_unknowns(self, problemType=None) -> list[str]:
        return ["t"]

    def Get_dof_n(self, problemType=None) -> int:
        return 1

    def _Check_dim_mesh_material(self) -> None:
        # In the case of a beam problem, we don't need to check this condition.
        pass

    def Results_nodeFields_elementFields(
        self, details=False
    ) -> tuple[list[str], list[str]]:
        nodesField = ["thermal", "thermalDot"]
        elementsField: list[str] = []
        return nodesField, elementsField

    def Get_problemTypes(self) -> list[ModelType]:
        return [ModelType.thermal]

    @property
    def thermalModel(self) -> Models.Thermal:
        """Thermal simulation model."""
        return self.model  # type: ignore [return-value]

    @property
    def thermal(self) -> _types.FloatArray:
        """Scalar temperature field.\n
        [ti, ....]"""
        return self._Get_u_n(self.problemType)

    @property
    def thermalDot(self) -> _types.FloatArray:
        """Time derivative of the scalar temperature field.\n
        [d(ti)/dt, ....]"""
        return self._Get_v_n(self.problemType)

    def Get_x0(self, problemType=None):
        if self.thermal.size != self.mesh.Nn:
            return np.zeros(self.mesh.Nn)
        else:
            return self.thermal

    def Construct_local_matrix_system(self, problemType):
        thermalModel = self.thermalModel
        mesh = self.mesh

        matrixType = MatrixType.rigi
        wJ_e_pg = mesh.Get_weightedJacobian_e_pg(matrixType)
        dN_e_pg = mesh.Get_dN_e_pg(matrixType)

        # conductivity part
        conductivity = thermalModel.k
        if thermalModel.isHeterogeneous:
            conductivity = Reshape_variable(conductivity, *wJ_e_pg.shape[:2])

        Kt_e = (conductivity * wJ_e_pg * dN_e_pg.T @ dN_e_pg).sum(axis=1)

        # reaction part
        rho = self.rho
        heatCapacity = thermalModel.c

        matrixType = MatrixType.mass
        wJ_e_pg = mesh.Get_weightedJacobian_e_pg(matrixType)
        reactionPart = mesh.Get_ReactionPart_e_pg(matrixType)

        if thermalModel.isHeterogeneous:
            rho = Reshape_variable(rho, *wJ_e_pg.shape[:2])
            heatCapacity = Reshape_variable(heatCapacity, *reactionPart.shape[:2])

        Ct_e = (rho * heatCapacity * reactionPart).sum(axis=1)

        # rescale
        if self.dim == 2:
            thickness = thermalModel.thickness
            Kt_e *= thickness
            Ct_e *= thickness

        return Kt_e, Ct_e, None, None

    def Save_Iter(self):
        iter = super().Save_Iter()

        iter["thermal"] = self.thermal

        if self.algo == AlgoType.parabolic:
            iter["thermalDot"] = self.thermalDot

        self._results.append(iter)

    def Set_Iter(self, iter: int = -1, resetAll=False) -> dict:
        results = super().Set_Iter(iter)

        if results is None:
            return

        u = results["thermal"]

        if self.algo == AlgoType.parabolic and "thermalDot" in results:
            v = results["thermalDot"]
        else:
            v = np.zeros_like(u)

        self._Set_solutions(self.problemType, u, v)

        return results

    def Results_Available(self) -> list[str]:
        options = []
        options.extend(["thermal", "thermalDot", "displacement_matrix"])
        return options

    def Result(
        self, result: str, nodeValues: bool = True, iter: Optional[int] = None
    ) -> Union[_types.FloatArray, float]:
        if iter is not None:
            self.Set_Iter(iter)

        if not self._Results_Check_Available(result):
            return None  # type: ignore [return-value]

        # begin cases ----------------------------------------------------

        if result == "thermal":
            values = self.thermal

        elif result == "thermalDot":
            values = self.thermalDot

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
        return super().Results_displacement_matrix()
