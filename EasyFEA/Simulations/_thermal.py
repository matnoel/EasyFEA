# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import TYPE_CHECKING

# utilities
from ..Utilities import _types

# fem
if TYPE_CHECKING:
    from ..FEM import Mesh
from ..FEM import Operators

# models
from .. import Models

# simu
from ._simu import _Simu
from .Problem._terms import Term
from .Problem._problem_type import ProblemType, TrialSolution


class Thermal(_Simu):
    r"""Heat-transfer (thermal) simulation.

    Solves the steady-state or transient temperature field driven by conduction, a volumetric source and Neumann/Dirichlet conditions, in 1D/2D/3D. Transient analyses use the parabolic time scheme. Material behavior (conductivity ``k`` and heat capacity ``ρc``) is supplied by a thermal model.

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

    T = ProblemType("thermal")
    """the temperature problem"""

    problemTypes = (T,)
    dofFields = {T: ("thermal", "thermalDot")}

    def __init__(
        self, mesh: "Mesh", model: Models.Thermal, folder: str = "", verbosity=False
    ):
        """Creates a thermal simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh used.
        model : IModel
            The model used.
        folder : str, optional
            save folder, by default "".
        verbosity : bool, optional
            If True, the simulation can write in the terminal. Defaults to False.
        """

        assert isinstance(model, Models.Thermal), "model must be a thermal model"
        super().__init__(mesh, model, folder, verbosity)

    def Get_unknowns(self, problemType=None) -> list[str]:
        return ["t"]

    def _Check_dim_mesh_material(self) -> None:
        pass

    def Results_nodeFields_elementFields(
        self, details=False
    ) -> tuple[list[str], list[str]]:
        nodesField = [
            name
            for role, name in zip("uv", self.dofFields[self.T])
            if role in self.timeScheme.roles
        ]
        elementsField: list[str] = []
        return nodesField, elementsField

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

    @property
    def thickness(self) -> float:
        """The model thickness on a 2D mesh, 1 otherwise."""
        return self.thermalModel.thickness if self.mesh.dim == 2 else 1.0

    def Get_terms(self, x: TrialSolution, problemType=None) -> list[Term]:
        thermalModel = self.thermalModel

        return [
            # conductivity
            Term(
                "K", Operators.Bilinear.GradUGradV, coef=thermalModel.k, constant=True
            ),
            # reaction
            Term(
                "C",
                Operators.Bilinear.UV,
                coef=self.rho * thermalModel.c,
                dof_n=1,
                constant=True,
            ),
        ]
