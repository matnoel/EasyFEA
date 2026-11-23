# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import TYPE_CHECKING
import numpy as np

# utilities
from ..Utilities import Tic, _types

# fem
if TYPE_CHECKING:
    from ..FEM import Mesh
from ..FEM import MatrixType, FeArray, Kinematics, Operators

# models
from ._utils import (
    Result_strain_or_stress_field_e,
    Stress_names,
    Strain_names,
    _Field_per_groupElem,
)
from ..Models.Elastic._laws import _Elastic

# simu
from ._simu import _Simu, UnknownValues
from .Problem._terms import Term
from .Problem._problem_type import ProblemType
from .TimeSchemes import Hyperbolic


class Elastic(_Simu):
    r"""Linear (small-strain) elasticity simulation.

    Solves the static, quasi-static or dynamic equilibrium of a linear-elastic body under body forces, surface tractions and Dirichlet conditions, in 1D/2D/3D. Transient and dynamic analyses use the available time schemes (Newmark, HHT, midpoint, …) and support Rayleigh damping. Material behavior is supplied by an elastic model (isotropic, anisotropic, …).

    Strong form:

    .. math::
        \diver{\Sig(\ub)} + \fb &= \rho \, \ddot{\ub} && \quad \text{in } \Omega, \\
        % 
        \Sig(\ub) \cdot \nb &= \tb && \quad \text{on } \partial\Omega_t, \\
        %
        \Sig(\ub) &= \Cbb : \Eps(\ub) && \quad \text{in } \Omega, \\
        % 
        \ub &= \ub_D && \quad \text{on } \partial\Omega_u,

    Weak form:

    .. math::
        \int_\Omega \Sig(\ub) : \Eps(\vb) \, \dO + \int_\Omega \rho \, \ddot{\ub} \cdot \vb \, \dO =
        \int _{\partial\Omega_t} \tb\cdot\vb \, \dS + \int _{\Omega} \fb\cdot\vb \, \dO \quad \forall \, \vb \in V
    
    The implemented elastic laws are available :ref:`here <models-elastic>`.
    """

    U = ProblemType("displacement")
    """the displacement problem"""

    problemTypes = (U,)
    dofFields = {U: ("displacement", "speed", "accel")}

    def __init__(
        self, mesh: "Mesh", model: _Elastic, folder: str = "", verbosity=False
    ):
        """Creates a elastic simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh used.
        model : _Elas
            The elastic model (or material) used.
        folder : str, optional
            save folder, by default "".
        verbosity : bool, optional
            If True, the simulation can write in the terminal. Defaults to False.
        """

        assert isinstance(model, _Elastic), "model must be a elastic model"
        super().__init__(mesh, model, folder, verbosity)

        # init
        self.Set_Rayleigh_Damping_Coefs()

    def Results_nodeFields_elementFields(
        self, details=False
    ) -> tuple[list[str], list[str]]:
        nodesField = ["displacement"]
        if details:
            elementsField = ["Svm", "Stress", "Strain"]
        else:
            elementsField = ["Svm", "Stress"]
        if isinstance(self.timeScheme, Hyperbolic._Hyperbolic):
            nodesField.extend(["speed", "accel"])
        return nodesField, elementsField

    def Get_unknowns(self, problemType=None) -> list[str]:
        dict_unknowns = {2: ["x", "y"], 3: ["x", "y", "z"]}
        return dict_unknowns[self.dim]

    @property
    def material(self) -> _Elastic:
        """elastic material"""
        return self.model  # type: ignore

    @property
    def displacement(self) -> _types.FloatArray:
        """Displacement vector field.\n
        2D [uxi, uyi, ...]\n
        3D [uxi, uyi, uzi, ...]"""
        return self._Get_u_n(self.problemType)

    @property
    def speed(self) -> _types.FloatArray:
        """Velocity vector field.\n
        2D [vxi, vyi, ...]\n
        3D [vxi, vyi, vzi, ...]"""
        return self._Get_v_n(self.problemType)

    @property
    def accel(self) -> _types.FloatArray:
        """Acceleration vector field.\n
        2D [axi, ayi, ...]\n
        3D [axi, ayi, azi, ...]"""
        return self._Get_a_n(self.problemType)

    def Get_terms(self, x: UnknownValues, problemType=None) -> list[Term]:

        stiffness = Term(
            "K",
            Operators.Bilinear.LinearizedElasticity,
            C=self.material.C,
            constant=True,
        )
        mass = Term(
            "M",
            Operators.Bilinear.UV,
            coef=self.rho,
            dof_n=self.dim,
            constant=True,
        )
        terms = [stiffness, mass]

        # Rayleigh damping C = coefK·K + coefM·M: the same integrations, scaled into C — so an undamped simulation (the default) never builds it at all.
        if self.__coefK != 0.0:
            terms.append(stiffness.Scaled(self.__coefK, "C"))
        if self.__coefM != 0.0:
            terms.append(mass.Scaled(self.__coefM, "C"))

        return terms

    def Set_Rayleigh_Damping_Coefs(self, coefM=0.0, coefK=0.0):
        r"""Sets damping coefficients \( C = coefK * K + coefM * M \)."""
        self.__coefM = coefM
        self.__coefK = coefK
        self.Need_Update()

    def Results_Available(self) -> list[str]:
        return [
            *super().Results_Available(),
            *Stress_names(self.dim),
            *Strain_names(self.dim),
            "Wdef",
            "Wdef_e",
        ]

    def _Result(
        self, result: str, nodeValues: bool = True
    ) -> _types.FloatArray | float:
        if result in Stress_names(self.dim):
            values_e = Result_strain_or_stress_field_e(self._Calc_Sigma(), result)
        elif result in Strain_names(self.dim):
            values_e = Result_strain_or_stress_field_e(self._Calc_Epsilon(), result)
        elif result == "Wdef":
            return float(self.mesh._Sum_on_ownedElements_e(self._Calc_Psi_Elas_e()))
        elif result == "Wdef_e":
            values_e = np.concatenate(self._Calc_Psi_Elas_e())
        else:
            return super()._Result(result, nodeValues)
        return self.Results_Reshape_values_e(values_e, nodeValues)

    def _Calc_Psi_Elas_e(self) -> list[_types.FloatArray]:
        r"""Strain energy of each element, one array per group: :math:`\int_e \psi \, \dO` with :math:`\psi = \frac{1}{2} \Sig : \Eps`."""

        tic = Tic()

        thickness = self.material.thickness if self.dim == 2 else 1
        matrixType = MatrixType.rigi

        space = self._Get_FunctionSpace()
        u = space.Get_nodeValues(self.displacement, self.dim)
        list_Wdef_e = [
            (
                thickness
                * groupElem.Get_weightedJacobian_e_pg(matrixType)
                * self.material.Compute_Psi(Kinematics(groupElem, u, matrixType))
            ).sum(1)
            for groupElem in space.Get_list_groupElem(self.dim)
        ]

        tic.Tac("PostProcessing", "Calc Psi Elas", False)

        return list_Wdef_e

    def _Calc_Epsilon(
        self, matrixType: MatrixType = MatrixType.rigi
    ) -> FeArray.FeArrayALike | dict:
        """Strain (Ne, pg, 3 or 6): an ``FeArray`` on one group, ``{groupElem: FeArray}`` on several."""
        space = self._Get_FunctionSpace()
        u = space.Get_nodeValues(self.displacement, self.dim)
        return _Field_per_groupElem(
            lambda groupElem: Kinematics(groupElem, u, matrixType).Compute_Epsilon(),
            space.Get_list_groupElem(),
        )

    def _Calc_Sigma(
        self, matrixType: MatrixType = MatrixType.rigi
    ) -> FeArray.FeArrayALike | dict:
        """Stress (Ne, pg, 3 or 6): an ``FeArray`` on one group, ``{groupElem: FeArray}`` on several."""
        space = self._Get_FunctionSpace()
        u = space.Get_nodeValues(self.displacement, self.dim)
        return _Field_per_groupElem(
            lambda groupElem: self.material.Compute_Sigma(
                Kinematics(groupElem, u, matrixType)
            ),
            space.Get_list_groupElem(),
        )

    def Results_dict_Energy(self) -> dict[str, float]:
        dict_energy = {r"$\Psi_{elas}$": float(self._Result("Wdef"))}
        return dict_energy

    def Results_Get_Iteration_Summary(self) -> str:
        summary = ""

        if not self._Results_Check_Available("Wdef"):
            return None  # type: ignore [return-value]

        # this rank's elements alone: the summary is printed by rank 0 only
        Wdef = self.Result("Wdef_e", nodeValues=False).sum()  # type: ignore [union-attr]
        summary += f"\nW def = {Wdef:.2f}"

        Svm = self.Result("Svm", nodeValues=False)
        summary += f"\n\nSvm max = {Svm.max():.2f}"  # type: ignore [union-attr]

        Evm = self.Result("Evm", nodeValues=False)
        summary += f"\n\nEvm max = {Evm.max() * 100:3.2f} %"  # type: ignore [union-attr]

        dx = self.Result("ux", nodeValues=True)
        summary += f"\n\nUx max = {dx.max():.2e}"  # type: ignore [union-attr]
        summary += f"\nUx min = {dx.min():.2e}"  # type: ignore [union-attr]

        dy = self.Result("uy", nodeValues=True)
        summary += f"\n\nUy max = {dy.max():.2e}"  # type: ignore [union-attr]
        summary += f"\nUy min = {dy.min():.2e}"  # type: ignore [union-attr]

        if self.dim == 3:
            dz = self.Result("uz", nodeValues=True)
            summary += f"\n\nUz max = {dz.max():.2e}"  # type: ignore [union-attr]
            summary += f"\nUz min = {dz.min():.2e}"  # type: ignore [union-attr]

        return summary

    def Results_displacement_matrix(self) -> _types.FloatArray:
        Nn = self.mesh.Nn
        space = self._Get_FunctionSpace()
        coord = space.Get_meshValues(self.displacement, self.dim)
        dim = coord.shape[1]

        displacement_matrix = np.zeros((Nn, 3))
        displacement_matrix[:, :dim] = coord

        return displacement_matrix
