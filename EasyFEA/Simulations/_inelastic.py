# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Optional, Union, TYPE_CHECKING

import numpy as np

from ..Utilities import Terminal, Tic, _types

if TYPE_CHECKING:
    from ..FEM import Mesh
    from ..FEM._utils import ElemType
from ..FEM import MatrixType, FeArray, Operators

from ..Models import ModelType, Result_strain_or_stress_field_e
from ..Models.InElastic._behavior import Behavior

from ._simu import _Simu


class InElastic(_Simu):
    r"""Quasi-static mechanics with a history-dependent material.

    Solves :math:`\diver{\Sig} + \fb = 0` by Newton-Raphson, where the stress comes from
    :meth:`~EasyFEA.Models.Behavior.Integrate` rather than from a fixed :math:`\Crm`. The
    material owns the constitutive integration; this class owns the equilibrium iteration and
    the internal-variable history.

    The state is committed in :meth:`Save_Iter` and read back in :meth:`Set_Iter`, so every
    integration starts from the **last converged** step rather than from the iterate in
    progress.
    """

    def __init__(
        self,
        mesh: "Mesh",
        model: Behavior,
        folder: str = "",
        verbosity: bool = False,
        absTol: float = 1e-6,
        relTol: float = 1e-10,
        incTol: float = 1e-11,
        maxIter: int = 20,
    ):
        """Creates a behavior simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh used.
        model : Behavior
            The material.
        folder : str, optional
            save folder, by default "".
        verbosity : bool, optional
            If True, the simulation can write in the terminal. Defaults to False.
        absTol, relTol, incTol : float, optional
            Newton-Raphson tolerances.
        maxIter : int, optional
            maximum Newton iterations, by default 20.
        """
        assert isinstance(model, Behavior), "model must be a Behavior"
        super().__init__(mesh, model, folder, verbosity)

        self._Solver_Set_Newton_Raphson_Algorithm(absTol, relTol, incTol, maxIter)

        self.__dt = 0.0
        self.__z: dict["ElemType", FeArray] = {}
        self.__zOld: dict["ElemType", FeArray] = {}

    @property
    def dt(self) -> float:
        """Time increment, read by a rate-dependent material."""
        return self.__dt

    @dt.setter
    def dt(self, value: float) -> None:
        assert value >= 0.0, "dt must be >= 0"
        self.__dt = value

    @property
    def material(self) -> Behavior:
        """The material."""
        return self.model  # type: ignore

    @property
    def displacement(self) -> _types.FloatArray:
        """Displacement vector field.\n
        2D [uxi, uyi, ...]\n
        3D [uxi, uyi, uzi, ...]"""
        return self._Get_u_n(self.problemType)

    def Get_unknowns(self, problemType=None) -> list[str]:
        return {2: ["x", "y"], 3: ["x", "y", "z"]}[self.dim]

    def Get_problemTypes(self) -> list[ModelType]:
        # a displacement problem, structurally identical to Simulations.Elastic
        return [ModelType.elastic]

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
    # Assembly
    # --------------------------------------------------------------------------

    def __Get_state(self, groupElem, matrixType: MatrixType) -> FeArray:
        """Committed state for a group, zeros if none yet."""
        elemType = groupElem.elemType
        if elemType not in self.__zOld:
            nPg = groupElem.Get_gauss(matrixType).nPg
            self.__zOld[elemType] = self.material.State_zeros(groupElem.Ne, nPg)
        return self.__zOld[elemType]

    def _Calc_Epsilon_e_pg(
        self,
        u: _types.FloatArray,
        groupElem,
        matrixType: MatrixType = MatrixType.rigi,
    ) -> FeArray.FeArrayALike:
        """Total strain field from the displacement vector field."""
        u_e = groupElem.Locates_sol_e(u, asFeArray=True)
        return groupElem.Get_B_e_pg(matrixType) @ u_e

    def Construct_local_matrix_system(
        self, problemType, matrixType: MatrixType = MatrixType.rigi
    ):
        """Per group: integrate the material, then assemble ``K_e = ∫BᵀC_alg B`` and
        ``F_e = -∫Bᵀσ``. The trial state is stashed, and committed only in :meth:`Save_Iter`.
        """
        thickness = self.material.thickness if self.dim == 2 else 1.0
        u = self._Solver_Get_Newton_Raphson_current_solution()

        out = {}
        for groupElem in self.mesh.Get_list_groupElem():
            eps_e_pg = self._Calc_Epsilon_e_pg(u, groupElem, matrixType)
            zOld_e_pg = self.__Get_state(groupElem, matrixType)
            # u_n is only overwritten once the Newton converges, so during assembly it is
            # still the displacement the increment started from
            epsOld_e_pg = self._Calc_Epsilon_e_pg(
                self._Get_u_n(self.problemType), groupElem, matrixType
            )

            sigma_e_pg, C_e_pg, z_e_pg, converged = self.material.Integrate(
                eps_e_pg, zOld_e_pg, self.__dt, epsOld_e_pg
            )
            assert C_e_pg is not None
            assert converged.all(), (
                f"constitutive integration did not converge at {int((~converged).sum())} of "
                f"{converged.size} Gauss points ({groupElem.elemType}) - reduce the load step"
            )

            tic = Tic()
            K_e = thickness * Operators.Bilinear.LinearizedElasticity(
                groupElem, C_e_pg, matrixType
            )
            F_e = -thickness * Operators.Linear.InternalForce(
                groupElem, sigma_e_pg, matrixType
            )
            tic.Tac("Matrix", f"Construct K_e and F_e ({groupElem.elemType})", False)

            self.__z[groupElem.elemType] = z_e_pg
            out[groupElem] = (K_e, None, None, F_e)

        return out

    # --------------------------------------------------------------------------
    # Iterations
    # --------------------------------------------------------------------------

    def Save_Iter(self, iter=None):
        if iter is None:
            iter = {}

        iter["displacement"] = self.displacement

        # the trial state becomes the committed one
        self.__zOld = {et: arr.copy() for et, arr in self.__z.items()}
        # copied again, so a later in-place write cannot reach the saved history
        iter["state"] = {et: arr.copy() for et, arr in self.__zOld.items()}

        return super().Save_Iter(iter)

    def Set_Iter(self, iter: int = -1, resetAll=False) -> dict:
        results = super().Set_Iter(iter)
        if results is None:
            return results

        u = results["displacement"]
        self._Set_solutions(self.problemType, u, np.zeros_like(u), np.zeros_like(u))

        self.__zOld = {et: a.copy() for et, a in results.get("state", {}).items()}
        self.__z = {et: a.copy() for et, a in self.__zOld.items()}

        return results

    # --------------------------------------------------------------------------
    # Results
    # --------------------------------------------------------------------------

    def Results_Available(self) -> list[str]:
        results = ["displacement", "displacement_norm", "displacement_matrix"]
        if self.dim == 2:
            results.extend(["ux", "uy"])
            results.extend(["Sxx", "Syy", "Sxy", "Exx", "Eyy", "Exy"])
        else:
            results.extend(["ux", "uy", "uz"])
            results.extend(["Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy"])
            results.extend(["Exx", "Eyy", "Ezz", "Eyz", "Exz", "Exy"])
        results.extend(["Svm", "Stress", "Evm", "Strain"])
        # scalar internal variables are plottable: alpha, d, ...
        results.extend(
            name
            for name, slot in self.material.layout.slots.items()
            if slot.stop - slot.start == 1
        )
        return results

    def __Result_state(self, name: str) -> _types.FloatArray:
        """Per-element value of a scalar internal variable, averaged over its Gauss points."""
        slot = self.material.layout.slots[name]
        assert (
            slot.stop - slot.start == 1
        ), f"'{name}' is a tensor internal variable; read it from Save_Iter's state instead"

        values = [
            np.mean(
                self.__Get_state(groupElem, MatrixType.rigi)[..., slot.start], axis=1
            )
            for groupElem in self.mesh.Get_list_groupElem()
            if groupElem.dim == self.dim
        ]
        return np.concatenate(values)

    def __indexResult(self, result: str) -> int:
        return {"x": 0, "y": 1, "z": 2}[result[-1]]

    def Result(
        self, result: str, nodeValues: bool = True, iter: Optional[int] = None
    ) -> Union[_types.FloatArray, float]:
        if iter is not None:
            self.Set_Iter(iter)

        if not self._Results_Check_Available(result):
            return None  # type: ignore [return-value]

        Nn = self.mesh.Nn
        u = self.displacement

        if result in ["ux", "uy", "uz"]:
            values = u.reshape(Nn, -1)[:, self.__indexResult(result)]

        elif result == "displacement":
            values = u

        elif result == "displacement_norm":
            values = np.linalg.norm(u.reshape(Nn, -1), axis=1)

        elif result == "displacement_matrix":
            values = self.Results_displacement_matrix()

        elif result in self.material.layout.slots:
            values = self.__Result_state(result)

        elif ("S" in result or "E" in result) and "_norm" not in result:
            isStress = "S" in result and result != "Strain"
            res = result if result in ["Strain", "Stress"] else result[-2:]

            def field_e_pg(groupElem):
                eps_e_pg = self._Calc_Epsilon_e_pg(u, groupElem)
                if not isStress:
                    return eps_e_pg
                # read the committed state, and only read it: Integrate would advance a
                # rate-dependent material by another dt every time a result was asked for
                zOld_e_pg = self.__Get_state(groupElem, MatrixType.rigi)
                return self.material.Compute_stress(eps_e_pg, zOld_e_pg)

            values = Result_strain_or_stress_field_e(
                field_e_pg=field_e_pg,
                list_groupElem=self.mesh.Get_list_groupElem(),
                result=res,
                coef=self.material.coef,
            )

        else:
            Terminal.MyPrintError(f"The result '{result}' is not implemented yet.")
            return None  # type: ignore [return-value]

        return self.Results_Reshape_values(values, nodeValues)

    def Results_Iter_Summary(
        self,
    ) -> tuple[list[int], list[tuple[str, _types.FloatArray]]]:
        return super().Results_Iter_Summary()

    def _Calc_psi(self) -> float:
        """Stored free energy over the domain, ``∫ psi dΩ``."""
        u = self.displacement
        thickness = self.material.thickness if self.dim == 2 else 1.0
        # same matrixType as assembly, so the cached state has the matching nPg
        matrixType = MatrixType.rigi

        total = 0.0
        for groupElem in self.mesh.Get_list_groupElem():
            if groupElem.dim != self.dim:
                continue
            eps_e_pg = self._Calc_Epsilon_e_pg(u, groupElem, matrixType)
            eps6_e_pg = self.material.Compute_strain_6d(
                eps_e_pg, self.__Get_state(groupElem, matrixType), self.__dt
            )
            psi_e_pg = self.material.Compute_psi(
                eps6_e_pg, self.__Get_state(groupElem, matrixType)
            )
            wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)
            total += float(np.sum(wJ_e_pg * psi_e_pg))

        return thickness * total

    def Results_dict_Energy(self) -> dict[str, float]:
        return {r"$\Psi$": self._Calc_psi()}

    def Results_displacement_matrix(self) -> _types.FloatArray:
        Nn = self.mesh.Nn
        coord = self.displacement.reshape((Nn, -1))
        matrix = np.zeros((Nn, 3))
        matrix[:, : coord.shape[1]] = coord
        return matrix
