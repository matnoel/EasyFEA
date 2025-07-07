# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Union, Optional
import numpy as np
from scipy import sparse
import pandas as pd

# utilities
from ..utilities import Display, Tic, _types
from ..utilities._observers import Observable

# fem
from ..fem import Mesh, MatrixType, FeArray

# materials
from .. import Models
from ..models import (
    ModelType,
    _IModel,
    Reshape_variable,
    Result_in_Strain_or_Stress_field,
)

# simu
from ._simu import _Simu


class PhaseFieldSimu(_Simu):
    def __init__(
        self,
        mesh: Mesh,
        model: Models.PhaseField,
        verbosity=False,
        useNumba=True,
        useIterativeSolvers=True,
    ):
        """Creates a damage simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh used.
        model : PhaseField
            The model used.
        verbosity : bool, optional
            If True, the simulation can write in the terminal. Defaults to False.
        useNumba : bool, optional
            If True, numba can be used. Defaults to True.
        useIterativeSolvers : bool, optional
            If True, iterative solvers can be used. Defaults to True.
        """

        assert isinstance(model, Models.PhaseField), "model must be a phase field model"
        super().__init__(mesh, model, verbosity, useNumba, useIterativeSolvers)

        # Init internal variable
        self.__psiP_e_pg: FeArray.FeArrayALike = np.empty(0, dtype=float)
        # old positive elastic energy density psiPlus(e, pg, 1) to use the miehe history field
        self.__old_psiP_e_pg: FeArray.FeArrayALike = np.empty(0, dtype=float)

        self.Solver_Set_Elliptic_Algorithm()

        self.Need_Update()

        self.phaseFieldModel.material._Add_observer(self)

        self.__resumeLoading = ""

    def Results_nodeFields_elementFields(
        self, details=False
    ) -> tuple[list[str], list[str]]:
        nodesField = ["displacement", "damage"]
        if details:
            elementsField = ["Svm", "Stress", "Strain", "psiP"]
        else:
            elementsField = ["Svm", "Stress"]
        return nodesField, elementsField

    def Get_unknowns(self, problemType=None) -> list[str]:
        if problemType == ModelType.damage:
            return ["d"]
        elif problemType in [ModelType.elastic, None]:
            _dict_unknowns = {2: ["x", "y"], 3: ["x", "y", "z"]}
            return _dict_unknowns[self.dim]
        else:
            raise ValueError("problem error")

    def Get_problemTypes(self) -> list[ModelType]:
        return [ModelType.damage, ModelType.elastic]

    def Get_lb_ub(
        self, problemType: ModelType
    ) -> tuple[_types.FloatArray, _types.FloatArray]:
        if problemType == ModelType.damage:
            solveur = self.phaseFieldModel.solver
            if solveur == "BoundConstrain":
                lb = self.damage
                lb[np.where(lb >= 1)] = 1 - np.finfo(float).eps
                ub = np.ones(lb.shape)
            else:
                lb, ub = np.array([]), np.array([])
        else:
            lb, ub = np.array([]), np.array([])

        return lb, ub

    def Get_dof_n(self, problemType=None) -> int:
        if problemType == ModelType.damage:
            return 1
        elif problemType in [ModelType.elastic, None]:
            return self.dim
        else:
            raise ValueError("problem error")

    @property
    def phaseFieldModel(self) -> Models.PhaseField:
        """damage model"""
        return self.model  # type: ignore [return-value]

    @property
    def displacement(self) -> _types.FloatArray:
        """Displacement vector field.\n
        2D [uxi, uyi, ...]\n
        3D [uxi, uyi, uzi, ...]"""
        return self._Get_u_n(ModelType.elastic)

    @property
    def damage(self) -> _types.FloatArray:
        """Damage scalar field.\n
        [di, ...]"""
        return self._Get_u_n(ModelType.damage)

    def Bc_dofs_nodes(
        self,
        nodes: _types.IntArray,
        unknowns: list[str],
        problemType=ModelType.elastic,
    ) -> _types.IntArray:
        return super().Bc_dofs_nodes(nodes, unknowns, problemType)

    def add_dirichlet(
        self,
        nodes: _types.IntArray,
        values: list,
        unknowns: list[str],
        problemType=ModelType.elastic,
        description="",
    ):
        return super().add_dirichlet(nodes, values, unknowns, problemType, description)

    def add_lineLoad(
        self,
        nodes: _types.IntArray,
        values: list,
        unknowns: list[str],
        problemType=ModelType.elastic,
        description="",
    ):
        return super().add_lineLoad(nodes, values, unknowns, problemType, description)

    def add_surfLoad(
        self,
        nodes: _types.IntArray,
        values: list,
        unknowns: list[str],
        problemType=ModelType.elastic,
        description="",
    ):
        return super().add_surfLoad(nodes, values, unknowns, problemType, description)

    def add_pressureLoad(
        self,
        nodes: _types.IntArray,
        magnitude: float,
        problemType=ModelType.elastic,
        description="",
    ) -> None:
        return super().add_pressureLoad(nodes, magnitude, problemType, description)

    def add_neumann(
        self,
        nodes: _types.IntArray,
        values: list,
        unknowns: list[str],
        problemType=ModelType.elastic,
        description="",
    ):
        return super().add_neumann(nodes, values, unknowns, problemType, description)

    def Get_K_C_M_F(
        self, problemType=None
    ) -> tuple[
        sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix
    ]:
        if problemType is None:
            problemType = ModelType.elastic

        # here always update to the last state
        if problemType == ModelType.elastic:
            if not self.__updatedDisplacement:  # type: ignore [has-type]
                self.__Assembly_elastic()
                self.__updatedDisplacement = True
            size = self.__Ku.shape[0]
            initcsr = sparse.csr_matrix((size, size))
            return self.__Ku.copy(), initcsr, initcsr, self.__Fu.copy()
        else:
            if not self.__updatedDamage:  # type: ignore [has-type]
                self.__Assembly_damage()
                self.__updatedDamage = True
            size = self.__Kd.shape[0]
            initcsr = sparse.csr_matrix((size, size))
            return self.__Kd.copy(), initcsr, initcsr, self.__Fd.copy()

    def _Update(self, observable: Observable, event: str) -> None:
        if isinstance(observable, _IModel):
            if event == "The model has been modified" and not self.needUpdate:
                self.Need_Update()
        elif isinstance(observable, Mesh):
            if event == "The mesh has been modified":
                self._Check_dim_mesh_material()
                self.Need_Update()
        else:
            Display.MyPrintError("Notification not yet implemented")

    @property
    def needUpdate(self) -> bool:
        return not self.__updatedDamage or not self.__updatedDisplacement

    def Need_Update(self, value=True) -> None:
        # the following functions help you avoid having to assemble matrices too often
        self.__updatedDamage = not value
        """The matrix system associated with the damage problem is updated."""
        self.__updatedDisplacement = not value
        """The matrix system associated with the displacement problem is updated."""

    def Get_x0(self, problemType=None):
        if problemType == ModelType.damage:
            if self.damage.size != self.mesh.Nn:
                return np.zeros(self.mesh.Nn)
            else:
                return self.damage
        elif problemType in [ModelType.elastic, None]:
            if self.displacement.size != self.mesh.Nn * self.dim:
                return np.zeros(self.mesh.Nn * self.dim)
            else:
                return self.displacement

    def Assembly(self) -> None:
        self.__Assembly_damage()
        self.__Assembly_elastic()

    def Solve(
        self, tolConv=1.0, maxIter=500, convOption=2
    ) -> tuple[_types.FloatArray, _types.FloatArray, sparse.csr_matrix, bool]:
        """Solves the iterative damage problem using the staggered scheme.

        Parameters
        ----------
        tolConv : float, optional
            threshold used to check convergence (𝜖), by default 1.0
        maxIter : int, optional
            Maximum iterations for convergence, by default 500
        convOption : int, optional
            - 0 -> convergence on damage np.max(np.abs(d_np1-dk)) equivalent normInf(d_np1-dk)  \n
            - 1 -> convergence on crack energy np.abs(psi_crack_n - psi_crack_np1)/psi_crack_np1    \n
            - 2 -> convergence on total energy np.abs(psi_tot_n - psi_tot_np1)/psi_tot_np1  \n
            eq (39) Ambati 2015 10.1007/s00466-014-1109-y   \n
            - 3 -> (convD <= tolConv) and (convU <= tolConv*0.999)  \n
            eq (25) Pech 2022 10.1016/j.engfracmech.2022.108591

        Returns
        -------
        _types.FloatArray, _types.FloatArray, csr_matrix, bool
            u_np1, d_np1, Ku, converged

            such that:\n
            - u_np1: displacement vector field\n
            - d_np1: damage scalar field\n
            - Ku: displacement stiffness matrix\n
            - converged: the solution has converged\n
        """

        assert tolConv > 0 and tolConv <= 1, "tolConv must be between 0 and 1."
        assert maxIter > 1, "Must be > 1."

        Niter = 0
        converged = False
        old_damage = self.damage

        solver = self.phaseFieldModel.solver

        if convOption == 2:
            fu = self._Solver_Apply_Neumann(ModelType.elastic).toarray().ravel()
            # A vector of zeros when no external body and surface forces are applied.

        tic = Tic()

        while not converged and Niter < maxIter:
            Niter += 1

            d_n = self.damage
            u_n = self.displacement

            if convOption == 1:
                E_n = self._Calc_Psi_Crack()

            elif convOption == 2:
                # eq (39) Ambati 2015 10.1007/s00466-014-1109-y
                # The work of external body and surface forces are added to remain as general as possible.
                E_n = (
                    self._Calc_Psi_Crack()
                    + self._Calc_Psi_Elas()
                    - self._Calc_Psi_Ext(fu)
                )

            # Compute damage field
            d_np1 = self.__Solve_damage()
            # new damage -> new displacement matrices
            self.__updatedDisplacement = False

            # Compute displacement field
            u_np1 = self.__Solve_elastic()
            # new displacement -> new damage matrices
            self.__updatedDamage = False

            if convOption == 0:
                convIter = np.max(np.abs(d_np1 - d_n))

            elif convOption in [1, 2]:
                E_np1 = self._Calc_Psi_Crack()
                if convOption == 2:
                    E_np1 += self._Calc_Psi_Elas() - self._Calc_Psi_Ext(fu)

                if E_np1 == 0:
                    convIter = np.abs(E_n - E_np1)
                else:
                    convIter = np.abs((E_n - E_np1) / E_np1)

            elif convOption == 3:
                # eq (25) Pech 2022 10.1016/j.engfracmech.2022.108591
                diffU = np.abs(u_np1 - u_n)
                diffU[u_np1 != 0] *= 1 / np.abs(u_np1[u_np1 != 0])
                diffD = np.abs(d_np1 - d_n)
                diffD[d_np1 != 0] *= 1 / np.abs(d_np1[d_np1 != 0])
                convU = np.sum(diffU)
                convD = np.sum(diffD)
                convIter = np.max([convD, convU])

            # check convergence
            if tolConv == 1 or d_np1.max() == 0:
                converged = True
            elif convOption == 3:
                converged = (convD <= tolConv) and (convU <= tolConv * 0.999)
            else:
                converged = convIter <= tolConv

        solverTypes = Models.PhaseField.SolverType

        if solver in [solverTypes.History, solverTypes.BoundConstrain]:
            d_np1 = d_np1

        elif solver == solverTypes.HistoryDamage:
            oldAndNewDamage = np.zeros((d_np1.shape[0], 2))
            oldAndNewDamage[:, 0] = old_damage
            oldAndNewDamage[:, 1] = d_np1
            d_np1 = np.max(oldAndNewDamage, 1)

        else:
            raise Exception("Unknown phase field solver.")

        timeIter = tic.Tac("Resolution phase field", "Phase Field iteration", False)

        # save solve config
        self.__tolConv = tolConv
        self.__convOption = convOption
        self.__maxIter = maxIter
        # save iter parameters
        self.__Niter = Niter
        self.__convIter = convIter
        self.__timeIter = timeIter

        Ku = self.__Ku.copy()

        return u_np1, d_np1, Ku, converged

    # ------------------------------------------- Elastic problem -------------------------------------------

    def __Construct_Elastic_Matrix(self) -> _types.FloatArray:
        """Computes the elementary stiffness matrices for the elastic problem."""

        matrixType = MatrixType.rigi

        # Data
        mesh = self.mesh

        B_dep_e_pg = mesh.Get_B_e_pg(matrixType)
        leftDepPart = mesh.Get_leftDispPart(
            matrixType
        )  # -> jacobian_e_pg * weight_pg * B_dep_e_pg'

        d = self.damage
        u = self.displacement

        phaseFieldModel = self.phaseFieldModel

        # compute strain field
        Epsilon_e_pg = self._Calc_Epsilon_e_pg(u, matrixType)

        # compute the splited stifness matrices for the given strain field.
        cP_e_pg, cM_e_pg = phaseFieldModel.Calc_C(Epsilon_e_pg)

        tic = Tic()

        # compute c such that: c = g(d) * cP + cM
        g_e_pg = phaseFieldModel.Get_g_e_pg(d, mesh, matrixType)
        cP_e_pg = g_e_pg * cP_e_pg

        c_e_pg = cP_e_pg + cM_e_pg

        # stiffness matrix for each element
        Ku_e = np.sum(leftDepPart @ c_e_pg @ B_dep_e_pg, axis=1)

        if self.dim == 2:
            thickness = self.phaseFieldModel.thickness
            Ku_e *= thickness

        tic.Tac("Matrix", "Construction Ku_e", self._verbosity)

        return Ku_e

    def __Assembly_elastic(self) -> sparse.csr_matrix:
        """Assemble the elastic problem."""

        # Data
        mesh = self.mesh
        Ndof = mesh.Nn * self.dim

        Ndof += self._Bc_Lagrange_dim(ModelType.elastic)

        Ku_e = self.__Construct_Elastic_Matrix()

        tic = Tic()

        linesVector_e = mesh.rowsVector_e.ravel()
        columnsVector_e = mesh.columnsVector_e.ravel()

        # Assembly
        self.__Ku = sparse.csr_matrix(
            (Ku_e.ravel(), (linesVector_e, columnsVector_e)), shape=(Ndof, Ndof)
        )
        """Kglob matrix for the displacement problem (Ndof, Ndof)"""

        self.__Fu = sparse.csr_matrix((Ndof, 1))
        """Fglob vector for the displacement problem (Ndof, 1)"""

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.spy(self.__Ku)
        # plt.show()

        tic.Tac("Matrix", "Assembly Ku and Fu", self._verbosity)
        # We ensure the matrices are always updated with the latest damage or displacement results.
        # Therefore, we don't specify that the matrices have been updated.
        return self.__Ku

    def __Solve_elastic(self) -> _types.FloatArray:
        """Computes the displacement field."""

        self._Solver_Solve(ModelType.elastic)

        return self.displacement

    # ------------------------------------------- Damage problem -------------------------------------------

    def __Calc_psiPlus_e_pg(self) -> FeArray.FeArrayALike:
        """Computes the positive energy density psi^+ (e, p)."""

        phaseFieldModel = self.phaseFieldModel

        u = self.displacement
        d = self.damage

        testu = isinstance(u, np.ndarray) and (u.shape[0] == self.mesh.Nn * self.dim)
        testd = isinstance(d, np.ndarray) and (d.shape[0] == self.mesh.Nn)

        assert testu or testd, "Dimension problem."

        Epsilon_e_pg = self._Calc_Epsilon_e_pg(u, MatrixType.mass)
        # here the mass term is important otherwise we under-integrate

        # Compute the elastic energy densities.
        psiP_e_pg, _ = phaseFieldModel.Calc_psi_e_pg(Epsilon_e_pg)

        if phaseFieldModel.solver == "History":
            # Get the old history field
            old_psiPlus_e_pg = self.__old_psiP_e_pg.copy()  # type: ignore [union-attr]

            if isinstance(old_psiPlus_e_pg, list) and len(old_psiPlus_e_pg) == 0:
                # No damage available yet
                old_psiPlus_e_pg = np.zeros_like(psiP_e_pg)

            if old_psiPlus_e_pg.shape != psiP_e_pg.shape:
                # the mesh has been changed, the value must be recalculated
                # here do nothing
                old_psiPlus_e_pg = np.zeros_like(psiP_e_pg)

            inc_H = psiP_e_pg - old_psiPlus_e_pg

            elements, gaussPoints = np.where(inc_H < 0)

            psiP_e_pg[elements, gaussPoints] = old_psiPlus_e_pg[elements, gaussPoints]

            # new = np.linalg.norm(psiP_e_pg)
            # old = np.linalg.norm(self.__old_psiP_e_pg)
            # assert new >= old, "Error"

        self.__psiP_e_pg = FeArray.asfearray(psiP_e_pg)

        return self.__psiP_e_pg

    def __Construct_Damage_Matrix(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        """Computes the elementary matrices for the damage problem."""

        pfm = self.phaseFieldModel

        # Data
        PsiP_e_pg = self.__Calc_psiPlus_e_pg()

        matrixType = MatrixType.mass

        mesh = self.mesh
        dN_e_pg = mesh.Get_dN_e_pg(matrixType)

        # K * Laplacien(d) + r * d = F

        tic = Tic()

        # Reaction part Kr_e = r_e_pg * jacobian_e_pg * weight_pg * N_pg' @ N_pg
        ReactionPart_e_pg = mesh.Get_ReactionPart_e_pg(matrixType)
        r_e_pg = pfm.Get_r_e_pg(PsiP_e_pg)
        Kr_e = (r_e_pg * ReactionPart_e_pg).sum(axis=1)

        # Diffusion part Kk_e -> k_e_pg * jacobian_e_pg * weight_pg * dN_e_pg' @ A @ dN_e_pg
        DiffusePart_e_pg = mesh.Get_DiffusePart_e_pg(matrixType)
        k = pfm.k
        A = pfm.A
        if pfm.isHeterogeneous:
            k = Reshape_variable(k, *PsiP_e_pg.shape[:2])
            A = Reshape_variable(A, *PsiP_e_pg.shape[:2])
        Kk_e = (k * DiffusePart_e_pg @ A @ dN_e_pg).sum(axis=1)

        # Source part Fd_e = f_e_pg * jacobian_e_pg * weight_pg * N_pg' @ N_pg
        SourcePart_e_pg = mesh.Get_SourcePart_e_pg(matrixType)
        f_e_pg = pfm.Get_f_e_pg(PsiP_e_pg)
        Fd_e = (f_e_pg * SourcePart_e_pg).sum(axis=1)

        Kd_e = Kr_e + Kk_e

        if self.dim == 2:
            thickness = pfm.thickness
            Kd_e *= thickness
            Fd_e *= thickness

        tic.Tac("Matrix", "Construct Kd_e and Fd_e", self._verbosity)

        return Kd_e, Fd_e

    def __Assembly_damage(self) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Assemble the elastic problem."""

        # Data
        mesh = self.mesh
        Ndof = mesh.Nn
        linesScalar_e = mesh.rowsScalar_e.ravel()
        columnsScalar_e = mesh.columnsScalar_e.ravel()

        # Additional dimension linked to the use of lagrange coefficients
        Ndof += self._Bc_Lagrange_dim(ModelType.damage)

        # Calculate elementary matrix
        Kd_e, Fd_e = self.__Construct_Damage_Matrix()

        # Assembly
        tic = Tic()

        self.__Kd = sparse.csr_matrix(
            (Kd_e.ravel(), (linesScalar_e, columnsScalar_e)), shape=(Ndof, Ndof)
        )
        """Kglob for damage problem (Ndof, Ndof)"""

        rows = mesh.connect.ravel()
        self.__Fd = sparse.csr_matrix(
            (Fd_e.ravel(), (rows, np.zeros_like(rows))), shape=(Ndof, 1)
        )
        """Fglob for damage problem (Ndof, 1)"""

        tic.Tac("Matrix", "Assembly Kd and Fd", self._verbosity)
        # We ensure the matrices are always updated with the latest damage or displacement results.
        # Therefore, we don't specify that the matrices have been updated.
        return self.__Kd, self.__Fd

    def __Solve_damage(self) -> _types.FloatArray:
        """Computes the damage field."""

        self._Solver_Solve(ModelType.damage)

        return self.damage

    def Save_Iter(self):
        iter = super().Save_Iter()

        # convergence informations
        iter["Niter"] = self.__Niter
        iter["timeIter"] = self.__timeIter
        iter["convIter"] = self.__convIter

        if self.phaseFieldModel.solver == self.phaseFieldModel.SolverType.History:
            # update old history field for next resolution
            self.__old_psiP_e_pg = self.__psiP_e_pg

        iter["displacement"] = self.displacement
        iter["damage"] = self.damage

        self._results.append(iter)

    def Set_Iter(self, iter: int = -1, resetAll=False) -> dict:
        results = super().Set_Iter(iter)

        if results is None:
            return

        damageType = ModelType.damage
        self._Set_solutions(damageType, results[damageType])

        displacementType = ModelType.elastic
        self._Set_solutions(displacementType, results["displacement"])

        # damage and displacement field will change thats why we need to update the assembled matrices
        self.__updatedDamage = False
        self.__updatedDisplacement = False

        if (
            resetAll
            and self.phaseFieldModel.solver == self.phaseFieldModel.SolverType.History
        ):
            # It's really useful to do this otherwise when we calculate psiP there will be a problem
            self.__old_psiP_e_pg = FeArray.zeros(*self.__old_psiP_e_pg.shape)
            # update psi+ with the current state
            self.__old_psiP_e_pg = self.__Calc_psiPlus_e_pg()

        return results

    def Results_Available(self) -> list[str]:
        results = []
        dim = self.dim

        results.extend(["displacement", "displacement_norm", "displacement_matrix"])

        if dim == 2:
            results.extend(["ux", "uy"])
            results.extend(["Sxx", "Syy", "Sxy"])
            results.extend(["Exx", "Eyy", "Exy"])

        elif dim == 3:
            results.extend(["ux", "uy", "uz"])
            results.extend(["Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy"])
            results.extend(["Exx", "Eyy", "Ezz", "Eyz", "Exz", "Exy"])

        results.extend(["Svm", "Stress", "Evm", "Strain"])

        results.extend(["damage", "psiP", "Psi_Crack"])
        results.extend(["Wdef"])

        return results

    def Result(
        self, result: str, nodeValues: bool = True, iter: Optional[int] = None
    ) -> Union[_types.FloatArray, float, None]:
        if iter is not None:
            self.Set_Iter(iter)

        if not self._Results_Check_Available(result):
            return None  # type: ignore [return-value]

        # begin cases ----------------------------------------------------

        Nn = self.mesh.Nn

        values = None

        if result in ["Wdef"]:
            return self._Calc_Psi_Elas()

        elif result == "Wdef_e":
            values = self._Calc_Psi_Elas()

        elif result == "Psi_Crack":
            return self._Calc_Psi_Crack()

        if result == "psiP":
            values_e_pg = self.__Calc_psiPlus_e_pg()
            values = np.mean(values_e_pg, axis=1)

        if result == "damage":
            values = self.damage  # type: ignore [assignment]

        elif result in ["ux", "uy", "uz"]:
            values_n = self.displacement.reshape(Nn, -1)
            values = values_n[:, self.__indexResult(result)]  # type: ignore [assignment]

        elif result == "displacement":
            values = self.displacement  # type: ignore [assignment]

        elif result == "displacement_norm":
            val_n = self.displacement.reshape(Nn, -1)
            values = np.linalg.norm(val_n, axis=1)

        elif result == "displacement_matrix":
            values = self.Results_displacement_matrix()  # type: ignore [assignment]

        elif ("S" in result or "E" in result) and ("_norm" not in result):
            # Strain and Stress calculation part

            coef = self.phaseFieldModel.material.coef

            displacement = self.displacement
            # Strain and stress for each element and gauss point
            Epsilon_e_pg = self._Calc_Epsilon_e_pg(displacement)
            Sigma_e_pg = self._Calc_Sigma_e_pg(Epsilon_e_pg)

            # Element average
            if "S" in result and result != "Strain":
                val_e = Sigma_e_pg.mean(1)
            elif "E" in result or result == "Strain":
                val_e = Epsilon_e_pg.mean(1)
            else:
                raise Exception("Wrong option")

            res = result if result in ["Strain", "Stress"] else result[-2:]

            values = Result_in_Strain_or_Stress_field(val_e, res, coef)  # type: ignore [assignment]

        if not isinstance(values, np.ndarray):
            Display.MyPrintError("This result option is not implemented yet.")
            return None  # type: ignore [return-value]

        # end cases ----------------------------------------------------

        return self.Results_Reshape_values(values, nodeValues)

    def __indexResult(self, result: str) -> int:
        if len(result) <= 2:
            if "x" in result:
                return 0
            elif "y" in result:
                return 1
            elif "z" in result:
                return 1
            else:
                raise ValueError("result error")
        else:
            raise ValueError("result error")

    def _Calc_Psi_Elas(self) -> float:
        """Computes of the kinematically admissible damaged deformation energy.\n
        Psi_Elas = 1/2 int_Ω Sig : Eps dΩ"""

        Ku = self.Get_K_C_M_F(ModelType.elastic)[0]

        tic = Tic()

        u = self.displacement.reshape(-1, 1)
        if np.linalg.norm(u) == 0:
            Psi_Elas = 0
        else:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                Psi_Elas = 1 / 2 * (u.T @ Ku @ u)[0, 0]

        tic.Tac("PostProcessing", "Calc Psi Elas", False)

        return Psi_Elas

    def _Calc_Psi_Crack(self) -> float:
        """Computes crack's energy."""

        Kd = self.Get_K_C_M_F(ModelType.damage)[0]

        tic = Tic()

        d = self.damage.reshape(-1, 1)
        if np.linalg.norm(d) == 0:
            Psi_Crack = 0
        else:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                Psi_Crack = 1 / 2 * (d.T @ Kd @ d)[0, 0]

        tic.Tac("PostProcessing", "Calc Psi Crack", False)

        return Psi_Crack

    def _Calc_Psi_Ext(self, f_n: _types.FloatArray) -> float:
        """Computes external's energy."""

        tic = Tic()

        u_n = self.displacement
        if np.linalg.norm(u_n) == 0 or np.linalg.norm(f_n) == 0:
            Psi_Ext = 0
        else:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                Psi_Ext = (u_n @ f_n)[0]

        tic.Tac("PostProcessing", "Calc Psi Ext", False)

        return Psi_Ext

    def _Calc_Epsilon_e_pg(
        self, sol: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray.FeArrayALike:
        """Computes strain field (Ne,pg,(3 or 6)).\n
        2D : [Exx Eyy sqrt(2)*Exy]\n
        3D : [Exx Eyy Ezz sqrt(2)*Eyz sqrt(2)*Exz sqrt(2)*Exy]

        Parameters
        ----------
        sol : _types.FloatArray
            Displacement vector

        Returns
        -------
        FeArray
            Computed strain field (Ne,pg,(3 or 6))
        """

        tic = Tic()
        sol_e = self.mesh.Locates_sol_e(sol, asFeArray=True)
        B_e_pg = self.mesh.Get_B_e_pg(matrixType)
        Epsilon_e_pg = B_e_pg @ sol_e

        tic.Tac("Matrix", "Epsilon_e_pg", False)

        return Epsilon_e_pg

    def _Calc_Sigma_e_pg(
        self, Epsilon_e_pg: FeArray.FeArrayALike, matrixType=MatrixType.rigi
    ) -> FeArray.FeArrayALike:
        """Computes stress field from strain field.\n
        2D : [Sxx Syy sqrt(2)*Sxy]\n
        3D : [Sxx Syy Szz sqrt(2)*Syz sqrt(2)*Sxz sqrt(2)*Sxy]

        Parameters
        ----------
        Epsilon_e_pg : FeArray.FeArrayALike
            Strain field (Ne,pg,(3 or 6))

        Returns
        -------
        FeArray
            Computed damaged stress field.
        """

        Epsilon_e_pg = FeArray.asfearray(Epsilon_e_pg)

        assert Epsilon_e_pg.shape[0] == self.mesh.Ne
        assert Epsilon_e_pg.shape[1] == self.mesh.Get_nPg(matrixType)

        d = self.damage

        phaseFieldModel = self.phaseFieldModel

        SigmaP_e_pg, SigmaM_e_pg = phaseFieldModel.Calc_Sigma_e_pg(Epsilon_e_pg)

        tic = Tic()

        # compute Sig such that: Sig = g(d) * SigP + SigM
        g_e_pg = phaseFieldModel.Get_g_e_pg(d, self.mesh, matrixType)
        SigmaP_e_pg = g_e_pg * SigmaP_e_pg
        Sigma_e_pg = SigmaP_e_pg + SigmaM_e_pg

        tic.Tac("Matrix", "Sigma_e_pg", False)

        return Sigma_e_pg

    def Results_Set_Bc_Summary(self, config: str):
        assert isinstance(config, str)
        self.__resumeLoading = config
        return self.__resumeLoading

    def Results_Get_Bc_Summary(self) -> str:
        return self.__resumeLoading

    def Results_Set_Iteration_Summary(
        self, iter: int, load: float, unitLoad: str, percentage=0.0, remove=False
    ) -> str:
        """Creates the iteration summary for the damage problem.

        Parameters
        ----------
        iter : int
            iteration
        load : float
            loading
        unitLoad : str
            loading unit
        percentage : float, optional
            percentage of simualtion performed, by default 0.0
        remove : bool, optional
            removes line from terminal after display, by default False
        """

        d = self.damage

        Niter = self.__Niter
        dincMax = self.__convIter
        timeIter = self.__timeIter

        min_d = d.min()
        max_d = d.max()
        summaryIter = f"{iter + 1:4d} : {load:4.3f} {unitLoad}, [{min_d:.2e}; {max_d:.2e}], {Niter}:{timeIter:4.3f} s, tol={dincMax:.2e}  "

        if remove:
            end = "\r"
        else:
            end = ""

        if percentage > 0:
            timeLeft = (1 / percentage - 1) * timeIter * iter
            timeCoef, unite = Tic.Get_time_unity(timeLeft)
            # Adds percentage and estimated time remaining
            summaryIter = (
                summaryIter + f"{percentage * 100:3.2f} % -> {timeCoef:3.2f} {unite}  "
            )

        Display.MyPrint(summaryIter, end=end)

        self.__resumeIter = summaryIter

        return summaryIter

    def Results_Get_Iteration_Summary(self) -> str:
        try:
            resumeIter = f"""
            tolConv = {self.__tolConv:.1e}
            maxIter = {self.__maxIter}
            convOption = {self.__convOption}\n\n
            """
        except AttributeError:
            resumeIter = ""

        resumeIter += self.__resumeIter

        return resumeIter

    def Results_dict_Energy(self) -> dict[str, float]:
        Psi_Elas = self._Calc_Psi_Elas()
        Psi_Crack = self._Calc_Psi_Crack()
        dict_energy = {
            r"$\Psi_{elas}$": Psi_Elas,
            r"$\Psi_{crack}$": Psi_Crack,
            r"$\Psi_{tot}$": Psi_Crack + Psi_Elas,
        }
        return dict_energy

    def Results_Iter_Summary(
        self,
    ) -> tuple[list[int], list[tuple[str, _types.FloatArray]]]:
        list_label_values = []

        resultats = self.results
        df = pd.DataFrame(resultats)
        iterations = np.arange(df.shape[0]).tolist()

        damageMaxIter = np.array([np.max(damage) for damage in df["damage"].values])
        list_label_values.append((r"$\phi$", damageMaxIter))

        convIter = df["convIter"].values
        list_label_values.append(("convIter", convIter))

        nombreIter = df["Niter"].values
        list_label_values.append(("Niter", nombreIter))

        tempsIter = df["timeIter"].values
        list_label_values.append(("time", tempsIter))

        return iterations, list_label_values

    def Results_displacement_matrix(self) -> _types.FloatArray:
        Nn = self.mesh.Nn
        coord = self.displacement.reshape((Nn, -1))
        dim = coord.shape[1]

        if dim == 1:
            # Here we add two columns
            coord = np.append(coord, np.zeros((Nn, 1)), axis=1)
            coord = np.append(coord, np.zeros((Nn, 1)), axis=1)
        elif dim == 2:
            # Here we add 1 column
            coord = np.append(coord, np.zeros((Nn, 1)), axis=1)

        return coord
