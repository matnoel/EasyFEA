# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Union, Callable, Optional, TYPE_CHECKING
import numpy as np

# utilities
from ..Utilities import Folder, Display, Tic, _types

# fem
if TYPE_CHECKING:
    from ..FEM import Mesh
from ..FEM import MatrixType, Mesher, FeArray

# models
from ..Models import ModelType, Reshape_variable, Result_in_Strain_or_Stress_field
from ..Models.Elastic._laws import _Elastic

# simu
from ._simu import _Simu
from .Solvers import AlgoType


class Elastic(_Simu):
    r"""Linearized elasticity.
    
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

    def __init__(self, mesh: "Mesh", model: _Elastic, verbosity=False):
        """Creates a elastic simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh used.
        model : _Elas
            The elastic model (or material) used.
        verbosity : bool, optional
            If True, the simulation can write in the terminal. Defaults to False.
        """

        assert isinstance(model, _Elastic), "model must be a elastic model"
        super().__init__(mesh, model, verbosity)

        # init
        self.Set_Rayleigh_Damping_Coefs()

        # Set solver petsc4py options, even if petsc4py is unavailable.
        self._Solver_Set_PETSc4Py_Options(pcType="lu")

    def Results_nodeFields_elementFields(
        self, details=False
    ) -> tuple[list[str], list[str]]:
        nodesField = ["displacement"]
        if details:
            elementsField = ["Svm", "Stress", "Strain"]
        else:
            elementsField = ["Svm", "Stress"]
        if self.algo in AlgoType.Get_Hyperbolic_Types():
            nodesField.extend(["speed", "accel"])
        return nodesField, elementsField

    def Get_unknowns(self, problemType=None) -> list[str]:
        dict_unknowns = {2: ["x", "y"], 3: ["x", "y", "z"]}
        return dict_unknowns[self.dim]

    def Get_problemTypes(self) -> list[ModelType]:
        return [ModelType.elastic]

    def Get_dof_n(self, problemType=None) -> int:
        return self.dim

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

    def Construct_local_matrix_system(self, problemType):

        mesh = self.mesh
        tic = Tic()

        # ------------------------------
        # Compute Stiffness
        # ------------------------------
        matrixType = MatrixType.rigi
        leftDepPart = mesh.Get_leftDispPart(matrixType)
        B_dep_e_pg = mesh.Get_B_e_pg(matrixType)

        if self.material.isHeterogeneous:
            matC = Reshape_variable(self.material.C, *B_dep_e_pg.shape[:2])
        else:
            matC = self.material.C

        Ku_e = (leftDepPart @ matC @ B_dep_e_pg).sum(axis=1)

        # ------------------------------
        # Compute Mass
        # ------------------------------
        matrixType = MatrixType.mass
        N_pg = FeArray.asfearray(mesh.Get_N_vector_pg(matrixType)[np.newaxis])
        wJ_e_pg = mesh.Get_weightedJacobian_e_pg(matrixType)

        rho_e_pg = Reshape_variable(self.rho, *wJ_e_pg.shape[:2])

        Mu_e = (rho_e_pg * wJ_e_pg * N_pg.T @ N_pg).sum(axis=1)

        if self.dim == 2:
            thickness = self.material.thickness
            Ku_e *= thickness
            Mu_e *= thickness

        tic.Tac("Matrix", "Construct Ku_e and Mu_e", self._verbosity)

        Cu_e = self.__coefK * Ku_e + self.__coefM * Mu_e

        return Ku_e, Cu_e, Mu_e, None

    def Set_Rayleigh_Damping_Coefs(self, coefM=0.0, coefK=0.0):
        """Sets damping coefficients \( C = coefK * K + coefM * M \)."""
        self.__coefM = coefM
        self.__coefK = coefK
        self.Need_Update()

    def Get_x0(self, problemType=None):
        algo = self.algo
        if self.displacement.size != self.mesh.Nn * self.dim:
            return np.zeros(self.mesh.Nn * self.dim)
        elif algo == AlgoType.elliptic:
            return self.displacement
        elif algo in AlgoType.Get_Hyperbolic_Types():
            return self.accel
        else:
            raise TypeError(f"Algo {algo} is not implemented here.")

    def Save_Iter(self):
        iter = super().Save_Iter()

        iter["displacement"] = self.displacement
        if self.algo in AlgoType.Get_Hyperbolic_Types():
            iter["speed"] = self.speed
            iter["accel"] = self.accel

        self._results.append(iter)

    def Set_Iter(self, iter: int = -1, resetAll=False) -> dict:
        results = super().Set_Iter(iter)

        if results is None:
            return

        u = results["displacement"]

        if (
            self.algo in AlgoType.Get_Hyperbolic_Types()
            and "speed" in results
            and "accel" in results
        ):
            v = results["speed"]
            a = results["accel"]
        else:
            v = np.zeros_like(u)
            a = np.zeros_like(u)

        self._Set_solutions(self.problemType, u, v, a)

        return results

    def Results_Available(self) -> list[str]:
        results = []
        dim = self.dim

        results.extend(["displacement", "displacement_norm", "displacement_matrix"])
        results.extend(["speed", "speed_norm"])
        results.extend(["accel", "accel_norm"])

        if dim == 2:
            results.extend(["ux", "uy"])
            results.extend(["vx", "vy"])
            results.extend(["ax", "ay"])
            results.extend(["Sxx", "Syy", "Sxy"])
            results.extend(["Exx", "Eyy", "Exy"])

        elif dim == 3:
            results.extend(["ux", "uy", "uz"])
            results.extend(["vx", "vy", "vz"])
            results.extend(["ax", "ay", "az"])
            results.extend(["Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy"])
            results.extend(["Exx", "Eyy", "Ezz", "Eyz", "Exz", "Exy"])

        results.extend(["Svm", "Stress", "Evm", "Strain"])

        results.extend(["Wdef", "Wdef_e", "ZZ1", "ZZ1_e"])

        return results

    def Result(
        self, result: str, nodeValues: bool = True, iter: Optional[int] = None
    ) -> Union[_types.FloatArray, float]:
        if iter is not None:
            self.Set_Iter(iter)

        if not self._Results_Check_Available(result):
            return None  # type: ignore [return-value]

        # begin cases ----------------------------------------------------

        Nn = self.mesh.Nn

        values = None

        if result in ["ux", "uy", "uz"]:
            values_n = self.displacement.reshape(Nn, -1)
            values = values_n[:, self.__indexResult(result)]

        elif result == "displacement":
            values = self.displacement

        elif result == "displacement_norm":
            val_n = self.displacement.reshape(Nn, -1)
            values = np.linalg.norm(val_n, axis=1)

        elif result == "displacement_matrix":
            values = self.Results_displacement_matrix()

        elif result in ["vx", "vy", "vz"]:
            values_n = self.speed.reshape(Nn, -1)
            values = values_n[:, self.__indexResult(result)]

        elif result == "speed":
            values = self.speed

        elif result == "speed_norm":
            val_n = self.speed.reshape(Nn, -1)
            values = np.linalg.norm(val_n, axis=1)

        elif result in ["ax", "ay", "az"]:
            values_n = self.accel.reshape(Nn, -1)
            values = values_n[:, self.__indexResult(result)]

        elif result == "accel":
            values = self.accel

        elif result == "accel_norm":
            val_n = self.accel.reshape(Nn, -1)
            values = np.linalg.norm(val_n, axis=1)

        elif result in ["Wdef"]:
            return self._Calc_Psi_Elas()

        elif result == "Wdef_e":
            values = self._Calc_Psi_Elas(returnScalar=False)

        elif result == "ZZ1":
            return self._Calc_ZZ1()[0]

        elif result == "ZZ1_e":
            values = self._Calc_ZZ1()[1]

        elif ("S" in result or "E" in result) and ("_norm" not in result):
            # Strain and Stress calculation part

            coef = self.material.coef

            displacement = self.displacement
            # Strain and stress for each element and gauss point
            Epsilon_e_pg = self._Calc_Epsilon_e_pg(displacement)
            Sigma_e_pg = self._Calc_Sigma_e_pg(Epsilon_e_pg)

            # Element average
            if "S" in result and result != "Strain":
                values_e_pg = Sigma_e_pg
            elif "E" in result or result == "Strain":
                values_e_pg = Epsilon_e_pg
            else:
                raise Exception("Wrong option")

            res = result if result in ["Strain", "Stress"] else result[-2:]

            values = Result_in_Strain_or_Stress_field(values_e_pg, res, coef).mean(1)

        if not isinstance(values, np.ndarray):
            Display.MyPrintError("This result option is not implemented yet.")
            return None  # type: ignore [return-value]

        # end cases ----------------------------------------------------

        return self.Results_Reshape_values(values, nodeValues)

    def _Calc_Psi_Elas(
        self, returnScalar=True, smoothedStress=False, matrixType=MatrixType.rigi
    ):
        """Computes the kinematically admissible deformation energy.
        Wdef = 1/2 int_Ω Sig : Eps dΩ"""

        tic = Tic()

        sol_u = self.displacement

        mesh = self.mesh

        Epsilon_e_pg = self._Calc_Epsilon_e_pg(sol_u, matrixType)
        weightedJacobian_pg = mesh.Get_weightedJacobian_e_pg(matrixType)
        N_pg = mesh.Get_N_pg(matrixType)

        if self.dim == 2:
            ep = self.material.thickness
        else:
            ep = 1

        Sigma_e_pg = self._Calc_Sigma_e_pg(Epsilon_e_pg, matrixType)

        if smoothedStress:
            Sigma_n = mesh.Get_Node_Values(np.mean(Sigma_e_pg, 1))

            Sigma_n_e = mesh.Locates_sol_e(Sigma_n)
            Sigma_e_pg = FeArray.asfearray(np.einsum("eni,pjn->epi", Sigma_n_e, N_pg))

        if returnScalar:
            Wdef = 1 / 2 * ep * (weightedJacobian_pg * Sigma_e_pg @ Epsilon_e_pg).sum()
        else:
            Wdef = 1 / 2 * ep * (weightedJacobian_pg * Sigma_e_pg @ Epsilon_e_pg).sum(1)

        tic.Tac("PostProcessing", "Calc Psi Elas", False)

        return Wdef

    def _Calc_ZZ1(self) -> tuple[float, _types.FloatArray]:
        """Computes the ZZ1 error.\n
        For more details, [F.Pled, Vers une stratégie robuste ... ingénierie mécanique] page 20/21\n
        Returns the global error and the error on each element.

        Returns
        -------
        error, error_e
        """

        W_e = self._Calc_Psi_Elas(False)
        Welas = np.sum(W_e)

        Ws_e = self._Calc_Psi_Elas(False, True)
        Ws = np.sum(Ws_e)

        error_e = np.abs(Ws_e - W_e).ravel() / Welas

        error: float = np.abs(Welas - Ws) / Welas

        return error, error_e

    def _Calc_Epsilon_e_pg(
        self, u: _types.FloatArray, matrixType=MatrixType.rigi
    ) -> FeArray.FeArrayALike:
        """Computes strain field from the displacement vector field.\n
        2D : [Exx Eyy sqrt(2)*Exy]\n
        3D : [Exx Eyy Ezz sqrt(2)*Eyz sqrt(2)*Exz sqrt(2)*Exy]

        Parameters
        ----------
        u : _types.FloatArray
            displacement vector (Ndof)

        Returns
        -------
        FeArray
            Computed strain field (Ne,pg,(3 or 6))
        """

        tic = Tic()
        u_e = self.mesh.Locates_sol_e(u, asFeArray=True)
        B_dep_e_pg = self.mesh.Get_B_e_pg(matrixType)
        Epsilon_e_pg = B_dep_e_pg @ u_e

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
            Computed stress field (Ne,pg,(3 or 6))
        """

        Epsilon_e_pg = FeArray.asfearray(Epsilon_e_pg)

        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]

        assert Ne == self.mesh.Ne
        assert nPg == self.mesh.Get_nPg(matrixType)

        tic = Tic()

        C = self.material.C
        if self.material.isHeterogeneous:
            C_e_pg = Reshape_variable(C, Ne, nPg)
        else:
            C_e_pg = FeArray.asfearray(C, True)

        Sigma_e_pg = C_e_pg @ Epsilon_e_pg

        tic.Tac("Matrix", "Sigma_e_pg", False)

        return Sigma_e_pg

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

    def Results_dict_Energy(self) -> dict[str, float]:
        dict_energy = {r"$\Psi_{elas}$": self._Calc_Psi_Elas()}
        return dict_energy

    def Results_Get_Iteration_Summary(self) -> str:
        summary = ""

        if not self._Results_Check_Available("Wdef"):
            return None  # type: ignore [return-value]

        Wdef = self.Result("Wdef")
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

    def Results_Iter_Summary(
        self,
    ) -> tuple[list[int], list[tuple[str, _types.FloatArray]]]:
        return super().Results_Iter_Summary()

    def Results_displacement_matrix(self) -> _types.FloatArray:
        Nn = self.mesh.Nn
        coord = self.displacement.reshape((Nn, -1))
        dim = coord.shape[1]

        displacement_matrix = np.zeros((Nn, 3))
        displacement_matrix[:, :dim] = coord

        return displacement_matrix


# ----------------------------------------------
# Other functions
# ----------------------------------------------
def Mesh_Optim_ZZ1(
    DoSimu: Callable[[str], Elastic],
    folder: str,
    threshold: float = 1e-2,
    iterMax: int = 20,
    coef: float = 1 / 2,
) -> Elastic:
    """Optimizes the mesh using ZZ1 error criterion.

    Parameters
    ----------
    DoSimu : Callable[[str], Displacement]
        Function that runs a simulation and takes a .pos file as argument for mesh optimization. The function must return a Displacement simulation.
    folder : str
        Folder in which .pos files are created and then deleted.
    threshold : float, optional
        targeted error, by default 1e-2
    iterMax : int, optional
        Maximum number of iterations, by default 20
    coef : float, optional
        mesh size division ratio, by default 1/2

    Returns
    -------
    Displacement
        Displacement simulation
    """

    i = -1
    error = 1
    optimGeom: Optional[str] = None
    # max=1
    while error >= threshold and i <= iterMax:
        i += 1

        # perform the simulation
        simu = DoSimu(optimGeom)  # type: ignore [arg-type]
        assert isinstance(
            simu, Elastic
        ), "DoSimu function must return a Displacement simulation"
        # get the current mesh
        mesh = simu.mesh

        if i > 0:
            # remove previous .pos file
            Folder.os.remove(optimGeom)  # type: ignore [arg-type]

        # Calculate the error with the ZZ1 method
        error, error_e = simu._Calc_ZZ1()  # type: ignore [assignment]

        print(f"error = {error * 100:.3f} %")

        # calculate the new mesh size for the associated error
        meshSize_n = mesh.Get_New_meshSize_n(error_e, coef)

        # build the .pos file that will be used to refine the mesh
        optimGeom = Mesher().Create_posFile(mesh.coord, meshSize_n, folder, f"pos{i}")

    if Folder.Exists(optimGeom):  # type: ignore [arg-type]
        # remove last .pos file
        Folder.os.remove(optimGeom)  # type: ignore [arg-type]

    return simu
