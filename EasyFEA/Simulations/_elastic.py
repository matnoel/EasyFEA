# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Union, Callable, Optional, TYPE_CHECKING
import numpy as np

# utilities
from ..Utilities import Folder, Display, Tic, _types

# fem
if TYPE_CHECKING:
    from ..FEM import Mesh
from ..FEM import MatrixType, Mesher, FeArray, Operators

# models
from ..Models import ModelType, Result_strain_or_stress_field_e
from ..Models.Elastic._laws import _Elastic

# simu
from ._simu import _Simu
from .Solvers import AlgoType


class Elastic(_Simu):
    r"""Linear (small-strain) elasticity simulation.

    Solves the static, quasi-static or dynamic equilibrium of a linear-elastic body under body forces, surface tractions and Dirichlet conditions, in 1D/2D/3D. Transient and dynamic analyses use the available time schemes (Newmark, HHT, midpoint, …) and support Rayleigh damping. Material behaviour is supplied by an elastic model (isotropic, anisotropic, …).

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

        tic = Tic()

        out = {}

        for groupElem in self.mesh.Get_list_groupElem():

            # compute stiffness
            K_e = Operators.Bilinear.LinearizedElasticity(groupElem, self.material.C)

            # compute mass
            M_e = Operators.Bilinear.UV(groupElem, self.rho, dof_n=self.dim)

            if self.dim == 2:
                thickness = self.material.thickness
                K_e *= thickness
                M_e *= thickness

            tic.Tac(
                "Matrix",
                f"Construct K_e and M_e ({groupElem.elemType})",
                self._verbosity,
            )

            C_e = self.__coefK * K_e + self.__coefM * M_e

            out[groupElem] = (K_e, C_e, M_e, None)

        return out

    def Set_Rayleigh_Damping_Coefs(self, coefM=0.0, coefK=0.0):
        r"""Sets damping coefficients \( C = coefK * K + coefM * M \)."""
        self.__coefM = coefM
        self.__coefK = coefK
        self.Need_Update()

    def Get_x0(self, problemType=None):
        if self.displacement.size != self.mesh.Nn * self.dim:
            return np.zeros(self.mesh.Nn * self.dim)
        else:
            return self.displacement

    def Save_Iter(self, iter=None):

        if iter is None:
            iter = {}

        iter["displacement"] = self.displacement
        if self.algo in AlgoType.Get_Hyperbolic_Types():
            iter["speed"] = self.speed
            iter["accel"] = self.accel

        return super().Save_Iter(iter)

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

            displacement = self.displacement

            isStrain = "E" in result or result == "Strain"
            isStress = "S" in result and result != "Strain"
            if not (isStrain or isStress):
                raise Exception("Wrong option")

            res = result if result in ["Strain", "Stress"] else result[-2:]

            def field_e_pg(groupElem):
                Eps = self._Calc_Epsilon_e_pg(displacement, groupElem)
                return self._Calc_Sigma_e_pg(Eps, groupElem) if isStress else Eps

            values = Result_strain_or_stress_field_e(
                field_e_pg=field_e_pg,
                list_groupElem=self.mesh.Get_list_groupElem(),
                result=res,
                coef=self.material.coef,
            )

        else:
            Display.MyPrintError(f"The result '{result}' is not implemented yet.")
            return None  # type: ignore [return-value]

        # end cases ----------------------------------------------------

        return self.Results_Reshape_values(values, nodeValues)

    def _Calc_Psi_Elas(
        self,
        returnScalar=True,
        smoothedStress=False,
        matrixType=MatrixType.rigi,
    ):
        r"""Computes the kinematically admissible deformation energy.

        .. math:: W_{def} = \frac{1}{2} \int_\Omega \Sig : \Eps \, \dO = \int_\Omega \psi \, \dO

        Parameters
        ----------
        returnScalar : bool, optional
            If True returns the total energy as a float, otherwise the per-element energy (Ne,), by default True.
        smoothedStress : bool, optional
            If True the energy density is built from the nodal-averaged (smoothed) stress field rather than the raw element stress; used by the ZZ1 error estimator, by default False.
        matrixType : MatrixType, optional
            integration scheme, by default MatrixType.rigi.
        """

        tic = Tic()

        sol_u = self.displacement
        thickness = self.material.thickness if self.dim == 2 else 1

        # strain and elastic energy density psi = 1/2 Sig : Eps, group by group
        # (each main-dimension group may have its own element type / number of Gauss points)
        list_groupElem = self.mesh.Get_list_groupElem(self.dim)
        list_Eps = [
            self._Calc_Epsilon_e_pg(sol_u, groupElem, matrixType)
            for groupElem in list_groupElem
        ]
        list_psi = [self.material.Calc_Psi_e_pg(Eps) for Eps in list_Eps]

        if smoothedStress:
            # ZZ1: rebuild psi from the element stresses averaged at the nodes, then projected back onto the Gauss points.
            list_Sig = [
                self._Calc_Sigma_e_pg(Eps, g, matrixType)
                for g, Eps in zip(list_groupElem, list_Eps)
            ]
            Sigma_n = self.mesh.Get_Node_Values(
                np.concatenate([np.mean(Sig, 1) for Sig in list_Sig])
            )
            list_psi = [
                self.material.Calc_Psi_e_pg(
                    Eps,
                    FeArray.asfearray(
                        np.einsum(
                            "eni,pjn->epi",
                            groupElem.Locates_sol_e(Sigma_n),
                            groupElem.Get_N_pg(matrixType),
                        )
                    ),
                )
                for groupElem, Eps in zip(list_groupElem, list_Eps)
            ]

        # integrate the energy density over each group: Wdef_e = int psi dOmega
        Wdef_e = np.concatenate(
            [
                np.asarray(
                    (
                        thickness
                        * groupElem.Get_weightedJacobian_e_pg(matrixType)
                        * psi
                    ).sum(1)
                )
                for groupElem, psi in zip(list_groupElem, list_psi)
            ]
        )

        tic.Tac("PostProcessing", "Calc Psi Elas", False)

        return float(Wdef_e.sum()) if returnScalar else Wdef_e

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
        self,
        u: _types.FloatArray,
        groupElem=None,
        matrixType=MatrixType.rigi,
    ) -> FeArray.FeArrayALike:
        """Computes the strain field from the displacement vector field (delegates to the material law ``Calc_Epsilon_e_pg``).\n
        2D : [Exx Eyy sqrt(2)*Exy]\n
        3D : [Exx Eyy Ezz sqrt(2)*Eyz sqrt(2)*Exz sqrt(2)*Exy]

        Parameters
        ----------
        u : _types.FloatArray
            displacement vector field (Ndof)
        groupElem : _GroupElem, optional
            element group on which to evaluate the strain, by default None (main group)
        matrixType : MatrixType, optional
            integration scheme, by default MatrixType.rigi

        Returns
        -------
        FeArray
            strain field (Ne, pg, (3 or 6))
        """
        if groupElem is None:
            groupElem = self.mesh.groupElem
        return self.material.Calc_Epsilon_e_pg(u, groupElem, matrixType)

    def _Calc_Sigma_e_pg(
        self,
        Epsilon_e_pg: FeArray.FeArrayALike,
        groupElem=None,
        matrixType=MatrixType.rigi,
    ) -> FeArray.FeArrayALike:
        """Computes the stress field from the strain field (Hooke's law, delegates to the material law ``Calc_Sigma_e_pg``).\n
        2D : [Sxx Syy sqrt(2)*Sxy]\n
        3D : [Sxx Syy Szz sqrt(2)*Syz sqrt(2)*Sxz sqrt(2)*Sxy]

        Parameters
        ----------
        Epsilon_e_pg : FeArray.FeArrayALike
            strain field (Ne, pg, (3 or 6))
        groupElem : _GroupElem, optional
            element group the strain field belongs to (used to check the array shape), by default None (main group)
        matrixType : MatrixType, optional
            integration scheme used for the shape check, by default MatrixType.rigi

        Returns
        -------
        FeArray
            stress field (Ne, pg, (3 or 6))
        """

        Epsilon_e_pg = FeArray.asfearray(Epsilon_e_pg)

        if groupElem is None:
            groupElem = self.mesh.groupElem

        assert Epsilon_e_pg.shape[0] == groupElem.Ne
        assert Epsilon_e_pg.shape[1] == groupElem.Get_gauss(matrixType).nPg

        tic = Tic()

        # constitutive law Sigma = C : Epsilon lives on the material model
        Sigma_e_pg = self.material.Calc_Sigma_e_pg(Epsilon_e_pg)

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
