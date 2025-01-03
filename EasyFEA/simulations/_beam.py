# Copyright (C) 2021-2024 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from typing import Union
import numpy as np
from scipy import sparse

# utilities
from ..utilities import Display, Tic
# fem
from ..fem import Mesh, MatrixType, LagrangeCondition
# materials
from .. import Materials
from ..materials import ModelType, Reshape_variable
# simu
from ._simu import _Simu

class BeamSimu(_Simu):

    def __init__(self, mesh: Mesh, model: Materials.BeamStructure, verbosity=False, useNumba=True, useIterativeSolvers=True):
        """Creates a Euler-Bernoulli beam simulation.

        Parameters
        ----------
        mesh : Mesh
            the mesh used.
        model : Beam_Structure | _Beam
            the model used.
        verbosity : bool, optional
            If True, the simulation can write in the terminal. Defaults to False.
        useNumba : bool, optional
            If True, numba can be used. Defaults to True.
        useIterativeSolvers : bool, optional
            If True, iterative solvers can be used. Defaults to True.
        """

        if isinstance(model, Materials._Beam):
            # changes the beam model as a beam structure
            model = Materials.BeamStructure([model])

        assert isinstance(model, Materials.BeamStructure), "model must be a beam model or a beam structure"
        super().__init__(mesh, model, verbosity, useNumba, useIterativeSolvers)

        # init
        self.Solver_Set_Elliptic_Algorithm()
        
        # turn beams into observable objects
        [beam._Add_observer(self) for beam in model.beams]
        
    def Results_nodesField_elementsField(self, details=False) -> tuple[list[str], list[str]]:
        if details:
            nodesField = ["displacement_matrix"]
            elementsField = ["Stress"]
        else:
            nodesField = ["displacement_matrix"]
            elementsField = ["Stress"]
        return nodesField, elementsField

    def Get_dofs(self, problemType=None) -> list[str]:
        dict_nbddl_directions = {
            1 : ["x"],
            3 : ["x","y","rz"],
            6 : ["x","y","z","rx","ry","rz"]
        }
        return dict_nbddl_directions[self.structure.dof_n]
    
    def Get_problemTypes(self) -> list[ModelType]:
        return [ModelType.beam]

    @property
    def structure(self) -> Materials.BeamStructure:
        """Beam structure."""
        return self.model

    def Get_dof_n(self, problemType=None) -> int:
        return self.structure.dof_n

    def _Check_dim_mesh_material(self) -> None:
        # In the case of a beam problem, we don't need to check this condition.
        pass

    @property
    def displacement(self) -> np.ndarray:
        """Displacement vector field.\n
        1D [uxi, ...]\n
        2D [uxi, uyi, rzi, ...]\n
        3D [uxi, uyi, uzi, rxi, ryi, rzi, ...]"""
        return self._Get_u_n(self.problemType)

    def add_surfLoad(self, nodes: np.ndarray, values: list, directions: list, problemType=None, description=""):
        Display.MyPrintError("Surface loads cannot be applied in beam problems.")
        return        

    def add_volumeLoad(self, nodes: np.ndarray, values: list, directions: list, problemType=None, description=""):
        Display.MyPrintError("Volumetric loads cannot be applied in beam problems.")
        return

    def add_connection_fixed(self, nodes: np.ndarray, description="Fixed"):
        """Adds a fixed connection.

        Parameters
        ----------
        nodes : np.ndarray
            nodes
        description : str, optional
            description, by default "Fixed"
        """

        beamModel = self.structure

        if beamModel.dim == 1:
            directions = ['x']
        elif beamModel.dim == 2:
            directions = ['x','y','rz']
        elif beamModel.dim == 3:
            directions = ['x','y','z','rx','ry','rz']

        description = f"Connection {description}"
        
        self.add_connection(nodes, directions, description)

    def add_connection_hinged(self, nodes: np.ndarray, directions=[''] ,description="Hinged"):
        """Adds a hinged connection.

        Parameters
        ----------
        nodes : np.ndarray
            nodes
        directions : list, optional
            directions, by default ['']
        description : str, optional
            description, by default "Hinged"
        """

        beamModel = self.structure
        
        if beamModel.dim == 1:
            return
        elif beamModel.dim == 2:
            directions = ['x','y']
        elif beamModel.dim == 3:
            directionsDeBase = ['x','y','z']
            directions = directionsDeBase
            if directions != ['']:
                # We will block rotation ddls that are not in directions.
                directionsRot = ['rx','ry','rz']
                for dir in directions:
                    if dir in directionsRot.copy():
                        directionsRot.remove(dir)
                directions.extend(directionsRot)

        description = f"Connection {description}"
        
        self.add_connection(nodes, directions, description)

    def add_connection(self, nodes: np.ndarray, directions: list[str], description: str):
        """Connects beams together in the specified directions.

        Parameters
        ----------
        nodes : np.ndarray
            nodes
        directions : list[str]
            directions
        description : str
            description
        """

        nodes = np.asarray(nodes)

        problemType = self.problemType        
        self._Check_dofs(problemType, directions)

        tic = Tic()
        
        if nodes.size > 1:
            # For each direction, we'll apply the conditions
            for d, dir in enumerate(directions):
                dofs = self.Bc_dofs_nodes(nodes, [dir], problemType)

                new_LagrangeBc = LagrangeCondition(problemType, nodes, dofs, [dir], [0], [1,-1], description)
                self._Bc_Add_Lagrange(new_LagrangeBc)
        else:
            self.add_dirichlet(nodes, [0]*len(directions), directions)

        tic.Tac("Boundary Conditions","Connection", self._verbosity)

        self._Bc_Add_Display(nodes, directions, description, problemType)

    def __Construct_Beam_Matrix(self) -> np.ndarray:
        """Constructs the elementary stiffness matrices for the beam problem."""

        # Data
        mesh = self.mesh
        if not mesh.groupElem.dim == 1: return
        groupElem = mesh.groupElem

        # Recovering the beam model
        beamStructure = self.structure
        
        matrixType=MatrixType.beam
        
        tic = Tic()
        
        jacobian_e_pg = mesh.Get_jacobian_e_pg(matrixType)
        weight_pg = mesh.Get_weight_pg(matrixType)

        D_e_pg = beamStructure.Calc_D_e_pg(groupElem)        

        B_e_pg = groupElem.Get_EulerBernoulli_B_e_pg(beamStructure)
        
        Kbeam_e = np.einsum('ep,p,epji,epjk,epkl->eil', jacobian_e_pg, weight_pg, B_e_pg, D_e_pg, B_e_pg, optimize='optimal')
            
        tic.Tac("Matrix","Construct Kbeam_e", self._verbosity)

        return Kbeam_e
    
    @property
    def mass(self) -> float:

        matrixType = MatrixType.mass

        mesh = self.mesh

        group = mesh.groupElem

        jacobian_e_p = group.Get_jacobian_e_pg(matrixType)
        
        weight_p = group.Get_weight_pg(matrixType)        

        rho_e_p = Reshape_variable(self.rho, mesh.Ne, weight_p.size)

        area_e = np.zeros(mesh.Ne)

        for beam in self.structure.beams:

            elements = mesh.Elements_Tags(beam.name)

            area_e[elements] = beam.area

        mass = float(np.einsum('ep,ep,p,e->', rho_e_p, jacobian_e_p, weight_p, area_e, optimize='optimal'))

        return mass
    
    @property
    def center(self) -> np.ndarray:
        """Center of mass / barycenter / inertia center"""

        matrixType = MatrixType.mass

        mesh = self.mesh

        group = mesh.groupElem

        coordo_e_p = group.Get_GaussCoordinates_e_p(matrixType)

        jacobian_e_p = group.Get_jacobian_e_pg(matrixType)
        weight_p = group.Get_weight_pg(matrixType)        

        rho_e_p = Reshape_variable(self.rho, mesh.Ne, weight_p.size)
        mass = self.mass

        area_e = np.zeros(mesh.Ne)
        for beam in self.structure.beams:
            elements = mesh.Elements_Tags(beam.name)
            area_e[elements] = beam.area

        center: np.ndarray = np.einsum('ep,e,ep,p,epi->i', rho_e_p, area_e, jacobian_e_p, weight_p, coordo_e_p, optimize='optimal') / mass

        if not isinstance(self.rho, np.ndarray):
            diff = np.linalg.norm(center - mesh.center)/np.linalg.norm(center)
            assert diff <= 1e-12

        return center

    def Assembly(self) -> None:

        # Data
        mesh = self.mesh

        model = self.structure

        nDof = mesh.Nn * model.dof_n

        Ku_beam = self.__Construct_Beam_Matrix()

        # Additional dimension linked to the use of lagrange coefficients
        nDof += self._Bc_Lagrange_dim(self.problemType)
        
        tic = Tic()

        linesVector_e = mesh.Get_linesVector_e(model.dof_n).ravel()
        columnsVector_e = mesh.Get_columnsVector_e(model.dof_n).ravel()

        # Assembly
        self.__Kbeam = sparse.csr_matrix((Ku_beam.ravel(), (linesVector_e, columnsVector_e)), shape=(nDof, nDof))
        """Kglob matrix for beam problem (nDof, nDof)"""

        self.__Fbeam = sparse.csr_matrix((nDof, 1))
        """Fglob vector for beam problem (nDof, 1)"""

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.spy(self.__Ku)
        # plt.show()

        tic.Tac("Matrix","Assembly Kbeam and Fbeam", self._verbosity)

    def Get_K_C_M_F(self, problemType=None) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        if self.needUpdate:
            self.Assembly()
            self.Need_Update(False)
        size = self.__Kbeam.shape[0]
        initcsr = sparse.csr_matrix((size, size))
        return self.__Kbeam.copy(), initcsr.copy(), initcsr.copy(), self.__Fbeam.copy()

    def Get_x0(self, problemType=None):
        if self.displacement.size != self.mesh.Nn*self.Get_dof_n(problemType):
            return np.zeros(self.mesh.Nn*self.Get_dof_n(problemType))
        else:
            return self.displacement

    def Save_Iter(self):

        iter = super().Save_Iter()
        
        iter['displacement'] = self.displacement
            
        self._results.append(iter)

    def Set_Iter(self, iter: int=-1, resetAll=False) -> dict:
        
        results = super().Set_Iter(iter)

        if results is None: return

        self._Set_u_n(self.problemType, results["displacement"])

        return results

    def Results_Available(self) -> list[str]:

        options = []
        dof_n = self.Get_dof_n(self.problemType)

        options.extend(["displacement", "displacement_norm", "displacement_matrix"])
        
        if dof_n == 1:
            options.extend(["ux"])
            options.extend(["fx"])
            options.extend(["ux'"])
            options.extend(["N"])
            options.extend(["Sxx"])            

        elif dof_n == 3:
            options.extend(["ux","uy","rz"])
            options.extend(["fx", "fy", "cz"])
            options.extend(["ux'", "rz'"])
            options.extend(["N", "Ty", "Mz"])
            options.extend(["Sxx", "Sxy"])

        elif dof_n == 6:
            options.extend(["ux", "uy", "uz", "rx", "ry", "rz"])
            options.extend(["fx","fy","fz","cx","cy","cz"])
            options.extend(["ux'", "rx'", "ry'", "rz'"])
            options.extend(["N", "Ty", "Tz", "Mx", "My", "Mz"])
            options.extend(["Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy"])
        
        options.extend(["Srain", "Stress"])

        return options

    def Result(self, result: str, nodeValues=True, iter=None) -> Union[np.ndarray, float]:

        if iter != None:
            self.Set_Iter(iter)
        
        if not self._Results_Check_Available(result): return None

        # begin cases ----------------------------------------------------

        dof_n = self.structure.dof_n
        Nn = self.mesh.Nn
        dofs = Nn*dof_n

        if result in ["ux","uy","uz","rx","ry","rz"]:            
            values_n = self.displacement.reshape(Nn, -1)
            index = self.__indexResult(result)
            values = values_n[:, index]

        elif result == "displacement":
            values = self.displacement

        elif result == "displacement_norm":
            values = np.linalg.norm(self.Results_displacement_matrix(), axis=1)
    
        elif result == "displacement_matrix":
            values = self.Results_displacement_matrix()        

        elif result in ["fx","fy","fz","cx","cy","cz"]:
        
            Kbeam = self.Get_K_C_M_F()[0]
            Kglob = Kbeam.tocsr()[:dofs].tocsc()[:,:dofs]
            force = Kglob @ self.displacement

            force_n = force.reshape(self.mesh.Nn, -1)
            index = self.__indexResult(result)
            values = force_n[:, index]

        elif result in ["N","Mx","My","Mz"]:

            Epsilon_e_pg = self._Calc_Epsilon_e_pg(self.displacement)

            internalForces_e_pg = self._Calc_InternalForces_e_pg(Epsilon_e_pg)
            values_e = internalForces_e_pg.mean(1)
            index = self.__indexResult(result)
            values = values_e[:, index]

        elif result in ["Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy"]:

            Epsilon_e_pg = self._Calc_Epsilon_e_pg(self.displacement)
            Sigma_e = self._Calc_Sigma_e_pg(Epsilon_e_pg).mean(1)
            index = self.__indexResult(result)
            values = Sigma_e[:,index]
        
        elif result in ["ux'", "rx'", "ry'", "rz'"]:

            coef = 1 if result == "Exx" else 1/2

            Epsilon_e = self._Calc_Epsilon_e_pg(self.displacement).mean(1)
            index = self.__indexResult(result)
            values = Epsilon_e[:,index] * coef

        # end cases ----------------------------------------------------
        
        return self.Results_Reshape_values(values, nodeValues)

    def __indexResult(self, result: str) -> int:

        # "Beam1D" : ["ux" "fx"]
        # "Beam2D : ["ux","uy","rz""fx", "fy", "cz"]
        # "Beam3D" : ["ux", "uy", "uz", "rx", "ry", "rz" "fx","fy","fz","cx","cy"]

        dim = self.dim

        if "ux" in result or "fx" in result:
            return 0
        elif ("uy" in result or "fy" in result) and dim >= 2:
            return 1
        elif ("uz" in result or "fz" in result) and dim == 3:
            return 2
        elif ("rx" in result or "cx" in result) and dim == 3:
            return 3
        elif ("ry" in result or "cy" in result) and dim == 3:
            return 4
        elif ("rz" in result or "cz" in result) and dim >= 2:
            if dim == 2:
                return 2
            elif dim == 3:
                return 5
        elif result == "N":
            return 0
        elif result == "Mx" and dim == 3:
            return 1
        elif result == "My" and dim == 3:
            return 2
        elif result == "Mz":
            if dim == 2:
                return 1
            elif dim == 3:
                return 3
                        
        if len(result) == 3 and result[0] == "E":
            # strain case
            indices = result[1:]
            if indices == 'xx':
                return 0
            elif indices == 'xy':
                return -1
            elif indices == 'xz':
                return 2
            elif indices == 'yz':
                return 1
        if len(result) == 3 and result[0] == "S":
            # stress case
            indices = result[1:]
            if indices == 'xx':
                return 0
            elif indices == 'yy':
                return 1
            elif indices == 'zz':
                return 2
            elif indices == 'yz':
                return 3
            elif indices == 'xz':
                return 4
            elif indices == 'xy':
                return -1

    def _Calc_Epsilon_e_pg(self, sol: np.ndarray) -> np.ndarray:
        """Construct deformations for each element and each Gauss point.\n
        a' denotes here da/dx \n
        1D -> [ux']\n
        2D -> [ux', rz']\n
        3D -> [ux', rx', ry', rz']
        """
        
        tic = Tic()

        dof_n = self.structure.dof_n
        assembly_e = self.mesh.groupElem.Get_assembly_e(dof_n)
        sol_e = sol[assembly_e]
        B_beam_e_pg = self.mesh.groupElem.Get_EulerBernoulli_B_e_pg(self.structure)
        Epsilon_e_pg = np.einsum('epij,ej->epi', B_beam_e_pg, sol_e, optimize='optimal')
        
        tic.Tac("Matrix", "Epsilon_e_pg", False)

        return Epsilon_e_pg
    
    def _Calc_InternalForces_e_pg(self, Epsilon_e_pg: np.ndarray) -> np.ndarray:
        """Calculation of internal forces.\n
        1D -> [N]\n
        2D -> [N, Mz]\n
        3D -> [N, Mx, My, Mz]
        """ 
        # .../FEMOBJECT/BASIC/MODEL/MATERIALS/@ELAS_BEAM/sigma.m

        matrixType = MatrixType.beam

        assert Epsilon_e_pg.shape[0] == self.mesh.Ne
        assert Epsilon_e_pg.shape[1] == self.mesh.Get_nPg(matrixType)

        tic = Tic()

        D_e_pg = self.structure.Calc_D_e_pg(self.mesh.groupElem)
        forces_e_pg: np.ndarray = np.einsum('epij,epj->epi', D_e_pg, Epsilon_e_pg, optimize='optimal')
            
        tic.Tac("Matrix", "InternalForces_e_pg", False)

        return forces_e_pg

    def _Calc_Sigma_e_pg(self, Epsilon_e_pg: np.ndarray) -> np.ndarray:
        """Calculates stresses from strains.\n
        1D -> [Sxx]\n
        2D -> [Sxx, Syy, Sxy]\n
        3D -> [Sxx, Syy, Szz, Syz, Sxz, Sxy]
        """
        # .../FEMOBJECT/BASIC/MODEL/MATERIALS/@ELAS_BEAM/sigma.m

        Nn = self.mesh.Nn
        Ne = self.mesh.Ne
        nPg = self.mesh.Get_nPg(MatrixType.beam)

        assert Epsilon_e_pg.shape[0] == Ne
        assert Epsilon_e_pg.shape[1] == nPg

        dim = self.structure.dim

        InternalForces_e_pg = self._Calc_InternalForces_e_pg(Epsilon_e_pg)

        tic = Tic()
        
        S_e_pg = np.zeros((Ne, nPg))
        Iy_e_pg = np.zeros_like(S_e_pg)
        Iz_e_pg = np.zeros_like(S_e_pg)
        J_e_pg = np.zeros_like(S_e_pg)
        mu_e_pg = np.zeros_like(S_e_pg)
        for beam in self.structure.beams:
            elems = self.mesh.Elements_Tags([beam.name])
            S_e_pg[elems] = beam.area
            Iy_e_pg[elems] = beam.Iy
            Iz_e_pg[elems] = beam.Iz
            J_e_pg[elems] = beam.J
            if isinstance(beam, Materials.Beam_Elas_Isot):
                mu_e_pg[elems] = beam.mu

        y_e_pg = np.sqrt(S_e_pg)
        z_e_pg = np.sqrt(S_e_pg)

        N_e_pg = InternalForces_e_pg[:,:,0]

        if dim == 1:
            # [Sxx]
            Sigma_e_pg = np.zeros((Ne, nPg, 1))            
            Sigma_e_pg[:,:,0] = N_e_pg/S_e_pg  # Sxx = N/S
        elif dim == 2:
            # [Sxx, Syy, Sxy]
            # [Sxx, 0, 0] for euler bernouilli
            Sigma_e_pg = np.zeros((Ne, nPg, 3))

            Mz_e_pg = InternalForces_e_pg[:,:,1]
            Sigma_e_pg[:,:,0] = N_e_pg/S_e_pg - (Mz_e_pg*y_e_pg/Iz_e_pg)  # Sxx = N/S - Mz*y/Iz
            Sigma_e_pg[:,:,1] = 0 # Syy = 0
            # Ty = 0 with euler bernoulli beam because uy' = rz
            Sigma_e_pg[:,:,2] = 0 # Sxy = Ty/S il faut calculer Ty
        elif dim == 3:
            # [Sxx, Syy, Szz, Syz, Sxz, Sxy]
            # [Sxx, 0, 0, 0, Sxz, Sxy] for 
            Sigma_e_pg = np.zeros((Ne, nPg, 6))

            Mx_e_pg = InternalForces_e_pg[:,:,1]
            My_e_pg = InternalForces_e_pg[:,:,2]
            Mz_e_pg = InternalForces_e_pg[:,:,3]
            
            Sigma_e_pg[:,:,0] = N_e_pg/S_e_pg + My_e_pg/Iy_e_pg*z_e_pg - Mz_e_pg/Iz_e_pg*y_e_pg # Sxx = N/S + My/Iy*z - Mz/Iz*y
            Sigma_e_pg[:,:,1] = 0 # Syy = 0
            Sigma_e_pg[:,:,2] = 0 # Szz = 0
            Sigma_e_pg[:,:,3] = 0 # Syz = 0
            # Ty = Tz = 0 with euler bernoulli beam
            Sigma_e_pg[:,:,4] = Mx_e_pg/J_e_pg*y_e_pg # Sxz = Tz/S + Mx/Ix*y 
            Sigma_e_pg[:,:,5] = - Mx_e_pg/J_e_pg*z_e_pg # Sxy = Ty/S - Mx/Ix*z

        # xAxis_e, yAxis_e = self.structure.Get_axis_e(self.mesh.groupElem)        
        # d = np.max((2,dim))
        # Ps, Pe = Materials.Get_Pmat(xAxis_e[:,:d], yAxis_e[:,:d], False)
        # Sigma_e_pg = np.einsum('eij,epj->epi',Ps, Sigma_e_pg, optimize='optimal')
            
        tic.Tac("Matrix", "Sigma_e_pg", False)

        return Sigma_e_pg

    def Results_dict_Energy(self) -> dict[str, float]:
        return super().Results_dict_Energy()

    def Results_Iter_Summary(self) -> list[tuple[str, np.ndarray]]:
        return super().Results_Iter_Summary()
    
    def Results_displacement_matrix(self) -> np.ndarray:
        
        Nn = self.mesh.Nn
        dof_n = self.Get_dof_n(self.problemType)        
        displacementRedim = self.displacement.reshape(Nn,-1)

        coordo = np.zeros((Nn, 3))

        if dof_n == 1:
            coordo[:,0] = displacementRedim[:,0]
        elif dof_n == 3:
            coordo[:,:2] = displacementRedim[:,:2]
        elif dof_n == 6:
            coordo[:,:3] = displacementRedim[:,:3]

        return coordo
    
    def Results_Get_Iteration_Summary(self) -> str:

        summary = ""

        # TODO to improve

        # Displacement display
        dx = self.Result("ux", nodeValues=True)
        summary += f"\n\nUx max = {dx.max():.2e}"
        summary += f"\nUx min = {dx.min():.2e}"

        if self.structure.dim > 1:
            dy = self.Result("uy", nodeValues=True)
            summary += f"\n\nUy max = {dy.max():.2e}"
            summary += f"\nUy min = {dy.min():.2e}"

        if self.dim == 3:
            dz = self.Result("uz", nodeValues=True)
            summary += f"\n\nUz max = {dz.max():.2e}"
            summary += f"\nUz min = {dz.min():.2e}"

        return summary