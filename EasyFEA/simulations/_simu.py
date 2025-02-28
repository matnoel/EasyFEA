# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from abc import ABC, abstractmethod
import pickle
from datetime import datetime
from typing import Union
import numpy as np
from scipy import sparse
import textwrap

from ..__about__ import __version__
# utilities
from ..utilities import Folder, Display, Tic
from ..utilities._observers import Observable, _IObserver
# fem
from ..fem import Mesh, MatrixType, BoundaryCondition, LagrangeCondition
# materials
from ..materials import ModelType, _IModel, Reshape_variable
# simu
from .Solvers import _Solve, _Solve_Axb, _Available_Solvers, ResolType, AlgoType

# ----------------------------------------------
# _Simu
# ----------------------------------------------
class _Simu(_IObserver, ABC):
    """
    The following classes inherit from the parent class _Simu:
        - ElasticSimu
        - PhaseFieldSimu
        - BeamSimu
        - ThermalSimu

    To create new simulation classes, take inspiration from existing classes.\n
    You'll need to respect the _Simu interface.\n
    The ThermalSimu class is quite simple to understand, see `simulations/_thermal.py`.\n

    To use the interface/inheritance, 14 methods need to be defined.

    General:
    --------
    
        - def Get_problemTypes(self) -> list[ModelType]:

        - def Get_directions(self, problemType=None) -> list[str]:

        - def Get_dof_n(self, problemType=None) -> int:

        These functions provides access to the available degrees of freedom (dofs).

    Solvers:
    --------

        - def Get_K_C_M_F(self, problemType=None) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:

        - def Get_x0(self, problemType=None):

        - def Assembly(self):

        These functions assemble the matrix system K u + C v + M a = F.

    Iterations:
    -----------

        - def Save_Iter(self) -> None:

        - def Set_Iter(self, index=-1) -> None:

        These functions are used to save or load iterations.

    Results:
    --------

        - def Results_Available(self) -> list[str]:

        - def Result(self, result: str, nodeValues=True, iter=None) -> float | np.ndarray:

        - def Results_Iter_Summary(self) -> tuple[list[int], list[tuple[str, np.ndarray]]]:

        - def Results_dict_Energy(self) -> dict[str, float]:

        - def Results_displacement_matrix(self) -> np.ndarray:

        - def Results_nodesField_elementsField(self, details=False) -> tuple[list[str], list[str]]:

        These functions are used to process the results.
    """

    # ----------------------------------------------
    # Abstract method
    # ----------------------------------------------

    @abstractmethod
    def Get_problemTypes(self) -> list[ModelType]:
        """Returns the problem types available through the simulation."""
        # A PhaseField simulation involves solving 2 problems. An elastic and a damage problem.
        pass

    @abstractmethod
    def Get_dofs(self, problemType=None) -> list[str]:
        """Returns a list of the degrees of freedom available in the simulation."""
        pass

    @abstractmethod
    def Get_dof_n(self, problemType=None) -> int:
        """Returns the degrees of freedom per node."""
        pass

    # Solvers
    @abstractmethod
    def Get_K_C_M_F(self, problemType=None) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        """Returns the assembled matrices of K u + C v + M a = F."""
        pass

    @abstractmethod
    def Get_x0(self, problemType=None) -> np.ndarray:
        """Returns the solution from the previous iteration."""
        return []

    @abstractmethod
    def Assembly(self) -> None:
        """Assembles the matrix system."""
        pass

    # Iterations

    @abstractmethod
    def Save_Iter(self) -> None:
        """Saves iteration results in _results."""
        iter = {}

        iter["indexMesh"] = self.__indexMesh
        # mesh identifier at this iteration

        return iter

    @abstractmethod
    def Set_Iter(self, iter: int=-1, resetAll=False) -> dict:
        """Sets the simulation to the specified iteration (usually the last one) and then reset the required variables if necessary (resetAll).\n
        Returns the simulation results dictionary."""
        
        iter = int(iter)
        assert isinstance(iter, int), "Must provide an integer."

        indexMax = len(self.results) - 1
        assert iter <= indexMax, f"The iter must be < {indexMax}]"

        # Retrieve the results stored in the pandas array
        results = self.results[iter]

        self.__Update_mesh(iter)

        return results.copy()

    # Results

    @abstractmethod
    def Results_Available(self) -> list[str]:
        """Returns a list of available results in the simulation."""
        pass

    @abstractmethod
    def Result(self, option: str, nodeValues=True, iter=None) -> Union[np.ndarray, float]:
        """Returns the result. Use Results_Available() to know the available results."""
        pass

    @abstractmethod
    def Results_Iter_Summary(self) -> tuple[list[int], list[tuple[str, np.ndarray]]]:
        """Returns the values to be displayed in Plot_Iter_Summary."""
        return [], []

    @abstractmethod
    def Results_dict_Energy(self) -> dict[str, float]:
        """Returns a dictionary containing the names and values of the calculated energies."""
        return {}

    @abstractmethod
    def Results_displacement_matrix(self) -> np.ndarray:
        """Returns displacements as a matrix [dx, dy, dz] (Nn,3)."""
        Nn = self.mesh.Nn
        return np.zeros((Nn, 3))

    @abstractmethod
    def Results_nodesField_elementsField(self, details=False) -> tuple[list[str], list[str]]:
        """Returns lists of nodesFields and elementsFields displayed in paraview."""
        return [], []

    # ----------------------------------------------
    # core functions
    # ----------------------------------------------

    def _Check_dofs(self, problemType: ModelType, directions: list) -> None:
        """Checks whether the specified directions are available for the problem."""
        dofs = self.Get_dofs(problemType)
        for d in directions:
            assert d in dofs, f"{d} is not in {dofs}"

    def __Check_problemTypes(self, problemType: ModelType) -> None:
        """Checks whether this type of problem is available through the simulation."""
        assert problemType in self.Get_problemTypes(), f"This type of problem is not available in this simulation ({self.Get_problemTypes()})"

    def _Check_dim_mesh_material(self) -> None:
        """Checks that the material dim matches the mesh dim."""
        dim = self.__model.dim
        assert dim == self.__mesh.dim and dim == self.__mesh.inDim, "Material and mesh must share the same dimensions and belong to the same space."

    def __str__(self) -> str:
        """Returns a string representation of the simulation.

        Returns
        -------
        str
            A string containing information about the simulation.
        """

        text = Display.Section("Mesh", False)
        text += str(self.mesh)

        text += Display.Section("Model", False)
        text += '\n' + str(self.model)
        
        text += '\n\nsolver : ' + str(self.solver)

        text += Display.Section("Boundary Conditions", False)
        text += '\n' + textwrap.dedent(self.Results_Get_Bc_Summary())

        text += Display.Section("Results", False)        
        text += '\n' + textwrap.dedent(self.Results_Get_Iteration_Summary())
        
        text += Display.Section("TicTac", False)
        
        if Tic.nTic() > 0:
            text += Tic.Resume(False)

        return text

    def __init__(self, mesh: Mesh, model: _IModel, verbosity=True, useNumba=True, useIterativeSolvers=True):
        """Creates a simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh used.
        model : _IModel
            The model used.
        verbosity : bool, optional
            If True, the simulation can write in the terminal. Defaults to True.
        useNumba : bool, optional
            If True and numba is installed numba can be used. Defaults to True.
        useIterativeSolvers : bool, optional
            If True, iterative solvers can be used. Defaults to True.
        """
    
        if verbosity:
            Display.Section("Simulation")

        if len(mesh.orphanNodes) > 0:
            raise Exception("The simulation cannot be created because orphan nodes have been detected in the mesh.\n See `Display.Plot_Nodes(mesh, mesh.orphanNodes)`")

        self.__model: _IModel = model

        self.__dim: int = model.dim
        """Simulation dimension."""

        self._results:list[dict] = []
        """Dictionary list containing the results."""

        # Fill in the first mesh
        self.__indexMesh: int = -1
        """Current mesh index in self.__listMesh"""
        self.__listMesh: list[Mesh] = []
        self.mesh = mesh

        self.rho = 1.0        

        self._Check_dim_mesh_material()

        self._verbosity = verbosity
        """The simulation can write in the terminal"""

        self.__algo = AlgoType.elliptic
        """System resolution algorithm during simulation."""
        # Basic algo solves stationary problems

        # Solver used for solving
        self.__solver = "scipy"  # Initialized just in case
        solvers = _Available_Solvers()  # Available solvers
        if "pypardiso" in solvers:
            self.solver = "pypardiso"
        elif "petsc" in solvers and useIterativeSolvers:
            self.solver = "petsc"
        elif useIterativeSolvers:
            self.solver = "cg"

        self.__Init_Sols_n()

        self.useNumba = useNumba

        self.__useIterativeSolvers: bool = useIterativeSolvers

        # Initialize Boundary conditions
        self.Bc_Init()
        
        # simulation will look for material and mesh modifications
        model._Add_observer(self)
        mesh._Add_observer(self)

    @property
    def model(self) -> _IModel:
        """model used"""
        return self.__model

    @property
    def rho(self) -> Union[float, np.ndarray]:
        """mass density"""
        return self.__rho

    @rho.setter
    def rho(self, value: Union[float, np.ndarray]):
        _IModel._Test_Sup0(value)
        self.__rho = value
        """mass density"""

    @property
    def mass(self) -> float:

        if self.dim == 1: return None

        matrixType = MatrixType.mass

        group = self.mesh.groupElem

        jacobian_e_p = group.Get_jacobian_e_pg(matrixType)
        
        weight_p = group.Get_weight_pg(matrixType)        

        rho_e_p = Reshape_variable(self.__rho, self.mesh.Ne, weight_p.size)

        mass = float(np.einsum('ep,ep,p->', rho_e_p, jacobian_e_p, weight_p, optimize='optimal'))

        if self.dim == 2:
            mass *= self.model.thickness

        return mass
    
    @property
    def center(self) -> np.ndarray:
        """Center of mass / barycenter / inertia center"""

        if self.dim == 1: return None

        matrixType = MatrixType.mass

        group = self.mesh.groupElem

        coord_e_p = group.Get_GaussCoordinates_e_p(matrixType)

        jacobian_e_p = group.Get_jacobian_e_pg(matrixType)
        weight_p = group.Get_weight_pg(matrixType)        

        rho_e_p = Reshape_variable(self.__rho, self.mesh.Ne, weight_p.size)
        mass = self.mass

        center: np.ndarray = np.einsum('ep,ep,p,epi->i', rho_e_p, jacobian_e_p, weight_p, coord_e_p, optimize='optimal') / mass

        if self.dim == 2:
            center *= self.model.thickness

        if not isinstance(self.__rho, np.ndarray):
            diff = np.linalg.norm(center - self.mesh.center)/np.linalg.norm(center)
            assert diff <= 1e-12

        return center

    # ----------------------------------------------
    # Solutions
    # ----------------------------------------------    

    # TODO Enable simulation creation from the variational formulation ?

    # Solutions
    @property
    def results(self) -> list[dict]:
        """Returns a copy of the list of dictionary containing the results from each iteration."""
        return self._results.copy()
    
    @property
    def Niter(self) -> int:
        return len(self._results)

    def __Init_Sols_n(self) -> None:
        """Initializes the solutions."""
        self.__dict_u_n = {}
        self.__dict_v_n = {}
        self.__dict_a_n = {}
        for problemType in self.Get_problemTypes():
            size = self.mesh.Nn * self.Get_dof_n(problemType)
            vectInit = np.zeros(size, dtype=float)
            self.__dict_u_n[problemType] = vectInit
            self.__dict_v_n[problemType] = vectInit
            self.__dict_a_n[problemType] = vectInit

    def __Check_New_Sol_Values(self, problemType: ModelType, values: np.ndarray) -> None:
        """Checks that the solution has the right size."""
        self.__Check_problemTypes(problemType)
        size = self.mesh.Nn * self.Get_dof_n(problemType)
        assert values.shape[0] == size, f"Must be size {size}"

    def _Get_u_n(self, problemType: ModelType) -> np.ndarray:
        """Returns the solution associated with the given problem."""
        return self.__dict_u_n[problemType].copy()

    def _Set_u_n(self, problemType: ModelType, values: np.ndarray) -> None:
        """Sets the solution associated with the given problem."""
        self.__Check_New_Sol_Values(problemType, values)
        self.__dict_u_n[problemType] = values

    def _Get_v_n(self, problemType: ModelType) -> np.ndarray:
        """Returns the speed solution associated with the given problem."""
        return self.__dict_v_n[problemType].copy()

    def _Set_v_n(self, problemType: ModelType, values: np.ndarray) -> None:
        """Sets the speed solution associated with the given problem."""
        self.__Check_New_Sol_Values(problemType, values)
        self.__dict_v_n[problemType] = values

    def _Get_a_n(self, problemType: ModelType) -> np.ndarray:
        """Returns the acceleration solution associated with the given problem."""
        return self.__dict_a_n[problemType].copy()

    def _Set_a_n(self, problemType: ModelType, values: np.ndarray) -> None:
        """Sets the acceleration solution associated with the given problem."""
        self.__Check_New_Sol_Values(problemType, values)
        self.__dict_a_n[problemType] = values

    # This method is overloaded in PhaseFieldSimu
    def Get_lb_ub(self, problemType: ModelType) -> tuple[np.ndarray, np.ndarray]:
        """Returns the lower bound and upper bound."""
        return np.array([]), np.array([])

    # Properties
    @property
    def problemType(self) -> ModelType:
        """Get the simulation problem type."""
        return self.__model.modelType

    @property
    def algo(self) -> AlgoType:
        """The algorithm used to solve the problem.\n
        (elliptic, parabolic, hyperbolic) see:\n
        - Solver_Set_Elliptic_Algorithm()\n
        K u = F
        - Solver_Set_Parabolic_Algorithm()\n
        K u + C v = F
        - Solver_Set_Newton_Raphson_Algorithm()\n
        K u + C v + M a = F
        """        
        return self.__algo

    @property
    def mesh(self) -> Mesh:
        """simulation's mesh."""
        return self.__mesh

    @mesh.setter
    def mesh(self, mesh: Mesh):        
        if isinstance(mesh, Mesh):
            # For all old meshes, delete the matrices
            listMesh: list[Mesh] = self.__listMesh
            [m._ResetMatrix() for m in listMesh]

            self.__indexMesh += 1
            self.__listMesh.append(mesh)
            self.__mesh = mesh

            # The mesh changes, so the matrices must be reconstructed
            self.Need_Update()
            # Initialize boundary conditions
            self.Bc_Init()
            # initialize the solutions
            self.__Init_Sols_n()

    @property
    def dim(self) -> int:
        """simulation's dimension"""
        return self.__dim

    @property
    def useNumba(self) -> bool:
        """the simulation can use numba functions"""
        return self.__useNumba

    @useNumba.setter
    def useNumba(self, value: bool):
        value = value
        self.__model.useNumba = value
        self.__useNumba = value

    def __Update_mesh(self, iter: int) -> None:
        """Updates the mesh for the specified iteration.

        Parameters
        ----------
        iter : int
            The iteration number to update the mesh.
        """
        indexMesh = self.results[iter]["indexMesh"]
        self.__mesh = self.__listMesh[indexMesh]
        self.Need_Update() # need to reconstruct matrices

    @property
    def needUpdate(self) -> bool:
        """The simulation needs to reconstruct matrices K, C, and M."""
        return self.__needUpdate
    
    def _Update(self, observable: Observable, event: str) -> None:
        if isinstance(observable, _IModel):
            if event == 'The model has been modified' and not self.needUpdate:
                self.Need_Update()
        elif isinstance(observable, Mesh):
            if event == 'The mesh has been modified':
                self._Check_dim_mesh_material()
                self.Need_Update()
        else:
            Display.MyPrintError("Notification not yet implemented")

    def Need_Update(self, value=True) -> None:
        """Sets whether the simulation needs to reconstruct matrices K, C, M and F."""
        self.__needUpdate = value

    # ----------------------------------------------
    # Solver
    # ----------------------------------------------

    @property
    def useIterativeSolvers(self) -> bool:
        """Iterative solvers can be used."""
        return self.__useIterativeSolvers

    @property
    def solver(self) -> str:
        """Solver used to solve the simulation."""
        return self.__solver

    @solver.setter
    def solver(self, value: str):

        # Retrieve usable solvers
        solvers = _Available_Solvers()

        if self.problemType != "damage":
            solvers.remove("BoundConstrain")

        if value in solvers:
            self.__solver = value
        else:
            Display.MyPrintError(f"The solver {value} cannot be used. The solver must be in {solvers}")

    def Solver_Set_Elliptic_Algorithm(self) -> None:
        """Sets the algorithm's resolution properties for an elliptic problem.

        Used to solve K u = F.
        """
        self.__algo = AlgoType.elliptic

    def Solver_Set_Parabolic_Algorithm(self, dt: float, alpha=1/2) -> None:
        """Sets the algorithm's resolution properties for a parabolic problem.

        Used to solve K u + C v = F.

        Parameters
        ----------
        dt : float
            The time increment.
        alpha : float, optional
            The alpha criterion, by default 1/2\n
            - 0 -> Forward Euler
            - 1 -> Backward Euler
            - 1/2 -> midpoint
        """
        self.__algo = AlgoType.parabolic

        assert dt > 0, "Time increment must be > 0"

        self.alpha = alpha
        self.dt = dt

    def Solver_Set_Newton_Raphson_Algorithm(self, dt: float, betha=1/4, gamma=1/2) -> None:
        """Sets the algorithm's resolution properties for a Newton-Raphson problem.

        Used to solve K u + C v + M a = F.

        Parameters
        ----------
        dt : float
            The time increment.
        betha : float, optional
            The coefficient betha, by default 1/4.
        gamma : float, optional
            The coefficient gamma, by default 1/2.
        """
        self.__algo = AlgoType.hyperbolic

        assert dt > 0, "Time increment must be > 0"

        self.betha = betha
        self.gamma = gamma
        self.dt = dt

    def Solve(self) -> np.ndarray:
        """Computes the solution field for the current boundary conditions.

        Returns
        -------
        np.ndarray
            The solution of the simulation.
        """        

        self._Solver_Solve(self.problemType)

        return self._Get_u_n(self.problemType)

    def _Solver_Solve(self, problemType: ModelType) -> None:
        """Solves the problem."""

        # Here you need to specify the type of problem because a simulation can have several physical models

        algo = self.__algo

        # Old solution
        u_n = self._Get_u_n(problemType)
        v_n = self._Get_v_n(problemType)
        a_n = self._Get_a_n(problemType)

        if len(self.Bc_Lagrange) > 0:
            # Lagrange conditions are applied.
            resolution = ResolType.r2
            x, lagrange = _Solve(self, problemType, resolution)
        else:
            resolution = ResolType.r1
            x = _Solve(self, problemType, resolution)            

        if algo == AlgoType.elliptic:
            u_np1 = x
            self._Set_u_n(problemType, u_np1)

        if algo == AlgoType.parabolic:
            # See Hughes 1987 Chapter 7

            u_np1 = x

            alpha = self.alpha
            dt = self.dt

            v_Tild_np1 = u_n + ((1 - alpha) * dt * v_n)
            v_np1 = (u_np1 - v_Tild_np1) / (alpha * dt)

            # New solutions
            self._Set_u_n(problemType, u_np1)
            self._Set_v_n(problemType, v_np1)

        elif algo == AlgoType.hyperbolic:
            # Accel formulation
            # See Hughes 1987 Chapter 7 or Pled 2020 3.7.3

            a_np1 = x

            dt = self.dt
            gamma = self.gamma
            betha = self.betha

            u_Tild_np1 = u_n + (dt * v_n) + dt**2/2 * (1 - 2 * betha) * a_n
            v_Tild_np1 = v_n + (1 - gamma) * dt * a_n

            u_np1 = u_Tild_np1 + betha * dt**2 * a_np1
            v_np1 = v_Tild_np1 + gamma * dt * a_np1

            # New solutions
            self._Set_u_n(problemType, u_np1)
            self._Set_v_n(problemType, v_np1)
            self._Set_a_n(problemType, a_np1)

    def _Solver_Apply_Neumann(self, problemType: ModelType) -> sparse.csr_matrix:
        """Fill in the Neumann boundary conditions by constructing b from A x = b.

        Parameters
        ----------
        problemType : ModelType
            problem type

        Returns
        -------
        sparse.csr_matrix
            The b vector as a csr_matrix.
        """
        
        algo = self.algo
        dofs = BoundaryCondition.Get_dofs(problemType, self.__Bc_Neumann)
        dofsValues = BoundaryCondition.Get_values(problemType, self.__Bc_Neumann)
        Ndof = self.mesh.Nn * self.Get_dof_n(problemType)

        # Additional dimension associated with the lagrangian multipliers
        Ndof += self._Bc_Lagrange_dim(problemType)

        b = sparse.csr_matrix((dofsValues, (dofs, np.zeros(len(dofs)))), shape=(Ndof, 1))

        K, C, M, F = self.Get_K_C_M_F(problemType)

        tic = Tic()

        u_n = self._Get_u_n(problemType)
        v_n = self._Get_v_n(problemType)
        a_n = self._Get_a_n(problemType)

        b = b + F

        if algo == AlgoType.parabolic:

            alpha = self.alpha
            dt = self.dt

            v_Tild_np1 = u_n + (1 - alpha) * dt * v_n
            v_Tild_np1 = sparse.csr_matrix(v_Tild_np1.reshape(-1, 1))

            b = b + C.dot(v_Tild_np1 / (alpha * dt))

        elif algo == AlgoType.hyperbolic:
            # Accel formulation

            if len(self.results) == 0 and (b.max() != 0 or b.min() != 0):
                # Initialize accel
                __, dofsUnknown = self.Bc_dofs_known_unknow(problemType)

                # don't change
                bb = b - K.dot(sparse.csr_matrix(u_n.reshape(-1, 1)))
                bb -= C.dot(sparse.csr_matrix(v_n.reshape(-1, 1)))                

                bbi = bb[dofsUnknown]
                Aii = M[dofsUnknown, :].tocsc()[:, dofsUnknown].tocsr()

                x0 = a_n[dofsUnknown]

                ai_n = _Solve_Axb(self, problemType, Aii, bbi, x0, [], [])

                a_n[dofsUnknown] = ai_n

                self._Set_a_n(problemType, a_n)

            a_n = self._Get_a_n(problemType)

            dt = self.dt
            gamma = self.gamma
            betha = self.betha

            uTild_np1 = u_n + (dt * v_n) + dt**2/2 * (1 - 2 * betha) * a_n
            vTild_np1 = v_n + (1 - gamma) * dt * a_n

            # dont change
            b -= K.dot(uTild_np1.reshape(-1, 1))
            b -= C.dot(vTild_np1.reshape(-1, 1))
            b = sparse.csr_matrix(b)

        tic.Tac("Solver", f"Neumann ({problemType}, {algo})", self._verbosity)

        return b

    def _Solver_Apply_Dirichlet(self, problemType: ModelType, b: sparse.csr_matrix, resolution: ResolType) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Fill in the Dirichlet conditions by constructing A and x from A x = b.

        Parameters
        ----------
        problemType : ModelType
            The type of problem.
        b : sparse.csr_matrix
            The b matrix.
        resolution : ResolutionType
            The resolution type.

        Returns
        -------
        tuple[sparse.csr_matrix, sparse.csr_matrix]
            The A and x matrices.
        """
        tic = Tic()

        algo = self.algo
        dofs = self.Bc_dofs_Dirichlet(problemType)
        dofsValues = self.Bc_values_Dirichlet(problemType)

        K, C, M, F = self.Get_K_C_M_F(problemType)

        if algo == AlgoType.elliptic:
            A = K

        if algo == AlgoType.parabolic:

            alpha = self.alpha
            dt = self.dt

            # Resolution in position
            A = K + C / (alpha * dt)

            # # Speed resolution
            # A = K * alpha * dt + M

        elif algo == AlgoType.hyperbolic:

            dt = self.dt
            gamma = self.gamma
            betha = self.betha

            # Accel formulation
            A = M + (K * betha * dt**2)
            A += (gamma * dt * C)

            solDotDot_n = self._Get_a_n(problemType)
            dofsValues = solDotDot_n[dofs]

        A, x = self.__Solver_Get_Dirichlet_A_x(problemType, resolution, A, b, dofsValues)

        tic.Tac("Solver", f"Dirichlet ({problemType}, {algo})", self._verbosity)

        return A, x

    def __Solver_Get_Dirichlet_A_x(self, problemType: ModelType, resolution: ResolType, A: sparse.csr_matrix, b: sparse.csr_matrix, dofsValues: np.ndarray):
        """Resizes the matrix system according to known degrees of freedom and resolution type.

        Parameters
        ----------
        problemType : ModelType
            The type of problem.
        resolution : ResolutionType
            The resolution type.
        A : sparse.csr_matrix
            The A matrix.
        b : sparse.csr_matrix
            The b matrix.
        dofsValues : np.ndarray
            The values of known degrees of freedom.

        Returns
        -------
        sparse.csr_matrix
            The A matrix after resizing.
        sparse.csr_matrix
            The x matrix after resizing.
        """
        dofs = self.Bc_dofs_Dirichlet(problemType)
        size = self.mesh.Nn * self.Get_dof_n(problemType)

        if resolution in [ResolType.r1, ResolType.r2]:

            # Here we return the solution with the known ddls
            x = sparse.csr_matrix((dofsValues, (dofs, np.zeros(len(dofs)))), shape=(size, 1), dtype=np.float64)

            return A, x

        elif resolution == ResolType.r3:
            # Penalization

            A = A.tolil()
            b = b.tolil()

            # Penalization A
            A[dofs] = 0.0
            A[dofs, dofs] = 1

            # Penalization b
            b[dofs] = dofsValues

            # Here we return A penalized
            return A.tocsr(), b.tocsr()
    
    # ----------------------------------------------
    # Boundary conditions
    # ----------------------------------------------
    
    def Bc_Init(self) -> None:
        """Initializes Dirichlet, Neumann and Lagrange boundary conditions"""
        # DIRICHLET
        self.__Bc_Dirichlet: list[BoundaryCondition] = []
        """Dirichlet conditions list[BoundaryCondition]"""
        # NEUMANN
        self.__Bc_Neumann: list[BoundaryCondition] = []
        """Neumann conditions list[BoundaryCondition]"""
        # LAGRANGE
        self.__Bc_Lagrange: list[LagrangeCondition] = []
        """Lagrange conditions list[BoundaryCondition]"""
        self.__Bc_Display = []
        """Boundary conditions for display list[BoundaryCondition]"""

    @property
    def Bc_Dirichlet(self) -> list[BoundaryCondition]:
        """Returns a copy of the Dirichlet conditions."""
        return self.__Bc_Dirichlet.copy()
    
    @property
    def Bc_Neuman(self) -> list[BoundaryCondition]:
        """Returns a copy of the Neumann conditions."""
        return self.__Bc_Neumann.copy()

    @property
    def Bc_Lagrange(self) -> list[LagrangeCondition]:
        """Returns a copy of the Lagrange conditions."""
        return self.__Bc_Lagrange.copy()
    
    def _Bc_Add_Lagrange(self, newBc: LagrangeCondition):
        """Adds Lagrange conditions."""
        assert isinstance(newBc, LagrangeCondition)
        self.__Bc_Lagrange.append(newBc)
        # triger the update because when we use lagrange multiplier we need to update the matrix system
        self.Need_Update()

    def _Bc_Lagrange_dim(self, problemType=None) -> int:
        """Calculates the dimension required to resize the system to use Lagrange multipliers."""
        if problemType is None:
            problemType = self.problemType
        # get the number of lagrange conditions applied to the problem
        # len(self.Bc_Lagrange) won't work because we need to filter the problemType
        nBc = BoundaryCondition.Get_nBc(problemType, self.Bc_Lagrange)
        if nBc > 0:
            nBc += len(self.Bc_dofs_Dirichlet(problemType))
        return nBc

    @property
    def Bc_Display(self) -> list[LagrangeCondition]:
        """Returns a copy of the boundary conditions for display."""
        return self.__Bc_Display.copy()

    def Bc_dofs_Dirichlet(self, problemType=None) -> list[int]:
        """Returns dofs related to Dirichlet conditions."""
        if problemType is None:
            problemType = self.problemType
        return BoundaryCondition.Get_dofs(problemType, self.__Bc_Dirichlet)

    def Bc_values_Dirichlet(self, problemType=None) -> list[float]:
        """Returns dofs values related to Dirichlet conditions."""
        if problemType is None:
            problemType = self.problemType
        return BoundaryCondition.Get_values(problemType, self.__Bc_Dirichlet)

    def Bc_dofs_known_unknow(self, problemType: ModelType) -> tuple[np.ndarray, np.ndarray]:
        """Returns known and unknown dofs."""
        tic = Tic()

        # Build known dofs
        dofsKnown = []

        dofsKnown = set(self.Bc_dofs_Dirichlet(problemType))

        # Build unknown dofs
        nDof = self.mesh.Nn * self.Get_dof_n(problemType)

        dofsUnknown = set(range(nDof)) - dofsKnown

        dofsKnown = np.asarray(list(dofsKnown), dtype=int)
        dofsUnknown = np.asarray(list(dofsUnknown), dtype=int)
        
        test = dofsKnown.size + dofsUnknown.size
        assert test == nDof, f"Problem under conditions dofsKnown + dofsUnknown - nDof = {test-nDof}"

        tic.Tac("Solver",f"Get dofs ({problemType})", self._verbosity)

        return dofsKnown, dofsUnknown

    def Bc_dofs_nodes(self, nodes: np.ndarray, directions: list[str], problemType=None) -> np.ndarray:
        """Returns degrees of freedom associated with the nodes, based on the problem type and directions.

        Parameters
        ----------
        nodes : np.ndarray
            nodes.
        directions : list
            directions (e.g ["x","y","rz"])
        problemType : str
            Problem type.

        Returns
        -------
        np.ndarray
            Degrees of freedom.
        """

        if problemType is None:
            problemType = self.problemType

        self.__Check_problemTypes(problemType)
        
        assert len(nodes) > 0, "Empty node list" 

        availableDirections = self.Get_dofs(problemType)
        
        return BoundaryCondition.Get_dofs_nodes(availableDirections, nodes, directions)

    def __Bc_evaluate(self, coordo: np.ndarray, values, option="nodes") -> np.ndarray:
        """Evaluates values at nodes or gauss points."""
        
        assert option in ["nodes","gauss"], f"Must be in ['nodes','gauss']"
        if option == "nodes":
            values_eval = np.zeros(coordo.shape[0])
        elif option == "gauss":
            values_eval = np.zeros((coordo.shape[0],coordo.shape[1]))
        
        if callable(values):
            # Evaluate function at coordinates   
            if option == "nodes":
                values_eval[:] = values(coordo[:,0], coordo[:,1], coordo[:,2])
            elif option == "gauss":
                values_eval[:,:] = values(coordo[:,:,0], coordo[:,:,1], coordo[:,:,2])
            
        else:            
            if option == "nodes":
                values_eval[:] = values
            elif option == "gauss":
                values_eval[:,:] = values

        return values_eval
    
    def add_dirichlet(self, nodes: np.ndarray, values: list, directions: list[str], problemType=None, description="") -> None:
        """Adds Dirichlet's boundary conditions.

        Parameters
        ----------
        nodes : np.ndarray
            nodes
        values : list
            list of values that can contains floats, arrays or functions or functions.\n
            e.g [10, lambda x,y,z: 10*x - 20*y + x*z, np.ndarray]\n
            The functions use the x, y and z nodes coordinates.\n
            Please note that the functions must take 3 input parameters in the order x, y, z, whether the problem is 1D, 2D or 3D.
        directions : list[str]
            directions where values will be applied (e.g ['y', 'x'])
        problemType : ModelType, optional
            problem type, if not specified, we take the basic problem of the problem
        description : str, optional
            Description of the condition, by default "".
        """

        if len(values) == 0 or len(values) != len(directions): return        

        if problemType is None:
            problemType = self.problemType

        self.__Check_problemTypes(problemType)
        
        assert len(nodes) > 0, "Empty node list"
        nodes = np.asarray(nodes)

        Nn = nodes.shape[0]
        coordo = self.mesh.coord
        coordo_n = coordo[nodes]

        # initialize the value vector for each nodes
        dofsValues_dir = np.zeros((Nn, len(directions)))        

        for d, dir in enumerate(directions):
            eval_n = self.__Bc_evaluate(coordo_n, values[d], option="nodes")
            dofsValues_dir[:,d] = eval_n.ravel()
        
        dofsValues = dofsValues_dir.ravel()
        
        dofs = self.Bc_dofs_nodes(nodes, directions, problemType)

        self.__Bc_Add_Dirichlet(problemType, nodes, dofsValues, dofs, directions, description)

    def add_neumann(self, nodes: np.ndarray, values: list, directions: list[str], problemType=None, description="") -> None:
        """Adds Neumann's boundary conditions.

        Parameters
        ----------
        nodes : np.ndarray
            nodes
        values : list
            list of values that can contains floats, arrays or functions or functions.\n
            e.g [10, lambda x,y,z: 10*x - 20*y + x*z, np.ndarray]\n
            The functions use the x, y and z nodes coordinates.\n
            Please note that the functions must take 3 input parameters in the order x, y, z, whether the problem is 1D, 2D or 3D.
        directions : list[str]
            directions where values will be applied (e.g ['y', 'x'])
        problemType : ModelType, optional
            problem type, if not specified, we take the basic problem of the problem
        description : str, optional
            Description of the condition, by default "".
        """
        
        if len(values) == 0 or len(values) != len(directions): return

        if problemType is None:
            problemType = self.problemType

        self.__Check_problemTypes(problemType)

        dofsValues, dofs = self.__Bc_pointLoad(problemType, nodes, values, directions)

        self.__Bc_Add_Neumann(problemType, nodes, dofsValues, dofs, directions, description)
        
    def add_lineLoad(self, nodes: np.ndarray, values: list, directions: list[str], problemType=None, description="") -> None:
        """Adds a linear load.

        Parameters
        ----------
        nodes : np.ndarray
            nodes
        values : list
            list of values that can contain floats, arrays or functions or lambda functions.\n
            e.g = [10, lambda x,y,z: 10*x - 20*y + x*z, np.ndarray] \n
            functions use x, y and z integration points coordinates (x,y,z are in this case arrays of dim (e,p)) \n
            Please note that the functions must take 3 input parameters in the order x, y, z, whether the problem is 1D, 2D or 3D.
        directions : list[str]
            directions where values will be applied (e.g ['y', 'x'])
        problemType : ModelType, optional
            problem type, if not specified, we take the basic problem of the problem
        description : str, optional
            Description of the condition, by default "".
        """

        if len(values) == 0 or len(values) != len(directions): return

        if problemType is None:
            problemType = self.problemType

        self.__Check_problemTypes(problemType)

        dofsValues, dofs, nodes = self.__Bc_lineLoad(problemType, nodes, values, directions)

        self.__Bc_Add_Neumann(problemType, nodes, dofsValues, dofs, directions, description)

    def add_surfLoad(self, nodes: np.ndarray, values: list, directions: list[str], problemType=None, description="") -> None:
        """Adds a surface load.
        
        Parameters
        ----------
        nodes : np.ndarray
            nodes
        values : list
            list of values that can contain floats, arrays or functions or lambda functions.\n
            e.g = [10, lambda x,y,z: 10*x - 20*y + x*z, np.ndarray] \n
            functions use x, y and z integration points coordinates (x,y,z are in this case arrays of dim (e,p)) \n
            Please note that the functions must take 3 input parameters in the order x, y, z, whether the problem is 1D, 2D or 3D.
        directions : list[str]
            directions where values will be applied (e.g ['y', 'x'])
        problemType : ModelType, optional
            problem type, if not specified, we take the basic problem of the problem
        description : str, optional
            Description of the condition, by default "".
        """

        if len(values) == 0 or len(values) != len(directions): return

        if problemType is None:
            problemType = self.problemType

        self.__Check_problemTypes(problemType)
            
        if self.__dim == 2:
            dofsValues, dofs, nodes = self.__Bc_lineLoad(problemType, nodes, values, directions)
            # multiplied by thickness
            dofsValues *= self.model.thickness
        elif self.__dim == 3:
            dofsValues, dofs, nodes = self.__Bc_surfload(problemType, nodes, values, directions)

        self.__Bc_Add_Neumann(problemType, nodes, dofsValues, dofs, directions, description)

    def add_pressureLoad(self, nodes: np.ndarray, magnitude: float, problemType=None, description="") -> None:
        """Adds a pressure.

        Parameters
        ----------
        nodes : np.ndarray
            nodes. (must belong to the edge of the mesh.)
        magnitude : float
            pressure magnitude
        problemType : str, optional
            problem type, if not specified, we take the basic problem of the problem
        description : str, optional
            Description of the condition, by default "".
        """

        if problemType is None:
            problemType = self.problemType

        self.__Check_problemTypes(problemType)

        if self.dim == 1:
            Display.MyPrintError("Cant apply pressure on 1D mesh.")
            return

        if len(self.Get_dofs(problemType)) == 0:
            Display.MyPrintError("Cant apply pressure on scalar problems.")
            return

        dofsValues, dofs, nodes = self.__Bc_pressureload(problemType, nodes, magnitude)

        directions = self.Get_dofs(problemType)[:self.mesh.inDim]

        self.__Bc_Add_Neumann(problemType, nodes, dofsValues, dofs, directions, description)

    def add_volumeLoad(self, nodes: np.ndarray, values: list, directions: list[str], problemType=None, description="") -> None:
        """Adds a volumetric load.
        
        Parameters
        ----------
        nodes : np.ndarray
            nodes
        values : list
            list of values that can contain floats, arrays or functions or lambda functions.\n
            e.g = [10, lambda x,y,z: 10*x - 20*y + x*z, np.ndarray] \n
            functions use x, y and z integration points coordinates (x,y,z are in this case arrays of dim (e,p)) \n
            Please note that the functions must take 3 input parameters in the order x, y, z, whether the problem is 1D, 2D or 3D.
        directions : list[str]
            directions where values will be applied (e.g ['y', 'x'])
        problemType : ModelType, optional
            problem type, if not specified, we take the basic problem of the problem
        description : str, optional
            Description of the condition, by default "".
        """
        
        if len(values) == 0 or len(values) != len(directions): return

        if problemType is None:
            problemType = self.problemType

        self.__Check_problemTypes(problemType)
        
        if self.__dim == 2:
            dofsValues, dofs, nodes = self.__Bc_surfload(problemType, nodes, values, directions)
            # multiplied by thickness
            dofsValues = dofsValues*self.model.thickness
        elif self.__dim == 3:
            dofsValues, dofs, nodes = self.__Bc_volumeload(problemType, nodes, values, directions)

        self.__Bc_Add_Neumann(problemType, nodes, dofsValues, dofs, directions, description)

    def __Bc_pointLoad(self, problemType: ModelType, nodes: np.ndarray, values: list, directions: list) -> tuple[np.ndarray , np.ndarray]:
        """Adds a point load."""

        Nn = nodes.shape[0]
        coordo = self.mesh.coordGlob
        coordo_n = coordo[nodes]

        # initialize the value vector for each node
        valeurs_ddl_dir = np.zeros((Nn, len(directions)))

        for d, dir in enumerate(directions):
            eval_n = self.__Bc_evaluate(coordo_n, values[d], option="nodes")
            if problemType == ModelType.beam:
                eval_n /= len(nodes)
            valeurs_ddl_dir[:,d] = eval_n.ravel()
        
        dofsValues = valeurs_ddl_dir.ravel()

        dofs = self.Bc_dofs_nodes(nodes, directions, problemType)

        return dofsValues, dofs

    def __Bc_Integration_Dim(self, dim: int, problemType: ModelType, nodes: np.ndarray, values: list, directions: list) -> tuple[np.ndarray , np.ndarray]:
        """Integrates on elements for the specified dimension."""

        dofsValues = np.array([])
        dofs = np.array([], dtype=int)
        Nodes = np.array([], dtype=int) # Nodes used by the elements
        # nodes != Nodes dont remove

        dof_n = self.Get_dof_n(problemType)

        Nn = self.mesh.Nn

        # For each group element
        for groupElem in self.mesh.Get_list_groupElem(dim):

            # Retrieve elements that exclusively use nodes
            elements = groupElem.Get_Elements_Nodes(nodes, exclusively=True)
            if elements.shape[0] == 0: continue
            connect = groupElem.connect[elements]
            Ne = elements.shape[0]
            Nodes = np.append(Nodes, np.reshape(connect, -1))

            # Get the coordinates of the Gauss points if you need to devaluate the function
            matrixType = MatrixType.mass
            coordo_e_p = groupElem.Get_GaussCoordinates_e_p(matrixType, elements)

            N_pg = groupElem.Get_N_pg(matrixType)

            # integration objects
            jacobian_e_pg = groupElem.Get_jacobian_e_pg(matrixType)[elements]
            gauss = groupElem.Get_gauss(matrixType)
            weight_pg = gauss.weights

            # initialize the matrix of values for each node used by the elements and each gauss point (Ne*nPe, dir)
            values_dofs_dir = np.zeros((Ne*groupElem.nPe, len(directions)))
            # initialize the dofs vector
            new_dofs = np.zeros_like(values_dofs_dir, dtype=int)

            # Integrated in every direction
            for d, dir in enumerate(directions):

                if isinstance(values[d], (int, float)) or callable(values[d]):
                    # evaluate on gauss points
                    eval_e_p = self.__Bc_evaluate(coordo_e_p, values[d], option="gauss")
                    # integrate the elements
                    values_e_p = np.einsum('ep,p,ep,pij->epij', jacobian_e_pg, weight_pg, eval_e_p, N_pg, optimize='optimal')

                else:
                    eval_n = np.zeros(Nn, dtype=float)
                    eval_n[nodes] = values[d]
                    eval_e = eval_n[groupElem.connect[elements]]

                    # integrate the elements
                    values_e_p = np.einsum('ep,p,ej,pij->epij', jacobian_e_pg, weight_pg, eval_e, N_pg, optimize='optimal')

                # sum over integration points
                values_e = np.sum(values_e_p, axis=1)
                # set calculated values and dofs
                values_dofs_dir[:,d] = values_e.ravel()
                new_dofs[:,d] = self.Bc_dofs_nodes(connect.ravel(), [dir], problemType)

            new_values_dofs = values_dofs_dir.ravel() # Put in vector form
            dofsValues = np.append(dofsValues, new_values_dofs)
            
            new_dofs = new_dofs.ravel() # Put in vector form
            dofs = np.append(dofs, new_dofs)

        return dofsValues, dofs, Nodes

    def __Bc_lineLoad(self, problemType: ModelType, nodes: np.ndarray, values: list, directions: list) -> tuple[np.ndarray , np.ndarray, np.ndarray]:
        """Adds a linear load.\n
        returns dofsValues, dofs, nodes"""
        
        self._Check_dofs(problemType, directions)

        dofsValues, dofs, nodes = self.__Bc_Integration_Dim(dim=1, problemType=problemType, nodes=nodes, values=values, directions=directions)

        return dofsValues, dofs, nodes
    
    def __Bc_surfload(self, problemType: ModelType, nodes: np.ndarray, values: list, directions: list) -> tuple[np.ndarray , np.ndarray, np.ndarray]:
        """Apply a surface force.\n
        returns dofsValues, dofs, nodes"""
        
        self._Check_dofs(problemType, directions)

        dofsValues, dofs, nodes = self.__Bc_Integration_Dim(dim=2, problemType=problemType, nodes=nodes, values=values, directions=directions)

        return dofsValues, dofs, nodes

    def __Bc_volumeload(self, problemType: ModelType, nodes: np.ndarray, values: list, directions: list) -> tuple[np.ndarray , np.ndarray, np.ndarray]:
        """Adds a volumetric load.\n
        returns dofsValues, dofs, nodes"""
        
        self._Check_dofs(problemType, directions)

        dofsValues, dofs, nodes = self.__Bc_Integration_Dim(dim=3, problemType=problemType, nodes=nodes, values=values, directions=directions)

        return dofsValues, dofs, nodes
    
    def __Bc_pressureload(self, problemType: ModelType, nodes: np.ndarray, magnitude: float) -> tuple[np.ndarray , np.ndarray, np.ndarray]:
        """Adds a pressure load.\n
        returns dofsValues, dofs, nodes"""
        
        # here we need to get the normal vector

        mesh = self.mesh
        dim = mesh.dim
        inDim = mesh.inDim

        if dim == 2:
            # will use 1D elements
            magnitude *= self.model.thickness
        else:
            magnitude *= -1

        normals, nodes = mesh.Get_normals(nodes)

        values = [val*magnitude for val in normals[:,:inDim].T]

        directions = self.Get_dofs(problemType)[:inDim]

        dofsValues, dofs, nodes = self.__Bc_Integration_Dim(dim-1, problemType=problemType, nodes=nodes, values=values, directions=directions)

        return dofsValues, dofs, nodes
    
    def __Bc_Add_Neumann(self, problemType: ModelType, nodes: np.ndarray, dofsValues: np.ndarray, dofs: np.ndarray, directions: list, description="") -> None:
        """Adds Neumann's boundary conditions.\n
        If a neumann condition is already applied to the dof, the condition will not be taken into account for the dof."""

        tic = Tic()

        self._Check_dofs(problemType, directions)

        new_Bc = BoundaryCondition(problemType, nodes, dofs, directions, dofsValues, f'Neumann {description}')
        self.__Bc_Neumann.append(new_Bc)

        tic.Tac("Boundary Conditions","Add Neumann condition ", self._verbosity)   
     
    def __Bc_Add_Dirichlet(self, problemType: ModelType, nodes: np.ndarray, dofsValues: np.ndarray, dofs: np.ndarray, directions: list, description="") -> None:
        """Adds Dirichlet's boundary conditions.\n
        If a Dirichlet's dof is entered more than once, the conditions are added together."""

        tic = Tic()

        self.__Check_problemTypes(problemType)

        new_Bc = BoundaryCondition(problemType, nodes, dofs, directions, dofsValues, f'Dirichlet {description}')

        self.__Bc_Dirichlet.append(new_Bc)

        tic.Tac("Boundary Conditions","Add Dirichlet condition", self._verbosity)
    
    # Functions to create links between degrees of freedom

    def _Bc_Add_Display(self, nodes: np.ndarray, directions: list[str], description: str, problemType=None) -> None:
        """Adds a display condition."""

        if problemType is None:
            problemType = self.problemType

        self.__Check_problemTypes(problemType)        

        dofs = self.Bc_dofs_nodes(nodes, directions, problemType)
        
        dofsValues =  np.array([0]*len(dofs))

        new_Bc = BoundaryCondition(problemType, nodes, dofs, directions, dofsValues, description)
        self.__Bc_Display.append(new_Bc)

    def Get_contact(self, masterMesh: Mesh, slaveNodes: np.ndarray=None, masterNodes: np.ndarray=None) -> tuple[np.ndarray, np.ndarray]:
        """Returns the simulation nodes detected in the master mesh with the associated displacement matrix to the interface.

        Parameters
        ----------
        masterMesh : Mesh
            master mesh
        slaveNodes : np.ndarray, optional
            slave nodes, by default None
        masterNodes : np.ndarray, optional
            master nodes, by default None

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            nodes, displacementMatrix
        """

        tic = Tic()

        assert self.mesh.dim == masterMesh.dim, "Must be same dimension"

        # Here the first element group is selected. Regardless of whether there are several group of the same dimension.
        masterGroup = masterMesh.Get_list_groupElem(masterMesh.dim-1)[0]
        # retrieve bounndary elements
        if masterNodes is None:
            elements = masterMesh.Elements_Nodes(masterGroup.nodes, False)
        else:
            elements = masterMesh.Elements_Nodes(masterNodes, False)        

        if slaveNodes is None:
            slaveGroup = self.mesh.Get_list_groupElem(masterMesh.dim-1)[0]
            slaveNodes = slaveGroup.nodes

        # update nodes coordinates
        newCoord = self.Results_displacement_matrix() + self.mesh.coord
        
        # get nodes in master mesh
        idx = masterMesh.groupElem.Get_Mapping(newCoord[slaveNodes], elements, False)[0]        
        idx = np.asarray(list(set(idx)), dtype=int)

        tic.Tac("PostProcessing","Get slave nodes in master mesh")

        if idx.size > 0:
            # slave nodes have been detected in the master mesh
            nodes: np.ndarray = slaveNodes[idx]

            sysCoord_e = masterGroup.sysCoord_e

            # get the elemGroup on the interface        
            gaussCoord_e_p = masterGroup.Get_GaussCoordinates_e_p(MatrixType.rigi)
            
            # empty new displacement
            listU: list[np.ndarray] = []
            # for each nodes in master mesh we will detects the shortest displacement vector to the interface
            for node in nodes:
                # vectors between the interface coordinates and the detected node
                vi_e_pg: np.ndarray  = gaussCoord_e_p - newCoord[node]               

                # distance between the interface coordinates and the detected node
                d_e_pg: np.ndarray = np.linalg.norm(vi_e_pg, axis=2)
                e, p = np.unravel_index(np.argmin(d_e_pg), d_e_pg.shape)
                # retrieves the nearest coordinate
                closeCoordo = np.reshape(gaussCoord_e_p[e,p], -1)
                
                # normal vector
                if masterGroup.dim == 1: # lines
                    normal_vect: np.ndarray = - sysCoord_e[e,:,1]
                elif masterGroup.dim == 2: # surfaces                
                    normal_vect: np.ndarray = sysCoord_e[e,:,2]
                else:
                    raise "The master group must be dimension 1 or 2. Must be lines or surfaces."
                
                # distance to project the node to the element
                d: float = np.abs((newCoord[node] - closeCoordo) @ normal_vect)
                # vector to the interface
                u: np.ndarray = d * normal_vect
                listU.append(u)

            # Apply the displacement to meet the interface 
            oldU: np.ndarray = self.Results_displacement_matrix()[nodes]
            displacementMatrix = np.asarray(listU) + oldU

        else:
            nodes = np.array([])
            displacementMatrix = np.array([])

        tic.Tac("PostProcessing","Get displacement")
        
        return nodes, displacementMatrix
    
    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    def _Results_Check_Available(self, result: str) -> bool:
        """Check that the result is available"""
        availableResults = self.Results_Available()
        if result in availableResults:
            return True
        else:
            Display.MyPrintError(f"\nFor a {self.problemType} problem result must be in : \n {availableResults}")
            return False

    def Results_Set_Iteration_Summary(self) -> None:
        """Sets the iteration's summary."""
        pass

    def Results_Get_Iteration_Summary(self) -> str:
        """Returns the iteration's summary."""
        return "Unspecified."

    def Results_Set_Bc_Summary(self) -> None:
        """Sets the simulation loading summary."""
        pass

    def Results_Get_Bc_Summary(self) -> str:
        """Returns the simulation loading summary."""
        return "Unspecified."
        
    def Results_Reshape_values(self, values: np.ndarray, nodeValues: bool) -> np.ndarray:
        """Reshapes input values based on whether they are stored at nodes or elements.

        Parameters
        ----------
        values : np.ndarray
            Input values to reshape.
        nodeValues : bool
            If True, the output will represent values at nodes; if False, values on elements will be derived.

        Returns
        -------
        np.ndarray
            Reshaped values on nodes or elements.

        Raises
        ------
        Exception
            Raised if unexpected conditions occur during the calculation.
        """
        mesh = self.mesh
        Nn = mesh.Nn
        Ne = mesh.Ne

        is1d = values.ndim == 1
        
        if nodeValues:
            shape = -1 if is1d else (Nn, -1)
            if values.size % Nn == 0:
                # values stored at nodes
                if is1d:
                    return values.ravel()
                else:
                    return values.reshape(Nn,-1)
            elif values.size % Ne == 0:
                # values stored at elements
                values_e = values.reshape(Ne, -1)
                # get node values from element values
                values_n = self.mesh.Get_Node_Values(values_e)
                return values_n.reshape(shape)
        else:
            shape = -1 if is1d else (Ne, -1)
            if values.size % Ne == 0:
                return values.reshape(shape)
            elif values.size % Nn == 0:
                # get values stored at nodes (Nn, i)
                values_n = values.reshape(Nn, -1)
                # get values on every elements and nPe (Ne, nPe, i)
                values_e_nPe = values_n[mesh.connect]
                # get values on elements by averaging over the nodes per elements (nPe)
                values_e: np.ndarray = np.mean(values_e_nPe, 1)
                return values_e.reshape(shape)

        # We should never reach this line of code if no unexpected conditions occurs
        raise Exception("Unexpected conditions occurred during the calculation.")
    
    def Save(self, folder: str, filename: str="simulation", additionalInfos:str="") -> None:
        """Saves the simulation and its summary in the folder. Saves the simulation as 'filename.pickle'."""
        # Empty matrices in element groups
        self.mesh._ResetMatrix()

        folder_EasyFEA = Folder.Dir(Folder.Dir()) # path the EasyFEA folder
        # this path will be removed in print

        # Save simulation
        path_simu = Folder.Join(folder, f"{filename}.pickle", mkdir=True)
        with open(path_simu, "wb") as file:
            pickle.dump(self, file)
        Display.MyPrint(f'Saved:\n{path_simu.replace(folder_EasyFEA,"")}\n', 'green')
        
        # Save simulation summary
        path_summary = Folder.Join(folder, "summary.txt", mkdir=True)
        summary = f"Simulation completed on: {datetime.now()}\n"
        summary += f"version: {__version__}"
        summary += str(self)
        if str(additionalInfos) != "":
            summary += Display.Section("Additional information", False)
            summary += '\n' + str(additionalInfos)

        with open(path_summary, 'w', encoding='utf8') as file:
            file.write(summary)
        Display.MyPrint(f'Saved:\n{path_summary.replace(folder_EasyFEA,"")}\n', 'green')

# ----------------------------------------------
# _Simu Functions
# ----------------------------------------------

def Load_Simu(folder: str, filename: str="simulation") -> _Simu:
    """Loads the simulation from the specified folder.

    Parameters
    ----------
    folder : str
        simulation's folder.
    filename : str, optional
        The simualtion's name, by default "simulation".

    Returns
    -------
    _Simu
        The loaded simulation.
    """

    folder_PythonEF = Folder.Dir(Folder.Dir())
    path_simu = Folder.Join(folder, f"{filename}.pickle")
    assert Folder.Exists(path_simu), f"The file {filename}.pickle cannot be found."

    try:
        with open(path_simu, 'rb') as file:
            simu: _Simu = pickle.load(file)
    except EOFError:
        Display.MyPrintError(f"The file:\n{path_simu}\nis empty or corrupted.")
        return None    
    
    Display.MyPrint(f'\nLoaded:\n{path_simu.replace(folder_PythonEF,"")}\n', 'green')

    return simu