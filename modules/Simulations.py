"""Module for creating simulations."""

from abc import ABC, abstractmethod
import os
import pickle
from datetime import datetime
from types import LambdaType
from typing import Union, cast
import numpy as np
import pandas as pd
from scipy import sparse

from Mesh import Mesh, MatrixType, ElemType
from BoundaryCondition import BoundaryCondition, LagrangeCondition
import Materials
from Materials import ModelType, IModel, _Displacement_Model, Beam_Structure, PhaseField_Model, Thermal_Model, Reshape_variable
from TicTac import Tic
from Interface_Solvers import ResolutionType, AlgoType, Solve, _Solve_Axb, Solvers
import Folder
from Display import myPrint, myPrintError

def Load_Simu(folder: str, verbosity=False):
    """
    Load the simulation from the specified folder.

    Parameters
    ----------
    folder : str
        The name of the folder where the simulation is saved.

    Returns
    -------
    Simu
        The loaded simulation.
    """

    path_simu = Folder.Join(folder, "simulation.pickle")
    error = "The file simulation.pickle cannot be found."
    assert Folder.Exists(path_simu), error

    with open(path_simu, 'rb') as file:
        simu = pickle.load(file)

    assert isinstance(simu, _Simu), 'Must be a simu object'

    if verbosity:
        myPrint(f'\nLoading:\n{path_simu}\n', 'green')
        print(simu.mesh)
        print(simu.model)
    return simu

class _Simu(ABC):
    """
    The following classes inherit from the parent class _Simu:
        - Simu_Displacement
        - Simu_Damage
        - Simu_Beam
        - Simu_Thermal

    To create new simulations, take inspiration from existing classes. You'll need to respect the interface with _Simu.
    The Simu_Thermal class is simple enough to understand the implementation.

    To use the interface/inheritance, 14 methods need to be defined.

    General:

        - def Get_problemTypes(self) -> list[ModelType]:

        - def Get_directions(self, problemType=None) -> list[str]:

        - def Get_dof_n(self, problemType=None) -> int:

    Solvers:

        - def Get_K_C_M_F(self, problemType=None) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:

        - def Get_x0(self, problemType=None):

        - def Assembly(self):

    Iterations:

        - def Save_Iter(self) -> None:

        - def Set_Iter(self, index=-1) -> None:

    Results:

        - def Results_Available(self) -> list[str]:

        - def Result(self, result: str, nodeValues=True, iter=None) -> float | np.ndarray:

        - def Results_Iter_Summary(self) -> tuple[list[int], list[tuple[str, np.ndarray]]]:

        - def Results_dict_Energy(self) -> dict[str, float]:

        - def Results_displacement_matrix(self) -> np.ndarray:

        - def Results_nodesField_elementsField(self, details=False) -> tuple[list[str], list[str]]:
    """

    # ================================================ ABSTRACT METHOD ================================================
    #
    @abstractmethod
    def Get_problemTypes(self) -> list[ModelType]:
        """Returns the problem types available through the simulation."""
        pass

    @abstractmethod
    def Get_directions(self, problemType=None) -> list[str]:
        """Returns a list of directions available in the simulation."""
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
        """Saves iteration results in results."""
        iter = {}

        iter["indexMesh"] = self.__indexMesh
        # mesh identifier at this iteration

        return iter

    @abstractmethod
    def Set_Iter(self, iter: int=-1) -> list[dict]:
        """Sets the simulation to the specified iteration (usually the last one) and returns the list of dictionary."""
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

    # ================================================ SIMU ================================================

    def _Check_Directions(self, problemType: ModelType, directions: list) -> None:
        """Checks whether the specified directions are available for the problem."""
        listDirections = self.Get_directions(problemType)
        for d in directions:
            assert d in listDirections, f"{d} is not in {listDirections}"

    def __Check_ProblemTypes(self, problemType: ModelType) -> None:
        """Checks whether this type of problem is available through the simulation."""
        assert problemType in self.Get_problemTypes(), f"This type of problem is not available in this simulation ({self.Get_problemTypes()})"

    def _Check_dim_mesh_material(self) -> None:
        """Checks that the material size matches the mesh size."""
        assert self.__model.dim == self.__mesh.dim, "The material must have the same dimensions as the mesh."

    def __str__(self) -> str:
        """
        Returns a string representation of the simulation.

        Returns
        -------
        str
            A string containing information about the simulation.
        """

        import Display

        text = Display.Section("Mesh", False)
        text += str(self.mesh)

        text += Display.Section("Model", False)
        text += '\n' + str(self.model)

        text += Display.Section("Loading", False)
        text += '\n' + self.Results_Get_Bc_Summary()

        text += Display.Section("Results", False)
        text += '\n' + self.Results_Get_Iteration_Summary()

        text += Display.Section("TicTac", False)
        text += Tic.Resume(False)

        return text

    def __init__(self, mesh: Mesh, model: IModel, verbosity=True, useNumba=True, useIterativeSolvers=True):
        """
        Creates a simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh the simulation will use.
        model : IModel
            The model used.
        verbosity : bool, optional
            If True, the simulation can write to the console. Defaults to True.
        useNumba : bool, optional
            If True and numba is installed numba can be used . Defaults to True.
        useIterativeSolvers : bool, optional
            If True, iterative solvers can be used. Defaults to True.
        """

        if verbosity:
            import Display
            Display.Section("Simulation")

        if len(mesh.orphanNodes) > 0:
            raise Exception("The simulation cannot be created because orphan nodes have been detected in the mesh.")

        self.__model: IModel = model

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
        """The simulation can write to the console"""

        self.__algo = AlgoType.elliptic
        """System resolution algorithm during simulation."""
        # Basic algo solves stationary problems

        # Solver used for solving
        self.__solver = "scipy"  # Initialized just in case
        solvers = Solvers()  # Available solvers
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

        self.Need_Update()

    @property
    def model(self) -> IModel:
        """Model used."""
        return self.__model

    @property
    def rho(self) -> Union[float, np.ndarray]:
        """Mass density."""
        return self.__rho

    @rho.setter
    def rho(self, value: Union[float, np.ndarray]):
        IModel._Test_Sup0(value)
        self.__rho = value
        """Mass density"""

    @property
    def mass(self) -> float:

        if self.dim == 1: return None

        matrixType = MatrixType.mass

        group = self.mesh.groupElem

        coordo_e_p = group.Get_GaussCoordinates_e_p(matrixType)

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

        coordo_e_p = group.Get_GaussCoordinates_e_p(matrixType)

        jacobian_e_p = group.Get_jacobian_e_pg(matrixType)
        weight_p = group.Get_weight_pg(matrixType)        

        rho_e_p = Reshape_variable(self.__rho, self.mesh.Ne, weight_p.size)
        mass = self.mass

        center: np.ndarray = np.einsum('ep,ep,p,epi->i', rho_e_p, jacobian_e_p, weight_p, coordo_e_p, optimize='optimal') / mass

        if self.dim == 2:
            center *= self.model.thickness

        if not isinstance(self.__rho, np.ndarray):
            diff = np.linalg.norm(center - self.mesh.center)/np.linalg.norm(center)
            assert diff <= 1e-12

        return center

    @property
    def useIterativeSolvers(self) -> bool:
        """Iterative solvers can be used."""
        return self.__useIterativeSolvers

    @property
    def solver(self) -> str:
        """Solver used to solve Ax=b."""
        return self.__solver

    @solver.setter
    def solver(self, value: str):

        # Retrieve usable solvers
        solvers = Solvers()

        if self.problemType != "damage":
            solvers.remove("BoundConstrain")

        if value in solvers:
            self.__solver = value
        else:
            myPrintError(f"The solver {value} cannot be used. The solver must be in {solvers}")

    def Save(self, folder: str) -> None:
        """Saves the simulation and its summary in the folder."""
        # Empty matrices in element groups
        self.mesh._ResetMatrix()

        folder_PythonEF = Folder.Get_Path(Folder.Get_Path()) # path the PythonEF folder
        # this path will be removed in print

        # Save simulation
        path_simu = Folder.New_File("simulation.pickle", folder)
        with open(path_simu, "wb") as file:
            pickle.dump(self, file)
        myPrint(f'\n{path_simu.replace(folder_PythonEF,"")} (saved)', 'green')
        
        # Save simulation summary
        path_summary = Folder.New_File("summary.txt", folder)
        summary = f"Simulation completed on: {datetime.now()}"
        summary += str(self)        
        with open(path_summary, 'w', encoding='utf8') as file:
            file.write(summary)
        myPrint(f'{path_summary.replace(folder_PythonEF,"")} (saved)', 'green')

    # TODO Enable simulation creation from the variational formulation ?

    # Solutions
    @property
    def results(self) -> list[dict]:
        """Returns a copy of the dictionary list containing the results of each iteration."""
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
            taille = self.mesh.Nn * self.Get_dof_n(problemType)
            vectInit = np.zeros(taille, dtype=float)
            self.__dict_u_n[problemType] = vectInit
            self.__dict_v_n[problemType] = vectInit
            self.__dict_a_n[problemType] = vectInit

    def __Check_New_Sol_Values(self, problemType: ModelType, values: np.ndarray) -> None:
        """Checks that the solution has the right size."""
        self.__Check_ProblemTypes(problemType)
        size = self.mesh.Nn * self.Get_dof_n(problemType)
        assert values.shape[0] == size, f"Must be size {size}"

    def get_u_n(self, problemType: ModelType) -> np.ndarray:
        """Returns the solution associated with the given problem."""
        return self.__dict_u_n[problemType].copy()

    def set_u_n(self, problemType: ModelType, values: np.ndarray) -> None:
        """Sets the solution associated with the given problem."""
        self.__Check_New_Sol_Values(problemType, values)
        self.__dict_u_n[problemType] = values

    def get_v_n(self, problemType: ModelType) -> np.ndarray:
        """Returns the speed solution associated with the given problem."""
        return self.__dict_v_n[problemType].copy()

    def set_v_n(self, problemType: ModelType, values: np.ndarray) -> None:
        """Sets the speed solution associated with the given problem."""
        self.__Check_New_Sol_Values(problemType, values)
        self.__dict_v_n[problemType] = values

    def get_a_n(self, problemType: ModelType) -> np.ndarray:
        """Returns the acceleration solution associated with the given problem."""
        return self.__dict_a_n[problemType].copy()

    def set_a_n(self, problemType: ModelType, values: np.ndarray) -> None:
        """Sets the acceleration solution associated with the given problem."""
        self.__Check_New_Sol_Values(problemType, values)
        self.__dict_a_n[problemType] = values

    # This method is overloaded in Simu_PhaseField
    def Get_lb_ub(self, problemType: ModelType) -> tuple[np.ndarray, np.ndarray]:
        """Returns the lower bound and upper bound."""
        return np.array([]), np.array([])

    # Properties
    @property
    def problemType(self) -> ModelType:
        """Get the simulation problem type.

        Returns
        -------
        ModelType
            The type of simulation problem.
        """
        return self.__model.modelType

    @property
    def algo(self) -> AlgoType:
        """The algorithm used to solve the problem.\n
        (elliptic, parabolic, hyperbolic) see:\n
        - self.Solver_Set_Elliptic_Algorithm()
        - self.Solver_Set_Parabolic_Algorithm()
        - self.Solver_Set_Newton_Raphson_Algorithm()
        """        
        return self.__algo

    @property
    def mesh(self) -> Mesh:
        """The simulation's mesh.
        """
        return self.__mesh

    @mesh.setter
    def mesh(self, mesh: Mesh):        
        if isinstance(mesh, Mesh):
            # For all old meshes, delete the matrices
            listMesh = cast(list[Mesh], self.__listMesh)
            [m._ResetMatrix() for m in listMesh]

            self.__indexMesh += 1
            self.__listMesh.append(mesh)
            self.__mesh = mesh

            # The mesh changes, so the matrices must be reconstructed
            self.Need_Update()
            # Initialize boundary conditions
            self.Bc_Init()

    @property
    def dim(self) -> int:
        """The simulation dimension.
        """
        return self.__dim

    @property
    def useNumba(self) -> bool:
        """The simulation can use numba functions.
        """
        return self.__useNumba

    @useNumba.setter
    def useNumba(self, value: bool):
        value = value
        self.__model.useNumba = value
        self.__useNumba = value

    def __Update_mesh(self, iter: int) -> None:
        """Updates the mesh with the specified iteration.

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
        """The simulation needs to reconstruct matrices K, C, and M.
        """
        return self.__matricesUpdated

    def Need_Update(self, value=True) -> None:
        """Set whether the simulation needs to reconstruct matrices K, C, and M.
        """
        self.__matricesUpdated = value

    # ================================================ Solver ================================================

    def Solver_Set_Elliptic_Algorithm(self) -> None:
        """Set the algorithm's resolution properties for an elliptic problem.

        Used to solve K u = F.
        """
        self.__algo = AlgoType.elliptic

    def Solver_Set_Parabolic_Algorithm(self, dt: float, alpha=1/2) -> None:
        """Set the algorithm's resolution properties for a parabolic problem.

        Used to solve K u + C v = F.

        Parameters
        ----------
        dt : float
            The time increment.
        alpha : float, optional
            The alpha criterion [0 -> Forward Euler, 1 -> Backward Euler, 1/2 -> midpoint], by default 1/2.
        """
        self.__algo = AlgoType.parabolic

        assert dt > 0, "Time increment must be > 0"

        self.alpha = alpha
        self.dt = dt

    def Solver_Set_Newton_Raphson_Algorithm(self, dt: float, betha=1/4, gamma=1/2) -> None:
        """Set the algorithm's resolution properties for a Newton-Raphson problem.

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
        """Solve the simulation for the current boundary conditions.

        Returns
        -------
        np.ndarray
            The solution of the simulation.
        """
        if self.needUpdate:
            self.Assembly()

        self._Solver_Solve(self.problemType)

        return self.get_u_n(self.problemType)

    def _Solver_Solve(self, problemType: ModelType) -> None:
        """Solve the problem.
        """
        # Here you need to specify the type of problem because a simulation can have several physical models

        algo = self.__algo

        # Old solution
        u_n = self.get_u_n(problemType)
        v_n = self.get_v_n(problemType)
        a_n = self.get_a_n(problemType)

        if len(self.Bc_Lagrange) > 0:
            # Lagrange conditions are applied.
            resolution = ResolutionType.r2
            x, lagrange = Solve(self, problemType, resolution)
        else:
            resolution = ResolutionType.r1
            x = Solve(self, problemType, resolution)            

        if algo == AlgoType.elliptic:
            u_np1 = x
            self.set_u_n(problemType, u_np1)

        if algo == AlgoType.parabolic:
            # See Hughes 1987 Chapter 7

            u_np1 = x

            alpha = self.alpha
            dt = self.dt

            v_Tild_np1 = u_n + ((1 - alpha) * dt * v_n)
            v_np1 = (u_np1 - v_Tild_np1) / (alpha * dt)

            # New solutions
            self.set_u_n(problemType, u_np1)
            self.set_v_n(problemType, v_np1)

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
            self.set_u_n(problemType, u_np1)
            self.set_v_n(problemType, v_np1)
            self.set_a_n(problemType, a_np1)

    def _Solver_Apply_Neumann(self, problemType: ModelType) -> sparse.csr_matrix:
        """Fill in the Neumann boundary conditions by constructing b of A x = b.

        Parameters
        ----------
        problemType : ModelType
            The type of problem.

        Returns
        -------
        sparse.csr_matrix
            The b vector as a csr_matrix.
        """
        tic = Tic()

        algo = self.algo
        dofs = BoundaryCondition.Get_dofs(problemType, self.__Bc_Neumann)
        dofsValues = BoundaryCondition.Get_values(problemType, self.__Bc_Neumann)
        nDof = self.mesh.Nn * self.Get_dof_n(problemType)

        # Additional dimension associated with the lagrangian multipliers
        nDof += self._Bc_Lagrange_dim(problemType)

        b = sparse.csr_matrix((dofsValues, (dofs, np.zeros(len(dofs)))), shape=(nDof, 1))

        K, C, M, F = self.Get_K_C_M_F(problemType)

        u_n = self.get_u_n(problemType)
        v_n = self.get_v_n(problemType)
        a_n = self.get_a_n(problemType)

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
                # Initializes accel
                dofsKnown, dofsUnknown = self.Bc_dofs_known_unknow(problemType)

                bb = b - K.dot(sparse.csr_matrix(u_n.reshape(-1, 1)))

                bb -= C.dot(sparse.csr_matrix(v_n.reshape(-1, 1)))

                bbi = bb[dofsUnknown]
                Aii = M[dofsUnknown, :].tocsc()[:, dofsUnknown].tocsr()

                x0 = a_n[dofsUnknown]

                ai_n = _Solve_Axb(self, problemType, Aii, bbi, x0, [], [])

                a_n[dofsUnknown] = ai_n

                self.set_a_n(problemType, a_n)

            a_n = self.get_a_n(problemType)

            dt = self.dt
            gamma = self.gamma
            betha = self.betha

            uTild_np1 = u_n + (dt * v_n) + dt**2/2 * (1 - 2 * betha) * a_n
            vTild_np1 = v_n + (1 - gamma) * dt * a_n

            b -= K.dot(uTild_np1.reshape(-1, 1))
            b -= C.dot(vTild_np1.reshape(-1, 1))
            b = sparse.csr_matrix(b)

        tic.Tac("Solver", f"Neumann ({problemType}, {algo})", self._verbosity)

        return b

    def _Solver_Apply_Dirichlet(self, problemType: ModelType, b: sparse.csr_matrix, resolution: ResolutionType) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Apply Dirichlet conditions by constructing A and x from A x = b.

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

            solDotDot_n = self.get_a_n(problemType)
            dofsValues = solDotDot_n[dofs]

        A, x = self.__Solver_Get_Dirichlet_A_x(problemType, resolution, A, b, dofsValues)

        tic.Tac("Solver", f"Dirichlet ({problemType}, {algo})", self._verbosity)

        return A, x

    def __Solver_Get_Dirichlet_A_x(self, problemType: ModelType, resolution: ResolutionType, A: sparse.csr_matrix, b: sparse.csr_matrix, dofsValues: np.ndarray):
        """Resize the matrix system according to known degrees of freedom and resolution type.

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

        if resolution in [ResolutionType.r1, ResolutionType.r2]:

            # Here we return the solution with the known ddls
            x = sparse.csr_matrix((dofsValues, (dofs, np.zeros(len(dofs)))), shape=(size, 1), dtype=np.float64)

            return A, x

        elif resolution == ResolutionType.r3:
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

    
    # ------------------------------------------- BOUNDARY CONDITIONS -------------------------------------------
    # Functions for setting simulation boundary conditions
    
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
        """Add Lagrange conditions."""
        assert isinstance(newBc, LagrangeCondition)
        self.__Bc_Lagrange.append(newBc)
        # triger the update cause when we use lagrange multiplier we need to update the matrix system
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
        try:
            return self.__Bc_Display.copy()
        except AttributeError:
            # the simu object is old and dont have self.__Bc_Display variable
            self.__Bc_Display = []
            return []

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
        """Retrieves known and unknown dofs."""
        tic = Tic()

        # Builds known dofs
        dofsKnown = []

        dofsKnown = self.Bc_dofs_Dirichlet(problemType)
        unique_dofs_unknown = np.unique(dofsKnown)

        # Builds unknown dofs
        nDof = self.mesh.Nn * self.Get_dof_n(problemType)

        dofsUnknown = list(range(nDof))

        dofsUnknown = list(set(dofsUnknown) - set(unique_dofs_unknown))        

        dofsKnown = np.asarray(dofsKnown, dtype=int)
        dofsUnknown = np.array(dofsUnknown, dtype=int)
        
        test = unique_dofs_unknown.shape[0] + dofsUnknown.shape[0]
        assert test == nDof, f"Problem under conditions dofsKnown + dofsUnknown - nDof = {test-nDof}"

        tic.Tac("Solver",f"Get dofs ({problemType})", self._verbosity)

        return dofsKnown, dofsUnknown

    def Bc_dofs_nodes(self, nodes: np.ndarray, directions: list[str], problemType=None) -> np.ndarray:
        """Get degrees of freedom associated with nodes based on the problem and directions.

        Parameters
        ----------
        nodes : np.ndarray
            Nodes.
        directions : list
            Directions.        
        problemType : str
            Problem type.

        Returns
        -------
        np.ndarray
            Degrees of freedom.
        """

        if problemType is None:
            problemType = self.problemType

        self.__Check_ProblemTypes(problemType)
        
        assert len(nodes) > 0, "Empty node list"
        nodes = np.asarray(nodes)

        if isinstance(directions, str):
            directions = [directions]

        dof_n = self.Get_dof_n(problemType)

        return BoundaryCondition.Get_dofs_nodes(dof_n, problemType, nodes, directions)

    def __Bc_evaluate(self, coordo: np.ndarray, values, option="nodes") -> np.ndarray:
        """Evaluates values at nodes or gauss points."""
        
        assert option in ["nodes","gauss"], f"Must be in ['nodes','gauss']"
        if option == "nodes":
            values_eval = np.zeros(coordo.shape[0])
        elif option == "gauss":
            values_eval = np.zeros((coordo.shape[0],coordo.shape[1]))
        
        if isinstance(values, LambdaType):
            # Evaluates function at coordinates   
            try:
                if option == "nodes":
                    values_eval[:] = values(coordo[:,0], coordo[:,1], coordo[:,2])
                elif option == "gauss":
                    values_eval[:,:] = values(coordo[:,:,0], coordo[:,:,1], coordo[:,:,2])
            except:
                raise Exception("Must provide a lambda function of the form\n lambda x,y,z, : f(x,y,z)")
        else:            
            if option == "nodes":
                values_eval[:] = values
            elif option == "gauss":
                values_eval[:,:] = values

        return values_eval
    
    def add_dirichlet(self, nodes: np.ndarray, values: list, directions: list[str], problemType=None, description="") -> None:
        """Add dirichlet conditions.

        Parameters
        ----------
        nodes : np.ndarray
            nodes
        values : list
            list of values that can contain floats, arrays or functions or lambda functions
            ex = [10, lambda x,y,z : 10*x - 20*y + x*z, np.ndarray] \n
            The functions use the x, y and z nodes coordinates. \n
            Please note that the functions must take 3 input parameters in the order x, y, z, whether the problem is 1D, 2D or 3D.
        directions : list[str]
            directions where values will be applied
            ex = ['y', 'x']
        problemType : ModelType, optional
            problem type, if not specified, we take the basic problem of the problem
        description : str, optional
            Description of the condition, by default "".
        """

        if len(values) == 0 or len(values) != len(directions): return        

        if problemType is None:
            problemType = self.problemType

        self.__Check_ProblemTypes(problemType)
        
        assert len(nodes) > 0, "Empty node list"
        nodes = np.asarray(nodes)

        Nn = nodes.shape[0]
        coordo = self.mesh.coordo
        coordo_n = coordo[nodes]

        # initializes the value vector for each nodes
        dofsValues_dir = np.zeros((Nn, len(directions)))        

        for d, dir in enumerate(directions):
            eval_n = self.__Bc_evaluate(coordo_n, values[d], option="nodes")
            dofsValues_dir[:,d] = eval_n.reshape(-1)
        
        dofsValues = dofsValues_dir.reshape(-1)
        
        dofs = self.Bc_dofs_nodes(nodes, directions, problemType)

        self.__Bc_Add_Dirichlet(problemType, nodes, dofsValues, dofs, directions, description)

    def add_neumann(self, nodes: np.ndarray, values: list, directions: list[str], problemType=None, description="") -> None:
        """Point force

        Parameters
        ----------
        nodes : np.ndarray
            nodes
        values : list
            list of values that can contain floats, arrays or functions or lambda functions
            ex = [10, lambda x,y,z : 10*x - 20*y + x*z, np.ndarray] \n
            The functions use the x, y and z nodes coordinates.\n
            Please note that the functions must take 3 input parameters in the order x, y, z, whether the problem is 1D, 2D or 3D.
        directions : list[str]
            directions where values will be applied
            ex = ['y', 'x']
        problemType : ModelType, optional
            problem type, if not specified, we take the basic problem of the problem
        description : str, optional
            Description of the condition, by default "".
        """
        
        if len(values) == 0 or len(values) != len(directions): return

        if problemType is None:
            problemType = self.problemType

        self.__Check_ProblemTypes(problemType)

        dofsValues, dofs = self.__Bc_pointLoad(problemType, nodes, values, directions)

        self.__Bc_Add_Neumann(problemType, nodes, dofsValues, dofs, directions, description)
        
    def add_lineLoad(self, nodes: np.ndarray, values: list, directions: list[str], problemType=None, description="") -> None:
        """Apply a linear force.

        Parameters
        ----------
        nodes : np.ndarray
            nodes
        values : list
            list of values that can contain floats, arrays or functions or lambda functions
            ex = [10, lambda x,y,z : 10*x - 20*y + x*z, np.ndarray] \n
            functions use x, y and z integration points coordinates (x,y,z are in this case arrays of dim (e,p)) \n
            Please note that the functions must take 3 input parameters in the order x, y, z, whether the problem is 1D, 2D or 3D.
        directions : list[str]
            directions where values will be applied
            ex = ['y', 'x']
        problemType : ModelType, optional
            problem type, if not specified, we take the basic problem of the problem
        description : str, optional
            Description of the condition, by default "".
        """

        if len(values) == 0 or len(values) != len(directions): return

        if problemType is None:
            problemType = self.problemType

        self.__Check_ProblemTypes(problemType)

        dofsValues, dofs = self.__Bc_lineLoad(problemType, nodes, values, directions)

        self.__Bc_Add_Neumann(problemType, nodes, dofsValues, dofs, directions, description)

    def add_surfLoad(self, nodes: np.ndarray, values: list, directions: list[str], problemType=None, description="") -> None:
        """Apply a surface force
        
        Parameters
        ----------
        nodes : np.ndarray
            nodes
        values : list
            list of values that can contain floats, arrays or functions or lambda functions
            ex = [10, lambda x,y,z : 10*x - 20*y + x*z, np.ndarray] \n
            functions use x, y and z integration points coordinates (x,y,z are in this case arrays of dim (e,p)) \n
            Please note that the functions must take 3 input parameters in the order x, y, z, whether the problem is 1D, 2D or 3D.
        directions : list[str]
            directions where values will be applied
            ex = ['y', 'x']
        problemType : ModelType, optional
            problem type, if not specified, we take the basic problem of the problem
        description : str, optional
            Description of the condition, by default "".
        """

        if len(values) == 0 or len(values) != len(directions): return

        if problemType is None:
            problemType = self.problemType

        self.__Check_ProblemTypes(problemType)
            
        if self.__dim == 2:
            dofsValues, dofs = self.__Bc_lineLoad(problemType, nodes, values, directions)
            # multiplied by thickness
            dofsValues *= self.model.thickness
        elif self.__dim == 3:
            dofsValues, dofs = self.__Bc_surfload(problemType, nodes, values, directions)

        self.__Bc_Add_Neumann(problemType, nodes, dofsValues, dofs, directions, description)

    def add_volumeLoad(self, nodes: np.ndarray, values: list, directions: list[str], problemType=None, description="") -> None:
        """Apply a volumetric force.
        
        Parameters
        ----------
        nodes : np.ndarray
            nodes
        values : list
            list of values that can contain floats, arrays or functions or lambda functions
            ex = [10, lambda x,y,z : 10*x - 20*y + x*z, np.ndarray] \n
            functions use x, y and z integration points coordinates (x,y,z are in this case arrays of dim (e,p)) \n
            Please note that the functions must take 3 input parameters in the order x, y, z, whether the problem is 1D, 2D or 3D.
        directions : list[str]
            directions where values will be applied
            ex = ['y', 'x']
        problemType : ModelType, optional
            problem type, if not specified, we take the basic problem of the problem
        description : str, optional
            Description of the condition, by default "".
        """
        
        if len(values) == 0 or len(values) != len(directions): return

        if problemType is None:
            problemType = self.problemType

        self.__Check_ProblemTypes(problemType)
        
        if self.__dim == 2:
            dofsValues, dofs = self.__Bc_surfload(problemType, nodes, values, directions)
            # multiplied by thickness
            dofsValues = dofsValues*self.model.thickness
        elif self.__dim == 3:
            dofsValues, dofs = self.__Bc_volumeload(problemType, nodes, values, directions)

        self.__Bc_Add_Neumann(problemType, nodes, dofsValues, dofs, directions, description)
    
    def __Bc_pointLoad(self, problemType: ModelType, nodes: np.ndarray, values: list, directions: list) -> tuple[np.ndarray , np.ndarray]:
        """Apply a point load."""

        Nn = nodes.shape[0]
        coordo = self.mesh.coordoGlob
        coordo_n = coordo[nodes]

        # initializes the value vector for each node
        valeurs_ddl_dir = np.zeros((Nn, len(directions)))

        for d, dir in enumerate(directions):
            eval_n = self.__Bc_evaluate(coordo_n, values[d], option="nodes")
            if problemType == ModelType.beam:
                eval_n /= len(nodes)
            valeurs_ddl_dir[:,d] = eval_n.reshape(-1)
        
        dofsValues = valeurs_ddl_dir.reshape(-1)

        dofs = self.Bc_dofs_nodes(nodes, directions, problemType)

        return dofsValues, dofs

    def __Bc_Integration_Dim(self, dim: int, problemType: ModelType, nodes: np.ndarray, values: list, directions: list) -> tuple[np.ndarray , np.ndarray]:
        """Integrating on elements for the specified dimension."""

        dofsValues = np.array([])
        dofs = np.array([], dtype=int)

        listGroupElemDim = self.mesh.Get_list_groupElem(dim)

        dof_n = self.Get_dof_n(problemType)

        # For each group element
        for groupElem in listGroupElemDim:

            # Retrieves elements that exclusively use nodes
            elements = groupElem.Get_Elements_Nodes(nodes, exclusively=True)
            if elements.shape[0] == 0: continue
            connect = groupElem.connect[elements]
            Ne = elements.shape[0]
            
            # retrieves the coordinates of the Gauss points if you need to devaluate the function
            matrixType = MatrixType.mass
            coordo_e_p = groupElem.Get_GaussCoordinates_e_p(matrixType, elements)
            nPg = coordo_e_p.shape[1]

            N_pg = groupElem.Get_N_pg(matrixType)

            # integration objects
            jacobian_e_pg = groupElem.Get_jacobian_e_pg(matrixType)[elements]
            gauss = groupElem.Get_gauss(matrixType)
            weight_pg = gauss.weights

            # initializes the matrix of values for each node used by the elements and each gauss point (Ne*nPe, dir)
            values_dofs_dir = np.zeros((Ne*groupElem.nPe, len(directions)))
            # initializes the dofs vector
            new_dofs = np.zeros_like(values_dofs_dir, dtype=int)

            # Integrated in every direction
            for d, dir in enumerate(directions):
                # evaluates values
                eval_e_p = self.__Bc_evaluate(coordo_e_p, values[d], option="gauss")
                # integrates the elements
                values_e_p = np.einsum('ep,p,ep,pij->epij', jacobian_e_pg, weight_pg, eval_e_p, N_pg, optimize='optimal')
                # sum over integration points
                values_e = np.sum(values_e_p, axis=1)
                # sets calculated values and dofs
                values_dofs_dir[:,d] = values_e.reshape(-1)
                new_dofs[:,d] = self.Bc_dofs_nodes(connect.reshape(-1), [dir], problemType)

            new_values_dofs = values_dofs_dir.reshape(-1) # Put in vector form
            dofsValues = np.append(dofsValues, new_values_dofs)
            
            new_dofs = new_dofs.reshape(-1) # Put in vector form
            dofs = np.append(dofs, new_dofs)

        return dofsValues, dofs

    def __Bc_lineLoad(self, problemType: ModelType, nodes: np.ndarray, values: list, directions: list) -> tuple[np.ndarray , np.ndarray]:
        """Apply a linear force."""
        
        self._Check_Directions(problemType, directions)

        dofsValues, dofs = self.__Bc_Integration_Dim(dim=1, problemType=problemType, nodes=nodes, values=values, directions=directions)

        return dofsValues, dofs
    
    def __Bc_surfload(self, problemType: ModelType, nodes: np.ndarray, values: list, directions: list) -> tuple[np.ndarray , np.ndarray]:
        """Apply a surface force."""
        
        self._Check_Directions(problemType, directions)

        dofsValues, dofs = self.__Bc_Integration_Dim(dim=2, problemType=problemType, nodes=nodes, values=values, directions=directions)

        return dofsValues, dofs

    def __Bc_volumeload(self, problemType: ModelType, nodes: np.ndarray, values: list, directions: list) -> tuple[np.ndarray , np.ndarray]:
        """Apply a volumetric force."""
        
        self._Check_Directions(problemType, directions)

        dofsValues, dofs = self.__Bc_Integration_Dim(dim=3, problemType=problemType, nodes=nodes, values=values, directions=directions)

        return dofsValues, dofs
    
    def __Bc_Add_Neumann(self, problemType: ModelType, nodes: np.ndarray, dofsValues: np.ndarray, dofs: np.ndarray, directions: list, description="") -> None:
        """Add Neumann conditions.\n
        If a neumann condition is already applied to the dof, the condition will not be taken into account for the dof."""

        tic = Tic()

        self._Check_Directions(problemType, directions)

        new_Bc = BoundaryCondition(problemType, nodes, dofs, directions, dofsValues, f'Neumann {description}')
        self.__Bc_Neumann.append(new_Bc)

        tic.Tac("Boundary Conditions","Add Neumann condition ", self._verbosity)   
     
    def __Bc_Add_Dirichlet(self, problemType: ModelType, nodes: np.ndarray, dofsValues: np.ndarray, dofs: np.ndarray, directions: list, description="") -> None:
        """Add Dirichlet conditions.\n
        If a Dirichlet's dof is entered more than once, the conditions are added together."""

        tic = Tic()

        self.__Check_ProblemTypes(problemType)

        new_Bc = BoundaryCondition(problemType, nodes, dofs, directions, dofsValues, f'Dirichlet {description}')

        self.__Bc_Dirichlet.append(new_Bc)

        tic.Tac("Boundary Conditions","Add Dirichlet condition", self._verbosity)
    
    # Functions to create links between degrees of freedom

    def _Bc_Add_Display(self, nodes: np.ndarray, directions: list[str], description: str, problemType=None) -> None:
        """Add condition for display"""

        if problemType is None:
            problemType = self.problemType

        self.__Check_ProblemTypes(problemType)        

        dofs = self.Bc_dofs_nodes(nodes, directions, problemType)
        
        dofsValues =  np.array([0]*len(dofs))

        new_Bc = BoundaryCondition(problemType, nodes, dofs, directions, dofsValues, description)
        self.__Bc_Display.append(new_Bc)

    def Get_contact(self, masterMesh: Mesh, slaveNodes: np.ndarray=None, masterNodes: np.ndarray=None) -> tuple[np.ndarray, np.ndarray]:
        """Retrieves the simulation nodes detected in the master mesh with the associated displacement matrix to the interface.

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
        newCoordo = self.Results_displacement_matrix() + self.mesh.coordo
        
        # check nodes in master mesh
        idx = masterMesh.groupElem.Get_Mapping(newCoordo[slaveNodes], elements)[0]        
        idx = np.asarray(list(set().union(idx)))

        tic.Tac("PostProcessing","Get slave nodes in master mesh")

        if idx.size > 0:
            # slave nodes have been detected in the master mesh
            nodes: np.ndarray = slaveNodes[idx]

            sysCoord_e = masterGroup.sysCoord_e

            # get the elemGroup on the interface        
            gaussCoordo_e_p = masterGroup.Get_GaussCoordinates_e_p(MatrixType.rigi)
            
            # empty new displacement
            listU: list[np.ndarray] = []
            # for each nodes in master mesh we will detects the shortest displacement vector to the interface
            for node in nodes:
                # vectors between the interface coordinates and the detected node
                vi_e_pg: np.ndarray  = gaussCoordo_e_p - newCoordo[node]               

                # distance between the interface coordinates and the detected node
                d_e_pg: np.ndarray = np.linalg.norm(vi_e_pg, axis=2)
                e, p = np.unravel_index(np.argmin(d_e_pg), d_e_pg.shape)
                # retrieves the nearest coordinate
                closeCoordo = np.reshape(gaussCoordo_e_p[e,p], -1)
                
                # normal vector
                if masterGroup.dim == 1: # lines
                    normal_vect: np.ndarray = - sysCoord_e[e,:,1]
                elif masterGroup.dim == 2: # surfaces                
                    normal_vect: np.ndarray = sysCoord_e[e,:,2]
                else:
                    raise "The master group must be dimension 1 or 2. Must be lines or surfaces."
                
                # distance to project the node to the element
                d: float = np.abs((newCoordo[node] - closeCoordo) @ normal_vect)
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
    
    # ------------------------------------------- Results ------------------------------------------- 

    # TODO jusque ici

    def _Results_Check_Available(self, result: str) -> bool:
        """Check that the result is available"""
        availableResults = self.Results_Available()
        if result in availableResults:
            return True
        else:
            myPrintError(f"\nFor a {self.problemType} problem result must be in : \n {availableResults}")
            return False

    def Results_Set_Iteration_Summary(self) -> None:
        """Sets the iteration summary."""
        pass

    def Results_Get_Iteration_Summary(self) -> str:
        """Iteration's summary."""
        return "Unknown"

    def Results_Set_Bc_Summary(self) -> None:
        """Simulation loading information"""
        pass

    def Results_Get_Bc_Summary(self) -> str:
        """Simulation loading summary"""
        return "Unknown load"
        
    @staticmethod
    def Results_Exract_Node_Values(mesh: Mesh, result_e: np.ndarray) -> np.ndarray:
        """Get node values from element values.\n
        The value of a node is calculated by averaging the values of the surrounding elements.

        Parameters
        ----------
        mesh : Mesh
            mesh
        result_e : np.ndarray
            element values (Ne, i)

        Returns
        -------
        np.ndarray
            nodes values (Nn, i)
        """

        assert mesh.Ne == result_e.shape[0], "Must be of size (Ne,i)"

        tic = Tic()

        Ne = mesh.Ne
        Nn = mesh.Nn

        if len(result_e.shape) == 1:
            # In this case it is a 1d vector
            # we need to reshape as
            result_e = result_e.reshape(Ne,1)
            isDim1 = True
        else:
            isDim1 = False
        
        nCols = result_e.shape[1]

        result_n = np.zeros((Nn, nCols), dtype=float)

        # connectivity of the nodes
        connect_n_e = mesh.Get_connect_n_e()
        # get elements per ndoes
        elements_n = np.reshape(np.sum(connect_n_e, axis=1), (mesh.Nn, 1))

        for c in range(nCols):
            values_e = result_e[:, c].reshape(mesh.Ne,1)
            values_n = (connect_n_e @ values_e) * 1/elements_n
            result_n[:,c] = values_n.reshape(-1)

        tic.Tac("PostProcessing","Element to nodes values", False)

        if isDim1:
            return result_n.reshape(-1)
        else:
            return result_n
        
    def Results_Reshape_values(self, values: np.ndarray, nodeValues: bool) -> np.ndarray:
        """
        Reshapes input values based on whether they are stored at nodes or elements.

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
                    return values.reshape(-1)
                else:
                    return values.reshape(Nn,-1)
            elif values.size % Ne == 0:
                # values stored at elements
                values_e = values.reshape(Ne, -1)
                # get node values from element values
                values_n = self.Results_Exract_Node_Values(mesh, values_e)
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

###################################################################################################

class Simu_Displacement(_Simu):

    def __init__(self, mesh: Mesh, model: _Displacement_Model, verbosity=False, useNumba=True, useIterativeSolvers=True):
        """
        Creates a displacement simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh the simulation will use.
        model : IModel
            The model used.
        verbosity : bool, optional
            If True, the simulation can write to the console. Defaults to False.
        useNumba : bool, optional
            If True, numba can be used. Defaults to True.
        useIterativeSolvers : bool, optional
            If True, iterative solvers can be used. Defaults to True.
        """
        assert model.modelType == ModelType.displacement, "The material must be displacement model"
        super().__init__(mesh, model, verbosity, useNumba, useIterativeSolvers)

        # init
        self.Set_Rayleigh_Damping_Coefs()
        self.Solver_Set_Elliptic_Algorithm()    

    def Results_nodesField_elementsField(self, details=False) -> tuple[list[str], list[str]]:
        nodesField = ["displacement_matrix"]
        if details:            
            elementsField = ["Stress", "Strain"]
        else:            
            elementsField = ["Stress"]
        if self.algo == AlgoType.hyperbolic: nodesField.extend(["speed", "accel"])
        return nodesField, elementsField
    
    def Get_directions(self, problemType=None) -> list[str]:
        dict_dim_directions = {
            2 : ["x", "y"],
            3 : ["x", "y", "z"]
        }
        return dict_dim_directions[self.dim]
    
    def Get_problemTypes(self) -> list[ModelType]:
        return [ModelType.displacement]
        
    def Get_dof_n(self, problemType=None) -> int:
        return self.dim

    @property
    def material(self) -> _Displacement_Model:
        """Elastic behavior."""
        return self.model

    @property
    def displacement(self) -> np.ndarray:
        """Displacement vector field.\n
        2D [uxi, uyi, ...]\n
        3D [uxi, uyi, uzi, ...]"""
        return self.get_u_n(self.problemType)

    @property
    def speed(self) -> np.ndarray:
        """Velocity vector field.\n
        2D [vxi, vyi, ...]\n
        3D [vxi, vyi, vzi, ...]"""
        return self.get_v_n(self.problemType)

    @property
    def accel(self) -> np.ndarray:
        """Acceleration vector field.\n
        2D [axi, ayi, ...]\n
        3D [axi, ayi, azi, ...]"""
        return self.get_a_n(self.problemType)

    def __Construct_Local_Matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Construct the elementary stiffness matrices for the displacement problem."""

        matrixType=MatrixType.rigi

        # Recovers matrices to work with
        mesh = self.mesh; Ne = mesh.Ne
        jacobian_e_pg = mesh.Get_jacobian_e_pg(matrixType)
        weight_pg = mesh.Get_weight_pg(matrixType)
        nPg = weight_pg.size

        N_vecteur_pg = mesh.Get_N_vector_pg(matrixType)
        rho = self.rho
        
        B_dep_e_pg = mesh.Get_B_e_pg(matrixType)
        leftDepPart = mesh.Get_leftDispPart(matrixType) # -> jacobian_e_pg * weight_pg * B_dep_e_pg'

        comportement = self.material

        tic = Tic()
        
        matC = comportement.C

        # Stifness
        matC = Reshape_variable(matC, Ne, nPg)
        Ku_e = np.sum(leftDepPart @ matC @ B_dep_e_pg, axis=1)
        
        # Mass
        rho_e_pg = Reshape_variable(rho, Ne, nPg)
        Mu_e = np.einsum(f'ep,p,pki,ep,pkj->eij', jacobian_e_pg, weight_pg, N_vecteur_pg, rho_e_pg, N_vecteur_pg, optimize="optimal")

        if self.dim == 2:
            thickness = self.material.thickness
            Ku_e *= thickness
            Mu_e *= thickness
        
        tic.Tac("Matrix","Construct Ku_e and Mu_e", self._verbosity)

        return Ku_e, Mu_e

    def Get_K_C_M_F(self, problemType=None) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        if problemType==None:
            problemType = self.problemType
        if self.needUpdate: self.Assembly()
        return self.__Ku.copy(), self.Get_Rayleigh_Damping(), self.__Mu.copy(), self.__Fu.copy()
 
    def Assembly(self) -> None:

        if self.needUpdate:

            # Data
            mesh = self.mesh        
            nDof = mesh.Nn*self.dim

            # Additional dimension linked to the use of lagrange coefficients
            nDof += self._Bc_Lagrange_dim(self.problemType)
                            
            Ku_e, Mu_e = self.__Construct_Local_Matrix()
            
            tic = Tic()

            linesVector_e = mesh.linesVector_e.reshape(-1)
            columnsVector_e = mesh.columnsVector_e.reshape(-1)

            # Assembly
            self.__Ku = sparse.csr_matrix((Ku_e.reshape(-1), (linesVector_e, columnsVector_e)), shape=(nDof, nDof))
            """Kglob matrix for the displacement problem (nDof, nDof)"""

            # Here I'm initializing Fu because I'd have to calculate the volumetric forces in __Construct_Local_Matrix.
            self.__Fu = sparse.csr_matrix((nDof, 1))
            """Fglob vector for the displacement problem (nDof, 1)"""

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.spy(self.__Ku)
            # plt.show()

            self.__Mu = sparse.csr_matrix((Mu_e.reshape(-1), (linesVector_e, columnsVector_e)), shape=(nDof, nDof))
            """Mglob matrix for the displacement problem (Nn*dim, Nn*dim)"""

            tic.Tac("Matrix","Assembly Ku, Mu and Fu", self._verbosity)

            self.Need_Update(False)

    def Set_Rayleigh_Damping_Coefs(self, coefM=0.0, coefK=0.0):
        """Set damping coefficients."""
        self.__coefM = coefM
        self.__coefK = coefK

    def Get_Rayleigh_Damping(self) -> sparse.csr_matrix:
        """Get damping matrix C."""
        if self.problemType == ModelType.displacement:
            try:
                return self.__coefM * self.__Mu + self.__coefK * self.__Ku
            except:
                # "Mu n'a pas été calculé"
                return None
        else:
            return None

    def Get_x0(self, problemType=None):
        algo = self.algo
        if self.displacement.size != self.mesh.Nn*self.dim:
            return np.zeros(self.mesh.Nn*self.dim)
        elif algo == AlgoType.elliptic:
            return self.displacement
        elif algo == AlgoType.hyperbolic:
            return self.accel
    
    def Save_Iter(self):
        
        iter = super().Save_Iter()

        iter['displacement'] = self.displacement
        if self.algo == AlgoType.hyperbolic:
            iter["speed"] = self.speed
            iter["accel"] = self.accel

        self._results.append(iter)
    
    def Set_Iter(self, iter= -1) -> list[dict]:
        
        results = super().Set_Iter(iter)

        if results is None: return

        displacementType = ModelType.displacement

        self.set_u_n(displacementType, results[displacementType])

        if self.algo == AlgoType.hyperbolic and "speed" in results and "accel" in results:
            self.set_v_n(displacementType, results["speed"])
            self.set_a_n(displacementType, results["accel"])
        else:
            initZeros = np.zeros_like(self.displacement)
            self.set_v_n(displacementType, initZeros)
            self.set_a_n(displacementType, initZeros)

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

        results.extend(["Svm", "Stress","Evm", "Strain"])
        
        results.extend(["Wdef","Wdef_e","ZZ1","ZZ1_e"])

        return results

    def Result(self, result: str, nodeValues=True, iter=None) -> Union[np.ndarray, float, None]:

        if iter != None:
            self.Set_Iter(iter)
        
        if not self._Results_Check_Available(result): return None

        # begin cases ----------------------------------------------------

        Nn = self.mesh.Nn

        values = None

        if result in ["Wdef"]:
            return self._Calc_Psi_Elas()

        elif result == "Wdef_e":
            values = self._Calc_Psi_Elas(returnScalar=False)
            
        elif result == "ZZ1":
            return self._Calc_ZZ1()[0]

        elif result == "ZZ1_e":
            error, values = self._Calc_ZZ1()

        elif result in ["ux", "uy", "uz"]:
            values_n = self.displacement.reshape(Nn, -1)
            values = values_n[:,self.__indexResult(result)]

        elif result == "displacement":
            values = self.displacement
        
        elif result == "displacement_norm":
            val_n = self.displacement.reshape(Nn, -1)
            values = np.linalg.norm(val_n, axis=1)

        elif result == "displacement_matrix":
            values = self.Results_displacement_matrix()

        elif result in ["vx", "vy", "vz"]:
            values_n = self.speed.reshape(Nn, -1)
            values = values_n[:,self.__indexResult(result)]

        elif result == "speed":
            values = self.speed
        
        elif result == "speed_norm":
            val_n = self.speed.reshape(Nn, -1)
            values = np.linalg.norm(val_n, axis=1)

        elif result in ["ax", "ay", "az"]:
            values_n = self.accel.reshape(Nn, -1)
            values = values_n[:,self.__indexResult(result)]
        
        elif result == "accel":
            values = self.accel
        
        elif result == "accel_norm":
            val_n = self.accel.reshape(Nn, -1)
            values = np.linalg.norm(val_n, axis=1)
        
        elif ("S" in result or "E" in result) and (not "_norm" in result):
            # Strain and Stress calculation part

            coef = self.material.coef

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
            
            values = Materials.Result_in_Strain_or_Stress_field(val_e, res, coef)

        if not isinstance(values, np.ndarray):
            myPrintError("This result option is not implemented yet.")
            return

        # end cases ----------------------------------------------------
        
        return self.Results_Reshape_values(values, nodeValues)

    def _Calc_Psi_Elas(self, returnScalar=True, smoothedStress=False, matrixType=MatrixType.rigi) -> float:
        """Calculation of the kinematically admissible deformation energy, damaged or not.
        Wdef = 1/2 int_Omega jacobian * weight * Sig : Eps dOmega thickness"""

        tic = Tic()
        
        sol_u  = self.displacement
        
        Epsilon_e_pg = self._Calc_Epsilon_e_pg(sol_u, matrixType)
        jacobian_e_pg = self.mesh.Get_jacobian_e_pg(matrixType)
        weight_pg = self.mesh.Get_weight_pg(matrixType)
        N_pg = self.mesh.Get_N_pg(matrixType)

        if self.dim == 2:
            ep = self.material.thickness
        else:
            ep = 1

        Sigma_e_pg = self._Calc_Sigma_e_pg(Epsilon_e_pg, matrixType)

        if smoothedStress:
            Sigma_n = self.Results_Exract_Node_Values(self.mesh, np.mean(Sigma_e_pg, 1))

            Sigma_n_e = self.mesh.Locates_sol_e(Sigma_n)
            Sigma_e_pg = np.einsum('eni,pjn->epi',Sigma_n_e, N_pg)

        if returnScalar:

            Wdef = 1/2 * np.einsum(',ep,p,epi,epi->', ep, jacobian_e_pg, weight_pg, Sigma_e_pg, Epsilon_e_pg, optimize='optimal')
            Wdef = float(Wdef)

        else:

            Wdef = 1/2 * np.einsum(',ep,p,epi,epi->e', ep, jacobian_e_pg, weight_pg, Sigma_e_pg, Epsilon_e_pg, optimize='optimal')

        tic.Tac("PostProcessing","Calc Psi Elas",False)
        
        return Wdef
    
    def _Calc_ZZ1(self) -> tuple[float, np.ndarray]:
        """Calculation of ZZ1 error. For more details, see
        [F.Pled, Vers une stratégie robuste ... ingénierie mécanique] page 20/21
        Returns the global error and the error on each element.

        Returns
        -------
        error, error_e
        """

        Wdef_e = self._Calc_Psi_Elas(False)
        Wdef = np.sum(Wdef_e)

        WdefLisse_e = self._Calc_Psi_Elas(False, True)
        WdefLisse = np.sum(WdefLisse_e)

        error_e: np.nd = np.abs(WdefLisse_e-Wdef_e).reshape(-1)/Wdef

        error: float = np.abs(Wdef-WdefLisse)/Wdef

        return error, error_e

    def _Calc_Epsilon_e_pg(self, sol: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Builds epsilon for each element and each gauss point.\n
        2D : [Exx Eyy sqrt(2)*Exy]\n
        3D : [Exx Eyy Ezz sqrt(2)*Eyz sqrt(2)*Exz sqrt(2)*Exy]

        Parameters
        ----------
        sol : np.ndarray
            Displacement vector

        Returns
        -------
        np.ndarray
            Deformations stored at elements and gauss points (Ne,pg,(3 or 6))
        """

        tic = Tic()        
        u_e = sol[self.mesh.assembly_e]
        B_dep_e_pg = self.mesh.Get_B_e_pg(matrixType)
        Epsilon_e_pg: np.ndarray = np.einsum('epij,ej->epi', B_dep_e_pg, u_e, optimize='optimal')
        
        tic.Tac("Matrix", "Epsilon_e_pg", False)

        return Epsilon_e_pg
                    
    def _Calc_Sigma_e_pg(self, Epsilon_e_pg: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Calculating stresses from strains.\n
        2D : [Sxx Syy sqrt(2)*Sxy]\n
        3D : [Sxx Syy Szz sqrt(2)*Syz sqrt(2)*Sxz sqrt(2)*Sxy]

        Parameters
        ----------
        Epsilon_e_pg : np.ndarray
            Deformations stored at elements and gauss points (Ne,pg,(3 or 6))

        Returns
        -------
        np.ndarray
            Returns damaged or undamaged constraints (Ne,pg,(3 or 6))
        """
        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]

        assert Ne == self.mesh.Ne
        assert nPg == self.mesh.Get_nPg(matrixType)

        tic = Tic()

        c = self.material.C
        
        c_e_p = Reshape_variable(c, Ne, nPg)

        Sigma_e_pg = c_e_p @ Epsilon_e_pg[:,:,:,np.newaxis]
        Sigma_e_pg: np.ndarray = Sigma_e_pg.reshape((Ne,nPg,-1))
            
        tic.Tac("Matrix", "Sigma_e_pg", False)

        return Sigma_e_pg

    def __indexResult(self, result: str) -> int:

        dim = self.dim

        if len(result) <= 2:
            "Case were ui, vi or ai"
            if "x" in result:
                return 0
            elif "y" in result:
                return 1
            elif "z" in result:
                return 2

    def Results_dict_Energy(self) -> dict[str, float]:
        dict_Energie = {
            r"$\Psi_{elas}$": self._Calc_Psi_Elas()
            }
        return dict_Energie

    def Results_Get_Iteration_Summary(self) -> str:        

        summary = ""

        if not self._Results_Check_Available("Wdef"):
            return
        
        Wdef = self.Result("Wdef")
        summary += f"\nW def = {Wdef:.2f}"
        
        Svm = self.Result("Svm", nodeValues=False)
        summary += f"\n\nSvm max = {Svm.max():.2f}"

        Evm = self.Result("Evm", nodeValues=False)
        summary += f"\n\nEvm max = {Evm.max()*100:3.2f} %"

        # Affichage des déplacements
        dx = self.Result("ux", nodeValues=True)
        summary += f"\n\nUx max = {dx.max():.2e}"
        summary += f"\nUx min = {dx.min():.2e}"

        dy = self.Result("uy", nodeValues=True)
        summary += f"\n\nUy max = {dy.max():.2e}"
        summary += f"\nUy min = {dy.min():.2e}"

        if self.dim == 3:
            dz = self.Result("uz", nodeValues=True)
            summary += f"\n\nUz max = {dz.max():.2e}"
            summary += f"\nUz min = {dz.min():.2e}"

        return summary

    def Results_Iter_Summary(self) -> tuple[list[int], list[tuple[str, np.ndarray]]]:
        return super().Results_Iter_Summary()

    def Results_displacement_matrix(self) -> np.ndarray:

        Nn = self.mesh.Nn
        coordo = self.displacement.reshape((Nn,-1))
        dim = coordo.shape[1]

        if dim == 1:
            # Here we add two columns
            coordo = np.append(coordo, np.zeros((Nn,1)), axis=1)
            coordo = np.append(coordo, np.zeros((Nn,1)), axis=1)
        elif dim == 2:
            # Here we add 1 column
            coordo = np.append(coordo, np.zeros((Nn,1)), axis=1)

        return coordo

###################################################################################################

class Simu_PhaseField(_Simu):

    def __init__(self, mesh: Mesh, model: PhaseField_Model, verbosity=False, useNumba=True, useIterativeSolvers=True):
        """
        Creates a damage simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh the simulation will use.
        model : IModel
            The model used.
        verbosity : bool, optional
            If True, the simulation can write to the console. Defaults to False.
        useNumba : bool, optional
            If True, numba can be used. Defaults to True.
        useIterativeSolvers : bool, optional
            If True, iterative solvers can be used. Defaults to True.
        """

        assert model.modelType == ModelType.damage, "The material must be damage model"
        super().__init__(mesh, model, verbosity, useNumba, useIterativeSolvers)

        # init résultats
        self.__psiP_e_pg = []
        self.__old_psiP_e_pg = [] # old positive elastic energy density psiPlus(e, pg, 1) to use the miehe history field
        self.Solver_Set_Elliptic_Algorithm()

    def Results_nodesField_elementsField(self, details=False) -> tuple[list[str], list[str]]:
        if details:
            nodesField = ["displacement_matrix", "damage"]
            elementsField = ["Stress", "Strain", "psiP"]
        else:
            nodesField = ["displacement_matrix", "damage"]
            elementsField = ["Stress"]
        return nodesField, elementsField

    def Get_directions(self, problemType=None) -> list[str]:        
        if problemType == ModelType.damage:
            return [""]
        elif problemType in [ModelType.displacement, None]:
            _dict_dim_directions_displacement = {
                2 : ["x", "y"],
                3 : ["x", "y", "z"]
            }
            return _dict_dim_directions_displacement[self.dim]
    
    def Get_problemTypes(self) -> list[ModelType]:
        return [ModelType.damage, ModelType.displacement]

    def Get_lb_ub(self, problemType: ModelType) -> tuple[np.ndarray, np.ndarray]:
        
        if problemType == ModelType.damage:
            solveur = self.phaseFieldModel.solver
            if solveur == "BoundConstrain":
                lb = self.damage
                lb[np.where(lb>=1)] = 1-np.finfo(float).eps
                ub = np.ones(lb.shape)
            else:
                lb, ub = np.array([]), np.array([])
        else:
            lb, ub = np.array([]), np.array([])
            
        return lb, ub

    def Get_dof_n(self, problemType=None) -> int:        
        if problemType == ModelType.damage:
            return 1
        elif problemType in [ModelType.displacement, None]:
            return self.dim

    @property
    def phaseFieldModel(self) -> PhaseField_Model:
        """Damage model"""
        return self.model

    @property
    def displacement(self) -> np.ndarray:
        """Displacement vector field.\n
        2D [uxi, uyi, ...]\n
        3D [uxi, uyi, uzi, ...]"""
        return self.get_u_n(ModelType.displacement)

    @property
    def damage(self) -> np.ndarray:
        """Damage scalar field.\n
        [di, ...]"""
        return self.get_u_n(ModelType.damage)
    
    def Bc_dofs_nodes(self, nodes: np.ndarray, directions: list[str], problemType=ModelType.displacement) -> np.ndarray:
        return super().Bc_dofs_nodes(nodes, directions, problemType)

    def add_dirichlet(self, nodes: np.ndarray, values: np.ndarray, directions: list[str], problemType=ModelType.displacement, description=""):        
        return super().add_dirichlet(nodes, values, directions, problemType, description)
    
    def add_lineLoad(self, nodes: np.ndarray, values: list, directions: list[str], problemType=ModelType.displacement, description=""):
        return super().add_lineLoad(nodes, values, directions, problemType, description)

    def add_surfLoad(self, nodes: np.ndarray, values: list, directions: list[str], problemType=ModelType.displacement, description=""):
        return super().add_surfLoad(nodes, values, directions, problemType, description)
        
    def add_neumann(self, nodes: np.ndarray, values: list, directions: list[str], problemType=ModelType.displacement, description=""):
        return super().add_neumann(nodes, values, directions, problemType, description)

    def Get_K_C_M_F(self, problemType=None) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        
        if problemType==None:
            problemType = ModelType.displacement

        size = self.mesh.Nn * self.Get_dof_n(problemType)
        initcsr = sparse.csr_matrix((size, size))

        if self.needUpdate:
            if problemType == ModelType.displacement:
                self.__Assembly_displacement()
            else:
                self.__Assembly_damage()

        try:    
            if problemType == ModelType.damage:            
                return self.__Kd.copy(), initcsr, initcsr, self.__Fd.copy()
            elif problemType == ModelType.displacement:            
                return self.__Ku.copy(), initcsr, initcsr, self.__Fu.copy()
        except AttributeError:
            myPrintError("System not yet assembled")
            return initcsr, initcsr, initcsr, initcsr

    def Get_x0(self, problemType=None):
        
        if problemType == ModelType.damage:
            if self.damage.size != self.mesh.Nn:
                return np.zeros(self.mesh.Nn)
            else:
                return self.damage
        elif problemType in [ModelType.displacement, None]:
            if self.displacement.size != self.mesh.Nn*self.dim:
                return np.zeros(self.mesh.Nn*self.dim)
            else:
                return self.displacement

    def Assembly(self):
        self.__Assembly_damage()
        self.__Assembly_displacement()
    
    def Solve(self, tolConv=1.0, maxIter=500, convOption=2) -> tuple[np.ndarray, np.ndarray, sparse.csr_matrix, bool]:
        """Solving the damage problem with the staggered scheme.

        Parameters
        ----------
        tolConv : float, optional
            tolerance between old and new damage, by default 1.0
        maxIter : int, optional
            Maximum number of iterations to reach convergence, by default 500
        convOption : int, optional
            0 -> convergence on damage np.max(np.abs(d_kp1-dk)) equivalent normInf(d_kp1-dk)\n
            1 -> convergence on crack energy np.abs(psi_crack_kp1 - psi_crack_k)/psi_crack_kp1 \n
            2 -> convergence on total energy np.abs(psi_tot_kp1 - psi_tot_k)/psi_tot_kp1

        Returns
        -------
        np.ndarray, np.ndarray, int, float
            u_np1, d_np1, Kglob, convergence

            such that :\n
            u_np1 : displacement vector field
            d_np1 : damage scalar field
            Kglob : displacement stiffness matrix
            convergence: the solution has converged
        """

        assert tolConv > 0 and tolConv <= 1 , "tolConv must be between 0 and 1."
        assert maxIter > 1 , "Must be > 1."

        Niter = 0
        convergence = False
        dn = self.damage

        solver = self.phaseFieldModel.solver
        regu = self.phaseFieldModel.regularization

        tic = Tic()

        while not convergence and Niter <= maxIter:
                    
            Niter += 1
            if convOption == 0:                    
                d_n = self.damage
            elif convOption == 1:
                psi_n = self._Calc_Psi_Crack()
            elif convOption == 2:
                psi_n = self._Calc_Psi_Crack() + self._Calc_Psi_Elas()
            elif convOption == 3:
                d_n = self.damage
                u_n = self.displacement

            # Damage
            self.__Assembly_damage()
            d_np1 = self.__Solve_damage()

            # Displacement
            Kglob = self.__Assembly_displacement()
            u_np1 = self.__Solve_displacement()

            if convOption == 0:                
                convIter = np.max(np.abs(d_np1 - d_n))

            elif convOption in [1,2]:
                psi_np1 = self._Calc_Psi_Crack()
                if convOption == 2:
                   psi_np1 += self._Calc_Psi_Elas()

                if psi_np1 == 0:
                    convIter = np.abs(psi_np1 - psi_n)
                else:
                    convIter = np.abs(psi_np1 - psi_n)/psi_np1

            elif convOption == 3:
                # eq (25) Pech 2022 10.1016/j.engfracmech.2022.108591
                diffU = np.abs(u_np1 - u_n); diffU[u_np1 != 0] *= 1/np.abs(u_np1[u_np1 != 0])
                diffD = np.abs(d_np1 - d_n); diffD[d_np1 != 0] *= 1/np.abs(d_np1[d_np1 != 0])
                convU = np.sum(diffU)
                convD = np.sum(diffD)
                convIter = np.max([convD, convU])

            # Convergence condition
            if tolConv == 1:
                convergence = True
            elif convOption == 3:
                convergence = (convD <= tolConv) and (convU <= tolConv*0.999)
            else:
                convergence = convIter <= tolConv
                
        solverTypes = PhaseField_Model.SolverType

        if solver in [solverTypes.History, solverTypes.BoundConstrain]:
            d_np1 = d_np1            
        elif solver == solverTypes.HistoryDamage:
            oldAndNewDamage = np.zeros((d_np1.shape[0], 2))
            oldAndNewDamage[:, 0] = dn
            oldAndNewDamage[:, 1] = d_np1
            d_np1 = np.max(oldAndNewDamage, 1)

        else:
            raise Exception("Solveur phase field unknown")

        timeIter = tic.Tac("Resolution phase field", "Phase Field iteration", False)

        self.__Niter = Niter
        self.__convIter = convIter
        self.__timeIter = timeIter
            
        return u_np1, d_np1, Kglob, convergence


    def __Construct_Displacement_Matrix(self) -> np.ndarray:
        """Construct the elementary stiffness matrices for the displacement problem."""

        matrixType=MatrixType.rigi

        # Data
        mesh = self.mesh
        
        # Recovers matrices to work with        
        B_dep_e_pg = mesh.Get_B_e_pg(matrixType)
        leftDepPart = mesh.Get_leftDispPart(matrixType) # -> jacobian_e_pg * weight_pg * B_dep_e_pg'

        d = self.damage
        u = self.displacement

        phaseFieldModel = self.phaseFieldModel
        
        # Calculates the deformation required for the split
        Epsilon_e_pg = self._Calc_Epsilon_e_pg(u, matrixType)

        # Split of the behavior law
        cP_e_pg, cM_e_pg = phaseFieldModel.Calc_C(Epsilon_e_pg)

        tic = Tic()
        
        # Damage : c = g(d) * cP + cM
        g_e_pg = phaseFieldModel.get_g_e_pg(d, mesh, matrixType)
        cP_e_pg = np.einsum('ep,epij->epij', g_e_pg, cP_e_pg, optimize='optimal')

        c_e_pg = cP_e_pg + cM_e_pg
        
        # Elemental stiffness matrix
        Ku_e = np.sum(leftDepPart @ c_e_pg @ B_dep_e_pg, axis=1)

        if self.dim == 2:
            thickness = self.phaseFieldModel.thickness
            Ku_e *= thickness
        
        tic.Tac("Matrix","Construction Ku_e", self._verbosity)

        return Ku_e
 
    def __Assembly_displacement(self) -> sparse.csr_matrix:
        """Construct the displacement problem."""

        # Data
        mesh = self.mesh        
        nDof = mesh.Nn*self.dim
        
        nDof += self._Bc_Lagrange_dim(ModelType.displacement)

        Ku_e = self.__Construct_Displacement_Matrix()

        tic = Tic()

        linesVector_e = mesh.linesVector_e.reshape(-1)
        columnsVector_e = mesh.columnsVector_e.reshape(-1)

        # Assembly
        self.__Ku = sparse.csr_matrix((Ku_e.reshape(-1), (linesVector_e, columnsVector_e)), shape=(nDof, nDof))
        """Kglob matrix for the displacement problem (nDof, nDof)"""
        
        self.__Fu = sparse.csr_matrix((nDof, 1))
        """Fglob vector for the displacement problem (nDof, 1)"""

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.spy(self.__Ku)
        # plt.show()        

        tic.Tac("Matrix","Assembly Ku and Fu", self._verbosity)
        return self.__Ku

        # # Here, we always want the matrices to be updated with the latest damage or displacement results.
        # # That's why we don't say the matrices have been updated

    def __Solve_displacement(self) -> np.ndarray:
        """Solving the displacement problem."""
            
        self._Solver_Solve(ModelType.displacement)

        # Split need to be update
        self.phaseFieldModel.Need_Split_Update()
       
        return self.displacement

    # ------------------------------------------- PROBLEME ENDOMMAGEMENT ------------------------------------------- 

    def __Calc_psiPlus_e_pg(self):
        """Calculation of the positive energy density.
        For each gauss point of all mesh elements, we calculate psi+.
        """

        phaseFieldModel = self.phaseFieldModel
        
        u = self.displacement
        d = self.damage

        testu = isinstance(u, np.ndarray) and (u.shape[0] == self.mesh.Nn*self.dim )
        testd = isinstance(d, np.ndarray) and (d.shape[0] == self.mesh.Nn )

        assert testu or testd, "Dimension problem."

        Epsilon_e_pg = self._Calc_Epsilon_e_pg(u, MatrixType.mass)
        # here the mass term is important otherwise we under-integrate

        # Energy calculation
        psiP_e_pg, psiM_e_pg = phaseFieldModel.Calc_psi_e_pg(Epsilon_e_pg)

        if phaseFieldModel.solver == "History":
            # Get the old history field
            old_psiPlus_e_pg = self.__old_psiP_e_pg.copy()
            
            if isinstance(old_psiPlus_e_pg, list) and len(old_psiPlus_e_pg) == 0:
                # No damage available yet
                old_psiPlus_e_pg = np.zeros_like(psiP_e_pg)
            
            if old_psiPlus_e_pg.shape != psiP_e_pg.shape:
                # the mesh has been changed, the value must be recalculated
                # here I do nothing
                old_psiPlus_e_pg = np.zeros_like(psiP_e_pg)

            inc_H = psiP_e_pg - old_psiPlus_e_pg

            elements, gaussPoints = np.where(inc_H < 0)

            psiP_e_pg[elements, gaussPoints] = old_psiPlus_e_pg[elements, gaussPoints]

            # new = np.linalg.norm(psiP_e_pg)
            # old = np.linalg.norm(self.__old_psiP_e_pg)
            # assert new >= old, "Erreur"
            
        self.__psiP_e_pg = psiP_e_pg

        return self.__psiP_e_pg
    
    def __Construct_Damage_Matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Construct the elementary matrices for the damage problem."""

        phaseFieldModel = self.phaseFieldModel

        # Data
        k = phaseFieldModel.k
        PsiP_e_pg = self.__Calc_psiPlus_e_pg()
        r_e_pg = phaseFieldModel.get_r_e_pg(PsiP_e_pg)
        f_e_pg = phaseFieldModel.get_f_e_pg(PsiP_e_pg)

        matrixType=MatrixType.mass

        mesh = self.mesh
        Ne = mesh.Ne
        nPg = r_e_pg.shape[1]

        # K * Laplacien(d) + r * d = F        
        ReactionPart_e_pg = mesh.Get_ReactionPart_e_pg(matrixType) # -> jacobian_e_pg * weight_pg * Nd_pg' * Nd_pg
        DiffusePart_e_pg = mesh.Get_DiffusePart_e_pg(matrixType, phaseFieldModel.A) # -> jacobian_e_pg, weight_pg, Bd_e_pg', A, Bd_e_pg
        SourcePart_e_pg = mesh.Get_SourcePart_e_pg(matrixType) # -> jacobian_e_pg, weight_pg, Nd_pg'
        
        tic = Tic()

        # Part that involves the reaction term r ->  jacobian_e_pg * weight_pg * r_e_pg * Nd_pg' * Nd_pg
        K_r_e = np.einsum('ep,epij->eij', r_e_pg, ReactionPart_e_pg, optimize='optimal')

        # The part that involves diffusion K -> jacobian_e_pg, weight_pg, k, Bd_e_pg', Bd_e_pg
        k_e_pg = Reshape_variable(k, Ne, nPg)
        K_K_e = np.einsum('ep,epij->eij', k_e_pg, DiffusePart_e_pg, optimize='optimal')
        
        # Source part Fd_e -> jacobian_e_pg, weight_pg, f_e_pg, Nd_pg'
        Fd_e = np.einsum('ep,epij->eij', f_e_pg, SourcePart_e_pg, optimize='optimal')
    
        Kd_e = K_r_e + K_K_e

        if self.dim == 2:
            # THICKNESS not used in femobject !
            thickness = phaseFieldModel.thickness
            Kd_e *= thickness
            Fd_e *= thickness
        
        tic.Tac("Matrix","Construc Kd_e and Fd_e", self._verbosity)        

        return Kd_e, Fd_e

    def __Assembly_damage(self) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Construct the damage problem."""
       
        # Data
        mesh = self.mesh
        nDof = mesh.Nn
        linesScalar_e = mesh.linesScalar_e.reshape(-1)
        columnsScalar_e = mesh.columnsScalar_e.reshape(-1)

        # Additional dimension linked to the use of lagrange coefficients        
        nDof += self._Bc_Lagrange_dim(ModelType.damage)
        
        # Calculating elementary matrix
        Kd_e, Fd_e = self.__Construct_Damage_Matrix()

        # Assemblage
        tic = Tic()        

        self.__Kd = sparse.csr_matrix((Kd_e.reshape(-1), (linesScalar_e, columnsScalar_e)), shape = (nDof, nDof))
        """Kglob for damage problem (Nn, Nn)"""
        
        lignes = mesh.connect.reshape(-1)
        self.__Fd = sparse.csr_matrix((Fd_e.reshape(-1), (lignes,np.zeros(len(lignes)))), shape = (nDof,1))
        """Fglob for damage problem (Nn, 1)"""        

        tic.Tac("Matrix","Assembly Kd and Fd", self._verbosity)

        # # Here, we always want the matrices to be updated with the latest damage or displacement results.
        # # That's why we don't say the matrices have been updated

        return self.__Kd, self.__Fd
    
    def __Solve_damage(self) -> np.ndarray:
        """Solving the damage problem."""
        
        self._Solver_Solve(ModelType.damage)

        return self.damage

    def Save_Iter(self):

        iter = super().Save_Iter()

        # convergence information        
        iter["Niter"] = self.__Niter
        iter["timeIter"] = self.__timeIter
        iter["convIter"] = self.__convIter
    
        if self.phaseFieldModel.solver == PhaseField_Model.SolverType.History:
            # update old history field for next resolution
            self.__old_psiP_e_pg = self.__psiP_e_pg
            
        iter["displacement"] = self.displacement
        iter["damage"] = self.damage

        self._results.append(iter)

    def Set_Iter(self, iter=-1) -> list[dict]:

        results = super().Set_Iter(iter)

        if results is None: return

        self.Need_Update() # damage field will change thats why we need to update the assembled matrices
        self.__old_psiP_e_pg = [] # It's really useful to do this otherwise when we calculate psiP there will be a problem

        damageType = ModelType.damage
        self.set_u_n(damageType, results[damageType])

        displacementType = ModelType.displacement
        self.set_u_n(displacementType, results[displacementType])

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
        
        results.extend(["Svm","Stress","Evm","Strain"])

        results.extend(["damage","psiP","Psi_Crack"])
        results.extend(["Wdef"])

        return results
    
    def Result(self, result: str, nodeValues=True, iter=None) -> Union[np.ndarray, float, None]:
        
        if iter != None:
            self.Set_Iter(iter)
        
        if not self._Results_Check_Available(result): return None

        # begin cases ----------------------------------------------------

        Nn = self.mesh.Nn

        values = None

        if result in ["Wdef"]:
            return self._Calc_Psi_Elas()

        elif result == "Wdef_e":
            values = self._Calc_Psi_Elas(returnScalar=False)

        elif result == "Psi_Crack":
            return self._Calc_Psi_Crack()

        if result == "psiP":
            values_e_pg = self.__Calc_psiPlus_e_pg()
            values = np.mean(values_e_pg, axis=1)

        if result == "damage":
            values = self.damage

        elif result in ["ux", "uy", "uz"]:
            values_n = self.displacement.reshape(Nn, -1)
            values = values_n[:,self.__indexResult(result)]

        elif result == "displacement":
            values = self.displacement
        
        elif result == "displacement_norm":
            val_n = self.displacement.reshape(Nn, -1)
            values = np.linalg.norm(val_n, axis=1)

        elif result == "displacement_matrix":
            values = self.Results_displacement_matrix()
        
        elif ("S" in result or "E" in result) and (not "_norm" in result):
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
            
            values = Materials.Result_in_Strain_or_Stress_field(val_e, res, coef)

        if not isinstance(values, np.ndarray):
            myPrintError("This result option is not implemented yet.")
            return

        # end cases ----------------------------------------------------
        
        return self.Results_Reshape_values(values, nodeValues)

    def __indexResult(self, resultat: str) -> int:

        dim = self.dim

        if len(resultat) <= 2:
            if "x" in resultat:
                return 0
            elif "y" in resultat:
                return 1
            elif "z" in resultat:
                return 1

    def _Calc_Psi_Elas(self) -> float:
        """Calculation of the kinematically admissible deformation energy, damaged or not.
        Wdef = 1/2 int_Omega jacobian * weight * Sig : Eps dOmega thickness"""

        tic = Tic()

        u = self.displacement.reshape(-1,1)
        Ku = self.__Assembly_displacement()
        
        Wdef = 1/2 * float(u.T @ Ku @ u)

        tic.Tac("PostProcessing","Calc Psi Elas",False)
        
        return Wdef

    def _Calc_Psi_Crack(self) -> float:
        """Calculating crack energy."""

        tic = Tic()
        
        d = self.damage.reshape(-1,1)
        Kd = self.__Assembly_damage()[0]
        Psi_Crack = 1/2 * float(d.T @ Kd @ d)

        tic.Tac("PostProcessing","Calc Psi Crack",False)

        return Psi_Crack

    def _Calc_Epsilon_e_pg(self, sol: np.ndarray, matrixType=MatrixType.rigi):
        """Builds epsilon for each element and each gauss point.\n
        2D : [Exx Eyy sqrt(2)*Exy]\n
        3D : [Exx Eyy Ezz sqrt(2)*Eyz sqrt(2)*Exz sqrt(2)*Exy]

        Parameters
        ----------
        sol : np.ndarray
            Displacement vector

        Returns
        -------
        np.ndarray
            Deformations stored at elements and gauss points (Ne,pg,(3 or 6))
        """
        
        tic = Tic()        
        u_e = sol[self.mesh.assembly_e]
        B_dep_e_pg = self.mesh.Get_B_e_pg(matrixType)
        Epsilon_e_pg = np.einsum('epij,ej->epi', B_dep_e_pg, u_e, optimize='optimal')            
        
        tic.Tac("Matrix", "Epsilon_e_pg", False)

        return Epsilon_e_pg

    def _Calc_Sigma_e_pg(self, Epsilon_e_pg: np.ndarray, matrixType=MatrixType.rigi) -> np.ndarray:
        """Calculating stresses from strains.\n
        2D : [Sxx Syy sqrt(2)*Sxy]\n
        3D : [Sxx Syy Szz sqrt(2)*Syz sqrt(2)*Sxz sqrt(2)*Sxy]

        Parameters
        ----------
        Epsilon_e_pg : np.ndarray
            Deformations stored at elements and gauss points (Ne,pg,(3 or 6))

        Returns
        -------
        np.ndarray
            Returns damaged or undamaged constraints (Ne,pg,(3 or 6))
        """

        assert Epsilon_e_pg.shape[0] == self.mesh.Ne
        assert Epsilon_e_pg.shape[1] == self.mesh.Get_nPg(matrixType)

        d = self.damage

        phaseFieldModel = self.phaseFieldModel

        SigmaP_e_pg, SigmaM_e_pg = phaseFieldModel.Calc_Sigma_e_pg(Epsilon_e_pg)
        
        # Damage : Sig = g(d) * SigP + SigM
        g_e_pg = phaseFieldModel.get_g_e_pg(d, self.mesh, matrixType)
        
        tic = Tic()
        
        SigmaP_e_pg = np.einsum('ep,epi->epi', g_e_pg, SigmaP_e_pg, optimize='optimal')

        Sigma_e_pg = SigmaP_e_pg + SigmaM_e_pg
            
        tic.Tac("Matrix", "Sigma_e_pg", False)

        return Sigma_e_pg

    def Results_Set_Bc_Summary(self, loadMax: float, listInc: list, listTreshold: list, listOption: list):
        assert len(listInc) == len(listTreshold) and len(listInc) == len(listOption), "Must be the same dimension."
        
        resumeChargement = 'Chargement :'
        resumeChargement += f'\n\tload max = {loadMax:.3}'

        for inc, treshold, option in zip(listInc, listTreshold, listOption):

            resumeChargement += f'\n\tinc = {inc} -> {option} < {treshold:.4e}'
        
        self.__resumeChargement = resumeChargement

        return self.__resumeChargement

    def Results_Get_Bc_Summary(self) -> str:
        try:
            return self.__resumeChargement
        except AttributeError:
            return ""

    def Results_Set_Iteration_Summary(self, iter: int, load: float, uniteLoad: str, percentage=0.0, remove=False) -> str:
        """Builds the iteration summary for the damage problem

        Parameters
        ----------
        iter : int
            iteration
        load : float
            loading
        uniteLoad : str
            loading unit
        percentage : float, optional
            percentage of simualtion performed, by default 0.0
        remove : bool, optional
            removes line from terminal after display, by default False
        """

        d = self.damage

        nombreIter = self.__Niter
        dincMax = self.__convIter
        timeIter = self.__timeIter

        min_d = d.min()
        max_d = d.max()
        summaryIter = f"{iter:4} : {load:4.3f} {uniteLoad}, [{min_d:.2e}; {max_d:.2e}], {nombreIter}:{timeIter:4.3f} s, tol={dincMax:.2e}  "
        
        if remove:
            end='\r'
        else:
            end=''

        if percentage > 0:
            timeLeft = (1/percentage-1)*timeIter*iter
            
            timeCoef, unite = Tic.Get_time_unity(timeLeft)

            # Adds percentage and estimated time remaining
            summaryIter = summaryIter+f"{np.round(percentage*100,2):3.2f} % -> {timeCoef:4.2f} {unite}  "

        myPrint(summaryIter, end=end)

        self.__resumeIter = summaryIter

    def Results_Get_Iteration_Summary(self) -> str:        
        return self.__resumeIter

    def Results_dict_Energy(self) -> dict[str, float]:
        PsiElas = self._Calc_Psi_Elas()
        PsiCrack = self._Calc_Psi_Crack()
        dict_Energie = {
            r"$\Psi_{elas}$": PsiElas,
            r"$\Psi_{crack}$": PsiCrack,
            r"$\Psi_{tot}$": PsiCrack+PsiElas
            }
        return dict_Energie

    def Results_Iter_Summary(self) -> list[tuple[str, np.ndarray]]:
        
        list_label_values = []
        
        resultats = self.results
        df = pd.DataFrame(resultats)
        iterations = np.arange(df.shape[0])
        
        damageMaxIter = np.array([np.max(damage) for damage in df["damage"].values])
        list_label_values.append((r"$\phi$", damageMaxIter))

        tolConvergence = df["convIter"].values
        list_label_values.append(("converg", tolConvergence))

        nombreIter = df["Niter"].values
        list_label_values.append(("Niter", nombreIter))

        tempsIter = df["timeIter"].values
        list_label_values.append(("time", tempsIter))
        
        return iterations, list_label_values
    
    def Results_displacement_matrix(self) -> np.ndarray:
        
        Nn = self.mesh.Nn
        coordo = self.displacement.reshape((Nn,-1))
        dim = coordo.shape[1]

        if dim == 1:
            # Here we add two columns
            coordo = np.append(coordo, np.zeros((Nn,1)), axis=1)
            coordo = np.append(coordo, np.zeros((Nn,1)), axis=1)
        elif dim == 2:
            # Here we add 1 column
            coordo = np.append(coordo, np.zeros((Nn,1)), axis=1)

        return coordo
    

###################################################################################################

class Simu_Beam(_Simu):

    def __init__(self, mesh: Mesh, model: Beam_Structure, verbosity=False, useNumba=True, useIterativeSolvers=True):
        """
        Creates a Euler-Bernoulli beam simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh the simulation will use.
        model : IModel
            The model used.
        verbosity : bool, optional
            If True, the simulation can write to the console. Defaults to False.
        useNumba : bool, optional
            If True, numba can be used. Defaults to True.
        useIterativeSolvers : bool, optional
            If True, iterative solvers can be used. Defaults to True.
        """
        
        assert model.modelType == ModelType.beam, "The material must be beam model"
        super().__init__(mesh, model, verbosity, useNumba, useIterativeSolvers)

        # init
        self.Solver_Set_Elliptic_Algorithm()

    def Results_nodesField_elementsField(self, details=False) -> tuple[list[str], list[str]]:
        if details:
            nodesField = ["displacement_matrix"]
            elementsField = ["Stress"]
        else:
            nodesField = ["displacement_matrix"]
            elementsField = ["Stress"]
        return nodesField, elementsField

    def Get_directions(self, problemType=None) -> list[str]:
        dict_nbddl_directions = {
            1 : ["x"],
            3 : ["x","y","rz"],
            6 : ["x","y","z","rx","ry","rz"]
        }
        return dict_nbddl_directions[self.structure.dof_n]
    
    def Get_problemTypes(self) -> list[ModelType]:
        return [ModelType.beam]

    @property
    def structure(self) -> Beam_Structure:
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
        return self.get_u_n(self.problemType)

    def add_surfLoad(self, nodes: np.ndarray, values: list, directions: list, problemType=None, description=""):
        myPrintError("It is impossible to apply a surface load in a beam problem.")
        return

    def add_volumeLoad(self, nodes: np.ndarray, values: list, directions: list, problemType=None, description=""):
        myPrintError("It is impossible to apply a volumetric load to a beam problem.")
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
        """Connects beams together in the specified directions

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
        self._Check_Directions(problemType, directions)

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
        """Construct the elementary stiffness matrices for the beam problem."""

        # Data
        mesh = self.mesh
        if not mesh.groupElem.dim == 1: return
        groupElem = mesh.groupElem

        # Recovering the beam model
        beamModel = self.structure
        
        matrixType=MatrixType.beam
        
        tic = Tic()
        
        jacobian_e_pg = mesh.Get_jacobian_e_pg(matrixType)
        weight_pg = mesh.Get_weight_pg(matrixType)

        D_e_pg = beamModel.Calc_D_e_pg(groupElem)        

        B_beam_e_pg = self._Get_B_beam_e_pg()
        
        Kbeam_e = np.einsum('ep,p,epji,epjk,epkl->eil', jacobian_e_pg, weight_pg, B_beam_e_pg, D_e_pg, B_beam_e_pg, optimize='optimal')
            
        tic.Tac("Matrix","Construct Kbeam_e", self._verbosity)

        return Kbeam_e
    
    @property
    def mass(self) -> float:

        matrixType = MatrixType.mass

        mesh = self.mesh

        group = mesh.groupElem

        coordo_e_p = group.Get_GaussCoordinates_e_p(matrixType)

        jacobian_e_p = group.Get_jacobian_e_pg(matrixType)
        
        weight_p = group.Get_weight_pg(matrixType)        

        rho_e_p = Reshape_variable(self.rho, mesh.Ne, weight_p.size)

        area_e = np.zeros(mesh.Ne)

        for beam in self.structure.listBeam:

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
        for beam in self.structure.listBeam:
            elements = mesh.Elements_Tags(beam.name)
            area_e[elements] = beam.area

        center: np.ndarray = np.einsum('ep,e,ep,p,epi->i', rho_e_p, area_e, jacobian_e_p, weight_p, coordo_e_p, optimize='optimal') / mass

        if not isinstance(self.rho, np.ndarray):
            diff = np.linalg.norm(center - mesh.center)/np.linalg.norm(center)
            assert diff <= 1e-12

        return center

    def _Get_N_beam_e_pg(self):
        """Euleur-Bernoulli BEAM"""

        # Exemple matlab : FEMOBJECT/BASIC/MODEL/ELEMENTS/@BEAM/calc_N.m
        
        tic = Tic()

        matrixType = MatrixType.beam

        # Récupération du maodel poutre
        struct = self.structure
        dim = struct.dim
        dof_n = struct.dof_n

        # Data
        mesh = self.mesh
        jacobian_e_pg = mesh.Get_jacobian_e_pg(matrixType)
        groupElem = mesh.groupElem
        elemType = groupElem.elemType
        nPe = groupElem.nPe
        Ne = jacobian_e_pg.shape[0]
        nPg = jacobian_e_pg.shape[1]

        # Recovers matrices to work with
        N_pg = mesh.Get_N_pg(matrixType)        
        if struct.dim > 1:
            Nv_e_pg = mesh.groupElem.Get_Nv_e_pg()
            dNv_e_pg = mesh.groupElem.Get_dNv_e_pg()

        if dim == 1:
            # u = [u1, . . . , un]
            
            # N = [N_i, . . . , N_n]

            idx_ux = np.arange(dof_n*nPe)

            Nbeam_e_pg = np.zeros((Ne, nPg, 1, dof_n*nPe))
            Nbeam_e_pg[:,:,0, idx_ux] = N_pg[:,:,0]
                
        elif dim == 2:
            # u = [u1, v1, rz1, . . . , un, vn, rzn]
            
            # N = [N_i, 0, 0, ... , N_n, 0, 0,]
            #     [0, Phi_i, Psi_i, ... , 0, Phi_i, Psi_i]
            #     [0, dPhi_i, dPsi_i, ... , 0, dPhi_i, dPsi_i]

            idx = np.arange(dof_n*nPe).reshape(nPe,-1)
            
            idx_ux = idx[:,0] # [0,3] (SEG2) [0,3,6] (SEG3)
            idx_uy = np.reshape(idx[:,1:], -1) # [1,2,4,5] (SEG2) [1,2,4,5,7,8] (SEG3)

            Nbeam_e_pg = np.zeros((Ne, nPg, 3, dof_n*nPe))
            
            Nbeam_e_pg[:,:,0, idx_ux] = N_pg[:,:,0] # traction / compression to get u
            Nbeam_e_pg[:,:,1, idx_uy] = Nv_e_pg[:,:,0] # flexion z to get v
            Nbeam_e_pg[:,:,2, idx_uy] = dNv_e_pg[:,:,0] # flexion z to get rz

        elif dim == 3:
            # u = [u1, v1, w1, rx1, ry1, rz1, . . . , un, vn, wn, rxn, ryn, rzn]

            # N = [N_i, 0, 0, 0, 0, 0, ... , N_n, 0, 0, 0, 0, 0]
            #     [0, Phi_i, 0, 0, 0, Psi_i, ... , 0, Phi_n, 0, 0, 0, Psi_n]
            #     [0, 0, dPhi_i, 0, -dPsi_i, 0, ... , 0, 0, dPhi_n, 0, -dPsi_n, 0]
            #     [0, 0, 0, N_i, 0, 0, ... , 0, 0, 0, N_n, 0, 0]
            #     [0, 0, -dPhi_i, 0, dPsi_i, 0, ... , 0, 0, -dPhi_n, 0, dPsi_n, 0]
            #     [0, dPhi_i, 0, 0, 0, dPsi_i, ... , 0, dPhi_i, 0, 0, 0, dPsi_n]

            idx = np.arange(dof_n*nPe).reshape(nPe,-1)
            idx_ux = idx[:,0] # [0,6] (SEG2) [0,6,12] (SEG3)
            idx_uy = np.reshape(idx[:,[1,5]], -1) # [1,5,7,11] (SEG2) [1,5,7,11,13,17] (SEG3)
            idx_uz = np.reshape(idx[:,[2,4]], -1) # [2,4,8,10] (SEG2) [2,4,8,10,14,16] (SEG3)
            idx_rx = idx[:,3] # [3,9] (SEG2) [3,9,15] (SEG3)
            idPsi = np.arange(1, nPe*2, 2) # [1,3] (SEG2) [1,3,5] (SEG3)

            Nvz_e_pg = Nv_e_pg.copy()
            Nvz_e_pg[:,:,0,idPsi] *= -1

            dNvz_e_pg = dNv_e_pg.copy()
            dNvz_e_pg[:,:,0,idPsi] *= -1

            Nbeam_e_pg = np.zeros((Ne, nPg, 6, dof_n*nPe))
            
            Nbeam_e_pg[:,:,0, idx_ux] = N_pg[:,:,0]
            Nbeam_e_pg[:,:,1, idx_uy] = Nv_e_pg[:,:,0]
            Nbeam_e_pg[:,:,2, idx_uz] = Nvz_e_pg[:,:,0]
            Nbeam_e_pg[:,:,3, idx_rx] = N_pg[:,:,0]
            Nbeam_e_pg[:,:,4, idx_uz] = -dNvz_e_pg[:,:,0] # ry = -uz'
            Nbeam_e_pg[:,:,5, idx_uy] = dNv_e_pg[:,:,0] # rz = uy'
        
        if dim > 1:
            # Construct the matrix used to change the matrix coordinates 
            P = np.zeros((self.mesh.Ne, 3, 3), dtype=float)
            for beam in struct.listBeam:
                elems = self.mesh.Elements_Tags([beam.name])
                P[elems] = beam._Calc_P()

            Pglob_e = np.zeros((Ne, dof_n*nPe, dof_n*nPe))            
            N = P.shape[1]
            lines = np.repeat(range(N), N)
            columns = np.array(list(range(N))*N)
            for n in range(dof_n*nPe//3):
                # apply P on the diagonal
                Pglob_e[:, lines + n*N, columns + n*N] = P[:,lines,columns]

            N_beam_e_pg = np.einsum('epij,ejk->epik', Nbeam_e_pg, Pglob_e, optimize='optimal')

        tic.Tac("Matrix","Construct N_beam_e_pg", False)

        return N_beam_e_pg

    def _Get_B_beam_e_pg(self):
        """Euleur-Bernoulli BEAM"""

        # Exemple matlab : FEMOBJECT/BASIC/MODEL/ELEMENTS/@BEAM/calc_B.m
        
        tic = Tic()

        matrixType = MatrixType.beam

        # Récupération du maodel poutre
        struct = self.structure
        dim = struct.dim
        dof_n = struct.dof_n

        # Data
        mesh = self.mesh
        jacobian_e_pg = mesh.Get_jacobian_e_pg(matrixType)
        groupElem = mesh.groupElem
        elemType = groupElem.elemType
        nPe = groupElem.nPe
        Ne = jacobian_e_pg.shape[0]
        nPg = jacobian_e_pg.shape[1]

        # Recovers matrices to work with
        dN_e_pg = mesh.Get_dN_e_pg(matrixType)
        if struct.dim > 1:
            ddNv_e_pg = mesh.groupElem.Get_ddNv_e_pg()

        if dim == 1:
            # u = [u1, . . . , un]
            
            # B = [dN_i, . . . , dN_n]

            idx_ux = np.arange(dof_n*nPe)

            B_beam_e_pg = np.zeros((Ne, nPg, 1, dof_n*nPe))
            B_beam_e_pg[:,:,0, idx_ux] = dN_e_pg[:,:,0]
                
        elif dim == 2:
            # u = [u1, v1, rz1, . . . , un, vn, rzn]
            
            # B = [dN_i, 0, 0, ... , dN_n, 0, 0,]
            #     [0, ddPhi_i, ddPsi_i, ... , 0, ddPhi_i, ddPsi_i]

            idx = np.arange(dof_n*nPe).reshape(nPe,-1)
            
            idx_ux = idx[:,0] # [0,3] (SEG2) [0,3,6] (SEG3)
            idx_uy = np.reshape(idx[:,1:], -1) # [1,2,4,5] (SEG2) [1,2,4,5,7,8] (SEG3)

            B_beam_e_pg = np.zeros((Ne, nPg, 2, dof_n*nPe))
            
            B_beam_e_pg[:,:,0, idx_ux] = dN_e_pg[:,:,0] # traction / compression
            B_beam_e_pg[:,:,1, idx_uy] = ddNv_e_pg[:,:,0] # flexion along z

        elif dim == 3:
            # u = [u1, v1, w1, rx1, ry1, rz1, . . . , un, vn, wn, rxn, ryn, rzn]

            # B = [dN_i, 0, 0, 0, 0, 0, ... , dN_n, 0, 0, 0, 0, 0]
            #     [0, 0, 0, dN_i, 0, 0, ... , 0, 0, 0, dN_n, 0, 0]
            #     [0, 0, ddPhi_i, 0, -ddPsi_i, 0, ... , 0, 0, ddPhi_n, 0, -ddPsi_n, 0]
            #     [0, ddPhi_i, 0, 0, 0, ddPsi_i, ... , 0, ddPhi_i, 0, 0, 0, ddPsi_n]

            idx = np.arange(dof_n*nPe).reshape(nPe,-1)
            idx_ux = idx[:,0] # [0,6] (SEG2) [0,6,12] (SEG3)
            idx_uy = np.reshape(idx[:,[1,5]], -1) # [1,5,7,11] (SEG2) [1,5,7,11,13,17] (SEG3)
            idx_uz = np.reshape(idx[:,[2,4]], -1) # [2,4,8,10] (SEG2) [2,4,8,10,14,16] (SEG3)
            idx_rx = idx[:,3] # [3,9] (SEG2) [3,9,15] (SEG3)
            
            idPsi = np.arange(1, nPe*2, 2) # [1,3] (SEG2) [1,3,5] (SEG3)
            ddNvz_e_pg = ddNv_e_pg.copy()
            ddNvz_e_pg[:,:,0,idPsi] *= -1 # RY = -UZ'

            B_beam_e_pg = np.zeros((Ne, nPg, 4, dof_n*nPe))
            
            B_beam_e_pg[:,:,0, idx_ux] = dN_e_pg[:,:,0] # traction / compression
            B_beam_e_pg[:,:,1, idx_rx] = dN_e_pg[:,:,0] # torsion
            B_beam_e_pg[:,:,2, idx_uz] = ddNvz_e_pg[:,:,0] # flexion along y
            B_beam_e_pg[:,:,3, idx_uy] = ddNv_e_pg[:,:,0] # flexion along z        

        if dim > 1:
            # Construct the matrix used to change the matrix coordinates 
            P = np.zeros((self.mesh.Ne, 3, 3), dtype=float)
            for beam in struct.listBeam:
                elems = self.mesh.Elements_Tags([beam.name])
                P[elems] = beam._Calc_P()

            Pglob_e = np.zeros((Ne, dof_n*nPe, dof_n*nPe))            
            N = P.shape[1]
            lines = np.repeat(range(N), N)
            columns = np.array(list(range(N))*N)
            for n in range(dof_n*nPe//3):
                # apply P on the diagonal
                Pglob_e[:, lines + n*N, columns + n*N] = P[:,lines,columns]

            B_beam_e_pg = np.einsum('epij,ejk->epik', B_beam_e_pg, Pglob_e, optimize='optimal')

        tic.Tac("Matrix","Construct B_beam_e_pg", False)

        return B_beam_e_pg

    def Assembly(self):

        if self.needUpdate:

            # Data
            mesh = self.mesh

            model = self.structure

            nDof = mesh.Nn * model.dof_n

            Ku_beam = self.__Construct_Beam_Matrix()

            # Additional dimension linked to the use of lagrange coefficients
            nDof += self._Bc_Lagrange_dim(self.problemType)
            
            tic = Tic()

            lignesVector_e = mesh.Get_linesVector_e(model.dof_n).reshape(-1)
            colonnesVector_e = mesh.Get_columnsVector_e(model.dof_n).reshape(-1)

            # Assembly
            self.__Kbeam = sparse.csr_matrix((Ku_beam.reshape(-1), (lignesVector_e, colonnesVector_e)), shape=(nDof, nDof))
            """Kglob matrix for beam problem (nDof, nDof)"""

            self.__Fbeam = sparse.csr_matrix((nDof, 1))
            """Fglob vector for beam problem (nDof, 1)"""

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.spy(self.__Ku)
            # plt.show()

            self.Need_Update(False)

            tic.Tac("Matrix","Assembly Kbeam and Fbeam", self._verbosity)

    def Get_K_C_M_F(self, problemType=None) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        if self.needUpdate: self.Assembly()
        size = self.mesh.Nn * self.Get_dof_n(problemType)
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

    def Set_Iter(self, iter=-1) -> list[dict]:
        
        results = super().Set_Iter(iter)

        if results is None: return

        self.set_u_n(self.problemType, results["displacement"])

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
        B_beam_e_pg = self._Get_B_beam_e_pg()
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
        for beam in self.structure.listBeam:
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

        # Affichage des déplacements
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

###################################################################################################

class Simu_Thermal(_Simu):

    def __init__(self, mesh: Mesh, model: Thermal_Model, verbosity=False, useNumba=True, useIterativeSolvers=True):
        """
        Creates a thermal simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh the simulation will use.
        model : IModel
            The model used.
        verbosity : bool, optional
            If True, the simulation can write to the console. Defaults to False.
        useNumba : bool, optional
            If True, numba can be used. Defaults to True.
        useIterativeSolvers : bool, optional
            If True, iterative solvers can be used. Defaults to True.
        """

        assert model.modelType == ModelType.thermal, "The material must be thermal model"
        super().__init__(mesh, model, verbosity, useNumba, useIterativeSolvers)

        # init
        self.Solver_Set_Elliptic_Algorithm()
    
    def Get_directions(self, problemType=None) -> list[str]:
        return [""]
    
    def Get_dof_n(self, problemType=None) -> int:
        return 1

    def Results_nodesField_elementsField(self, details=False) -> tuple[list[str], list[str]]:
        nodesField = ["thermal", "thermalDot"]
        elementsField = []
        return nodesField, elementsField
    
    def Get_problemTypes(self) -> list[ModelType]:
        return [ModelType.thermal]

    @property
    def thermalModel(self) -> Thermal_Model:
        """Thermal simulation model."""
        return self.model

    @property
    def thermal(self) -> np.ndarray:
        """Scalar temperature field.\n
        [ti, ....]"""
        return self.get_u_n(self.problemType)

    @property
    def thermalDot(self) -> np.ndarray:
        """Time derivative of the scalar temperature field.\n
        [d(ti)/dt, ....]"""
        return self.get_v_n(self.problemType)

    def Get_x0(self, problemType=None):
        if self.thermal.size != self.mesh.Nn:
            return np.zeros(self.mesh.Nn)
        else:
            return self.thermal

    def Get_K_C_M_F(self, problemType=None) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        if self.needUpdate: self.Assembly()
        size = self.mesh.Nn * self.Get_dof_n(problemType)
        initcsr = sparse.csr_matrix((size, size))
        return self.__Kt.copy(), self.__Ct.copy(), initcsr, self.__Ft.copy()

    def __Construct_Thermal_Matrix(self) -> tuple[np.ndarray, np.ndarray]:

        thermalModel = self.thermalModel

        # Data
        k = thermalModel.k
        rho = self.rho
        c = thermalModel.c

        matrixType=MatrixType.rigi

        mesh = self.mesh

        jacobian_e_pg = mesh.Get_jacobian_e_pg(matrixType)
        weight_pg = mesh.Get_weight_pg(matrixType)
        N_e_pg = mesh.Get_N_pg(matrixType)
        D_e_pg = mesh.Get_dN_e_pg(matrixType)
        Ne = mesh.Ne
        nPg = weight_pg.size

        k_e_pg = Reshape_variable(k, Ne, nPg)

        Kt_e = np.einsum('ep,p,epji,ep,epjk->eik', jacobian_e_pg, weight_pg, D_e_pg, k_e_pg, D_e_pg, optimize="optimal")

        rho_e_pg = Reshape_variable(rho, Ne, nPg)
        c_e_pg = Reshape_variable(c, Ne, nPg)

        Ct_e = np.einsum('ep,p,pji,ep,ep,pjk->eik', jacobian_e_pg, weight_pg, N_e_pg, rho_e_pg, c_e_pg, N_e_pg, optimize="optimal")

        if self.dim == 2:
            epaisseur = thermalModel.thickness
            Kt_e *= epaisseur
            Ct_e *= epaisseur

        return Kt_e, Ct_e

    def Assembly(self) -> None:
        """Construct the matrix system for the thermal problem in stationary or transient regime."""

        if self.needUpdate:
       
            # Data
            mesh = self.mesh
            nDof = mesh.Nn
            linesScalar_e = mesh.linesScalar_e.reshape(-1)
            columnsScalar_e = mesh.columnsScalar_e.reshape(-1)

            # Additional dimension linked to the use of lagrange coefficients
            nDof += self._Bc_Lagrange_dim(self.problemType)
            
            # Calculating elementary matrices
            Kt_e, Mt_e = self.__Construct_Thermal_Matrix()
            
            tic = Tic()

            self.__Kt = sparse.csr_matrix((Kt_e.reshape(-1), (linesScalar_e, columnsScalar_e)), shape = (nDof, nDof))
            """Kglob for thermal problem (Nn, Nn)"""
            
            self.__Ft = sparse.csr_matrix((nDof, 1))
            """Fglob vector for thermal problem (Nn, 1)."""

            self.__Ct = sparse.csr_matrix((Mt_e.reshape(-1), (linesScalar_e, columnsScalar_e)), shape = (nDof, nDof))
            """Mglob for thermal problem (Nn, Nn)"""

            tic.Tac("Matrix","Assembly Kt, Mt and Ft", self._verbosity)

            self.Need_Update(False)

    def Save_Iter(self):

        iter = super().Save_Iter()
        
        iter['thermal'] = self.thermal

        if self.algo == AlgoType.parabolic:
            iter['thermalDot'] = self.thermalDot
            
        self._results.append(iter)

    def Set_Iter(self, iter=-1) -> list[dict]:
        
        results = super().Set_Iter(iter)

        if results is None: return

        self.set_u_n(ModelType.thermal, results["thermal"])

        if self.algo == AlgoType.parabolic and "thermalDot" in results:
            self.set_v_n(ModelType.thermal, results["thermalDot"])
        else:
            self.set_v_n(ModelType.thermal, np.zeros_like(self.thermal))

        return results

    def Results_Available(self) -> list[str]:
        options = []
        options.extend(["thermal", "thermalDot", "displacement_matrix"])
        return options
        
    def Result(self, result: str, nodeValues=True, iter=None) -> Union[np.ndarray, float, None]:

        if iter != None:
            self.Set_Iter(iter)
        
        if not self._Results_Check_Available(result): return None

        # begin cases ----------------------------------------------------

        if result == "thermal":
            values = self.thermal

        elif result == "thermalDot":
            values = self.thermalDot

        elif result == "displacement_matrix":
            values = self.Results_displacement_matrix()

        # end cases ----------------------------------------------------
        
        return self.Results_Reshape_values(values, nodeValues)

    def Results_Iter_Summary(self) -> list[tuple[str, np.ndarray]]:
        return super().Results_Iter_Summary()

    def Results_dict_Energy(self) -> list[tuple[str, float]]:
        return super().Results_dict_Energy()
    
    def Results_displacement_matrix(self) -> np.ndarray:
        return super().Results_displacement_matrix()