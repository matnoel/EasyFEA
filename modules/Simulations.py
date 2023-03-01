from abc import ABC, abstractmethod, abstractproperty

import os
import pickle
from colorama import Fore
from datetime import datetime
from types import LambdaType
from typing import cast
import platform

import numpy as np
import pandas as pd
from scipy import sparse

# Meme si pas utilisé laissé l'acces
from Mesh import Mesh, MatriceType, ElemType
from BoundaryCondition import BoundaryCondition, LagrangeCondition
from Materials import ModelType, IModel, Displacement_Model, Beam_Model, PhaseField_Model, Thermal_Model
from TicTac import Tic
import CalcNumba
from Interface_Solveurs import ResolutionType, AlgoType, _SolveProblem, _Solve_Axb, Solvers
import Folder

def Load_Simu(folder: str, verbosity=False):
    """Charge la simulation depuis le dossier

    Parameters
    ----------
    folder : str
        nom du dossier dans lequel simulation est sauvegardée

    Returns
    -------
    Simu
        simu
    """

    filename = Folder.Join([folder, "simulation.pickle"])
    erreur = Fore.RED + "Le fichier simulation.pickle est introuvable" + Fore.WHITE
    assert os.path.exists(filename), erreur

    with open(filename, 'rb') as file:
        simu = pickle.load(file)

    assert isinstance(simu, Simu)

    if verbosity:
        print(Fore.CYAN + f'\nChargement de :\n{filename}\n' + Fore.WHITE)
        simu.mesh.Resume()
        print(simu.model.resume)
    return simu

class Simu(ABC):
    """Classe mère de :\n
     - Simu_Displacement
     - Simu_Damage
     - Simu_Beam
     - Simu_Thermal

    # 14 méthodes à définir pour respecter l'interface / l'héritage

    - directions ddls et matériau :
    
        def Get_problemTypes(self) -> list[ModelType]:
            
        def Get_Resultats_disponibles(self) -> list[str]:

        def Get_Directions(self, problemType=None):
        
        def Get_nbddl_n(self, problemType=None) -> int:

    - Solveur :
    
        def Get_K_C_M_F(self, problemType=None) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:    
        
        def Get_x0(self, problemType=None):
        
        def Assemblage(self):

    - Itérations :
    
        def Save_Iteration(self) -> dict:
        
        def Update_iter(self, index=-1) -> list[dict]:

    - Resultats :
    
        def Get_Resultat(self, option: str, nodeValues=True, iter=None):
        
        def Resultats_Get_ResumeIter_values(self) -> tuple[list[int], list[tuple[str, np.ndarray]]]:    
    """

    # ================================================ ABSTRACT METHOD ================================================
    #     
    @abstractmethod
    def Get_problemTypes(self) -> list[ModelType]:
        """Problèmes/modèles disponibles par la simulation"""
        pass
    
    @abstractmethod
    def Get_Directions(self, problemType=None) -> list[str]:
        """Liste de directions disponibles dans la simulation"""
        pass
    
    @abstractmethod
    def Get_Resultats_disponibles(self) -> list[str]:
        """Donne la liste de résultats auxquelles la peut accéder"""
        pass

    @abstractmethod
    def Get_nbddl_n(self, problemType=None) -> int:
        """Degrés de liberté par noeud"""
        pass

    # Solveurs
    @abstractmethod
    def Get_K_C_M_F(self, problemType=None) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        """Renvoie les matrices assemblées de K u + C v + M a = F"""
        pass
    
    @abstractmethod
    def Get_x0(self, problemType=None):
        """Renvoie la solution de l'itération précédente"""
        return []

    @abstractmethod
    def Assemblage(self):
        """Assemblage du système matriciel"""
        pass

    # Itérations
    @abstractmethod
    def Save_Iteration(self) -> dict:
        """Sauvegarde les résultats de l'itération dans results
        """
        iter = {}

        iter["indexMesh"] = self.__indexMesh
        # identifiant du maillage à cette itération

        return iter
    
    @abstractmethod
    def Update_iter(self, index=-1) -> list[dict]:
        """Mets la simulation à l'itération renseignée (de base la dernière) et renvoie la liste de dictionnaire"""
        index = int(index)
        assert isinstance(index, int), print("Doit fournir un entier")

        indexMax = len(self._results)-1
        assert index <= indexMax, f"L'index doit etre < {indexMax}]"

        # On va venir récupérer les résultats stockés dans le tableau pandas
        results =  self._results[index]

        self.__Update_mesh(index)

        return results

    # Resultats

    @abstractmethod
    def Get_Resultat(self, option: str, nodeValues=True, iter=None) -> np.ndarray | float:
        """Renvoie le résultat de la simulation (np.ndarray ou float)
        """
        pass

    @abstractmethod
    def Resultats_Get_ResumeIter_values(self) -> tuple[list[int], list[tuple[str, np.ndarray]]]:
        """Renvoie les valeurs a afficher dans Plot_ResumeIter
        """
        return [], []

    @abstractmethod
    def Resultats_Get_dict_Energie(self) -> dict[str, float]:
        """Renvoie une liste de tuple contenant les noms et les valeurs des énergies calculées"""
        return {}
    
    @abstractmethod
    def Resultats_matrice_displacement(self) -> np.ndarray:
        """Renvoie les déplacements sous la forme d'une matrice [dx, dy, dz] (Nn,3)"""
        Nn = self.mesh.Nn
        return np.array((Nn,3))

    @abstractmethod
    def Paraview_nodesField_elementsField(self, details=False) -> tuple[list[str], list[str]]:
        """Renvoie les listes détaillées ou non pour la récupération des nodesField et elementsField affichés dans paraview.
        """
        return [], []

    # ================================================ SIMU ================================================

    def Check_Directions(self, problemType : ModelType, directions:list):
        """Vérifie si les directions renseignées sont possibles pour le problème"""
        listDirections = self.Get_Directions(problemType)
        for d in directions: assert d in listDirections, f"{d} n'est pas dans [{listDirections}]"

    def __Check_ProblemTypes(self, problemType : ModelType):
        """Vérifie si ce type de problème est disponible par la simulation"""
        assert problemType in self.Get_problemTypes(), f"Ce type de problème n'est pas disponible dans cette simulation ({self.Get_problemTypes()})"
    
    def Check_dim_mesh_materiau(self) -> None:
        """On vérifie que la dimension du matériau correspond à la dimension du maillage"""
        assert self.__model.dim == self.__mesh.dim, "Le matériau doit avoir la même dimension que le maillage"

    def __init__(self, mesh: Mesh, model: IModel, verbosity=True, useNumba=True):
        """Creation d'une simulation

        Args:
            dim (int): Dimension de la simulation (2D ou 3D)
            mesh (Mesh): Maillage que la simulation va utiliser
            model (IModel): Modèle utilisé
            verbosity (bool, optional): La simulation peut écrire dans la console. Defaults to True.
        """
        
        if verbosity:
            import Affichage
            Affichage.NouvelleSection("Simulation")

        self.__results = []
        """liste de dictionnaire qui contient les résultats"""

        # Renseigne le premier maillage
        self.__indexMesh = -1
        """index du maillage actuel dans self.__listMesh"""
        self.__listMesh = cast(list[Mesh], [])
        self.mesh = mesh

        self.__rho = 1
        """masse volumique"""
        
        self.__model = model

        self.Check_dim_mesh_materiau()

        self.__dim = model.dim
        """dimension de la simulation 2D ou 3D"""
        self._verbosity = verbosity
        """la simulation peut écrire dans la console"""
        
        self.__algo = AlgoType.elliptic
        """algorithme de résolution du système lors de la simulation"""
        # de base l'algo résout des problèmes stationnaires
        
        # Solveur utilisé lors de la résolution
        self.__solver = "scipy" # initialise au cas ou
        if "arm" in platform.machine():
            # peut pas utiliser pypardiso ;(
            self.solver = "petsc"
        else:
            self.solver = "pypardiso"        

        self.__Init_Sols_n()

        self.useNumba = useNumba

        # Conditions Limites
        self.Bc_Init()

        self.Need_Update()

    @property
    def model(self) -> IModel:
        """Modèle utilisé"""
        return self.__model

    @property
    def rho(self) -> float:
        """masse volumique"""
        return self.__rho

    @rho.setter
    def rho(self, value: float):
        assert value > 0.0 , "Doit être supérieur à 0"
        self.__rho = value

    @property
    def solver(self) -> str:
        """Solveur utilisé lors de la résolution"""
        return self.__solver
    
    @solver.setter
    def solver(self, value: str):

        # récupère les solveurs utilisables
        solvers = Solvers()

        if self.problemType != "damage":
            solvers.remove("BoundConstrain")

        if value in solvers:
            self.__solver = value
        else:
            print(Fore.RED + f"Le solveur {value} n'est pas utilisable. Le solveur doit être dans {solvers}"+ Fore.WHITE)

    def Save(self, folder:str):
        "Sauvegarde la simulation et son résumé dans le dossier"    
        # Il faut vider les matrices dans les groupes d'éléments
        self.mesh.ResetMatrices()
    
        # returns current date and time
        dateEtHeure = datetime.now()
        resume = f"Simulation lancée le : {dateEtHeure}"
        nomSimu = "simulation.pickle"
        filename = Folder.Join([folder, nomSimu])
        print(Fore.GREEN + f'\nSauvegarde de :')
        print(Fore.GREEN + f'  - {nomSimu}' + Fore.WHITE)
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # Sauvagarde la simulation
        with open(filename, "wb") as file:
            pickle.dump(self, file)

        # Sauvegarde le résumé de la simulation
        resume += self.Resultats_Resume(False)
        nomResume = "résumé.txt"
        print(Fore.GREEN + f'  - {nomResume} \n' + Fore.WHITE)
        filenameResume = Folder.Join([folder, nomResume])

        with open(filenameResume, 'w', encoding='utf8') as file:
            file.write(resume)

    # TODO Permettre de creer des simulation depuis le formulation variationnelle ?    

    # SOLUTIONS

    @property
    def _results(self) -> list[dict]:
        """Renvoie la liste de dictionnaire qui contient les résultats de chaque itération
        """
        return self.__results

    def __Init_Sols_n(self):
        """On vient initialiser les solutions"""
        self.__dict_u_n = {}
        self.__dict_v_n = {}
        self.__dict_a_n = {}
        for problemType in self.Get_problemTypes():
            taille = self.mesh.Nn * self.Get_nbddl_n(problemType)
            vectInit = np.zeros(taille, dtype=float)
            self.__dict_u_n[problemType] = vectInit
            self.__dict_v_n[problemType] = vectInit
            self.__dict_a_n[problemType] = vectInit    

    def __Check_New_Sol_Values(self, problemType: ModelType, values: np.ndarray):
        """Vérifie que la solution renseignée est de la bonne taille"""
        self.__Check_ProblemTypes(problemType)
        taille = self.mesh.Nn * self.Get_nbddl_n(problemType)
        assert values.shape[0] == taille, f"Doit être de taille {taille}"

    def get_u_n(self, problemType: ModelType) -> np.ndarray:
        """Renvoie la solution associée au problème renseigné"""
        return self.__dict_u_n[problemType].copy()
    
    def set_u_n(self, problemType: ModelType, values: np.ndarray):
        """Renseigne la solution associée au problème renseigné"""
        self.__Check_New_Sol_Values(problemType, values)
        self.__dict_u_n[problemType] = values

    def get_v_n(self, problemType: ModelType) -> np.ndarray:
        """Renvoie la solution en vitesse associée au problème renseigné"""
        return self.__dict_v_n[problemType].copy()

    def set_v_n(self, problemType: ModelType, values: np.ndarray):
        """Renseigne la solution en vitesse associée au problème renseigné"""
        self.__Check_New_Sol_Values(problemType, values)
        self.__dict_v_n[problemType] = values

    def get_a_n(self, problemType: ModelType) -> np.ndarray:
        """Renvoie la solution en accel associée au problème renseigné"""
        return self.__dict_a_n[problemType].copy()

    def set_a_n(self, problemType: ModelType, values: np.ndarray):
        """Renseigne la solution en vitesse associée au problème renseigné"""
        self.__Check_New_Sol_Values(problemType, values)
        self.__dict_a_n[problemType] = values

    # cette méthode est surchargée dans __Simu_PhaseField
    def Get_lb_ub(self, problemType: ModelType) -> tuple[np.ndarray, np.ndarray]:
        """Lower bound et upper bound pour BoundConstrain"""
        return np.array([]), np.array([])

    # Properties
    @property
    def problemType(self) -> ModelType:
        """problème de la simulation"""
        return self.__model.modelType

    @property
    def algo(self) -> AlgoType:
        """algorithme utilisé lors de la résolution"""
        return self.__algo
    
    @property
    def mesh(self) -> Mesh:
        """maillage de la simulation"""
        return self.__mesh
    
    @mesh.setter
    def mesh(self, mesh: Mesh):
        """applique un nouveau maillage"""
        if isinstance(mesh, Mesh):
            # Pour tous les anciens maillages, j'efface les matrices
            listMesh = cast(list[Mesh], self.__listMesh)
            [m.ResetMatrices() for m in listMesh]

            self.__indexMesh += 1
            self.__listMesh.append(mesh)
            self.__mesh = mesh

            # le maillage change, il faut donc reconstruire les matrices
            self.Need_Update()

    @property
    def dim(self) -> int:
        """dimension de la simulation"""
        return self.__dim

    @property
    def use3DBeamModel(self) -> bool:
        return False

    @property
    def useCholesky(self) -> bool:
        """La matrice A de Ax=b est définie symétrique positive"""
        return True

    @property
    def A_isSymetric(self) -> bool:
        """La matrice A de Ax=b est symétrique"""
        return True   

    @property
    def useNumba(self) -> bool:
        """La simulation peut utiliser des fonctions numba"""
        return self.__useNumba
    
    @useNumba.setter
    def useNumba(self, value: bool):
        self.__model.useNumba = value
        self.__useNumba = value

    def __Update_mesh(self, index: int):
        """Met à jour le maillage à l'index renseigné"""
        indexMesh = self._results[index]["indexMesh"]
        self.__mesh = self.__listMesh[indexMesh]
        self.Need_Update()

    @property
    def needUpdate(self) -> bool:
        """La simulation à besoin de reconstruire ces matrices K, C et M"""
        return self.__matricesUpdated

    def Need_Update(self, value=True):
        """Renseigne si la simulation a besoin de reconstruire ces matrices K, C et M"""
        self.__matricesUpdated = value 

    # ================================================ Solveur ================================================

    def Solver_Set_Elliptic_Algorithm(self):
        """Renseigne les propriétés de résolution de l'algorithme\n
        Pour résolution K u = F"""
        self.__algo = AlgoType.elliptic

    def Solver_Set_Parabolic_Algorithm(self, dt=0.1, alpha=1/2):
        """Renseigne les propriétés de résolution de l'algorithme\n
        Pour résolution K u + C v = F

        Parameters
        ----------
        alpha : float, optional
            critère alpha [0 -> Forward Euler, 1 -> Backward Euler, 1/2 -> midpoint], by default 1/2
        dt : float, optional
            incrément temporel, by default 0.1
        """

        self.__algo = AlgoType.parabolic

        # assert alpha >= 0 and alpha <= 1, "alpha doit être compris entre [0, 1]"
        # Est-il possible d’avoir au-delà de 1 ?

        assert dt > 0, "l'incrément temporel doit être > 0"

        self.alpha = alpha
        self.dt = dt

    def Solver_Set_Newton_Raphson_Algorithm(self, betha=1/4, gamma=1/2, dt=0.1):
        """Renseigne les propriétés de résolution de l'algorithme\n
        Pour résolution K u + C v + M a = F

        Parameters
        ----------
        betha : float, optional
            coef betha, by default 1/4
        gamma : float, optional
            coef gamma, by default 1/2
        dt : float, optional
            incrément temporel, by default 0.1
        """

        self.__algo = AlgoType.hyperbolic

        # assert alpha >= 0 and alpha <= 1, "alpha doit être compris entre [0, 1]"
        # Est-il possible d’avoir au-delà de 1 ?

        assert dt > 0, "l'incrément temporel doit être > 0"

        self.betha = betha
        self.gamma = gamma
        self.dt = dt

    def Solve(self) -> np.ndarray:
        """Resolution de la simulation et renvoie la solution
        """

        if self.needUpdate: self.Assemblage()

        self._SolveProblem(self.problemType)
        
        return self.get_u_n(self.problemType)

    def _SolveProblem(self, problemType : ModelType):
        """Resolution du problème.\n
        Il faut privilégier l'utilisation de Solve()"""        
        # ici il faut spécifier le type de problème, car une simulation peut posséder plusieurs Modèles physiques        

        algo = self.__algo

        if len(self.Bc_ddls_Lagrange(problemType)) > 0:
            # Des condtions de lagrange son renseigné
            resolution = ResolutionType.r2
        else:
            resolution = ResolutionType.r1

        # Ancienne solution
        u_n = self.get_u_n(problemType)
        v_n = self.get_v_n(problemType)
        a_n = self.get_a_n(problemType)

        x = _SolveProblem(self, problemType, resolution)
        
        if algo == AlgoType.elliptic:

            u_np1 = x

            self.set_u_n(problemType, u_np1)

        if algo == AlgoType.parabolic:

            u_np1 = x

            alpha = self.alpha
            dt = self.dt

            v_Tild_np1 = u_n + ((1-alpha) * dt * v_n)

            v_np1 = (u_np1 - v_Tild_np1)/(alpha*dt)

            # Nouvelles solutions
            self.set_u_n(problemType, u_np1)
            self.set_v_n(problemType, v_np1)
            
        elif algo == AlgoType.hyperbolic:
            # Formulation en accel

            a_np1 = x

            dt = self.dt
            gamma = self.gamma
            betha = self.betha

            u_Tild_np1 = u_n + (dt * v_n) + dt**2/2 * (1-2*betha) * a_n
            v_Tild_np1 = v_n + (1-gamma) * dt * a_n
            
            u_np1 = u_Tild_np1 + betha * dt**2 * a_np1
            v_np1 = v_Tild_np1 + gamma * dt * a_np1

            # Nouvelles solutions
            self.set_u_n(problemType, u_np1)
            self.set_v_n(problemType, v_np1)
            self.set_a_n(problemType, a_np1)    

    def _Apply_Neumann(self, problemType: ModelType) -> sparse.csr_matrix:
        """Renseigne les conditions limites de neumann en construisant b de A x = b"""
        tic = Tic()
        
        algo = self.algo
        ddls = self.Bc_ddls_Neumann(problemType)
        valeurs_ddls = self.Bc_values_Neumann(problemType)
        taille = self.mesh.Nn * self.Get_nbddl_n(problemType)

        # Dimension supplémentaire lié a l'utilisation des coefs de lagrange
        dimSupl = len(self.Bc_Lagrange)
        if dimSupl > 0:
            dimSupl += len(self.Bc_ddls_Dirichlet(problemType))
            taille += dimSupl
            
        b = sparse.csr_matrix((valeurs_ddls, (ddls,  np.zeros(len(ddls)))), shape = (taille,1))

        K, C, M, F = self.Get_K_C_M_F(problemType)

        u_n = self.get_u_n(problemType)
        v_n = self.get_v_n(problemType)
        a_n = self.get_a_n(problemType)

        b = b + F

        if algo == AlgoType.parabolic:            

            alpha = self.alpha
            dt = self.dt

            v_Tild_np1 = u_n + (1-alpha) * dt * v_n
            v_Tild_np1 = sparse.csr_matrix(v_Tild_np1.reshape(-1, 1))
            
            b = b + C.dot(v_Tild_np1/(alpha*dt))

        elif algo == AlgoType.hyperbolic:
            # Formulation en accel

            if len(self._results) == 0 and (b.max() != 0 or b.min() != 0):
                # initialise l'accel
                ddl_Connues, ddl_Inconnues = self.Bc_ddls_connues_inconnues(problemType)

                bb = b - K.dot(sparse.csr_matrix(u_n.reshape(-1, 1)))
                
                bb -= C.dot(sparse.csr_matrix(v_n.reshape(-1, 1)))

                bbi = bb[ddl_Inconnues]
                Aii = M[ddl_Inconnues, :].tocsc()[:, ddl_Inconnues].tocsr()

                x0 = a_n[ddl_Inconnues]

                ai_n = _Solve_Axb(simu=self, problemType=problemType, A=Aii, b=bbi, x0=x0, lb=[], ub=[], useCholesky=False, A_isSymetric=False, verbosity=self._verbosity)

                a_n[ddl_Inconnues] = ai_n 

                self.set_a_n(problemType, a_n)
            
            a_n = self.get_a_n(problemType)

            dt = self.dt
            gamma = self.gamma
            betha = self.betha

            uTild_np1 = u_n + (dt * v_n) + dt**2/2 * (1-2*betha) * a_n
            vTild_np1 = v_n + (1-gamma) * dt * a_n
            
            b -= K.dot(uTild_np1.reshape(-1,1))
            b -= C.dot(vTild_np1.reshape(-1,1))
            b = sparse.csr_matrix(b)

        tic.Tac("Solver",f"Neumann ({problemType}, {algo})", self._verbosity)

        return b

    def _Apply_Dirichlet(self, problemType: ModelType, b: sparse.csr_matrix, resolution: ResolutionType) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Applique les conditions de dirichlet en construisant Ax de Ax=b"""

        tic = Tic()

        algo = self.algo
        ddls = self.Bc_ddls_Dirichlet(problemType)
        valeurs_ddls = self.BC_values_Dirichlet(problemType)

        K, C, M, F = self.Get_K_C_M_F(problemType)
        
        if algo == AlgoType.elliptic:            
            A = K

        if algo == AlgoType.parabolic:

            alpha = self.alpha
            dt = self.dt
            
            # Resolution en position
            A = K + C/(alpha * dt)
                
            # # Résolution en vitesse
            # A = K * alpha * dt + M

        elif algo == AlgoType.hyperbolic:

            dt = self.dt
            gamma = self.gamma
            betha = self.betha

            # Formulation en accel
            A = M + (K * betha * dt**2)
            A += (gamma * dt * C)

            solDotDot_n = self.get_v_n(problemType)

            valeurs_ddls = solDotDot_n[ddls]        

        A, x = self.__Get_Dirichlet_A_x(problemType, resolution, A, b, valeurs_ddls)

        tic.Tac("Solver",f"Dirichlet ({problemType}, {algo})", self._verbosity)

        return A, x

    def __Get_Dirichlet_A_x(self, problemType: ModelType, resolution: ResolutionType, A: sparse.csr_matrix, b: sparse.csr_matrix, valeurs_ddls: np.ndarray):

        ddls = self.Bc_ddls_Dirichlet(problemType)        
        taille = self.mesh.Nn * self.Get_nbddl_n(problemType)

        if resolution in [ResolutionType.r1, ResolutionType.r2]:
                
            # ici on renvoie la solution avec les ddls connues
            x = sparse.csr_matrix((valeurs_ddls, (ddls,  np.zeros(len(ddls)))), shape = (taille,1), dtype=np.float64)

            # l,c ,v = sparse.find(x)

            return A, x

        elif resolution == ResolutionType.r3:
            # Pénalisation

            A = A.tolil()
            b = b.tolil()
            
            # Pénalisation A
            A[ddls] = 0.0
            A[ddls, ddls] = 1

            # Pénalisation b
            b[ddls] = valeurs_ddls

            # ici on renvoie A pénalisé
            return A.tocsr(), b.tocsr()
    
    # ------------------------------------------- CONDITIONS LIMITES -------------------------------------------
    # Fonctions pour le renseignement des conditions limites de la simulation    


    @staticmethod
    def __Bc_Init_List_BoundaryCondition() -> list[BoundaryCondition]:
        return []

    @staticmethod
    def __Bc_Init_List_LagrangeCondition() -> list[LagrangeCondition]:
        return []
    
    def Bc_Init(self):
        """Initialise les conditions limites de Dirichlet, Neumann et Lagrange"""
        # DIRICHLET
        self.__Bc_Dirichlet = Simu.__Bc_Init_List_BoundaryCondition()
        """Conditions de Dirichlet list(BoundaryCondition)"""
        # NEUMANN
        self.__Bc_Neumann = Simu.__Bc_Init_List_BoundaryCondition()
        """Conditions de Neumann list(BoundaryCondition)"""
        # LAGRANGE
        self.__Bc_Lagrange = Simu.__Bc_Init_List_LagrangeCondition()
        """Conditions de Lagrange list(BoundaryCondition)"""
        self.__Bc_LagrangeAffichage = []
        """Conditions de Lagrange list(BoundaryCondition)"""

    @property
    def Bc_Dirichlet(self) -> list[BoundaryCondition]:
        """Renvoie une copie des conditions de Dirichlet"""
        return self.__Bc_Dirichlet.copy()
    
    @property
    def Bc_Neuman(self) -> list[BoundaryCondition]:
        """Renvoie une copie des conditions de Neumann"""
        return self.__Bc_Neumann.copy()

    @property
    def Bc_Lagrange(self) -> list[LagrangeCondition]:
        """Renvoie une copie des conditions de Lagrange"""
        return self.__Bc_Lagrange.copy()
    
    def _Bc_Add_Lagrange(self, newBc: LagrangeCondition):
        """Ajoute les conditions de Lagrange"""
        assert isinstance(newBc, LagrangeCondition)
        self.__Bc_Lagrange.append(newBc)
    
    @property
    def Bc_LagrangeAffichage(self) -> list[LagrangeCondition]:
        """Renvoie une copie des conditions de Lagrange pour l'affichage"""
        return self.__Bc_LagrangeAffichage.copy()

    def Bc_ddls_Dirichlet(self, problemType=None) -> list[int]:
        """Renvoie les ddls liés aux conditions de Dirichlet"""
        if problemType == None:
            problemType = self.problemType
        return BoundaryCondition.Get_ddls(problemType, self.__Bc_Dirichlet)

    def BC_values_Dirichlet(self, problemType=None) -> list[float]:
        """Renvoie les valeurs ddls liés aux conditions de Dirichlet"""
        if problemType == None:
            problemType = self.problemType
        return BoundaryCondition.Get_values(problemType, self.__Bc_Dirichlet)
    
    def Bc_ddls_Neumann(self, problemType=None) -> list[int]:
        """Renvoie les ddls liés aux conditions de Neumann"""
        if problemType == None:
            problemType = self.problemType
        return BoundaryCondition.Get_ddls(problemType, self.__Bc_Neumann)
    
    def Bc_values_Neumann(self, problemType=None) -> list[float]:
        """Renvoie les valeurs ddls liés aux conditions de Neumann"""
        if problemType == None:
            problemType = self.problemType
        return BoundaryCondition.Get_values(problemType, self.__Bc_Neumann)

    def Bc_ddls_Lagrange(self, problemType=None) -> list[int]:
        """Renvoie les ddls liés aux conditions de Lagrange"""
        if problemType == None:
            problemType = self.problemType
        return BoundaryCondition.Get_ddls(problemType, self.__Bc_Lagrange)
    
    def Bc_values_Lagrange(self, problemType=None) -> list[float]:
        """Renvoie les valeurs ddls liés aux conditions de Lagrange"""
        if problemType == None:
            problemType = self.problemType
        return BoundaryCondition.Get_values(problemType, self.__Bc_Lagrange)

    def Bc_ddls_connues_inconnues(self, problemType: ModelType):
        """Récupère les ddl Connues et Inconnues
        Returns:
            list(int), list(int): ddl_Connues, ddl_Inconnues
        """
        tic = Tic()

        # Construit les ddls connues
        ddls_Connues = []

        ddls_Connues = self.Bc_ddls_Dirichlet(problemType)
        unique_ddl_Connues = np.unique(ddls_Connues)

        # Construit les ddls inconnues

        taille = self.mesh.Nn * self.Get_nbddl_n(problemType)

        ddls_Inconnues = list(range(taille))

        ddls_Inconnues = list(set(ddls_Inconnues) - set(unique_ddl_Connues))        

        ddls_Connues = np.asarray(ddls_Connues)                        
        ddls_Inconnues = np.array(ddls_Inconnues)
        
        verifTaille = unique_ddl_Connues.shape[0] + ddls_Inconnues.shape[0]
        assert verifTaille == taille, f"Problème dans les conditions ddls_Connues + ddls_Inconnues - taille = {verifTaille-taille}"

        tic.Tac("Solver",f"Construit ddls ({problemType})", self._verbosity)

        return ddls_Connues, ddls_Inconnues    

    def __Bc_evalue(self, coordo: np.ndarray, valeurs, option="noeuds") -> np.ndarray:
        """evalue les valeurs aux noeuds ou aux points de gauss"""
        
        assert option in ["noeuds","gauss"], f"Doit être dans ['noeuds','gauss']"
        if option == "noeuds":
            valeurs_eval = np.zeros(coordo.shape[0])
        elif option == "gauss":
            valeurs_eval = np.zeros((coordo.shape[0],coordo.shape[1]))
        
        if isinstance(valeurs, LambdaType):
            # Evalue la fonction aux coordonnées
            try:
                if option == "noeuds":
                    valeurs_eval[:] = valeurs(coordo[:,0], coordo[:,1], coordo[:,2])
                elif option == "gauss":
                    valeurs_eval[:,:] = valeurs(coordo[:,:,0], coordo[:,:,1], coordo[:,:,2])
            except:
                raise Exception("Doit fournir une fonction lambda de la forme\n lambda x,y,z, : f(x,y,z)")
        else:            
            if option == "noeuds":
                valeurs_eval[:] = valeurs
            elif option == "gauss":
                valeurs_eval[:,:] = valeurs

        return valeurs_eval
    
    def add_dirichlet(self, noeuds: np.ndarray, valeurs: list, directions: list, problemType=None, description=""):
        """Ajoute des conditions de dirichlet

        Parameters
        ----------
        noeuds : np.ndarray
            noeuds
        valeurs : list
            liste de valeurs qui peuvent contenir des float ou des fonctions lambda à évaluer
            ex = [10, lambda x,y,z : 10*x - 20*y + x*z] \n
            Les fonctions utilisent les coordonnées x, y et z des noeuds renseignés
            Attention, les fonctions à évaluer doivent obligatoirement prendre 3 paramètres d'entrée dans l'ordre x, y, z que le problème soit 1D, 2D ou 3D
        directions : list
            directions ou on va appliquer les valeurs
        problemType : ModelType, optional
            type du problème, si non renseingé, on prend le le problème de base du problem
        description : str, optional
            Description de la condition, by default ""
        """

        if len(valeurs) == 0 or len(valeurs) != len(directions): return        

        if problemType == None:
            problemType = self.problemType

        self.__Check_ProblemTypes(problemType)
        
        assert len(noeuds) > 0, "Liste de noeuds vides"
        noeuds = np.asarray(noeuds)

        Nn = noeuds.shape[0]
        coordo = self.mesh.coordo
        coordo_n = coordo[noeuds]

        # initialise le vecteur de valeurs pour chaque noeuds
        valeurs_ddl_dir = np.zeros((Nn, len(directions)))

        for d, dir in enumerate(directions):
            eval_n = self.__Bc_evalue(coordo_n, valeurs[d], option="noeuds")
            valeurs_ddl_dir[:,d] = eval_n.reshape(-1)
        
        valeurs_ddls = valeurs_ddl_dir.reshape(-1)

        nbddl_n = self.Get_nbddl_n(problemType)

        ddls = BoundaryCondition.Get_ddls_noeuds(nbddl_n, problemType, noeuds, directions)

        self.__Bc_Add_Dirichlet(problemType, noeuds, valeurs_ddls, ddls, directions, description)

    def add_neumann(self, noeuds: np.ndarray, valeurs: list, directions: list, problemType=None, description=""):
        """Force ponctuelle

        Parameters
        ----------
        noeuds : np.ndarray
            noeuds
        valeurs : list
            liste de valeurs qui peuvent contenir des float ou des fonctions lambda à évaluer
            ex = [10, lambda x,y,z : 10*x - 20*y + x*z] \n
            Les fonctions utilisent les coordonnées x, y et z des noeuds renseignés
            Attention, les fonctions à évaluer doivent obligatoirement prendre 3 paramètres d'entrée dans l'ordre x, y, z que le problème soit 1D, 2D ou 3D
        directions : list
            directions ou on va appliquer les valeurs
        problemType : ModelType, optional
            type du problème, si non renseingé, on prend le le problème de base du problem
        description : str, optional
            Description de la condition, by default ""
        """
        
        if len(valeurs) == 0 or len(valeurs) != len(directions): return

        if problemType == None:
            problemType = self.problemType

        self.__Check_ProblemTypes(problemType)

        valeurs_ddls, ddls = self.__Bc_pointLoad(problemType, noeuds, valeurs, directions)

        self.__Bc_Add_Neumann(problemType, noeuds, valeurs_ddls, ddls, directions, description)
        
    def add_lineLoad(self, noeuds: np.ndarray, valeurs: list, directions: list, problemType=None, description=""):
        """Applique une force linéique

        Parameters
        ----------
        noeuds : np.ndarray
            noeuds
        valeurs : list
            liste de valeurs qui peuvent contenir des float ou des fonctions lambda à évaluer
            ex = [10, lambda x,y,z : 10*x - 20*y + x*z] \n
            les fonctions utilisent les coordonnées x, y et z des points d'intégrations
            Attention, les fonctions à évaluer doivent obligatoirement prendre 3 paramètres d'entrée dans l'ordre x, y, z que le problème soit 1D, 2D ou 3D
        directions : list
            directions ou on va appliquer les valeurs
        problemType : ModelType, optional
            type du problème, si non renseingé, on prend le le problème de base du problem
        description : str, optional
            Description de la condition, by default ""
        """

        if len(valeurs) == 0 or len(valeurs) != len(directions): return

        if problemType == None:
            problemType = self.problemType

        self.__Check_ProblemTypes(problemType)

        valeurs_ddls, ddls = self.__Bc_lineLoad(problemType, noeuds, valeurs, directions)

        self.__Bc_Add_Neumann(problemType, noeuds, valeurs_ddls, ddls, directions, description)

    def add_surfLoad(self, noeuds: np.ndarray, valeurs: list, directions: list, problemType=None, description=""):
        """Applique une force surfacique
        
        Parameters
        ----------
        noeuds : np.ndarray
            noeuds
        valeurs : list
            liste de valeurs qui peuvent contenir des float ou des fonctions lambda à évaluer
            ex = [10, lambda x,y,z : 10*x - 20*y + x*z] \n
            les fonctions utilisent les coordonnées x, y et z des points d'intégrations
            Attention, les fonctions à évaluer doivent obligatoirement prendre 3 paramètres d'entrée dans l'ordre x, y, z que le problème soit 1D, 2D ou 3D
        directions : list
            directions ou on va appliquer les valeurs
        problemType : ModelType, optional
            type du problème, si non renseingé, on prend le le problème de base du problem
        description : str, optional
            Description de la condition, by default ""
        """

        if len(valeurs) == 0 or len(valeurs) != len(directions): return

        if problemType == None:
            problemType = self.problemType

        self.__Check_ProblemTypes(problemType)
            
        if self.__dim == 2:
            valeurs_ddls, ddls = self.__Bc_lineLoad(problemType, noeuds, valeurs, directions)
            # multiplie par l'epaisseur
            valeurs_ddls *= self.model.epaisseur
        elif self.__dim == 3:
            valeurs_ddls, ddls = self.__Bc_surfload(problemType, noeuds, valeurs, directions)

        self.__Bc_Add_Neumann(problemType, noeuds, valeurs_ddls, ddls, directions, description)

    def add_volumeLoad(self, noeuds: np.ndarray, valeurs: list, directions: list, problemType=None, description=""):
        """Applique une force volumique
        
        Parameters
        ----------
        noeuds : np.ndarray
            noeuds
        valeurs : list
            liste de valeurs qui peuvent contenir des float ou des fonctions lambda à évaluer
            ex = [10, lambda x,y,z : 10*x - 20*y + x*z] \n
            les fonctions utilisent les coordonnées x, y et z des points d'intégrations
            Attention, les fonctions à évaluer doivent obligatoirement prendre 3 paramètres d'entrée dans l'ordre x, y, z que le problème soit 1D, 2D ou 3D
        directions : list
            directions ou on va appliquer les valeurs
        problemType : ModelType, optional
            type du problème, si non renseingé, on prend le le problème de base du problem
        description : str, optional
            Description de la condition, by default ""
        """
        
        if len(valeurs) == 0 or len(valeurs) != len(directions): return

        if problemType == None:
            problemType = self.problemType

        self.__Check_ProblemTypes(problemType)
        
        if self.__dim == 2:
            valeurs_ddls, ddls = self.__Bc_surfload(problemType, noeuds, valeurs, directions)
            # multiplie par l'epaisseur
            valeurs_ddls = valeurs_ddls*self.model.epaisseur
        elif self.__dim == 3:
            valeurs_ddls, ddls = self.__Bc_volumeload(problemType, noeuds, valeurs, directions)

        self.__Bc_Add_Neumann(problemType, noeuds, valeurs_ddls, ddls, directions, description)
    
    def __Bc_pointLoad(self, problemType : ModelType, noeuds: np.ndarray, valeurs: list, directions: list):
        """Applique une force linéique\n
        Renvoie valeurs_ddls, ddls"""

        Nn = noeuds.shape[0]
        coordo = self.mesh.coordo
        coordo_n = coordo[noeuds]

        # initialise le vecteur de valeurs pour chaque noeuds
        valeurs_ddl_dir = np.zeros((Nn, len(directions)))

        for d, dir in enumerate(directions):
            eval_n = self.__Bc_evalue(coordo_n, valeurs[d], option="noeuds")
            if problemType == ModelType.beam:
                eval_n /= len(noeuds)
            valeurs_ddl_dir[:,d] = eval_n.reshape(-1)
        
        valeurs_ddls = valeurs_ddl_dir.reshape(-1)

        nbddl_n = self.Get_nbddl_n(problemType)

        ddls = BoundaryCondition.Get_ddls_noeuds(nbddl_n, problemType, noeuds, directions)

        return valeurs_ddls, ddls

    def __Bc_IntegrationDim(self, dim: int, problemType : ModelType, noeuds: np.ndarray, valeurs: list, directions: list):
        """Intégration des valeurs sur les éléments"""

        valeurs_ddls=np.array([])
        ddls=np.array([], dtype=int)

        listGroupElemDim = self.mesh.Get_list_groupElem(dim)

        if len(listGroupElemDim) > 1:
            exclusivement=True
        else:
            exclusivement=True

        nbddl_n = self.Get_nbddl_n(problemType)

        # Récupération des matrices pour le calcul
        for groupElem in listGroupElemDim:

            # Récupère les éléments qui utilisent exclusivement les noeuds
            elements = groupElem.Get_Elements_Nodes(noeuds, exclusivement=exclusivement)
            if elements.shape[0] == 0: continue
            connect_e = groupElem.connect_e[elements]
            Ne = elements.shape[0]
            
            # récupère les coordonnées des points de gauss dans le cas ou on a besoin dévaluer la fonction
            matriceType = MatriceType.masse
            coordo_e_p = groupElem.Get_coordo_e_p(matriceType,elements)
            nPg = coordo_e_p.shape[1]

            N_pg = groupElem.Get_N_pg(matriceType)

            # objets d'integration
            jacobien_e_pg = groupElem.Get_jacobien_e_pg(matriceType)[elements]
            gauss = groupElem.Get_gauss(matriceType)
            poid_pg = gauss.poids

            # initialise le vecteur de valeurs pour chaque element et chaque pts de gauss
            valeurs_ddl_dir = np.zeros((Ne*groupElem.nPe, len(directions)))

            # Intègre sur chaque direction
            for d, dir in enumerate(directions):
                eval_e_p = self.__Bc_evalue(coordo_e_p, valeurs[d], option="gauss")
                valeurs_e_p = np.einsum('ep,p,ep,pij->epij', jacobien_e_pg, poid_pg, eval_e_p, N_pg, optimize='optimal')
                valeurs_e = np.sum(valeurs_e_p, axis=1)
                valeurs_ddl_dir[:,d] = valeurs_e.reshape(-1)

            new_valeurs_ddls = valeurs_ddl_dir.reshape(-1)
            valeurs_ddls = np.append(valeurs_ddls, new_valeurs_ddls)
            
            new_ddls = BoundaryCondition.Get_ddls_connect(nbddl_n, problemType, connect_e, directions)
            ddls = np.append(ddls, new_ddls)

        return valeurs_ddls, ddls

    def __Bc_lineLoad(self, problemType : ModelType, noeuds: np.ndarray, valeurs: list, directions: list):
        """Applique une force linéique\n
        Renvoie valeurs_ddls, ddls"""
        
        self.Check_Directions(problemType, directions)

        valeurs_ddls, ddls = self.__Bc_IntegrationDim(dim=1, problemType=problemType, noeuds=noeuds, valeurs=valeurs, directions=directions)

        return valeurs_ddls, ddls
    
    def __Bc_surfload(self, problemType : ModelType, noeuds: np.ndarray, valeurs: list, directions: list):
        """Applique une force surfacique\n
        Renvoie valeurs_ddls, ddls"""
        
        self.Check_Directions(problemType, directions)

        valeurs_ddls, ddls = self.__Bc_IntegrationDim(dim=2, problemType=problemType, noeuds=noeuds, valeurs=valeurs, directions=directions)

        return valeurs_ddls, ddls

    def __Bc_volumeload(self, problemType : ModelType, noeuds: np.ndarray, valeurs: list, directions: list):
        """Applique une force surfacique\n
        Renvoie valeurs_ddls, ddls"""
        
        self.Check_Directions(problemType, directions)

        valeurs_ddls, ddls = self.__Bc_IntegrationDim(dim=3, problemType=problemType, noeuds=noeuds, valeurs=valeurs, directions=directions)

        return valeurs_ddls, ddls
    
    def __Bc_Add_Neumann(self, problemType : ModelType, noeuds: np.ndarray, valeurs_ddls: np.ndarray, ddls: np.ndarray, directions: list, description=""):
        """Ajoute les conditions de Neumann"""

        tic = Tic()

        self.Check_Directions(problemType, directions)

        if problemType in [ModelType.damage,ModelType.thermal]:
            
            new_Bc = BoundaryCondition(problemType, noeuds, ddls, directions, valeurs_ddls, f'Neumann {description}')

        elif problemType in [ModelType.displacement,ModelType.beam]:

            # Si un ddl est déja connue dans les conditions de dirichlet
            # alors on enlève la valeur et le ddl associé
            
            ddl_Dirchlet = self.Bc_ddls_Dirichlet(problemType)
            valeursTri_ddls = list(valeurs_ddls)
            ddlsTri = list(ddls.copy())

            def tri(d, ddl):
                if ddl in ddl_Dirchlet:
                    ddlsTri.remove(ddl)
                    valeursTri_ddls.remove(valeurs_ddls[d])

            [tri(d, ddl) for d, ddl in enumerate(ddls)]

            valeursTri_ddls = np.array(valeursTri_ddls)
            ddlsTri = np.array(ddlsTri)

            new_Bc = BoundaryCondition(problemType, noeuds, ddlsTri, directions, valeursTri_ddls, f'Neumann {description}')

        self.__Bc_Neumann.append(new_Bc)

        tic.Tac("Boundary Conditions","Condition Neumann", self._verbosity)   
     
    def __Bc_Add_Dirichlet(self, problemType : ModelType, noeuds: np.ndarray, valeurs_ddls: np.ndarray, ddls: np.ndarray, directions: list, description=""):
        """Ajoute les conditions de Dirichlet"""

        tic = Tic()

        self.__Check_ProblemTypes(problemType)

        if problemType in [ModelType.damage,ModelType.thermal]:
            
            new_Bc = BoundaryCondition(problemType, noeuds, ddls, directions, valeurs_ddls, f'Dirichlet {description}')

        elif problemType in [ModelType.displacement,ModelType.beam]:

            new_Bc = BoundaryCondition(problemType, noeuds, ddls, directions, valeurs_ddls, f'Dirichlet {description}')

            # Verifie si les ddls de Neumann ne coincidenet pas avec dirichlet
            ddl_Neumann = self.Bc_ddls_Neumann(problemType)
            for d in ddls: 
                assert d not in ddl_Neumann, "On ne peut pas appliquer conditions dirchlet et neumann aux memes ddls"

        self.__Bc_Dirichlet.append(new_Bc)

        tic.Tac("Boundary Conditions","Condition Dirichlet", self._verbosity)

    # ------------------------------------------- LIAISONS ------------------------------------------- 
    # Fonctions pour créer des liaisons entre degré de liberté    

    def Bc_Add_LagrangeAffichage(self,noeuds: np.ndarray, directions: list[str], description: str):
        
        # Ajoute une condition pour l'affichage
        nbddl = self.Get_nbddl_n(self.problemType)
        
        # Prend le premier noeuds de la liaison
        noeuds1 = np.array([noeuds[0]])

        ddls = BoundaryCondition.Get_ddls_noeuds(param=nbddl,  problemType=ModelType.beam, noeuds=noeuds1, directions=directions)
        valeurs_ddls =  np.array([0]*len(ddls))

        new_Bc = BoundaryCondition(ModelType.beam, noeuds1, ddls, directions, valeurs_ddls, description)
        self.__Bc_LagrangeAffichage.append(new_Bc)
    
    # ------------------------------------------- POST TRAITEMENT ------------------------------------------- 

    def Resultats_Check_Options_disponibles(self, resultat: str) -> bool:
        """Verification que le résultat est bien calculable
        """
        listResultats = self.Get_Resultats_disponibles()
        if resultat in listResultats:
            return True
        else:
            print(f"\nPour un problème ({self.problemType}) l'option doit etre dans : \n {listResultats}")
            return False

    def Resultats_Set_Resume_Iteration(self):
        """Renseigne le resumé de l'itération"""
        pass

    def Resultats_Get_Resume_Iteration(self) -> str:
        """Resumé de l'itération"""
        return "Non renseigné"

    def Resultats_Set_Resume_Chargement(self):
        """Renseigne de chargement de la simulation"""
        pass

    def Resultats_Get_Resume_Chargement(self) -> str:
        """Résumé de chargement de la simulation"""
        return "Non renseigné"
        
    def Resultats_Resume(self, verbosity=True) -> str:

        import Affichage

        resume = Affichage.NouvelleSection("Maillage", False)
        resume += self.mesh.Resume(False)

        resume += Affichage.NouvelleSection("Materiau", False)
        resume += '\n' + self.model.resume

        resume += Affichage.NouvelleSection("Chargement", False)
        resume += '\n' + self.Resultats_Get_Resume_Chargement()

        resume += Affichage.NouvelleSection("Résultat", False)
        resume += '\n' + self.Resultats_Get_Resume_Iteration()

        resume += Affichage.NouvelleSection("TicTac", False)
        resume += Tic.Resume(False)

        if verbosity:
            print(resume)

        return resume    
    
    def Resultats_InterpolationAuxNoeuds(self, resultat_e: np.ndarray):
        """Pour chaque noeuds on récupère les valeurs des élements autour de lui pour on en fait la moyenne
        """
        return Simu.Resultats_InterpolationAuxNoeuds(self.__mesh, resultat_e=resultat_e)
        
    @staticmethod
    def Resultats_InterpolationAuxNoeuds(mesh: Mesh, resultat_e: np.ndarray):
        """Pour chaque noeuds on récupère les valeurs des élements autour de lui pour on en fait la moyenne
        """

        tic = Tic()

        Ne = mesh.Ne
        Nn = mesh.Nn

        if len(resultat_e.shape) == 1:
            resultat_e = resultat_e.reshape(Ne,1)
            isDim1 = True
        else:
            isDim1 = False
        
        Ncolonnes = resultat_e.shape[1]

        resultat_n = np.zeros((Nn, Ncolonnes))

        for c in range(Ncolonnes):

            valeurs_e = resultat_e[:, c]

            connect_n_e = mesh.Get_connect_n_e()
            nombreApparition = np.array(np.sum(connect_n_e, axis=1)).reshape(mesh.Nn,1)
            valeurs_n_e = connect_n_e.dot(valeurs_e.reshape(mesh.Ne,1))
            valeurs_n = valeurs_n_e/nombreApparition

            resultat_n[:,c] = valeurs_n.reshape(-1)

        tic.Tac("Post Traitement","Interpolation aux noeuds", False)

        if isDim1:
            return resultat_n.reshape(-1)
        else:
            return resultat_n

###################################################################################################

class Simu_Displacement(Simu):

    def __init__(self, mesh: Mesh, model: Displacement_Model, verbosity=False, useNumba=True):
        """Creation d'une simulation de déplacement"""
        assert model.modelType == ModelType.displacement, "Le matériau doit être de type displacement"
        super().__init__(mesh, model, verbosity, useNumba)

        # init
        self.Set_Rayleigh_Damping_Coefs()
        self.Solver_Set_Elliptic_Algorithm()

    def Get_Resultats_disponibles(self) -> list[str]:

        options = []
        dim = self.dim
        
        if dim == 2:
            options.extend(["ux", "uy", "uz", "amplitude", "displacement", "matrice_displacement"])
            options.extend(["vx", "vy", "speed", "amplitudeSpeed"])
            options.extend(["ax", "ay", "accel", "amplitudeAccel"])
            options.extend(["Sxx", "Syy", "Sxy", "Svm","Stress"])
            options.extend(["Exx", "Eyy", "Exy", "Evm","Strain"])

        elif dim == 3:
            options.extend(["ux", "uy", "uz","amplitude","displacement", "matrice_displacement"])
            options.extend(["vx", "vy", "vz", "speed", "amplitudeSpeed"])
            options.extend(["ax", "ay", "az", "accel", "amplitudeAccel"])
            options.extend(["Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy", "Svm","Stress"])
            options.extend(["Exx", "Eyy", "Ezz", "Eyz", "Exz", "Exy", "Evm","Strain"])
        
        options.extend(["Wdef","Psi_Elas","energy"])

        return options

    def Paraview_nodesField_elementsField(self, details=False) -> tuple[list[str], list[str]]:
        if details:
            nodesField = ["matrice_displacement", "speed", "accel"]
            elementsField = ["Stress", "Strain"]
        else:
            nodesField = ["matrice_displacement", "speed", "accel"]
            elementsField = ["Stress"]                        
        return nodesField, elementsField
    
    def Get_Directions(self, problemType=None) -> list[str]:
        dict_dim_directions = {
            2 : ["x", "y"],
            3 : ["x", "y", "z"]
        }
        return dict_dim_directions[self.dim]
    
    def Get_problemTypes(self) -> list[ModelType]:
        return [ModelType.displacement]
        
    def Get_nbddl_n(self, problemType=None) -> int:
        return self.dim

    @property
    def comportement(self) -> Displacement_Model:
        """Modèle de déplacement élastique de la simulation"""
        return self.model

    @property
    def displacement(self) -> np.ndarray:
        """Copie du champ vectoriel de déplacement"""
        return self.get_u_n(self.problemType)

    @property
    def speed(self) -> np.ndarray:
        """Copie du champ vectoriel de vitesse"""
        return self.get_v_n(self.problemType)

    @property
    def accel(self) -> np.ndarray:
        """Copie du champ vectoriel d'accélération"""
        return self.get_a_n(self.problemType)

    def __ConstruitMatElem_Dep(self):
        """Construit les matrices de rigidités élementaires pour le problème en déplacement
        """

        useNumba = self.useNumba
        # useNumba = False

        matriceType=MatriceType.rigi

        # Recupère les matrices pour travailler
        mesh = self.mesh
        jacobien_e_pg = mesh.Get_jacobien_e_pg(matriceType)
        poid_pg = mesh.Get_poid_pg(matriceType)
        N_vecteur_e_pg = mesh.Get_N_vecteur_pg(matriceType)
        rho = self.rho
        
        B_dep_e_pg = mesh.Get_B_dep_e_pg(matriceType)
        leftDepPart = mesh.Get_leftDepPart(matriceType) # -> jacobien_e_pg * poid_pg * B_dep_e_pg'

        comportement = self.comportement

        tic = Tic()

        # Ici on le matériau est homogène
        matC = comportement.C

        # Matrices rigi
        if useNumba:
            # Plus rapide
            Ku_e = CalcNumba.epij_jk_epkl_to_eil(leftDepPart, matC, B_dep_e_pg)
        else:
            # Ku_e = np.einsum('ep,p,epki,kl,eplj->eij', jacobien_e_pg, poid_pg, B_dep_e_pg, matC, B_dep_e_pg, optimize='optimal')
            Ku_e = np.einsum('epij,jk,epkl->eil', leftDepPart, matC, B_dep_e_pg, optimize='optimal')           
        
        # Matrices masse
        Mu_e = np.einsum('ep,p,pki,,pkj->eij', jacobien_e_pg, poid_pg, N_vecteur_e_pg, rho, N_vecteur_e_pg, optimize="optimal")            

        if self.dim == 2:
            epaisseur = self.comportement.epaisseur
            Ku_e *= epaisseur
            Mu_e *= epaisseur
        
        tic.Tac("Matrices","Construction Ku_e et Mu_e", self._verbosity)

        return Ku_e, Mu_e

    def Get_K_C_M_F(self, problemType=None) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        if problemType==None:
            problemType = self.problemType
        if self.needUpdate: self.Assemblage()
        return self.__Ku.copy(), self.Get_Rayleigh_Damping(), self.__Mu.copy(), self.__Fu.copy()
 
    def Assemblage(self):

        if self.needUpdate:

            # Data
            mesh = self.mesh        
            taille = mesh.Nn*self.dim

            # Dimension supplémentaire lié a l'utilisation des coefs de lagrange
            dimSupl = len(self.Bc_Lagrange)
            if dimSupl > 0:
                dimSupl += len(self.Bc_ddls_Dirichlet(ModelType.displacement))
                taille += dimSupl

            # Construit dict_Ku_e
            Ku_e, Mu_e = self.__ConstruitMatElem_Dep()            

            # Prépare assemblage
            lignesVector_e = mesh.lignesVector_e
            colonnesVector_e = mesh.colonnesVector_e
            
            tic = Tic()

            # Assemblage
            self.__Ku = sparse.csr_matrix((Ku_e.reshape(-1), (lignesVector_e.reshape(-1), colonnesVector_e.reshape(-1))), shape=(taille, taille))
            """Matrice Kglob pour le problème en déplacement (Nn*dim, Nn*dim)"""

            # Ici j'initialise Fu calr il faudrait calculer les forces volumiques dans __ConstruitMatElem_Dep !!!
            self.__Fu = sparse.csr_matrix((taille, 1))
            """Vecteur Fglob pour le problème en déplacement (Nn*dim, 1)"""

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.spy(self.__Ku)
            # plt.show()

            self.__Mu = sparse.csr_matrix((Mu_e.reshape(-1), (lignesVector_e.reshape(-1), colonnesVector_e.reshape(-1))), shape=(taille, taille))
            """Matrice Mglob pour le problème en déplacement (Nn*dim, Nn*dim)"""

            tic.Tac("Matrices","Assemblage Ku, Mu et Fu", self._verbosity)

            self.Need_Update()

    def Set_Rayleigh_Damping_Coefs(self, coefM=0.0, coefK=0.0):
        self.__coefM = coefM
        self.__coefK = coefK

    def Get_Rayleigh_Damping(self) -> sparse.csr_matrix:
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

    
    def Save_Iteration(self):
        
        iter = super().Save_Iteration()

        iter['displacement'] = self.displacement
        if self.algo == AlgoType.parabolic:
            iter["speed"] = self.speed
            iter["accel"] = self.accel

        self._results.append(iter)
    
    def Update_iter(self, index= -1):
        
        results = super().Update_iter(index)

        if results == None: return

        displacementType = ModelType.displacement

        self.set_u_n(displacementType, results[displacementType])

        if self.algo == AlgoType.hyperbolic and "speed" in results and "accel" in results:
            self.set_v_n(displacementType, results["speed"])
            self.set_a_n(displacementType, results["accel"])
        else:
            initZeros = np.zeros_like(self.displacement)
            self.set_v_n(displacementType, initZeros)
            self.set_a_n(displacementType, initZeros)

    def Get_Resultat(self, option: str, nodeValues=True, iter=None):
        
        dim = self.dim
        Ne = self.mesh.Ne
        Nn = self.mesh.Nn
        
        if not self.Resultats_Check_Options_disponibles(option): return None

        if iter != None:
            self.Update_iter(iter)

        if option in ["Wdef","Psi_Elas"]:
            return self.__Calc_Psi_Elas()

        if option == "energy":
            psi_e = self.__Calc_Psi_Elas(returnScalar=False)

            if nodeValues:
                return self.Resultats_InterpolationAuxNoeuds(psi_e)
            else:
                return psi_e

        if option == "displacement":
            return self.displacement

        if option == "speed":
            return self.speed
        
        if option == "accel":
            return self.accel

        if option == 'matrice_displacement':
            return self.Resultats_matrice_displacement()

        displacement = self.displacement

        coef = self.comportement.coef

        # Deformation et contraintes pour chaque element et chaque points de gauss        
        Epsilon_e_pg = self.__Calc_Epsilon_e_pg(displacement)
        Sigma_e_pg = self.__Calc_Sigma_e_pg(Epsilon_e_pg)

        # Moyenne sur l'élement
        Epsilon_e = np.mean(Epsilon_e_pg, axis=1)
        Sigma_e = np.mean(Sigma_e_pg, axis=1)
        
        if ("S" in option or "E" in option) and option != "amplitudeSpeed":

            if "S" in option and option != "Strain":
                val_e = Sigma_e
            
            elif "E" in option or option == "Strain":
                val_e = Epsilon_e

            else:
                raise Exception("Mauvaise option")

            if dim == 2:

                val_xx_e = val_e[:,0]
                val_yy_e = val_e[:,1]
                val_xy_e = val_e[:,2]/coef
                
                # TODO Ici il faudrait calculer Szz si deformation plane
                
                val_vm_e = np.sqrt(val_xx_e**2+val_yy_e**2-val_xx_e*val_yy_e+3*val_xy_e**2)

                if "xx" in option:
                    resultat_e = val_xx_e
                elif "yy" in option:
                    resultat_e = val_yy_e
                elif "xy" in option:
                    resultat_e = val_xy_e
                elif "vm" in option:
                    resultat_e = val_vm_e

            elif dim == 3:

                val_xx_e = val_e[:,0]
                val_yy_e = val_e[:,1]
                val_zz_e = val_e[:,2]
                val_yz_e = val_e[:,3]/coef
                val_xz_e = val_e[:,4]/coef
                val_xy_e = val_e[:,5]/coef               

                val_vm_e = np.sqrt(((val_xx_e-val_yy_e)**2+(val_yy_e-val_zz_e)**2+(val_zz_e-val_xx_e)**2+6*(val_xy_e**2+val_yz_e**2+val_xz_e**2))/2)

                if "xx" in option:
                    resultat_e = val_xx_e
                elif "yy" in option:
                    resultat_e = val_yy_e
                elif "zz" in option:
                    resultat_e = val_zz_e
                elif "yz" in option:
                    resultat_e = val_yz_e
                elif "xz" in option:
                    resultat_e = val_xz_e
                elif "xy" in option:
                    resultat_e = val_xy_e
                elif "vm" in option:
                    resultat_e = val_vm_e

            if option in ["Stress","Strain"]:
                resultat_e = np.append(val_e, val_vm_e.reshape((Ne,1)), axis=1)

            if nodeValues:
                resultat_n = self.Resultats_InterpolationAuxNoeuds(resultat_e)
                return resultat_n
            else:
                return resultat_e
        
        else:

            Nn = self.mesh.Nn

            if option in ["ux", "uy", "uz", "amplitude"]:
                resultat_ddl = self.displacement
            elif option in ["vx", "vy", "vz", "amplitudeSpeed"]:
                resultat_ddl = self.speed
            elif option in ["ax", "ay", "az", "amplitudeAccel"]:
                resultat_ddl = self.accel

            resultat_ddl = resultat_ddl.reshape(Nn, -1)

            index = self.__indexResulat(option)
            
            if nodeValues:

                if "amplitude" in option:
                    return np.sqrt(np.sum(resultat_ddl**2,axis=1))
                else:
                    if len(resultat_ddl.shape) > 1:
                        return resultat_ddl[:,index]
                    else:
                        return resultat_ddl.reshape(-1)
                        
            else:

                # recupere pour chaque element les valeurs de ses noeuds
                resultat_e_n = self.mesh.Localises_sol_e(resultat_ddl)
                resultat_e = resultat_e_n.mean(axis=1)

                if "amplitude" in option:
                    return np.sqrt(np.sum(resultat_e**2, axis=1))
                elif option in ["speed", "accel"]:
                    return resultat_e.reshape(-1)
                else:
                    if len(resultat_e.shape) > 1:
                        return resultat_e[:,index]
                    else:
                        return resultat_e.reshape(-1)

    def __Calc_Psi_Elas(self, returnScalar=True) -> float:
        """Calcul de l'energie de deformation cinématiquement admissible endommagé ou non
        Calcul de Wdef = 1/2 int_Omega jacobien * poid * Sig : Eps dOmega x epaisseur"""

        tic = Tic()

        sol_u  = self.displacement

        matriceType = MatriceType.rigi
        Epsilon_e_pg = self.__Calc_Epsilon_e_pg(sol_u, matriceType)
        jacobien_e_pg = self.mesh.Get_jacobien_e_pg(matriceType)
        poid_pg = self.mesh.Get_poid_pg(matriceType)

        if self.dim == 2:
            ep = self.comportement.epaisseur
        else:
            ep = 1

        Sigma_e_pg = self.__Calc_Sigma_e_pg(Epsilon_e_pg, matriceType)

        if returnScalar:

            Wdef = 1/2 * np.einsum(',ep,p,epi,epi->', ep, jacobien_e_pg, poid_pg, Sigma_e_pg, Epsilon_e_pg, optimize='optimal')        

            Wdef = float(Wdef)

        else:

            Wdef = 1/2 * np.einsum(',ep,p,epi,epi->e', ep, jacobien_e_pg, poid_pg, Sigma_e_pg, Epsilon_e_pg, optimize='optimal')        


        tic.Tac("PostTraitement","Calcul Psi Elas",False)
        
        return Wdef

    def __Calc_Epsilon_e_pg(self, sol: np.ndarray, matriceType=MatriceType.rigi):
        """Construit epsilon pour chaque element et chaque points de gauss

        Parameters
        ----------
        u : np.ndarray
            Vecteur des déplacements

        Returns
        -------
        np.ndarray
            Deformations stockées aux éléments et points de gauss (Ne,pg,(3 ou 6))
        """

        # useNumba = self.useNumba
        useNumba = False
        
        tic = Tic()

        u_e = self.mesh.Localises_sol_e(sol)
        B_dep_e_pg = self.mesh.Get_B_dep_e_pg(matriceType)
        if useNumba: # Moins rapide
            Epsilon_e_pg = CalcNumba.epij_ej_to_epi(B_dep_e_pg, u_e)
        else: # Plus rapide
            Epsilon_e_pg = np.einsum('epij,ej->epi', B_dep_e_pg, u_e, optimize='optimal')
        
        tic.Tac("Matrices", "Epsilon_e_pg", False)

        return Epsilon_e_pg
                    
    def __Calc_Sigma_e_pg(self, Epsilon_e_pg: np.ndarray, matriceType=MatriceType.rigi) -> np.ndarray:
        """Calcul les contraintes depuis les deformations

        Parameters
        ----------
        Epsilon_e_pg : np.ndarray
            Deformations stockées aux éléments et points de gauss (Ne,pg,(3 ou 6))

        Returns
        -------
        np.ndarray
            Renvoie les contrainres endommagé ou non (Ne,pg,(3 ou 6))
        """

        assert Epsilon_e_pg.shape[0] == self.mesh.Ne
        assert Epsilon_e_pg.shape[1] == self.mesh.Get_nPg(matriceType)

        useNumba = self.useNumba

        tic = Tic()

        c = self.comportement.C
        if useNumba:
            # Plus rapide sur les gros système > 500 000 ddl (ordre de grandeur)
            # sinon legerement plus lent
            Sigma_e_pg = CalcNumba.ij_epj_to_epi(c, Epsilon_e_pg)
        else:
            # Plus rapide sur les plus petits système
            Sigma_e_pg = np.einsum('ik,epk->epi', c, Epsilon_e_pg, optimize='optimal')
            
        tic.Tac("Matrices", "Sigma_e_pg", False)

        return Sigma_e_pg

    def Resultats_Resume(self, verbosity=True) -> str:
        resume = super().Resultats_Resume(verbosity)

        return resume

    def __indexResulat(self, resultat: str) -> int:

        dim = self.dim

        if len(resultat) <= 2:
            if "x" in resultat:
                return 0
            elif "y" in resultat:
                return 1
            elif "z" in resultat:
                return 1

        else:

            if "xx" in resultat:
                return 0
            elif "yy" in resultat:
                return 1
            elif "zz" in resultat:
                return 2
            elif "yz" in resultat:
                return 3
            elif "xz" in resultat:
                return 4
            elif "xy" in resultat:
                if dim == 2:
                    return 2
                elif dim == 3:
                    return 5

    def Resultats_Get_dict_Energie(self) -> list[tuple[str, float]]:
        dict_Energie = {
            r"$\Psi_{elas}$": self.__Calc_Psi_Elas()
            }
        return dict_Energie

    def Resultats_Get_Resume_Iteration(self) -> str:
        """Ecrit un résumé de la simulation dans le terminal"""

        resume = ""

        if not self.Resultats_Check_Options_disponibles("Wdef"):
            return
        
        Wdef = self.Get_Resultat("Wdef")
        resume += f"\nW def = {Wdef:.2f}"
        
        Svm = self.Get_Resultat("Svm", nodeValues=False)
        resume += f"\n\nSvm max = {Svm.max():.2f}"

        # Affichage des déplacements
        dx = self.Get_Resultat("ux", nodeValues=True)
        resume += f"\n\nUx max = {dx.max():.2e}"
        resume += f"\nUx min = {dx.min():.2e}"

        dy = self.Get_Resultat("uy", nodeValues=True)
        resume += f"\n\nUy max = {dy.max():.2e}"
        resume += f"\nUy min = {dy.min():.2e}"

        if self.dim == 3:
            dz = self.Get_Resultat("uz", nodeValues=True)
            resume += f"\n\nUz max = {dz.max():.2e}"
            resume += f"\nUz min = {dz.min():.2e}"

        if self._verbosity: print(resume)

        return resume

    def Resultats_Get_ResumeIter_values(self) -> tuple[list[int], list[tuple[str, np.ndarray]]]:
        return super().Resultats_Get_ResumeIter_values()

    def Resultats_matrice_displacement(self) -> np.ndarray:

        Nn = self.mesh.Nn
        coordo = self.displacement.reshape((Nn,-1))
        dim = coordo.shape[1]

        if dim == 1:
            # Ici on rajoute deux colonnes
            coordo = np.append(coordo, np.zeros((Nn,1)), axis=1)
            coordo = np.append(coordo, np.zeros((Nn,1)), axis=1)
        elif dim == 2:
            # Ici on rajoute 1 colonne
            coordo = np.append(coordo, np.zeros((Nn,1)), axis=1)

        return coordo

###################################################################################################

class Simu_PhaseField(Simu):

    def __init__(self, mesh: Mesh, model: PhaseField_Model, verbosity=False, useNumba=True):
        """Creation d'une simulation d'endommagement en champ de phase"""

        assert model.modelType == ModelType.damage, "Le matériau doit être de type damage"
        super().__init__(mesh, model, verbosity, useNumba)

        # init résultats
        self.__psiP_e_pg = []
        self.__old_psiP_e_pg = [] #ancienne densitée d'energie elastique positive PsiPlus(e, pg, 1) pour utiliser le champ d'histoire de miehe
        self.Solver_Set_Elliptic_Algorithm()

    def Get_Resultats_disponibles(self) -> list[str]:

        options = []
        dim = self.dim
        
        if dim == 2:
            options.extend(["ux", "uy", "uz", "amplitude", "displacement", "matrice_displacement"])
            # options.extend(["ax", "ay", "accel", "amplitudeAccel"])
            # options.extend(["vx", "vy", "speed", "amplitudeSpeed"])
            options.extend(["Sxx", "Syy", "Sxy", "Svm","Stress"])
            options.extend(["Exx", "Eyy", "Exy", "Evm","Strain"])

        elif dim == 3:
            options.extend(["ux", "uy", "uz","amplitude","displacement", "matrice_displacement"])
            # options.extend(["ax", "ay", "az", "accel", "amplitudeAccel"])
            # options.extend(["vx", "vy", "vz", "speed", "amplitudeSpeed"])
            options.extend(["Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy", "Svm","Stress"])
            options.extend(["Exx", "Eyy", "Ezz", "Eyz", "Exz", "Exy", "Evm","Strain"])
        
        options.extend(["damage","psiP","Psi_Crack"])
        options.extend(["Wdef","Psi_Elas"])

        return options

    def Paraview_nodesField_elementsField(self, details=False) -> tuple[list[str], list[str]]:
        if details:
            nodesField = ["matrice_displacement", "damage"]
            elementsField = ["Stress", "Strain", "psiP"]
        else:
            nodesField = ["matrice_displacement", "damage"]
            elementsField = ["Stress"]
        return nodesField, elementsField

    def Get_Directions(self, problemType=None) -> list[str]:        
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

    @property
    def useCholesky(self) -> bool:
        return False

    @property
    def A_isSymetric(self) -> bool:
        return False

    def Get_lb_ub(self, problemType: ModelType) -> tuple[np.ndarray, np.ndarray]:
        
        if problemType == ModelType.damage:
            solveur = self.phaseFieldModel.solveur
            if solveur == "BoundConstrain":
                lb = self.damage
                lb[np.where(lb>=1)] = 1-np.finfo(float).eps
                ub = np.ones(lb.shape)
            else:
                lb, ub = np.array([]), np.array([])
        else:
            lb, ub = np.array([]), np.array([])
            
        return lb, ub

    def Get_nbddl_n(self, problemType=None) -> int:        
        if problemType == ModelType.damage:
            return 1
        elif problemType in [ModelType.displacement, None]:
            return self.dim

    @property
    def phaseFieldModel(self) -> PhaseField_Model:
        """Modèle d'endommagement de la simulation"""
        return self.model

    @property
    def displacement(self) -> np.ndarray:
        """Copie du champ vectoriel de déplacement"""
        return self.get_u_n(ModelType.displacement)

    @property
    def damage(self) -> np.ndarray:
        """Copie du champ scalaire d'endommagement"""
        return self.get_u_n(ModelType.damage)

    def add_dirichlet(self, noeuds: np.ndarray, valeurs: np.ndarray, directions: list, problemType=ModelType.displacement, description=""):        
        return super().add_dirichlet(noeuds, valeurs, directions, problemType, description)
    
    def add_lineLoad(self, noeuds: np.ndarray, valeurs: list, directions: list, problemType=ModelType.displacement, description=""):
        return super().add_lineLoad(noeuds, valeurs, directions, problemType, description)

    def add_surfLoad(self, noeuds: np.ndarray, valeurs: list, directions: list, problemType=ModelType.displacement, description=""):
        return super().add_surfLoad(noeuds, valeurs, directions, problemType, description)
        
    def add_neumann(self, noeuds: np.ndarray, valeurs: list, directions: list, problemType=ModelType.displacement, description=""):
        return super().add_neumann(noeuds, valeurs, directions, problemType, description)

    def Get_K_C_M_F(self, problemType=None) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        
        if problemType==None:
            problemType = ModelType.displacement

        taille = self.mesh.Nn * self.Get_nbddl_n(problemType)
        initcsr = sparse.csr_matrix((taille, taille))

        try:    
            if problemType == ModelType.damage:            
                return self.__Kd.copy(), initcsr, initcsr, self.__Fd.copy()
            elif problemType == ModelType.displacement:            
                return self.__Ku.copy(), initcsr, initcsr, self.__Fu.copy()
        except AttributeError:
            print("Sytème pas encore assemblé")
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

    def Assemblage(self):
        self.__Assemblage_d()
        self.__Assemblage_u()
    
    def Solve(self, tolConv=1.0, maxIter=500, convOption=0) -> tuple[np.ndarray, np.ndarray, sparse.csr_matrix, bool]:
        """Résolution du problème d'endommagement de façon étagée

        Parameters
        ----------
        tolConv : float, optional
            Tolérance de convergence entre l'ancien et le nouvelle endommagement, by default 1.0
        maxIter : int, optional
            Nombre d'itération maximum pour atteindre la convergence, by default 500
        convOption : int, optional
            0 -> convergence sur l'endommagement np.max(np.abs(d_kp1-dk)) équivalent normInf(d_kp1-dk)\n
            1 -> convergence sur l'energie de fissure np.abs(psi_crack_kp1 - psi_crack_k)/psi_crack_k

        Returns
        -------
        np.ndarray, np.ndarray, int, float
            u, d, Kglob, nombreIter, dincMax\n

            tel que :\n
            u : champ vectorielle de déplacement
            d : champ scalaire d'endommagement
            Kglob : matrice de rigidité en déplacement
            convergence : la solution a convérgé
        """

        assert tolConv > 0 and tolConv <= 1 , "tolConv doit être compris entre 0 et 1"
        assert maxIter > 1 , "Doit être > 1"

        nombreIter = 0
        convergence = False
        dn = self.damage

        solveur = self.phaseFieldModel.solveur

        tic = Tic()

        while not convergence and nombreIter <= maxIter:
                    
            nombreIter += 1
            # Ancien endommagement dans la procedure de la convergence
            if convOption == 0:
                dk = self.damage
            elif convOption == 1:
                psi_crack_k = self.__Calc_Psi_Crack()
            elif convOption == 2:
                psi_tot_k = self.__Calc_Psi_Crack() + self.__Calc_Psi_Elas()

            # Damage
            self.__Assemblage_d()
            d_kp1 = self.__Solve_d()            
            # Displacement
            Kglob = self.__Assemblage_u()            
            u_np1 = self.__Solve_u()
            
            if convOption == 1:
                psi_crack_kp1 = self.__Calc_Psi_Crack()
            elif convOption == 2:
                psi_tot_kp1 = self.__Calc_Psi_Crack() + self.__Calc_Psi_Elas()

            if convOption == 0:
                convIter = np.max(np.abs(d_kp1-dk))
            elif convOption == 1:
                convIter = np.abs(psi_crack_kp1 - psi_crack_k)/psi_crack_k
            elif convOption == 2:
                convIter = np.abs(psi_tot_kp1 - psi_tot_k)/psi_tot_k
            
            if tolConv == 1.0:
                convergence=True
            else:
                # Condition de convergence
                # convergence = dincMax <= tolConv and iterConv > 1 # idée de florent
                convergence = convIter <= tolConv

        solveurTypes = PhaseField_Model.SolveurType

        if solveur in [solveurTypes.History, solveurTypes.BoundConstrain]:
            d_np1 = d_kp1
            
        elif solveur == solveurTypes.HistoryDamage:
            oldAndNewDamage = np.zeros((d_kp1.shape[0], 2))
            oldAndNewDamage[:, 0] = dn
            oldAndNewDamage[:, 1] = d_kp1
            d_np1 = np.max(oldAndNewDamage, 1)

        else:
            raise Exception("Solveur phase field inconnue")

        temps = tic.Tac("Resolution phase field", "Resolution Phase Field", False)

        self.__nombreIter = nombreIter
        self.__convIter = convIter
        self.__tempsIter = temps
            
        return u_np1, d_np1, Kglob, convergence


    def __ConstruitMatElem_Dep(self) -> np.ndarray:
        """Construit les matrices de rigidités élementaires pour le problème en déplacement

        Returns:
            dict_Ku_e: les matrices elementaires pour chaque groupe d'element
        """

        useNumba = self.useNumba
        # useNumba = False

        matriceType=MatriceType.rigi

        # Data
        mesh = self.mesh
        nPg = mesh.Get_nPg(matriceType)
        
        # Recupère les matrices pour travailler
        
        B_dep_e_pg = mesh.Get_B_dep_e_pg(matriceType)
        leftDepPart = mesh.Get_leftDepPart(matriceType) # -> jacobien_e_pg * poid_pg * B_dep_e_pg'

        d = self.damage
        u = self.displacement

        phaseFieldModel = self.phaseFieldModel
        
        # Calcul la deformation nécessaire pour le split
        Epsilon_e_pg = self.__Calc_Epsilon_e_pg(u, matriceType)

        # Split de la loi de comportement
        cP_e_pg, cM_e_pg = phaseFieldModel.Calc_C(Epsilon_e_pg)

        tic = Tic()
        
        # Endommage : c = g(d) * cP + cM
        g_e_pg = phaseFieldModel.get_g_e_pg(d, mesh, matriceType)
        useNumba=False
        if useNumba:
            # Moins rapide
            cP_e_pg = CalcNumba.ep_epij_to_epij(g_e_pg, cP_e_pg)
            # testcP_e_pg = np.einsum('ep,epij->epij', g_e_pg, cP_e_pg, optimize='optimal') - cP_e_pg
        else:
            # Plus rapide
            cP_e_pg = np.einsum('ep,epij->epij', g_e_pg, cP_e_pg, optimize='optimal')

        c_e_pg = cP_e_pg+cM_e_pg
        
        # Matrice de rigidité élementaire
        useNumba = self.useNumba
        if useNumba:
            # Plus rapide
            Ku_e = CalcNumba.epij_epjk_epkl_to_eil(leftDepPart, c_e_pg, B_dep_e_pg)
        else:
            # Ku_e = np.einsum('ep,p,epki,epkl,eplj->eij', jacobien_e_pg, poid_pg, B_dep_e_pg, c_e_pg, B_dep_e_pg, optimize='optimal')
            Ku_e = np.einsum('epij,epjk,epkl->eil', leftDepPart, c_e_pg, B_dep_e_pg, optimize='optimal') 

        if self.dim == 2:
            epaisseur = self.phaseFieldModel.epaisseur
            Ku_e *= epaisseur
        
        tic.Tac("Matrices","Construction Ku_e", self._verbosity)

        return Ku_e
 
    def __Assemblage_u(self):
        """Construit K global pour le problème en deplacement
        """

        # Data
        mesh = self.mesh        
        taille = mesh.Nn*self.dim

        # Dimension supplémentaire lié a l'utilisation des coefs de lagrange
        dimSupl = len(self.Bc_Lagrange)
        if dimSupl > 0:
            dimSupl += len(self.Bc_ddls_Dirichlet(ModelType.displacement))
            taille += dimSupl

        Ku_e = self.__ConstruitMatElem_Dep()        

        # Prépare assemblage
        lignesVector_e = mesh.lignesVector_e
        colonnesVector_e = mesh.colonnesVector_e
        
        tic = Tic()

        # Assemblage
        self.__Ku = sparse.csr_matrix((Ku_e.reshape(-1), (lignesVector_e.reshape(-1), colonnesVector_e.reshape(-1))), shape=(taille, taille))
        """Matrice Kglob pour le problème en déplacement (Nn*dim, Nn*dim)"""

        # Ici j'initialise Fu calr il faudrait calculer les forces volumiques dans __ConstruitMatElem_Dep !!!
        self.__Fu = sparse.csr_matrix((taille, 1))
        """Vecteur Fglob pour le problème en déplacement (Nn*dim, 1)"""

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.spy(self.__Ku)
        # plt.show()        

        tic.Tac("Matrices","Assemblage Ku et Fu", self._verbosity)
        return self.__Ku

    def __Solve_u(self) -> np.ndarray:
        """Resolution du problème de déplacement"""
            
        self._SolveProblem(ModelType.displacement)

        # On renseigne au model phase field qu'il va falloir mettre à jour le split
        self.phaseFieldModel.Need_Split_Update()
       
        return self.displacement

    # ------------------------------------------- PROBLEME ENDOMMAGEMENT ------------------------------------------- 

    def __Calc_psiPlus_e_pg(self):
        """Calcul de la densité denergie positive\n
        Pour chaque point de gauss de tout les éléments du maillage on va calculer psi+
       
        Returns:
            np.ndarray: self.__psiP_e_pg
        """

        phaseFieldModel = self.phaseFieldModel
        
        u = self.displacement
        d = self.damage

        testu = isinstance(u, np.ndarray) and (u.shape[0] == self.mesh.Nn*self.dim )
        testd = isinstance(d, np.ndarray) and (d.shape[0] == self.mesh.Nn )

        assert testu or testd,"Problème de dimension"

        Epsilon_e_pg = self.__Calc_Epsilon_e_pg(u, MatriceType.masse)
        # ici le therme masse est important sinon on sous intègre

        # Calcul l'energie
        psiP_e_pg, psiM_e_pg = phaseFieldModel.Calc_psi_e_pg(Epsilon_e_pg)

        if phaseFieldModel.solveur == "History":
            # Récupère l'ancien champ d'histoire
            old_psiPlus_e_pg = self.__old_psiP_e_pg
            
            if isinstance(old_psiPlus_e_pg, list) and len(old_psiPlus_e_pg) == 0:
                # Pas encore d'endommagement disponible
                old_psiPlus_e_pg = np.zeros_like(psiP_e_pg)

            inc_H = psiP_e_pg - old_psiPlus_e_pg

            elements, pdGs = np.where(inc_H < 0)

            psiP_e_pg[elements, pdGs] = old_psiPlus_e_pg[elements, pdGs]

            # new = np.linalg.norm(psiP_e_pg)
            # old = np.linalg.norm(self.__old_psiP_e_pg)
            # assert new >= old, "Erreur"
            
        self.__psiP_e_pg = psiP_e_pg


        return self.__psiP_e_pg
    
    def __ConstruitMatElem_Pfm(self):
        """Construit les matrices élementaires pour le problème d'endommagement

        Returns:
            Kd_e, Fd_e: les matrices elementaires pour chaque element
        """

        phaseFieldModel = self.phaseFieldModel

        # Data
        k = phaseFieldModel.k
        PsiP_e_pg = self.__Calc_psiPlus_e_pg()
        r_e_pg = phaseFieldModel.get_r_e_pg(PsiP_e_pg)
        f_e_pg = phaseFieldModel.get_f_e_pg(PsiP_e_pg)

        matriceType=MatriceType.masse

        mesh = self.mesh

        # Probleme de la forme K*Laplacien(d) + r*d = F        
        ReactionPart_e_pg = mesh.Get_phaseField_ReactionPart_e_pg(matriceType) # -> jacobien_e_pg * poid_pg * Nd_pg' * Nd_pg
        DiffusePart_e_pg = mesh.Get_phaseField_DiffusePart_e_pg(matriceType) # -> jacobien_e_pg, poid_pg, Bd_e_pg', Bd_e_pg
        SourcePart_e_pg = mesh.Get_phaseField_SourcePart_e_pg(matriceType) # -> jacobien_e_pg, poid_pg, Nd_pg'
        
        tic = Tic()

        # useNumba = self.__useNumba
        useNumba = False        
        if useNumba:
            # Moin rapide et beug Fd_e

            Kd_e, Fd_e = CalcNumba.Construit_Kd_e_and_Fd_e(r_e_pg, ReactionPart_e_pg,
            k, DiffusePart_e_pg,
            f_e_pg, SourcePart_e_pg)
            # testFd_e = np.einsum('ep,epij->eij', f_e_pg, SourcePart_e_pg, optimize='optimal') - Fd_e

            # assert np.max(testFd_e) == 0, "Erreur"

            # K_r_e = np.einsum('ep,epij->eij', r_e_pg, ReactionPart_e_pg, optimize='optimal')
            # K_K_e = np.einsum('epij->eij', DiffusePart_e_pg * k, optimize='optimal')
            # testFd_e = np.einsum('ep,epij->eij', f_e_pg, SourcePart_e_pg, optimize='optimal') - Fd_e
            # testKd_e = K_r_e+K_K_e - Kd_e

        else:
            # Plus rapide sans beug

            # jacobien_e_pg = mesh.Get_jacobien_e_pg(matriceType)
            # poid_pg = mesh.Get_poid_pg(matriceType)
            # Nd_pg = mesh.Get_N_scalaire_pg(matriceType)
            # Bd_e_pg = mesh.Get_B_sclaire_e_pg(matriceType)

            # Partie qui fait intervenir le therme de reaction r ->  jacobien_e_pg * poid_pg * r_e_pg * Nd_pg' * Nd_pg
            K_r_e = np.einsum('ep,epij->eij', r_e_pg, ReactionPart_e_pg, optimize='optimal')
            # K_r_e = np.einsum('ep,p,ep,pki,pkj->eij', jacobien_e_pg, poid_pg, r_e_pg, Nd_pg, Nd_pg, optimize='optimal')

            # Partie qui fait intervenir le therme de diffusion K -> jacobien_e_pg, poid_pg, k, Bd_e_pg', Bd_e_pg
            K_K_e = np.einsum('epij->eij', DiffusePart_e_pg * k, optimize='optimal')
            # K_K_e = np.einsum('ep,p,,epki,epkj->eij', jacobien_e_pg, poid_pg, k, Bd_e_pg, Bd_e_pg, optimize='optimal')
            
            # Construit Fd_e -> jacobien_e_pg, poid_pg, f_e_pg, Nd_pg'
            Fd_e = np.einsum('ep,epij->eij', f_e_pg, SourcePart_e_pg, optimize='optimal') #Ici on somme sur les points d'integrations
        
            Kd_e = K_r_e + K_K_e

        tic.Tac("Matrices","Construction Kd_e et Fd_e", self._verbosity)

        return Kd_e, Fd_e

    def __Assemblage_d(self):
        """Construit Kglobal pour le problème d'endommagement
        """
       
        # Data
        mesh = self.mesh
        taille = mesh.Nn
        lignesScalar_e = mesh.lignesScalar_e
        colonnesScalar_e = mesh.colonnesScalar_e

        # Dimension supplémentaire lié a l'utilisation des coefs de lagrange
        dimSupl = len(self.Bc_Lagrange)
        if dimSupl > 0:
            dimSupl += len(self.Bc_ddls_Dirichlet(ModelType.damage))
            taille += dimSupl
        
        # Calul les matrices elementaires
        Kd_e, Fd_e = self.__ConstruitMatElem_Pfm()

        # Assemblage
        tic = Tic()        

        self.__Kd = sparse.csr_matrix((Kd_e.reshape(-1), (lignesScalar_e.reshape(-1), colonnesScalar_e.reshape(-1))), shape = (taille, taille))
        """Kglob pour le problème d'endommagement (Nn, Nn)"""
        
        lignes = mesh.connect.reshape(-1)
        self.__Fd = sparse.csr_matrix((Fd_e.reshape(-1), (lignes,np.zeros(len(lignes)))), shape = (taille,1))
        """Fglob pour le problème d'endommagement (Nn, 1)"""        

        tic.Tac("Matrices","Assemblage Kd et Fd", self._verbosity)        

        return self.__Kd, self.__Fd    
    
    def __Solve_d(self) -> np.ndarray:
        """Resolution du problème d'endommagement"""
        
        self._SolveProblem(ModelType.damage)

        return self.damage

    def Save_Iteration(self):

        iter = super().Save_Iteration()

        # informations de convergence
        
        iter["nombreIter"] = self.__nombreIter
        iter["tempsIter"] = self.__tempsIter
        iter["convIter"] = self.__convIter
    
        if self.phaseFieldModel.solveur == PhaseField_Model.SolveurType.History:
            # mets à jour l'ancien champ histoire pour la prochaine résolution 
            self.__old_psiP_e_pg = self.__psiP_e_pg
            
        iter['displacement'] = self.displacement
        iter['damage'] = self.damage

        self._results.append(iter)

    def Update_iter(self, index=-1):

        results = super().Update_iter(index)

        if results == None: return

        self.__old_psiP_e_pg = [] # Il vraiment utile de faire ça sinon quand on calculer psiP il va y avoir un problème

        damageType = ModelType.damage
        self.set_u_n(damageType, results[damageType])

        displacementType = ModelType.displacement
        self.set_u_n(displacementType, results[displacementType])

        self.phaseFieldModel.Need_Split_Update()

    def Get_Resultat(self, option: str, nodeValues=True, iter=None):
        
        dim = self.dim
        Ne = self.mesh.Ne
        Nn = self.mesh.Nn
        
        if not self.Resultats_Check_Options_disponibles(option): return None

        if iter != None:
            self.Update_iter(iter)

        if option in ["Wdef","Psi_Elas"]:
            return self.__Calc_Psi_Elas()

        if option == "Psi_Crack":
            return self.__Calc_Psi_Crack()

        if option == "damage":
            return self.damage

        if option == "psiP":
            resultat_e_pg = self.__Calc_psiPlus_e_pg()
            resultat_e = np.mean(resultat_e_pg, axis=1)

            if nodeValues:
                return self.Resultats_InterpolationAuxNoeuds(resultat_e)
            else:
                return resultat_e

        if option == "displacement":
            return self.displacement

        if option == "speed":
            return self.speed
        
        if option == "accel":
            return self.accel

        if option == 'matrice_displacement':
            return self.Resultats_matrice_displacement()

        displacement = self.displacement

        coef = self.phaseFieldModel.comportement.coef       

        # Deformation et contraintes pour chaque element et chaque points de gauss        
        Epsilon_e_pg = self.__Calc_Epsilon_e_pg(displacement)
        Sigma_e_pg = self.__Calc_Sigma_e_pg(Epsilon_e_pg)

        # Moyenne sur l'élement
        Epsilon_e = np.mean(Epsilon_e_pg, axis=1)
        Sigma_e = np.mean(Sigma_e_pg, axis=1)
        
        if "S" in option or "E" in option and option != "amplitudeSpeed":

            if "S" in option and option != "Strain":
                val_e = Sigma_e
            
            elif "E" in option or option == "Strain":
                val_e = Epsilon_e

            else:
                raise Exception("Mauvaise option")

            if dim == 2:

                val_xx_e = val_e[:,0]
                val_yy_e = val_e[:,1]
                val_xy_e = val_e[:,2]/coef
                
                # TODO Ici il faudrait calculer Szz si deformation plane
                
                val_vm_e = np.sqrt(val_xx_e**2+val_yy_e**2-val_xx_e*val_yy_e+3*val_xy_e**2)

                if "xx" in option:
                    resultat_e = val_xx_e
                elif "yy" in option:
                    resultat_e = val_yy_e
                elif "xy" in option:
                    resultat_e = val_xy_e
                elif "vm" in option:
                    resultat_e = val_vm_e

            elif dim == 3:

                val_xx_e = val_e[:,0]
                val_yy_e = val_e[:,1]
                val_zz_e = val_e[:,2]
                val_yz_e = val_e[:,3]/coef
                val_xz_e = val_e[:,4]/coef
                val_xy_e = val_e[:,5]/coef               

                val_vm_e = np.sqrt(((val_xx_e-val_yy_e)**2+(val_yy_e-val_zz_e)**2+(val_zz_e-val_xx_e)**2+6*(val_xy_e**2+val_yz_e**2+val_xz_e**2))/2)

                if "xx" in option:
                    resultat_e = val_xx_e
                elif "yy" in option:
                    resultat_e = val_yy_e
                elif "zz" in option:
                    resultat_e = val_zz_e
                elif "yz" in option:
                    resultat_e = val_yz_e
                elif "xz" in option:
                    resultat_e = val_xz_e
                elif "xy" in option:
                    resultat_e = val_xy_e
                elif "vm" in option:
                    resultat_e = val_vm_e

            if option in ["Stress","Strain"]:
                resultat_e = np.append(val_e, val_vm_e.reshape((Ne,1)), axis=1)

            if nodeValues:
                resultat_n = self.Resultats_InterpolationAuxNoeuds(resultat_e)
                return resultat_n
            else:
                return resultat_e
        
        else:

            Nn = self.mesh.Nn

            if option in ["ux", "uy", "uz", "amplitude"]:
                resultat_ddl = self.displacement
            elif option in ["vx", "vy", "vz", "amplitudeSpeed"]:
                resultat_ddl = self.speed
            elif option in ["ax", "ay", "az", "amplitudeAccel"]:
                resultat_ddl = self.accel

            resultat_ddl = resultat_ddl.reshape(Nn, -1)

            index = self.__indexResulat(option)
            
            if nodeValues:

                if "amplitude" in option:
                    return np.sqrt(np.sum(resultat_ddl**2,axis=1))
                else:
                    if len(resultat_ddl.shape) > 1:
                        return resultat_ddl[:,index]
                    else:
                        return resultat_ddl.reshape(-1)
                        
            else:

                # recupere pour chaque element les valeurs de ses noeuds
                resultat_e_n = self.mesh.Localises_sol_e(resultat_ddl)
                resultat_e = resultat_e_n.mean(axis=1)

                if "amplitude" in option:
                    return np.sqrt(np.sum(resultat_e**2, axis=1))
                elif option in ["speed", "accel"]:
                    return resultat_e.reshape(-1)
                else:
                    if len(resultat_e.shape) > 1:
                        return resultat_e[:,index]
                    else:
                        return resultat_e.reshape(-1)

    def __indexResulat(self, resultat: str) -> int:

        dim = self.dim

        if len(resultat) <= 2:
            if "x" in resultat:
                return 0
            elif "y" in resultat:
                return 1
            elif "z" in resultat:
                return 1

        else:

            if "xx" in resultat:
                return 0
            elif "yy" in resultat:
                return 1
            elif "zz" in resultat:
                return 2
            elif "yz" in resultat:
                return 3
            elif "xz" in resultat:
                return 4
            elif "xy" in resultat:
                if dim == 2:
                    return 2
                elif dim == 3:
                    return 5

    def __Calc_Psi_Elas(self) -> float:
        """Calcul de l'energie de deformation cinématiquement admissible endommagé ou non
        Calcul de Wdef = 1/2 int_Omega jacobien * poid * Sig : Eps dOmega x epaisseur"""

        tic = Tic()

        sol_u  = self.displacement

        matriceType = MatriceType.rigi
        Epsilon_e_pg = self.__Calc_Epsilon_e_pg(sol_u, matriceType)
        jacobien_e_pg = self.mesh.Get_jacobien_e_pg(matriceType)
        poid_pg = self.mesh.Get_poid_pg(matriceType)

        if self.dim == 2:
            ep = self.phaseFieldModel.epaisseur
        else:
            ep = 1

        d = self.damage

        phaseFieldModel = self.phaseFieldModel
        psiP_e_pg, psiM_e_pg = phaseFieldModel.Calc_psi_e_pg(Epsilon_e_pg)

        # Endommage : psiP_e_pg = g(d) * PsiP_e_pg 
        g_e_pg = phaseFieldModel.get_g_e_pg(d, self.mesh, matriceType)
        psiP_e_pg = np.einsum('ep,ep->ep', g_e_pg, psiP_e_pg, optimize='optimal')
        psi_e_pg = psiP_e_pg + psiM_e_pg

        Wdef = np.einsum(',ep,p,ep->', ep, jacobien_e_pg, poid_pg, psi_e_pg, optimize='optimal')
        
        Wdef = float(Wdef)

        tic.Tac("PostTraitement","Calcul Psi Elas",False)
        
        return Wdef

    def __Calc_Psi_Crack(self) -> float:
        """Calcul l'energie de fissure"""

        tic = Tic()

        pfm = self.phaseFieldModel 

        matriceType = MatriceType.masse

        Gc = pfm.Gc
        l0 = pfm.l0
        c0 = pfm.c0

        d_n = self.damage
        d_e = self.mesh.Localises_sol_e(d_n)

        jacobien_e_pg = self.mesh.Get_jacobien_e_pg(matriceType)
        poid_pg = self.mesh.Get_poid_pg(matriceType)
        Nd_pg = self.mesh.Get_N_scalaire_pg(matriceType)
        Bd_e_pg = self.mesh.Get_dN_sclaire_e_pg(matriceType)

        grad_e_pg = np.einsum('epij,ej->epi',Bd_e_pg,d_e, optimize='optimal')
        diffuse_e_pg = grad_e_pg**2

        gradPart = np.einsum('ep,p,,epi->',jacobien_e_pg, poid_pg, Gc*l0/c0, diffuse_e_pg, optimize='optimal')

        alpha_e_pg = np.einsum('pij,ej->epi', Nd_pg, d_e, optimize='optimal')
        if pfm.regularization == PhaseField_Model.RegularizationType.AT2:
            alpha_e_pg = alpha_e_pg**2
        
        alphaPart = np.einsum('ep,p,,epi->',jacobien_e_pg, poid_pg, Gc/(c0*l0), alpha_e_pg, optimize='optimal')

        if self.dim == 2:
            ep = self.phaseFieldModel.epaisseur
        else:
            ep = 1

        Psi_Crack = (alphaPart + gradPart)*ep

        tic.Tac("PostTraitement","Calcul Psi Crack",False)

        return Psi_Crack

    def __Calc_Epsilon_e_pg(self, sol: np.ndarray, matriceType=MatriceType.rigi):
        """Construit epsilon pour chaque element et chaque points de gauss

        Parameters
        ----------
        u : np.ndarray
            Vecteur des déplacements

        Returns
        -------
        np.ndarray
            Deformations stockées aux éléments et points de gauss (Ne,pg,(3 ou 6))
        """

        useNumba = self.useNumba
        useNumba = False
        
        tic = Tic()

        u_e = self.mesh.Localises_sol_e(sol)
        B_dep_e_pg = self.mesh.Get_B_dep_e_pg(matriceType)
        if useNumba: # Moins rapide
            Epsilon_e_pg = CalcNumba.epij_ej_to_epi(B_dep_e_pg, u_e)
        else: # Plus rapide
            Epsilon_e_pg = np.einsum('epij,ej->epi', B_dep_e_pg, u_e, optimize='optimal')
        
        tic.Tac("Matrices", "Epsilon_e_pg", False)

        return Epsilon_e_pg

    def __Calc_Sigma_e_pg(self, Epsilon_e_pg: np.ndarray, matriceType=MatriceType.rigi) -> np.ndarray:
        """Calcul les contraintes depuis les deformations

        Parameters
        ----------
        Epsilon_e_pg : np.ndarray
            Deformations stockées aux éléments et points de gauss (Ne,pg,(3 ou 6))

        Returns
        -------
        np.ndarray
            Renvoie les contrainres endommagé ou non (Ne,pg,(3 ou 6))
        """

        assert Epsilon_e_pg.shape[0] == self.mesh.Ne
        assert Epsilon_e_pg.shape[1] == self.mesh.Get_nPg(matriceType)

        useNumba = self.useNumba

        d = self.damage

        phaseFieldModel = self.phaseFieldModel

        SigmaP_e_pg, SigmaM_e_pg = phaseFieldModel.Calc_Sigma_e_pg(Epsilon_e_pg)
        
        # Endommage : Sig = g(d) * SigP + SigM
        g_e_pg = phaseFieldModel.get_g_e_pg(d, self.mesh, matriceType)

        useNumba = False
        tic = Tic()
        if useNumba:
            # Moins rapide 
            SigmaP_e_pg = CalcNumba.ep_epi_to_epi(g_e_pg, SigmaP_e_pg)
        else:
            # Plus rapide
            SigmaP_e_pg = np.einsum('ep,epi->epi', g_e_pg, SigmaP_e_pg, optimize='optimal')

        Sigma_e_pg = SigmaP_e_pg + SigmaM_e_pg
            
        tic.Tac("Matrices", "Sigma_e_pg", False)

        return Sigma_e_pg

    def Resultats_Set_Resume_Chargement(self, loadMax: float, listInc: list, listTreshold: list, listOption: list):        
        
        assert len(listInc) == len(listTreshold) and len(listInc) == len(listOption), "Doit etre de la même dimension"
        
        resumeChargement = 'Chargement :'
        resumeChargement += f'\n\tload max = {loadMax:.3}'

        for inc, treshold, option in zip(listInc, listTreshold, listOption):

            resumeChargement += f'\n\tinc = {inc} -> {option} < {treshold}'
        
        self.__resumeChargement = resumeChargement

        return self.__resumeChargement

    def Resultats_Get_Resume_Chargement(self) -> str:
        try:
            return self.__resumeChargement
        except AttributeError:
            return ""

    def Resultats_Set_Resume_Iteration(self, resol: int, load: float, uniteLoad: str, pourcentage=0.0, remove=False):
        """Construit le résumé de l'itération pour le problème d'endommagement

        Parameters
        ----------
        resol : int
            identifiant de la résolution
        load : float
            chargement imposé
        uniteLoad : str, optional
            unité du chargement, by default "m"
        pourcentage : float, optional
            pourcentage de la simualtion réalisée, by default 0.0
        remove : bool, optional
            supprime la ligne dans le terminal apres l'affichage, by default False
        """

        d = self.damage

        nombreIter = self.__nombreIter
        dincMax = self.__convIter
        temps = self.__tempsIter

        min_d = d.min()
        max_d = d.max()
        resumeIter = f"{resol:4} : ud = {np.round(load,3)} {uniteLoad},  d = [{min_d:.2e}; {max_d:.2e}], {nombreIter}:{np.round(temps,3)} s, tol={dincMax:.2e}  "
        
        if remove:
            end='\r'
        else:
            end=''

        if pourcentage > 0:
            tempsRestant = (1/pourcentage-1)*temps*resol
            
            tempsCoef, unite = Tic.Get_temps_unite(tempsRestant)

            # Rajoute le pourcentage et lestimation du temps restant
            resumeIter = resumeIter+f"{np.round(pourcentage*100,2)} % -> {np.round(tempsCoef,1)} {unite}   "

        print(resumeIter, end=end)

        self.__resumeIter = resumeIter

    def Resultats_Get_Resume_Iteration(self) -> str:        
        return self.__resumeIter

    def Resultats_Get_dict_Energie(self) -> list[tuple[str, float]]:
        PsiElas = self.__Calc_Psi_Elas()
        PsiCrack = self.__Calc_Psi_Crack()
        dict_Energie = {
            r"$\Psi_{elas}$": PsiElas,
            r"$\Psi_{crack}$": PsiCrack,
            r"$\Psi_{tot}$": PsiCrack+PsiElas
            }
        return dict_Energie

    def Resultats_Get_ResumeIter_values(self) -> list[tuple[str, np.ndarray]]:
        
        list_label_values = []
        
        resultats = self._results
        df = pd.DataFrame(resultats)
        iterations = np.arange(df.shape[0])
        
        damageMaxIter = np.max(list(df["damage"].values), axis=1)
        list_label_values.append((r"$\phi$", damageMaxIter))

        tolConvergence = df["convIter"].values
        list_label_values.append(("convergence", tolConvergence))

        nombreIter = df["nombreIter"].values
        list_label_values.append(("nombre Iter", nombreIter))

        tempsIter = df["tempsIter"].values
        list_label_values.append(("temps Iter", tempsIter))
        
        return iterations, list_label_values
    
    def Resultats_matrice_displacement(self) -> np.ndarray:
        
        Nn = self.mesh.Nn
        coordo = self.displacement.reshape((Nn,-1))
        dim = coordo.shape[1]

        if dim == 1:
            # Ici on rajoute deux colonnes
            coordo = np.append(coordo, np.zeros((Nn,1)), axis=1)
            coordo = np.append(coordo, np.zeros((Nn,1)), axis=1)
        elif dim == 2:
            # Ici on rajoute 1 colonne
            coordo = np.append(coordo, np.zeros((Nn,1)), axis=1)

        return coordo
    

###################################################################################################

class Simu_Beam(Simu):

    def __init__(self, mesh: Mesh, model: Beam_Model, verbosity=False, useNumba=True):
        """Creation d'une simulation poutre"""

        assert model.modelType == ModelType.beam, "Le matériau doit être de type beam"
        super().__init__(mesh, model, verbosity, useNumba)

        # init
        self.Solver_Set_Elliptic_Algorithm()
    
    def Get_Resultats_disponibles(self) -> list[str]:

        options = []
        nbddl_n = self.Get_nbddl_n(self.problemType)
        
        if nbddl_n == 1:
            options.extend(["ux", "beamDisplacement", "matrice_displacement"])
            options.extend(["fx"])

        elif nbddl_n == 3:
            options.extend(["ux","uy","rz", "amplitude", "beamDisplacement", "matrice_displacement"])
            options.extend(["fx", "fy", "cz", "Exx", "Exy", "Sxx", "Sxy"])            

        elif nbddl_n == 6:
            options.extend(["ux", "uy", "uz", "rx", "ry", "rz", "amplitude", "beamDisplacement", "matrice_displacement"])
            options.extend(["fx","fy","fz","cx","cy","cz"])
        
        options.extend(["Srain", "Stress"])        

        return options

    def Paraview_nodesField_elementsField(self, details=False) -> tuple[list[str], list[str]]:
        if details:
            nodesField = ["matrice_displacement"]
            elementsField = ["Stress"]
        else:
            nodesField = ["matrice_displacement"]
            elementsField = ["Stress"]
        return nodesField, elementsField

    def Get_Directions(self, problemType=None) -> list[str]:
        dict_nbddl_directions = {
            1 : ["x"],
            3 : ["x","y","rz"],
            6 : ["x","y","z","rx","ry","rz"]
        }
        return dict_nbddl_directions[self.beamModel.nbddl_n]
    
    def Get_problemTypes(self) -> list[ModelType]:
        return [ModelType.beam]

    @property
    def beamModel(self) -> Beam_Model:
        """Modèle poutre de la simulation"""
        return self.model

    @property
    def useCholesky(self) -> bool:
        return False

    @property
    def A_isSymetric(self) -> bool:
        return False

    def Get_nbddl_n(self, problemType=None) -> int:
        return self.beamModel.nbddl_n

    def Check_dim_mesh_materiau(self) -> None:
        # Dans le cadre d'un problème de poutre on à pas besoin de verifier cette condition
        pass
    
    @property
    def use3DBeamModel(self) -> bool:
        if self.beamModel.dim == 3:
            return True
        else:
            return False

    @property
    def beamDisplacement(self) -> np.ndarray:
        """Copie du champ vectoriel de déplacement pour le problème poutre"""
        return self.get_u_n(self.problemType)

    def add_surfLoad(self, noeuds: np.ndarray, valeurs: list, directions: list, problemType=None, description=""):
        print("Il est impossible d'appliquer une charge surfacique dans un problème poutre")
        return

    def add_volumeLoad(self, noeuds: np.ndarray, valeurs: list, directions: list, problemType=None, description=""):
        print("Il est impossible d'appliquer une charge volumique sur un problème poutre")
        return

    def add_liaison_Encastrement(self, noeuds: np.ndarray, description="Encastrement"):

        beamModel = self.beamModel

        if beamModel.dim == 1:
            directions = ['x']
        elif beamModel.dim == 2:
            directions = ['x','y','rz']
        elif beamModel.dim == 3:
            directions = ['x','y','z','rx','ry','rz']

        description = f"Liaison {description}"
        
        self.add_liaisonPoutre(noeuds, directions, description)

    def add_liaison_Rotule(self, noeuds: np.ndarray, directions=[''] ,description="Rotule"):

        beamModel = self.beamModel
        
        if beamModel.dim == 1:
            return
        elif beamModel.dim == 2:
            directions = ['x','y']
        elif beamModel.dim == 3:
            directionsDeBase = ['x','y','z']
            if directions != ['']:
                # On va bloquer les ddls de rotations que ne sont pas dans directions
                directionsRot = ['rx','ry','rz']
                for dir in directions:
                    if dir in directionsRot.copy():
                        directionsRot.remove(dir)

            directions = directionsDeBase
            directions.extend(directionsRot)

        description = f"Liaison {description}"
        
        self.add_liaisonPoutre(noeuds, directions, description)

    def add_liaisonPoutre(self, noeuds: np.ndarray, directions: list[str], description: str):        

        nbddl_n = self.Get_nbddl_n(self.problemType)
        problemType = self.problemType

        # Verficiation
        self.Check_Directions(problemType, directions)

        tic = Tic()

        # On va venir pour chaque directions appliquer les conditions
        for d, dir in enumerate(directions):
            ddls = BoundaryCondition.Get_ddls_noeuds(nbddl_n,  problemType=problemType, noeuds=noeuds, directions=[dir])

            new_LagrangeBc = LagrangeCondition(problemType, noeuds, ddls, [dir], [0], [1,-1], description)

            self._Bc_Add_Lagrange(new_LagrangeBc)
        
        # Il n'est pas possible de poser u1-u2 + v1-v2 = 0
        # Il faut appliquer une seule condition à la fois

        tic.Tac("Boundary Conditions","Liaison", self._verbosity)

        self.Bc_Add_LagrangeAffichage(noeuds, directions, description)

    def __ConstruitMatElem_Beam(self) -> np.ndarray:
        """Construit les matrices de rigidités élementaires pour le problème en déplacement

        Returns:
            Ku_beam: les matrices elementaires pour chaque groupe d'element 
        """

        # Data
        mesh = self.mesh
        if not mesh.groupElem.dim == 1: return
        groupElem = mesh.groupElem

        # Récupération du maodel poutre
        beamModel = self.beamModel
        if not isinstance(beamModel, Beam_Model): return
        
        matriceType=MatriceType.beam
        
        tic = Tic()
        
        jacobien_e_pg = mesh.Get_jacobien_e_pg(matriceType)
        poid_pg = mesh.Get_poid_pg(matriceType)

        D_e_pg = beamModel.Calc_D_e_pg(groupElem, matriceType)
        
        B_beam_e_pg = self.__Get_B_beam_e_pg(matriceType)
        
        Kbeam_e = np.einsum('ep,p,epji,epjk,epkl->eil', jacobien_e_pg, poid_pg, B_beam_e_pg, D_e_pg, B_beam_e_pg, optimize='optimal')
            
        tic.Tac("Matrices","Construction Kbeam_e", self._verbosity)

        return Kbeam_e

    def __Get_B_beam_e_pg(self, matriceType: str):

        # Exemple matlab : FEMOBJECT/BASIC/MODEL/ELEMENTS/@BEAM/calc_B.m

        tic = Tic()

        # Récupération du maodel poutre
        beamModel = self.beamModel
        assert isinstance(beamModel, Beam_Model)
        dim = beamModel.dim
        nbddl_n = beamModel.nbddl_n

        # Data
        mesh = self.mesh
        jacobien_e_pg = mesh.Get_jacobien_e_pg(matriceType)
        groupElem = mesh.groupElem
        elemType = groupElem.elemType
        nPe = groupElem.nPe
        Ne = jacobien_e_pg.shape[0]
        nPg = jacobien_e_pg.shape[1]

        # Recupère les matrices pour travailler
        dN_e_pg = mesh.Get_dN_sclaire_e_pg(matriceType)
        if beamModel.dim > 1:
            ddNv_e_pg = mesh.Get_ddNv_sclaire_e_pg(matriceType)

        if dim == 1:
            # u = [u1, . . . , un]
            
            # repu = np.arange(0,3*nPe,3) # [0,3] (SEG2) [0,3,6] (SEG3)
            if elemType == ElemType.SEG2:
                repu = [0,1]
            elif elemType == ElemType.SEG3:
                repu = [0,1,2]
            elif elemType == ElemType.SEG4:
                repu = [0,1,2,3]
            elif elemType == ElemType.SEG5:
                repu = [0,1,2,3,4]

            B_e_pg = np.zeros((Ne, nPg, 1, nbddl_n*nPe))
            B_e_pg[:,:,0, repu] = dN_e_pg[:,:,0]
                

        elif dim == 2:
            # u = [u1, v1, rz1, . . . , un, vn, rzn]
            
            # repu = np.arange(0,3*nPe,3) # [0,3] (SEG2) [0,3,6] (SEG3)
            if elemType == ElemType.SEG2:
                repu = [0,3]
                repv = [1,2,4,5]
            elif elemType == ElemType.SEG3:
                repu = [0,3,6]
                repv = [1,2,4,5,7,8]
            elif elemType == ElemType.SEG4:
                repu = [0,3,6,9]
                repv = [1,2,4,5,7,8,10,11]
            elif elemType == ElemType.SEG5:
                repu = [0,3,6,9,12]
                repv = [1,2,4,5,7,8,10,11,13,14]

            B_e_pg = np.zeros((Ne, nPg, 2, nbddl_n*nPe))
            
            B_e_pg[:,:,0, repu] = dN_e_pg[:,:,0]
            B_e_pg[:,:,1, repv] = ddNv_e_pg[:,:,0]

        elif dim == 3:
            # u = [u1, v1, w1, rx1, ry1, rz1, u2, v2, w2, rx2, ry2, rz2]

            if elemType == ElemType.SEG2:
                repux = [0,6]
                repvy = [1,5,7,11]
                repvz = [2,4,8,10]
                reptx = [3,9]
            elif elemType == ElemType.SEG3:
                repux = [0,6,12]
                repvy = [1,5,7,11,13,17]
                repvz = [2,4,8,10,14,16]
                reptx = [3,9,15]
            elif elemType == ElemType.SEG4:
                repux = [0,6,12,18]
                repvy = [1,5,7,11,13,17,19,23]
                repvz = [2,4,8,10,14,16,20,22]
                reptx = [3,9,15,21]
            elif elemType == ElemType.SEG5:
                repux = [0,6,12,18,24]
                repvy = [1,5,7,11,13,17,19,23,25,29]
                repvz = [2,4,8,10,14,16,20,22,26,28]
                reptx = [3,9,15,21,27]

            B_e_pg = np.zeros((Ne, nPg, 4, nbddl_n*nPe))
            
            B_e_pg[:,:,0, repux] = dN_e_pg[:,:,0]
            B_e_pg[:,:,1, reptx] = dN_e_pg[:,:,0]
            B_e_pg[:,:,3, repvy] = ddNv_e_pg[:,:,0]
            ddNvz_e_pg = ddNv_e_pg.copy()
            ddNvz_e_pg[:,:,0,[1,3]] *= -1 # RY = -UZ'
            B_e_pg[:,:,2, repvz] = ddNvz_e_pg[:,:,0]

        # Fait la rotation si nécessaire

        if dim == 1:
            B_beam_e_pg = B_e_pg
        else:
            P = mesh.groupElem.sysCoord_e
            Pglob_e = np.zeros((Ne, nbddl_n*nPe, nbddl_n*nPe))
            Ndim = nbddl_n*nPe
            N = P.shape[1]
            listlignes = np.repeat(range(N), N)
            listColonnes = np.array(list(range(N))*N)
            for e in range(nbddl_n*nPe//3):
                Pglob_e[:,listlignes + e*N, listColonnes + e*N] = P[:,listlignes,listColonnes]

            B_beam_e_pg = np.einsum('epij,ejk->epik', B_e_pg, Pglob_e, optimize='optimal')

        tic.Tac("Matrices","Construction B_beam_e_pg", False)

        return B_beam_e_pg

    def Assemblage(self):

        if self.needUpdate:

            # Data
            mesh = self.mesh

            model = self.beamModel

            assert isinstance(model, Beam_Model)

            taille = mesh.Nn * model.nbddl_n

            Ku_beam = self.__ConstruitMatElem_Beam()

            # Dimension supplémentaire lié a l'utilisation des coefs de lagrange
            dimSupl = len(self.Bc_Lagrange)
            if dimSupl > 0:
                dimSupl += len(self.Bc_ddls_Dirichlet(ModelType.beam))                
                taille += dimSupl

            # Prépare assemblage
            lignesVector_e = mesh.Get_lignesVectorBeam_e(model.nbddl_n)
            colonnesVector_e = mesh.Get_colonnesVectorBeam_e(model.nbddl_n)
            
            tic = Tic()

            # Assemblage
            self.__Kbeam = sparse.csr_matrix((Ku_beam.reshape(-1), (lignesVector_e.reshape(-1), colonnesVector_e.reshape(-1))), shape=(taille, taille))
            """Matrice Kglob pour le problème poutre (Nn*nbddl_e, Nn*nbddl_e)"""

            # Ici j'initialise Fu calr il faudrait calculer les forces volumiques dans __ConstruitMatElem_Dep !!!
            self.__Fbeam = sparse.csr_matrix((taille, 1))
            """Vecteur Fglob pour le problème poutre (Nn*nbddl_e, 1)"""

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.spy(self.__Ku)
            # plt.show()

            tic.Tac("Matrices","Assemblage Kbeam et Fbeam", self._verbosity)

    def Get_K_C_M_F(self, problemType=None) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        taille = self.mesh.Nn * self.Get_nbddl_n(problemType)
        initcsr = sparse.csr_matrix((taille, taille))
        return self.__Kbeam, initcsr, initcsr, self.__Fbeam

    def Get_x0(self, problemType=None):
        if self.beamDisplacement.size != self.mesh.Nn*self.Get_nbddl_n(problemType):
            return np.zeros(self.mesh.Nn*self.Get_nbddl_n(problemType))
        else:
            return self.beamDisplacement

    def Save_Iteration(self):

        iter = super().Save_Iteration()
        
        iter['beamDisplacement'] = self.beamDisplacement
            
        self._results.append(iter)

    def Update_iter(self, index=-1):
        
        results = super().Update_iter(index)

        if results == None: return

        self.set_u_n(self.problemType, results["beamDisplacement"])

    def Get_Resultat(self, option: str, nodeValues=True, iter=None):
        
        if not self.Resultats_Check_Options_disponibles(option): return None

        if iter != None:
            self.Update_iter(iter)

        if option == "beamDisplacement":
            return self.beamDisplacement.copy()
    
        if option == 'matrice_displacement':
            return self.Resultats_matrice_displacement()

        if option in ["fx","fy","fz","cx","cy","cz"]:

            force = np.array(self.__Kbeam.dot(self.beamDisplacement))
            force_redim = force.reshape(self.mesh.Nn, -1)
            index = Simu.Resultats_indexe_option(self.dim, self.problemType, option)
            resultat_ddl = force_redim[:, index]

        else:

            resultat_ddl = self.beamDisplacement
            resultat_ddl = resultat_ddl.reshape((self.mesh.Nn,-1))

        index = self.__indexResulat(option)


        # Deformation et contraintes pour chaque element et chaque points de gauss        
        Epsilon_e_pg = self.__Calc_Epsilon_e_pg(self.beamDisplacement)
        Sigma_e_pg = self.__Calc_Sigma_e_pg(Epsilon_e_pg)

        if option == "Stress":
            return np.mean(Sigma_e_pg, axis=1)


        if nodeValues:
            if option == "amplitude":
                return np.sqrt(np.sum(resultat_ddl,axis=1))
            else:
                if len(resultat_ddl.shape) > 1:
                    return resultat_ddl[:,index]
                else:
                    return resultat_ddl.reshape(-1)
        else:
            # recupere pour chaque element les valeurs de ses noeuds
                resultat_e_n = self.mesh.Localises_sol_e(resultat_ddl)
                resultat_e = resultat_e_n.mean(axis=1)

                if option == "amplitude":
                    return np.sqrt(np.sum(resultat_e**2, axis=1))
                elif option in ["speed", "accel"]:
                    return resultat_e
                else:
                    if len(resultat_ddl.shape) > 1:
                        return resultat_e[:,index]
                    else:
                        return resultat_e.reshape(-1)

    def __indexResulat(self, resultat: str) -> int:

        # "Beam1D" : ["ux" "fx"]
        # "Beam2D : ["ux","uy","rz""fx", "fy", "cz"]
        # "Beam3D" : ["ux", "uy", "uz", "rx", "ry", "rz" "fx","fy","fz","cx","cy"]

        dim = self.dim

        if len(resultat) <= 2:
            if "ux" in resultat or "fx" in resultat:
                return 0
            elif "uy" in resultat or "fy" in resultat:
                return 1
            elif "uz" in resultat or "fz" in resultat:
                return 2
            elif "rx" in resultat or "cx" in resultat:
                return 3
            elif "ry" in resultat or "cy" in resultat:
                return 4
            elif "rz" in resultat or "cz" in resultat:
                if dim == 2:
                    return 2
                else:
                    return 5

    def __Calc_Epsilon_e_pg(self, sol: np.ndarray, matriceType=MatriceType.rigi):
        """Construit les déformations pour chaque element et chaque points de gauss
        """
        
        tic = Tic()

        nbddl_n = self.beamModel.nbddl_n
        assemblyBeam_e = self.mesh.groupElem.Get_assembly_e(nbddl_n)
        sol_e = sol[assemblyBeam_e]
        B_beam_e_pg = self.__Get_B_beam_e_pg(matriceType)
        Epsilon_e_pg = np.einsum('epij,ej->epi', B_beam_e_pg, sol_e, optimize='optimal')
        
        tic.Tac("Matrices", "Epsilon_e_pg", False)

        return Epsilon_e_pg

    def __Calc_Sigma_e_pg(self, Epsilon_e_pg: np.ndarray, matriceType=MatriceType.rigi) -> np.ndarray:
        """Calcul les contraintes depuis les deformations
        """

        assert Epsilon_e_pg.shape[0] == self.mesh.Ne
        assert Epsilon_e_pg.shape[1] == self.mesh.Get_nPg(matriceType)

        tic = Tic()

        D_e_pg = self.beamModel.Calc_D_e_pg(self.mesh.groupElem, matriceType)
        Sigma_e_pg = np.einsum('epij,epj->epi', D_e_pg, Epsilon_e_pg, optimize='optimal')
            
        tic.Tac("Matrices", "Sigma_e_pg", False)

        return Sigma_e_pg

    def Resultats_Resume(self, verbosity=True) -> str:
        return super().Resultats_Resume(verbosity)

    def Resultats_Get_dict_Energie(self) -> list[tuple[str, float]]:
        return super().Resultats_Get_dict_Energie()

    def Resultats_Get_ResumeIter_values(self) -> list[tuple[str, np.ndarray]]:
        return super().Resultats_Get_ResumeIter_values()
    
    def Resultats_matrice_displacement(self) -> np.ndarray:
        
        Nn = self.mesh.Nn
        beamDisplacementRedim = self.beamDisplacement.reshape((Nn,-1))
        nbddl = self.Get_nbddl_n(self.problemType)

        coordo = np.zeros((Nn, 3))

        if nbddl == 1:
            coordo[:,0] = beamDisplacementRedim[:,0]
        elif nbddl == 3:
            coordo[:,:2] = beamDisplacementRedim[:,:2]
        elif nbddl == 6:
            coordo[:,:3] = beamDisplacementRedim[:,:3]

        return coordo

###################################################################################################

class Simu_Thermal(Simu):

    def __init__(self, mesh: Mesh, model: Thermal_Model, verbosity=False, useNumba=True):
        """Creation d'une simulation thermique"""

        assert model.modelType == ModelType.thermal, "Le matériau doit être de type thermal"
        super().__init__(mesh, model, verbosity, useNumba)

        # init
        self.Solver_Set_Elliptic_Algorithm()
    
    def Get_Directions(self, problemType=None) -> list[str]:
        return [""]

    def Get_Resultats_disponibles(self) -> list[str]:
        options = []
        options.extend(["thermal", "thermalDot"])
        return options

    def Paraview_nodesField_elementsField(self, details=False) -> tuple[list[str], list[str]]:
        nodesField = ["thermal", "thermalDot"]
        elementsField = []
        return nodesField, elementsField
    
    def Get_problemTypes(self) -> list[ModelType]:
        return [ModelType.thermal]
    
    def Get_nbddl_n(self, problemType=None) -> int:
        return 1

    @property
    def thermalModel(self) -> Thermal_Model:
        """Modèle thermique de la simulation"""
        return self.model

    @property
    def thermal(self) -> np.ndarray:
        """Copie du champ scalaire de température"""
        return self.get_u_n(self.problemType)

    @property
    def thermalDot(self) -> np.ndarray:
        """Copie de la dérivée du champ scalaire de température"""
        return self.get_v_n(self.problemType)

    def Get_x0(self, problemType=None):
        if self.thermal.size != self.mesh.Nn:
            return np.zeros(self.mesh.Nn)
        else:
            return self.thermal

    def Get_K_C_M_F(self, problemType=None) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        if self.needUpdate: self.Assemblage()
        taille = self.mesh.Nn * self.Get_nbddl_n(problemType)
        initcsr = sparse.csr_matrix((taille, taille))
        return self.__Kt.copy(), self.__Ct.copy(), initcsr, self.__Ft.copy()

    def __ConstruitMatElem_Thermal(self):

        thermalModel = self.thermalModel

        # Data
        k = thermalModel.k

        matriceType=MatriceType.rigi

        mesh = self.mesh

        jacobien_e_pg = mesh.Get_jacobien_e_pg(matriceType)
        poid_pg = mesh.Get_poid_pg(matriceType)
        N_e_pg = mesh.Get_N_scalaire_pg(matriceType)
        D_e_pg = mesh.Get_dN_sclaire_e_pg(matriceType)

        Kt_e = np.einsum('ep,p,epji,,epjk->eik', jacobien_e_pg, poid_pg, D_e_pg, k, D_e_pg, optimize="optimal")

        rho = self.rho
        c = thermalModel.c

        Mt_e = np.einsum('ep,p,pji,,,pjk->eik', jacobien_e_pg, poid_pg, N_e_pg, rho, c, N_e_pg, optimize="optimal")

        return Kt_e, Mt_e

    def Assemblage(self):
        """Construit du systeme matricielle pour le problème thermique en régime stationnaire ou transitoire
        """

        if self.needUpdate:
       
            # Data
            mesh = self.mesh
            taille = mesh.Nn
            lignesScalar_e = mesh.lignesScalar_e
            colonnesScalar_e = mesh.colonnesScalar_e

            # Dimension supplémentaire lié a l'utilisation des coefs de lagrange
            dimSupl = len(self.Bc_Lagrange)
            if dimSupl > 0:
                dimSupl += len(self.Bc_ddls_Dirichlet(ModelType.thermal))
                taille += dimSupl
            
            # Calul les matrices elementaires
            Kt_e, Mt_e = self.__ConstruitMatElem_Thermal()

            # Assemblage
            tic = Tic()

            self.__Kt = sparse.csr_matrix((Kt_e.reshape(-1), (lignesScalar_e.reshape(-1), colonnesScalar_e.reshape(-1))), shape = (taille, taille))
            """Kglob pour le problème thermique (Nn, Nn)"""
            
            self.__Ft = sparse.csr_matrix((taille, 1))
            """Vecteur Fglob pour le problème en thermique (Nn, 1)"""

            self.__Ct = sparse.csr_matrix((Mt_e.reshape(-1), (lignesScalar_e.reshape(-1), colonnesScalar_e.reshape(-1))), shape = (taille, taille))
            """Mglob pour le problème thermique (Nn, Nn)"""

            tic.Tac("Matrices","Assemblage Kt, Mt et Ft", self._verbosity)

            self.Need_Update()

    def Save_Iteration(self):

        iter = super().Save_Iteration()
        
        iter['thermal'] = self.thermal

        if self.algo == AlgoType.parabolic:
            iter['thermalDot'] = self.thermalDot
            
        self._results.append(iter)

    def Update_iter(self, index=-1):
        
        results = super().Update_iter(index)

        if results == None: return

        self.set_u_n(ModelType.thermal, results["thermal"])

        if self.algo == AlgoType.parabolic and "thermalDot" in results:
            self.set_v_n(ModelType.thermal, results["thermalDot"])
        else:
            self.set_v_n(ModelType.thermal, np.zeros_like(self.thermal))

    def Get_Resultat(self, option: str, nodeValues=True, iter=None):
        
        if not self.Resultats_Check_Options_disponibles(option): return None

        if iter != None:
            self.Update_iter(iter)

        if option == "thermal":
            return self.thermal

        if option == "thermalDot":
            return self.thermalDot

    def Resultats_Get_ResumeIter_values(self) -> list[tuple[str, np.ndarray]]:
        return super().Resultats_Get_ResumeIter_values()
    
    def Resultats_Resume(self, verbosity=True) -> str:
        return super().Resultats_Resume(verbosity)

    def Resultats_Get_dict_Energie(self) -> list[tuple[str, float]]:
        return super().Resultats_Get_dict_Energie()
    
    def Resultats_matrice_displacement(self) -> np.ndarray:
        Nn = self.mesh.Nn
        return np.array((Nn,3))