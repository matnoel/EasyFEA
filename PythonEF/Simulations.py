from abc import ABC, abstractmethod
from types import LambdaType
from typing import List, cast
import numpy as np
from scipy import sparse

# Meme si pas utilisé laissé l'acces
from Mesh import Mesh, MatriceType
from BoundaryCondition import BoundaryCondition, LagrangeCondition
from Materials import ModelType, _Materiau, Beam_Model, _Materiau_Beam, _Materiau_Displacement, _Materiau_PhaseField, _Materiau_Thermal
from TicTac import Tic
import CalcNumba
import Interface_Solveurs

def Create_Simu(mesh: Mesh, materiau: _Materiau, verbosity=True, useNumba=True):

    params = (mesh, materiau, verbosity ,useNumba)

    if materiau.modelType == ModelType.displacement:
        simu = __Simu_Displacement(*params)
    elif materiau.modelType == ModelType.beam:
        simu = __Simu_Beam(*params)
    elif materiau.modelType == ModelType.damage:
        simu = __Simu_PhaseField(*params)
    elif materiau.modelType == ModelType.thermal:
        simu = __Simu_Thermal(*params)
    else:
        raise "Modèle physique inconnue pour la création d'une simulation"

    return simu

class _Simu(ABC):
    """Classe mère de :\n
     - Simu_Displacement
     - Simu_Damage
     - Simu_Beam
     - Simu_Thermal"""

    @staticmethod
    def Check_ProblemTypes(problemType : ModelType):
        """Verifie si ce type de probleme connue"""
        list_problemType = list(ModelType)
        assert problemType in list_problemType, "Ce type de probleme n'est pas implémenté"

    @abstractmethod
    def Check_Directions(self, problemType : ModelType, directions:list):
        """Verifie si les directions renseignées sont possible pour le probleme"""
        pass
    
    def Check_dim_mesh_materiau(self) -> None:
        """On verifie que la dimension du materiau correspond a la dimension du maillage"""
        assert self.materiau.dim == self.mesh.dim, "Le materiau doit avoir la meme dimension que le maillage"

    def __init__(self, mesh: Mesh, materiau: _Materiau, verbosity=True, useNumba=True):
        """Creation d'une simulation

        Args:
            dim (int): Dimension de la simulation (2D ou 3D)
            mesh (Mesh): Maillage que la simulation va utiliser
            materiau (Materiau): Materiau utilisé
            verbosity (bool, optional): La simulation ecrira dans la console. Defaults to True.
        """
        
        if verbosity:
            import Affichage
            Affichage.NouvelleSection("Simulation")

        # Renseigne le premier maillage
        self.__iterMesh = -1
        self.__listMesh = cast(list[Mesh], [])
        self.mesh = mesh

        # Renseigne le matériau, Il n'est pas possible de changer de matériau
        self.__materiau = materiau
        """materiau de la simulation"""
        self.Check_dim_mesh_materiau()

        self.__dim = materiau.dim
        """dimension de la simulation 2D ou 3D"""
        self._verbosity = verbosity
        """la simulation peut ecrire dans la console"""
        
        self.__algo = Interface_Solveurs.AlgoType.elliptic

        self.useNumba = useNumba

        # Conditions Limites
        self.Bc_Init()

    # TODO Permettre de creer des simulation depuis le formulation variationnelle ?

    @property
    def problemType(self) -> ModelType:
        """problème de la simulation"""
        return self.materiau.modelType

    @property
    def algo(self) -> Interface_Solveurs.AlgoType:
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
            # Pour tout les anciens maillage j'efface les matrices
            listMesh = cast(List[Mesh], self.__listMesh)
            [m.ResetMatrices() for m in listMesh]

            self.__iterMesh += 1
            self.__listMesh.append(mesh)
            self.__mesh = mesh

    @property
    def dim(self) -> int:
        """dimension de la simulation"""
        return self.__dim
    
    @abstractmethod
    def nbddl_n(self, problemType="") -> int:
        """degrés de libertés par noeud"""
        pass

    @property
    def use3DBeamModel(self) -> bool:
        return False

    @property
    def useNumba(self) -> bool:
        return self.__useNumba
    
    @useNumba.setter
    def useNumba(self, value: bool):
        self.materiau.useNumba = value
        self.__useNumba = value

    # Fonctions à redefinir pour chaque heritié de simu

    @property
    @abstractmethod
    def materiau(self) -> _Materiau:
        """Matériau de la simulation"""
        return self.__materiau

    @property
    @abstractmethod    
    def results(self) -> List[dict]:
        """Renvoie la liste de dictionnaire qui stocke les résultats
        """
        pass

    @abstractmethod
    def Assemblage(self):
        """Assemblage de la simulation"""
        pass

    @abstractmethod
    def Solve(self):
        """Résolution de la simulation"""
        pass

    @abstractmethod
    def Save_Iteration(self, nombreIter=None, tempsIter=None, dincMax=None) -> dict:
        """Sauvegarde les résultats de l'itération

        Parameters
        ----------
        nombreIter : int, optional
            nombre d'itération pour arriver à la convergence, by default None
        tempsIter : float, optional
            temps nécessaire à une itération, by default None
        dincMax : float, optional
            tolérance de convergence d'endommagement, by default None
        """

        iter = {}

        iter["iterMesh"] = self.__iterMesh

        if nombreIter != None and tempsIter != None and dincMax != None:
            iter["nombreIter"] = nombreIter
            iter["tempsIter"] = tempsIter
            iter["dincMax"] = dincMax

        return iter
    
    @abstractmethod
    def Update_iter(self, index=-1) -> dict:
        """Met la simulation à l'iteration renseignée"""
        index = int(index)
        assert isinstance(index, int), print("Doit fournir un entier")

        indexMax = len(self.results)-1
        assert index <= indexMax, f"L'index doit etre < {indexMax}]"

        # On va venir récupérer les resultats stocké dans le tableau pandas
        results =  self.results[index]

        self.__Update_mesh(index)

        return results

    def __Update_mesh(self, index: int):
        """Met à jour le maillage à l'index renseigné"""
        iterMesh = self.results[index]["iterMesh"]
        self.__mesh = self.__listMesh[iterMesh]

    @abstractmethod
    def Get_Resultat(self, option: str, valeursAuxNoeuds=True, iter=None):
        """ Renvoie le résultat de la simulation
        """
        pass
    
    # ------------------------------------------------- SOLVEUR -------------------------------------------------
    # Fonctions pour l'interface avec le solveur
        
    def Solveur(self, problemType : ModelType, algo: Interface_Solveurs.AlgoType) -> np.ndarray:
        """Resolution du de la simulation et renvoie la solution\n
        Prépare dans un premier temps A et b pour résoudre Ax=b\n
        On va venir appliquer les conditions limites pour résoudre le système"""
        # ici il faut specifier le type de probleme car une simulation peut posséder plusieurs Modèle physique

        resolution = 1

        self.__algo = algo

        if len(self.__Bc_Lagrange) > 0:
            resolution = 2
        
        if resolution == 1:
            
            x = Interface_Solveurs.Solveur_1(self, problemType)

        elif resolution == 2:

            x = Interface_Solveurs.Solveur_2(self, problemType)

        elif resolution == 3:
            
            x = Interface_Solveurs.Solveur_3(self, problemType)

        return x

    def Solveur_Parabolic_Properties(self, dt=0.1, alpha=1/2):
        """Renseigne les propriétes de résolution de l'algorithme

        Parameters
        ----------
        alpha : float, optional
            critère alpha [0 -> Forward Euler, 1 -> Backward Euler, 1/2 -> midpoint], by default 1/2
        dt : float, optional
            incrément temporel, by default 0.1
        """

        # assert alpha >= 0 and alpha <= 1, "alpha doit être compris entre [0, 1]"
        # Est-il possible davoir au dela de 1 ?

        assert dt > 0, "l'incrément temporel doit être > 0"

        self.alpha = alpha
        self.dt = dt

    def Solveur_Newton_Raphson_Properties(self, betha=1/4, gamma=1/2, dt=0.1):
        """Renseigne les propriétes de résolution de l'algorithme

        Parameters
        ----------
        betha : float, optional
            coef betha, by default 1/4
        gamma : float, optional
            coef gamma, by default 1/2
        dt : float, optional
            incrément temporel, by default 0.1
        """

        # assert alpha >= 0 and alpha <= 1, "alpha doit être compris entre [0, 1]"
        # Est-il possible davoir au dela de 1 ?

        assert dt > 0, "l'incrément temporel doit être > 0"

        self.betha = betha
        self.gamma = gamma
        self.dt = dt

    # ------------------------------------------- CONDITIONS LIMITES -------------------------------------------
    # Fonctions pour le renseignement des conditions limites de la simulation

    @staticmethod
    def __Bc_Init_List_BoundaryCondition() -> list[BoundaryCondition]:
        return []

    @staticmethod
    def __Bc_Init_List_LagrangeCondition() -> list[LagrangeCondition]:
        return []
    
    def Bc_Init(self):
        """Initie les conditions limites de dirichlet, Neumann et Lagrange"""
        # DIRICHLET
        self.__Bc_Dirichlet = _Simu.__Bc_Init_List_BoundaryCondition()
        """Conditions de Dirichlet list(BoundaryCondition)"""
        # NEUMANN
        self.__Bc_Neumann = _Simu.__Bc_Init_List_BoundaryCondition()
        """Conditions de Neumann list(BoundaryCondition)"""
        # LAGRANGE
        self.__Bc_Lagrange = _Simu.__Bc_Init_List_LagrangeCondition()
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
    
    @property
    def Bc_LagrangeAffichage(self) -> list[LagrangeCondition]:
        """Renvoie une copie des conditions de Lagrange pour l'affichage"""
        return self.__Bc_LagrangeAffichage.copy()

    def Bc_ddls_Dirichlet(self, problemType=None) -> list[int]:
        """Renvoie les ddls liés aux conditions de Dirichlet"""
        if problemType == None:
            problemType = self.problemType
        return BoundaryCondition.Get_ddls(problemType, self.__Bc_Dirichlet)

    def BC_values_Dirichlet(self, problemType=None) -> list:
        """Renvoie les valeurs ddls liés aux conditions de Dirichlet"""
        if problemType == None:
            problemType = self.problemType
        return BoundaryCondition.Get_values(problemType, self.__Bc_Dirichlet)
    
    def Bc_ddls_Neumann(self, problemType=None) -> list:
        """Renvoie les ddls liés aux conditions de Neumann"""
        if problemType == None:
            problemType = self.problemType
        return BoundaryCondition.Get_ddls(problemType, self.__Bc_Neumann)
    
    def Bc_values_Neumann(self, problemType=None) -> list:
        """Renvoie les valeurs ddls liés aux conditions de Neumann"""
        if problemType == None:
            problemType = self.problemType
        return BoundaryCondition.Get_values(problemType, self.__Bc_Neumann)

    def Bc_ddls_Lagrange(self, problemType=None) -> list:
        """Renvoie les ddls liés aux conditions de Lagrange"""
        if problemType == None:
            problemType = self.problemType
        return BoundaryCondition.Get_ddls(problemType, self.__Bc_Lagrange)
    
    def Bc_values_Lagrange(self, problemType=None) -> list:
        """Renvoie les valeurs ddls liés aux conditions de Lagrange"""
        if problemType == None:
            problemType = self.problemType
        return BoundaryCondition.Get_values(problemType, self.__Bc_Lagrange)

    def __Bc_evalue(self, coordo: np.ndarray, valeurs, option="noeuds"):
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
                raise "Doit fournir une fonction lambda de la forme\n lambda x,y,z, : f(x,y,z)"
        else:            
            if option == "noeuds":
                valeurs_eval[:] = valeurs
            elif option == "gauss":
                valeurs_eval[:,:] = valeurs

        return valeurs_eval
    
    def add_dirichlet(self, noeuds: np.ndarray, valeurs: np.ndarray, directions: list, problemType=None, description=""):

        if len(valeurs) == 0 or len(valeurs) != len(directions): return

        if problemType == None:
            problemType = self.problemType

        _Simu.Check_ProblemTypes(problemType)
        
        Nn = noeuds.shape[0]
        coordo = self.mesh.coordo
        coordo_n = coordo[noeuds]

        # initialise le vecteur de valeurs pour chaque noeuds
        valeurs_ddl_dir = np.zeros((Nn, len(directions)))

        for d, dir in enumerate(directions):
            eval_n = self.__Bc_evalue(coordo_n, valeurs[d], option="noeuds")
            valeurs_ddl_dir[:,d] = eval_n.reshape(-1)
        
        valeurs_ddls = valeurs_ddl_dir.reshape(-1)

        nbddl_n = self.nbddl_n(problemType)

        ddls = BoundaryCondition.Get_ddls_noeuds(nbddl_n, problemType, noeuds, directions)

        self.__Bc_Add_Dirichlet(problemType, noeuds, valeurs_ddls, ddls, directions, description)

    def add_pointLoad(self, noeuds: np.ndarray, valeurs: list, directions: list, problemType=None, description=""):
        """Pour le probleme donné applique une force ponctuelle\n
        valeurs est une liste de constantes ou de fonctions\n
        ex: valeurs = [lambda x,y,z : f(x,y,z) ou -10]

        les fonctions doivent être de la forme lambda x,y,z : f(x,y,z)\n
        les fonctions utilisent les coordonnées x, y et z des noeuds renseignés
        """
        
        if len(valeurs) == 0 or len(valeurs) != len(directions): return

        if problemType == None:
            problemType = self.problemType

        _Simu.Check_ProblemTypes(problemType)

        valeurs_ddls, ddls = self.__Bc_pointLoad(problemType, noeuds, valeurs, directions)

        self.__Bc_Add_Neumann(problemType, noeuds, valeurs_ddls, ddls, directions, description)
        
    def add_lineLoad(self, noeuds: np.ndarray, valeurs: list, directions: list, problemType=None, description=""):
        """Pour le probleme donné applique une force linéique\n
        valeurs est une liste de constantes ou de fonctions\n
        ex: valeurs = [lambda x,y,z : f(x,y,z) ou -10]

        les fonctions doivent être de la forme lambda x,y,z : f(x,y,z)\n
        les fonctions utilisent les coordonnées x, y et z des points d'intégrations
        """
        if len(valeurs) == 0 or len(valeurs) != len(directions): return

        if problemType == None:
            problemType = self.problemType

        _Simu.Check_ProblemTypes(problemType)

        valeurs_ddls, ddls = self.__Bc_lineLoad(problemType, noeuds, valeurs, directions)

        self.__Bc_Add_Neumann(problemType, noeuds, valeurs_ddls, ddls, directions, description)

    def add_surfLoad(self, noeuds: np.ndarray, valeurs: list, directions: list, problemType=None, description=""):
        """Pour le probleme donné applique une force surfacique\n
        valeurs est une liste de constantes ou de fonctions\n
        ex: valeurs = [lambda x,y,z : f(x,y,z) ou -10]

        les fonctions doivent être de la forme lambda x,y,z : f(x,y,z)\n
        les fonctions utilisent les coordonnées x, y et z des points d'intégrations\n
        Si probleme poutre on integre sur la section de la poutre
        """
        if len(valeurs) == 0 or len(valeurs) != len(directions): return

        if problemType == None:
            problemType = self.problemType

        _Simu.Check_ProblemTypes(problemType)

        if problemType == ModelType.beam:
            # valeurs_ddls, ddls = self.__Bc_pointLoad(problemType, noeuds, valeurs, directions)
            # # multiplie par la surface de la section
            # # ici il peut y avoir un probleme si il ya plusieurs poutres et donc des sections différentes
            # valeurs_ddls *= self.materiau.beamModel.poutre.section.aire
            print("Il est impossible d'appliquer une charge surfacique sur un probleme poutre")
            return
            
        if self.__dim == 2:
            valeurs_ddls, ddls = self.__Bc_lineLoad(problemType, noeuds, valeurs, directions)
            # multiplie par l'epaisseur
            valeurs_ddls *= self.materiau.epaisseur
        elif self.__dim == 3:
            valeurs_ddls, ddls = self.__Bc_surfload(problemType, noeuds, valeurs, directions)

        self.__Bc_Add_Neumann(problemType, noeuds, valeurs_ddls, ddls, directions, description)

    def add_volumeLoad(self, noeuds: np.ndarray, valeurs: list, directions: list, problemType=None, description=""):
        """Pour le probleme donné applique une force volumique\n
        valeurs est une liste de constantes ou de fonctions\n
        ex: valeurs = [lambda x,y,z : f(x,y,z) ou -10]

        les fonctions doivent être de la forme lambda x,y,z : f(x,y,z)\n
        les fonctions utilisent les coordonnées x, y et z des points d'intégrations
        """
        
        if len(valeurs) == 0 or len(valeurs) != len(directions): return

        if problemType == None:
            problemType = self.problemType

        _Simu.Check_ProblemTypes(problemType)

        if problemType == ModelType.beam:
            # valeurs_ddls, ddls = self.__Bc_lineLoad(problemType, noeuds, valeurs, directions)
            # # multiplie par la surface de la section
            # # ici il peut y avoir un probleme si il ya plusieurs poutres et donc des sections différentes
            # valeurs_ddls *= self.materiau.beamModel.poutre.section.aire
            print("Il est impossible d'appliquer une charge volumique sur un probleme poutre")
            return

        if self.__dim == 2:
            valeurs_ddls, ddls = self.__Bc_surfload(problemType, noeuds, valeurs, directions)
            # multiplie par l'epaisseur
            valeurs_ddls = valeurs_ddls*self.materiau.epaisseur
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

        nbddl_n = self.nbddl_n(problemType)

        ddls = BoundaryCondition.Get_ddls_noeuds(nbddl_n, problemType, noeuds, directions)

        return valeurs_ddls, ddls

    def __Bc_IntegrationDim(self, dim: int, problemType : ModelType, noeuds: np.ndarray, valeurs: list, directions: list):
        """Intégration des valeurs sur les elements"""

        valeurs_ddls=np.array([])
        ddls=np.array([], dtype=int)

        listGroupElemDim = self.mesh.Get_list_groupElem(dim)

        if len(listGroupElemDim) > 1:
            exclusivement=True
        else:
            exclusivement=True

        nbddl_n = self.nbddl_n(problemType)

        # Récupération des matrices pour le calcul
        for groupElem in listGroupElemDim:

            # Récupère les elements qui utilisent exclusivement les noeuds
            elements = groupElem.get_elementsIndex(noeuds, exclusivement=exclusivement)
            if elements.shape[0] == 0: continue
            connect_e = groupElem.connect_e[elements]
            Ne = elements.shape[0]
            
            # récupère les coordonnées des points de gauss dans le cas ou on a besoin dévaluer la fonction
            matriceType = MatriceType.masse
            coordo_e_p = groupElem.get_coordo_e_p(matriceType,elements)
            nPg = coordo_e_p.shape[1]

            N_pg = groupElem.get_N_pg(matriceType)

            # objets d'integration
            jacobien_e_pg = groupElem.get_jacobien_e_pg(matriceType)[elements]
            gauss = groupElem.get_gauss(matriceType)
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
        
        _Simu.Check_Directions(self.__dim, problemType, directions)

        valeurs_ddls, ddls = self.__Bc_IntegrationDim(dim=1, problemType=problemType, noeuds=noeuds, valeurs=valeurs, directions=directions)

        return valeurs_ddls, ddls
    
    def __Bc_surfload(self, problemType : ModelType, noeuds: np.ndarray, valeurs: list, directions: list):
        """Applique une force surfacique\n
        Renvoie valeurs_ddls, ddls"""
        
        _Simu.Check_Directions(self.__dim, problemType, directions)

        valeurs_ddls, ddls = self.__Bc_IntegrationDim(dim=2, problemType=problemType, noeuds=noeuds, valeurs=valeurs, directions=directions)

        return valeurs_ddls, ddls

    def __Bc_volumeload(self, problemType : ModelType, noeuds: np.ndarray, valeurs: list, directions: list):
        """Applique une force surfacique\n
        Renvoie valeurs_ddls, ddls"""
        
        _Simu.Check_Directions(self.__dim, problemType, directions)

        valeurs_ddls, ddls = self.__Bc_IntegrationDim(dim=3, problemType=problemType, noeuds=noeuds, valeurs=valeurs, directions=directions)

        return valeurs_ddls, ddls
    
    def __Bc_Add_Neumann(self, problemType : ModelType, noeuds: np.ndarray, valeurs_ddls: np.ndarray, ddls: np.ndarray, directions: list, description=""):
        """Ajoute les conditions de Neumann"""

        tic = Tic()

        _Simu.Check_Directions(self.__dim, problemType, directions)

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

        _Simu.Check_ProblemTypes(problemType)

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

    # TODO à passer dans __Simu_beam

    def add_liaison_Encastrement(self, noeuds: np.ndarray, description="Encastrement"):
        
        if not isinstance(self.materiau, _Materiau_Beam):
            print("La simulation n'est pas un probleme poutre")
            return

        beamModel = self.materiau.beamModel        

        if beamModel.dim == 1:
            directions = ['x']
        elif beamModel.dim == 2:
            directions = ['x','y','rz']
        elif beamModel.dim == 3:
            directions = ['x','y','z','rx','ry','rz']

        description = f"Liaison {description}"
        
        self.add_liaisonPoutre(noeuds, directions, description)

    def add_liaison_Rotule(self, noeuds: np.ndarray, directions=[''] ,description="Rotule"):
        
        if not isinstance(self.materiau, _Materiau_Beam):
            print("La simulation n'est pas un probleme poutre")
            return

        beamModel = self.materiau.beamModel
        
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

    def add_liaisonPoutre(self, noeuds: np.ndarray, directions: List[str], description: str):
        
        if not isinstance(self.materiau, _Materiau_Beam):
            print("La simulation n'est pas un probleme poutre")
            return

        nbddl_n = self.nbddl_n()
        problemType = self.problemType

        # Verficiation
        _Simu.Check_Directions(self.__dim, problemType, directions)

        tic = Tic()

        # On va venir pour chaque directions appliquer les conditions
        for d, dir in enumerate(directions):
            ddls = BoundaryCondition.Get_ddls_noeuds(nbddl_n,  problemType=problemType, noeuds=noeuds, directions=[dir])

            new_LagrangeBc = LagrangeCondition(problemType, noeuds, ddls, [dir], [0], [1,-1], description)

            self.__Bc_Lagrange.append(new_LagrangeBc)
        
        # Il n'est pas possible de poser u1-u2 + v1-v2 = 0
        # Il faut appliquer une seule condition à la fois

        tic.Tac("Boundary Conditions","Liaison", self._verbosity)

        self.__Bc_Add_LagrangeAffichage(noeuds, directions, description)

    def __Bc_Add_LagrangeAffichage(self,noeuds: np.ndarray, directions: List[str], description: str):
        
        # Ajoute une condition pour l'affichage
        nbddl = self.nbddl_n()
        
        # Prend le premier noeuds de la liaison
        noeuds1 = np.array([noeuds[0]])

        ddls = BoundaryCondition.Get_ddls_noeuds(param=nbddl,  problemType=ModelType.beam, noeuds=noeuds1, directions=directions)
        valeurs_ddls =  np.array([0]*len(ddls))

        new_Bc = BoundaryCondition(ModelType.beam, noeuds1, ddls, directions, valeurs_ddls, description)
        self.__Bc_LagrangeAffichage.append(new_Bc)
    
    # ------------------------------------------- POST TRAITEMENT ------------------------------------------- 
    
    @staticmethod
    def Resultats_indexe_option(dim: int, problemType : ModelType,  option: str):
        """Donne l'indexe pour accéder à l'option en fonction du type de problème"""
        
        if problemType in [ModelType.displacement,ModelType.damage]:
            # "Stress" : ["Sxx", "Syy", "Sxy"]
            # "Stress" : ["Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy"]
            # "Accel" : ["vx", "vy", "vz", "ax", "ay", "az"]

            if option in ["x","fx","dx","vx","ax"]:
                return 0
            elif option in ["y","fy","dy","vy","ay"]:
                return 1
            elif option in ["z","fz","dz","vz","az"]:
                return 2
            elif option in ["Sxx","Exx"]:
                return 0
            elif option in ["Syy","Eyy"]:
                return 1
            elif option in ["Sxy","Exy"]:
                if dim == 2:
                    return 2
                elif dim == 3:
                    return 5
            elif option in ["Syy","Eyy"]:
                return 1
            elif option in ["Szz","Ezz"]:
                return 2
            elif option in ["Syz","Eyz"]:
                return 3
            elif option in ["Sxz","Exz"]:
                return 4
        elif problemType == ModelType.beam:

            # "Beam1D" : ["u" "fx"]
            # "Beam2D : ["u","v","rz""fx", "fy", "cz"]
            # "Beam3D" : ["u", "v", "w", "rx", "ry", "rz" "fx","fy","fz","cx","cy"]

            if option in ["u","fx"]:
                return 0
            elif option in ["v","fy"]:
                return 1
            elif option in ["w","fz"]:
                return 2
            elif option in ["rx"]:
                return 3
            elif option in ["ry"]:
                return 4
            elif option in ["rz"]:
                if dim == 2: 
                    return 2
                elif dim == 3: 
                    return 5

    @staticmethod
    def __Resultats_Categories(dim: int) -> dict:
        if dim == 1:
            options = {
                "Beam" : ["u", "beamDisplacement", "coordoDef"],
                "Beam Load" : ["fx","Srain","Stress"]
            }
        elif dim == 2:
            options = {
                "Thermal" : ["thermal", "thermalDot"],
                "Displacement" : ["dx", "dy", "dz", "amplitude", "displacement", "coordoDef"],
                "Accel" : ["ax", "ay", "accel", "amplitudeAccel"],
                "Speed" : ["vx", "vy", "speed", "amplitudeSpeed"],
                "Stress" : ["Sxx", "Syy", "Sxy", "Svm","Stress"],
                "Strain" : ["Exx", "Eyy", "Exy", "Evm","Strain"],
                "Beam" : ["u","v","rz", "amplitude", "beamDisplacement", "coordoDef"],
                "Beam Load" : ["fx", "fy", "cz", "Exx", "Exy", "Sxx", "Sxy", "Srain", "Stress"],
                "Energie" :["Wdef","Psi_Elas"],
                "Damage" :["damage","psiP","Psi_Crack"]
            }
        elif dim == 3:
            options = {
                "Thermal" : ["thermal", "thermalDot"],
                "Displacement" : ["dx", "dy", "dz","amplitude","displacement", "coordoDef"],
                "Accel" : ["ax", "ay", "az", "accel", "amplitudeAccel"],
                "Speed" : ["vx", "vy", "vz", "speed","amplitudeSpeed"],
                "Stress" : ["Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy", "Svm","Stress"],
                "Strain" : ["Exx", "Eyy", "Ezz", "Eyz", "Exz", "Exy", "Evm","Strain"],
                "Beam" : ["u", "v", "w", "rx", "ry", "rz", "amplitude", "beamDisplacement", "coordoDef"],
                "Beam Load" : ["fx","fy","fz","cx","cy","cz","Srain","Stress"],
                "Energie" :["Wdef","Psi_Elas"]
            }
        return options

    @staticmethod
    def __Resultats_Options(dim: int, problemType : ModelType) -> List[str]:
        """Donne la liste la liste de résultats auxquelles la simulation a accès"""

        categories = _Simu.__Resultats_Categories(dim)
        
        if problemType == ModelType.displacement:
            keys = ["Stress", "Strain", "Displacement", "Accel", "Speed", "Energie"]
        elif problemType == ModelType.damage:
            keys = ["Stress", "Strain", "Displacement", "Energie", "Damage"]
        elif problemType == ModelType.beam:
            keys = ["Beam", "Beam Load"]
        elif problemType == ModelType.thermal:
            keys = ["Thermal"]

        listOption = []
        [listOption.extend(categories[key]) for key in keys]

        return listOption

    def Resultats_Verification(self, option) -> bool:
        """Verification que l'option est bien calculable
        """
        # Construit la liste d'otions pour les résultats en 2D ou 3D
        
        listOptions = self.__Resultats_Options(self.dim, self.problemType)
        if option not in listOptions:
            print(f"\nPour un probleme ({self.problemType}) l'option doit etre dans : \n {listOptions}")
            return False
        else:
            return True
    
    def Resultats_InterpolationAuxNoeuds(self, resultat_e: np.ndarray):
        """Pour chaque noeuds on récupère les valeurs des élements autour de lui pour on en fait la moyenne
        """

        tic = Tic()

        Ne = self.__mesh.Ne
        Nn = self.__mesh.Nn

        if len(resultat_e.shape) == 1:
            resultat_e = resultat_e.reshape(Ne,1)
            isDim1 = True
        else:
            isDim1 = False
        
        Ncolonnes = resultat_e.shape[1]

        resultat_n = np.zeros((Nn, Ncolonnes))

        for c in range(Ncolonnes):

            valeurs_e = resultat_e[:, c]

            connect_n_e = self.__mesh.connect_n_e
            nombreApparition = np.array(np.sum(connect_n_e, axis=1)).reshape(self.__mesh.Nn,1)
            valeurs_n_e = connect_n_e.dot(valeurs_e.reshape(self.__mesh.Ne,1))
            valeurs_n = valeurs_n_e/nombreApparition

            resultat_n[:,c] = valeurs_n.reshape(-1)

        tic.Tac("Post Traitement","Interpolation aux noeuds", False)

        if isDim1:
            return resultat_n.reshape(-1)
        else:
            return resultat_n

    @abstractmethod
    def Resultats_Resume(self, verbosity=True) -> str:

        import Affichage

        resume = Affichage.NouvelleSection("Maillage", verbosity)
        resume += self.mesh.Resume(verbosity)

        resume += Affichage.NouvelleSection("Materiau", verbosity)
        resume += '\n' + self.materiau.Get_Resume(verbosity)

        resume += Affichage.NouvelleSection("Temps", verbosity)
        resume += Tic.getResume(verbosity)

        return resume
    
    def Resultats_GetCoordUglob(self):
        """Renvoie les déplacements sous la forme [dx, dy, dz] (Nn,3)        """

        Nn = self.__mesh.Nn
        dim = self.__dim
        problemType = self.problemType

        if problemType in [ModelType.displacement,ModelType.beam,ModelType.damage]:

            if problemType in [ModelType.displacement,ModelType.damage]:
                Uglob = self.displacement
            else:
                Uglob = self.beamDisplacement
                listDim = np.arange(dim)

            coordo = Uglob.reshape((Nn,-1))
           
            if problemType in [ModelType.displacement,ModelType.damage]:
                if dim == 2:
                    coordo = np.append(coordo, np.zeros((Nn,1)), axis=1)
            else:
                coordo = coordo[:, listDim]

                if self.dim == 1:
                    # Ici on rajoute deux colonnes
                    coordo = np.append(coordo, np.zeros((Nn,1)), axis=1)
                    coordo = np.append(coordo, np.zeros((Nn,1)), axis=1)
                elif self.dim == 2:
                    # Ici on rajoute 1 colonne
                    coordo = np.append(coordo, np.zeros((Nn,1)), axis=1)

            return coordo
        else:
            return None

###################################################################################################

class __Simu_Displacement(_Simu):

    __dict_dim_directions = {
        2 : ["x", "y"],
        3 : ["x", "y", "z"]
    }

    def Check_Directions(self, problemType: ModelType, directions: list):
        listDirections = __Simu_Displacement.__dict_dim_directions[self.dim]
        for d in directions: assert d in listDirections, f"{d} doit être dans [{listDirections}]"

    def __init__(self, mesh: Mesh, materiau: _Materiau_Displacement, verbosity=True, useNumba=True):
        """Creation d'une simulation de déplacement"""
        assert materiau.modelType == ModelType.displacement, "Le materiau doit être de type displacement"
        super().__init__(mesh, materiau, verbosity, useNumba)

        # resultats
        self.__init_results()
    
    def __init_results(self):

        self.__displacement = np.zeros(self.mesh.Nn*self.dim)

        if self.materiau.ro > 0:
            self.Set_Rayleigh_Damping_Coefs()
            self.__speed = np.zeros_like(self.__displacement)                
            self.accel = np.zeros_like(self.__displacement)

        self.__results = [] #liste de dictionnaire qui contient les résultats

        self.Solveur_Parabolic_Properties() # Renseigne les propriétes de résolution de l'algorithme
        self.Solveur_Newton_Raphson_Properties() # Renseigne les propriétes de résolution de l'algorithme
    
    def nbddl_n(self, problemType="") -> int:
        return self.dim

    @property
    def materiau(self) -> _Materiau_Displacement:
        return super().materiau

    @property
    def results(self) -> List[dict]:
        return self.__results

    @property
    def displacement(self) -> np.ndarray:
        """Copie du champ vectoriel de déplacement"""
        return self.__displacement.copy()

    @property
    def speed(self) -> np.ndarray:
        """Copie du champ vectoriel de vitesse"""
        if self.materiau.ro > 0:
            return self.__speed.copy()
        else:
            return None
    
    # @property
    # def accel(self) -> np.ndarray:
    #     """Copie du champ vectoriel d'accéleration"""
    #     if self.problemType in [ProblemType.displacement, ProblemType.damage] and self.materiau.ro:
    #         return self.accel.copy()
    #     else:
    #         return None
    

    def ConstruitMatElem_Dep(self, steadyState=True) -> np.ndarray:
        """Construit les matrices de rigidités élementaires pour le problème en déplacement
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

        comportement = self.materiau.comportement

        tic = Tic()

        # Ici on le materiau est homogène
        matC = comportement.get_C()

        if useNumba:
            # Plus rapide
            Ku_e = CalcNumba.epij_jk_epkl_to_eil(leftDepPart, matC, B_dep_e_pg)
        else:
            # Ku_e = np.einsum('ep,p,epki,kl,eplj->eij', jacobien_e_pg, poid_pg, B_dep_e_pg, matC, B_dep_e_pg, optimize='optimal')
            Ku_e = np.einsum('epij,jk,epkl->eil', leftDepPart, matC, B_dep_e_pg, optimize='optimal')
           
        
        if not steadyState:
            jacobien_e_pg = mesh.Get_jacobien_e_pg(matriceType)
            poid_pg = mesh.Get_poid_pg(matriceType)
            N_vecteur_e_pg = mesh.Get_N_vecteur_pg(matriceType)
            ro = self.materiau.ro

            Mu_e = np.einsum('ep,p,pki,,pkj->eij', jacobien_e_pg, poid_pg, N_vecteur_e_pg, ro, N_vecteur_e_pg, optimize="optimal")

        if self.dim == 2:
            epaisseur = self.materiau.epaisseur
            Ku_e *= epaisseur
            if not steadyState:
                Mu_e *= epaisseur
        
        tic.Tac("Matrices","Construction Ku_e", self._verbosity)

        if steadyState:
            return Ku_e
        else:
            return Ku_e, Mu_e
 
    def Assemblage(self, steadyState=True):
        """Construit K global pour le problème en deplacement
        """

        # Data
        mesh = self.mesh        
        taille = mesh.Nn*self.dim

        # Construit dict_Ku_e
        if steadyState:
            Ku_e = self.ConstruitMatElem_Dep(steadyState)
        else:
            Ku_e, Mu_e = self.ConstruitMatElem_Dep(steadyState)

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

        if steadyState:
            tic.Tac("Matrices","Assemblage Ku et Fu", self._verbosity)
            return self.__Ku
        else:
            self.__Mu = sparse.csr_matrix((Mu_e.reshape(-1), (lignesVector_e.reshape(-1), colonnesVector_e.reshape(-1))), shape=(taille, taille))
            """Matrice Mglob pour le problème en déplacement (Nn*dim, Nn*dim)"""

            tic.Tac("Matrices","Assemblage Ku, Mu et Fu", self._verbosity)
            return self.__Ku, self.__Mu

    @property
    def Mu(self) -> sparse.csr_matrix:
        return self.__Mu.copy()
    @property
    def Ku(self) -> sparse.csr_matrix:
        return self.__Ku.copy()
    @property
    def Fu(self) -> sparse.csr_matrix:
        return self.__Fu.copy()

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

    def Solve(self, steadyState=True) -> np.ndarray:
        """Resolution du probleme de déplacement"""

        if steadyState:
            x = self.Solveur(ModelType.displacement, Interface_Solveurs.AlgoType.elliptic)
            assert x.shape[0] == self.mesh.Nn*self.dim

            self.__displacement = x

            return self.__displacement.copy()
        else:
            x = self.Solveur(ModelType.displacement, Interface_Solveurs.AlgoType.hyperbolic)
            assert x.shape[0] == self.mesh.Nn*self.dim
            
            # Formulation en accel

            a_np1 = np.array(x)
                

            u_n = self.displacement
            v_n = self.speed
            a_n = self.accel

            dt = self.dt
            gamma = self.gamma
            betha = self.betha

            uTild_np1 = u_n + (dt * v_n) + dt**2/2 * (1-2*betha) * a_n
            vTild_np1 = v_n + (1-gamma) * dt * a_n

            # Formulation en accel
            u_np1 = uTild_np1 + betha * dt**2 * a_np1
            v_np1 = vTild_np1 + gamma * dt * a_np1

            # # Formulation en déplacement
            # a_np1 = (u_np1 - uTild_np1)/(betha*dt**2)
            # v_np1 = vTild_np1 + (gamma * dt * a_np1)

            

            self.__displacement = u_np1
            self.__speed = v_np1
            self.accel = a_np1

            return self.displacement
        
    
    def Save_Iteration(self, nombreIter=None, tempsIter=None, dincMax=None):
        
        iter = super().Save_Iteration(nombreIter, tempsIter, dincMax)

        iter['displacement'] = self.__displacement
        try:
            iter["speed"] = self.__speed
            iter["accel"] = self.accel
        except:
            pass

        self.__results.append(iter)

    
    def Update_iter(self, index= -1):
        
        results = super().Update_iter(index)

        if results == None: return

        self.__displacement = results["displacement"]
        try :
            self.__speed = results["speed"]
            self.accel = results["accel"]
        except:
            # La première itération à été réalisé en régime stationnaire
            self.__speed = np.zeros_like(self.__displacement)
            self.accel = np.zeros_like(self.__displacement)
            

    def Get_Resultat(self, option: str, valeursAuxNoeuds=True, iter=None):
        
        dim = self.dim
        Ne = self.mesh.Ne
        Nn = self.mesh.Nn
        
        if not self.Resultats_Verification(option): return None

        if iter != None:
            self.Update_iter(iter)

        if option in ["Wdef","Psi_Elas"]:
                return self.__Calc_Psi_Elas()

        if option == "displacement":
            return self.displacement

        if option == "speed":
            return self.speed
        
        if option == "accel":
            return self.accel

        if option == 'coordoDef':
            return self.Resultats_GetCoordUglob()

        displacement = self.displacement

        coef = self.materiau.comportement.coef

        # TODO fusionner avec Stress ?

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
                raise "Erreur"

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

            if valeursAuxNoeuds:
                resultat_n = self.Resultats_InterpolationAuxNoeuds(resultat_e)
                return resultat_n
            else:
                return resultat_e
        
        else:

            Nn = self.mesh.Nn

            if option in ["dx", "dy", "dz", "amplitude"]:
                resultat_ddl = self.displacement
            elif option in ["vx", "vy", "vz", "amplitudeSpeed"]:
                resultat_ddl = self.speed
            elif option in ["ax", "ay", "az", "amplitudeAccel"]:
                resultat_ddl = self.accel

            resultat_ddl = resultat_ddl.reshape(Nn, -1)

            index = _Simu.Resultats_indexe_option(self.dim, self.problemType, option)
            
            if valeursAuxNoeuds:

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

    def __Calc_Psi_Elas(self):
        """Calcul de l'energie de deformation cinématiquement admissible endommagé ou non
        Calcul de Wdef = 1/2 int_Omega jacobien * poid * Sig : Eps dOmega x epaisseur"""

        tic = Tic()

        sol_u  = self.__displacement

        matriceType = MatriceType.rigi
        Epsilon_e_pg = self.__Calc_Epsilon_e_pg(sol_u, matriceType)
        jacobien_e_pg = self.mesh.Get_jacobien_e_pg(matriceType)
        poid_pg = self.mesh.Get_poid_pg(matriceType)

        if self.dim == 2:
            ep = self.materiau.epaisseur
        else:
            ep = 1

        Sigma_e_pg = self.__Calc_Sigma_e_pg(Epsilon_e_pg, matriceType)
            
        Wdef = 1/2 * np.einsum(',ep,p,epi,epi->', ep, jacobien_e_pg, poid_pg, Sigma_e_pg, Epsilon_e_pg, optimize='optimal')

        # # Calcul par Element fini
        # u_e = self.__mesh.Localises_sol_e(sol_u)
        # Ku_e = self.__ConstruitMatElem_Dep()
        # Wdef = 1/2 * np.einsum('ei,eij,ej->', u_e, Ku_e, u_e, optimize='optimal')
            

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
            Deformations stockées aux elements et points de gauss (Ne,pg,(3 ou 6))
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
            Deformations stockées aux elements et points de gauss (Ne,pg,(3 ou 6))

        Returns
        -------
        np.ndarray
            Renvoie les contrainres endommagé ou non (Ne,pg,(3 ou 6))
        """

        assert Epsilon_e_pg.shape[0] == self.mesh.Ne
        assert Epsilon_e_pg.shape[1] == self.mesh.Get_nPg(matriceType)

        useNumba = self.useNumba

        tic = Tic()

        c = self.materiau.comportement.get_C()
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

        resume += self.Resultats_Displacement(verbosity)

        return resume

    def Resultats_Displacement(self, verbosity=True):
        """Ecrit un résumé de la simulation dans le terminal"""

        resume = ""

        if not self.Resultats_Verification("Wdef"):
            return
        
        Wdef = self.Get_Resultat("Wdef")
        resume += f"\nW def = {Wdef:.2f}"
        
        Svm = self.Get_Resultat("Svm", valeursAuxNoeuds=False)
        resume += f"\n\nSvm max = {Svm.max():.2f}"

        # Affichage des déplacements
        dx = self.Get_Resultat("dx", valeursAuxNoeuds=True)
        resume += f"\n\nUx max = {dx.max():.2e}"
        resume += f"\nUx min = {dx.min():.2e}"

        dy = self.Get_Resultat("dy", valeursAuxNoeuds=True)
        resume += f"\n\nUy max = {dy.max():.2e}"
        resume += f"\nUy min = {dy.min():.2e}"

        if self.dim == 3:
            dz = self.Get_Resultat("dz", valeursAuxNoeuds=True)
            resume += f"\n\nUz max = {dz.max():.2e}"
            resume += f"\nUz min = {dz.min():.2e}"

        if verbosity: print(resume)

        return resume

###################################################################################################

class __Simu_PhaseField(_Simu):

    def Check_Directions(self, problemType: ModelType, directions: list):
        # Rien d'implémenté car aucune direction n'est nécessaire pour cette simulation
        pass

    def __init__(self, mesh: Mesh, materiau: _Materiau_PhaseField, verbosity=True, useNumba=True):
        assert materiau.modelType == ModelType.damage, "Le materiau doit être de type damage"
        super().__init__(mesh, materiau, verbosity, useNumba)

        # resultats
        self.__init_results()
    
    def __init_results(self):

        self.__displacement = np.zeros(self.mesh.Nn*self.dim)
        self.__damage = np.zeros(self.mesh.Nn)
        self.__psiP_e_pg = []
        self.__old_psiP_e_pg = [] #ancienne densitée d'energie elastique positive PsiPlus(e, pg, 1) pour utiliser le champ d'histoire de miehe

        self.__results = [] #liste de dictionnaire qui contient les résultats

        self.Solveur_Parabolic_Properties() # Renseigne les propriétes de résolution de l'algorithme
        self.Solveur_Newton_Raphson_Properties() # Renseigne les propriétes de résolution de l'algorithme
    
    def nbddl_n(self, problemType="") -> int:
        if problemType == ModelType.damage:
            return 1
        elif problemType == ModelType.displacement:
            return self.dim

    @property
    def materiau(self) -> _Materiau_PhaseField:
        return super().materiau

    @property
    def comportement(self):
        return self.materiau.phaseFieldModel.comportement

    @property
    def results(self) -> List[dict]:
        return self.__results

    @property
    def displacement(self) -> np.ndarray:
        """Copie du champ vectoriel de déplacement"""
        return self.__displacement.copy()

    @property
    def damage(self) -> np.ndarray:
        """Copie du champ scalaire d'endommagement"""
        return self.__damage.copy()

    def add_dirichlet(self, noeuds: np.ndarray, valeurs: np.ndarray, directions: list, problemType=None, description=""):
        if problemType == None:
            problemType = ModelType.displacement
        return super().add_dirichlet(noeuds, valeurs, directions, problemType, description)
    
    def add_lineLoad(self, noeuds: np.ndarray, valeurs: list, directions: list, problemType=None, description=""):
        if problemType == None:
            problemType = ModelType.displacement
        return super().add_lineLoad(noeuds, valeurs, directions, problemType, description)

    def add_surfLoad(self, noeuds: np.ndarray, valeurs: list, directions: list, problemType=None, description=""):
        if problemType == None:
            problemType = ModelType.displacement
        return super().add_surfLoad(noeuds, valeurs, directions, problemType, description)
        
    def add_pointLoad(self, noeuds: np.ndarray, valeurs: list, directions: list, problemType=None, description=""):
        if problemType == None:
            problemType = ModelType.displacement
        return super().add_pointLoad(noeuds, valeurs, directions, problemType, description)
    

    def Assemblage(self):
        print("Utiliser Assemblage_u() et Assemblage_d()")
    
    def Solve(self):
        print("Utiliser Solve_u() et Solve_d()")

    def ConstruitMatElem_Dep(self) -> np.ndarray:
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

        d = self.__damage
        u = self.__displacement

        phaseFieldModel = self.materiau.phaseFieldModel
        
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
            epaisseur = self.materiau.epaisseur
            Ku_e *= epaisseur
        
        tic.Tac("Matrices","Construction Ku_e", self._verbosity)

        return Ku_e
 
    def Assemblage_u(self):
        """Construit K global pour le problème en deplacement
        """

        # Data
        mesh = self.mesh        
        taille = mesh.Nn*self.dim

        Ku_e = self.ConstruitMatElem_Dep()
        

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
    
    @property
    def Ku(self) -> sparse.csr_matrix:
        return self.__Ku.copy()
    @property
    def Fu(self) -> sparse.csr_matrix:
        return self.__Fu.copy()

    def Solve_u(self, steadyState=True) -> np.ndarray:
        """Resolution du probleme de déplacement"""
            
        displacement = self.Solveur(ModelType.displacement, Interface_Solveurs.AlgoType.elliptic)

        # Si c'est un problement d'endommagement on renseigne au model phase field qu'il va falloir mettre à jour le split
        if self.problemType == ModelType.damage:
            self.materiau.phaseFieldModel.Need_Split_Update()
        
        assert displacement.shape[0] == self.mesh.Nn*self.dim

        self.__displacement = displacement
       
        return self.__displacement.copy()

    # ------------------------------------------- PROBLEME ENDOMMAGEMENT ------------------------------------------- 

    def Calc_psiPlus_e_pg(self):
        """Calcul de la densité denergie positive\n
        Pour chaque point de gauss de tout les elements du maillage on va calculer psi+
       
        Returns:
            np.ndarray: self.__psiP_e_pg
        """

        phaseFieldModel = self.materiau.phaseFieldModel
        
        u = self.__displacement
        d = self.__damage

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

        phaseFieldModel = self.materiau.phaseFieldModel

        # Data
        k = phaseFieldModel.k
        PsiP_e_pg = self.Calc_psiPlus_e_pg()
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

    def Assemblage_d(self):
        """Construit Kglobal pour le probleme d'endommagement
        """
       
        # Data
        mesh = self.mesh
        taille = mesh.Nn
        lignesScalar_e = mesh.lignesScalar_e
        colonnesScalar_e = mesh.colonnesScalar_e
        
        # Calul les matrices elementaires
        Kd_e, Fd_e = self.__ConstruitMatElem_Pfm()

        # Assemblage
        tic = Tic()        

        self.__Kd = sparse.csr_matrix((Kd_e.reshape(-1), (lignesScalar_e.reshape(-1), colonnesScalar_e.reshape(-1))), shape = (taille, taille))
        """Kglob pour le probleme d'endommagement (Nn, Nn)"""
        
        lignes = mesh.connect.reshape(-1)
        self.__Fd = sparse.csr_matrix((Fd_e.reshape(-1), (lignes,np.zeros(len(lignes)))), shape = (taille,1))
        """Fglob pour le probleme d'endommagement (Nn, 1)"""        

        tic.Tac("Matrices","Assemblage Kd et Fd", self._verbosity)       

        return self.__Kd, self.__Fd

    @property
    def Kd(self) -> sparse.csr_matrix:
        return self.__Kd.copy()
    @property
    def Fd(self) -> sparse.csr_matrix:
        return self.__Fd.copy()
    
    def Solve_d(self) -> np.ndarray:
        """Resolution du problème d'endommagement"""
        
        damage = self.Solveur(ModelType.damage, Interface_Solveurs.AlgoType.elliptic)

        assert damage.shape[0] == self.mesh.Nn

        self.__damage = damage

        return self.__damage.copy()

    def Save_Iteration(self, nombreIter=None, tempsIter=None, dincMax=None):

        iter = super().Save_Iteration(nombreIter, tempsIter, dincMax)
        
        if self.materiau.phaseFieldModel.solveur == "History":
            # mets à jour l'ancien champ histoire pour la prochaine résolution 
            self.__old_psiP_e_pg = self.__psiP_e_pg
            
        iter['displacement'] = self.__displacement
        iter['damage'] = self.__damage

        self.__results.append(iter)

    def Update_iter(self, index=-1):

        results = super().Update_iter(index)

        if results == None: return

        self.__old_psiP_e_pg = [] # TODO est il vraiment utile de faire ça ?
        self.__damage = results["damage"]
        self.__displacement = results["displacement"]
        try:
            self.materiau.phaseFieldModel.Need_Split_Update()
        except:
            # Il est possible que cette version du modèle d'endomagement ne possède pas cette fonction
            pass

    def Get_Resultat(self, option: str, valeursAuxNoeuds=True, iter=None):
        
        dim = self.dim
        Ne = self.mesh.Ne
        Nn = self.mesh.Nn
        
        if not self.Resultats_Verification(option): return None

        if iter != None:
            self.Update_iter(iter)

        if option in ["Wdef","Psi_Elas"]:
            return self.__Calc_Psi_Elas()

        if option == "Psi_Crack":
            return self.__Calc_Psi_Crack()

        if option == "damage":
            return self.damage

        if option == "psiP":
            resultat_e_pg = self.Calc_psiPlus_e_pg()
            resultat_e = np.mean(resultat_e_pg, axis=1)

            if valeursAuxNoeuds:
                return self.Resultats_InterpolationAuxNoeuds(resultat_e)
            else:
                return resultat_e

        if option == "displacement":
            return self.displacement

        if option == "speed":
            return self.speed
        
        if option == "accel":
            return self.accel

        if option == 'coordoDef':
            return self.Resultats_GetCoordUglob()

        displacement = self.displacement

        coef = self.materiau.comportement.coef

        # TODO fusionner avec Stress ?

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
                raise "Erreur"

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

            if valeursAuxNoeuds:
                resultat_n = self.Resultats_InterpolationAuxNoeuds(resultat_e)
                return resultat_n
            else:
                return resultat_e
        
        else:

            Nn = self.mesh.Nn

            if option in ["dx", "dy", "dz", "amplitude"]:
                resultat_ddl = self.displacement
            elif option in ["vx", "vy", "vz", "amplitudeSpeed"]:
                resultat_ddl = self.speed
            elif option in ["ax", "ay", "az", "amplitudeAccel"]:
                resultat_ddl = self.accel

            resultat_ddl = resultat_ddl.reshape(Nn, -1)

            index = _Simu.Resultats_indexe_option(self.dim, self.problemType, option)
            
            if valeursAuxNoeuds:

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

    def __Calc_Psi_Elas(self):
        """Calcul de l'energie de deformation cinématiquement admissible endommagé ou non
        Calcul de Wdef = 1/2 int_Omega jacobien * poid * Sig : Eps dOmega x epaisseur"""

        tic = Tic()

        sol_u  = self.__displacement

        matriceType = MatriceType.rigi
        Epsilon_e_pg = self.__Calc_Epsilon_e_pg(sol_u, matriceType)
        jacobien_e_pg = self.mesh.Get_jacobien_e_pg(matriceType)
        poid_pg = self.mesh.Get_poid_pg(matriceType)

        if self.dim == 2:
            ep = self.materiau.epaisseur
        else:
            ep = 1

        d = self.__damage

        phaseFieldModel = self.materiau.phaseFieldModel
        psiP_e_pg, psiM_e_pg = phaseFieldModel.Calc_psi_e_pg(Epsilon_e_pg)

        # Endommage : psiP_e_pg = g(d) * PsiP_e_pg 
        g_e_pg = phaseFieldModel.get_g_e_pg(d, self.mesh, matriceType)
        psiP_e_pg = np.einsum('ep,ep->ep', g_e_pg, psiP_e_pg, optimize='optimal')
        psi_e_pg = psiP_e_pg + psiM_e_pg

        Wdef = np.einsum(',ep,p,ep->', ep, jacobien_e_pg, poid_pg, psi_e_pg, optimize='optimal')

        tic.Tac("PostTraitement","Calcul Psi Elas",False)
        
        return Wdef

    def __Calc_Psi_Crack(self):
        """Calcul l'energie de fissure"""

        tic = Tic()

        pfm = self.materiau.phaseFieldModel 

        matriceType = MatriceType.masse

        Gc = pfm.Gc
        l0 = pfm.l0
        c0 = pfm.c0

        d_n = self.__damage
        d_e = self.mesh.Localises_sol_e(d_n)

        jacobien_e_pg = self.mesh.Get_jacobien_e_pg(matriceType)
        poid_pg = self.mesh.Get_poid_pg(matriceType)
        Nd_pg = self.mesh.Get_N_scalaire_pg(matriceType)
        Bd_e_pg = self.mesh.Get_dN_sclaire_e_pg(matriceType)

        grad_e_pg = np.einsum('epij,ej->epi',Bd_e_pg,d_e, optimize='optimal')
        diffuse_e_pg = grad_e_pg**2

        gradPart = np.einsum('ep,p,,epi->',jacobien_e_pg, poid_pg, Gc*l0/c0, diffuse_e_pg, optimize='optimal')

        alpha_e_pg = np.einsum('pij,ej->epi', Nd_pg, d_e, optimize='optimal')
        if pfm.regularization == "AT2":
            alpha_e_pg = alpha_e_pg**2
        
        alphaPart = np.einsum('ep,p,,epi->',jacobien_e_pg, poid_pg, Gc/(c0*l0), alpha_e_pg, optimize='optimal')

        if self.dim == 2:
            ep = self.materiau.epaisseur
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
            Deformations stockées aux elements et points de gauss (Ne,pg,(3 ou 6))
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
            Deformations stockées aux elements et points de gauss (Ne,pg,(3 ou 6))

        Returns
        -------
        np.ndarray
            Renvoie les contrainres endommagé ou non (Ne,pg,(3 ou 6))
        """

        assert Epsilon_e_pg.shape[0] == self.mesh.Ne
        assert Epsilon_e_pg.shape[1] == self.mesh.Get_nPg(matriceType)

        useNumba = self.useNumba

        d = self.__damage

        phaseFieldModel = self.materiau.phaseFieldModel

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

    def Resultats_Resume(self, verbosity=True) -> str:
        return super().Resultats_Resume(verbosity)

###################################################################################################

class __Simu_Beam(_Simu):

    __dict_dim_directions = {
        1 : ["x"],
        2 : ["x","y","rz"],
        3 : ["x","y","z","rx","ry","rz"]
    }

    def Check_Directions(self, problemType: ModelType, directions: list):
        listDirections = __Simu_Beam.__dict_dim_directions[self.dim]
        for d in directions: assert d in listDirections, f"{d} doit être dans [{listDirections}]"

    def Check_dim_mesh_materiau(self) -> None:
        # Dans le cadre d'un probleme de poutre on à pas besoin de verifier cette condition
        pass

    def __init__(self, mesh: Mesh, materiau: _Materiau_Beam, verbosity=True, useNumba=True):
        assert materiau.modelType == ModelType.beam, "Le materiau doit être de type beam"
        super().__init__(mesh, materiau, verbosity, useNumba)

        # resultats
        self.__init_results()
    
    def __init_results(self):

        self.__beamDisplacement = np.zeros(self.mesh.Nn*self.materiau.beamModel.nbddl_n)

        self.__results = [] #liste de dictionnaire qui contient les résultats

        self.Solveur_Parabolic_Properties() # Renseigne les propriétes de résolution de l'algorithme
        self.Solveur_Newton_Raphson_Properties() # Renseigne les propriétes de résolution de l'algorithme
    
    def nbddl_n(self, problemType="") -> int:
        return self.materiau.beamModel.nbddl_n

    @property
    def materiau(self) -> _Materiau_Beam:
        return super().materiau
    
    @property
    def use3DBeamModel(self) -> bool:
        if self.materiau.beamModel.dim == 3:
            return True
        else:
            return False

    @property
    def results(self) -> List[dict]:
        return self.__results

    @property
    def beamDisplacement(self) -> np.ndarray:
        """Copie du champ vectoriel de déplacement pour le problème poutre"""
        return self.__beamDisplacement.copy()

    def ConstruitMatElem_Beam(self) -> np.ndarray:
        """Construit les matrices de rigidités élementaires pour le problème en déplacement

        Returns:
            Ku_beam: les matrices elementaires pour chaque groupe d'element 
        """

        # Data
        mesh = self.mesh
        if not mesh.groupElem.dim == 1: return
        groupElem = mesh.groupElem

        # Récupération du maodel poutre
        beamModel = self.materiau.beamModel
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
        beamModel = self.materiau.beamModel
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
            if elemType == "SEG2":
                repu = [0,1]
            elif elemType == "SEG3":
                repu = [0,1,2]
            elif elemType == "SEG4":
                repu = [0,1,2,3]
            elif elemType == "SEG5":
                repu = [0,1,2,3,4]

            B_e_pg = np.zeros((Ne, nPg, 1, nbddl_n*nPe))
            B_e_pg[:,:,0, repu] = dN_e_pg[:,:,0]
                

        elif dim == 2:
            # u = [u1, v1, rz1, . . . , un, vn, rzn]
            
            # repu = np.arange(0,3*nPe,3) # [0,3] (SEG2) [0,3,6] (SEG3)
            if elemType == "SEG2":
                repu = [0,3]
                repv = [1,2,4,5]
            elif elemType == "SEG3":
                repu = [0,3,6]
                repv = [1,2,4,5,7,8]
            elif elemType == "SEG4":
                repu = [0,3,6,9]
                repv = [1,2,4,5,7,8,10,11]
            elif elemType == "SEG5":
                repu = [0,3,6,9,12]
                repv = [1,2,4,5,7,8,10,11,13,14]

            B_e_pg = np.zeros((Ne, nPg, 2, nbddl_n*nPe))
            
            B_e_pg[:,:,0, repu] = dN_e_pg[:,:,0]
            B_e_pg[:,:,1, repv] = ddNv_e_pg[:,:,0]

        elif dim == 3:
            # u = [u1, v1, w1, rx1, ry1, rz1, u2, v2, w2, rx2, ry2, rz2]

            if elemType == "SEG2":
                repux = [0,6]
                repvy = [1,5,7,11]
                repvz = [2,4,8,10]
                reptx = [3,9]
            elif elemType == "SEG3":
                repux = [0,6,12]
                repvy = [1,5,7,11,13,17]
                repvz = [2,4,8,10,14,16]
                reptx = [3,9,15]
            elif elemType == "SEG4":
                repux = [0,6,12,18]
                repvy = [1,5,7,11,13,17,19,23]
                repvz = [2,4,8,10,14,16,20,22]
                reptx = [3,9,15,21]
            elif elemType == "SEG5":
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
        """Construit K global pour le problème en deplacement
        """

        # Data
        mesh = self.mesh

        model = self.materiau.beamModel

        assert isinstance(model, Beam_Model)

        taille = mesh.Nn * model.nbddl_n

        Ku_beam = self.ConstruitMatElem_Beam()

        # Dimension supplémentaire lié a l'utilisation des coefs de lagrange
        dimSupl = len(self.Bc_Lagrange)
        if dimSupl > 0:
            dimSupl += len(self.Bc_ddls_Dirichlet(MatriceType.beam))

        # Prépare assemblage
        lignesVector_e = mesh.lignesVectorBeam_e(model.nbddl_n)
        colonnesVector_e = mesh.colonnesVectorBeam_e(model.nbddl_n)
        
        tic = Tic()

        # Assemblage
        self.__Kbeam = sparse.csr_matrix((Ku_beam.reshape(-1), (lignesVector_e.reshape(-1), colonnesVector_e.reshape(-1))), shape=(taille+dimSupl, taille+dimSupl))
        """Matrice Kglob pour le problème poutre (Nn*nbddl_e, Nn*nbddl_e)"""

        # Ici j'initialise Fu calr il faudrait calculer les forces volumiques dans __ConstruitMatElem_Dep !!!
        self.__Fbeam = sparse.csr_matrix((taille+dimSupl, 1))
        """Vecteur Fglob pour le problème poutre (Nn*nbddl_e, 1)"""

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.spy(self.__Ku)
        # plt.show()

        tic.Tac("Matrices","Assemblage Kbeam et Fbeam", self._verbosity)
        return self.__Kbeam

    @property
    def Kbeam(self) -> sparse.csr_matrix:
        return self.__Kbeam
    @property
    def Fbeam(self) -> sparse.csr_matrix:
        return self.__Fbeam

    def Solve(self) -> np.ndarray:
        """Resolution du probleme poutre"""

        beamDisplacement = self.Solveur(ModelType.beam, Interface_Solveurs.AlgoType.elliptic)
        
        assert beamDisplacement.shape[0] == self.mesh.Nn*self.materiau.beamModel.nbddl_n

        self.__beamDisplacement = beamDisplacement
       
        return self.__beamDisplacement.copy()

    def Save_Iteration(self, nombreIter=None, tempsIter=None, dincMax=None):

        iter = super().Save_Iteration(nombreIter, tempsIter, dincMax)
        
        iter['beamDisplacement'] = self.__beamDisplacement
            
        self.__results.append(iter)

    def Update_iter(self, index=-1):
        
        results = super().Update_iter(index)

        if results == None: return

        self.__beamDisplacement = results["beamDisplacement"]

    def Get_Resultat(self, option: str, valeursAuxNoeuds=True, iter=None):
        
        if not self.Resultats_Verification(option): return None

        if iter != None:
            self.Update_iter(iter)

        if option == "beamDisplacement":
            return self.beamDisplacement.copy()
    
        if option == 'coordoDef':
            return self.Resultats_GetCoordUglob()

        if option in ["fx","fy","fz","cx","cy","cz"]:

            force = np.array(self.__Kbeam.dot(self.__beamDisplacement))
            force_redim = force.reshape(self.mesh.Nn, -1)
            index = _Simu.Resultats_indexe_option(self.dim, self.problemType, option)
            resultat_ddl = force_redim[:, index]

        else:

            resultat_ddl = self.beamDisplacement
            resultat_ddl = resultat_ddl.reshape((self.mesh.Nn,-1))

        index = _Simu.Resultats_indexe_option(self.dim, self.problemType, option)


        # Deformation et contraintes pour chaque element et chaque points de gauss        
        Epsilon_e_pg = self.__Calc_Epsilon_e_pg(self.beamDisplacement)
        Sigma_e_pg = self.__Calc_Sigma_e_pg(Epsilon_e_pg)

        if option == "Stress":
            return np.mean(Sigma_e_pg, axis=1)


        if valeursAuxNoeuds:
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

    def __Calc_Epsilon_e_pg(self, sol: np.ndarray, matriceType=MatriceType.rigi):
        """Construit les déformations pour chaque element et chaque points de gauss
        """
        
        tic = Tic()

        nbddl_n = self.materiau.beamModel.nbddl_n
        assemblyBeam_e = self.mesh.groupElem.get_assemblyBeam_e(nbddl_n)
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

        D_e_pg = self.materiau.beamModel.Calc_D_e_pg(self.mesh.groupElem, matriceType)
        Sigma_e_pg = np.einsum('epij,epj->epi', D_e_pg, Epsilon_e_pg, optimize='optimal')
            
        tic.Tac("Matrices", "Sigma_e_pg", False)

        return Sigma_e_pg

    def Resultats_Resume(self, verbosity=True) -> str:
        return super().Resultats_Resume(verbosity)

###################################################################################################

class __Simu_Thermal(_Simu):

    def Check_Directions(self, problemType: ModelType, directions: list):
        # Rien d'implémenté car aucune direction n'est nécessaire pour cette simulation
        pass

    def __init__(self, mesh: Mesh, materiau: _Materiau_Thermal, verbosity=True, useNumba=True):
        assert materiau.modelType == ModelType.thermal, "Le materiau doit être de type thermal"
        super().__init__(mesh, materiau, verbosity, useNumba)

        # resultats
        self.__init_results()
    
    def __init_results(self):        

        self.__thermal = np.zeros(self.mesh.Nn)

        if self.materiau.thermalModel.c > 0 and self.materiau.ro > 0:
            # Il est possible de calculer la matrice de masse et donc de résoudre un problème parabolic au lieu d'elliptic
            self.__thermalDot = np.zeros_like(self.__thermal)

        self.__results = [] #liste de dictionnaire qui contient les résultats

        self.Solveur_Parabolic_Properties() # Renseigne les propriétes de résolution de l'algorithme
        self.Solveur_Newton_Raphson_Properties() # Renseigne les propriétes de résolution de l'algorithme
    
    def nbddl_n(self, problemType="") -> int:
        return 1

    @property
    def materiau(self) -> _Materiau_Thermal:
        return super().materiau

    @property
    def results(self) -> List[dict]:
        return self.__results

    @property
    def thermal(self) -> np.ndarray:
        """Copie du champ scalaire de température"""
        return self.__thermal.copy()

    @property
    def thermalDot(self) -> np.ndarray:
        """Copie de la dérivée du champ scalaire de température"""
        if self.materiau.thermalModel.c > 0 and self.materiau.ro > 0:
            return self.__thermalDot.copy()
        else:
            return None
    
    

    def __ConstruitMatElem_Thermal(self, steadyState: bool) -> np.ndarray:

        thermalModel = self.materiau.thermalModel

        # Data
        k = thermalModel.k

        matriceType=MatriceType.rigi

        mesh = self.mesh

        jacobien_e_pg = mesh.Get_jacobien_e_pg(matriceType)
        poid_pg = mesh.Get_poid_pg(matriceType)
        N_e_pg = mesh.Get_N_scalaire_pg(matriceType)
        D_e_pg = mesh.Get_dN_sclaire_e_pg(matriceType)

        Kt_e = np.einsum('ep,p,epji,,epjk->eik', jacobien_e_pg, poid_pg, D_e_pg, k, D_e_pg, optimize="optimal")

        if steadyState:
            return Kt_e
        else:
            ro = self.materiau.ro
            c = thermalModel.c

            Mt_e = np.einsum('ep,p,pji,,,pjk->eik', jacobien_e_pg, poid_pg, N_e_pg, ro, c, N_e_pg, optimize="optimal")

            return Kt_e, Mt_e


    def Assemblage(self, steadyState=True) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Construit du systeme matricielle pour le probleme thermique en régime stationnaire ou transitoire
        """
       
        # Data
        mesh = self.mesh
        taille = mesh.Nn
        lignesScalar_e = mesh.lignesScalar_e
        colonnesScalar_e = mesh.colonnesScalar_e
        
        # Calul les matrices elementaires
        if steadyState:
            Kt_e = self.__ConstruitMatElem_Thermal(steadyState)
        else:
            Kt_e, Mt_e = self.__ConstruitMatElem_Thermal(steadyState)
        

        # Assemblage
        tic = Tic()

        self.__Kt = sparse.csr_matrix((Kt_e.reshape(-1), (lignesScalar_e.reshape(-1), colonnesScalar_e.reshape(-1))), shape = (taille, taille))
        """Kglob pour le probleme thermique (Nn, Nn)"""
        
        self.__Ft = sparse.csr_matrix((taille, 1))
        """Vecteur Fglob pour le problème en thermique (Nn, 1)"""

        if not steadyState:
            self.__Mt = sparse.csr_matrix((Mt_e.reshape(-1), (lignesScalar_e.reshape(-1), colonnesScalar_e.reshape(-1))), shape = (taille, taille))
            """Mglob pour le probleme thermique (Nn, Nn)"""

        tic.Tac("Matrices","Assemblage Kt et Ft", self._verbosity)       

        return self.__Kt, self.__Ft

    @property
    def Kt(self) -> sparse.csr_matrix:
        return self.__Kt.copy()
    @property
    def Mt(self) -> sparse.csr_matrix:
        return self.__Mt.copy()
    @property
    def Ft(self) -> sparse.csr_matrix:
        return self.__Ft.copy()

    def Solve(self, steadyState=True) -> np.ndarray:
        """Resolution du problème thermique

        Parameters
        ----------
        isStatic : bool, optional
            Le problème est stationnaire, by default True

        Returns
        -------
        np.ndarray
            vecteur solution
        """

        if steadyState:
            thermal_np1 = self.Solveur(ModelType.thermal, Interface_Solveurs.AlgoType.elliptic)
            # TODO que faire pour -> quand plusieurs types -> np.ndarray ou tuple[np.ndarray, np.ndarray] ?
        else:

            thermal_np1 = self.Solveur(ModelType.thermal, Interface_Solveurs.AlgoType.parabolic)
            thermalDot = self.thermalDot

            alpha = self.alpha
            dt = self.dt

            thermalDotTild_np1 = self.__thermal + ((1-alpha) * dt * thermalDot)

            thermalDot_np1 = (thermal_np1 - thermalDotTild_np1)/(alpha*dt)

            self.__thermalDot = thermalDot_np1

        assert thermal_np1.shape[0] == self.mesh.Nn

        self.__thermal = thermal_np1

        return self.thermal

    def Save_Iteration(self, nombreIter=None, tempsIter=None, dincMax=None):

        iter = super().Save_Iteration(nombreIter, tempsIter, dincMax)
        
        iter['thermal'] = self.__thermal

        try:
            iter['thermalDot'] = self.__thermalDot                
        except:
            pass
            
        self.__results.append(iter)

    def Update_iter(self, index=-1):
        
        results = super().Update_iter(index)

        if results == None: return

        self.__thermal = results["thermal"]
        try:
            self.__thermalDot = results["thermalDot"]
        except:
            # Résultat non disponible
            pass
            

    def Get_Resultat(self, option: str, valeursAuxNoeuds=True, iter=None):
        
        if not self.Resultats_Verification(option): return None

        if iter != None:
            self.Update_iter(iter)

        if option == "thermal":
            return self.thermal

        if option == "thermalDot":
            try:
                return self.thermalDot
            except:
                print("La simulation thermique est realisé en état d'équilibre")

    
    def Resultats_Resume(self, verbosity=True) -> str:
        return super().Resultats_Resume(verbosity)