import numpy as np
import inspect

class BoundaryCondition:
    """Classe de condition limite"""

    def __init__(self, problemType: str, noeuds: np.ndarray, ddls: np.ndarray, directions: np.ndarray, valeurs_ddls: np.ndarray, description: str):
        """Construit une boundary conditions

        Parameters
        ----------
        problemType : str
            type de probleme qui doit etre contenue dans ["damage", "displacement","thermal","beam"]
        noeuds : np.ndarray
            noeuds sur lesquels on applique une condition
        ddls : np.ndarray
            degrés de liberté associés aux noeuds et aux directions
        directions : list
            directions associées doit etre dans "d" ou ["x","y","z"] en fonction du problème
        valeurs_ddls : np.ndarray
            valeurs appliquées
        fonction : list
            fonction de la condition
        description : str
            description de la condition
        """

        assert problemType in ["damage", "displacement", "thermal","beam"]
        self.__problemType = problemType

        if problemType in ["damage", "thermal"]:
            directions = None
        elif problemType == "displacement":
            for d in directions: assert d in ["x","y","z"], "Erreur de direction"
        elif problemType == "beam":
            for d in directions: assert d in ["x","y","z","rx","ry","rz"], "Erreur de direction"
        self.__directions = directions

        self.__noeuds = noeuds

        self.__ddls = np.asarray(ddls, dtype=int)
        self.__valeurs_ddls = valeurs_ddls

        self.description = description
        """description de la condition"""


    @property
    def problemType(self) -> str:
        """type de problème"""
        return self.__problemType

    @property
    def noeuds(self) -> np.ndarray:
        """noeuds sur lesquels on applique une condition"""
        return self.__noeuds

    @property
    def ddls(self) -> np.ndarray:
        """degrés de liberté associés aux noeuds et aux directions"""
        return self.__ddls

    @property
    def valeurs_ddls(self) -> np.ndarray:
        """valeurs que l'on applique au ddls"""
        return self.__valeurs_ddls

    @property
    def directions(self) -> list:
        """directions associées"""
        return self.__directions

    @staticmethod
    def Get_ddls(problemType: str, list_Bc_Conditions: list) -> list:
        """Renvoie les ddls du probleme et de la liste de conditions donné

        Parameters
        ----------
        problemType : str
            type de probleme qui doit etre contenue dans ["damage", "displacement"]
        list_Bc_Conditions : list de BoundaryCondition
            liste de conditions limites

        Returns
        -------
        np.ndarray
            degrés de liberté
        """
        
        ddls = []
        for bc in list_Bc_Conditions:
            assert isinstance(bc, BoundaryCondition)            
            if bc.problemType == problemType:
                ddls.extend(bc.ddls)
        return ddls

    @staticmethod
    def Get_values(problemType: str, list_Bc_Conditions: list) -> list:
        """Renvoie les ddls du probleme et de la liste de conditions donné

        Parameters
        ----------
        problemType : str
            type de probleme qui doit etre contenue dans ["damage", "displacement"]
        list_Bc_Conditions : list de BoundaryCondition
            liste de conditions limites

        Returns
        -------
        np.ndarray
            degrés de liberté
        """
        
        values = []
        for bc in list_Bc_Conditions:
            assert isinstance(bc, BoundaryCondition)            
            if bc.problemType == problemType:
                values.extend(bc.valeurs_ddls)
        return values
    
    @staticmethod
    def Get_ddls_connect(param: int, problemType:str, connect_e: np.ndarray, directions: list) -> np.ndarray:
        """Construit les ddls liées au noeuds de la matrice de connection

        Parameters
        ----------
        param : int
            parametre du probleme beam -> nbddl_e sinon dim
        problemType : str
            type de probleme qui doit etre contenue dans ["damage", "displacement"]
        connect_e : np.ndarray
            matrice de connectivité
        directions : list
            directions

        Returns
        -------
        np.ndarray
            degrés de liberté
        """
        if problemType in ["damage","thermal"]:
            return connect_e.reshape(-1)
        elif problemType == "displacement":
            dim = param
            indexes = {
                "x": 0,
                "y": 1,
                "z": 2,
            }
            listeIndex=[]
            for dir in directions:
                listeIndex.append(indexes[dir])

            Ne = connect_e.shape[0]
            nPe = connect_e.shape[1]

            connect_e_repet = np.repeat(connect_e, len(directions), axis=0).reshape(-1,nPe)
            listIndex = np.repeat(np.array(listeIndex*nPe), Ne, axis=0).reshape(-1,nPe)

            ddls_dir = np.array(connect_e_repet*dim + listIndex, dtype=int)

            return ddls_dir.reshape(-1)

        elif problemType == "beam":

            nbddl_e = param
            dim = param

            if nbddl_e == 1:
                # 1D
                indexes = {
                    "x": 0,
                }
            elif nbddl_e == 3:
                # 2D
                indexes = {
                    "x": 0,
                    "y": 1,
                    "rz": 2,
                }
            elif nbddl_e == 6:
                # 3D
                indexes = {
                    "x": 0,
                    "y": 1,
                    "z": 2,
                    "rx": 3,
                    "ry": 4,
                    "rz": 5
                }
            
            listeIndex=[]
            for dir in directions:
                listeIndex.append(indexes[dir])

            Ne = connect_e.shape[0]
            nPe = connect_e.shape[1]

            connect_e_repet = np.repeat(connect_e, len(directions), axis=0).reshape(-1,nPe)
            listIndex = np.repeat(np.array(listeIndex*nPe), Ne, axis=0).reshape(-1,nPe)

            ddls_dir = np.array(connect_e_repet*dim + listIndex, dtype=int)

            return ddls_dir.reshape(-1)
    
    @staticmethod
    def Get_ddls_noeuds(param: int, problemType:str, noeuds:np.ndarray, directions: list) -> np.ndarray:
        """Récupère les ddls liés aux noeuds en fonction du problème et des directions

        Parameters
        ----------
        param : int
            parametre du probleme beam -> nbddl_e sinon dim
        problemType : str
            type de probleme qui doit etre contenue dans ["damage", "displacement","thermal", "beam"]
        noeuds : np.ndarray
            noeuds
        directions : list
            directions

        Returns
        -------
        np.ndarray
            liste de ddls
        """

        if problemType in ["damage","thermal"]:
            return noeuds.reshape(-1)
        elif problemType == "displacement":
            ddls_dir = np.zeros((noeuds.shape[0], len(directions)), dtype=int)
            dim = param
            for d, direction in enumerate(directions):
                if direction == "x":
                    index = 0
                elif direction == "y":
                    index = 1
                elif direction == "z":
                    assert dim == 3, "Une étude 2D ne permet pas d'appliquer des forces suivant z"
                    index = 2
                else:
                    "Direction inconnue"
                ddls_dir[:,d] = noeuds * dim + index

            return ddls_dir.reshape(-1)

        elif problemType == "beam":
            ddls_dir = np.zeros((noeuds.shape[0], len(directions)), dtype=int)

            nbddl_e = param

            if nbddl_e == 1:
                dimModel = "1D"
            elif nbddl_e == 3:
                dimModel = "2D"
            elif nbddl_e == 6:
                dimModel = "3D"

            for d, direction in enumerate(directions):

                if direction == "x":
                    index = 0
                elif direction == "y":
                    if dimModel in ["2D","3D"]:
                        index = 1
                    else:
                        raise "Il faut réaliser une Etude poutre 2D ou 3D pour accéder aux ddls suivant y"
                elif direction == "z":
                    assert dimModel == "3D", "Il faut réaliser une Etude poutre 3D pour accéder aux ddls suivant z"
                    index = 2
                elif direction == "rx":
                    if dimModel == "3D":
                        # modèle poutre 3D
                        index = 3
                    else:
                        raise "Il faut réaliser une Etude poutre 3D pour acceder aux ddls rx"
                elif direction == "ry":
                    if dimModel == "3D":
                        # modèle poutre 3D
                        index = 4
                    else:
                        raise "Il faut réaliser une Etude poutre 3D pour acceder aux ddls ry" 
                elif direction == "rz":
                    if dimModel == "2D":
                        # modèle poutre 2D
                        index = 2
                    elif dimModel == "3D":
                        # modèle poutre 3D
                        index = 5
                    else:
                        raise "Il faut réaliser une Etude poutre 2D ou 3D pour acceder aux ddls rz"
                else:
                    raise "Direction inconnue"
                ddls_dir[:,d] = noeuds * nbddl_e + index

            return ddls_dir.reshape(-1)
        else:
            print("Problème inconnu")

class LagrangeCondition(BoundaryCondition):

    def __init__(self, problemType: str, noeuds: np.ndarray, ddls: np.ndarray, directions: np.ndarray, valeurs_ddls: np.ndarray, lagrangeCoefs: np.ndarray, description= ""):
        """Construit une condition de lagrange sur la base d'une boundary conditions"""
        super().__init__(problemType, noeuds, ddls, directions, valeurs_ddls, description)

        self.__lagrangeCoefs = lagrangeCoefs
    
    @property
    def lagrangeCoefs(self):
        return self.__lagrangeCoefs


