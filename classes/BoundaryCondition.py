
from typing import cast
import numpy as np

class BoundaryCondition:

    def __init__(self, problemType: str, noeuds: np.ndarray,
    ddls: np.ndarray, directions: list, valeurs_ddls: np.ndarray,
    description: str):

        assert problemType in ["damage", "displacement"]
        self.__problemType = problemType

        self.__noeuds = noeuds

        self.__ddls = ddls
        self.__valeurs_ddls = valeurs_ddls
        self.__directions = directions
        
        self.description = description

    def __get_problemType(self):
        return self.__problemType
    problemType = cast(str, property(__get_problemType))

    def __get_noeuds(self):
        return self.__noeuds
    noeuds = cast(np.ndarray, property(__get_noeuds))

    def __get_ddls(self):
        return self.__ddls
    ddls = cast(np.ndarray, property(__get_ddls))

    def __get_valeurs(self):
        return self.__valeurs_ddls
    valeurs_ddls = cast(np.ndarray, property(__get_valeurs))

    def __get_directions(self):
        return self.__directions
    directions = cast(list, property(__get_directions))

    # Methodes statiques pour construire les ddls

    @staticmethod
    def Get_ddls(problemType, list_Bc_Conditions: list):
        """Renvoie les ddls du probleme et de la liste de conditions donné"""                        
        ddls = []
        for bc in list_Bc_Conditions:
            assert isinstance(bc, BoundaryCondition)            
            if bc.problemType == problemType:
                ddls.extend(bc.ddls)
        return np.array(ddls)
    
    @staticmethod
    def Get_ddls_connect(dim: int, problemType:str, connect_e: np.ndarray, directions: list):
        """Construit les ddls liées au noeuds de la matrice de connection
        """    
        if problemType == "damage":
            return connect_e.reshape(-1)
        elif problemType == "displacement":
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
    
    @staticmethod
    def Get_ddls_noeuds(dim: int, problemType:str, noeuds:np.ndarray, directions: list):
        """Récupère les ddls liés aux noeuds en fonction du problème et des directions

        Args:
            dim (int): dimension du problème
            problemType (str): [displacement, damage]
            noeuds (np.ndarray): list des noeuds
            directions (list): directions ["x","y","z"]  (Pas forcément dans l'ordre)

        Returns:
            np.ndarray: liste de ddls
        """

        if problemType == "damage":
            return noeuds.reshape(-1)
        elif problemType == "displacement":
            ddls_dir = np.zeros((noeuds.shape[0], len(directions)), dtype=int)
            for d, direction in enumerate(directions):
                if direction == "x":
                    index = 0
                elif direction == "y":
                    index = 1
                elif direction == "z":
                    assert dim == 3,"Une étude 2D ne permet pas d'appliquer des forces suivant z"
                    index = 2
                else:
                    "Direction inconnue"
                ddls_dir[:,d] = noeuds * dim + index

            return ddls_dir.reshape(-1)
        else:
            print("Problème inconnu")

