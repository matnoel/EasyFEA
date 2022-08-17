
from typing import List, cast
import numpy as np

class BoundaryCondition:
    """Classe de condition limite"""

    def __init__(self, problemType: str, noeuds: np.ndarray,
    ddls: np.ndarray, directions: list, valeurs_ddls: np.ndarray,
    description: str):
        """Construit une boundary conditions

        Parameters
        ----------
        problemType : str
            type de probleme qui doit etre contenue dans ["damage", "displacement"]
        noeuds : np.ndarray
            noeuds sur lesquels on applique une condition
        ddls : np.ndarray
            degrés de liberté associés aux noeuds et aux directions
        directions : list
            directions associées doit etre dans "d" ou ["x","y","z"] en fonction du problème
        valeurs_ddls : np.ndarray
            valeurs appliquées
        description : str
            description de la condition
        """

        assert problemType in ["damage", "displacement"]
        self.__problemType = problemType

        if problemType == "damage":
            assert directions == "d", "Erreur de direction"
        elif problemType == "displacement":
            for d in directions: assert d in ["x","y","z"], "Erreur de direction"
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
    def Get_ddls(problemType: str, list_Bc_Conditions: list) -> np.ndarray:
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
        return np.array(ddls)
    
    @staticmethod
    def Get_ddls_connect(dim: int, problemType:str, connect_e: np.ndarray, directions: list) -> np.ndarray:
        """Construit les ddls liées au noeuds de la matrice de connection

        Parameters
        ----------
        dim : int
            dimension du probleme
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
    def Get_ddls_noeuds(dim: int, problemType:str, noeuds:np.ndarray, directions: list) -> np.ndarray:
        """Récupère les ddls liés aux noeuds en fonction du problème et des directions

        Parameters
        ----------
        dim : int
            dimension du problème
        problemType : str
            type de probleme qui doit etre contenue dans ["damage", "displacement"]
        noeuds : np.ndarray
            noeuds
        directions : list
            directions

        Returns
        -------
        np.ndarray
            liste de ddls
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

