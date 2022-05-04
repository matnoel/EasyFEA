
from typing import cast
import numpy as np

class BoundaryCondition:

    def __init__(self, dim: int, problemType: str,
    noeuds: np.ndarray, directions: list, valeurs: np.ndarray,
    description: str, marker='.', color='blue'):

        self.__dim = dim
        
        assert problemType in ["damage", "displacement"]
        self.__problemType = problemType

        self.name = description

        self.__noeuds = np.array(noeuds, dtype=int)

        self.__valeurs = valeurs

        match problemType:
            case "damage":
                assert len(directions) == 0, "Doit renseigner une liste vide"
            case "displacement":
                assert isinstance(directions[0], str), "Doit être une liste de chaine de caractère"
                for d in directions: assert d in ["x", "y", "z"], "Doit être compris dans ['x', 'y', 'z']"
        self.__directions = directions

        self.marker = marker
        self.color = color   

    def __get_dim(self):
        return self.__dim
    dim = property(__get_dim)

    def __get_problemType(self):
        return self.__problemType
    problemType = property(__get_problemType)

    def __get_directions(self):
        return self.__directions
    directions = property(__get_directions)

    def __get_ddls(self):        
        match self.__problemType:
            case "damage":
                return self.__noeuds
            case "displacement":
                ddls = []
                directions = self.__directions
                directions.sort()
                for direction in directions:
                    if direction == "x":
                        ddls.extend(self.__noeuds * self.__dim)
                    if direction == "y":
                        ddls.extend(self.__noeuds * self.__dim + 1)
                    if direction == "z":
                        assert self.__dim == 3,"Une étude 2D ne permet pas d'appliquer des forces suivant z"
                        ddls.extend(self.__noeuds * self.__dim + 2)
                return np.unique(np.array(ddls).reshape(-1))
    ddls = cast(np.ndarray, property(__get_ddls))

    def __get_valeurs(self):
        return self.__valeurs
    valeurs = property(__get_valeurs)

    def __get_directions(self):
        return self.__directions
    directions = property(__get_directions)

