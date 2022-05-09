
from typing import cast
import numpy as np

class BoundaryCondition:

    def __init__(self, dim: int, problemType: str,
    ddls: np.ndarray, directions: list, valeurs: np.ndarray,
    description: str, marker='.', color='blue'):

        self.__dim = dim
        
        assert problemType in ["damage", "displacement"]
        self.__problemType = problemType

        self.name = description

        self.__ddls = ddls

        self.__valeurs = valeurs
                
        self.__directions = directions

        self.marker = marker
        self.color = color   

    def __get_dim(self):
        return self.__dim
    dim = property(__get_dim)

    def __get_problemType(self):
        return self.__problemType
    problemType = property(__get_problemType)

    def __get_ddls(self):
        return self.__ddls
    ddls = cast(np.ndarray, property(__get_ddls))

    def __get_valeurs(self):
        return self.__valeurs
    valeurs = cast(np.ndarray, property(__get_valeurs))

    def __get_directions(self):
        return self.__directions
    directions = property(__get_directions)

