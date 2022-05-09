
from typing import cast
import numpy as np

class BoundaryCondition:

    def __init__(self, problemType: str,
    ddls: np.ndarray, directions: list, valeurs_ddls: np.ndarray,
    description: str, marker='.', color='blue'):

        assert problemType in ["damage", "displacement"]
        self.__problemType = problemType

        self.description = description

        self.__ddls = ddls

        self.__valeurs = valeurs_ddls
                
        self.__directions = directions

        self.marker = marker
        self.color = color

    def __get_problemType(self):
        return self.__problemType
    problemType = property(__get_problemType)

    def __get_ddls(self):
        return self.__ddls
    ddls = cast(np.ndarray, property(__get_ddls))

    def __get_valeurs(self):
        return self.__valeurs
    valeurs_ddls = cast(np.ndarray, property(__get_valeurs))

    def __get_directions(self):
        return self.__directions
    directions = property(__get_directions)

