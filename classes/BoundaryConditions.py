
from typing import cast
import numpy as np

class BoundaryConditions:    

    def __init__(self, problemType: str, name: str, ddls: np.ndarray, marker='.', color='blue'):
        
        assert problemType in ["damage", "displacement"]
        self.__problemType = problemType
        self.name = name
        self.__ddls = ddls
        self.marker = marker
        self.color = color   

    def __get_problemType(self):
        return self.__problemType
    problemType = property(__get_problemType)

    def __get_ddls(self):
        return self.__ddls
    ddls = cast(str, property(__get_ddls))

    def __get_directions(self):
        return self.__directions
    directions = property(__get_directions)