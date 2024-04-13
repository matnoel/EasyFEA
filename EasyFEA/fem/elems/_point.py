"""Point element module."""

from ... import np, plt

from .._group_elems import _GroupElem

class POINT(_GroupElem):
    
    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return super().origin

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def faces(self) -> list[int]:
        return [0]

    def _Ntild(self) -> np.ndarray:
        pass

    def _dNtild(self) -> np.ndarray:
        pass

    def _ddNtild(self) -> np.ndarray:
        pass
    
    def _dddNtild(self) -> np.ndarray:
        pass

    def _ddddNtild(self) -> np.ndarray:
        pass