from typing import cast
import numpy as np

class Point:
    """Classe Point"""

    def __init__(self, x=0.0, y=0.0, z=0.0, isOpen=False):
        """Construit un point

        Parameters
        ----------
        x : float, optional
            coordo en x, by default 0.0
        y : float, optional
            coordo en y, by default 0.0
        z : float, optional
            coordo en z, by default 0.0
        isOpen : bool, optional
            le point peut s'ouvrir, by default False
        """
        self.x = x
        self.y = y
        self.z = z
        self.coordo = np.array([x, y, z]).reshape(1,3)
        self.isOpen = isOpen

class Line:
    """Clase Line"""

    @staticmethod
    def distance(pt1: Point, pt2: Point) -> float:
        """Calcul la distance entre deux points"""
        length = np.sqrt((pt1.x-pt2.x)**2 + (pt1.y-pt2.y)**2 + (pt1.z-pt2.z)**2)
        return np.abs(length)
    
    @staticmethod
    def get_vecteurUnitaire(pt1: Point, pt2: Point) -> np.ndarray:
        """Construit le vecteur unitaire qui passe entre deux points"""
        length = Line.distance(pt1, pt2)        
        v = np.array([pt2.x-pt1.x, pt2.y-pt1.y, pt2.z-pt1.z])/length
        return v   

    def __init__(self, pt1: Point, pt2: Point, taille=0.0):
        """Construit une ligne

        Parameters
        ----------
        pt1 : Point
            premier point
        pt2 : Point
            deuxième point
        taille : float, optional
            taille qui sera utilisée pour la construction du maillage, by default 0.0
        """
        self.pt1 = pt1
        self.pt2 = pt2
        self.coordo = np.array([[pt1.x, pt1.y, pt1.z], [pt2.x, pt2.y, pt2.z]]).reshape(2,3)

        assert taille >= 0
        self.taille = taille
    
    @property
    def vecteurUnitaire(self) -> np.ndarray:
        """Construction du vecteur unitaire pour les deux points de la ligne"""
        return Line.get_vecteurUnitaire(self.pt1, self.pt2)

    @property
    def length(self) -> float:
        """Calcul la longeur de la ligne"""
        return Line.distance(self.pt1, self.pt2)

class Domain:
    """Classe Domain"""

    def __init__(self, pt1: Point, pt2: Point, taille=0.0):
        """Construit d'un domaine entre 2 points\n
        Ce domaine n'est pas tourné !

        Parameters
        ----------
        pt1 : Point
            point 1
        pt2 : Point
            point 2
        taille : float, optional
            taille qui sera utilisée pour la construction du maillage, by default 0.0
        """
        self.pt1 = pt1
        self.pt2 = pt2

        assert taille >= 0
        self.taille = taille

class Circle:
    """Classe Circle"""

    def __init__(self, center: Point, diam: float, taille=0.0, isCreux=True):
        """Construction d'un cercle en fonction de son centre et de son diamètre \n
        Ce cercle sera projeté dans le plan (x,y)

        Parameters
        ----------
        center : Point
            point 1
        diam : float
            diamètre
        taille : float, optional
            taille qui sera utilisée pour la construction du maillage, by default 0.0
        isCreux : bool, optional
            le cercle est creux, by default True
        """
        
        assert diam > 0.0

        self.center = center
        self.diam = diam

        assert taille >= 0
        self.taille = taille

        self.isCreux = isCreux
