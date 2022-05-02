from typing import cast
import numpy as np

class Point:

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.coordo = np.array([x, y, z]).reshape(1,3)

class Line:

    @staticmethod
    def distance(pt1: Point, pt2: Point):
        length = np.sqrt((pt1.x-pt2.x)**2 + (pt1.y-pt2.y)**2 + (pt1.z-pt2.z)**2)
        return cast(float, np.abs(length))
    
    @staticmethod
    def get_vecteurUnitaire(pt1: Point, pt2: Point):
        length = Line.distance(pt1, pt2)        
        v = np.array([pt2.x-pt1.x, pt2.y-pt1.y, pt2.z-pt1.z])/length
        return v   

    def __init__(self, pt1: Point, pt2: Point):
        self.pt1 = pt1
        self.pt2 = pt2
        self.coordo = np.array([[pt1.x, pt1.y, pt1.z], [pt2.x, pt2.y, pt2.z]]).reshape(2,3)
    
    def __get_vecteurUnitaire(self):        
        return Line.get_vecteurUnitaire(self.pt1, self.pt2)
    vecteurUnitaire = cast(np.ndarray, property(__get_vecteurUnitaire))

    def __get_length(self):        
        return Line.distance(self.pt1, self.pt2)
    length = cast(float, property(__get_length))

class Domain:

    def __init__(self, pt1: Point, pt2: Point):
        self.pt1 = pt1
        self.pt2 = pt2