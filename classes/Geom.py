import numpy as np

class Point:

    def __init__(self, x=0.0, y=0.0, z=0.0):

        self.x = x
        self.y = y
        self.z = z

class Line:

    def distance(pt1: Point, pt2: Point):
        length = np.sqrt((pt1.x-pt2.x)**2 + (pt1.y-pt2.y)**2 + (pt1.z-pt2.z)**2)
        return length

    def __init__(self, pt1: Point, pt2: Point):
        self.pt1 = pt1
        self.pt2 = pt2

    def __get_length(self):        
        return Line.distance(self.pt1, self.pt2)
    length = property(__get_length)

class Domain:

    def __init__(self, pt1: Point, pt2: Point):
        self.pt1 = pt1
        self.pt2 = pt2