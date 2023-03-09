import numpy as np

class Point:
    """Classe Point"""

    def __init__(self, x=0.0, y=0.0, z=0.0, isOpen=False, r=0.0):
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
        r : float, optional
            rayon utilisé pour le congé
        """
        self.__x = x
        self.__y = y
        self.__z = z        
        self.__r = r
        self.__coordo = np.array([x, y, z])
        self.__isOpen = isOpen

    @property
    def x(self) -> float:
        """coordo x du point"""
        return self.__x

    @property
    def y(self) -> float:
        """coordo y du point"""
        return self.__y

    @property
    def z(self) -> float:
        """coordo z du point"""
        return self.__z

    @property
    def r(self) -> float:
        """rayon utilisé pour le congé"""
        return self.__r

    @property
    def coordo(self) -> np.ndarray:
        """coordonnées x,y,z (3,)"""
        return self.__coordo

    @property
    def isOpen(self):
        """Le point est ouvert"""
        return self.__isOpen
    
    def __radd__(self, value):
        return self.__add__(value)

    def __add__(self, value):
        if isinstance(value, float) and isinstance(value, int):
            x = self.x + value
            y = self.y + value
            z = self.z + value
            return Point(x, y, z, self.isOpen, self.r)
        elif isinstance(value, list):
            x = self.x + value[0] if len(value) > 0 else self.x
            y = self.y + value[1] if len(value) > 1 else self.y
            z = self.z + value[2] if len(value) > 2 else self.z
            return Point(x, y, z, self.isOpen, self.r)
        elif isinstance(value, Point):
            x = self.x + value.x
            y = self.y + value.y
            z = self.z + value.z
            return Point(x, y, z, self.isOpen, self.r)
        elif isinstance(value, list[Point]):
            points = [] 
            for point in value:
                x = self.x + point.x
                y = self.y + point.y
                z = self.z + point.z
            points.append(Point(x, y, z, self.isOpen, self.r))
        elif isinstance(value, np.ndarray):
            points = []
            for x, y, z in zip(value[:,0], value[:,1], value[:,2]):
                x = self.x + x
                y = self.y + y
                z = self.z + z
                points.append(Point(x, y, z, self.isOpen, self.r))

    def __rsub__(self, value):
        return self.__add__(value)
    
    def __sub__(self, value):
        if isinstance(value, float) and isinstance(value, int):
            x = self.x - value
            y = self.y - value
            z = self.z - value
            return Point(x, y, z, self.isOpen, self.r)
        elif isinstance(value, list):
            x = self.x - value[0] if len(value) > 0 else self.x
            y = self.y - value[1] if len(value) > 1 else self.y
            z = self.z - value[2] if len(value) > 2 else self.z
            return Point(x, y, z, self.isOpen, self.r)
        elif isinstance(value, Point):
            x = self.x - value.x
            y = self.y - value.y
            z = self.z - value.z
            return Point(x, y, z, self.isOpen, self.r)
        elif isinstance(value, list[Point]):
            points = [] 
            for point in value:
                x = self.x - point.x
                y = self.y - point.y
                z = self.z - point.z
            points.append(Point(x, y, z, self.isOpen, self.r))
        elif isinstance(value, np.ndarray):
            points = []
            for x, y, z in zip(value[:,0], value[:,1], value[:,2]):
                x = self.x - x
                y = self.y - y
                z = self.z - z
                points.append(Point(x, y, z, self.isOpen, self.r))

    

class Geom:

    def __init__(self, points: list[Point], meshSize: float, name: str):
        """Construit un objet géométrique

        Parameters
        ----------
        points : list[Point]
            liste de points pour construire l'objet géométrique
        meshSize : float
            taille de maillage qui sera utilisé pour creer le maillage >= 0
        name : str
            nom de l'objet
        """
        assert meshSize >= 0
        self.__meshSize = meshSize

        self.__points = points

        self.__name = name

    @property
    def meshSize(self) -> bool:
        """Taille d'element utilisé pour le maillage"""
        return self.__meshSize

    @property
    def points(self) -> list[Point]:
        """Points utilisés pour construire l'objet"""
        return self.__points

    @property
    def name(self) -> str:
        """Nom de l'objet"""
        return self.__name

class PointsList(Geom):
    """Classe PointsList"""

    __nbPointsList = 0

    def __init__(self, points: list[Point], meshSize=0.0, isCreux=False):
        """Construit une liste de point

        Parameters
        ----------
        points : list[Point]
            liste de points pour construire l'objet géométrique
        meshSize : float, optional
            taille de maillage qui sera utilisé pour creer le maillage >= 0, by default 0.0
        isCreux : bool, optional
            le domaine formé est creux, by default False
        """

        self.isCreux=isCreux

        PointsList.__nbPointsList += 1
        name = f"PointsList{PointsList.__nbPointsList}"
        super().__init__(points, meshSize, name)

class Line(Geom):
    """Classe Line"""

    __nbLine = 0

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

    def __init__(self, pt1: Point, pt2: Point, meshSize=0.0, isOpen=False):
        """Construit une ligne

        Parameters
        ----------
        pt1 : Point
            premier point
        pt2 : Point
            deuxième point
        meshSize : float, optional
            taille qui sera utilisée pour la construction du maillage, by default 0.0
        """
        self.pt1 = pt1
        self.pt2 = pt2
        self.coordo = np.array([[pt1.x, pt1.y, pt1.z], [pt2.x, pt2.y, pt2.z]]).reshape(2,3)

        self.__isOpen = isOpen

        Line.__nbLine += 1
        name = f"Line{Line.__nbLine}"
        Geom.__init__(self, points=[pt1, pt2], meshSize=meshSize, name=name)
    

    @property
    def isOpen(self) -> bool:
        """Renvoie si la ligne peut s'ouvrir pour représenter une fissure"""
        return self.__isOpen
    
    @property
    def vecteurUnitaire(self) -> np.ndarray:
        """Construction du vecteur unitaire pour les deux points de la ligne"""
        return Line.get_vecteurUnitaire(self.pt1, self.pt2)

    @property
    def length(self) -> float:
        """Calcul la longeur de la ligne"""
        return Line.distance(self.pt1, self.pt2)

class Domain(Geom):
    """Classe Domain"""

    __nbDomain = 0

    def __init__(self, pt1: Point, pt2: Point, meshSize=0.0, isCreux=False):
        """Construit d'un domaine entre 2 points\n
        Ce domaine n'est pas tourné !

        Parameters
        ----------
        pt1 : Point
            point 1
        pt2 : Point
            point 2
        meshSize : float, optional
            taille qui sera utilisée pour la construction du maillage, by default 0.0
        isCreux : bool, optional
            le domaine est creux, by default False
        """
        self.pt1 = pt1
        self.pt2 = pt2

        self.isCreux = isCreux

        Domain.__nbDomain += 1
        name = f"Domain{Domain.__nbDomain}"
        Geom.__init__(self, points=[pt1, pt2], meshSize=meshSize, name=name)

class Circle(Geom):
    """Classe Circle"""

    __nbCircle = 0

    def __init__(self, center: Point, diam: float, meshSize=0.0, isCreux=True):
        """Construction d'un cercle en fonction de son centre et de son diamètre \n
        Ce cercle sera projeté dans le plan (x,y)

        Parameters
        ----------
        center : Point
            point 1
        diam : float
            diamètre
        meshSize : float, optional
            taille qui sera utilisée pour la construction du maillage, by default 0.0
        isCreux : bool, optional
            le cercle est creux, by default True
        """
        
        assert diam > 0.0

        self.center = center
        self.diam = diam
        
        self.isCreux = isCreux

        Circle.__nbCircle += 1
        name = f"Circle{Circle.__nbCircle}"
        Geom.__init__(self, points=[center], meshSize=meshSize, name=name)

class Section:

    def __init__(self, mesh):
        """Section

        Parameters
        ----------
        mesh : Mesh
            Maillage
        """

        from Mesh import Mesh
        assert isinstance(mesh, Mesh), "Doit être un maillage 2D"

        assert mesh.dim == 2, "Doit être un maillage 2D"
        
        self.__mesh = mesh

    @property
    def mesh(self):
        """Maillage de la section"""
        return self.__mesh

    @property
    def epaisseur(self) -> float:
        """Epaisseur de la section"""
        # ici l'epaisseur est suivant x
        coordo = self.__mesh.coordo
        epaisseur = np.abs(coordo[:,0].max() - coordo[:,0].min())
        return epaisseur
    
    @property
    def hauteur(self) -> float:
        """Hauteur de la section"""
        # ici l'epaisseur est suivant x
        coordo = self.__mesh.coordo
        hauteur = np.abs(coordo[:,1].max() - coordo[:,1].min())
        return hauteur
    
    @property
    def aire(self) -> float:
        """Surface de la section"""
        return self.__mesh.aire

    @property
    def Iy(self) -> float:
        """Moment quadratique de la section suivant y\n
        int_S z^2 dS """
        return self.__mesh.Ix

    @property
    def Iz(self) -> float:
        """Moment quadratique de la section suivant z\n
        int_S y^2 dS """
        return self.__mesh.Iy

    @property
    def Iyz(self) -> float:
        """Moment quadratique de la section suivant yz\n
        int_S y z dS """
        return self.__mesh.Ixy

    @property
    def J(self) -> float:
        """Moment quadratique polaire\n
        J = Iz + Iy
        """
        return self.__mesh.J

# Fonctions pour faire des caluls de distances d'angles etc

def normalize_vect(vect: np.ndarray) -> np.ndarray:
    """Renvoie le vecteur normalisé"""
    return vect / np.linalg.norm(vect)

def angleBetween_a_b(a: np.ndarray, b: np.ndarray) -> float:
    """calcul l'angle entre le vecteur a et le vecteur b
    https://math.stackexchange.com/questions/878785/how-to-find-an-angle-in-range0-360-between-2-vectors"""

    assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray), "a et b doivent être des np.array"
    assert a.shape == (3,) and b.shape == (3,), "a et b doivent être des vecteur 3D"
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    cosAngle = a.dot(b)/(norm_a * norm_b)
    sinAngle = np.cross(a,b)/(norm_a * norm_b)
    angle = np.arctan2(sinAngle, cosAngle)
    
    return angle[-1]

def matriceJacobienne(i: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Calcul la matrice jacobienne de passage de la base (i,j,k) vers la base (x,y,z)\n
    p(x,y) = matrice.dot(p(i,j))

    Parameters
    ----------
    i : np.ndarray
        vecteur i
    k : np.ndarray
        vecteur

    Returns
    -------
    np.ndarray
        La matrice jacobienne            
    """        

    i = normalize_vect(i)
    k = normalize_vect(k)
    
    j = np.cross(k, i)
    j = normalize_vect(j)

    F = np.zeros((3,3))

    F[:,0] = i
    F[:,1] = j
    F[:,2] = k

    return F

def PointsRayon(P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, r: float):
                
    # vecteurs
    i = P1-P0
    j = P2-P0
    
    n = np.cross(i, j) # vecteur normal au plan formé par i, j

    F = matriceJacobienne

    if r > 0:
        # angle de i vers k            
        betha = angleBetween_a_b(i, j)/2
        
        d = np.abs(r)/np.tan(betha) # disante entre P0 et A sur i et disante entre P0 et B sur j

        d *= np.sign(betha)

        A = F(i, n) @ np.array([d,0,0]) + P0
        B = F(j, n) @ np.array([d,0,0]) + P0
        C = F(i, n) @ np.array([d, r,0]) + P0
    else:
        d = np.abs(r)
        A = F(i, n) @ np.array([d,0,0]) + P0
        B = F(j, n) @ np.array([d,0,0]) + P0
        C = P0

    return A, B, C