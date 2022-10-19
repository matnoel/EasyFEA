from typing import List

from TicTac import Tic

from Mesh import Mesh, GroupElem
import CalcNumba as CalcNumba
import numpy as np
import Affichage as Affichage
from inspect import stack
from Geom import Poutre

from scipy.linalg import sqrtm

class LoiDeComportement(object):

    @staticmethod
    def get_LoisDeComportement():
        liste = [Elas_Isot, Elas_IsotTrans, Elas_Anisot]
        return liste

    @staticmethod
    def get_P(axis_1: np.ndarray, axis_2: np.ndarray):
        """Création de la matrice de changement de base P\n
        on utilise P pour passer des coordonnées du matériau au coordonnée global \n
        
        Tet que :\n

        C et S en [11, 22, 33, racine(2)*23, racine(2)*13, racine(2)*12]

        C_global = P' * C_materiau * P et S_global = P' * S_materiau * P

        Ici j'utilise Chevalier 1988 : Comportements élastique et viscoélastique des composites
        mais avec la transformation parce qu'on est en mandel\n

        Ici on peut se permettre d'écrire ça car on travail en mandel\n
        
        """

        axis_1 = axis_1/np.linalg.norm(axis_1)
        axis_2 = axis_2/np.linalg.norm(axis_2)

        # Detection si les 2 vecteurs sont bien perpendiculaires
        # tt = axis_1.dot(axis_2)
        if not np.isclose(axis_1.dot(axis_2), 0, 1e-12):
            theta = np.pi/2
            rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
            axis_2 = rot.dot(axis_1)
            tt = axis_1.dot(axis_2)
            print("Création d'un nouvel axe transverse. axis_l et axis_t ne sont pas perpendiculaire")

        axis_3 = np.cross(axis_1, axis_2, axis=0)
        axis_3 = axis_3/np.linalg.norm(axis_3)


        # Construit la matrice de chamgement de base x = [P] x'
        # P passe de la base global a la base du matériau
        p = np.zeros((3,3))
        p[:,0] = axis_1
        p[:,1] = axis_2
        p[:,2] = axis_3

        p11 = p[0,0]; p12 = p[0,1]; p13 = p[0,2]
        p21 = p[1,0]; p22 = p[1,1]; p23 = p[1,2]
        p31 = p[2,0]; p32 = p[2,1]; p33 = p[2,2]

        A = np.array([[p21*p31, p11*p31, p11*p21],
                      [p22*p32, p12*p32, p12*p22],
                      [p23*p33, p13*p33, p13*p23]])
        
        B = np.array([[p12*p13, p22*p23, p32*p33],
                      [p11*p13, p21*p23, p31*p33],
                      [p11*p12, p21*p22, p31*p32]])

        D1 = p.T**2

        D2 = np.array([[p22*p33 + p32*p23, p12*p33 + p32*p13, p12*p23 + p22*p13],
                       [p21*p33 + p31*p23, p11*p33 + p31*p13, p11*p23 + p21*p13],
                       [p21*p32 + p31*p22, p11*p32 + p31*p12, p11*p22 + p21*p12]])

        coef = np.sqrt(2)
        # coef = 1
        M = np.concatenate( (np.concatenate((D1, coef*A), axis=1),
                            np.concatenate((coef*B, D2), axis=1)), axis=0)        
        # print(f"M =\n{M}")
        
        return M

    """Classe des lois de comportements C de (Sigma = C * Epsilon)
    (Elas_isot, ...)
    """
    def __init__(self, dim: int, C: np.ndarray, S: np.ndarray, epaisseur: float):
        
        self.__dim = dim
        """dimension lié a la loi de comportement"""

        if dim == 2:
            assert epaisseur > 0 , "Doit être supérieur à 0"
            self.__epaisseur = epaisseur
        
        self.__C = C
        """Loi de comportement pour la loi de Lamé en kelvin mandel"""

        self.__S = S
        """Loi de comportement pour la loi de Hooke en kelvin mandel"""

    @property
    def coef(self) -> float:
        """Coef lié à la notation de kelvin mandel=racine(2)"""
        return np.sqrt(2)
    

    def get_C(self):
        """Renvoie une copie de la loi de comportement pour la loi de Lamé en Kelvin Mandel\n
        En 2D:
        -----
        C -> C : Epsilon = Sigma [Sxx, Syy, racine(2)*Sxy]\n
        En 3D:
        -----
        C -> C : Epsilon = Sigma [Sxx, Syy, Szz, racine(2)*Syz, racine(2)*Sxz, racine(2)*Sxy]
        """        
        return self.__C.copy()

    def get_S(self):
        """Renvoie une copie de la loi de comportement pour la loi de Hooke en Kelvin Mandel\n
        En 2D:
        -----        
        S -> S : Sigma = Epsilon [Exx, Eyy, racine(2)*Exy]\n
        En 3D:
        -----        
        S -> S : Sigma = Epsilon [Exx, Eyy, Ezz, racine(2)*Eyz, racine(2)*Exz, racine(2)*Exy]
        """
        return self.__S.copy()

    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def epaisseur(self) -> float:
        if self.__dim == 2:
            return self.__epaisseur
        else:
            return 1.0

    @property
    def nom(self) -> str:
        return type(self).__name__

    @property
    def resume(self) -> str:
        return ""

    @staticmethod
    def AppliqueCoefSurBrigi(dim: int, B_rigi_e_pg: np.ndarray):

        if dim == 2:
            coord=2
        elif dim == 3:
            coord=[3,4,5]
        else:
            raise "Pas implémenté"

        coef = np.sqrt(2)

        B_rigi_e_pg[:,:,coord,:] = B_rigi_e_pg[:,:,coord,:]/coef

        return B_rigi_e_pg
    
    @staticmethod
    def ApplyKelvinMandelCoefTo_Matrice(dim: int, Matrice: np.ndarray):        
        """Applique ces coefs à la matrice\n
        si 2D:
        \n
        [1,1,r2]\n
        [1,1,r2]\n
        [r2, r2, 2]]\n

        \nsi 3D:
        \n
        [1,1,1,r2,r2,r2]\n
        [1,1,1,r2,r2,r2]\n
        [1,1,1,r2,r2,r2]\n
        [r2,r2,r2,2,2,2]\n
        [r2,r2,r2,2,2,2]\n
        [r2,r2,r2,2,2,2]]\n
        
        """

        r2 = np.sqrt(2)

        if dim == 2:            
            transform = np.array([  [1,1,r2],
                                    [1,1,r2],
                                    [r2, r2, 2]])
        elif dim == 3:
            transform = np.array([  [1,1,1,r2,r2,r2],
                                    [1,1,1,r2,r2,r2],
                                    [1,1,1,r2,r2,r2],
                                    [r2,r2,r2,2,2,2],
                                    [r2,r2,r2,2,2,2],
                                    [r2,r2,r2,2,2,2]])
        else:
            raise "Pas implémenté"

        matriceMandelCoef = Matrice*transform

        return matriceMandelCoef

    @staticmethod
    def Apply_P(P: np.ndarray, Matrice: np.ndarray):
        matrice_P = np.einsum('ji,jk,kl->il',P, Matrice, P, optimize='optimal')

        # on verfie que les invariants du tenseur ne change pas !
        # if np.linalg.norm(P.T-P) <= 1e-12:
        test_det_c = np.linalg.det(matrice_P) - np.linalg.det(Matrice)
        assert test_det_c <1e-12
        test_trace_c = np.trace(matrice_P) - np.trace(Matrice)
        assert test_trace_c <1e-12
        
        return matrice_P

class Elas_Isot(LoiDeComportement):   

    def __init__(self, dim: int, E=210000.0, v=0.3, contraintesPlanes=True, epaisseur=1.0):
        """Creer la matrice de comportement d'un matériau : Elastique isotrope

        Parameters
        ----------
        dim : int
            Dimension de la simulation 2D ou 3D
        E : float, optional
            Module d'elasticité du matériau en MPa (> 0)
        v : float, optional
            Coef de poisson ]-1;0.5]
        contraintesPlanes : bool
            Contraintes planes si dim = 2 et True, by default True        
        """       

        # Vérification des valeurs
        assert dim in [2,3], "doit être en 2 et 3"
        self.__dim = dim

        assert E > 0.0, "Le module élastique doit être > 0 !"
        self.E=E
        """Module de Young"""

        poisson = "Le coef de poisson doit être compris entre ]-1;0.5["
        assert v > -1.0 and v < 0.5, poisson
        self.v=v
        """Coef de poisson"""        

        if dim == 2:
            self.contraintesPlanes = contraintesPlanes
            """type de simplification 2D"""

        C, S = self.__Comportement()

        LoiDeComportement.__init__(self, dim, C, S, epaisseur)

    @property
    def resume(self) -> str:
        resume = f"\nElas_Isot :"
        resume += f"\nE = {self.E:.2e}, v = {self.v}"
        if self.__dim == 2:
            resume += f"\nCP = {self.contraintesPlanes}, ep = {self.epaisseur:.2e}"            
        return resume
    

    def get_lambda(self):

        E=self.E
        v=self.v
        
        l = E*v/((1+v)*(1-2*v))

        if self.__dim == 2 and self.contraintesPlanes:
            l = E*v/(1-v**2)
        
        return l
    
    def get_mu(self):
        """Coef de cisaillement"""
        
        E=self.E
        v=self.v

        mu = E/(2*(1+v))

        return mu
    
    def get_bulk(self):
        """Module de compression"""

        E=self.E
        v=self.v

        mu = self.get_mu()
        l = self.get_lambda()
        
        bulk = l + 2*mu/self.dim        

        return bulk

    def __Comportement(self):
        """"Construit les matrices de comportement en kelvin mandel\n
        
        En 2D:
        -----

        C -> C : Epsilon = Sigma [Sxx Syy racine(2)*Sxy]\n
        S -> S : Sigma = Epsilon [Exx Eyy racine(2)*Exy]

        En 3D:
        -----

        C -> C : Epsilon = Sigma [Sxx Syy Szz racine(2)*Syz racine(2)*Sxz racine(2)*Sxy]\n
        S -> S : Sigma = Epsilon [Exx Eyy Ezz racine(2)*Eyz racine(2)*Exz racine(2)*Exy]

        """

        E=self.E
        v=self.v

        dim = self.__dim

        mu = self.get_mu()
        l = self.get_lambda()

        if dim == 2:

            # Attention ici ça marche car lambda change en fonction de la simplification 2D

            cVoigt = np.array([ [l + 2*mu, l, 0],
                                [l, l + 2*mu, 0],
                                [0, 0, mu]])

            # if self.contraintesPlanes:
            #     # C = np.array([  [4*(mu+l), 2*l, 0],
            #     #                 [2*l, 4*(mu+l), 0],
            #     #                 [0, 0, 2*mu+l]]) * mu/(2*mu+l)

            #     cVoigt = np.array([ [1, v, 0],
            #                         [v, 1, 0],
            #                         [0, 0, (1-v)/2]]) * E/(1-v**2)
                
            # else:
            #     cVoigt = np.array([ [l + 2*mu, l, 0],
            #                         [l, l + 2*mu, 0],
            #                         [0, 0, mu]])

            #     # C = np.array([  [1, v/(1-v), 0],
            #     #                 [v/(1-v), 1, 0],
            #     #                 [0, 0, (1-2*v)/(2*(1-v))]]) * E*(1-v)/((1+v)*(1-2*v))

        elif dim == 3:
            
            cVoigt = np.array([ [l+2*mu, l, l, 0, 0, 0],
                                [l, l+2*mu, l, 0, 0, 0],
                                [l, l, l+2*mu, 0, 0, 0],
                                [0, 0, 0, mu, 0, 0],
                                [0, 0, 0, 0, mu, 0],
                                [0, 0, 0, 0, 0, mu]])
        
        c = LoiDeComportement.ApplyKelvinMandelCoefTo_Matrice(dim, cVoigt)

        s = np.linalg.inv(c)

        return c, s

class BeamModel():   

    def __init__(self, dim: int, listePoutres: List[Poutre], list_E: List[float], list_v=[]):
        """Creation du model poutre

        Parameters
        ----------
        dim : int
            dimension utilisée [1,2,3]
        E : float, optional
            Module d'elasticité du matériau en MPa (> 0)
        v : float, optional
            Coef de poisson ]-1;0.5]
        """

        # Effectue les verifications
        assert len(listePoutres) == len(list_E), "Doit fournir autant de coef matériau que de poutres"
        assert len(listePoutres) == len(list_v), "Doit fournir autant de coef matériau que de poutres"

        for E in list_E: assert E > 0, "Le module élastique doit être > 0 !" 
        for v in list_v: assert v > -1.0 and v < 0.5, "Le coef de poisson doit être compris entre ]-1;0.5["
        
        self.__dim = dim
        self.__listePoutres = listePoutres
        self.__list_E = list_E
        self.__list_v = list_v

        self.__list_D = []

        for poutre, E, v in zip(listePoutres, list_E, list_v):

            assert isinstance(poutre, Poutre)
            A = poutre.section.aire
        
            if dim == 1:
                # u = [u1, . . . , un]
                D = np.diag([E*A])
            elif dim == 2:
                # u = [u1, v1, rz1, . . . , un, vn, rzn]
                Iz = poutre.section.Iz
                D = np.diag([E*A, E*Iz])
            elif dim == 3:
                # u = [u1, v1, w1, rx1, ry1 rz1, . . . , un, vn, wn, rxn, ryn rzn]
                Iy = poutre.section.Iy
                Iz = poutre.section.Iz
                J = poutre.section.J
                mu = E/(2*(1+v))
                D = np.diag([E*A, mu*J, E*Iy, E*Iz])
            
            self.__list_D.append(D)

    def Calc_D_e_pg(self, groupElem: GroupElem, matriceType: str):
        # Construction de D_e_pg: 
        listePoutres = self.__listePoutres
        list_D = self.__list_D
        # Pour chaque poutre, on va construire la loi de comportement
        Ne = groupElem.Ne
        nPg = groupElem.get_gauss(matriceType).nPg
        D_e_pg = np.zeros((Ne, nPg, list_D[0].shape[0], list_D[0].shape[0]))
        for poutre, D in zip(listePoutres, list_D):
            # recupère les element
            elements = groupElem.Get_Elements_PhysicalGroup(poutre.idPoutre)
            D_e_pg[elements] = D

        return D_e_pg

    @property
    def dim(self) -> int:
        """Dimension du model \n
        1D -> traction compression \n 
        2D -> traction compression + fleche + flexion \n
        3D -> tout \n"""
        return self.__dim

    @property
    def nbddl_n(self) -> int:
        """Nombdre de ddl par noeud
        1D -> [u1, . . ., un]\n
        2D -> [u1, v1, rz1, . . ., un, vn, rzn]\n
        3D -> [u1, v1, w1, rx1, ry1, rz1, . . ., u2, v2, w2, rx2, ry2, rz2]"""
        if self.__dim == 1:
            return 1 # u
        elif self.__dim == 2:
            return 3 # u v rz
        elif self.__dim == 3:
            return 6 # u v w rx ry rz
        return self.__dim
    
    @property
    def listePoutres(self) -> List[Poutre]:
        """Liste des poutres"""
        return self.__listePoutres

    @property
    def nbPoutres(self) -> int:
        """Nombre de poutre"""
        return len(self.__listePoutres)

    @property
    def liste_E(self) -> List[float]:
        """Liste des modules élastiques"""
        return self.__liste_E

    @property
    def liste_v(self) -> List[float]:
        """Liste des coef de poisson"""
        return self.__liste_v

    @property
    def list_D(self) -> List[np.ndarray]:
        """liste de loi de comportement"""
        return self.__list_D

    @property
    def resume(self) -> str:
        resume = f"\nModel poutre:"
        resume += f"\nNombre de Poutre = {self.nbPoutres} :\n"
        # Réalise un résumé pour chaque poutre
        for poutre, E, v in zip(self.__listePoutres, self.__list_E, self.__list_v):
            resume += poutre.resume
            if isinstance(E, int):
                resume += f"\n\tE = {E:6}, v = {v}"
            else:
                resume += f"\n\tE = {E:6.2}, v = {v}"
        return resume

class Elas_IsotTrans(LoiDeComportement):

    def __init__(self, dim: int, El: float, Et: float, Gl: float, vl: float, vt: float, axis_l=np.array([1,0,0]), axis_t=np.array([0,1,0]), contraintesPlanes=True, epaisseur=1.0):

        # Vérification des valeurs
        assert dim in [2,3], "doit être en 2 et 3"
        self.__dim = dim

        erreurCoef = f"Les modules El, Et et Gl doivent être > 0 !"
        for i, E in enumerate([El, Et, Gl]): assert E > 0.0, erreurCoef
        self.El=El
        """Module de Young longitudinale"""
        self.Et=Et
        """Module de Young transverse"""
        self.Gl=Gl
        """Module de Cisaillent longitudinale"""

        erreurPoisson = lambda i :f"Les coefs de poisson vt et vl doivent être compris entre ]-1;0.5["
        # TODO a mettre à jour Peut ne pas être vrai ? -> J'ai vu que cetait de -1 a 1
        for v in [vl, vt]: assert v > -1.0 and v < 0.5, erreurPoisson
        # -1<vt<1
        # -1<vl<0.5
        # Regarder torquato 328
        self.vl=vl
        """Coef de poisson longitudianale"""
        self.vt=vt
        """Coef de poisson transverse"""

        if dim == 2:
            self.contraintesPlanes = contraintesPlanes
            """type de simplification 2D"""

        # Création de la matrice de changement de base

        self.__axis1 = axis_l
        self.__axis2 = axis_t
        
        P = self.get_P(axis_1=axis_l, axis_2=axis_t)

        if np.linalg.norm(axis_l-np.array([1,0,0]))<1e-12 and np.linalg.norm(axis_t-np.array([0,1,0]))<1e-12:
            useSameAxis=True
        else:
            useSameAxis=False

        C, S = self.__Comportement(P, useSameAxis)

        LoiDeComportement.__init__(self, dim, C, S, epaisseur)

    @property
    def Gt(self) -> float:
        
        Et = self.Et
        vt = self.vt

        Gt = Et/(2*(1+vt))

        return Gt

    @property
    def kt(self) -> float:
        # Source : torquato 2002
        El = self.El
        Et = self.Et
        vtt = self.vt
        vtl = self.vl
        kt = El*Et/((2*(1-vtt)*El)-(4*vtl**2*Et))

        return kt

    @property
    def resume(self) -> str:
        resume = f"\nElas_IsotTrans :"
        resume += f"\nEl = {self.El:.2e}, Et = {self.El:.2e}, Gl = {self.Gl:.2e}"
        resume += f"\nvl = {self.vl}, vt = {self.vt}"
        resume += f"\naxi_l = {self.__axis1},  axi_t = {self.__axis2}"
        if self.__dim == 2:
            resume += f"\nCP = {self.contraintesPlanes}, ep = {self.epaisseur:.2e}"            
        return resume

    def __Comportement(self, P, useSameAxis: bool):
        """"Construit les matrices de comportement en kelvin mandel\n
        
        En 2D:
        -----

        C -> C : Epsilon = Sigma [Sxx Syy racine(2)*Sxy]\n
        S -> S : Sigma = Epsilon [Exx Eyy racine(2)*Exy]

        En 3D:
        -----

        C -> C : Epsilon = Sigma [Sxx Syy Szz racine(2)*Syz racine(2)*Sxz racine(2)*Sxy]\n
        S -> S : Sigma = Epsilon [Exx Eyy Ezz racine(2)*Eyz racine(2)*Exz racine(2)*Exy]

        """

        dim = self.__dim

        El = self.El
        Et = self.Et
        vt = self.vt
        vl = self.vl
        Gl = self.Gl
        Gt = self.Gt

        kt = self.kt

        # Matrice de souplesse et de rigidité en mandel dans la base du matériau
        # [11, 22, 33, sqrt(2)*23, sqrt(2)*13, sqrt(2)*12]

        material_sM = np.array([[1/El, -vl/El, -vl/El, 0, 0, 0],
                      [-vl/El, 1/Et, -vt/Et, 0, 0, 0],
                      [-vl/El, -vt/Et, 1/Et, 0, 0, 0],
                      [0, 0, 0, 1/(2*Gt), 0, 0],
                      [0, 0, 0, 0, 1/(2*Gl), 0],
                      [0, 0, 0, 0, 0, 1/(2*Gl)]])

        material_cM = np.array([[El+4*vl**2*kt, 2*kt*vl, 2*kt*vl, 0, 0, 0],
                      [2*kt*vl, kt+Gt, kt-Gt, 0, 0, 0],
                      [2*kt*vl, kt-Gt, kt+Gt, 0, 0, 0],
                      [0, 0, 0, 2*Gt, 0, 0],
                      [0, 0, 0, 0, 2*Gl, 0],
                      [0, 0, 0, 0, 0, 2*Gl]])

        # # Verifie que C = S^-1
        # assert np.linalg.norm(material_sM - np.linalg.inv(material_cM)) < 1e-10        
        # assert np.linalg.norm(material_cM - np.linalg.inv(material_sM)) < 1e-10

        # Effectue le changement de base pour orienter le matériau dans lespace
        global_sM = self.Apply_P(P, material_sM)
        global_cM = self.Apply_P(P, material_cM)
        
        # verification que si les axes ne change pas on obtient bien la meme loi de comportement
        test_diff_c = global_cM - material_cM
        if useSameAxis: assert(np.linalg.norm(test_diff_c)<1e-12)

        # verification que si les axes ne change pas on obtient bien la meme loi de comportement
        test_diff_s = global_sM - material_sM
        if useSameAxis: assert np.linalg.norm(test_diff_s) < 1e-12
        
        c = global_cM
        s = global_sM

        if dim == 2:
            x = np.array([0,1,5])
            
            if self.contraintesPlanes == True:                
                s = global_sM[x,:][:,x]
                c = np.linalg.inv(s)
            else:
                c = global_cM[x,:][:,x]
                # s = global_sM[x,:][:,x]
                s = np.linalg.inv(c)

                # testS = np.linalg.norm(s-s2)/np.linalg.norm(s2)
            
        
        return c, s

class Elas_Anisot(LoiDeComportement):   

    def __init__(self, dim: int, C_voigt: np.ndarray, axis1:np.ndarray, axis2=None, contraintesPlanes=True, epaisseur=1.0):
        """Création d'une loi de comportement elastique anisotrope

        Parameters
        ----------
        dim : int
            dimension
        C_voigt : np.ndarray
            matrice de rigidité en notation de voigt dans la base d'anisotropie
        axis1 : np.ndarray
            vecteur de l'axe1
        axis2 : np.ndarray, optional
            vecteur de l'axe2, by default None
        contraintesPlanes : bool, optional
            simplification 2D, by default True
        epaisseur : float, optional
            epaisseur du matériau, by default 1.0

        Returns
        -------
        Elas_Anisot
            Loi de comportemet anisotrope
        """

        # Vérification des valeurs
        assert dim in [2,3], "doit être en 2 et 3"
        self.__dim = dim
        
        # Verification sur la matrice
        if dim == 2:
            assert C_voigt.shape == (3,3), "La matrice doit être de dimension 3x3"
        else:
            assert C_voigt.shape == (6,6), "La matrice doit être de dimension 6x6"
        assert np.linalg.norm(C_voigt.T - C_voigt) <= 1e-12, "La matrice n'est pas symétrique"

        # Verification et construction des vecteurs
        assert axis1.size == 3, "Doit fournir un vecteur" 
        self.__axis1 = axis1
        def Calc_axis2():
            theta = np.pi/2
            rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
            axis2 = rot.dot(axis1)
            return axis2

        if axis2 == None:
            axis2 = Calc_axis2()
        else:
            assert axis2.size == 3, "Doit fournir un vecteur"
            if not np.isclose(axis1.dot(axis2), 0, 1e-12):
                axis2 = Calc_axis2()

        self.__axis2 = axis2

        # Construction de la matrice de rotation
        P = self.get_P(axis_1=axis1, axis_2=axis2)

        # Application des coef
        C_mandel = self.ApplyKelvinMandelCoefTo_Matrice(dim, C_voigt)

        # Passage de la matrice en 3D pour faire la rotation

        if dim == 2:

            C_mandel_global = np.zeros((6,6))

            listIndex = np.array([0,1,5])
            
            for i, I in enumerate(listIndex):
                for j, J in enumerate(listIndex):
                    C_mandel_global[I,J] = C_mandel[i, j]

        else:
            C_mandel_global = C_voigt

        C_mandelP_global = self.Apply_P(P, C_mandel_global)

        if dim == 2:
            listIndex = np.array([0,1,5])
            C_mandelP = C_mandelP_global[listIndex, :][:, listIndex]
        else:
            C_mandelP = C_mandelP_global

        S_mandelP = np.linalg.inv(C_mandelP)

        if dim == 2:
            self.contraintesPlanes = contraintesPlanes
            """type de simplification 2D"""

        LoiDeComportement.__init__(self, dim, C_mandelP, S_mandelP, epaisseur)

    @property
    def resume(self) -> str:
        resume = f"\nElas_Anisot :"
        resume += f"\n{self.get_C()}"
        resume += f"\naxi1 = {self.__axis1},  axi2 = {self.__axis2}"
        if self.__dim == 2:
            resume += f"\nCP = {self.contraintesPlanes}, ep = {self.epaisseur:.2e}"
        return resume


class PhaseFieldModel:

    @staticmethod
    def get_splits() -> List[str]:
        """splits disponibles"""
        __splits = ["Bourdin","Amor"]
        __splits.extend(["Miehe","He","Stress"])
        __splits.extend(["AnisotMiehe","AnisotMiehe_PM","AnisotMiehe_MP","AnisotMiehe_NoCross"])
        __splits.extend(["AnisotStress","AnisotStress_PM","AnisotStress_MP","AnisotStress_NoCross"])
        
        return __splits
    
    @staticmethod
    def get_regularisations() -> List[str]:
        """regularisations disponibles"""
        __regularizations = ["AT1","AT2"]
        return __regularizations

    @staticmethod
    def get_solveurs() -> List[str]:
        """solveurs disponibles"""
        __solveurs = ["History", "HistoryDamage", "BoundConstrain"]
        return __solveurs

    @property
    def k(self) -> float:
        Gc = self.__Gc
        l0 = self.__l0

        k = Gc * l0

        if self.__regularization == "AT1":
            k = 3/4 * k

        return k

    def get_r_e_pg(self, PsiP_e_pg: np.ndarray) -> np.ndarray:
        Gc = self.__Gc
        l0 = self.__l0

        r = 2 * PsiP_e_pg

        if self.__regularization == "AT2":
            r = r + (Gc/l0)
        
        return r

    def get_f_e_pg(self, PsiP_e_pg: np.ndarray) -> np.ndarray:
        Gc = self.__Gc
        l0 = self.__l0

        f = 2 * PsiP_e_pg

        if self.__regularization == "AT1":            
            f = f - ( (3*Gc) / (8*l0) )            
            absF = np.abs(f)
            f = (f+absF)/2
        
        return f

    def get_g_e_pg(self, d_n: np.ndarray, mesh: Mesh, matriceType: str, k_residu=1e-12) -> np.ndarray:
        """Fonction de dégradation en energies / contraintes
        k_residu=1e-10
        Args:
            d_n (np.ndarray): Endomagement localisé aux noeuds (Nn,1)
            mesh (Mesh): maillage
        """
        d_e_n = mesh.Localises_sol_e(d_n)
        Nd_pg = mesh.Get_N_scalaire_pg(matriceType)

        d_e_pg = np.einsum('pij,ej->ep', Nd_pg, d_e_n, optimize='optimal')        

        if self.__regularization in ["AT1","AT2"]:
            g_e_pg = (1-d_e_pg)**2 + k_residu
        else:
            raise "Pas implémenté"

        assert mesh.Ne == g_e_pg.shape[0]
        assert mesh.Get_nPg(matriceType) == g_e_pg.shape[1]
        
        return g_e_pg

    @property
    def resume(self) -> str:
        resum = '\nPhaseField :'        
        resum += f'\nsplit : {self.__split}'
        resum += f'\nregularisation : {self.__regularization}'
        resum += f'\nGc : {self.__Gc:.2e}'
        resum += f'\nl0 : {self.__l0:.2e}'
        return resum

    @property
    def split(self) -> str:
        return self.__split

    @property
    def regularization(self) -> str:
        return self.__regularization
    
    @property
    def comportement(self) -> LoiDeComportement:
        return self.__comportement

    @property
    def solveur(self):
        """Solveur d'endommagement"""
        return self.__solveur

    @property
    def Gc(self):
        """Taux de libération d'énergie critque [J/m^2]"""
        return self.__Gc

    @property
    def l0(self):
        """Largeur de régularisation de la fissure"""
        return self.__l0

    @property
    def c0(self):
        """Paramètre de mise à l'échelle permettant dissiper exactement l'énergie de fissure"""
        if self.__regularization == "AT1":
            c0 = 8/3
        elif self.__regularization == "AT2":
            c0 = 2
        return c0
    
    @property
    def useNumba(self) -> bool:
        return self.__useNumba
    
    @useNumba.setter
    def useNumba(self, val: bool):
        self.__useNumba = val

    def __init__(self, comportement: LoiDeComportement,split: str, regularization: str, Gc: float, l_0: float,solveur="History"):
        """Crétation d'un modèle à gradient d'endommagement

        Parameters
        ----------
        loiDeComportement : LoiDeComportement
            Loi de comportement du matériau (Elas_Isot, Elas_IsotTrans)
        split : str
            Split de la densité d'energie élastique (voir PhaseFieldModel.get_splits())
        regularization : str
            Modèle de régularisation de la fissure AT1 ou AT2
        Gc : float
            Taux de restitution d'energie critique en J.m^-2
        l_0 : float
            Demie largeur de fissure 
        solveur : str, optional
            Type de résolution de l'endommagement, by default "History" (voir PhaseFieldModel.get_solveurs())
        """
    
        assert isinstance(comportement, LoiDeComportement), "Doit être une loi de comportement"
        self.__comportement = comportement

        assert split in PhaseFieldModel.get_splits(), f"Doit être compris dans {PhaseFieldModel.get_splits()}"
        if not isinstance(comportement, Elas_Isot):
            assert not split in ["Amor", "Miehe", "Stress"], "Ces splits ne sont implémentés que pour Elas_Isot"
        self.__split =  split
        """Split de la densité d'energie elastique"""
        
        assert regularization in PhaseFieldModel.get_regularisations(), f"Doit être compris dans {PhaseFieldModel.get_regularisations()}"
        self.__regularization = regularization
        """Modèle de régularisation de la fissure ["AT1","AT2"]"""

        assert Gc > 0, "Doit être supérieur à 0" 
        self.__Gc = Gc
        """Taux de libération d'énergie critque [J/m^2]"""

        assert l_0 > 0, "Doit être supérieur à 0"
        self.__l0 = l_0
        """Largeur de régularisation de la fissure"""

        self.__solveur = solveur
        """Solveur d'endommagement"""

        self.__useNumba = False
        """Utilise ou non les fonctions numba"""
        
            
    def Calc_psi_e_pg(self, Epsilon_e_pg: np.ndarray):
        """Calcul de la densité d'energie elastique\n
        psiP_e_pg = 1/2 SigmaP_e_pg * Epsilon_e_pg\n
        psiM_e_pg = 1/2 SigmaM_e_pg * Epsilon_e_pg\n
        Tel que :\n
        SigmaP_e_pg = cP_e_pg * Epsilon_e_pg\n
        SigmaM_e_pg = cM_e_pg * Epsilon_e_pg        
        """
        
        # Data
        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]

        SigmaP_e_pg, SigmaM_e_pg = self.Calc_Sigma_e_pg(Epsilon_e_pg)

        useNumba = self.__useNumba
        # useNumba = False

        tic = Tic()

        if useNumba:
            # Plus rapide
            psiP_e_pg, psiM_e_pg = CalcNumba.Calc_psi_e_pg(Epsilon_e_pg, SigmaP_e_pg, SigmaM_e_pg)
        else:
            psiP_e_pg = 1/2 * np.einsum('epi,epi->ep', SigmaP_e_pg, Epsilon_e_pg, optimize='optimal').reshape((Ne, nPg))
            psiM_e_pg = 1/2 * np.einsum('epi,epi->ep', SigmaM_e_pg, Epsilon_e_pg, optimize='optimal').reshape((Ne, nPg))

        tic.Tac("Matrices PFM", "psiP_e_pg et psiM_e_pg", False)

        return psiP_e_pg, psiM_e_pg

    def Calc_Sigma_e_pg(self, Epsilon_e_pg: np.ndarray):
        """Calcul la contrainte en fonction de la deformation et du split\n
        Ici on calcul :\n
        SigmaP_e_pg = cP_e_pg * Epsilon_e_pg \n
        SigmaM_e_pg = cM_e_pg * Epsilon_e_pg

        Parameters
        ----------
        Epsilon_e_pg : np.ndarray
            deformations stockées aux elements et Points de Gauss

        Returns
        -------
        np.ndarray
            SigmaP_e_pg, SigmaM_e_pg : les contraintes stockées aux elements et Points de Gauss
        """       

        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]
        comp = Epsilon_e_pg.shape[2]

        useNumba = self.__useNumba
        # useNumba = False

        cP_e_pg, cM_e_pg = self.Calc_C(Epsilon_e_pg)

        tic = Tic()

        if useNumba:
            # Plus rapide
            SigmaP_e_pg, SigmaM_e_pg = CalcNumba.Calc_Sigma_e_pg(Epsilon_e_pg, cP_e_pg, cM_e_pg)
        else:
            SigmaP_e_pg = np.einsum('epij,epj->epi', cP_e_pg, Epsilon_e_pg, optimize='optimal').reshape((Ne, nPg, comp))
            SigmaM_e_pg = np.einsum('epij,epj->epi', cM_e_pg, Epsilon_e_pg, optimize='optimal').reshape((Ne, nPg, comp))

        tic.Tac("Matrices PFM", "SigmaP_e_pg et SigmaM_e_pg", False)

        return SigmaP_e_pg, SigmaM_e_pg
    
    def Calc_C(self, Epsilon_e_pg: np.ndarray, verif=False):
        """Calcul la loi de comportement en fonction du split

        Parameters
        ----------
        Epsilon_e_pg : np.ndarray
            deformations stockées aux élements et points de gauss (Pas utilisé si bourdin)

        Returns
        -------
        np.ndarray
            Renvoie cP_e_pg, cM_e_pg
        """

        # Ici faire en sorte que l'on passe que 1 fois par itération pour eviter de faire les calculs plusieurs fois
        # Il se trouve que ça ne marche pas l'endommagement n'évolue pas
        # On passe ici 2 fois par itération
        # Une fois pour calculer l'energie et une fois pour calculer K_u

        tic = Tic()

        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]
            
        if self.__split == "Bourdin":
            c = self.__comportement.get_C()
            c = c[np.newaxis, np.newaxis,:,:]
            c = np.repeat(c, Ne, axis=0)
            c = np.repeat(c, nPg, axis=1)

            cP_e_pg = c
            cM_e_pg = np.zeros_like(cP_e_pg)

        elif self.__split == "Amor":
            cP_e_pg, cM_e_pg = self.__Split_Amor(Epsilon_e_pg)

        elif self.__split in ["Miehe","AnisotMiehe","AnisotMiehe_PM","AnisotMiehe_MP","AnisotMiehe_NoCross"]:
            cP_e_pg, cM_e_pg = self.__Split_Miehe(Epsilon_e_pg, verif=verif)
        
        elif self.__split in ["Stress","AnisotStress","AnisotStress_PM","AnisotStress_MP","AnisotStress_NoCross"]:
            cP_e_pg, cM_e_pg = self.__Split_Stress(Epsilon_e_pg, verif=verif)

        elif self.__split in ["He","HeStress"]:
            cP_e_pg, cM_e_pg = self.__Split_He(Epsilon_e_pg, verif=verif)
        
        else: 
            raise "Split inconnue"

        fonctionQuiAppelle = stack()[2].function

        if fonctionQuiAppelle == "Calc_psi_e_pg":
            matrice = "masse"
        else:
            matrice = "rigi"


        # tic.Tac("Matrices PFM",f"cP_e_pg et cM_e_pg ({matrice})", False)
        tic.Tac("Matrices PFM",f"cP_e_pg et cM_e_pg", False)

        return cP_e_pg, cM_e_pg

    def __Split_Amor(self, Epsilon_e_pg: np.ndarray):

        assert isinstance(self.__comportement, Elas_Isot), f"Implémenté que pour un matériau Elas_Isot"
        
        loiDeComportement = self.__comportement                

        bulk = loiDeComportement.get_bulk()
        mu = loiDeComportement.get_mu()

        Rp_e_pg, Rm_e_pg = self.__Rp_Rm(Epsilon_e_pg)

        dim = self.__comportement.dim

        if dim == 2:
            Ivoigt = np.array([1,1,0]).reshape((3,1))
            taille = 3
        else:
            Ivoigt = np.array([1,1,1,0,0,0]).reshape((6,1))
            taille = 6

        IxI = np.array(Ivoigt.dot(Ivoigt.T))

        # Projecteur deviatorique
        Pdev = np.eye(taille) - 1/dim * IxI
        partieDeviateur = 2*mu*Pdev

        # projetcteur spherique
        # useNumba = self.__useNumba
        useNumba=False
        if useNumba:
            # Moins rapide
            cP_e_pg, cM_e_pg = CalcNumba.Split_Amor(Rp_e_pg, Rm_e_pg, partieDeviateur, IxI, bulk)
        else:
            
            # # Mois rapide que einsum
            # IxI = IxI[np.newaxis, np.newaxis, :, :].repeat(Rp_e_pg.shape[0], axis=0).repeat(Rp_e_pg.shape[1], axis=1)

            # Rp_e_pg = Rp_e_pg[:,:, np.newaxis, np.newaxis].repeat(taille, axis=2).repeat(taille, axis=3)
            # Rm_e_pg = Rm_e_pg[:,:, np.newaxis, np.newaxis].repeat(taille, axis=2).repeat(taille, axis=3)

            # spherP_e_pg = Rp_e_pg * IxI
            # spherM_e_pg = Rm_e_pg * IxI

            spherP_e_pg = np.einsum('ep,ij->epij', Rp_e_pg, IxI, optimize='optimal')
            spherM_e_pg = np.einsum('ep,ij->epij', Rm_e_pg, IxI, optimize='optimal')            
        
            cP_e_pg = bulk*spherP_e_pg + partieDeviateur
            cM_e_pg = bulk*spherM_e_pg

        return cP_e_pg, cM_e_pg

    def __Rp_Rm(self, vecteur_e_pg: np.ndarray):
        """Renvoie Rp_e_pg, Rm_e_pg"""

        Ne = vecteur_e_pg.shape[0]
        nPg = vecteur_e_pg.shape[1]

        dim = self.__comportement.dim

        tr_Eps = np.zeros((Ne, nPg))

        tr_Eps = vecteur_e_pg[:,:,0] + vecteur_e_pg[:,:,1]

        if dim == 3:
            tr_Eps += vecteur_e_pg[:,:,2]

        Rp_e_pg = (1+np.sign(tr_Eps))/2
        Rm_e_pg = (1+np.sign(-tr_Eps))/2

        return Rp_e_pg, Rm_e_pg
    
    def __Split_Miehe(self, Epsilon_e_pg: np.ndarray, verif=False):

        dim = self.__comportement.dim
        assert dim == 2, "Implémenté que en 2D"

        useNumba = self.__useNumba

        projP_e_pg, projM_e_pg = self.__Decomposition_Spectrale(Epsilon_e_pg, verif)

        if self.__split == "Miehe":
            
            assert isinstance(self.__comportement, Elas_Isot), f"Implémenté que pour un matériau Elas_Isot"

            # Calcul Rp et Rm
            Rp_e_pg, Rm_e_pg = self.__Rp_Rm(Epsilon_e_pg)
            
            # Calcul IxI
            I = np.array([1,1,0]).reshape((3,1))
            IxI = I.dot(I.T)

            # Calcul partie sphérique
            spherP_e_pg = np.einsum('ep,ij->epij', Rp_e_pg, IxI, optimize='optimal')
            spherM_e_pg = np.einsum('ep,ij->epij', Rm_e_pg, IxI, optimize='optimal')

            # Calcul de la loi de comportement
            lamb = self.__comportement.get_lambda()
            mu = self.__comportement.get_mu()

            cP_e_pg = lamb*spherP_e_pg + 2*mu*projP_e_pg
            cM_e_pg = lamb*spherM_e_pg + 2*mu*projM_e_pg

            # projecteurs = {
            #     "projP_e_pg" : projP_e_pg,
            #     "projM_e_pg" : projM_e_pg,
            #     "spherP_e_pg" : spherP_e_pg,
            #     "spherM_e_pg" : spherM_e_pg
            # }
        
        elif self.__split in ["AnisotMiehe","AnisotMiehe_PM","AnisotMiehe_MP","AnisotMiehe_NoCross"]:
            
            c = self.__comportement.get_C()

            tic = Tic()
            
            if useNumba:
                # Plus rapide
                Cpp, Cpm, Cmp, Cmm = CalcNumba.Get_Anisot_C(projP_e_pg, c, projM_e_pg)
            else:
                Cpp = np.einsum('epji,jk,epkl->epil', projP_e_pg, c, projP_e_pg, optimize='optimal')
                Cpm = np.einsum('epji,jk,epkl->epil', projP_e_pg, c, projM_e_pg, optimize='optimal')
                Cmm = np.einsum('epji,jk,epkl->epil', projM_e_pg, c, projM_e_pg, optimize='optimal')
                Cmp = np.einsum('epji,jk,epkl->epil', projM_e_pg, c, projP_e_pg, optimize='optimal')

            tic.Tac("Matrices PFM","Anisot : Cpp, Cpm, Cmp, Cmm", False)
            
            if self.__split ==  "AnisotMiehe":

                cP_e_pg = Cpp + Cpm + Cmp
                cM_e_pg = Cmm 

            elif self.__split ==  "AnisotMiehe_PM":
                
                cP_e_pg = Cpp + Cpm
                cM_e_pg = Cmm + Cmp

            elif self.__split ==  "AnisotMiehe_MP":
                
                cP_e_pg = Cpp + Cmp
                cM_e_pg = Cmm + Cpm

            elif self.__split ==  "AnisotMiehe_NoCross":
                
                cP_e_pg = Cpp
                cM_e_pg = Cmm + Cpm + Cmp
            
        else:
            raise "Split inconnue"

        return cP_e_pg, cM_e_pg

    
    def __Split_Stress(self, Epsilon_e_pg: np.ndarray, verif=False):
        """Construit Cp et Cm pour le split en contraintse"""

        # Récupère les contraintes
        # Ici le matériau est supposé homogène
        loiDeComportement = self.__comportement
        C = loiDeComportement.get_C()    
        Sigma_e_pg = np.einsum('ij,epj->epi',C, Epsilon_e_pg, optimize='optimal')

        # Construit les projecteurs tel que SigmaP = Pp : Sigma et SigmaM = Pm : Sigma                    
        projP_e_pg, projM_e_pg = self.__Decomposition_Spectrale(Sigma_e_pg, verif)

        if self.__split == "Stress":
        
            assert isinstance(loiDeComportement, Elas_Isot)

            E = loiDeComportement.E
            v = loiDeComportement.v

            c = loiDeComportement.get_C()

            # Calcul Rp et Rm
            Rp_e_pg, Rm_e_pg = self.__Rp_Rm(Sigma_e_pg)
            
            # Calcul IxI
            I = np.array([1,1,0]).reshape((3,1))
            IxI = I.dot(I.T)

            RpIxI_e_pg = np.einsum('ep,ij->epij',Rp_e_pg, IxI, optimize='optimal')
            RmIxI_e_pg = np.einsum('ep,ij->epij',Rm_e_pg, IxI, optimize='optimal')

            if loiDeComportement.contraintesPlanes:
                sP_e_pg = (1+v)/E*projP_e_pg - v/E * RpIxI_e_pg
                sM_e_pg = (1+v)/E*projM_e_pg - v/E * RmIxI_e_pg
            else:
                sP_e_pg = (1+v)/E*projP_e_pg - v*(1+v)/E * RpIxI_e_pg
                sM_e_pg = (1+v)/E*projM_e_pg - v*(1+v)/E * RmIxI_e_pg
            
            useNumba = self.__useNumba
            if useNumba:
                # Plus rapide
                cP_e_pg, cM_e_pg = CalcNumba.Get_Cp_Cm_Stress(c, sP_e_pg, sM_e_pg)
            else:
                cT = c.T
                cP_e_pg = np.einsum('ij,epjk,kl->epil', cT, sP_e_pg, c, optimize='optimal')
                cM_e_pg = np.einsum('ij,epjk,kl->epil', cT, sM_e_pg, c, optimize='optimal')

            # # Ici c'est un test pour verifier que cT : S : c = inv(S)

            # detP_e_pg = np.linalg.det(sP_e_pg); e_pg_detPnot0 = np.where(detP_e_pg!=0)
            # detM_e_pg = np.linalg.det(sM_e_pg); e_pg_detMnot0 = np.where(detM_e_pg!=0)

            # invSP_e_pg = np.zeros(sP_e_pg.shape)
            # invSM_e_pg = np.zeros(sM_e_pg.shape)
            
            # invSP_e_pg[e_pg_detPnot0] = np.linalg.inv(sP_e_pg[e_pg_detPnot0])
            # invSM_e_pg[e_pg_detMnot0] = np.linalg.inv(sM_e_pg[e_pg_detMnot0])

            # testP = np.linalg.norm(invSP_e_pg-cP_e_pg)/np.linalg.norm(cP_e_pg)
            # testM = np.linalg.norm(invSM_e_pg-cM_e_pg)/np.linalg.norm(cM_e_pg)
            # pass
        
        elif self.__split in ["AnisotStress","AnisotStress_PM","AnisotStress_MP","AnisotStress_NoCross"]:

            # Construit les ppc_e_pg = Pp : C et ppcT_e_pg = transpose(Pp : C)
            Cp_e_pg = np.einsum('epij,jk->epik', projP_e_pg, C, optimize='optimal')
            Cm_e_pg = np.einsum('epij,jk->epik', projM_e_pg, C, optimize='optimal')
            
            tic = Tic()

            if self.__split != "AnisotStress":
                # Construit Cp et Cm
                S = loiDeComportement.get_S()
                if self.__useNumba:
                    # Plus rapide
                    Cpp, Cpm, Cmp, Cmm = CalcNumba.Get_Anisot_C(Cp_e_pg, S, Cm_e_pg)
                else:
                    Cpp = np.einsum('epji,jk,epkl->epil', Cp_e_pg, S, Cp_e_pg, optimize='optimal')
                    Cpm = np.einsum('epji,jk,epkl->epil', Cp_e_pg, S, Cm_e_pg, optimize='optimal')
                    Cmm = np.einsum('epji,jk,epkl->epil', Cm_e_pg, S, Cm_e_pg, optimize='optimal')
                    Cmp = np.einsum('epji,jk,epkl->epil', Cm_e_pg, S, Cp_e_pg, optimize='optimal')

                tic.Tac("Matrices PFM","Anisot : Cpp, Cpm, Cmp, Cmm", False)

            if self.__split ==  "AnisotStress":

                # cP_e_pg = Cpp + Cpm + Cmp
                # cM_e_pg = Cmm 

                cP_e_pg = Cp_e_pg
                cM_e_pg = Cm_e_pg

            elif self.__split ==  "AnisotStress_PM":
                
                cP_e_pg = Cpp + Cpm
                cM_e_pg = Cmm + Cmp

            elif self.__split ==  "AnisotStress_MP":
                
                cP_e_pg = Cpp + Cmp
                cM_e_pg = Cmm + Cpm

            elif self.__split ==  "AnisotStress_NoCross":
                
                cP_e_pg = Cpp
                cM_e_pg = Cmm + Cpm + Cmp
        
        else:
            raise "Split inconnue"

        return cP_e_pg, cM_e_pg

    def __Split_He(self, Epsilon_e_pg: np.ndarray, verif=False):
            
        # Ici le matériau est supposé homogène
        loiDeComportement = self.__comportement

        if self.__split == "He":
            
            C = loiDeComportement.get_C() 

            sqrtC = sqrtm(C)
            
            if verif :
                # Verif C^1/2 * C^1/2 = C
                testC = np.dot(sqrtC, sqrtC) - C
                assert np.linalg.norm(testC)/np.linalg.norm(C) < 1e-12

            inv_sqrtC = np.linalg.inv(sqrtC)

            # On calcule les nouveaux vecteurs
            Epsilont_e_pg = np.einsum('ij,epj->epi', sqrtC, Epsilon_e_pg, optimize='optimal')

            # On calcule les projecteurs
            projPt_e_pg, projMt_e_pg = self.__Decomposition_Spectrale(Epsilont_e_pg, verif)

            projPt_e_pg_x_sqrtC = np.einsum('epij,jk->epik', projPt_e_pg, sqrtC, optimize='optimal')
            projMt_e_pg_x_sqrtC = np.einsum('epij,jk->epik', projMt_e_pg, sqrtC, optimize='optimal')

            projP_e_pg = np.einsum('ij,epjk->epik', inv_sqrtC, projPt_e_pg_x_sqrtC, optimize='optimal')
            projPT_e_pg =  np.transpose(projP_e_pg, (0,1,3,2))
            projM_e_pg = np.einsum('ij,epjk->epik', inv_sqrtC, projMt_e_pg_x_sqrtC, optimize='optimal')
            projMT_e_pg = np.transpose(projM_e_pg, (0,1,3,2))

            cP_e_pg = np.einsum('epij,jk,epkl->epil', projPT_e_pg, C, projP_e_pg, optimize='optimal')
            cM_e_pg = np.einsum('epij,jk,epkl->epil', projMT_e_pg, C, projM_e_pg, optimize='optimal')

            vecteur_e_pg = Epsilon_e_pg.copy()
            mat = C.copy()

        elif self.__split == "HeStress":

            pass

        if verif:
            # Verification de la décomposition et de l'orthogonalité            
            vecteurP = np.einsum('epij,epj->epi', projP_e_pg, vecteur_e_pg, optimize='optimal')
            vecteurM = np.einsum('epij,epj->epi', projM_e_pg, vecteur_e_pg, optimize='optimal')
            
            # Et+:Et- = 0 deja dans vérifié dans decomp spec
            
            # Décomposition vecteur_e_pg = vecteurP_e_pg + vecteurM_e_pg
            decomp = vecteur_e_pg-(vecteurP + vecteurM)
            if np.linalg.norm(vecteur_e_pg) > 0:
                verifDecomp = np.linalg.norm(decomp)/np.linalg.norm(vecteur_e_pg)
                assert verifDecomp < 1e-12

            # Orthogonalité E+:C:E-
            ortho_vP_vM = np.abs(np.einsum('epi,ij,epj->ep',vecteurP, mat, vecteurM, optimize='optimal'))
            ortho_vM_vP = np.abs(np.einsum('epi,ij,epj->ep',vecteurM, mat, vecteurP, optimize='optimal'))
            ortho_v_v = np.abs(np.einsum('epi,ij,epj->ep', vecteur_e_pg, mat, vecteur_e_pg, optimize='optimal'))
            if ortho_v_v.min() > 0:
                vertifOrthoEpsPM = np.max(ortho_vP_vM/ortho_v_v)
                tvertifOrthoEpsPM = ortho_vP_vM/ortho_v_v
                assert vertifOrthoEpsPM < 1e-12
                vertifOrthoEpsMP = np.max(ortho_vM_vP/ortho_v_v)
                assert vertifOrthoEpsMP < 1e-12
                
        return cP_e_pg, cM_e_pg

    
    def __Decomposition_Spectrale(self, vecteur_e_pg: np.ndarray, verif=False):
        """Calcul projP et projM tel que :\n

        vecteur_e_pg = [1 1 racine(2)] \n
        
        vecteurP = projP : vecteur -> [1, 1, racine(2)] si mandel\n
        vecteurM = projM : vecteur -> [1, 1, racine(2)] si mandel\n

        renvoie projP, projM
        """

        tic = Tic()

        useNumba = self.__useNumba

        coef = self.__comportement.coef

        dim = self.__comportement.dim
        assert dim == 2, "Implémenté que en 2D"

        Ne = vecteur_e_pg.shape[0]
        nPg = vecteur_e_pg.shape[1]

        # Reconsruit le tenseur des deformations [e,pg,dim,dim]
        matrice_e_pg = np.zeros((Ne,nPg,2,2))
        matrice_e_pg[:,:,0,0] = vecteur_e_pg[:,:,0]
        matrice_e_pg[:,:,1,1] = vecteur_e_pg[:,:,1]
        matrice_e_pg[:,:,0,1] = vecteur_e_pg[:,:,2]/coef
        matrice_e_pg[:,:,1,0] = vecteur_e_pg[:,:,2]/coef
        
        
        # invariants du tenseur des deformations [e,pg]
        # trace_e_pg = np.trace(matrice_e_pg, axis1=2, axis2=3)
        trace_e_pg = np.einsum('epii->ep', matrice_e_pg, optimize='optimal')
        
        # determinant_e_pg = np.linalg.det(matrice_e_pg)
        a_e_pg = matrice_e_pg[:,:,0,0]
        b_e_pg = matrice_e_pg[:,:,0,1]
        c_e_pg = matrice_e_pg[:,:,1,0]
        d_e_pg = matrice_e_pg[:,:,1,1]
        determinant_e_pg = (a_e_pg*d_e_pg)-(c_e_pg*b_e_pg)
        
        # probleme Elas_Isot False Stress delta négatif -> si v est grand (0.49999)

        # Calculs des valeurs propres [e,pg]
        delta = trace_e_pg**2 - (4*determinant_e_pg)
        val_e_pg = np.zeros((Ne,nPg,2))
        val_e_pg[:,:,0] = (trace_e_pg - np.sqrt(delta))/2
        val_e_pg[:,:,1] = (trace_e_pg + np.sqrt(delta))/2
        
        # Constantes pour calcul de m1 = (matrice_e_pg - v2*I)/(v1-v2)
        v2I = np.einsum('ep,ij->epij', val_e_pg[:,:,1], np.eye(2), optimize='optimal')
        v1_m_v2 = val_e_pg[:,:,0] - val_e_pg[:,:,1]
        
        # identifications des elements et points de gauss ou vp1 != vp2
        # elements, pdgs = np.where(v1_m_v2 != 0)
        elements, pdgs = np.where(val_e_pg[:,:,0] != val_e_pg[:,:,1])
        
        # construction des bases propres m1 et m2 [e,pg,dim,dim]
        M1 = np.zeros((Ne,nPg,2,2))
        M1[:,:,0,0] = 1
        if elements.size > 0:
            m1_tot = np.einsum('epij,ep->epij', matrice_e_pg-v2I, 1/v1_m_v2, optimize='optimal')
            M1[elements, pdgs] = m1_tot[elements, pdgs]            
        M2 = np.eye(2) - M1
        
        if verif:
            # test ortho entre M1 et M2 
            verifOrtho_M1M2 = np.einsum('epij,epij->ep', M1, M2, optimize='optimal')
            assert np.abs(verifOrtho_M1M2).max() < 1e-10, "Orthogonalité entre M1 et M2 non vérifié"
        
        # Passage des bases propres sous la forme dun vecteur [e,pg,3]  ou [e,pg,6]
        m1 = np.zeros((Ne,nPg,3)); m2 = np.zeros((Ne,nPg,3))
        m1[:,:,0] = M1[:,:,0,0];   m2[:,:,0] = M2[:,:,0,0]
        m1[:,:,1] = M1[:,:,1,1];   m2[:,:,1] = M2[:,:,1,1]
        # m1[:,:,2] = M1[:,:,0,1];   m2[:,:,2] = M2[:,:,0,1]
        m1[:,:,2] = M1[:,:,0,1]*coef;   m2[:,:,2] = M2[:,:,0,1]*coef # Ici on met pas le coef pour que ce soit en [1 1 1]
        
        # Calcul de mixmi [e,pg,3,3] ou [e,pg,6,6]        
        m1xm1 = np.einsum('epi,epj->epij', m1, m1, optimize='optimal')
        m2xm2 = np.einsum('epi,epj->epij', m2, m2, optimize='optimal')
        
        # Récupération des parties positives et négatives des valeurs propres [e,pg,2]
        valp = (val_e_pg+np.abs(val_e_pg))/2
        valm = (val_e_pg-np.abs(val_e_pg))/2
        
        # Calcul des di [e,pg,2]
        dvalp = np.heaviside(val_e_pg, 0.5)
        dvalm = np.heaviside(-val_e_pg, 0.5)
        
        # Calcul des Beta Plus [e,pg,1]
        BetaP = dvalp[:,:,0].copy()
        BetaP[elements,pdgs] = (valp[elements,pdgs,0]-valp[elements,pdgs,1])/v1_m_v2[elements,pdgs]
        
        # Calcul de Beta Moin [e,pg,1]
        BetaM = dvalm[:,:,0].copy()
        BetaM[elements,pdgs] = (valm[elements,pdgs,0]-valm[elements,pdgs,1])/v1_m_v2[elements,pdgs]
        
        # Calcul de gamma [e,pg,2]
        gammap = dvalp - np.repeat(BetaP.reshape((Ne,nPg,1)),2, axis=2)
        gammam = dvalm - np.repeat(BetaM.reshape((Ne,nPg,1)), 2, axis=2)
        
        matriceI = np.eye(3)

        if useNumba:
            # Plus rapide
            projP, projM = CalcNumba.Get_projP_projM(BetaP, gammap, BetaM, gammam, m1xm1, m2xm2)
        else:
            # Projecteur P tel que vecteur_e_pg = projP_e_pg : vecteur_e_pg
            BetaP_x_matriceI = np.einsum('ep,ij->epij', BetaP, matriceI, optimize='optimal')
            gamma1P_x_m1xm1 = np.einsum('ep,epij->epij', gammap[:,:,0], m1xm1, optimize='optimal')
            gamma2P_x_m2xm2 = np.einsum('ep,epij->epij', gammap[:,:,1], m2xm2, optimize='optimal')
            projP = BetaP_x_matriceI + gamma1P_x_m1xm1 + gamma2P_x_m2xm2

            # Projecteur M tel que EpsM = projM : Eps
            BetaM_x_matriceI = np.einsum('ep,ij->epij', BetaM, matriceI, optimize='optimal')
            gamma1M_x_m1xm1 = np.einsum('ep,epij->epij', gammam[:,:,0], m1xm1, optimize='optimal')
            gamma2M_x_m2xm2 = np.einsum('ep,epij->epij', gammam[:,:,1], m2xm2, optimize='optimal')
            projM = BetaM_x_matriceI + gamma1M_x_m1xm1 + gamma2M_x_m2xm2

        if verif:
            # Verification de la décomposition et de l'orthogonalité
            # projecteur en [1; 1; 1]
            vecteurP = np.einsum('epij,epj->epi', projP, vecteur_e_pg, optimize='optimal')
            vecteurM = np.einsum('epij,epj->epi', projM, vecteur_e_pg, optimize='optimal')           
            
            # Décomposition vecteur_e_pg = vecteurP_e_pg + vecteurM_e_pg
            decomp = vecteur_e_pg-(vecteurP + vecteurM)
            if np.linalg.norm(vecteur_e_pg) > 0:
                verifDecomp = np.linalg.norm(decomp)/np.linalg.norm(vecteur_e_pg)
                assert verifDecomp < 1e-12

            # Orthogonalité
            ortho_vP_vM = np.abs(np.einsum('epi,epi->ep',vecteurP, vecteurM, optimize='optimal'))
            ortho_vM_vP = np.abs(np.einsum('epi,epi->ep',vecteurM, vecteurP, optimize='optimal'))
            ortho_v_v = np.abs(np.einsum('epi,epi->ep', vecteur_e_pg, vecteur_e_pg, optimize='optimal'))
            if ortho_v_v.min() > 0:
                vertifOrthoEpsPM = np.max(ortho_vP_vM/ortho_v_v)
                tvertifOrthoEpsPM = ortho_vP_vM/ortho_v_v
                assert vertifOrthoEpsPM < 1e-12
                vertifOrthoEpsMP = np.max(ortho_vM_vP/ortho_v_v)
                assert vertifOrthoEpsMP < 1e-12
        
        tic.Tac("Matrices PFM", "Decomp spectrale", False)
            
        return projP, projM

class ThermalModel:

    def __init__(self, dim:int, k: float, c=0.0, epaisseur=1.0):
        """Construction d'un modèle thermique

        Parameters
        ----------
        dim : int
            dimension du modèle
        k : float
            conduction thermique [W m^-1]
        c : float, optional
            capacité thermique massique [J K^-1 kg^-1], by default 0.0
        epaisseur : float, optional
            epaisseur de la pièce, by default 1.0
        """
        assert dim in [1,2,3]
        self.__dim = dim

        self.__k = k

        self.__c = c
        
        assert epaisseur > 0, "Doit être supérieur à 0"
        self.__epaisseur = epaisseur
    

    @property
    def dim(self) -> int:
        """dimension du modèle"""
        return self.__dim

    @property
    def k(self) -> float:
        """conduction thermique [W . m^-1]"""
        return self.__k

    @property
    def c(self) -> float:
        """capacité thermique massique [J K^-1 kg^-1]"""
        return self.__c

    @property
    def epaisseur(self) -> float:
        """epaisseur de la pièce"""
        return self.__epaisseur
        

class Materiau:

    @property
    def problemType(self) -> str:
        return self.__problemType
    
    @property
    def dim(self) -> int:
        if self.__problemType == "thermal":
            return self.__thermalModel.dim
        if self.__problemType == "beam":
            return self.__beamModel.dim
        else:
            return self.comportement.dim

    @property
    def epaisseur(self) -> float:
        if self.__problemType == "thermal":
            return self.thermalModel.epaisseur
        else:
            return self.comportement.epaisseur

    @property
    def ro(self) -> float:
        """masse volumique"""
        return self.__ro
    
    @property
    def comportement(self) -> LoiDeComportement:
        if self.__problemType in ["thermal","beam"]:
            return None
        else:
            if self.isDamaged:
                return self.__phaseFieldModel.comportement
            else:
                return self.__comportement

    @property
    def thermalModel(self) -> ThermalModel:
        if self.__problemType == "thermal":
            return self.__thermalModel
        else:
            return None
    
    @property
    def beamModel(self) -> BeamModel:
        if self.__problemType == "beam":
            return self.__beamModel
        else:
            return None
    
    @property
    def isDamaged(self) -> bool:
        if self.__problemType == "damage":
            return True
        else:
            return False
    
    @property
    def phaseFieldModel(self) -> PhaseFieldModel:
        """Modèle d'endommagement"""
        if self.isDamaged:
            return self.__phaseFieldModel
        else:
            # Le matériau n'est pas endommageable (pas de modèle PhaseField)
            return None

    def __init__(self, model=None, ro=8100.0, verbosity=False):
        """Creer un materiau avec la loi de comportement ou le phase field model communiqué

        Parameters
        ----------                        
        ro : float, optional
            Masse volumique en kg.m^-3
        epaisseur : float, optional
            epaisseur du matériau si en 2D > 0 !
        """
        if verbosity:
            Affichage.NouvelleSection("Matériau")

        if isinstance(model, LoiDeComportement):
            self.__problemType = "displacement"
            self.__comportement = model
            self.__phaseFieldModel = None
        elif isinstance(model, PhaseFieldModel):
            self.__problemType = "damage"
            self.__phaseFieldModel = model
        elif isinstance(model, ThermalModel):
            self.__problemType = "thermal"
            self.__thermalModel = model
        elif isinstance(model, BeamModel):
            self.__problemType = "beam"
            self.__beamModel = model
        else:
            raise "Model inconnue"

        assert ro > 0 , "Doit être supérieur à 0"
        self.__ro = ro

        self.__verbosity = verbosity

        if self.__verbosity:
            self.Resume()

    def Resume(self, verbosity=True):
        resume = ""

        if self.__problemType == "damage":
            resume += self.__phaseFieldModel.comportement.resume
            resume += '\n' + self.__phaseFieldModel.resume
        elif self.__problemType == "displacement":
            resume += self.__comportement.resume
        elif self.__problemType == "thermal":
            pass
        elif self.__problemType == "beam":
            resume += self.__beamModel.resume

        if verbosity: print(resume)
        return resume

