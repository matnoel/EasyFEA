from abc import ABC, abstractmethod, abstractproperty
from typing import List
from enum import Enum
from TicTac import Tic

from Mesh import Mesh, GroupElem
import CalcNumba as CalcNumba
import numpy as np
import Affichage as Affichage
from Geom import Line, Section

from scipy.linalg import sqrtm

class ModelType(str, Enum):
    """Modèles physiques"""

    displacement = "displacement"
    damage = "damage"
    thermal = "thermal"
    beam = "beam"

class IModel(ABC):
    """Interface d'un modèle
    """

    @abstractproperty
    def modelType(self) -> ModelType:
        """Identifiant du modèle"""
        pass
    
    @abstractproperty
    def dim(self) -> int:
        """dimension du modèle"""
        pass
    
    @abstractproperty
    def epaisseur(self) -> float:
        """epaisseur à utiliser dans le modèle"""
        pass

    @abstractproperty
    def resume(self) -> str:
        """résumé du modèle pour l'affichage"""
        pass

    @property
    def nom(self) -> str:
        """nom du modèle pour l'affichage"""
        return type(self).__name__

    @property
    def useNumba(self) -> bool:
        """Renvoie si le modèle peut utiliser les fonctions numba"""
        return self.__useNumba

    @useNumba.setter
    def useNumba(self, value: bool):
        self.__useNumba = value

    @property
    def needUpdate(self) -> bool:
        """Le model à besoin d'être mis à jour"""
        return self.__needUpdate

    def Need_Update(self, value=True):
        """Renseigne si le model à besoin d'être mis à jour"""
        self.__needUpdate = value

    @staticmethod
    def _Test_Sup0(value: float|np.ndarray):
        texteErreur = "Doit être > 0 !"
        if isinstance(value, (float, int)):
            assert value > 0.0, texteErreur
        if isinstance(value, np.ndarray):
            assert value.min() > 0.0, texteErreur

    @staticmethod
    def _Test_Borne(value: float|np.ndarray, bInf=-1, bSup=0.5):
        texteErreur = f"Doit être compris entre ]{bInf};{bSup}["
        if isinstance(value, (float, int)):
            assert value > bInf and value < bSup, texteErreur
        if isinstance(value, np.ndarray):
            assert value.min() > bInf and value.max() < bSup, texteErreur

class Displacement_Model(IModel):
    """Classe des lois de comportements élastiques
    (Elas_isot, Elas_IsotTrans, Elas_Anisot ...)
    """
    def __init__(self, dim: int, epaisseur: float):
        
        self.__dim = dim
        """dimension lié a la loi de comportement"""

        if dim == 2:
            assert epaisseur > 0 , "Doit être supérieur à 0"
            self.__epaisseur = epaisseur

        if self.dim == 2:
            self.__simplification = "CP" if self.contraintesPlanes else "DP"
        else:
            self.__simplification = "3D"

        self.Need_Update()

    @property
    def modelType(self) -> ModelType:
        return ModelType.displacement

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
    def contraintesPlanes(self) -> bool:
        """Le modèle utilise une simplification contraintes planes"""
        return False

    @property
    def simplification(self) -> str:
        """Simplification utilisé pour le modèle"""
        return self.__simplification

    @abstractmethod
    def _Update(self):
        """Mets à jour la loi de comportement C et S"""
        pass
    
    @abstractproperty
    def resume(self) -> str:
        pass

    # Model
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

    @property
    def coef(self) -> float:
        """Coef lié à la notation de kelvin mandel=racine(2)"""
        return np.sqrt(2)    

    @property
    def C(self) -> np.ndarray:
        """Loi de comportement pour la loi de Lamé en Kelvin Mandel\n
        En 2D : C -> C : Epsilon = Sigma [Sxx, Syy, racine(2)*Sxy]\n
        En 3D : C -> C : Epsilon = Sigma [Sxx, Syy, Szz, racine(2)*Syz, racine(2)*Sxz, racine(2)*Sxy]
        """
        if self.needUpdate:
            self._Update()
            self.Need_Update(False)
        return self.__C.copy()

    @C.setter
    def C(self, array: np.ndarray):
        self.__C = array

    @property
    def S(self) -> np.ndarray:
        """Loi de comportement pour la loi de Hooke en Kelvin Mandel\n
        En 2D : S -> S : Sigma = Epsilon [Exx, Eyy, racine(2)*Exy]\n
        En 3D : S -> S : Sigma = Epsilon [Exx, Eyy, Ezz, racine(2)*Eyz, racine(2)*Exz, racine(2)*Exy]
        """
        if self.needUpdate:
            self._Update()
            self.Need_Update(False)
        return self.__S.copy()
    
    @S.setter
    def S(self, array: np.ndarray):
        self.__S = array

    @staticmethod
    def AppliqueCoefSurBrigi(dim: int, B_rigi_e_pg: np.ndarray) -> np.ndarray:

        if dim == 2:
            coord=2
        elif dim == 3:
            coord=[3,4,5]
        else:
            raise Exception("Pas implémenté")

        coef = np.sqrt(2)

        B_rigi_e_pg[:,:,coord,:] = B_rigi_e_pg[:,:,coord,:]/coef

        return B_rigi_e_pg
    
    @staticmethod
    def ApplyKelvinMandelCoefTo_Matrice(dim: int, Matrice: np.ndarray) -> np.ndarray:        
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
            raise Exception("Pas implémenté")

        matriceMandelCoef = Matrice*transform

        return matriceMandelCoef

    @staticmethod
    def Apply_P(P: np.ndarray, Matrice: np.ndarray) -> np.ndarray:

        shape = Matrice.shape
        if len(shape) == 2:
            matrice_P = np.einsum('ji,jk,kl->il',P, Matrice, P, optimize='optimal')
            axis1, axis2 = 0, 1
        elif len(shape) == 3:
            matrice_P = np.einsum('ji,ejk,kl->eil',P, Matrice, P, optimize='optimal')
            axis1, axis2 = 1, 2
        elif len(shape) == 4:
            matrice_P = np.einsum('ji,epjk,kl->epil',P, Matrice, P, optimize='optimal')
            axis1, axis2 = 2, 3
        else:
            raise Exception("Doit être de dimension (ij) ou (eij) ou (epij)")

        # on verfie que les invariants du tenseur ne change pas !
        # if np.linalg.norm(P.T-P) <= 1e-12:
        tr1 = np.trace(matrice_P, 0, axis1, axis2)
        tr2 = np.trace(Matrice, 0, axis1, axis2)
        diffTrace = np.linalg.norm(tr1-tr2)
        if diffTrace > 1e-12:
            test_trace_c = diffTrace/np.linalg.norm(tr2)
            assert test_trace_c <1e-12, "La trace n'est pas conservé pendant la transformation"
        detMatrice = np.linalg.det(Matrice)
        if np.max(detMatrice) > 1e-12:
            test_det_c = np.linalg.norm(np.linalg.det(matrice_P) - detMatrice)/np.linalg.norm(detMatrice)
            assert test_det_c <1e-10, "Le determinant n'est pas conservé pendant la transformation"
        
        return matrice_P

class Elas_Isot(Displacement_Model):    

    def __init__(self, dim: int, E=210000.0, v=0.3, contraintesPlanes=True, epaisseur=1.0):
        """Creer la matrice de comportement d'un matériau : Elastique isotrope

        Parameters
        ----------
        dim : int
            Dimension de la simulation 2D ou 3D
        E : float|np.ndarray, optional
            Module d'elasticité du matériau en MPa (> 0)
        v : float|np.ndarray, optional
            Coef de poisson ]-1;0.5]
        contraintesPlanes : bool
            Contraintes planes si dim = 2 et True, by default True        
        """       

        # Vérification des valeurs
        assert dim in [2,3], "doit être en 2 et 3"
        self.__dim = dim
        
        self.E=E
        self.v=v

        self.__contraintesPlanes = contraintesPlanes if dim == 2 else False
        """type de simplification 2D"""

        Displacement_Model.__init__(self, dim, epaisseur)

        self._Update()

    @property
    def contraintesPlanes(self) -> bool:
        return self.__contraintesPlanes

    def _Update(self):
        C, S = self.__Comportement()
        # try:        
        #     C, S = self.__Comportement()
        # except ValueError:
        #     raise Exception(str(_erreurConstMateriau))
        self.C = C
        self.S = S        

    @property
    def resume(self) -> str:
        resume = f"\n{self.nom} :"
        resume += f"\nE = {self.E:.2e}, v = {self.v}"
        if self.dim == 2:
            resume += f"\nCP = {self.contraintesPlanes}, ep = {self.epaisseur:.2e}"            
        return resume

    @property
    def E(self) -> float|np.ndarray:
        """Module de Young"""
        return self.__E
    
    @E.setter
    def E(self, value):
        self._Test_Sup0(value)
        self.Need_Update()
        self.__E = value

    @property
    def v(self) -> float|np.ndarray:
        """Coef de poisson"""
        return self.__v
    
    @v.setter
    def v(self, value: float):
        self._Test_Borne(value)
        self.Need_Update()
        self.__v = value

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

        dtype = object if True in [isinstance(p, np.ndarray) for p in [E, v]] else float

        if dim == 2:

            # Attention ici ça marche car lambda change en fonction de la simplification 2D

            cVoigt = np.array([ [l + 2*mu, l, 0],
                                [l, l + 2*mu, 0],
                                [0, 0, mu]], dtype=dtype)

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
                                [0, 0, 0, 0, 0, mu]], dtype=dtype)
            
        cVoigt = Uniform_Array(cVoigt)
        
        c = Displacement_Model.ApplyKelvinMandelCoefTo_Matrice(dim, cVoigt)

        s = np.linalg.inv(c)

        return c, s

class Elas_IsotTrans(Displacement_Model):

    def __init__(self, dim: int, El: float, Et: float, Gl: float, vl: float, vt: float, axis_l=np.array([1,0,0]), axis_t=np.array([0,1,0]), contraintesPlanes=True, epaisseur=1.0):

        # Vérification des valeurs
        assert dim in [2,3], "doit être en 2 et 3"
        self.__dim = dim

        self.El=El        
        self.Et=Et        
        self.Gl=Gl
        
        self.vl=vl        
        self.vt=vt        

        self.__contraintesPlanes = contraintesPlanes if dim == 2 else False
        """type de simplification 2D"""       

        # Création de la matrice de changement de base
        self.__axis1 = axis_l
        self.__axis2 = axis_t

        Displacement_Model.__init__(self, dim, epaisseur)

        self._Update()

    @property
    def contraintesPlanes(self) -> bool:
        return self.__contraintesPlanes

    @property
    def Gt(self) -> float|np.ndarray:
        
        Et = self.Et
        vt = self.vt

        Gt = Et/(2*(1+vt))

        return Gt

    @property
    def El(self) -> float|np.ndarray:
        """Module de Young longitudinale"""
        return self.__El

    @El.setter
    def El(self, value: float|np.ndarray):
        self._Test_Sup0(value)
        self.Need_Update()
        self.__El = value

    @property
    def Et(self) -> float|np.ndarray:
        """Module de Young transverse"""
        return self.__Et
    
    @Et.setter
    def Et(self, value: float|np.ndarray):
        self._Test_Sup0(value)
        self.Need_Update()
        self.__Et = value

    @property
    def Gl(self) -> float|np.ndarray:
        """Module de Cisaillent longitudinale"""
        return self.__Gl

    @Gl.setter
    def Gl(self, value: float|np.ndarray):
        self._Test_Sup0(value)
        self.Need_Update()
        self.__Gl = value

    @property
    def vl(self) -> float|np.ndarray:
        """Coef de poisson longitudianale"""
        return self.__vl

    @vl.setter
    def vl(self, value: float|np.ndarray):
        # -1<vt<1
        # -1<vl<0.5
        # Regarder torquato 328
        self._Test_Borne(value, -1, 1)
        self.Need_Update()
        self.__vl = value
    
    @property
    def vt(self) -> float|np.ndarray:
        """Coef de poisson transverse"""
        return self.__vt

    @vt.setter
    def vt(self, value: float|np.ndarray):
        # -1<vt<1
        # -1<vl<0.5
        # Regarder torquato 328
        self._Test_Borne(value)
        self.Need_Update()
        self.__vt = value

    @property
    def kt(self) -> float|np.ndarray:
        # Source : torquato 2002
        El = self.El
        Et = self.Et
        vtt = self.vt
        vtl = self.vl
        kt = El*Et/((2*(1-vtt)*El)-(4*vtl**2*Et))

        return kt

    def _Update(self):
        axis_l, axis_t = self.__axis1, self.__axis2
        P = self.get_P(axis_1=axis_l, axis_2=axis_t)

        if np.linalg.norm(axis_l-np.array([1,0,0]))<1e-12 and np.linalg.norm(axis_t-np.array([0,1,0]))<1e-12:
            useSameAxis=True
        else:
            useSameAxis=False
        
        try:
            C, S = self.__Comportement(P, useSameAxis)
        except ValueError:
            raise Exception(str(_erreurConstMateriau))

        self.C = C
        self.S = S

    @property
    def resume(self) -> str:
        resume = f"\nElas_IsotTrans :"
        resume += f"\nEl = {self.El:.2e}, Et = {self.Et:.2e}, Gl = {self.Gl:.2e}"
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
        
        dtype = object if isinstance(kt, np.ndarray) else float

        # Matrice de souplesse et de rigidité en mandel dans la base du matériau
        # [11, 22, 33, sqrt(2)*23, sqrt(2)*13, sqrt(2)*12]

        material_sM = np.array([[1/El, -vl/El, -vl/El, 0, 0, 0],
                      [-vl/El, 1/Et, -vt/Et, 0, 0, 0],
                      [-vl/El, -vt/Et, 1/Et, 0, 0, 0],
                      [0, 0, 0, 1/(2*Gt), 0, 0],
                      [0, 0, 0, 0, 1/(2*Gl), 0],
                      [0, 0, 0, 0, 0, 1/(2*Gl)]], dtype=dtype)
        
        material_sM = Uniform_Array(material_sM)

        material_cM = np.array([[El+4*vl**2*kt, 2*kt*vl, 2*kt*vl, 0, 0, 0],
                      [2*kt*vl, kt+Gt, kt-Gt, 0, 0, 0],
                      [2*kt*vl, kt-Gt, kt+Gt, 0, 0, 0],
                      [0, 0, 0, 2*Gt, 0, 0],
                      [0, 0, 0, 0, 2*Gl, 0],
                      [0, 0, 0, 0, 0, 2*Gl]], dtype=dtype)
        
        material_cM = Uniform_Array(material_cM)

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

            shape = c.shape
            
            if self.contraintesPlanes == True:
                if len(shape) == 2:
                    s = global_sM[x,:][:,x]
                elif len(shape) == 3:
                    s = global_sM[:,x,:][:,:,x]
                elif len(shape) == 4:
                    s = global_sM[:,:,x,:][:,:,:,x]
                    
                c = np.linalg.inv(s)
            else:                
                if len(shape) == 2:
                    c = global_cM[x,:][:,x]
                elif len(shape) == 3:
                    c = global_cM[:,x,:][:,:,x]
                elif len(shape) == 4:
                    c = global_cM[:,:,x,:][:,:,:,x]
                
                s = np.linalg.inv(c)

                # testS = np.linalg.norm(s-s2)/np.linalg.norm(s2)            
        
        return c, s

class Elas_Anisot(Displacement_Model):

    @property
    def resume(self) -> str:
        resume = f"\n{self.nom}) :"
        resume += f"\n{self.C}"
        resume += f"\naxi1 = {self.__axis1},  axi2 = {self.__axis2}"
        if self.__dim == 2:
            resume += f"\nCP = {self.contraintesPlanes}, ep = {self.epaisseur:.2e}"
        return resume

    def __init__(self, dim: int, C: np.ndarray, axis1=None, axis2=None, useVoigtNotation=True, contraintesPlanes=True, epaisseur=1.0):
        """Création d'une loi de comportement elastique anisotrope

        Parameters
        ----------
        dim : int
            dimension
        C : np.ndarray
            matrice de rigidité dans la base d'anisotropie
        axis1 : np.ndarray, optional
            vecteur de l'axe1, by default None
        axis2 : np.ndarray, optional
            vecteur de l'axe2, by default None
        useVoigtNotation : bool, optional
            la loi de comportement utilise l'a notation de voigt, by default True
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

        self.__contraintesPlanes = contraintesPlanes if dim == 2 else False
        """type de simplification 2D"""

        if isinstance(axis1, np.ndarray):        
            # Verification et construction des vecteurs
            assert len(axis1) == 3, "Doit fournir un vecteur" 
        else:
            axis1 = np.array([1,0,0])
        self.__axis1 = axis1

        def Calc_axis2():
            theta = np.pi/2
            rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
            axis2 = rot.dot(axis1)
            return axis2

        if isinstance(axis2, np.ndarray):
            assert axis2.size == 3, "Doit fournir un vecteur"
            if not np.isclose(axis1.dot(axis2), 0, 1e-12):
                axis2 = Calc_axis2()
        else:
            axis2 = Calc_axis2()
        self.__axis2 = axis2

        Displacement_Model.__init__(self, dim, epaisseur)

        self.Set_C(C, useVoigtNotation)

    def _Update(self):
        # ici ne fait rien car pour mettre a jour les loi on utilise Set_C
        return super()._Update()

    def Set_C(self, C: np.ndarray, useVoigtNotation=True, update_S=True):
        """Mets à jour la loi de comportement C et S

        Parameters
        ----------
        C : np.ndarray
           Loi de comportement pour la loi de Lamé
        useVoigtNotation : bool, optional
            La loi de comportement utilise la notation de kevin mandel, by default True
        update_S : bool, optional
            Met à jour la matrice de souplesse, by default True
        """
        
        C_mandelP = self.__Comportement(C, useVoigtNotation)
        self.C = C_mandelP
        
        if update_S:
            S_mandelP = np.linalg.inv(C_mandelP)
            self.S = S_mandelP
    
    def __Comportement(self, C: np.ndarray, useVoigtNotation: bool):

        dim = self.__dim

        shape = C.shape

        # Verification sur la matrice
        if dim == 2:
            assert (shape[-2], shape[-1]) == (3,3), "La matrice doit être de dimension 3x3"
        else:
            assert (shape[-2], shape[-1]) == (6,6), "La matrice doit être de dimension 6x6"
        testSym = np.linalg.norm(C.T - C)/np.linalg.norm(C)
        assert testSym <= 1e-12, "La matrice n'est pas symétrique"

        # Construction de la matrice de rotation
        P = self.get_P(axis_1=self.__axis1, axis_2=self.__axis2)

        # Application des coef si nécessaire
        if useVoigtNotation:
            C_mandel = self.ApplyKelvinMandelCoefTo_Matrice(dim, C)
        else:
            C_mandel = C.copy()

        # Passage de la matrice en 3D pour faire la rotation

        if dim == 2:

            listIndex = np.array([0,1,5])

            if len(shape)==2:
                C_mandel_global = np.zeros((6,6))
                for i, I in enumerate(listIndex):
                    for j, J in enumerate(listIndex):
                        C_mandel_global[I,J] = C_mandel[i,j]
            if len(shape)==3:
                C_mandel_global = np.zeros((shape[0],6,6))
                for i, I in enumerate(listIndex):
                    for j, J in enumerate(listIndex):
                        C_mandel_global[:,I,J] = C_mandel[:,i,j]
            elif len(shape)==4:
                C_mandel_global = np.zeros((shape[0],shape[1],6,6))
                for i, I in enumerate(listIndex):
                    for j, J in enumerate(listIndex):
                        C_mandel_global[:,:,I,J] = C_mandel[:,:,i,j]
        else:
            C_mandel_global = C

        C_mandelP_global = self.Apply_P(P, C_mandel_global)

        if dim == 2:
            listIndex = np.array([0,1,5])

            if len(shape)==2:
                C_mandelP = C_mandelP_global[listIndex,:][:,listIndex]
            if len(shape)==3:
                C_mandelP = C_mandelP_global[:,listIndex,:][:,:,listIndex]
            elif len(shape)==4:
                C_mandelP = C_mandelP_global[:,:,listIndex,:][:,:,:,listIndex]
            
        else:
            C_mandelP = C_mandelP_global

        return C_mandelP

    @property
    def contraintesPlanes(self) -> bool:
        return self.__contraintesPlanes

class Poutre_Elas_Isot():

    @property
    def resume(self) -> str:
        resume = ""

        resume += f"\n{self.__name} :"
        resume += f"\n  S = {self.__section.aire:.2}, Iz = {self.__section.aire:.2}, Iy = {self.__section.aire:.2}, J = {self.__section.J:.2}"

        return resume

    # Nombre de poutres crées
    __nbPoutre=0

    def __init__(self, line: Line, section: Section, E: float, v:float):
        """Construction d'une poutre élastique isotrope

        Parameters
        ----------
        line : Line
            Ligne de la fibre moyenne
        section : Section
            Section de la poutre
        E : float
            module elastique
        v : float
            coef de poisson
        """

        self.__line = line

        self.__section = section

        assert E > 0.0, "Le module élastique doit être > 0 !"
        self.__E=E

        poisson = "Le coef de poisson doit être compris entre ]-1;0.5["
        assert v > -1.0 and v < 0.5, poisson
        self.__v=v

        # Verifie si la section est symétrique Iyz = 0
        Iyz = section.Iyz 
        assert Iyz <=  1e-12, "La section doit être symétrique"

        Poutre_Elas_Isot.__nbPoutre += 1
        self.__name = f"Poutre{Poutre_Elas_Isot.__nbPoutre}"

    @property
    def line(self) -> Line:
        """Ligne fibre moyenne de la poutre"""
        return self.__line

    @property
    def section(self) -> Section:
        """Section de la poutre"""
        return self.__section

    @property
    def name(self) -> str:
        """Identifiant de la poutre"""
        return self.__name

    @property
    def E(self) -> float:
        """Le module élastique"""
        return self.__E

    @property
    def v(self) -> float:
        """Coef de poisson"""
        return self.__v    

class Beam_Model(IModel):

    __modelType = ModelType.beam

    @property
    def modelType(self) -> ModelType:
        return Beam_Model.__modelType

    @property
    def dim(self) -> int:
        """Dimension du model \n
        1D -> traction compression \n 
        2D -> traction compression + fleche + flexion \n
        3D -> tout \n"""
        return self.__dim
    
    @property
    def epaisseur(self) -> float:
        """Le modèle poutre peut posséder plusieurs poutres et donc des sections différentes\n
        Il faut regarder dans la section de la poutre qui nous intéresse"""
        return None

    @property
    def resume(self) -> str:
        resume = f"\n{self.nom} :"
        resume += f"\nNombre de Poutre = {self.nbPoutres} :\n"
        # Réalise un résumé pour chaque poutre
        
        def __resumePoutreElast(resume: str, poutre: Poutre_Elas_Isot, E: float, v: float):
            resume += poutre.resume
            if isinstance(E, int):
                resume += f"\n\tE = {E:6}, v = {v}"
            else:
                resume += f"\n\tE = {E:6.2}, v = {v}"

        [__resumePoutreElast(resume, poutre, E, v) for poutre, E, v in zip(self.__listePoutres, self.liste_E, self.liste_v)]
            
        return resume

    def __init__(self, dim: int, listePoutres: list[Poutre_Elas_Isot]):
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
        
        self.__dim = dim
        self.__listePoutres = listePoutres
        self.__list_D = []

        list_E = self.liste_E
        list_v = self.liste_v

        for poutre, E, v in zip(listePoutres, list_E, list_v):
            
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
        nPg = groupElem.Get_gauss(matriceType).nPg
        D_e_pg = np.zeros((Ne, nPg, list_D[0].shape[0], list_D[0].shape[0]))
        for poutre, D in zip(listePoutres, list_D):
            # recupère les element
            elements = groupElem.Get_Elements_Tag(poutre.name)
            D_e_pg[elements] = D

        return D_e_pg    

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
    def listePoutres(self) -> List[Poutre_Elas_Isot]:
        """Liste des poutres"""
        return self.__listePoutres

    @property
    def nbPoutres(self) -> int:
        """Nombre de poutre"""
        return len(self.__listePoutres)

    @property
    def liste_E(self) -> List[float]:
        """Liste des modules élastiques"""
        return [poutre.E for poutre in self.__listePoutres]

    @property
    def liste_v(self) -> List[float]:
        """Liste des coef de poisson"""
        return [poutre.v for poutre in self.__listePoutres]

    @property
    def list_D(self) -> List[np.ndarray]:
        """liste de loi de comportement"""
        return self.__list_D

class PhaseField_Model(IModel):

    class RegularizationType(str, Enum):
        """Régularisation de la fissure"""

        AT1 = "AT1"
        AT2 = "AT2"

    class SplitType(str, Enum):
        """Splits utilisables"""

        Bourdin = "Bourdin"
        Amor = "Amor"
        Miehe = "Miehe"
        He = "He"
        Stress = "Stress"
        Zhang = "Zhang"

        AnisotStrain = "AnisotStrain"
        AnisotStrain_PM = "AnisotStrain_PM"
        AnisotStrain_MP = "AnisotStrain_MP"
        AnisotStrain_NoCross = "AnisotStrain_NoCross"

        AnisotStress = "AnisotStress"
        AnisotStress_PM = "AnisotStress_PM"
        AnisotStress_MP = "AnisotStress_MP"
        AnisotStress_NoCross = "AnisotStress_NoCross"

    class SolveurType(str, Enum):
        History = "History"
        HistoryDamage = "HistoryDamage"
        BoundConstrain = "BoundConstrain"    

    def __init__(self, comportement: Displacement_Model, split: str, regularization: RegularizationType, Gc: float, l_0: float, solveur=SolveurType.History, A=None):
        """Crétation d'un modèle à gradient d'endommagement

        Parameters
        ----------
        loiDeComportement : LoiDeComportement
            Loi de comportement du matériau (Elas_Isot, Elas_IsotTrans)
        split : str
            Split de la densité d'energie élastique (voir PhaseFieldModel.get_splits())
        regularization : RegularizationType
            Modèle de régularisation de la fissure AT1 ou AT2
        Gc : float
            Taux de restitution d'energie critique en J.m^-2
        l_0 : float
            Demie largeur de fissure 
        solveur : SolveurType, optional
            Type de résolution de l'endommagement, by default History (voir SolveurType)        
        A : np.ndarray, optional
            Matrice caractérisant la direction de l'anisotropie du modèle pour l'énergie de fissure
        """
    
        assert isinstance(comportement, Displacement_Model), "Doit être une loi de comportement"
        self.__comportement = comportement

        assert split in PhaseField_Model.get_splits(), f"Doit être compris dans {PhaseField_Model.get_splits()}"
        if not isinstance(comportement, Elas_Isot):
            assert not split in PhaseField_Model.__splits_Isot, "Ces splits ne sont implémentés que pour Elas_Isot"
        self.__split =  split
        """Split de la densité d'energie elastique"""
        
        assert regularization in PhaseField_Model.get_regularisations(), f"Doit être compris dans {PhaseField_Model.get_regularisations()}"
        self.__regularization = regularization
        """Modèle de régularisation de la fissure ["AT1","AT2"]"""
        
        self.Gc = Gc

        assert l_0 > 0, "Doit être supérieur à 0"
        self.__l0 = l_0
        """Largeur de régularisation de la fissure"""

        self.__solveur = solveur
        """Solveur d'endommagement"""

        if not isinstance(A, np.ndarray):
            self.__A = np.eye(self.dim)
        else:
            dim = self.dim
            assert A.shape[-2] == dim and A.shape[-1] == dim, "Mauvaise dimension"
            self.__A = A

        self.__useNumba = True
        """Utilise ou non les fonctions numba"""

        self.Need_Split_Update()

    @property
    def modelType(self) -> ModelType:
        return ModelType.damage

    @property
    def dim(self) -> int:
        return self.__comportement.dim

    @property
    def epaisseur(self) -> float:
        return self.__comportement.epaisseur

    @property
    def resume(self) -> str:
        resume = self.__comportement.resume
        resume += f'\n\n{self.nom} :'
        resume += f'\nsplit : {self.__split}'
        resume += f'\nregularisation : {self.__regularization}'
        resume += f'\nGc : {self.__Gc:.2e}'
        resume += f'\nl0 : {self.__l0:.2e}'
        return resume

    # Phase field   

    __splits_Isot = [SplitType.Amor, SplitType.Miehe, SplitType.Stress]
    __split_Anisot = [SplitType.Bourdin, SplitType.He, SplitType.Zhang,
                    SplitType.AnisotStrain, SplitType.AnisotStrain_PM, SplitType.AnisotStrain_MP, SplitType.AnisotStrain_NoCross,
                    SplitType.AnisotStress, SplitType.AnisotStress_PM, SplitType.AnisotStress_MP, SplitType.AnisotStress_NoCross]

    @staticmethod
    def get_splits() -> List[SplitType]:
        """splits disponibles"""
        return list(PhaseField_Model.SplitType)    
    
    @staticmethod
    def get_regularisations() -> List[RegularizationType]:
        """regularisations disponibles"""
        __regularizations = list(PhaseField_Model.RegularizationType)
        return __regularizations    

    @staticmethod
    def get_solveurs() -> List[SolveurType]:
        """solveurs disponibles"""
        __solveurs = list(PhaseField_Model.SolveurType)
        return __solveurs

    @property
    def k(self) -> float:
        Gc = self.__Gc
        l0 = self.__l0

        k = Gc * l0

        if self.__regularization == PhaseField_Model.RegularizationType.AT1:
            k = 3/4 * k

        return k

    def get_r_e_pg(self, PsiP_e_pg: np.ndarray) -> np.ndarray:
        
        Gc = Resize_variable(self.__Gc, PsiP_e_pg.shape[0], PsiP_e_pg.shape[1])
        
        l0 = self.__l0        
        r = 2 * PsiP_e_pg

        if self.__regularization == PhaseField_Model.RegularizationType.AT2:
            r = r + (Gc/l0)
        
        return r

    def get_f_e_pg(self, PsiP_e_pg: np.ndarray) -> np.ndarray:
        
        Gc = Resize_variable(self.__Gc, PsiP_e_pg.shape[0], PsiP_e_pg.shape[1])
        l0 = self.__l0

        f = 2 * PsiP_e_pg

        if self.__regularization == PhaseField_Model.RegularizationType.AT1:
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

        if self.__regularization in PhaseField_Model.get_regularisations():
            g_e_pg = (1-d_e_pg)**2 + k_residu
        else:
            raise Exception("Pas implémenté")

        assert mesh.Ne == g_e_pg.shape[0]
        assert mesh.Get_nPg(matriceType) == g_e_pg.shape[1]
        
        return g_e_pg
    
    @property
    def A(self) -> np.ndarray:
        """Matrice caractérisant la direction de l'anisotropie du modèle pour l'énergie de fissure"""
        return self.__A

    @property
    def split(self) -> str:
        return self.__split

    @property
    def regularization(self) -> str:
        return self.__regularization
    
    @property
    def comportement(self) -> Displacement_Model:
        """modele en déplacement"""
        return self.__comportement

    @property
    def solveur(self):
        """Solveur d'endommagement"""
        return self.__solveur

    @property
    def Gc(self):
        """Taux de libération d'énergie critque [J/m^2]"""
        return self.__Gc
    
    @Gc.setter
    def Gc(self, value):
        self._Test_Sup0(value)
        self.Need_Update()
        self.__Gc = value

    @property
    def l0(self):
        """Largeur de régularisation de la fissure"""
        return self.__l0

    @property
    def c0(self):
        """Paramètre de mise à l'échelle permettant dissiper exactement l'énergie de fissure"""
        if self.__regularization == PhaseField_Model.RegularizationType.AT1:
            c0 = 8/3
        elif self.__regularization == PhaseField_Model.RegularizationType.AT2:
            c0 = 2
        return c0
    
    @property
    def useNumba(self) -> bool:
        return self.__useNumba
    
    @useNumba.setter
    def useNumba(self, val: bool):
        self.__useNumba = val
            
    def Calc_psi_e_pg(self, Epsilon_e_pg: np.ndarray):
        """Calcul de la densité d'energie elastique\n
        psiP_e_pg = 1/2 SigmaP_e_pg * Epsilon_e_pg\n
        psiM_e_pg = 1/2 SigmaM_e_pg * Epsilon_e_pg\n
        Tel que :\n
        SigmaP_e_pg = cP_e_pg * Epsilon_e_pg\n
        SigmaM_e_pg = cM_e_pg * Epsilon_e_pg        
        """

        SigmaP_e_pg, SigmaM_e_pg = self.Calc_Sigma_e_pg(Epsilon_e_pg)

        tic = Tic()

        psiP_e_pg = np.sum(1/2 * Epsilon_e_pg * SigmaP_e_pg, -1)
        psiM_e_pg = np.sum(1/2 * Epsilon_e_pg * SigmaM_e_pg, -1)

        tic.Tac("Matrices", "psiP_e_pg et psiM_e_pg", False)

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
        
        cP_e_pg, cM_e_pg = self.Calc_C(Epsilon_e_pg)

        tic = Tic()
        
        Epsilon_e_pg = Epsilon_e_pg.reshape((Ne,nPg,comp,1))

        SigmaP_e_pg = np.reshape(cP_e_pg @ Epsilon_e_pg, (Ne,nPg,-1))
        SigmaM_e_pg = np.reshape(cM_e_pg @ Epsilon_e_pg, (Ne,nPg,-1))

        tic.Tac("Matrices", "SigmaP_e_pg et SigmaM_e_pg", False)

        return SigmaP_e_pg, SigmaM_e_pg

    def Need_Split_Update(self):
        """Initialise le dictionnaire qui stocke la décompositon de la loi de comportement"""
        self.__dict_cP_e_pg_And_cM_e_pg = {}
    
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
        # Une fois pour calculer l'energie (psiP) donc pour construire une matrice de masse et une fois pour calculer K_u soit une matrice rigi, on est donc obligé de passé 2 fois dedans

        Ne = Epsilon_e_pg.shape[0]
        nPg = Epsilon_e_pg.shape[1]

        key = f"({Ne}, {nPg})"

        if key in self.__dict_cP_e_pg_And_cM_e_pg:
            # Si la clé est renseigné on récupère la solution stockée

            cP_e_pg = self.__dict_cP_e_pg_And_cM_e_pg[key][0]
            cM_e_pg = self.__dict_cP_e_pg_And_cM_e_pg[key][1]
        
        else:

            if self.__split == PhaseField_Model.SplitType.Bourdin:
                cP_e_pg, cM_e_pg = self.__Split_Bourdin(Ne, nPg)

            elif self.__split == PhaseField_Model.SplitType.Amor:
                cP_e_pg, cM_e_pg = self.__Split_Amor(Epsilon_e_pg)

            elif self.__split == PhaseField_Model.SplitType.Miehe or "Strain" in self.__split:
                cP_e_pg, cM_e_pg = self.__Split_Miehe(Epsilon_e_pg, verif=verif)
            
            elif self.__split == PhaseField_Model.SplitType.Zhang or "Stress" in self.__split:
                cP_e_pg, cM_e_pg = self.__Split_Stress(Epsilon_e_pg, verif=verif)

            elif self.__split == PhaseField_Model.SplitType.He:
                cP_e_pg, cM_e_pg = self.__Split_He(Epsilon_e_pg, verif=verif)
            
            else: 
                raise Exception("Split inconnue")

            self.__dict_cP_e_pg_And_cM_e_pg[key] = (cP_e_pg, cM_e_pg)

        return cP_e_pg, cM_e_pg

    def __Split_Bourdin(self, Ne: int, nPg: int):
        
        tic = Tic()
        c = self.__comportement.C
        c = c[np.newaxis, np.newaxis,:,:]
        c = np.repeat(c, Ne, axis=0)
        c = np.repeat(c, nPg, axis=1)

        cP_e_pg = c
        cM_e_pg = np.zeros_like(cP_e_pg)
        tic.Tac("Decomposition",f"cP_e_pg et cM_e_pg", False)

        return cP_e_pg, cM_e_pg

    def __Split_Amor(self, Epsilon_e_pg: np.ndarray):

        assert isinstance(self.__comportement, Elas_Isot), f"Implémenté que pour un matériau Elas_Isot"
        
        tic = Tic()
        
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

        tic.Tac("Decomposition",f"cP_e_pg et cM_e_pg", False)

        return cP_e_pg, cM_e_pg

    def __Rp_Rm(self, vecteur_e_pg: np.ndarray):
        """Renvoie Rp_e_pg, Rm_e_pg"""

        Ne = vecteur_e_pg.shape[0]
        nPg = vecteur_e_pg.shape[1]

        dim = self.__comportement.dim

        trace = np.zeros((Ne, nPg))

        trace = vecteur_e_pg[:,:,0] + vecteur_e_pg[:,:,1]

        if dim == 3:
            trace += vecteur_e_pg[:,:,2]

        Rp_e_pg = (1+np.sign(trace))/2
        Rm_e_pg = (1+np.sign(-trace))/2

        return Rp_e_pg, Rm_e_pg
    
    def __Split_Miehe(self, Epsilon_e_pg: np.ndarray, verif=False):

        dim = self.__comportement.dim
        # assert dim == 2, "Implémenté que en 2D"

        useNumba = self.__useNumba

        projP_e_pg, projM_e_pg = self.__Decomposition_Spectrale(Epsilon_e_pg, verif)

        tic = Tic()

        if self.__split == PhaseField_Model.SplitType.Miehe:
            
            assert isinstance(self.__comportement, Elas_Isot), f"Implémenté que pour un matériau Elas_Isot"

            # Calcul Rp et Rm
            Rp_e_pg, Rm_e_pg = self.__Rp_Rm(Epsilon_e_pg)
            
            # Calcul IxI
            if dim == 2:
                I = np.array([1,1,0]).reshape((3,1))
            elif dim == 3:
                I = np.array([1,1,1,0,0,0]).reshape((6,1))
            IxI = I.dot(I.T)

            # Calcul partie sphérique
            spherP_e_pg = np.einsum('ep,ij->epij', Rp_e_pg, IxI, optimize='optimal')
            spherM_e_pg = np.einsum('ep,ij->epij', Rm_e_pg, IxI, optimize='optimal')

            # Calcul de la loi de comportement
            lamb = self.__comportement.get_lambda()
            mu = self.__comportement.get_mu()

            cP_e_pg = lamb*spherP_e_pg + 2*mu*projP_e_pg
            cM_e_pg = lamb*spherM_e_pg + 2*mu*projM_e_pg
        
        elif "Strain" in self.__split:
            
            c = self.__comportement.C

            # TODO a optim ?
            # ici fonctionne pas si c est heterogène
            
            if useNumba:
                # Plus rapide
                Cpp, Cpm, Cmp, Cmm = CalcNumba.Get_Anisot_C(projP_e_pg, c, projM_e_pg)
            else:
                Cpp = np.einsum('epji,jk,epkl->epil', projP_e_pg, c, projP_e_pg, optimize='optimal')
                Cpm = np.einsum('epji,jk,epkl->epil', projP_e_pg, c, projM_e_pg, optimize='optimal')
                Cmm = np.einsum('epji,jk,epkl->epil', projM_e_pg, c, projM_e_pg, optimize='optimal')
                Cmp = np.einsum('epji,jk,epkl->epil', projM_e_pg, c, projP_e_pg, optimize='optimal')
            
            if self.__split == PhaseField_Model.SplitType.AnisotStrain:

                cP_e_pg = Cpp + Cpm + Cmp
                cM_e_pg = Cmm 

            elif self.__split == PhaseField_Model.SplitType.AnisotStrain_PM:
                
                cP_e_pg = Cpp + Cpm
                cM_e_pg = Cmm + Cmp

            elif self.__split == PhaseField_Model.SplitType.AnisotStrain_MP:
                
                cP_e_pg = Cpp + Cmp
                cM_e_pg = Cmm + Cpm

            elif self.__split == PhaseField_Model.SplitType.AnisotStrain_NoCross:
                
                cP_e_pg = Cpp
                cM_e_pg = Cmm + Cpm + Cmp
            
        else:
            raise Exception("Split inconnue")

        tic.Tac("Decomposition",f"cP_e_pg et cM_e_pg", False)

        return cP_e_pg, cM_e_pg

    
    def __Split_Stress(self, Epsilon_e_pg: np.ndarray, verif=False):
        """Construit Cp et Cm pour le split en contraintse"""

        # Récupère les contraintes
        # Ici le matériau est supposé homogène
        loiDeComportement = self.__comportement
        C = loiDeComportement.C    
        Sigma_e_pg = np.einsum('ij,epj->epi',C, Epsilon_e_pg, optimize='optimal')

        # Construit les projecteurs tel que SigmaP = Pp : Sigma et SigmaM = Pm : Sigma                    
        projP_e_pg, projM_e_pg = self.__Decomposition_Spectrale(Sigma_e_pg, verif)

        tic = Tic()

        if self.__split == PhaseField_Model.SplitType.Stress:
        
            assert isinstance(loiDeComportement, Elas_Isot)

            E = loiDeComportement.E
            v = loiDeComportement.v

            c = loiDeComportement.C

            dim = self.dim

            # Calcul Rp et Rm
            Rp_e_pg, Rm_e_pg = self.__Rp_Rm(Sigma_e_pg)
            
            # Calcul IxI
            if dim == 2:
                I = np.array([1,1,0]).reshape((3,1))
            else:
                I = np.array([1,1,1,0,0,0]).reshape((6,1))
            IxI = I.dot(I.T)

            RpIxI_e_pg = np.einsum('ep,ij->epij',Rp_e_pg, IxI, optimize='optimal')
            RmIxI_e_pg = np.einsum('ep,ij->epij',Rm_e_pg, IxI, optimize='optimal')

            if dim == 2:
                if loiDeComportement.contraintesPlanes:
                    sP_e_pg = (1+v)/E*projP_e_pg - v/E * RpIxI_e_pg
                    sM_e_pg = (1+v)/E*projM_e_pg - v/E * RmIxI_e_pg
                else:
                    sP_e_pg = (1+v)/E*projP_e_pg - v*(1+v)/E * RpIxI_e_pg
                    sM_e_pg = (1+v)/E*projM_e_pg - v*(1+v)/E * RmIxI_e_pg
            elif dim == 3:
                mu = loiDeComportement.get_mu()

                sP_e_pg = 1/(2*mu)*projP_e_pg - v/E * RpIxI_e_pg
                sM_e_pg = 1/(2*mu)*projM_e_pg - v/E * RmIxI_e_pg
            
            useNumba = self.__useNumba
            if useNumba:
                # Plus rapide
                cP_e_pg, cM_e_pg = CalcNumba.Get_Cp_Cm_Stress(c, sP_e_pg, sM_e_pg)
            else:
                cT = c.T
                cP_e_pg = np.einsum('ij,epjk,kl->epil', cT, sP_e_pg, c, optimize='optimal')
                cM_e_pg = np.einsum('ij,epjk,kl->epil', cT, sM_e_pg, c, optimize='optimal')
        
        elif self.__split == PhaseField_Model.SplitType.Zhang or "Stress" in self.__split:

            # Construit les ppc_e_pg = Pp : C et ppcT_e_pg = transpose(Pp : C)
            Cp_e_pg = np.einsum('epij,jk->epik', projP_e_pg, C, optimize='optimal')
            Cm_e_pg = np.einsum('epij,jk->epik', projM_e_pg, C, optimize='optimal')

            if self.__split == PhaseField_Model.SplitType.Zhang:
                cP_e_pg = Cp_e_pg
                cM_e_pg = Cm_e_pg
            
            else:
                # Construit Cp et Cm
                S = loiDeComportement.S
                if self.__useNumba:
                    # Plus rapide
                    Cpp, Cpm, Cmp, Cmm = CalcNumba.Get_Anisot_C(Cp_e_pg, S, Cm_e_pg)
                else:
                    Cpp = np.einsum('epji,jk,epkl->epil', Cp_e_pg, S, Cp_e_pg, optimize='optimal')
                    Cpm = np.einsum('epji,jk,epkl->epil', Cp_e_pg, S, Cm_e_pg, optimize='optimal')
                    Cmm = np.einsum('epji,jk,epkl->epil', Cm_e_pg, S, Cm_e_pg, optimize='optimal')
                    Cmp = np.einsum('epji,jk,epkl->epil', Cm_e_pg, S, Cp_e_pg, optimize='optimal')                    
                
                if self.__split == PhaseField_Model.SplitType.AnisotStress:

                    cP_e_pg = Cpp + Cpm + Cmp
                    cM_e_pg = Cmm                    

                elif self.__split == PhaseField_Model.SplitType.AnisotStress_PM:
                    
                    cP_e_pg = Cpp + Cpm
                    cM_e_pg = Cmm + Cmp

                elif self.__split == PhaseField_Model.SplitType.AnisotStress_MP:
                    
                    cP_e_pg = Cpp + Cmp
                    cM_e_pg = Cmm + Cpm

                elif self.__split == PhaseField_Model.SplitType.AnisotStress_NoCross:
                    
                    cP_e_pg = Cpp
                    cM_e_pg = Cmm + Cpm + Cmp
            
                else:
                    raise Exception("Split inconnue")

        tic.Tac("Decomposition",f"cP_e_pg et cM_e_pg", False)

        return cP_e_pg, cM_e_pg

    def __Split_He(self, Epsilon_e_pg: np.ndarray, verif=False):
            
        # Ici le matériau est supposé homogène
        loiDeComportement = self.__comportement

        C = loiDeComportement.C 

        # Mettre ça direct dans la loi de comportement ?

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

        tic = Tic()

        projPt_e_pg_x_sqrtC = np.einsum('epij,jk->epik', projPt_e_pg, sqrtC, optimize='optimal')
        projMt_e_pg_x_sqrtC = np.einsum('epij,jk->epik', projMt_e_pg, sqrtC, optimize='optimal')

        projP_e_pg = np.einsum('ij,epjk->epik', inv_sqrtC, projPt_e_pg_x_sqrtC, optimize='optimal')
        projPT_e_pg =  np.transpose(projP_e_pg, (0,1,3,2))
        projM_e_pg = np.einsum('ij,epjk->epik', inv_sqrtC, projMt_e_pg_x_sqrtC, optimize='optimal')
        projMT_e_pg = np.transpose(projM_e_pg, (0,1,3,2))

        cP_e_pg = np.einsum('epij,jk,epkl->epil', projPT_e_pg, C, projP_e_pg, optimize='optimal')
        cM_e_pg = np.einsum('epij,jk,epkl->epil', projMT_e_pg, C, projM_e_pg, optimize='optimal')

        tic.Tac("Decomposition",f"cP_e_pg et cM_e_pg", False)

        if verif:
            vecteur_e_pg = Epsilon_e_pg.copy()
            mat = C.copy()    

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
                assert vertifOrthoEpsPM < 1e-12
                vertifOrthoEpsMP = np.max(ortho_vM_vP/ortho_v_v)
                assert vertifOrthoEpsMP < 1e-12

        return cP_e_pg, cM_e_pg

    def __valeursPropres_vecteursPropres_matricesPropres(self, vecteur_e_pg: np.ndarray, verif=False) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:

        dim = self.__comportement.dim

        coef = self.__comportement.coef
        Ne = vecteur_e_pg.shape[0]
        nPg = vecteur_e_pg.shape[1]

        tic = Tic()

        # Reconsruit le tenseur des deformations [e,pg,dim,dim]
        matrice_e_pg = np.zeros((Ne,nPg,dim,dim))
        for d in range(dim):
            matrice_e_pg[:,:,d,d] = vecteur_e_pg[:,:,d]
        if dim == 2:
            # [x, y, xy]
            # xy
            matrice_e_pg[:,:,0,1] = vecteur_e_pg[:,:,2]/coef
            matrice_e_pg[:,:,1,0] = vecteur_e_pg[:,:,2]/coef
        else:
            # [x, y, z, yz, xz, xy]
            # yz
            matrice_e_pg[:,:,1,2] = vecteur_e_pg[:,:,3]/coef
            matrice_e_pg[:,:,2,1] = vecteur_e_pg[:,:,3]/coef
            # xz
            matrice_e_pg[:,:,0,2] = vecteur_e_pg[:,:,4]/coef
            matrice_e_pg[:,:,2,0] = vecteur_e_pg[:,:,4]/coef
            # xy
            matrice_e_pg[:,:,0,1] = vecteur_e_pg[:,:,5]/coef
            matrice_e_pg[:,:,1,0] = vecteur_e_pg[:,:,5]/coef

        tic.Tac("Decomposition", "vecteur_e_pg -> matrice_e_pg", False)

        # trace_e_pg = np.trace(matrice_e_pg, axis1=2, axis2=3)
        trace_e_pg = np.einsum('epii->ep', matrice_e_pg, optimize='optimal')

        if self.dim == 2:
            # invariants du tenseur des deformations [e,pg]            

            a_e_pg = matrice_e_pg[:,:,0,0]
            b_e_pg = matrice_e_pg[:,:,0,1]
            c_e_pg = matrice_e_pg[:,:,1,0]
            d_e_pg = matrice_e_pg[:,:,1,1]
            determinant_e_pg = (a_e_pg*d_e_pg)-(c_e_pg*b_e_pg)

            tic.Tac("Decomposition", "Invariants", False)

            # Calculs des valeurs propres [e,pg]
            delta = trace_e_pg**2 - (4*determinant_e_pg)
            val_e_pg = np.zeros((Ne,nPg,2))
            val_e_pg[:,:,0] = (trace_e_pg - np.sqrt(delta))/2
            val_e_pg[:,:,1] = (trace_e_pg + np.sqrt(delta))/2

            tic.Tac("Decomposition", "Valeurs propres", False)
            
            # Constantes pour calcul de m1 = (matrice_e_pg - v2*I)/(v1-v2)
            v2I = np.einsum('ep,ij->epij', val_e_pg[:,:,1], np.eye(2), optimize='optimal')
            v1_m_v2 = val_e_pg[:,:,0] - val_e_pg[:,:,1]
            
            # identifications des elements et points de gauss ou vp1 != vp2
            # elements, pdgs = np.where(v1_m_v2 != 0)
            elems, pdgs = np.where(val_e_pg[:,:,0] != val_e_pg[:,:,1])
            
            # construction des bases propres m1 et m2 [e,pg,dim,dim]
            M1 = np.zeros((Ne,nPg,2,2))
            M1[:,:,0,0] = 1
            if elems.size > 0:
                m1_tot = np.einsum('epij,ep->epij', matrice_e_pg-v2I, 1/v1_m_v2, optimize='optimal')
                M1[elems, pdgs] = m1_tot[elems, pdgs]
            M2 = np.eye(2) - M1

            tic.Tac("Decomposition", "Projecteurs propres", False)
        
        elif self.dim == 3:

            version = 'eigh' # 'matlab', 'matthieu', 'eigh'            

            if version in ['matlab', 'matthieu']:

                a11_e_pg = matrice_e_pg[:,:,0,0]; a12_e_pg = matrice_e_pg[:,:,0,1]; a13_e_pg = matrice_e_pg[:,:,0,2]
                a21_e_pg = matrice_e_pg[:,:,1,0]; a22_e_pg = matrice_e_pg[:,:,1,1]; a23_e_pg = matrice_e_pg[:,:,1,2]
                a31_e_pg = matrice_e_pg[:,:,2,0]; a32_e_pg = matrice_e_pg[:,:,2,1]; a33_e_pg = matrice_e_pg[:,:,2,2]

                determinant_e_pg = a11_e_pg * ((a22_e_pg*a33_e_pg)-(a32_e_pg*a23_e_pg)) - a12_e_pg * ((a21_e_pg*a33_e_pg)-(a31_e_pg*a23_e_pg)) + a13_e_pg * ((a21_e_pg*a32_e_pg)-(a31_e_pg*a22_e_pg))            

                # Invariants
                I1_e_pg = trace_e_pg
                mat_mat = np.einsum('epij,epjk->epik', matrice_e_pg, matrice_e_pg, optimize='optimal')
                trace_mat_mat = np.einsum('epii->ep', mat_mat, optimize='optimal')
                I2_e_pg = (trace_e_pg**2 - trace_mat_mat)/2
                I3_e_pg = determinant_e_pg

                tic.Tac("Decomposition", "Invariants", False)

                g = I1_e_pg**2 - 3*I2_e_pg            
                elems0 = np.unique(np.where(g == 0)[0])
                filtreNot0 = g != 0
                elemsNot0 = np.unique(np.where(filtreNot0)[0])

                racine_g = np.sqrt(g)
                racine_g_ij = racine_g.reshape((Ne, nPg, 1, 1)).repeat(3, axis=2).repeat(3, axis=3)            
                
                arg = (2*I1_e_pg**3 - 9*I1_e_pg*I2_e_pg + 27*I3_e_pg)/2/g**(3/2) # -1 <= arg <= 1
                
                theta = np.arccos(arg)/3 # Lode's angle such that 0 <= theta <= pi/3

                elemsMin = np.unique(np.where(arg == 1)[0]) # positions of double minimum eigenvalue            
                elemsMax = np.unique(np.where(arg == -1)[0]) # positions of double maximum eigenvalue

                elemsNot0 = np.setdiff1d(elemsNot0, elemsMin)
                elemsNot0 = np.setdiff1d(elemsNot0, elemsMax)

            def __Normalize(M1, M2, M3):
                M1 = np.einsum('epij,ep->epij', M1, 1/np.linalg.norm(M1, axis=(2,3)))
                M2 = np.einsum('epij,ep->epij', M2, 1/np.linalg.norm(M2, axis=(2,3)))
                M3 = np.einsum('epij,ep->epij', M3, 1/np.linalg.norm(M3, axis=(2,3)))

                return M1, M2, M3

            if version == 'eigh':

                valnum, vectnum = np.linalg.eigh(matrice_e_pg)

                tic.Tac("Decomposition", "np.linalg.eigh", False)

                func_Mi = lambda mi: np.einsum('epi,epj->epij', mi, mi, optimize='optimal')

                M1 = func_Mi(vectnum[:,:,:,0])
                M2 = func_Mi(vectnum[:,:,:,1])
                M3 = func_Mi(vectnum[:,:,:,2])
                
                val_e_pg = valnum

                tic.Tac("Decomposition", "Valeurs et projecteurs propres", False)

            elif version == 'matlab':

                # Initialisation des valeurs propres
                val_e_pg = (I1_e_pg/3).reshape((Ne, nPg, 1)).repeat(3, axis=2)

                # Case of double minimum eigenvalue
                if elemsMin.size > 0:
                    val_e_pg[elemsMin] = val_e_pg[elemsMin] + np.einsum('ep,i->epi', racine_g, np.array([-1, -1, 2])/3, optimize='optimal')[elemsMin]

                # Case of double maximum eigenvalue
                if elemsMax.size > 0:
                    val_e_pg[elemsMax] = val_e_pg[elemsMax] + np.einsum('ep,i->epi', racine_g, np.array([-2, 1, 1])/3, optimize='optimal')[elemsMax]

                # Case of 3 distinct eigenvalues
                if elemsNot0.size > 0:
                    thet = np.zeros_like(val_e_pg)
                    thet[:,:,0] = np.cos(2*np.pi/3+theta)
                    thet[:,:,1] = np.cos(2*np.pi/3-theta)
                    thet[:,:,2] = np.cos(theta)
                    thet = thet * 2/3

                    val_e_pg[elemsNot0] = val_e_pg[elemsNot0] + np.einsum('ep,epi->epi', racine_g, thet, optimize='optimal')[elemsNot0]

                tic.Tac("Decomposition", "Valeurs propres", False)

                # Initialisation des projecteurs propres
                M1 = np.zeros_like(matrice_e_pg); M1[:,:,0,0] = 1            
                M3 = np.zeros_like(matrice_e_pg); M3[:,:,2,2] = 1

                eye3 = np.zeros_like(matrice_e_pg)
                eye3[:,:,0,0] = 1; eye3[:,:,1,1] = 1; eye3[:,:,2,2] = 1
                I_rg = np.einsum('ep,epij->epij', I1_e_pg - racine_g, eye3/3, optimize='optimal')

                # Case of double minimum eigenvalue
                M3[elemsMin] = (matrice_e_pg[elemsMin] - I_rg[elemsMin])/racine_g_ij[elemsMin]
                M1[elemsMin] = (eye3[elemsMin] - M3[elemsMin])/2

                # Case of double maximum eigenvalue
                M1[elemsMax] = (I_rg[elemsMax] - matrice_e_pg[elemsMax])/racine_g_ij[elemsMax]
                M3[elemsMax] = (eye3[elemsMax] - M1[elemsMax])/2

                # Case of 3 distinct eigenvalues            
                if elemsNot0.size > 0:
                    matNot0 = matrice_e_pg[elemsNot0]
                    E1not0 = val_e_pg[elemsNot0, :, 0]; E1not0_eye3 = np.einsum('ep,ij->epij', E1not0, np.ones((3,3)), optimize='optimal')
                    E2not0 = val_e_pg[elemsNot0, :, 1]; E2not0_eye3 = np.einsum('ep,ij->epij', E2not0, np.ones((3,3)), optimize='optimal')
                    E3not0 = val_e_pg[elemsNot0, :, 2]; E3not0_eye3 = np.einsum('ep,ij->epij', E3not0, np.ones((3,3)), optimize='optimal')

                    M1[elemsNot0] = (matNot0-E2not0_eye3)/(E1not0_eye3-E2not0_eye3) * (matNot0-E3not0_eye3)/(E1not0_eye3-E3not0_eye3)
                    M3[elemsNot0] = (matNot0-E1not0_eye3)/(E3not0_eye3-E1not0_eye3) * (matNot0-E2not0_eye3)/(E3not0_eye3-E2not0_eye3)

                M2 = eye3 - (M1 + M3)

                M1, M2, M3 = __Normalize(M1, M2, M3)

                tic.Tac("Decomposition", "Projecteurs propres", False)

            elif version == 'matthieu':

                E1 = I1_e_pg/3 + 2/3 * racine_g * np.cos(2*np.pi/3 + theta)
                E2 = I1_e_pg/3 + 2/3 * racine_g * np.cos(2*np.pi/3 - theta)
                E3 = I1_e_pg/3 + 2/3 * racine_g * np.cos(theta)

                # Initialisation des valeurs propres            
                val_e_pg = (I1_e_pg/3).reshape((Ne, nPg, 1)).repeat(3, axis=2)
                val_e_pg[elemsNot0, :, 0] = E1[elemsNot0]
                val_e_pg[elemsNot0, :, 1] = E2[elemsNot0]
                val_e_pg[elemsNot0, :, 2] = E3[elemsNot0]

                tic.Tac("Decomposition", "Valeurs propres", False)

                # Initialisation des projecteurs propres
                M1 = np.zeros_like(matrice_e_pg); M1[:,:,0,0] = 1
                M2 = np.zeros_like(matrice_e_pg); M2[:,:,1,1] = 1
                M3 = np.zeros_like(matrice_e_pg); M3[:,:,2,2] = 1

                # g=0 -> E1 = E2 = E3 -> 3 valeurs propres identiques
                # ici ne fait rien car déja initialisé

                eye3 = np.zeros_like(matrice_e_pg)
                eye3[:,:,0,0] = 1; eye3[:,:,1,1] = 1; eye3[:,:,2,2] = 1
                I_rg = np.einsum('ep,epij->epij', I1_e_pg - racine_g, eye3/3, optimize='optimal')                

                # g!=0 & E1 < E2 = E3 -> 2 valeurs propres maximums
                #             
                # elems_E2E3 = np.unique(np.where(filtreNot0 & (np.abs(E2-E3) <= tol))[0])
                # E2[elems_E2E3] = E3[elems_E2E3]

                elems_Max = np.unique(np.where(filtreNot0 & (E1<E2) & (E2==E3))[0])
                M1[elems_Max] = ((I_rg[elems_Max] - matrice_e_pg[elems_Max])/racine_g_ij[elems_Max])
                M2[elems_Max] = M3[elems_Max] = (eye3[elems_Max] - M1[elems_Max])/2

                # g!=0 & E1 == E2 < E3  -> 2 valeurs propres minimums

                # elems_E1E2 = np.unique(np.where(filtreNot0 & (np.abs(E1-E2) <= tol))[0])
                # E1[elems_E1E2] = E2[elems_E1E2]

                elems_Min = np.unique(np.where(filtreNot0 & (E1==E2) & (E2<E3))[0])
                M3[elems_Min] = ((matrice_e_pg[elems_Min] - I_rg[elems_Min])/racine_g_ij[elems_Min])
                M1[elems_Min] = M2[elems_Min] = (eye3[elems_Min] - M3[elems_Min])/2

                # g!=0 & E1 < E2 < E3  -> 3 valeurs propres distinctes
                elems_Dist = np.unique(np.where(filtreNot0 & (E1<E2) & (E2<E3))[0])
                # idxDist = idx0

                E1_ij = E1.reshape((Ne,nPg,1,1)).repeat(3, axis=2).repeat(3, axis=3)[elems_Dist]
                E2_ij = E2.reshape((Ne,nPg,1,1)).repeat(3, axis=2).repeat(3, axis=3)[elems_Dist]
                E3_ij = E3.reshape((Ne,nPg,1,1)).repeat(3, axis=2).repeat(3, axis=3)[elems_Dist]
                matr_dist = matrice_e_pg[elems_Dist]
                eye3_dist = eye3[elems_Dist]
                
                M1[elems_Dist] = ((matr_dist - E2_ij*eye3_dist)/(E1_ij-E2_ij) * (matr_dist - E3_ij*eye3_dist)/(E1_ij-E3_ij))
                M2[elems_Dist] = ((matr_dist - E1_ij*eye3_dist)/(E2_ij-E1_ij) * (matr_dist - E3_ij*eye3_dist)/(E2_ij-E3_ij))
                M3[elems_Dist] = ((matr_dist - E1_ij*eye3_dist)/(E3_ij-E1_ij) * (matr_dist - E2_ij*eye3_dist)/(E3_ij-E2_ij))

                M1, M2, M3 = __Normalize(M1, M2, M3)

                tic.Tac("Decomposition", "Projecteurs propres", False)

        # Passage des bases propres sous la forme dun vecteur [e,pg,3] ou [e,pg,6]
        if dim == 2:
            # [x, y, xy]
            m1 = np.zeros((Ne,nPg,3)); m2 = np.zeros_like(m1)
            m1[:,:,0] = M1[:,:,0,0];   m2[:,:,0] = M2[:,:,0,0]
            m1[:,:,1] = M1[:,:,1,1];   m2[:,:,1] = M2[:,:,1,1]            
            m1[:,:,2] = M1[:,:,0,1]*coef;   m2[:,:,2] = M2[:,:,0,1]*coef

            list_m = [m1, m2]

            list_M = [M1, M2]

        elif dim == 3:
            # [x, y, z, yz, xz, xy]
            m1 = np.zeros((Ne,nPg,6)); m2 = np.zeros_like(m1);  m3 = np.zeros_like(m1)
            m1[:,:,0] = M1[:,:,0,0];   m2[:,:,0] = M2[:,:,0,0]; m3[:,:,0] = M3[:,:,0,0]
            m1[:,:,1] = M1[:,:,1,1];   m2[:,:,1] = M2[:,:,1,1]; m3[:,:,1] = M3[:,:,1,1]
            m1[:,:,2] = M1[:,:,2,2];   m2[:,:,2] = M2[:,:,2,2]; m3[:,:,2] = M3[:,:,2,2]
            
            m1[:,:,3] = M1[:,:,1,2]*coef;   m2[:,:,3] = M2[:,:,1,2]*coef;   m3[:,:,3] = M3[:,:,1,2]*coef
            m1[:,:,4] = M1[:,:,0,2]*coef;   m2[:,:,4] = M2[:,:,0,2]*coef;   m3[:,:,4] = M3[:,:,0,2]*coef
            m1[:,:,5] = M1[:,:,0,1]*coef;   m2[:,:,5] = M2[:,:,0,1]*coef;   m3[:,:,5] = M3[:,:,0,1]*coef

            list_m = [m1, m2, m3]

            list_M = [M1, M2, M3]

        tic.Tac("Decomposition", "Vecteurs propres", False)
        
        if verif:
            
            valnum, vectnum = np.linalg.eigh(matrice_e_pg)

            func_Mi = lambda mi: np.einsum('epi,epj->epij', mi, mi, optimize='optimal')
            func_ep_epij = lambda ep, epij : np.einsum('ep,epij->epij', ep, epij, optimize='optimal')

            M1_num = func_Mi(vectnum[:,:,:,0])
            M2_num = func_Mi(vectnum[:,:,:,1])

            matrice = func_ep_epij(val_e_pg[:,:,0], M1) + func_ep_epij(val_e_pg[:,:,1], M2)

            matrice_num = func_ep_epij(valnum[:,:,0], M1_num) + func_ep_epij(valnum[:,:,1], M2_num)
            
            if dim == 3:                
                M3_num = func_Mi(vectnum[:,:,:,2])
                matrice = matrice + func_ep_epij(val_e_pg[:,:,2], M3)
                matrice_num = matrice_num + func_ep_epij(valnum[:,:,2], M3_num)            

            if matrice_e_pg.max() > 0:
                ecart_matrice = matrice - matrice_e_pg
                erreurMatrice = np.linalg.norm(ecart_matrice)/np.linalg.norm(matrice_e_pg)
                assert erreurMatrice <= 1e-12, "matrice != E1*M1 + E2*M2 + E3*M3 != matrice_e_pg"

            if matrice.max() > 0:
                erreurMatriceNumMatrice = np.linalg.norm(matrice_num - matrice)/np.linalg.norm(matrice)
                assert erreurMatriceNumMatrice <= 1e-12, "matrice != matrice_num"

            # verification si les valeurs prores sont bonnes
            if valnum.max() > 0:
                testval = np.linalg.norm(val_e_pg - valnum)/np.linalg.norm(valnum)
                assert testval <= 1e-12, "Erreur dans le calcul de valeurs propres"            

            # verification si les projecteurs propres sont corrects
            def erreur_Mi_Minum(Mi, mi_num):
                Mi_num = np.einsum('epi,epj->epij', mi_num, mi_num, optimize='optimal')
                ecart = Mi_num-Mi
                erreur = np.linalg.norm(ecart)/np.linalg.norm(Mi)
                assert erreur <= 1e-12, "Erreur dans le calcul des projecteurs propres"

            if dim == 3:
                pass

            erreur_Mi_Minum(M1, vectnum[:,:,:,0])
            erreur_Mi_Minum(M2, vectnum[:,:,:,1])
            if dim == 3:
                erreur_Mi_Minum(M3, vectnum[:,:,:,2])

            # test ortho entre M1 et M2 
            verifOrtho_M1M2 = np.einsum('epij,epij->ep', M1, M2, optimize='optimal')
            textTest = "Orthogonalité non vérifié"
            assert np.abs(verifOrtho_M1M2).max() < 1e-9, textTest

            if dim == 3:
                verifOrtho_M1M3 = np.einsum('epij,epij->ep', M1, M3, optimize='optimal')
                assert np.abs(verifOrtho_M1M3).max() < 1e-10, textTest
                verifOrtho_M2M3 = np.einsum('epij,epij->ep', M2, M3, optimize='optimal')
                assert np.abs(verifOrtho_M2M3).max() < 1e-10, textTest

        return val_e_pg, list_m, list_M
    
    def __Decomposition_Spectrale(self, vecteur_e_pg: np.ndarray, verif=False):
        """Calcul projP et projM tel que :\n

        vecteur_e_pg = [1 1 racine(2)] \n
        
        vecteurP = projP : vecteur -> [1, 1, racine(2)] si mandel\n
        vecteurM = projM : vecteur -> [1, 1, racine(2)] si mandel\n

        renvoie projP, projM
        """

        useNumba = self.__useNumba        

        dim = self.__comportement.dim        

        Ne = vecteur_e_pg.shape[0]
        nPg = vecteur_e_pg.shape[1]
        
        # récupération des valeurs, vecteurs et matrices propres
        val_e_pg, list_m, list_M = self.__valeursPropres_vecteursPropres_matricesPropres(vecteur_e_pg, verif)

        tic = Tic()
        
        # Récupération des parties positives et négatives des valeurs propres [e,pg,2]
        valp = (val_e_pg+np.abs(val_e_pg))/2
        valm = (val_e_pg-np.abs(val_e_pg))/2
        
        # Calcul des di [e,pg,2]
        dvalp = np.heaviside(val_e_pg, 0.5)
        dvalm = np.heaviside(-val_e_pg, 0.5)

        if dim == 2:
            # vecteurs propres
            m1, m2 = list_m[0], list_m[1]

            # elements et pdgs ou les valeurs propres 1 et 2 sont différentes
            elems, pdgs = np.where(val_e_pg[:,:,0] != val_e_pg[:,:,1])

            v1_m_v2 = val_e_pg[:,:,0] - val_e_pg[:,:,1] # val1 - val2

            # Calcul des Beta Plus [e,pg,1]            
            BetaP = dvalp[:,:,0].copy() # ici bien mettre copy sinon quand modification Beta modifie en meme temps dvalp !
            BetaP[elems,pdgs] = (valp[elems,pdgs,0]-valp[elems,pdgs,1])/v1_m_v2[elems,pdgs]
            
            # Calcul de Beta Moin [e,pg,1]
            BetaM = dvalm[:,:,0].copy()
            BetaM[elems,pdgs] = (valm[elems,pdgs,0]-valm[elems,pdgs,1])/v1_m_v2[elems,pdgs]
            
            # Calcul de gamma [e,pg,2]
            gammap = dvalp - np.repeat(BetaP.reshape((Ne,nPg,1)),2, axis=2)
            gammam = dvalm - np.repeat(BetaM.reshape((Ne,nPg,1)), 2, axis=2)

            tic.Tac("Decomposition", "Betas et gammas", False)

            if useNumba:
                # Plus rapide
                projP, projM = CalcNumba.Get_projP_projM_2D(BetaP, gammap, BetaM, gammam, m1, m2)

            else:
                # Calcul de mixmi [e,pg,3,3] ou [e,pg,6,6]
                m1xm1 = np.einsum('epi,epj->epij', m1, m1, optimize='optimal')
                m2xm2 = np.einsum('epi,epj->epij', m2, m2, optimize='optimal')

                matriceI = np.eye(3)
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

            tic.Tac("Decomposition", "projP et projM", False)

        elif dim == 3:
            m1, m2, m3 = list_m[0], list_m[1], list_m[2]

            M1, M2, M3 = list_M[0], list_M[1], list_M[2]            

            coef = np.sqrt(2)

            thetap = dvalp.copy()/2
            thetam = dvalm.copy()/2

            funcFiltreComp = lambda vi, vj: vi != vj
            
            elems, pdgs = np.where(funcFiltreComp(val_e_pg[:,:,0], val_e_pg[:,:,1]))
            v1_m_v2 = val_e_pg[elems,pdgs,0]-val_e_pg[elems,pdgs,1]
            thetap[elems, pdgs, 0] = (valp[elems,pdgs,0]-valp[elems,pdgs,1])/(2*v1_m_v2)
            thetam[elems, pdgs, 0] = (valm[elems,pdgs,0]-valm[elems,pdgs,1])/(2*v1_m_v2)

            elems, pdgs = np.where(funcFiltreComp(val_e_pg[:,:,0], val_e_pg[:,:,2]))
            v1_m_v3 = val_e_pg[elems,pdgs,0]-val_e_pg[elems,pdgs,2]
            thetap[elems, pdgs, 1] = (valp[elems,pdgs,0]-valp[elems,pdgs,2])/(2*v1_m_v3)
            thetam[elems, pdgs, 1] = (valm[elems,pdgs,0]-valm[elems,pdgs,2])/(2*v1_m_v3)

            elems, pdgs = np.where(funcFiltreComp(val_e_pg[:,:,1], val_e_pg[:,:,2]))
            v2_m_v3 = val_e_pg[elems,pdgs,1]-val_e_pg[elems,pdgs,2]
            thetap[elems, pdgs, 2] = (valp[elems,pdgs,1]-valp[elems,pdgs,2])/(2*v2_m_v3)
            thetam[elems, pdgs, 2] = (valm[elems,pdgs,1]-valm[elems,pdgs,2])/(2*v2_m_v3)

            tic.Tac("Decomposition", "thetap et thetam", False)

            if useNumba:
                # Beaucoup plus rapide (environ 2x plus rapide)

                G12_ijkl, G13_ijkl, G23_ijkl = CalcNumba.Get_G12_G13_G23(M1, M2, M3)

                listI = [0]*6; listI.extend([1]*6); listI.extend([2]*6); listI.extend([1]*6); listI.extend([0]*12)
                listJ = [0]*6; listJ.extend([1]*6); listJ.extend([2]*18); listJ.extend([1]*6)
                listK = [0,1,2,1,0,0]*6
                listL = [0,1,2,2,2,1]*6
                
                colonnes = np.arange(0,6, dtype=int).reshape((1,6)).repeat(6,axis=0).reshape(-1)
                lignes = np.sort(colonnes)

                def __get_Gab_ij(Gab_ijkl: np.ndarray):
                    Gab_ij = np.zeros((Ne, nPg, 6, 6))

                    Gab_ij[:,:,lignes, colonnes] = Gab_ijkl[:,:,listI,listJ,listK,listL]
                
                    Gab_ij[:,:,:3,3:6] = Gab_ij[:,:,:3,3:6] * coef
                    Gab_ij[:,:,3:6,:3] = Gab_ij[:,:,3:6,:3] * coef
                    Gab_ij[:,:,3:6,3:6] = Gab_ij[:,:,3:6,3:6] * 2

                    return Gab_ij

                G12_ij = __get_Gab_ij(G12_ijkl)
                G13_ij = __get_Gab_ij(G13_ijkl)
                G23_ij = __get_Gab_ij(G23_ijkl)

                list_mi = [m1, m2, m3]
                list_Gab = [G12_ij, G13_ij, G23_ij]

                projP, projM = CalcNumba.Get_projP_projM_3D(dvalp, dvalm, thetap, thetam, list_mi, list_Gab)
            
            else:

                def __Construction_Gij(Ma, Mb):

                    Gij = np.zeros((Ne, nPg, 6, 6))

                    part1 = lambda Ma, Mb: np.einsum('epik,epjl->epijkl', Ma, Mb, optimize='optimal')
                    part2 = lambda Ma, Mb: np.einsum('epil,epjk->epijkl', Ma, Mb, optimize='optimal')

                    Gijkl = part1(Ma, Mb) + part2(Ma, Mb) + part1(Mb, Ma) + part2(Mb, Ma)

                    listI = [0]*6; listI.extend([1]*6); listI.extend([2]*6); listI.extend([1]*6); listI.extend([0]*12)
                    listJ = [0]*6; listJ.extend([1]*6); listJ.extend([2]*18); listJ.extend([1]*6)
                    listK = [0,1,2,1,0,0]*6
                    listL = [0,1,2,2,2,1]*6
                    
                    colonnes = np.arange(0,6, dtype=int).reshape((1,6)).repeat(6,axis=0).reshape(-1)
                    lignes = np.sort(colonnes)

                    # # ici je construit une matrice pour verfier que les numéros sont bons
                    # ma = np.zeros((6,6), dtype=np.object0)
                    # for lin,col,i,j,k,l in zip(lignes, colonnes, listI, listJ, listK, listL):
                    #     text = f"{i+1}{j+1}{k+1}{l+1}"
                    #     ma[lin,col] = text
                    #     pass

                    Gij[:,:,lignes, colonnes] = Gijkl[:,:,listI,listJ,listK,listL]
                    
                    Gij[:,:,:3,3:6] = Gij[:,:,:3,3:6] * coef
                    Gij[:,:,3:6,:3] = Gij[:,:,3:6,:3] * coef
                    Gij[:,:,3:6,3:6] = Gij[:,:,3:6,3:6] * 2

                    return Gij

                G12 = __Construction_Gij(M1, M2)
                G13 = __Construction_Gij(M1, M3)
                G23 = __Construction_Gij(M2, M3)

                tic.Tac("Decomposition", "Gab", False)

                m1xm1 = np.einsum('epi,epj->epij', m1, m1, optimize='optimal')
                m2xm2 = np.einsum('epi,epj->epij', m2, m2, optimize='optimal')
                m3xm3 = np.einsum('epi,epj->epij', m3, m3, optimize='optimal')

                tic.Tac("Decomposition", "mixmi", False)

                # func = lambda ep, epij: np.einsum('ep,epij->epij', ep, epij, optimize='optimal')
                func = lambda ep, epij: ep[:,:,np.newaxis,np.newaxis].repeat(epij.shape[2], axis=2).repeat(epij.shape[3], axis=3) * epij

                projP = func(dvalp[:,:,0], m1xm1) + func(dvalp[:,:,1], m2xm2) + func(dvalp[:,:,2], m3xm3) + func(thetap[:,:,0], G12) + func(thetap[:,:,1], G13) + func(thetap[:,:,2], G23)

                projM = func(dvalm[:,:,0], m1xm1) + func(dvalm[:,:,1], m2xm2) + func(dvalm[:,:,2], m3xm3) + func(thetam[:,:,0], G12) + func(thetam[:,:,1], G13) + func(thetam[:,:,2], G23)

            tic.Tac("Decomposition", "projP et projM", False)

        if verif:
            # Verification de la décomposition et de l'orthogonalité
            # projecteur en [1; 1; 1]
            vecteurP = np.einsum('epij,epj->epi', projP, vecteur_e_pg, optimize='optimal')
            vecteurM = np.einsum('epij,epj->epi', projM, vecteur_e_pg, optimize='optimal')           
            
            # Décomposition vecteur_e_pg = vecteurP_e_pg + vecteurM_e_pg
            decomp = vecteur_e_pg-(vecteurP + vecteurM)
            if np.linalg.norm(vecteur_e_pg) > 0:                
                verifDecomp = np.linalg.norm(decomp)/np.linalg.norm(vecteur_e_pg)
                assert verifDecomp <= 1e-12, "vecteur_e_pg != vecteurP_e_pg + vecteurM_e_pg"

            # Orthogonalité
            ortho_vP_vM = np.abs(np.einsum('epi,epi->ep',vecteurP, vecteurM, optimize='optimal'))
            ortho_vM_vP = np.abs(np.einsum('epi,epi->ep',vecteurM, vecteurP, optimize='optimal'))
            ortho_v_v = np.abs(np.einsum('epi,epi->ep', vecteur_e_pg, vecteur_e_pg, optimize='optimal'))
            if ortho_v_v.min() > 0:
                vertifOrthoEpsPM = np.max(ortho_vP_vM/ortho_v_v)
                assert vertifOrthoEpsPM <= 1e-12
                vertifOrthoEpsMP = np.max(ortho_vM_vP/ortho_v_v)
                assert vertifOrthoEpsMP <= 1e-12
            
        return projP, projM

class Thermal_Model(IModel):

    __modelType = ModelType.thermal

    @property
    def modelType(self) -> ModelType:
        return Thermal_Model.__modelType
    
    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def epaisseur(self) -> float:        
        return self.__epaisseur

    @property
    def resume(self) -> str:
        resume = f'\n{self.nom} :'
        resume += f'\nconduction thermique (k)  : {self.__k}'
        resume += f'\ncapacité thermique massique (c) : {self.__c}'
        return resume

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

        # ThermalModel Anisot avec un coef de diffusion différents pour chaque direction ! k devient une matrice

        self.__c = c
        
        assert epaisseur > 0, "Doit être supérieur à 0"
        self.__epaisseur = epaisseur

        self.Need_Update()

    @property
    def k(self) -> float|np.ndarray:
        """conduction thermique [W . m^-1]"""
        return self.__k

    @property
    def c(self) -> float|np.ndarray:
        """capacité thermique massique [J K^-1 kg^-1]"""
        return self.__c

_erreurConstMateriau = "Il faut faire attention aux dimensions des constantes matériaux.\nSi les constantes matériaux sont dans des array, ces array doivent être de même dimension."

def Resize_variable(variable: int|float|np.ndarray, Ne: int, nPg: int):
    """Redimensionne la variable pour quelle soit sous la forme ep.."""

    if isinstance(variable, (int,float)):
        return np.ones((Ne, nPg)) * variable
    
    elif isinstance(variable, np.ndarray):
        shape = variable.shape
        if len(shape) == 1:
            if shape[0] == Ne:
                variable = variable[:,np.newaxis].repeat(nPg, axis=1)
                return variable
            elif shape[0] == nPg:
                variable = variable[np.newaxis].repeat(Ne, axis=0)
                return variable
            else:
                raise Exception("La variable renseigné doit être de dimension (e) ou (p)")

        if len(shape) == 2:
            if shape == (Ne, nPg):
                return variable
            else:
                variable = variable[np.newaxis, np.newaxis]
                variable = variable.repeat(Ne, axis=0)
                variable = variable.repeat(nPg, axis=1)
                return variable
            
        elif len(shape) == 3:
            if shape[0] == Ne:
                variable = variable[:, np.newaxis].repeat(nPg, axis=1)
                return variable
            elif shape[0] == nPg:
                variable = variable[np.newaxis].repeat(Ne, axis=0)
                return variable
            else:
                raise Exception("La variable renseigné doit êrtre de dimension (eij) ou (pij)")



def Uniform_Array(array: np.ndarray):
    """Redimensionne l'array"""

    dimI, dimJ = array.shape
    
    shapes = [np.shape(array[i,j]) for i in range(dimI) for j in range(dimJ) if len(np.shape(array[i,j]))>0]
    if len(shapes) > 0:
        idx = np.argmax([len(shape) for shape in shapes])
        shape = shapes[idx]
    else:
        shape = ()

    shapeNew = list(shape); shapeNew.extend(array.shape)

    newArray = np.zeros(shapeNew)
    def SetMat(i,j):
        values = array[i,j]
        if isinstance(values, (int, float)):
            values = np.ones(shape) * values
        if len(shape) == 0:
            newArray[i,j] = values
        elif len(shape) == 1:
            newArray[:,i,j] = values
        elif len(shape) == 2:
            newArray[:,:,i,j] = values
        else:
            raise Exception("Les constantes matériaux doivent être au maximum de dimension (Ne, nPg)")
    [SetMat(i,j) for i in range(dimI) for j in range(dimJ)]

    return newArray