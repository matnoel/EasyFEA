import numpy as np

class Element:

    __listElement2D = ["TRI3", "TRI6", "QUAD4", "QUAD8"]
    
    __listElement3D = ["TETRA4"]
    

    @staticmethod
    def get_Types2D():
        """type d'elements disponibles en 2D"""
        return Element.__listElement2D.copy()
    
    @staticmethod
    def get_Types3D():
        """type d'elements disponibles en 3D"""
        return Element.__listElement3D.copy()

    def __get_ElementType(self):
        """Renvoie le type de l'élément en fonction du nombre de noeuds par élement
        """        
        if self.__dim == 2:        
            switch = {
                3 : "TRI3",
                6 : "TRI6",
                4 : "QUAD4",
                8 : "QUAD8",
            }                
            return switch[self.nPe]
        if self.__dim == 3:
            switch = {
                4 : "TETRA4",                
            }                
            return switch[self.nPe]
    type = property(__get_ElementType) 
    """type de l'élement"""

    def __get_nPg(self):
        return self.gauss.shape[0]
    nPg = property(__get_nPg)
    """nombre de points d'intégrations"""

    def __init__(self, dim: int, nPe: int):
        """Constructeur d'element, on construit Be et le jacobien !

        Parameters
        ----------
        dim : int
            Numéro de l'élement (>=0)
        nPe : int
            Nombre de noeud par element
        """
        assert dim in [2,3], "Dimesion compris entre 2D et 3D"
        
        # Création des variables de la classe        
        self.__dim = dim
        """dimension de l'élement [2,3]"""
        
        self.nPe = nPe
        """noeuds par élément"""
        
        self.gauss = np.zeros((0,3))
        """ points d'intégrations (pt ksi eta poid) """
        
        self.N_rigi_pg = np.zeros((dim, nPe*dim))
        """ Fonctions de formes vectorielles (pg, dim, nPe*dim) : \n 
            [Ni 0 . . . Nn 0 \n
             0 Ni . . . 0 Nn]"""

        # [N1 ... Nn]
        self.N_mass_pg = np.zeros((1, nPe))
        """ Fonctions de formes scalaires (pg, 1, nPe) : \n
            [Ni . . . Nn]"""

        self.dN_pg = np.zeros((dim, nPe))
        """ Dérivées des fonctions de formes dans l'element de référence (pg, dim, nPe) : \n
        [Ni,ksi . . . Nn,ksi \n
        Ni,eta . . . Nn,eta]"""
                     
        self.__Construit_B_N()

    def __Construit_B_N(self):
        """Construit les fonctions de forme et leur dérivée pour l'element de référence"""
        
        if self.__dim == 2:        
            # Triangle à 3 noeuds ou 6 noeuds Application linéaire
            if self.nPe == 3 or self.nPe == 6:
                self.__Construit_B_N_Triangle()
            elif self.nPe == 4 or self.nPe == 8:
                self.__Construit_B_N_Quadrangle()
        elif self.__dim == 3:
            if self.nPe == 4:
                self.__Construit_B_N_Tetraedre()                

    def __Construit_B_N_Triangle(self):

        # TRI3
        if self.nPe == 3:  
            
            # Points de gauss
            ksi = 1/3
            eta = 1/3
            poid = 1/2
            self.gauss = np.array([ksi, eta, poid]).reshape((1,3))

            # Calcul N aux points de gauss
            N1t = 1-ksi-eta
            N2t = ksi
            N3t = eta
            Ntild = [N1t, N2t, N3t]

            self.N_rigi_pg = self.__ConstruitN(Ntild).reshape((1,2,-1))

            self.N_mass_pg = np.array(Ntild).reshape((1,1,-1))

            self.dN_pg = np.array([[-1, 1, 0],[-1, 0, 1]]).reshape((1,2,-1))

        # TRI6  
        if self.nPe == 6:
            
            # Points de gauss
            ksis = [1/6, 2/3, 1/6]
            etas = [1/6, 1/6, 2/3]
            poids = [1/6] * 3
            self.gauss = np.array([ksis, etas, poids]).T
            
            def Construit_Ntild(ksi, eta):
                # Code aster (Fonctions de forme et points d'intégration des élé[...])
                N1t = -(1-ksi-eta)*(1-2*(1-ksi-eta))
                N2t = -ksi*(1-2*ksi)
                N3t = -eta*(1-2*eta)
                N4t = 4*ksi*(1-ksi-eta)
                N5t = 4*ksi*eta
                N6t = 4*eta*(1-ksi-eta)
                return np.array([N1t, N2t, N3t, N4t, N5t, N6t])

            def Construit_dNtild(ksi, eta):
                dN1t = np.array([4*ksi+4*eta-3] *2)
                dN2t = np.array([4*ksi-1, 0])
                dN3t = np.array([0, 4*eta-1])
                dN4t = np.array([4-8*ksi-4*eta, -4*ksi])
                dN5t = np.array([4*eta, 4*ksi])
                dN6t = np.array([-4*eta, 4-4*ksi-8*eta])
                return np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t]).T
            
            N_rigi_pg = []
            N_mass_pg = []
            dN_pg = []

            for pg in range(len(ksis)):
                
                ksi = ksis[pg] 
                eta = etas[pg]
                
                Ntild = Construit_Ntild(ksi, eta)                
                
                N_rigi = self.__ConstruitN(Ntild)
                N_rigi_pg.append(N_rigi)

                N_mass = self.__ConstruitN(Ntild, vecteur=False)
                N_mass_pg.append(N_mass)
                
                dNtild = Construit_dNtild(ksi, eta)
                dN_pg.append(dNtild)
            
            self.N_rigi_pg = np.array(N_rigi_pg)
            self.N_mass_pg = np.array(N_mass_pg)
            self.dN_pg = np.array(dN_pg)
            

    def __Construit_B_N_Quadrangle(self):
        
        if self.nPe == 4:
            
            # Points de gauss
            UnSurRacine3 = 1/np.sqrt(3) 
            ksis = [-UnSurRacine3, UnSurRacine3, UnSurRacine3, -UnSurRacine3]
            etas = [-UnSurRacine3, -UnSurRacine3, UnSurRacine3, UnSurRacine3]
            poids = [1]*4
            self.gauss = np.array([ksis, etas, poids]).T

            def Construit_Ntild(ksi, eta):
                N1t = (1-ksi)*(1-eta)/4
                N2t = (1+ksi)*(1-eta)/4
                N3t = (1+ksi)*(1+eta)/4
                N4t = (1-ksi)*(1+eta)/4
                return np.array([N1t, N2t, N3t, N4t])
            
            def Construit_dNtild(ksi, eta):
                dN1t = np.array([(eta-1)/4, (ksi-1)/4])
                dN2t = np.array([(1-eta)/4, (-ksi-1)/4])
                dN3t = np.array([(1+eta)/4, (1+ksi)/4])
                dN4t = np.array([(-eta-1)/4, (1-ksi)/4])                
                return np.array([dN1t, dN2t, dN3t, dN4t]).T
            
            N_rigi_pg = []
            N_mass_pg = []
            dN_pg = []

            for pg in range(len(ksis)):
                
                ksi = ksis[pg] 
                eta = etas[pg]
                
                Ntild = Construit_Ntild(ksi, eta)                
                
                N_rigi = self.__ConstruitN(Ntild)
                N_rigi_pg.append(N_rigi)

                N_mass = self.__ConstruitN(Ntild, vecteur=False)
                N_mass_pg.append(N_mass)
                
                dNtild = Construit_dNtild(ksi, eta)
                dN_pg.append(dNtild)
            
            self.N_rigi_pg = np.array(N_rigi_pg)
            self.N_mass_pg = np.array(N_mass_pg)
            self.dN_pg = np.array(dN_pg)
              
        elif self.nPe ==8:
            
            # Points de gauss
            UnSurRacine3 = 1/np.sqrt(3) 
            ksis = [-UnSurRacine3, UnSurRacine3, UnSurRacine3, -UnSurRacine3]
            etas = [-UnSurRacine3, -UnSurRacine3, UnSurRacine3, UnSurRacine3]
            poids = [1]*4
            self.gauss = np.array([ksis, etas, poids]).T

            def Construit_Ntild(Ksi, eta):
                N1t = (1-ksi)*(1-eta)*(-1-ksi-eta)/4
                N2t = (1+ksi)*(1-eta)*(-1+ksi-eta)/4
                N3t = (1+ksi)*(1+eta)*(-1+ksi+eta)/4
                N4t = (1-ksi)*(1+eta)*(-1-ksi+eta)/4
                N5t = (1-ksi**2)*(1-eta)/2
                N6t = (1+ksi)*(1-eta**2)/2
                N7t = (1-ksi**2)*(1+eta)/2
                N8t = (1-ksi)*(1-eta**2)/2
                return np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t])

            def Construit_dNtild(ksi, eta):
                dN1t = np.array([(1-eta)*(2*ksi+eta)/4, (1-ksi)*(ksi+2*eta)/4])
                dN2t = np.array([(1-eta)*(2*ksi-eta)/4, -(1+ksi)*(ksi-2*eta)/4])
                dN3t = np.array([(1+eta)*(2*ksi+eta)/4, (1+ksi)*(ksi+2*eta)/4])
                dN4t = np.array([-(1+eta)*(-2*ksi+eta)/4, (1-ksi)*(-ksi+2*eta)/4])
                dN5t = np.array([-ksi*(1-eta), -(1-ksi**2)/2])
                dN6t = np.array([(1-eta**2)/2, -eta*(1+ksi)])
                dN7t = np.array([-ksi*(1+eta), (1-ksi**2)/2])                
                dN8t = np.array([-(1-eta**2)/2, -eta*(1-ksi)])
                                
                return np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t]).T

            N_rigi_pg = []
            N_mass_pg = []
            dN_pg = []

            for pg in range(len(ksis)):
                
                ksi = ksis[pg] 
                eta = etas[pg]
                
                Ntild = Construit_Ntild(ksi, eta)                
                
                N_rigi = self.__ConstruitN(Ntild)
                N_rigi_pg.append(N_rigi)

                N_mass = self.__ConstruitN(Ntild, vecteur=False)
                N_mass_pg.append(N_mass)
                
                dNtild = Construit_dNtild(ksi, eta)
                dN_pg.append(dNtild)
            
            self.N_rigi_pg = np.array(N_rigi_pg)
            self.N_mass_pg = np.array(N_mass_pg)
            self.dN_pg = np.array(dN_pg)
            
    def __Construit_B_N_Tetraedre(self):
        if self.nPe == 4:                       
            
            # Points de gauss
            x = 1/4
            y = 1/4
            z = 1/4
            poid = 1/6
            self.gauss = np.array([x, y, z, poid]).reshape((1,4))

            # Construit Ntild
            N1t = 1-x-y-z
            N2t = x
            N3t = y
            N4t = z            
            Ntild = np.array([N1t, N2t, N3t, N4t])
            
            self.N_rigi_pg = self.__ConstruitN(Ntild).reshape((1,3,-1))

            self.N_mass_pg = self.__ConstruitN(Ntild, vecteur=False).reshape((1,1,-1))

            # Construit dNtild
            dN1t = np.array([-1, -1, -1])
            dN2t = np.array([1, 0, 0])
            dN3t = np.array([0, 1, 0])
            dN4t = np.array([0, 0, 1])
            dNtild = np.array([dN1t, dN2t, dN3t, dN4t]).T
            self.dN_pg = dNtild.reshape((1,3,-1))
    
    def __ConstruitN(self, Ntild: np.ndarray, vecteur=True):
        """Construit la matrice de fonction de forme

        Parameters
        ----------
        Ntild : list des fonctions Ntild
            Fonctions Ntild
        vecteur : bool, optional
            Option qui permet de construire N pour un probleme de déplacement ou un problème thermique, by default True

        Returns
        -------
        ndarray
            Renvoie la matrice Ntild
        """
        if vecteur:
            N_pg = np.zeros((self.__dim, len(Ntild)*self.__dim))
            
            colonne = 0
            for nt in Ntild:
                for ligne in range(self.__dim):
                    N_pg[ligne, colonne] = nt
                    colonne += 1
        else:
            N_pg = np.zeros((1, len(Ntild)))
            colonne = 0
            for nt in Ntild:
                N_pg[0, colonne] = nt
                colonne += 1            

        return N_pg

# ====================================

import unittest
import os

class Test_Element(unittest.TestCase):
    
    def setUp(self):
        self.element = Element(1,3)  

    def test_BienCree(self):
        self.assertIsInstance(self.element, Element)        

if __name__ == '__main__':        
    try:
        os.system("cls")    #nettoie terminal
        unittest.main(verbosity=2)    
    except:
        print("")