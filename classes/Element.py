from typing import cast

import numpy as np
try:
    from classes.Noeud import Noeud
except:
    from Noeud import Noeud

class Element:

    def get_nbFaces(self):
        if self.__dim == 2:
            return 1
        else:
            # TETRA4
            if self.nPe == 4:
                return 4
    
    def get_ElementType(self):
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
    type = property(get_ElementType) 

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
        
        self.nPe = nPe

        self.nPg = 0

        # définit aux points de gauss
        self.listPoid_pg = []

        # [N1 0 ... Nn 0
        #  0 N1 ... 0 Nn]
        self.listN_rigi_pg = []

        # [N1 ... Nn]
        self.listN_mass_pg = []

        # [N1,ksi ... Nn,ksi
        #  N1,eta ... Nn,eta]
        self.listdN_pg = []
                     
        self.__Construit_B_N()

    def __Construit_B_N(self):
        
        # Construit les fonctions de forme et leur dérivée pour l'element de référence

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
            self.listPoid_pg = [1/2]
            self.nPg = len(self.listPoid_pg)

            # Calcul N aux points de gauss
            N1t = 1-ksi-eta
            N2t = ksi
            N3t = eta
            Ntild = [N1t, N2t, N3t]

            self.listN_rigi_pg.append(self.__ConstruitN(Ntild))

            self.listN_mass_pg.append(np.array(Ntild))

            self.listdN_pg.append(np.array([[-1, 1, 0],[-1, 0, 1]]))

        # TRI6  
        if self.nPe == 6:
            
            # Points de gauss
            ksis = [1/6, 2/3, 1/6]
            etas = [1/6, 1/6, 2/3]

            self.listPoid_pg = [1/6] * 3
            self.nPg = len(self.listPoid_pg)           
            
            def Construit_Ntild(ksi, eta):
                # Code aster (Fonctions de forme et points d'intégration des élé[...])
                N1t = -(1-ksi-eta)*(1-2*(1-ksi-eta))
                N2t = -ksi*(1-2*ksi)
                N3t = -eta*(1-2*eta)
                N4t = 4*ksi*(1-ksi-eta)
                N5t = 4*ksi*eta
                N6t = 4*eta*(1-ksi-eta)
                return [N1t, N2t, N3t, N4t, N5t, N6t]

            def Construit_dNtild(ksi, eta):
                dN1t = np.array([4*ksi+4*eta-3] *2)
                dN2t = np.array([4*ksi-1, 0])
                dN3t = np.array([0, 4*eta-1])
                dN4t = np.array([4-8*ksi-4*eta, -4*ksi])
                dN5t = np.array([4*eta, 4*ksi])
                dN6t = np.array([-4*eta, 4-4*ksi-8*eta])
                return [dN1t, dN2t, dN3t, dN4t, dN5t, dN6t]
            
            for pg in range(len(ksis)):
                
                ksi = ksis[pg] 
                eta = etas[pg]
                
                Ntild = Construit_Ntild(ksi, eta)                
                
                N_rigi = self.__ConstruitN(Ntild)
                self.listN_rigi_pg.append(N_rigi)

                N_mass = self.__ConstruitN(Ntild, vecteur=False)
                self.listN_mass_pg.append(N_mass)
                
                dNtild = Construit_dNtild(ksi, eta)
                self.listdN_pg.append(dNtild)

    def __Construit_B_N_Quadrangle(self):
        """Construit la matrice Be d'un element quadrillatère
        """
        if self.nPe == 4:
            
            # Points de gauss
            UnSurRacine3 = 1/np.sqrt(3) 
            ksis = [-UnSurRacine3, UnSurRacine3, UnSurRacine3, -UnSurRacine3]
            etas = [-UnSurRacine3, -UnSurRacine3, UnSurRacine3, UnSurRacine3]
            self.listPoid_pg = [1] * 4
            self.nPg = len(self.listPoid_pg)

            def Construit_Ntild(ksi, eta):
                N1t = (1-ksi)*(1-eta)/4
                N2t = (1+ksi)*(1-eta)/4
                N3t = (1+ksi)*(1+eta)/4
                N4t = (1-ksi)*(1+eta)/4
                return [N1t, N2t, N3t, N4t]
            
            def Construit_dNtild(ksi, eta):
                dN1t = np.array([(eta-1)/4, (ksi-1)/4])
                dN2t = np.array([(1-eta)/4, (-ksi-1)/4])
                dN3t = np.array([(1+eta)/4, (1+ksi)/4])
                dN4t = np.array([(-eta-1)/4, (1-ksi)/4])                
                return [dN1t, dN2t, dN3t, dN4t]
            
            for pg in range(len(ksis)):
                
                ksi = ksis[pg] 
                eta = etas[pg]
                
                Ntild = Construit_Ntild(ksi, eta)
                
                N_rigi = self.__ConstruitN(Ntild)
                self.listN_rigi_pg.append(N_rigi)

                N_mass = self.__ConstruitN(Ntild, vecteur=False)                
                self.listN_mass_pg.append(N_mass)

                dNtild = Construit_dNtild(ksi, eta)
                self.listdN_pg.append(dNtild)            
              
        elif self.nPe ==8:
            
            # Points de gauss
            UnSurRacine3 = 1/np.sqrt(3) 
            ksis = [-UnSurRacine3, UnSurRacine3, UnSurRacine3, -UnSurRacine3]
            etas = [-UnSurRacine3, -UnSurRacine3, UnSurRacine3, UnSurRacine3]
            self.listPoid_pg = [1] * 4
            self.nPg = len(self.listPoid_pg)

            def Construit_Ntild(Ksi, eta):
                N1t = (1-ksi)*(1-eta)*(-1-ksi-eta)/4
                N2t = (1+ksi)*(1-eta)*(-1+ksi-eta)/4
                N3t = (1+ksi)*(1+eta)*(-1+ksi+eta)/4
                N4t = (1-ksi)*(1+eta)*(-1-ksi+eta)/4
                N5t = (1-ksi**2)*(1-eta)/2
                N6t = (1+ksi)*(1-eta**2)/2
                N7t = (1-ksi**2)*(1+eta)/2
                N8t = (1-ksi)*(1-eta**2)/2
                return [N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t]

            def Construit_dNtild(ksi, eta):
                dN1t = np.array([(1-eta)*(2*ksi+eta)/4, (1-ksi)*(ksi+2*eta)/4])
                dN2t = np.array([(1-eta)*(2*ksi-eta)/4, -(1+ksi)*(ksi-2*eta)/4])
                dN3t = np.array([(1+eta)*(2*ksi+eta)/4, (1+ksi)*(ksi+2*eta)/4])
                dN4t = np.array([-(1+eta)*(-2*ksi+eta)/4, (1-ksi)*(-ksi+2*eta)/4])
                dN5t = np.array([-ksi*(1-eta), -(1-ksi**2)/2])
                dN6t = np.array([(1-eta**2)/2, -eta*(1+ksi)])
                dN7t = np.array([-ksi*(1+eta), (1-ksi**2)/2])                
                dN8t = np.array([-(1-eta**2)/2, -eta*(1-ksi)])
                                
                return [dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t]

            for pg in range(len(ksis)):
                
                ksi = ksis[pg] 
                eta = etas[pg]
                
                Ntild = Construit_Ntild(ksi, eta)
                
                N_rigi = self.__ConstruitN(Ntild)
                self.listN_rigi_pg.append(N_rigi)

                N_mass = self.__ConstruitN(Ntild, vecteur=False)
                self.listN_mass_pg.append(N_mass)

                dNtild = Construit_dNtild(ksi, eta)
                self.listdN_pg.append(dNtild)
            
    def __Construit_B_N_Tetraedre(self):
        if self.nPe == 4:                       
            
            # Points de gauss
            x = 1/4
            y = 1/4
            z = 1/4
            self.listPoid_pg = [1/6]
            self.nPg = len(self.listPoid_pg)

            # Construit Ntild
            N1t = 1-x-y-z
            N2t = x
            N3t = y
            N4t = z            
            Ntild = [N1t, N2t, N3t, N4t]
            
            self.listN_rigi_pg.append(self.__ConstruitN(Ntild))

            self.listN_mass_pg.append(self.__ConstruitN(Ntild, vecteur=False))

            # Construit dNtild
            dN1t = np.array([-1, -1, -1])
            dN2t = np.array([1, 0, 0])
            dN3t = np.array([0, 1, 0])
            dN4t = np.array([0, 0, 1])
            dNtild = [dN1t, dN2t, dN3t, dN4t]
            self.listdN_pg.append(dNtild)
    
    def ConstruitB_pg(self, list_dNtild: list, invF: np.ndarray, vecteur=True):  
        """Construit la matrice Be depuis les fonctions de formes de l'element
        de reference et l'inverserse de la matrice F

        Parameters
        ----------
        list_Ntild : list
            Liste des vecteurs Ntildix et y
        invF : np.ndarray
            Inverse de la matrice F

        Returns
        -------
        np.ndarray
            si dim = 2
            Renvoie une matrice de dim (3,len(list_Ntild)*2)
            
            si dim = 3
            Renvoie une matrice de dim (6,len(list_Ntild)*3)
        """
        
        # list_dNtild = np.array(list_dNtild)
        
        if vecteur:
            if self.__dim == 2:            
                B_pg = np.zeros((3,len(list_dNtild)*2))      

                colonne = 0
                for dNt in list_dNtild:            
                    dNdx = invF[0].dot(dNt)
                    dNdy = invF[1].dot(dNt)
                    
                    B_pg[0, colonne] = dNdx
                    B_pg[1, colonne+1] = dNdy
                    B_pg[2, colonne] = dNdy; B_pg[2, colonne+1] = dNdx    
                    colonne += 2
            elif self.__dim == 3:
                B_pg = np.zeros((6,len(list_dNtild)*3))

                colonne = 0
                for dNt in list_dNtild:            
                    dNdx = invF[0].dot(dNt)
                    dNdy = invF[1].dot(dNt)
                    dNdz = invF[2].dot(dNt)
                    
                    B_pg[0, colonne] = dNdx
                    B_pg[1, colonne+1] = dNdy
                    B_pg[2, colonne+2] = dNdz
                    B_pg[3, colonne] = dNdy; B_pg[3, colonne+1] = dNdx
                    B_pg[4, colonne+1] = dNdz; B_pg[4, colonne+2] = dNdy
                    B_pg[5, colonne] = dNdz; B_pg[5, colonne+2] = dNdx
                    colonne += 3
        else:
            # Construit B comme pour un probleme de thermique
            B_pg = np.zeros((self.__dim, len(list_dNtild)))
            
            for i in range(len(list_dNtild)):
                dNt = list_dNtild[i]
                for j in range(self.__dim):
                    # j=0 dNdx, j=1 dNdy, j=2 dNdz
                    dNdj = invF[j].dot(dNt)
                    B_pg[j, i] = dNdj

        return B_pg   
    
    def __ConstruitN(self, list_Ntild: list, vecteur=True):
        """Construit la matrice de fonction de forme

        Parameters
        ----------
        list_Ntild : list des fonctions Ntild
            Fonctions Ntild
        vecteur : bool, optional
            Option qui permet de construire N pour un probleme de déplacement ou un problème thermique, by default True

        Returns
        -------
        ndarray
            Renvoie la matrice Ntild
        """
        if vecteur:
            N_pg = np.zeros((self.__dim, len(list_Ntild)*self.__dim))
            
            colonne = 0
            for nt in list_Ntild:
                for ligne in range(self.__dim):
                    N_pg[ligne, colonne] = nt
                    colonne += 1
        else:
            N_pg = np.zeros((1, len(list_Ntild)))
            colonne = 0
            for nt in list_Ntild:
                N_pg[0, colonne] = nt
                colonne += 1            

        return N_pg

# ====================================

import unittest
import os

class Test_Element(unittest.TestCase):
    
    def setUp(self):

        n1 = Noeud(0, [0, 0, 0])
        n2 = Noeud(1, [1, 0, 0])
        n3 = Noeud(3, [0, 1, 0])

        noeuds = [n1, n2, n3]

        self.element = Element(1,noeuds,2)  

    def test_BienCree(self):
        self.assertIsInstance(self.element, Element)
        self.assertEqual(len(self.element.noeuds), 3)
        self.assertListEqual(self.element.assembly, [0, 1, 2, 3, 6, 7])

if __name__ == '__main__':        
    try:
        os.system("cls")    #nettoie terminal
        unittest.main(verbosity=2)    
    except:
        print("")