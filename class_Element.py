from typing import cast

from class_Noeud import Noeud
import numpy as np

class Element:    

    def get_nPe(self):
        """Renvoie le nombre de noeud par element

        Returns
        -------
        int
            Noeud par element
        """
        return int(len(self.noeuds))
    nPe = property(get_nPe)
    
    def get_nbFaces(self):
        if self.__dim == 2:
            return 1
        else:
            # TETRA4
            if self.nPe == 4:
                return 4 

    def __init__(self,id: int, noeuds: list, dim: int):
        """Constructeur d'element, on construit Be et le jacobien !

        Parameters
        ----------
        id : int
            Numéro de l'élement (>=0)
        noeuds : list
            Liste de noeuds de l'element
        """

        # Vérification des variables
        assert isinstance(id, int),"Doit être un entier"
        assert id >= 0,"Doit être >= 0"

        assert isinstance(noeuds, list),"Doit être une liste"
        assert len(noeuds) > 0, "La liste est vide"
        assert isinstance(noeuds[0], Noeud), "Doit être une liste de noeud"
        
        assert dim in [2,3], "Dimesion compris entre 2D et 3D"
        
        # Création des variables de la classe        
        self.__dim = dim
        self.id = id        
        self.noeuds = noeuds
        self.type = Element.getElementType(self.__dim, self.nPe)
        
        # Construit la matrice assembly
        self.assembly = []
        for n in self.noeuds:            
            # Pour chaque noeud on ajoute lelement    
            n.AddElement(self)            
            if dim == 2:
                self.assembly.append(n.id * 2)
                self.assembly.append(n.id * 2 + 1)                
            elif dim ==3:
                self.assembly.append(n.id * 3)
                self.assembly.append(n.id * 3 + 1)
                self.assembly.append(n.id * 3 + 2)
        
    def ConstruitKe(self, C: np.array):
        
        self.__listJacobien = []
        self.listBeAuNoeuds = []
        self.__C = C
        
        # Construiction Ke        
        taille = self.nPe*self.__dim        
        self.__Ke = np.zeros((taille, taille))

        if self.__dim == 2:        
            # Triangle à 3 noeuds ou 6 noeuds Application linéaire
            if self.nPe == 3 or self.nPe == 6:
                self.__ConstruitKeTriangle()
            elif self.nPe == 4 or self.nPe == 8:
                self.__ConstruitKeQuadrangle()
        elif self.__dim == 3:
            if self.nPe == 4:
                self.__ConstruitKeTetraedre()

        return self.__Ke

        
        
    def __ConstruitKeTriangle(self):
        """Construit la matrice Ke d'un element triangulaire
        """
            
        if self.nPe == 3:        
            
            # Calul du jacobien pour une application lineaire
            matriceCoef = np.array([[0, 0, 1],
                                    [1, 0, 1],
                                    [0, 1, 1]])

            constX, constY = self.CalculLesConstantes(matriceCoef)
                
            alpha = constX[0]
            beta = constX[1]
            gamma = constX[2]                    

            a = constY[0]
            b = constY[1]
            c = constY[2]

            F = np.array([  [alpha, beta],
                            [a, b]   ])
            
            jacobien = np.linalg.det(F)
            # jacobien = alpha * b - beta * a 
            invF = np.linalg.inv(F)
            # invF = 1/jacobien * np.array([  [b, -beta],
            #                                 [-a, alpha]  ])
            
            # Calcul de Be
            dN1t = np.array([-1, -1])
            dN2t = np.array([1, 0])
            dN3t = np.array([0, 1])
            
            self.__listJacobien.append(jacobien)
            
            Be = self.ConstruitBe([dN1t, dN2t, dN3t], invF)
            
            self.__Ke = jacobien * 1/2 * Be.T.dot(self.__C).dot(Be)
            
            self.listBeAuNoeuds = [Be] * 3            
            
        if self.nPe == 6:
            
            matriceCoef = np.array([[0, 0, 0, 0, 0, 1],
                                    [1, 0, 0, 1, 0, 1],
                                    [0, 1, 0, 0, 1, 1],
                                    [1/4, 0, 0, 1/2, 0, 1],
                                    [1/4, 1/4, 1/4, 1/2, 1/2, 1],
                                    [0, 1/4, 0, 0, 1/2, 1]])
            
            constX, constY = self.CalculLesConstantes(matriceCoef)
            
            alpha = constX[0]
            beta = constX[1]
            gamma = constX[2]
            delta = constX[3]
            epsilon = constX[4]
            phi = constX[5]

            a = constY[0]
            b = constY[1]
            c = constY[2]
            d = constY[3]
            e = constY[4]
            f = constY[5]
            
            def ConstruitNtild(ksi, eta):
                N1t = np.array([4*ksi+4*eta-3] *2)
                N2t = np.array([4*ksi-1, 0])
                N3t = np.array([0, 4*eta-1])
                N4t = np.array([4-8*ksi-4*eta, -4*ksi])
                N5t = np.array([4*eta, 4*ksi])
                N6t = np.array([-4*eta, 4-4*ksi-8*eta])
                return [N1t, N2t, N3t, N4t, N5t, N6t]
            
            def ConstruitF(ksi, eta):
                F = np.array([[alpha*2*ksi+gamma*eta+delta, beta*2*eta+gamma*ksi+epsilon],
                              [a*2*ksi+c*eta+d, b*2*eta+c*ksi+e]])                  
                return F               
            
            # Pour chaque point d'integration on calcul Be
            ksis = [1/6, 2/3, 1/6]
            etas = [1/6, 1/6, 2/3]
            poids = [1/6] * 3

            pg = 0
            while pg < len(ksis):
                
                ksi = ksis[pg] 
                eta = etas[pg]
                poid = poids[pg]
                
                Ntild = ConstruitNtild(ksi, eta)
                
                F = ConstruitF(ksi, eta)
                
                invF = np.linalg.inv(F)

                jacobien = np.linalg.det(F)
                
                Be = self.ConstruitBe(Ntild, invF)
                
                self.__listJacobien.append(jacobien)
                self.__Ke = self.__Ke + jacobien * poid * Be.T.dot(self.__C).dot(Be)
                
                pg += 1
            
            # Pour chaque noeuds on calcul Be  
            ksis = [0, 1, 0, 1/2, 1/2, 0]
            etas = [0, 0, 1, 0, 1/2, 1/2]
            
            i = 0
            while i < len(ksis):
                
                ksi = ksis[i]
                eta = etas[i]
                
                Ntild = ConstruitNtild(ksi, eta)               
                
                F = ConstruitF(ksi, eta)

                invF = np.linalg.inv(F)

                Be = self.ConstruitBe(Ntild, invF)
                self.listBeAuNoeuds.append(Be)
                
                i += 1
    
    def __ConstruitKeQuadrangle(self):
        """Construit la matrice Be d'un element quadrillatère
        """
        if self.nPe == 4:
            matriceCoef = np.array([[1, -1, -1, 1],
                                    [-1, 1, -1, 1],
                                    [1, 1, 1, 1],
                                    [-1, -1, 1, 1]])
            
            constX, constY = self.CalculLesConstantes(matriceCoef)
            
            alpha = constX[0]
            beta = constX[1]
            gamma = constX[2]
            delta = constX[3]            

            a = constY[0]
            b = constY[1]
            c = constY[2]
            d = constY[3]
            
            def ConstruitNtild(ksi, eta):
                N1t = np.array([(eta-1)/4, (ksi-1)/4])
                N2t = np.array([(1-eta)/4, (-ksi-1)/4])
                N3t = np.array([(1+eta)/4, (1+ksi)/4])
                N4t = np.array([(-eta-1)/4, (1-ksi)/4])                
                return [N1t, N2t, N3t, N4t]
            
            def ConstruitF(ksi, eta):
                F = np.array([[alpha*eta+beta, alpha*ksi+gamma],
                              [a*eta+b, a*ksi+c]])                  
                return F                              
            
            # Pour chaque point d'integration on calcul Be
            UnSurRacine3 = 1/np.sqrt(3) 
            ksis = [-UnSurRacine3, UnSurRacine3, UnSurRacine3, -UnSurRacine3]
            etas = [-UnSurRacine3, -UnSurRacine3, UnSurRacine3, UnSurRacine3]
            poids = [1] * 4
            
            pg = 0
            while pg < len(ksis):
                
                ksi = ksis[pg] 
                eta = etas[pg]
                poid = poids[pg]
                
                Ntild = ConstruitNtild(ksi, eta)
                
                F = ConstruitF(ksi, eta)    
                
                invF = np.linalg.inv(F)

                jacobien = np.linalg.det(F)
                
                Be = self.ConstruitBe(Ntild, invF)
                
                self.__listJacobien.append(jacobien)                
                self.__Ke = self.__Ke + jacobien * poid * Be.T.dot(self.__C).dot(Be)
                
                pg += 1
                
            # Pour chaque noeuds on calcul Be    
            
            ksis = [-1, 1, 1, -1]
            etas = [-1, -1, 1, 1]
            
            i = 0
            while i < len(ksis):
                
                ksi = ksis[i]
                eta = etas[i]
                
                Ntild = ConstruitNtild(ksi, eta)               
                
                F = ConstruitF(ksi, eta)

                invF = np.linalg.inv(F)
                                             
                Be = self.ConstruitBe(Ntild, invF)
                self.listBeAuNoeuds.append(Be)
                
                i += 1  
            
              
        elif self.nPe ==8:
            matriceCoef = np.array([[-1, -1, 1, 1, 1, -1, -1, 1],
                                    [-1, 1, 1, 1, -1, 1, -1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, -1, 1, 1, -1, -1, 1, 1],
                                    [0, 0, 0, 1, 0, 0, -1, 1],
                                    [0, 0, 1, 0, 0, 1, 0, 1],
                                    [0, 0, 0, 1, 0, 0, 1, 1],
                                    [0, 0, 1, 0, 0, -1, 0, 1]])
            
            constX, constY = self.CalculLesConstantes(matriceCoef)
            
            a = constX[0]
            b = constX[1]
            c = constX[2]
            d = constX[3]
            e = constX[4]
            f = constX[5]
            g = constX[6]
            h = constX[7]              

            a2 = constY[0]
            b2 = constY[1]
            c2 = constY[2]
            d2 = constY[3]
            e2 = constY[4]
            f2 = constY[5]
            g2 = constY[6]
            h2 = constY[7]
            
            def ConstruitNtild(ksi, eta):
                N1t = np.array([(1-eta)*(2*ksi+eta)/4, (1-ksi)*(ksi+2*eta)/4])
                N2t = np.array([(1-eta)*(2*ksi-eta)/4, -(1+ksi)*(ksi-2*eta)/4])
                N3t = np.array([(1+eta)*(2*ksi+eta)/4, (1+ksi)*(ksi+2*eta)/4])
                N4t = np.array([-(1+eta)*(-2*ksi+eta)/4, (1-ksi)*(-ksi+2*eta)/4])
                N5t = np.array([-ksi*(1-eta), -(1-ksi**2)/2])
                N6t = np.array([(1-eta**2)/2, -eta*(1+ksi)])
                N7t = np.array([-ksi*(1+eta), (1-ksi**2)/2])                
                N8t = np.array([-(1-eta**2)/2, -eta*(1-ksi)])
                                
                return [N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t]                               
                
            
            def ConstruitF(ksi, eta):
                F = np.array([[2*a*ksi*eta+b*eta**2+2*c*ksi+e*eta+f, a*ksi**2+2*b*eta*ksi+2*d*eta+e*ksi+g],
                              [2*a2*ksi*eta+b2*eta**2+2*c2*ksi+e2*eta+f2, a2*ksi**2+2*b2*eta*ksi+2*d2*eta+e2*ksi+g2]])                 
                return F                              
            
            # Pour chaque point d'integration on calcul Be            
            UnSurRacine3 = 1/np.sqrt(3) 
            ksis = [-UnSurRacine3, UnSurRacine3, UnSurRacine3, -UnSurRacine3]
            etas = [-UnSurRacine3, -UnSurRacine3, UnSurRacine3, UnSurRacine3]
            poids = [1] * 4
            
            pg = 0
            while pg < len(ksis):
                
                ksi = ksis[pg] 
                eta = etas[pg]
                poid = poids[pg]
                
                Ntild = ConstruitNtild(ksi, eta)
                
                F = ConstruitF(ksi, eta)    
                
                invF = np.linalg.inv(F)
                
                jacobien = np.linalg.det(F)
                
                Be = self.ConstruitBe(Ntild, invF)
                
                self.__listJacobien.append(jacobien)
                self.__Ke = self.__Ke + jacobien * poid * Be.T.dot(self.__C).dot(Be)
                
                pg += 1
                
            # Pour chaque noeuds on calcul Be               
            ksis = [-1, 1, 1, -1, 0, 1, 0, -1]
            etas = [-1, -1, 1, 1,-1, 0, 1, 0]
            
            i = 0
            while i < len(ksis):
                
                ksi = ksis[i]
                eta = etas[i]
                
                Ntild = ConstruitNtild(ksi, eta)               
                
                F = ConstruitF(ksi, eta)
                
                invF = np.linalg.inv(F)
   
                Be = self.ConstruitBe(Ntild, invF)
                self.listBeAuNoeuds.append(Be)
                
                i += 1
    
    def __ConstruitKeTetraedre(self):
        if self.nPe == 4:
            matriceCoef = np.array([[0, 0, 0, 1],
                                    [1, 0, 0, 1],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 1]])        
           
            constX, constY, constZ = self.CalculLesConstantes(matriceCoef)
            
            a1 = constX[0]
            b1 = constX[1]
            c1 = constX[2]
            d1 = constX[3]            

            a2 = constY[0]
            b2 = constY[1]
            c2 = constY[2]
            d2 = constY[3]
            
            a3 = constZ[0]
            b3 = constZ[1]
            c3 = constZ[2]
            d3 = constZ[3]
            
            # F = np.array([[a1, a2, a3],
            #               [b1, b2, b3],                              
            #               [c1, c2, c3]])

            F = np.array([[a1, b1, c1],
                          [a2, b2, c2],                              
                          [a3, b3, c3]])                          
            
            invF = np.linalg.inv(F)

            jacobien = np.linalg.det(F)
            
            self.__listJacobien.append(jacobien)
            
            N1t = np.array([-1, -1, -1])
            N2t = np.array([1, 0, 0])
            N3t = np.array([0, 1, 0])
            N4t = np.array([0, 0, 1])
            
            Be = self.ConstruitBe([N1t, N2t, N3t, N4t], invF)
            
            self.__Ke = jacobien * 1/6 * Be.T.dot(self.__C).dot(Be)
            
            self.listBeAuNoeuds = [Be] * 4
    
    def CalculLesConstantes(self, matriceCoef: np.ndarray):
        """Determine les constantes pour passer de l'element de reference a lelement reele

        Parameters
        ----------
        matriceCoef
            Matrice de coef

        Returns
        -------
        tuple
            Retourne les vecteurs de constantes 
        """
        vectX = []
        vectY = []
        vectZ = []
        for noeud in self.noeuds:
            noeud = cast(Noeud, noeud)
            
            vectX.append(noeud.coordo[0])
            vectY.append(noeud.coordo[1])
            vectZ.append(noeud.coordo[2])
        
        vectX = np.array(vectX)
        vectY = np.array(vectY)
        vectZ = np.array(vectZ)

        inv_matriceCoef = np.linalg.inv(matriceCoef)
        
        constX = inv_matriceCoef.dot(vectX)
        constY = inv_matriceCoef.dot(vectY)
        constZ = inv_matriceCoef.dot(vectZ)

        if self.__dim == 2:
            return constX, constY
        elif self.__dim == 3:
            return constX, constY, constZ

    def ConstruitBe(self, list_Ntild: list, invF: np.ndarray):  
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
        
        list_Ntild = np.array(list_Ntild)
        # Transpose la matrice F inversé
        invFT = np.array(invF).T
        
        if self.__dim == 2:            
            Be = np.zeros((3,len(list_Ntild)*2))      

            i = 0
            for nt in list_Ntild:            
                dNdx = invFT[0].dot(nt)
                dNdy = invFT[1].dot(nt)
                
                Be[0, i] = dNdx
                Be[1, i+1] = dNdy
                Be[2, i] = dNdy; Be[2, i+1] = dNdx    
                i += 2
        elif self.__dim == 3:
            Be = np.zeros((6,len(list_Ntild)*3))      

            i = 0
            for nt in list_Ntild:            
                dNdx = invFT[0].dot(nt)
                dNdy = invFT[1].dot(nt)
                dNdz = invFT[2].dot(nt)
                
                Be[0, i] = dNdx
                Be[1, i+1] = dNdy
                Be[2, i+2] = dNdz
                Be[3, i] = dNdy; Be[3, i+1] = dNdx
                Be[4, i+1] = dNdz; Be[4, i+2] = dNdy
                Be[5, i] = dNdz; Be[5, i+2] = dNdx
                i += 3
            pass
                
                
        return Be   
    
    @staticmethod    
    def getElementType(dim: int, i: int):
        """Renvoie le type de l'élément en fonction du nombre de noeuds par élement

        Parameters
        ----------
        i : int
            Nombre de noeud par element

        Returns
        -------
        str
            Renvoie un string qui caractérise l'element
        """
        
        if dim == 2:        
            switch = {
                3 : "Triangle à 3 noeuds",
                6 : "Triangle à 6 noeuds",
                4 : "Quadrillatère à 4 noeuds",
                8 : "Quadrillatère à 8 noeuds",
            }                
            return switch[i]
        if dim == 3:
            switch = {
                4 : "Tétraèdre à 4 noeuds",                
            }                
            return switch[i]

    def RenvoieLesNumsDeNoeudsTriés(self):
        """Trie les noeuds de l'element et les renvoie dessiner le maillage

        Returns
        -------
        liste de Noeud
            Renvoie le numéro des noeuds triés
        """

        assert len(self.noeuds)>0, "L'élément n'a pas de noeuds"       

        noeuds = []

        if self.nPe == 3 or self.nPe == 4:
            for n in self.noeuds:
                noeuds.append(n.id) 
        elif self.nPe == 6:
            for i in [0, 3, 1, 4, 2, 5]:
                noeuds.append(self.noeuds[i].id)        
        elif self.nPe == 8:
            for i in [0, 4, 1, 5, 2, 6, 3, 7]:
                noeuds.append(self.noeuds[i].id)
                
        return list(noeuds)

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

    def test_RenvoiePolygon(self):
        numTrie  = self.element.RenvoieLesNumsDeNoeudsTriés()                    
        self.assertIsInstance(numTrie, list)        
        self.assertListEqual(numTrie,[0,1,3])

if __name__ == '__main__':        
    try:
        os.system("cls")    #nettoie terminal
        unittest.main(verbosity=2)    
    except:
        print("")