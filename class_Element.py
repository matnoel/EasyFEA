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
        
        # Construit la matrice assembly
        self.assembly = []
        for n in self.noeuds:
            n = cast(Noeud, n)
            # Pour chaque noeud on ajoute lelement    
            n.AddElement(self)            
            for d in range(dim):
                self.assembly.append(n.id * dim + d)
        
        self.__Construit_B_N()

    def __Construit_B_N(self):
        
        self.listJacobien_pg = []
        self.listB_u_pg = []
        self.listB_d_pg = []
        self.listN_u_pg = []
        self.listN_d_pg = []

        self.listB_u_n = []

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

            # Calul du jacobien pour chaque point de gauss
            matriceCoef = np.array([[0, 0, 1],
                                    [1, 0, 1],
                                    [0, 1, 1]])

            constX, constY = self.__CalculLesConstantes(matriceCoef)
                
            alpha = constX[0]
            beta = constX[1]
            gamma = constX[2]                    

            a = constY[0]
            b = constY[1]
            c = constY[2]

            F = np.array([  [alpha, a],
                            [beta, b]   ])
            
            jacobien = np.linalg.det(F)
            # jacobien = alpha * b - beta * a             
            self.listJacobien_pg.append(jacobien)

            # Points de gauss
            ksi = 1/3
            eta = 1/3
            self.listPoid_pg = [1/2]

            # Calcul N aux points de gauss
            N1t = 1-ksi-eta
            N2t = ksi
            N3t = eta
            Ntild = [N1t, N2t, N3t]

            N_u_pg = self.__ConstruitN_pg(Ntild)
            self.listN_u_pg.append(N_u_pg)

            N_d_pg = self.__ConstruitN_pg(Ntild, vecteur=False)            
            self.listN_d_pg.append(N_d_pg)

            # Calcul de B au points de gauss
            dN1t = np.array([-1, -1])
            dN2t = np.array([1, 0])
            dN3t = np.array([0, 1])
            dNtild = [dN1t, dN2t, dN3t]


            invF = np.linalg.inv(F)
            # invF = 1/jacobien * np.array([  [b, -beta],
            #                                 [-a, alpha]  ])

            B_u_pg = self.__ConstruitB_pg(dNtild, invF)
            self.listB_u_pg.append(B_u_pg)

            self.listB_u_n = [B_u_pg] * 3

            B_d_pg = self.__ConstruitB_pg(dNtild, invF, vecteur=False)
            self.listB_d_pg.append(B_d_pg)
        # TRI6  
        if self.nPe == 6:
            
            # Points de gauss
            ksis = [1/6, 2/3, 1/6]
            etas = [1/6, 1/6, 2/3]
            self.listPoid_pg = [1/6] * 3

            matriceCoef = np.array([[0, 0, 0, 0, 0, 1],
                                    [1, 0, 0, 1, 0, 1],
                                    [0, 1, 0, 0, 1, 1],
                                    [1/4, 0, 0, 1/2, 0, 1],
                                    [1/4, 1/4, 1/4, 1/2, 1/2, 1],
                                    [0, 1/4, 0, 0, 1/2, 1]])
            
            constX, constY = self.__CalculLesConstantes(matriceCoef)
            
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
            
            def ConstruitF(ksi, eta):
                F = np.array([[alpha*2*ksi+gamma*eta+delta, a*2*ksi+c*eta+d],
                              [beta*2*eta+gamma*ksi+epsilon, b*2*eta+c*ksi+e]])
                return F   

            # Pour chaque pg on calcul jacobien B et N
            for pg in range(len(ksis)):
                
                ksi = ksis[pg] 
                eta = etas[pg]
                
                Ntild = Construit_Ntild(ksi, eta)
                dNtild = Construit_dNtild(ksi, eta)                
                F = ConstruitF(ksi, eta)                
                invF = np.linalg.inv(F)

                jacobien = np.linalg.det(F)
                self.listJacobien_pg.append(jacobien)
                
                B_u_pg = self.__ConstruitB_pg(dNtild, invF)
                self.listB_u_pg.append(B_u_pg)

                B_d_pg = self.__ConstruitB_pg(dNtild, invF, vecteur=False)
                self.listB_d_pg.append(B_d_pg)

                N_u_pg = self.__ConstruitN_pg(Ntild)
                self.listN_u_pg.append(N_u_pg)

                N_d_pg = self.__ConstruitN_pg(Ntild, vecteur=False)                
                self.listN_d_pg.append(N_d_pg)
                

            
            # Pour chaque noeuds on calcul Be  
            ksis = [0, 1, 0, 1/2, 1/2, 0]
            etas = [0, 0, 1, 0, 1/2, 1/2]
            
            for p in range(len(ksis)):
                
                ksi = ksis[p]
                eta = etas[p]
                
                dNtild = Construit_dNtild(ksi, eta)               
                
                F = ConstruitF(ksi, eta)

                invF = np.linalg.inv(F)

                B_n = self.__ConstruitB_pg(dNtild, invF)
                self.listB_u_n.append(B_n)

    def __Construit_B_N_Quadrangle(self):
        """Construit la matrice Be d'un element quadrillatère
        """
        if self.nPe == 4:
            matriceCoef = np.array([[1, -1, -1, 1],
                                    [-1, 1, -1, 1],
                                    [1, 1, 1, 1],
                                    [-1, -1, 1, 1]])
            
            constX, constY = self.__CalculLesConstantes(matriceCoef)
            
            alpha = constX[0]
            beta = constX[1]
            gamma = constX[2]
            delta = constX[3]            

            a = constY[0]
            b = constY[1]
            c = constY[2]
            d = constY[3]

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
            
            def ConstruitF(ksi, eta):
                F = np.array([[alpha*eta+beta, a*eta+b],
                              [alpha*ksi+gamma, a*ksi+c]])
                return F                              
            
            # Pour chaque point d'integration on calcul Be
            UnSurRacine3 = 1/np.sqrt(3) 
            ksis = [-UnSurRacine3, UnSurRacine3, UnSurRacine3, -UnSurRacine3]
            etas = [-UnSurRacine3, -UnSurRacine3, UnSurRacine3, UnSurRacine3]
            self.listPoid_pg = [1] * 4
            
            for pg in range(len(ksis)):
                
                ksi = ksis[pg] 
                eta = etas[pg]
                
                Ntild = Construit_Ntild(ksi, eta)
                dNtild = Construit_dNtild(ksi, eta)                
                F = ConstruitF(ksi, eta)                    
                invF = np.linalg.inv(F)

                jacobien = np.linalg.det(F)
                self.listJacobien_pg.append(jacobien)

                B_pg = self.__ConstruitB_pg(dNtild, invF)
                self.listB_u_pg.append(B_pg)

                B_d_pg = self.__ConstruitB_pg(dNtild, invF, vecteur=False)
                self.listB_d_pg.append(B_d_pg)

                N_u_pg = self.__ConstruitN_pg(Ntild)
                self.listN_u_pg.append(N_u_pg)

                N_d_pg = self.__ConstruitN_pg(Ntild, vecteur=False)                
                self.listN_d_pg.append(N_d_pg)

                
                
            # Pour chaque noeuds on calcul Be    
            
            ksis = [-1, 1, 1, -1]
            etas = [-1, -1, 1, 1]
            
            for i in range(len(ksis)):
                
                ksi = ksis[i]
                eta = etas[i]
                
                dNtild = Construit_dNtild(ksi, eta)               
                
                F = ConstruitF(ksi, eta)

                invF = np.linalg.inv(F)
                                             
                B_n = self.__ConstruitB_pg(dNtild, invF)
                self.listB_u_n.append(B_n)
            
              
        elif self.nPe ==8:
            matriceCoef = np.array([[-1, -1, 1, 1, 1, -1, -1, 1],
                                    [-1, 1, 1, 1, -1, 1, -1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, -1, 1, 1, -1, -1, 1, 1],
                                    [0, 0, 0, 1, 0, 0, -1, 1],
                                    [0, 0, 1, 0, 0, 1, 0, 1],
                                    [0, 0, 0, 1, 0, 0, 1, 1],
                                    [0, 0, 1, 0, 0, -1, 0, 1]])
            
            constX, constY = self.__CalculLesConstantes(matriceCoef)
            
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
                
            
            def ConstruitF(ksi, eta):
                F = np.array([[2*a*ksi*eta+b*eta**2+2*c*ksi+e*eta+f, 2*a2*ksi*eta+b2*eta**2+2*c2*ksi+e2*eta+f2],
                              [a*ksi**2+2*b*eta*ksi+2*d*eta+e*ksi+g, a2*ksi**2+2*b2*eta*ksi+2*d2*eta+e2*ksi+g2]])
                return F
            
            # Pour chaque point d'integration on calcul Be            
            UnSurRacine3 = 1/np.sqrt(3) 
            ksis = [-UnSurRacine3, UnSurRacine3, UnSurRacine3, -UnSurRacine3]
            etas = [-UnSurRacine3, -UnSurRacine3, UnSurRacine3, UnSurRacine3]
            self.listPoid_pg = [1] * 4

            for pg in range(len(ksis)):
                
                ksi = ksis[pg] 
                eta = etas[pg]
                
                Ntild = Construit_Ntild(ksi, eta)
                dNtild = Construit_dNtild(ksi, eta)                
                F = ConstruitF(ksi, eta)                    
                invF = np.linalg.inv(F)
                
                jacobien = np.linalg.det(F)
                self.listJacobien_pg.append(jacobien)

                B_pg = self.__ConstruitB_pg(dNtild, invF)                
                self.listB_u_pg.append(B_pg)

                B_d_pg = self.__ConstruitB_pg(dNtild, invF, vecteur=False)
                self.listB_d_pg.append(B_d_pg)

                N_u_pg = self.__ConstruitN_pg(Ntild)
                self.listN_u_pg.append(N_u_pg)

                N_d_pg = self.__ConstruitN_pg(Ntild, vecteur=False)
                self.listN_d_pg.append(N_d_pg)
                
                
            # Pour chaque noeuds on calcul Be               
            ksis = [-1, 1, 1, -1, 0, 1, 0, -1]
            etas = [-1, -1, 1, 1,-1, 0, 1, 0]
            
            for i in range(len(ksis)):
                
                ksi = ksis[i]
                eta = etas[i]
                
                dNtild = Construit_dNtild(ksi, eta)               
                
                F = ConstruitF(ksi, eta)
                
                invF = np.linalg.inv(F)
   
                B_n = self.__ConstruitB_pg(dNtild, invF)
                self.listB_u_n.append(B_n)
    
    def __Construit_B_N_Tetraedre(self):
        if self.nPe == 4:
            matriceCoef = np.array([[0, 0, 0, 1],
                                    [1, 0, 0, 1],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 1]])        
           
            constX, constY, constZ = self.__CalculLesConstantes(matriceCoef)
            
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
            
            F = np.array([[a1, a2, a3],
                          [b1, b2, b3],
                          [c1, c2, c3]])
            
            invF = np.linalg.inv(F)

            jacobien = np.linalg.det(F)            
            self.listJacobien_pg.append(jacobien)
            
            # Points de gauss
            x = 1/4
            y = 1/4
            z = 1/4
            self.listPoid_pg = [1/6]

            # Construit Ntild
            N1t = 1-x-y-z
            N2t = x
            N3t = y
            N4t = z            
            Ntild = [N1t, N2t, N3t, N4t]

            # Construit dNtild
            dN1t = np.array([-1, -1, -1])
            dN2t = np.array([1, 0, 0])
            dN3t = np.array([0, 1, 0])
            dN4t = np.array([0, 0, 1])
            dNtild = [dN1t, dN2t, dN3t, dN4t]            
            
            B_pg = self.__ConstruitB_pg(dNtild, invF)
            self.listB_u_pg.append(B_pg)
            
            self.listB_u_n = [B_pg] * 4

            B_d_pg = self.__ConstruitB_pg(dNtild, invF, vecteur=False)
            self.listB_d_pg.append(B_d_pg)
            
            N_u_pg = self.__ConstruitN_pg(Ntild)
            self.listN_u_pg.append(N_u_pg)

            N_d_pg = self.__ConstruitN_pg(Ntild, vecteur=False)            
            self.listN_d_pg.append(N_d_pg)

    
    def __CalculLesConstantes(self, matriceCoef: np.ndarray):
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
        vectX = np.array([self.noeuds[i].coordo[0] for i in range(self.nPe)])
        vectY = np.array([self.noeuds[i].coordo[1] for i in range(self.nPe)])
        vectZ = np.array([self.noeuds[i].coordo[2] for i in range(self.nPe)])

        constX = np.linalg.solve(matriceCoef, vectX)
        constY = np.linalg.solve(matriceCoef, vectY)
        constZ = np.linalg.solve(matriceCoef, vectZ)

        if self.__dim == 2:
            return constX, constY
        elif self.__dim == 3:
            return constX, constY, constZ

    def __ConstruitB_pg(self, list_dNtild: list, invF: np.ndarray, vecteur=True):  
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
    
    def __ConstruitN_pg(self, list_Ntild: list, vecteur=True):
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