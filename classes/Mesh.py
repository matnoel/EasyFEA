import numpy as np

from Element import Element
from TicTac import TicTac

class Mesh:
    
    def __get_Ne(self):
        """Renvoie le nombre d'éléments du maillage        
        """
        return int(len(self.__connect))
    Ne = property(__get_Ne)
    
    def __get_Nn(self):
        """Renvoie le nombre d'éléments du maillage        
        """
        return int(len(self.__coordo))
    Nn = property(__get_Nn)

    def __get_dim(self):
        return self.__dim
    dim = property(__get_dim)

    def __get_coordo(self):
        return self.__coordo.copy()
    coordo = property(__get_coordo)

    def __get_connect(self):
        return self.__connect.copy()
    connect = property(__get_connect)

    def __get_assembly(self):
        return self.__assembly_e.copy()
    assembly_e = property(__get_assembly)        

    def __get_lignesVector_e(self):
        return self.__lignesVector_e.copy()
    lignesVector_e = property(__get_lignesVector_e)

    def __get_colonnesVector_e(self):
        return self.__colonnesVector_e.copy()
    colonnesVector_e = property(__get_colonnesVector_e)

    def __get_lignesScalar_e(self):
        return self.__lignesScalar_e.copy()
    lignesScalar_e = property(__get_lignesScalar_e)

    def __get_colonnesScalar_e(self):
        return self.__colonnesScalar_e.copy()
    colonnesScalar_e = property(__get_colonnesScalar_e)

    def __get_poid_pg(self):
        return self.__poid_pg.copy()
    poid_pg = property(__get_poid_pg)

    def __get_jacobien_e_pg(self):
        return self.__jacobien_e_pg.copy()
    jacobien_e_pg = property(__get_jacobien_e_pg)

    def __get_N_mass_pg(self):
        return self.__N_mass_pg.copy()
    N_mass_pg = property(__get_N_mass_pg)

    def __get_N_rigi_pg(self):
        return self.__N_rigi_pg.copy()
    N_rigi_pg = property(__get_N_rigi_pg)

    def __get_B_mass_e_pg(self):
        return self.__B_mass_e_pg.copy()
    B_mass_e_pg = property(__get_B_mass_e_pg)

    def __get_B_rigi_e_pg(self):
        return self.__B_rigi_e_pg.copy()
    B_rigi_e_pg = property(__get_B_rigi_e_pg)

    def get_nbFaces(self):
        if self.__dim == 2:
            return 1
        else:
            # TETRA4
            if self.__connect.shape[1] == 4:
                return 4

    def get_connectTriangle(self):
        """Transforme la matrice de connectivité pour la passer dans le trisurf en 2D
            ou construit les faces pour la 3D
            Par exemple pour un quadrangle on construit deux triangles
            pour un triangle à 6 noeuds on construit 4 triangles
            POur la 3D on construit des faces pour passer en Poly3DCollection
            """

        if len(self.__connectPourTriangle) == 0:
                     
            # npe = self.__connect.shape[1]
                
            # if self.__dim == 2:            
            #     # TRI3
            #     if npe == 3:
            #         self.__connectPourTriangle = self.__connect
            #     # TRI6
            #     elif npe == 6:
            #         n1 = self.__connect[:,0]
            #         n2 = self.__connect[:,1]
            #         n3 = self.__connect[:,2]
            #         n4 = self.__connect[:,3]
            #         n5 = self.__connect[:,4]
            #         n6 = self.__connect[:,5]

            #         self.__connectPourTriangle = np.array([[n1, n4, n6],[n4, n2, n5],[n6, n5, n3],[n4, n5, n6]]).T.reshape((-1,3))
            #     # QUAD4
            #     elif npe == 4:
            #         n1 = self.__connect[:,0]
            #         n2 = self.__connect[:,1]
            #         n3 = self.__connect[:,2]
            #         n4 = self.__connect[:,3]

            #         self.__connectPourTriangle = np.array([[n1, n2, n4],[n2, n3, n4]]).T.reshape((-1,3))
            #     # QUAD8
            #     elif npe == 8:
            #         n1 = self.__connect[:,0]
            #         n2 = self.__connect[:,1]
            #         n3 = self.__connect[:,2]
            #         n4 = self.__connect[:,3]
            #         n5 = self.__connect[:,4]
            #         n6 = self.__connect[:,5]
            #         n7 = self.__connect[:,6]
            #         n8 = self.__connect[:,7]

            #         self.__connectPourTriangle = np.array([[n5, n6, n8],[n6, n7, n8],[n1, n5, n8],[n5, n2, n6],[n6, n3, n7],[n7, n4, n8]]).T.reshape((-1,3))
                    
            # elif self.__dim ==3:
            #     pass

            for listIdNoeuds in self.__connect:
                npe = len(listIdNoeuds)
                
                if self.__dim == 2:            
                    # TRI3
                    if npe == 3:
                        self.__connectPourTriangle = self.__connect
                        break            
                    # TRI6
                    elif npe == 6:
                        n1 = listIdNoeuds[0]
                        n2 = listIdNoeuds[1]
                        n3 = listIdNoeuds[2]
                        n4 = listIdNoeuds[3]
                        n5 = listIdNoeuds[4]
                        n6 = listIdNoeuds[5]

                        self.__connectPourTriangle.append([n1, n4, n6])
                        self.__connectPourTriangle.append([n4, n2, n5])
                        self.__connectPourTriangle.append([n6, n5, n3])
                        self.__connectPourTriangle.append([n4, n5, n6])                    
                    # QUAD4
                    elif npe == 4:
                        n1 = listIdNoeuds[0]
                        n2 = listIdNoeuds[1]
                        n3 = listIdNoeuds[2]
                        n4 = listIdNoeuds[3]                

                        self.__connectPourTriangle.append([n1, n2, n4])
                        self.__connectPourTriangle.append([n2, n3, n4])                    
                    # QUAD8
                    elif npe == 8:
                        n1 = listIdNoeuds[0]
                        n2 = listIdNoeuds[1]
                        n3 = listIdNoeuds[2]
                        n4 = listIdNoeuds[3]
                        n5 = listIdNoeuds[4]
                        n6 = listIdNoeuds[5]
                        n7 = listIdNoeuds[6]
                        n8 = listIdNoeuds[7]

                        self.__connectPourTriangle.append([n5, n6, n8])
                        self.__connectPourTriangle.append([n6, n7, n8])
                        self.__connectPourTriangle.append([n1, n5, n8])
                        self.__connectPourTriangle.append([n5, n2, n6])
                        self.__connectPourTriangle.append([n6, n3, n7])
                        self.__connectPourTriangle.append([n7, n4, n8])                    
                    
                elif self.__dim ==3:
                    pass

        return self.__connectPourTriangle
    
    def get_connect_Faces(self):
        """Construit les faces pour chaque element

        Returns
        -------
        list de list
            Renvoie une liste de face
        """
        if len(self.__connectPolygon) == 0:

            npe = self.__connect.shape[1]

            if self.__dim == 2:
                # TRI3
                if npe == 3:
                    n1 = self.__connect[:,0]
                    n2 = self.__connect[:,1]
                    n3 = self.__connect[:,2]

                    self.__connectPolygon = np.array([n1, n2, n3, n1]).T
                # TRI6
                elif npe == 6:
                    n1 = self.__connect[:,0]
                    n2 = self.__connect[:,1]
                    n3 = self.__connect[:,2]
                    n4 = self.__connect[:,3]
                    n5 = self.__connect[:,4]
                    n6 = self.__connect[:,5]

                    self.__connectPolygon = np.array([n1, n4, n2, n5, n3, n6, n1]).T
                # QUAD4
                elif npe == 4:
                    # self.__connectPolygon = self.__connect
                    # break
                    n1 = self.__connect[:,0]
                    n2 = self.__connect[:,1]
                    n3 = self.__connect[:,2]
                    n4 = self.__connect[:,3]

                    self.__connectPolygon = np.array([n1, n2, n3, n4, n1]).T
                # QUAD8
                elif npe == 8:
                    n1 = self.__connect[:,0]
                    n2 = self.__connect[:,1]
                    n3 = self.__connect[:,2]
                    n4 = self.__connect[:,3]
                    n5 = self.__connect[:,4]
                    n6 = self.__connect[:,5]
                    n7 = self.__connect[:,6]
                    n8 = self.__connect[:,7]

                    self.__connectPolygon = np.array([n1, n5, n2, n6, n3, n7, n4, n8, n1]).T
            elif self.__dim == 3:
                # TETRA4
                if npe == 4:
                    n1 = self.__connect[:,0]
                    n2 = self.__connect[:,1]
                    n3 = self.__connect[:,2]
                    n4 = self.__connect[:,3]
                                    
                    self.__connectPolygon = np.array([[n1 ,n2, n3],[n1, n2, n4],[n1, n3, n4],[n2, n3, n4]]).T
     
        return self.__connectPolygon

    def __init__(self, dim: int, coordo, connect, verbosity=True):
        """Création du maillage depuis coordo et connection

        Parameters
        ----------
        coordo : list
            Coordonnées des noeuds dim(Nn,3), by default []
        connection : list
            Matrice de connection dim(Ne,nPe), by default []
        affichageMaillage : bool, optional
            Affichage après la construction du maillage, by default True
        """
    
        # Vérfication
        assert isinstance(coordo, np.ndarray) and isinstance(coordo[0], np.ndarray),"Doit fournir une liste de ndarray de ndarray !"
        
        assert isinstance(connect, np.ndarray) and isinstance(connect[0], np.ndarray),"Doit fournir une liste de liste"       

        self.__dim = dim

        self.__verbosity = verbosity

        self.__coordo = np.array(coordo)
        self.__connect = np.array(connect)
       
        self.__ConstruitMatricesPourCalculEf()        

        self.__connectPourTriangle = []
        self.__connectPolygon = []
        
        if verbosity:
            print("\nNe = {}, Nn = {}, nbDdl = {}".format(self.Ne,self.Nn,self.Nn*self.__dim)) 
    
    def __ConstruitMatricesPourCalculEf(self, verification = False):
        """Construit les matrices nécessaire au calcul des matrices elementaire
        
        F_e_pg : Matrice jacobienne
        
        invF_e_pg : Inverse matrice jacobienne
        
        jacobien_e_pg : Jacobien
        
        N_rigi_pg : Matrice des fonctions de forme dans element de référence (ksi, eta)
        exemple : [N1(ksi,eta) 0 N2(ksi,eta) 0 Nn(ksi,eta) 0
                        0 N1(ksi,eta) 0 N2(ksi,eta) 0Nn(ksi,eta)]        
        
        N_mass_pg : Matrice des fonctions de forme dans element de référence
        dN_e_pg : Derivé des fonctions de forme dans la base réele
        exemple : [dN1,x dN2,x dNn,x
                        dN1,y dN2,y dNn,y]
        
        B_mass_pg : dN_e_pg

        B_rigi_pg : [dN1,x 0 dN2,x 0 dNn,x 0
                    0 dN1,y 0 dN2,y 0 dNn,y
                    dN1,y dN1,x dN2,y dN2,x dN3,y dN3,x]
        """
        
        tic = TicTac()

        # Data
        dim = self.__dim
        connect = self.__connect
        coordo = self.__coordo
        listElement = range(self.Ne)        

        element = Element(dim, len(connect[0]))
        nPe = element.nPe;  listnPe = list(range(nPe))
        nPg = element.nPg;  listPg = list(range(nPg))
        gauss = element.gauss
        nodes_n = coordo[:,range(dim)]
        taille = nPe*dim

        # Construit la matrice d'assemblage
        self.__assembly_e = np.zeros((self.Ne, nPe*dim), dtype=np.int64)
        self.__assembly_e[:, list(range(0, taille, dim))] = np.array(self.connect) * dim
        self.__assembly_e[:, list(range(1, taille, dim))] = np.array(self.connect) * dim + 1            
        if dim == 3:            
            self.__assembly_e[:, list(range(2, taille, dim))] = np.array(self.connect) * dim + 2

        # Construit les lignes et colonnes ou il y aura des valeurs dans la matrice d'assemblage
        
        # lignes_e = np.array([[i]*taille for i in self.__assembly_e]).reshape(self.Ne,-1)
        self.__lignesVector_e = np.array([[[i]*taille for i in self.__assembly_e[e]] for e in listElement]).reshape(self.Ne,-1)
        self.__lignesScalar_e = np.array([[[i]*nPe for i in self.__connect[e]] for e in listElement]).reshape(self.Ne,-1)

        # colonnes_e = np.array([[i]*taille for i in self.__assembly_e]).reshape(self.Ne,-1)
        # colonnes_e = np.array([[self.__assembly_e]*taille]).reshape(self.Ne,-1)
        self.__colonnesVector_e = np.array([[[self.__assembly_e[e]]*taille] for e in listElement]).reshape(self.Ne,-1)
        self.__colonnesScalar_e = np.array([[[self.__connect[e]]*nPe] for e in listElement]).reshape(self.Ne,-1)

        # Poid
        self.__poid_pg = np.array(gauss[:,-1])

        # coordo localisé
        nodes_e = np.array(nodes_n[connect])

        dN_pg = np.array(element.dN_pg)

        # Matrice jacobienne
        self.__F_e_pg = np.array(np.einsum('pik,ekj->epij', dN_pg, nodes_e, optimize=False))

        # Inverse Matrice jacobienne
        self.__invF_e_pg = np.array(np.linalg.inv(self.__F_e_pg))

        # jacobien
        self.__jacobien_e_pg = np.array(np.linalg.det(self.__F_e_pg))

        # Fonctions de formes dans l'element isoparamétrique pour un scalaire ou un vecteur
        self.__N_rigi_pg = np.array(element.N_rigi_pg)
        self.__N_mass_pg = np.array(element.N_mass_pg).reshape(nPg, 1,-1)

        # Derivé des fonctions de formes dans la base réele
        dN_e_pg = np.array(np.einsum('epik,pkj->epij', self.__invF_e_pg, element.dN_pg, optimize=True))

        # Assemble les matrice Epsilons pour un scalaire
        self.__B_mass_e_pg = dN_e_pg

        # Assemble les matrice Epsilons pour un vecteur
        colonnes0 = np.arange(0, nPe*dim, dim)
        colonnes1 = np.arange(1, nPe*dim, dim)

        if self.__dim == 2:
            self.__B_rigi_e_pg = np.array([[np.zeros((3, nPe*dim))]*element.nPg]*self.Ne)
            
            dNdx = dN_e_pg[:,:,0,listnPe]
            dNdy = dN_e_pg[:,:,1,listnPe]

            self.__B_rigi_e_pg[:,:,0,colonnes0] = dNdx
            self.__B_rigi_e_pg[:,:,1,colonnes1] = dNdy
            self.__B_rigi_e_pg[:,:,2,colonnes0] = dNdy; self.__B_rigi_e_pg[:,:,2,colonnes1] = dNdx
        else:
            self.__B_rigi_e_pg = np.array([[np.zeros((6, nPe*dim))]*element.nPg]*self.Ne)

            dNdx = dN_e_pg[:,:,0,listnPe]
            dNdy = dN_e_pg[:,:,1,listnPe]
            dNdz = dN_e_pg[:,:,2,listnPe]

            colonnes2 = np.arange(2, nPe*dim, dim)

            self.__B_rigi_e_pg[:,:,0,colonnes0] = dNdx
            self.__B_rigi_e_pg[:,:,1,colonnes1] = dNdy
            self.__B_rigi_e_pg[:,:,2,colonnes2] = dNdz
            self.__B_rigi_e_pg[:,:,3,colonnes1] = dNdz; self.__B_rigi_e_pg[:,:,3,colonnes2] = dNdy
            self.__B_rigi_e_pg[:,:,4,colonnes0] = dNdz; self.__B_rigi_e_pg[:,:,4,colonnes2] = dNdx
            self.__B_rigi_e_pg[:,:,5,colonnes0] = dNdy; self.__B_rigi_e_pg[:,:,5,colonnes1] = dNdx

        if verification:

            # Verification assemblage
            assembly_e_test = np.array([[int(n * dim + d)for n in connect[e] for d in range(dim)] for e in listElement])
            testAssembly = self.__assembly_e - assembly_e_test
            assert testAssembly.mean() == 0, "Erreur dans la construction de la matrice d'assemblage"
            
            # Verification lignes_e 
            lignes_e_test = np.array([[i for i in self.__assembly_e[e] for j in self.__assembly_e[e]] for e in listElement])
            testLignes = lignes_e_test - self.__lignesVector_e
            assert testLignes.mean() == 0, "Erreur dans la constuction de lignes_e"

            # Verification lignes_e 
            colonnes_e_test = np.array([[j for i in self.__assembly_e[e] for j in self.__assembly_e[e]] for e in listElement])
            testColonnes = colonnes_e_test - self.colonnesVector_e
            assert testColonnes.mean() == 0, "Erreur dans la constuction de lignes_e"

            list_B_rigi_e_pg = []

            for e in listElement:
                list_B_rigi_pg = []
                for pg in listPg:
                    if dim == 2:
                        B_rigi_pg = np.zeros((3, nPe*dim))
                        colonne = 0
                        dN = self.__dN_e_pg[e,pg]
                        for n in range(nPe):
                            dNdx = dN[0, n]
                            dNdy = dN[1, n]
                            
                            # B rigi
                            B_rigi_pg[0, colonne] = dNdx
                            B_rigi_pg[1, colonne+1] = dNdy
                            B_rigi_pg[2, colonne] = dNdy; B_rigi_pg[2, colonne+1] = dNdx
                            
                            colonne += 2
                        list_B_rigi_pg.append(B_rigi_pg)    
                    else:
                        B_rigi_pg = np.zeros((6, nPe*dim))
                        
                        colonne = 0
                        for n in range(nPe):
                            dNdx = dN[0, n]
                            dNdy = dN[1, n]
                            dNdz = dN[2, n]                        
                            
                            B_rigi_pg[0, colonne] = dNdx
                            B_rigi_pg[1, colonne+1] = dNdy
                            B_rigi_pg[2, colonne+2] = dNdz
                            B_rigi_pg[3, colonne] = dNdy; B_rigi_pg[3, colonne+1] = dNdx
                            B_rigi_pg[4, colonne+1] = dNdz; B_rigi_pg[4, colonne+2] = dNdy
                            B_rigi_pg[5, colonne] = dNdz; B_rigi_pg[5, colonne+2] = dNdx
                            colonne += 3
                        list_B_rigi_pg.append(B_rigi_pg)
                        
                    
                list_B_rigi_e_pg.append(list_B_rigi_pg)
            
                test = np.array(list_B_rigi_e_pg)-self.__B_rigi_e_pg
                assert test.max() == 0 and test.min() == 0, "Erreur dans la construiction de B"
            
        tic.Tac("Mesh","Construit les matrices EF", self.__verbosity)

    def Get_Nodes(self, conditionX=True, conditionY=True, conditionZ=True):
        """Renvoie la liste de noeuds qui respectent la les condtions

        Args:
            conditionX (bool, optional): Conditions suivant x. Defaults to True.
            conditionY (bool, optional): Conditions suivant y. Defaults to True.
            conditionZ (bool, optional): Conditions suivant z. Defaults to True.

        Exemples de contitions:
            x ou toto ça n'a pas d'importance
            condition = lambda x: x < 40 and x > 20
            condition = lambda x: x == 40
            condition = lambda x: x >= 0

        Returns:
            list(int): lite des noeuds qui respectent les conditions
        """

        verifX = isinstance(conditionX, bool)
        verifY = isinstance(conditionY, bool)
        verifZ = isinstance(conditionZ, bool)

        listNoeud = list(range(self.Nn))
        if verifX and verifY and verifZ:
            return listNoeud

        coordoX = self.__coordo[:,0]
        coordoY = self.__coordo[:,1]
        coordoZ = self.__coordo[:,2]
        
        arrayVrai = np.array([True]*self.Nn)
        
        # Verification suivant X
        if verifX:
            valideConditionX = arrayVrai
        else:
            try:
                valideConditionX = conditionX(coordoX)
            except:
                valideConditionX = [conditionX(coordoX[n]) for n in listNoeud]

        # Verification suivant Y
        if verifY:
            valideConditionY = arrayVrai
        else:
            try:
                valideConditionY = conditionY(coordoY)
            except:
                valideConditionY = [conditionY(coordoY[n]) for n in listNoeud]
        
        # Verification suivant Z
        if verifZ:
            valideConditionZ = arrayVrai
        else:
            try:
                valideConditionZ = conditionZ(coordoZ)
            except:
                valideConditionZ = [conditionZ(coordoZ[n]) for n in listNoeud]
        
        conditionsTotal = valideConditionX * valideConditionY * valideConditionZ

        noeuds = list(np.where(conditionsTotal)[0])
        
        return noeuds

    def Localise_e(self, sol: np.ndarray):
        tailleVecteur = self.Nn * self.__dim
        tailleScalaire = self.Nn

        if sol.shape[0] == tailleVecteur:
            sol_e = sol[self.__assembly_e]
        else:
            sol_e = sol[self.__connect]
        
        return sol_e
        

            

# TEST ==============================

import unittest
import os

class Test_Mesh(unittest.TestCase):
    def setUp(self):
        
        coordo = []
        
        coordo.append(np.array([0, 0, 0]))
        coordo.append(np.array([1, 0, 0]))
        coordo.append(np.array([0, 1, 0]))
        
        connect = [[0, 1, 2]]
        
        self.mesh = Mesh(2, np.array(coordo), connect)

    def test_BienCree(self):
        self.assertIsInstance(self.mesh, Mesh)

if __name__ == '__main__':        
    try:
        os.system("cls")
        unittest.main(verbosity=2)
    except:
        print("")        