from typing import cast

from Geom import *
from Gauss import Gauss
from TicTac import TicTac
from matplotlib import pyplot as plt
import numpy as np

class GroupElem:

        def __init__(self, gmshId: int, elementTags: np.ndarray, nodeTags: np.ndarray, coordo: np.ndarray, verbosity=False):
                
                self.__gmshId = gmshId

                # Elements
                self.__elementTags = elementTags
                self.__connect = nodeTags.reshape(self.Ne, self.nPe)
                
                # Noeuds
                self.__nodes = np.unique(nodeTags)
                self.__coordo = cast(np.ndarray, coordo[self.__nodes])

                self.__TestImportation()

                self.__verbosity = verbosity

                # Dictionnaires pour chaque types de matrices
                if self.dim > 0:
                    self.__dict_dN_e_pg = {}
                    self.__dict_F_e_pg = {}                
                    self.__dict_invF_e_pg = {}                
                    self.__dict_jacobien_e_pg = {}                
        
        ################################################ METHODS ##################################################

        def __get_elemType(self):
            return GroupElem.Get_ElemInFos(self.__gmshId)[0]
        elemType = cast(str, property(__get_elemType))

        def __get_nPe(self):
            return GroupElem.Get_ElemInFos(self.__gmshId)[1]
        nPe = cast(int, property(__get_nPe))

        def __get_dim(self):
            return GroupElem.Get_ElemInFos(self.__gmshId)[2]
        dim = cast(int, property(__get_dim))

        def __get_Ne(self):
            return self.__elementTags.shape[0]
        Ne = cast(int, property(__get_Ne))

        def __get_nodes(self):
            return self.__nodes.copy()
        nodes = cast(np.ndarray, property(__get_nodes))

        def __get_Nn(self):
            return self.__nodes.shape[0]
        Nn = property(__get_Nn)

        def __get_connect(self):
            return self.__connect.copy()
        connect = cast(np.ndarray, property(__get_connect))
        """matrice de connection de l'element (Ne, nPe)"""

        def get_assembly(self, dim=None):
            nPe = self.nPe
            if dim == None:
                dim = self.dim
            taille = nPe*dim

            connect = self.connect
            assembly = np.zeros((self.Ne, taille), dtype=np.int64)

            for d in range(dim):
                assembly[:, np.arange(d, taille, dim)] = np.array(connect) * dim + d

            return assembly
        assembly_e = cast(np.ndarray, property(get_assembly))
        """matrice d'assemblage (Ne, nPe*dim)"""

        def __get_coordo(self):
            return self.__coordo.copy()
        coordo = cast(np.ndarray, property(__get_coordo))
        """matrice de coordonnées de l'element (Nn, 3)"""

        def __get_nbFaces(self):
            match self.dim:
                case (0,1):
                    return 0
                case 2:
                    return 1
                case 3:
                    match self.elemType:
                        case "TETRA4":
                            return 4
        nbFaces = cast(int, property(__get_nbFaces))

        def get_gauss(self, matriceType: str):
            return Gauss(self.elemType, matriceType)

        def get_N_pg(self, matriceType: str, isScalaire: bool):
            """Fonctions de formes dans la base de réference

            Args:
                matriceType (str): ["rigi","masse"]
                isScalaire (bool): type de matrice N\n

            Returns:
                np.ndarray: . Fonctions de formes vectorielles (pg, dim, nPe*dim), dans la base (ksi, eta ...)\n
                                [Ni 0 . . . Nn 0 \n
                                0 Ni . . . 0 Nn]

                            . Fonctions de formes scalaires (pg, 1, nPe), dans la base (ksi, eta ...)\n
                                [Ni . . . Nn]
            """
            if self.dim == 0: return

            N_pg = self.__get_N_pg(matriceType)

            if not isinstance(N_pg, np.ndarray): return

            if isScalaire:
                return N_pg
            else:
                dim = self.dim
                taille = N_pg.shape[2]*dim
                N_vect_pg = np.zeros((N_pg.shape[0] ,dim , taille))

                for d in range(dim):
                    N_vect_pg[:, d, np.arange(d, taille, dim)] = N_pg[:,0,:]
                
                return N_vect_pg
        
        def get_dN_e_pg(self, matriceType: str):
            assert matriceType in GroupElem.get_MatriceType()

            if matriceType not in self.__dict_dN_e_pg.keys():

                invF_e_pg = self.get_invF_e_pg(matriceType)

                dN_pg = self.get_dN_pg(matriceType)

                # Derivé des fonctions de formes dans la base réele
                dN_e_pg = np.array(np.einsum('epik,pkj->epij', invF_e_pg, dN_pg, optimize=True))
                self.__dict_dN_e_pg[matriceType] = dN_e_pg

            return self.__dict_dN_e_pg[matriceType]
        
        def get_F_e_pg(self, matriceType: str):
            """Renvoie la matrice jacobienne
            """
            if self.dim == 0: return
            if matriceType not in self.__dict_F_e_pg.keys():

                nodes_n = self.coordo[:, range(self.dim)]
                nodes_e = nodes_n[self.connect]

                dN_pg = self.get_dN_pg(matriceType)

                F_e_pg = np.array(np.einsum('pik,ekj->epij', dN_pg, nodes_e, optimize=True))                        
                
                self.__dict_F_e_pg[matriceType] = F_e_pg

            return cast(np.ndarray, self.__dict_F_e_pg[matriceType])
        
        def get_jacobien_e_pg(self, matriceType:str):
            """Renvoie les jacobiens
            """
            if self.dim == 0: return
            if matriceType not in self.__dict_jacobien_e_pg.keys():

                F_e_pg = self.get_F_e_pg(matriceType)

                jacbobien_e_pg = np.array(np.linalg.det(F_e_pg))

                self.__dict_jacobien_e_pg[matriceType] = jacbobien_e_pg

            return cast(np.ndarray, self.__dict_jacobien_e_pg[matriceType])
        
        def get_invF_e_pg(self, matriceType: str):
            """Renvoie l'inverse de la matrice jacobienne
            """
            if self.dim == 0: return
            if matriceType not in self.__dict_invF_e_pg.keys():

                F_e_pg = self.get_F_e_pg(matriceType)

                match self.dim:
                    case 1:
                        invF_e_pg = 1/F_e_pg
                    case (2|3):
                        invF_e_pg = np.array(np.linalg.inv(F_e_pg))

                self.__dict_invF_e_pg[matriceType] = invF_e_pg

            return cast(np.ndarray, self.__dict_invF_e_pg[matriceType])

        def __get_N_pg(self, matriceType: str):
            """Fonctions de formes vectorielles (pg), dans la base (ksi, eta ...)\n
            [N1, N2, . . . ,Nn]
            """
            if self.dim == 0: return

            match self.elemType:

                case "SEG2":

                    N1t = lambda x: 0.5*(1-x)
                    N2t = lambda x: 0.5*(1+x)

                    Ntild = np.array([N1t, N2t])
                
                case "SEG3":

                    N1t = lambda x: -0.5*(1-x)*x
                    N2t = lambda x: 0.5*(1+x)*x
                    N3t = lambda x: (1+x)*(1-x)

                    Ntild = np.array([N1t, N2t])

                case "TRI3":

                    N1t = lambda ksi,eta: 1-ksi-eta
                    N2t = lambda ksi,eta: ksi
                    N3t = lambda ksi,eta: eta
                    
                    Ntild = np.array([N1t, N2t, N3t])

                case "TRI6":

                    N1t = lambda ksi,eta: -(1-ksi-eta)*(1-2*(1-ksi-eta))
                    N2t = lambda ksi,eta: -ksi*(1-2*ksi)
                    N3t = lambda ksi,eta: -eta*(1-2*eta)
                    N4t = lambda ksi,eta: 4*ksi*(1-ksi-eta)
                    N5t = lambda ksi,eta: 4*ksi*eta
                    N6t = lambda ksi,eta: 4*eta*(1-ksi-eta)
                    
                    Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t])
                
                case "QUAD4":

                    N1t = lambda ksi,eta: (1-ksi)*(1-eta)/4
                    N2t = lambda ksi,eta: (1+ksi)*(1-eta)/4
                    N3t = lambda ksi,eta: (1+ksi)*(1+eta)/4
                    N4t = lambda ksi,eta: (1-ksi)*(1+eta)/4
                    
                    Ntild = np.array([N1t, N2t, N3t, N4t])

                case "QUAD8":

                    N1t = lambda ksi,eta: (1-ksi)*(1-eta)*(-1-ksi-eta)/4
                    N2t = lambda ksi,eta: (1+ksi)*(1-eta)*(-1+ksi-eta)/4
                    N3t = lambda ksi,eta: (1+ksi)*(1+eta)*(-1+ksi+eta)/4
                    N4t = lambda ksi,eta: (1-ksi)*(1+eta)*(-1-ksi+eta)/4
                    N5t = lambda ksi,eta: (1-ksi**2)*(1-eta)/2
                    N6t = lambda ksi,eta: (1+ksi)*(1-eta**2)/2
                    N7t = lambda ksi,eta: (1-ksi**2)*(1+eta)/2
                    N8t = lambda ksi,eta: (1-ksi)*(1-eta**2)/2
                    
                    Ntild =  np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t])                    

                case "TETRA4":

                    N1t = lambda x,y,z: 1-x-y-z
                    N2t = lambda x,y,z: x
                    N3t = lambda x,y,z: y
                    N4t = lambda x,y,z: z

                    Ntild = np.array([N1t, N2t, N3t, N4t])
                
                case _: 
                    # print("Type inconnue")
                    # raise "Type inconnue"
                    return
            
            # Evalue aux points de gauss

            gauss = self.get_gauss(matriceType)            
            coord = gauss.coord
            nPg = gauss.nPg

            N_pg = np.zeros((nPg, 1, len(Ntild)))

            for pg in range(nPg):
                for n, Nt in enumerate(Ntild):
                    match coord.shape[1]:
                        case 1:
                            N_pg[pg, 0, n] = Nt(coord[pg,0])
                        case 2:
                            N_pg[pg, 0, n] = Nt(coord[pg,0], coord[pg,1])
                        case 3:
                            N_pg[pg, 0, n] = Nt(coord[pg,0], coord[pg,1], coord[pg,2])

            return N_pg
        
        def get_dN_pg(self, matriceType: str):
            """Dérivées des fonctions de formes dans l'element de référence (pg, dim, nPe), dans la base (ksi, eta ...) \n
            [Ni,ksi . . . Nn,ksi\n
            Ni,eta . . . Nn,eta]
            """
            if self.dim == 0: return

            match self.elemType:

                case "SEG2":

                    dN1t = [lambda x: -0.5]
                    dN2t = [lambda x: 0.5]

                    dNtild = np.array([dN1t, dN2t])
                
                case "SEG3":

                    dN1t = [lambda x: x-0.5]
                    dN2t = [lambda x: x+0.5]
                    dN3t = [lambda x: -2*x]

                    dNtild = np.array([dN1t, dN2t, dN3t])

                case "TRI3":

                    dN1t = [lambda ksi,eta: -1, lambda ksi,eta: -1]
                    dN2t = [lambda ksi,eta: 1,  lambda ksi,eta: 0]
                    dN3t = [lambda ksi,eta: 0,  lambda ksi,eta: 1]

                    dNtild = np.array([dN1t, dN2t, dN3t])

                case "TRI6":

                    dN1t = [lambda ksi,eta: 4*ksi+4*eta-3,  lambda ksi,eta: 4*ksi+4*eta-3]
                    dN2t = [lambda ksi,eta: 4*ksi-1,        lambda ksi,eta: 0]
                    dN3t = [lambda ksi,eta: 0,              lambda ksi,eta: 4*eta-1]
                    dN4t = [lambda ksi,eta: 4-8*ksi-4*eta,  lambda ksi,eta: -4*ksi]
                    dN5t = [lambda ksi,eta: 4*eta,          lambda ksi,eta: 4*ksi]
                    dN6t = [lambda ksi,eta: -4*eta,         lambda ksi,eta: 4-4*ksi-8*eta]
                    
                    dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t])
                
                case "QUAD4":
                    
                    dN1t = [lambda ksi,eta: (eta-1)/4,  lambda ksi,eta: (ksi-1)/4]
                    dN2t = [lambda ksi,eta: (1-eta)/4,  lambda ksi,eta: (-ksi-1)/4]
                    dN3t = [lambda ksi,eta: (1+eta)/4,  lambda ksi,eta: (1+ksi)/4]
                    dN4t = [lambda ksi,eta: (-eta-1)/4, lambda ksi,eta: (1-ksi)/4]
                    
                    dNtild = [dN1t, dN2t, dN3t, dN4t]

                case "QUAD8":
                   
                    dN1t = [lambda ksi,eta: (1-eta)*(2*ksi+eta)/4,      lambda ksi,eta: (1-ksi)*(ksi+2*eta)/4]
                    dN2t = [lambda ksi,eta: (1-eta)*(2*ksi-eta)/4,      lambda ksi,eta: -(1+ksi)*(ksi-2*eta)/4]
                    dN3t = [lambda ksi,eta: (1+eta)*(2*ksi+eta)/4,      lambda ksi,eta: (1+ksi)*(ksi+2*eta)/4]
                    dN4t = [lambda ksi,eta: -(1+eta)*(-2*ksi+eta)/4,    lambda ksi,eta: (1-ksi)*(-ksi+2*eta)/4]
                    dN5t = [lambda ksi,eta: -ksi*(1-eta),               lambda ksi,eta: -(1-ksi**2)/2]
                    dN6t = [lambda ksi,eta: (1-eta**2)/2,               lambda ksi,eta: -eta*(1+ksi)]
                    dN7t = [lambda ksi,eta: -ksi*(1+eta),               lambda ksi,eta: (1-ksi**2)/2]
                    dN8t = [lambda ksi,eta: -(1-eta**2)/2,              lambda ksi,eta: -eta*(1-ksi)]
                                    
                    dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t])

                case "TETRA4":
                    
                    dN1t = [lambda x,y,z: -1,   lambda x,y,z: -1,   lambda x,y,z: -1]
                    dN2t = [lambda x,y,z: 1,    lambda x,y,z: 0,    lambda x,y,z: 0]
                    dN3t = [lambda x,y,z: 0,    lambda x,y,z: 1,    lambda x,y,z: 0]
                    dN4t = [lambda x,y,z: 0,    lambda x,y,z: 0,    lambda x,y,z: 1]

                    dNtild = np.array([dN1t, dN2t, dN3t, dN4t])
                
                case _: 
                    # print("Type inconnue")
                    # raise "Type inconnue"
                    return
            
            # Evaluation aux points de gauss
            gauss = self.get_gauss(matriceType)
            coord = gauss.coord

            dim = self.dim
            nPg = gauss.nPg

            dN_pg = np.zeros((nPg, dim, len(dNtild)))

            for pg in range(nPg):
                for n, Nt in enumerate(dNtild):
                    for d in range(dim):
                        func = Nt[d]
                        match coord.shape[1]:
                            case 1:
                                dN_pg[pg, d, n] = func(coord[pg,0])
                            case 2:
                                dN_pg[pg, d, n] = func(coord[pg,0], coord[pg,1])
                            case 3:
                                dN_pg[pg, d, n] = func(coord[pg,0], coord[pg,1], coord[pg,2])

            return dN_pg        

        def Get_Nodes_Conditions(self, conditionX=True, conditionY=True, conditionZ=True):
            """Renvoie la liste de noeuds qui respectent les condtions

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

            noeuds = np.where(conditionsTotal)[0]
            
            return noeuds
        
        def Get_Nodes_Line(self, line: Line):
            
            vectUnitaire = line.vecteurUnitaire

            coordo = self.__coordo

            vect = coordo-line.coordo[0]

            prodScalaire = np.einsum('i,ni-> n', vectUnitaire, vect, optimize=True)
            prodVecteur = np.cross(vect, vectUnitaire)
            norm = np.linalg.norm(prodVecteur, axis=1)

            eps = np.finfo(float).eps

            noeuds = np.where((norm<eps) & (prodScalaire>=-eps) & (prodScalaire<=line.length+eps))            

            return noeuds
        
        def Get_Nodes_Domain(self, domain: Domain):
            """Renvoie la liste de noeuds qui sont dans le domaine"""

            coordo = self.__coordo

            eps = np.finfo(float).eps

            noeuds = np.where(  (coordo[:,0] >= domain.pt1.x-eps) & (coordo[:,0] <= domain.pt2.x+eps) &
                                (coordo[:,1] >= domain.pt1.y-eps) & (coordo[:,1] <= domain.pt2.y+eps) &
                                (coordo[:,2] >= domain.pt1.z-eps) & (coordo[:,2] <= domain.pt2.z+eps))
            
            return noeuds
        
        def Localise_sol_e(self, sol: np.ndarray):
            """localise les valeurs de noeuds sur les elements"""
            tailleVecteur = self.Nn * self.dim

            if sol.shape[0] == tailleVecteur:
                sol_e = sol[self.assembly_e]
            else:
                sol_e = sol[self.__connect]
            
            return sol_e

        def get_connectTriangle(self):
            """Transforme la matrice de connectivité pour la passer dans le trisurf en 2D\n
            Par exemple pour un quadrangle on construit deux triangles
            pour un triangle à 6 noeuds on construit 4 triangles
            """
            assert self.dim == 2
            match self.elemType:
                case "TRI3":
                    return self.__connect[:,[0,1,2]]
                case "TRI6":
                    return np.array(self.__connect[:, [0,3,5,3,1,4,5,4,2,3,4,5]]).reshape(-1,3)
                case "QUAD4":
                    return np.array(self.__connect[:, [0,1,3,1,2,3]]).reshape(-1,3)
                case "QUAD8":
                    return np.array(self.__connect[:, [4,5,7,5,6,7,0,4,7,4,1,5,5,2,6,6,3,7]]).reshape(-1,3)

        def get_connect_Faces(self):
            """Récupère les identifiants des noeud constuisant les faces

            Returns
            -------
            list de list
                Renvoie une liste de face
            """
            assert self.dim in [2,3]
            nPe = self.nPe
            match self.elemType:
                case "TRI3":
                    return self.__connect[:, [0,1,2,0]]
                case "TRI6":
                    return self.__connect[:, [0,3,1,4,2,5,0]]
                case "QUAD4":
                    return self.__connect[:, [0,1,2,3,0]]
                case "QUAD8":
                    return self.__connect[:, [0,4,1,5,2,6,3,7,0]]
                case "TETRA4":
                    # Ici par elexemple on va creer 3 faces, chaque face est composé des identifiants des noeuds
                    return np.array(self.__connect[:, [0,1,2,0,1,3,0,2,3,1,2,3]]).reshape(self.Ne*nPe,-1)

        def __TestImportation(self):
            """Test si il n'existe pas un noeud en trop
            """

            Nmax = self.__nodes.max()
            ecart = Nmax - (self.Nn-1)
            
            if ecart != 0:
                # Si l'écart et différent de 0 alors il ya un noeud qui à été dédoublé
                # Il faut alors creer un nouveau noeud dans coordo
                # Pour connaitre les coordo du nouveau noeud on va utiliser les segments

                coordo = self.__coordo
                coordo = np.append(coordo, [0,0.0005,0]).reshape(-1,3)

                if self.dim == 2:
                    fig, ax = plt.subplots()

                    ax.scatter(coordo[:,0], coordo[:,1], marker='.')

                    connectFaces = self.connect[:,[0,1,2,0]]

                    for e in range(self.Ne):
                            co = coordo[connectFaces[e]]
                            ax.plot(co[:,0], co[:,1])
                            plt.pause(0.5)
                    pass

        ################################################ STATIC ##################################################

        @staticmethod
        def get_MatriceType():
            liste = ["rigi", "masse"]
            return liste

        @staticmethod
        def get_Types2D():
            """type d'elements disponibles en 2D"""
            liste2D = ["TRI3", "TRI6", "QUAD4", "QUAD8"]
            return liste2D
        
        @staticmethod
        def get_Types3D():
            """type d'elements disponibles en 3D"""
            liste3D = ["TETRA4"]
            return liste3D

        @staticmethod
        def Get_ElemInFos(gmshId: int):
                """Renvoie le nom le nombre de noeuds par element et la dimension de l'élement en fonction du type

                Args:
                    type (int): type de l'identifiant sur gmsh

                Returns:
                    tuple: (type, nPe, dim)
                """

                match gmshId:
                        case 1: 
                                type = "SEG2"; nPe = 2; dim = 1
                        case 2: 
                                type = "TRI3"; nPe = 3; dim = 2
                        case 3: 
                                type = "QUAD4"; nPe = 4; dim = 2 
                        case 4: 
                                type = "TETRA4"; nPe = 4; dim = 3
                        case 5: 
                                type = "CUBE8"; nPe = 8; dim = 3
                        case 6: 
                                type = "PRISM6"; nPe = 6; dim = 3
                        case 7: 
                                type = "PYRA5"; nPe = 5; dim = 3
                        case 8: 
                                type = "SEG3"; nPe = 3; dim = 1
                        case 9: 
                                type = "TRI6"; nPe = 6; dim = 2
                        case 10: 
                                type = "QUAD9"; nPe = 9; dim = 2
                        case 11: 
                                type = "TETRA10"; nPe = 10; dim = 3
                        case 12: 
                                type = "CUBE27"; nPe = 27; dim = 3
                        case 13: 
                                type = "PRISM18"; nPe = 18; dim = 3
                        case 14: 
                                type = "PYRA14"; nPe = 17; dim = 3
                        case 15: 
                                type = "POINT"; nPe = 1; dim = 0
                        case 16: 
                                type = "QUAD8"; nPe = 8; dim = 2
                        case 18: 
                                type = "PRISM15"; nPe = 15; dim = 3
                        case 19: 
                                type = "PYRA13"; nPe = 13; dim = 3
                        case _: 
                                raise "Type inconnue"
                return type, nPe, dim
        
        
# ====================================

import unittest
import os

class Test_GroupElem(unittest.TestCase):
    
    def setUp(self):
        self.elements = []
    
    def test_creation2D(self):
        from Interface_Gmsh import Interface_Gmsh
        from Mesh import Mesh

        list_mesh2D = Interface_Gmsh.Construction2D(L=1, h=1, taille=0.5)

        for mesh in list_mesh2D:

            mesh = cast(Mesh, mesh)
            
            mesh.assembly_e
            mesh.colonnesScalar_e
            mesh.colonnesVector_e
            mesh.colonnesScalar_e
            mesh.get_N_scalaire_pg("rigi")
            mesh.get_N_vecteur_pg("rigi")

if __name__ == '__main__':        
    try:
        os.system("cls")    #nettoie terminal
        unittest.main(verbosity=2)    
    except:
        print("")