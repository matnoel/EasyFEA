import unittest
from Materiau import LoiDeComportement, Materiau, Elas_Isot, PhaseFieldModel
from typing import cast
import numpy as np
import Affichage
from Mesh import Mesh
from Interface_Gmsh import Interface_Gmsh
from Simu import Simu
import Dossier
from GroupElem import GroupElem
from TicTac import TicTac
import matplotlib.pyplot as plt

class Test_Simu(unittest.TestCase):    
    
    def CreationDesSimusElastique2D(self):
        
        dim = 2

        # Paramètres géométrie
        L = 120;  #mm
        h = 120;    
        b = 13

        # Charge a appliquer
        P = -800 #N

        # Paramètres maillage
        taille = L/2

        comportement = Elas_Isot(dim, epaisseur=b)

        materiau = Materiau(comportement, verbosity=False)

        self.simulations2DElastique = []

        listMesh2D = Interface_Gmsh.Construction2D(L=L, h=h, taille=taille)

        # Pour chaque type d'element 2D
        for mesh in listMesh2D:

            mesh = cast(Mesh, mesh)
            
            simu = Simu(mesh, materiau, verbosity=False)

            simu.Assemblage_u()

            noeuds_en_0 = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == 0)
            noeuds_en_L = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == L)

            simu.add_dirichlet("displacement", noeuds_en_0, [0, 0], ["x","y"], description="Encastrement")
            # simu.add_lineLoad("displacement",noeuds_en_L, [-P/h], ["y"])
            simu.add_dirichlet("displacement",noeuds_en_L, [lambda x,y,z: 1], ['x'])
            simu.add_surfLoad("displacement",noeuds_en_L, [P/h/b], ["y"])

            self.simulations2DElastique.append(simu)

    def CreationDesSimusElastique3D(self):

        dir_path = Dossier.GetPath()
        fichier = Dossier.Join([dir_path, "models", "part.stp"])

        dim = 3

        # Paramètres géométrie
        L = 120  #mm
        h = 13    
        b = 13

        P = -800 #N

        # Paramètres maillage        
        taille = L/2

        comportement = Elas_Isot(dim)

        materiau = Materiau(comportement, verbosity=False)
        
        self.simulations3DElastique = []

        listMesh3D = Interface_Gmsh.Construction3D(L=L, h=h, b=b, taille=taille)

        # Pour chaque type d'element 2D
        for mesh in listMesh3D:

            mesh = cast(Mesh, mesh)
            
            simu = Simu(mesh, materiau, verbosity=False)

            simu.Assemblage_u()

            noeuds_en_0 = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == 0)
            noeuds_en_L = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == L)

            simu.add_dirichlet("displacement", noeuds_en_0, [0, 0], ["x","y"], description="Encastrement")
            # simu.add_lineLoad("displacement",noeuds_en_L, [-P/h], ["y"])
            simu.add_dirichlet("displacement",noeuds_en_L, [lambda x,y,z: 1], ['x'])
            simu.add_surfLoad("displacement",noeuds_en_L, [P/h/b], ["y"])

            self.simulations3DElastique.append(simu)
    
    def setUp(self):
        self.CreationDesSimusElastique2D()
        self.CreationDesSimusElastique3D()  

    def test_ResolutionDesSimulationsElastique2D(self):
        # Pour chaque type de maillage on simule
        ax=None        
        for simu in self.simulations2DElastique:
            simu = cast(Simu, simu)
            simu.Solve_u()
            fig, ax, cb = Affichage.Plot_Result(simu, "amplitude", affichageMaillage=True)
            plt.pause(0.00001)
            plt.close(fig)

    def test_ResolutionDesSimulationsElastique3D(self):
        # Pour chaque type de maillage on simule
        ax=None
        for simu in self.simulations3DElastique:
            simu = cast(Simu, simu)
            simu.Solve_u()
            
            fig, ax, cb = Affichage.Plot_Result(simu, "amplitude", affichageMaillage=True)
            plt.pause(0.00001)
            plt.close(fig)

    def test__ConstruitMatElem_Dep(self):
        for simu in self.simulations2DElastique:
            simu = cast(Simu, simu)
            Ke_e = simu.ConstruitMatElem_Dep()
            self.__VerificationConstructionKe(simu, Ke_e)

    # ------------------------------------------- Vérifications ------------------------------------------- 

    def __VerificationConstructionKe(self, simu: Simu, Ke_e, d=[]):
            """Ici on verifie quon obtient le meme resultat quavant verification vectorisation

            Parameters
            ----------
            Ke_e : nd.array par element
                Matrice calculé en vectorisant        
            d : ndarray
                Champ d'endommagement
            """

            tic = TicTac()

            matriceType = "rigi"

            # Data
            mesh = simu.mesh
            nPg = mesh.get_nPg(matriceType)
            listPg = list(range(nPg))
            Ne = mesh.Ne            
            materiau = simu.materiau
            C = materiau.comportement.get_C()

            listKe_e = []

            B_dep_e_pg = mesh.get_B_dep_e_pg(matriceType)            

            jacobien_e_pg = mesh.get_jacobien_e_pg(matriceType)
            poid_pg = mesh.get_poid_pg(matriceType)
            for e in range(Ne):            
                # Pour chaque poing de gauss on construit Ke
                Ke = 0
                for pg in listPg:
                    jacobien = jacobien_e_pg[e,pg]
                    poid = poid_pg[pg]
                    B_pg = B_dep_e_pg[e,pg]

                    K = jacobien * poid * B_pg.T.dot(C).dot(B_pg)

                    if len(d)==0:   # probleme standart
                        
                        Ke += K
                    else:   # probleme endomagement
                        
                        de = np.array([d[mesh.connect[e]]])
                        
                        # Bourdin
                        g = (1-mesh.N_mass_pg[pg].dot(de))**2
                        # g = (1-de)**2
                        
                        Ke += g * K
                # # print(Ke-listeKe[e.id])
                if mesh.dim == 2:
                    listKe_e.append(Ke)
                else:
                    listKe_e.append(Ke)                

            tic.Tac("Matrices","Calcul des matrices elementaires (boucle)", False)
            
            # Verification
            Ke_comparaison = np.array(listKe_e)*simu.materiau.comportement.epaisseur
            test = Ke_e - Ke_comparaison

            test = np.testing.assert_array_almost_equal(Ke_e, Ke_comparaison, verbose=False)

            self.assertIsNone(test)
            

if __name__ == '__main__':        
    try:
        Affichage.Clear()
        unittest.main(verbosity=2)    
    except:
        print("")   
