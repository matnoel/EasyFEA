import unittest
import os

from Interface_Gmsh import Interface_Gmsh
import numpy as np

from Geom import *
import Affichage
import matplotlib.pyplot as plt

class Test_InterfaceGmsh(unittest.TestCase):

        def setUp(self):
                self.list_mesh2D = Interface_Gmsh.Construction2D()
                self.list_mesh3D = Interface_Gmsh.Construction3D()
        
        def test_Construction2D(self):
                nbMesh = len(self.list_mesh2D)
                nrows = 4
                ncols = 8
                assert nbMesh == nrows*ncols
                fig, ax = plt.subplots(nrows, ncols)
                lignes = np.repeat(np.arange(nrows), ncols)
                colonnes = np.repeat(np.arange(ncols).reshape(1,-1), nrows, axis=0).reshape(-1)
                for m, mesh2D in enumerate(self.list_mesh2D):
                        axx = ax[lignes[m],colonnes[m]]
                        Affichage.Plot_Maillage(mesh2D, ax= axx)
                        Affichage.Plot_NoeudsMaillage(mesh2D, showId=False, ax=axx, c='black')
                        axx.set_title("")
                        axx.get_xaxis().set_visible(False)
                        axx.get_yaxis().set_visible(False)
                        plt.pause(0.00005)
                
                plt.show()
        
        def test_Importation3D(self):
                for mesh3D in self.list_mesh3D:
                        Affichage.Plot_NoeudsMaillage(mesh3D, showId=True)
                        plt.pause(0.00005)
           
if __name__ == '__main__':        
    try:
        os.system("cls")
        unittest.main(verbosity=2)
    except:
        print("")