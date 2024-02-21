import unittest
import os

from Gmsh_Interface import Mesher
import numpy as np

from Geoms import *
import Display as Display
import matplotlib.pyplot as plt

class Test_InterfaceGmsh(unittest.TestCase):

    def test_Construct_2D(self):
        """Builds all 2D meshes"""
        list_mesh2D = Mesher.Construct_2D_meshes()
        nbMesh = len(list_mesh2D)
        nrows = 3
        ncols = 10
        # assert nbMesh == nrows*ncols , "Not enough space"
        fig, ax = plt.subplots(nrows, ncols)
        lignes = np.repeat(np.arange(nrows), ncols)
        colonnes = np.repeat(np.arange(ncols).reshape(1,-1), nrows, axis=0).ravel()
        for m, mesh2D in enumerate(list_mesh2D):
            axx = ax[lignes[m],colonnes[m]]
            Display.Plot_Mesh(mesh2D, ax= axx)
            Display.Plot_Nodes(mesh2D, showId=False, ax=axx, c='black')
            axx.set_title("")
            axx.get_xaxis().set_visible(False)
            axx.get_yaxis().set_visible(False)
            plt.pause(1e-12)

            Display.Plot_Tags(mesh2D)
            plt.pause(1e-12)
            plt.close()
        
        # plt.show()
    
    def test_Construct_3D(self):
        """Builds all 3D meshes"""
        list_mesh3D = Mesher.Construct_3D_meshes()
        for mesh3D in list_mesh3D:
            ax = Display.Plot_Mesh(mesh3D)
            Display.Plot_Nodes(mesh3D, showId=False, ax=ax, c='black')
            plt.pause(1e-12)
            plt.close()

            Display.Plot_Tags(mesh3D)
            plt.pause(1e-12)
            plt.close()

        # plt.show()

if __name__ == '__main__':
    unittest.main(verbosity=2)