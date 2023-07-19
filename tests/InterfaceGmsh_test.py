import unittest
import os

from Interface_Gmsh import Interface_Gmsh
import numpy as np

from Geom import *
import Display as Display
import matplotlib.pyplot as plt

class Test_InterfaceGmsh(unittest.TestCase):

    def test_Construction2D(self):
        list_mesh2D = Interface_Gmsh.Construction2D()
        nbMesh = len(list_mesh2D)
        nrows = 3
        ncols = 10
        # assert nbMesh == nrows*ncols , "Pas assez de place"
        fig, ax = plt.subplots(nrows, ncols)
        lignes = np.repeat(np.arange(nrows), ncols)
        colonnes = np.repeat(np.arange(ncols).reshape(1,-1), nrows, axis=0).reshape(-1)
        for m, mesh2D in enumerate(list_mesh2D):
            axx = ax[lignes[m],colonnes[m]]
            Display.Plot_Mesh(mesh2D, ax= axx)
            Display.Plot_Nodes(mesh2D, showId=False, ax=axx, c='black')
            axx.set_title("")
            axx.get_xaxis().set_visible(False)
            axx.get_yaxis().set_visible(False)
            plt.pause(1e-12)

            Display.Plot_Model(mesh2D)
            plt.pause(1e-12)
            plt.close()
        
        # plt.show()
    
    def test_Construction3D(self):
        list_mesh3D = Interface_Gmsh.Construction3D()
        for mesh3D in list_mesh3D:
            ax = Display.Plot_Mesh(mesh3D)
            Display.Plot_Nodes(mesh3D, showId=False, ax=ax, c='black')
            plt.pause(1e-12)
            plt.close()

            Display.Plot_Model(mesh3D)
            plt.pause(1e-12)
            plt.close()

        # plt.show()
        
if __name__ == '__main__':        
    try:
        import Display
        Display.Clear()
        unittest.main(verbosity=2)
    except:
        print("")