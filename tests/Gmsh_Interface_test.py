# Copyright (C) 2021-2024 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

import unittest

from EasyFEA import Mesher, Mesh, plt, np
from EasyFEA.Geoms import *
from EasyFEA import Display as Display

class Test_Gmsh_Interface(unittest.TestCase):

    def test_Plot_2D(self):
        """Builds all 2D meshes"""
        list_mesh2D = Mesher._Construct_2D_meshes()
        nbMesh = len(list_mesh2D)
        nrows = 3
        ncols = 10
        # assert nbMesh == nrows*ncols , "Not enough space"
        fig, axs = plt.subplots(nrows, ncols)
        axs: list[Display.Axes] = np.ravel(axs)
        
        for m, mesh2D in enumerate(list_mesh2D):
            ax = axs[m]
            ax.axis('off')
            Display.Plot_Mesh(mesh2D, ax= ax)
            Display.Plot_Nodes(mesh2D, showId=False, ax=ax, c='black')
            ax.set_title("")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.pause(1e-12)

            Display.Plot_Tags(mesh2D)
            plt.pause(1e-12)
            plt.close()
        
        # plt.show()
    
    def test_Plot_3D(self):
        """Builds all 3D meshes"""
        list_mesh3D = Mesher._Construct_3D_meshes(useImport3D=True)
        for mesh3D in list_mesh3D:
            ax = Display.Plot_Mesh(mesh3D)
            Display.Plot_Nodes(mesh3D, showId=False, ax=ax, c='black')
            plt.pause(1e-12)
            ax.axis('off')
            plt.close()

            Display.Plot_Tags(mesh3D)
            plt.pause(1e-12)
            plt.close()

        # plt.show()
            
    def test_MoveMesh(self):
        """Check that if you move the mesh the properties remain the same."""

        meshes = Mesher._Construct_2D_meshes()
        meshes.extend(Mesher._Construct_3D_meshes())

        def testSameMesh(mesh1: Mesh, mesh2: Mesh) -> None:

            testSame(mesh1.dim, mesh2.dim)
            testSame(mesh1.Nn, mesh2.Nn)
            testSame(mesh1.Ne, mesh2.Ne)

            testSame(mesh1.length, mesh2.length) # same length
            testSame(mesh1.area, mesh2.area) # same area
            if mesh1.dim == 2:
                self.assertIs(mesh1.volume, None)
                self.assertIs(mesh2.volume, None)
            else:                
                testSame(mesh1.volume, mesh2.volume) # same volume

        def testSame(val1, val2) -> None:
            """Test if the values are the same"""
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                test = np.linalg.norm(val1-val2)/np.linalg.norm(val2)
            else:
                test = np.abs((val1-val2)/val2)
            self.assertTrue(test <= 1e-12)

        for mesh in meshes:

            # Translate
            meshTranslate = mesh.copy()
            dec = np.random.rand(3)
            meshTranslate.Translate(*dec)            
            testSame(meshTranslate.center, mesh.center+dec) # same center
            testSameMesh(meshTranslate, mesh)

            # Rotate
            rot = np.random.rand() * 360
            meshRotate = mesh.copy()
            newCenter = Rotate_coord(mesh.center, rot, mesh.center, (1,0))            
            meshRotate.Rotate(rot, meshRotate.center, (1,0))
            testSame(meshRotate.center, newCenter)  # same center
            testSameMesh(meshRotate, mesh)
            axis = (1,3,-1)
            newCenter = Rotate_coord(newCenter, rot, newCenter, axis)
            meshRotate.Rotate(rot, meshRotate.center, axis)
            testSame(meshRotate.center, newCenter) # same center
            testSameMesh(meshRotate, mesh)


if __name__ == '__main__':
    unittest.main(verbosity=2)