from typing import cast
import unittest
import os

class Test_GroupElem(unittest.TestCase):
    
    def setUp(self):
        self.elements = []
    
    def test_creation2D(self):
        from PythonEF.Interface_Gmsh import Interface_Gmsh
        from PythonEF.Mesh import Mesh

        list_mesh2D = Interface_Gmsh.Construction2D(L=1, h=1, taille=0.5)

        for mesh in list_mesh2D:
            
            mesh = cast(Mesh, mesh)
            
            mesh.assembly_e
            mesh.colonnesScalar_e
            mesh.colonnesVector_e
            mesh.colonnesScalar_e
            mesh.Get_N_scalaire_pg("rigi")
            mesh.Get_N_vecteur_pg("rigi")

if __name__ == '__main__' or __file__ == '__file__':
    try:
        os.system("cls")    #nettoie terminal
        unittest.main(verbosity=2)    
    except:
        print("")