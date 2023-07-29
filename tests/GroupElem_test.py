from typing import cast
import unittest
import os

class Test_GroupElem(unittest.TestCase):
    
    def setUp(self):
        self.elements = []
    
    def test_creation2D(self):
        from Interface_Gmsh import Interface_Gmsh
        from Mesh import Mesh

        list_mesh2D = Interface_Gmsh.Construction_2D(L=1, h=1, taille=0.5)

        for mesh in list_mesh2D:
            
            mesh = cast(Mesh, mesh)
            
            mesh.assembly_e
            mesh.columnsScalar_e
            mesh.columnsVector_e
            mesh.columnsScalar_e
            mesh.Get_N_pg("rigi")
            mesh.Get_N_vector_pg("rigi")

if __name__ == '__main__' or __file__ == '__file__':
    try:
        import Display
        Display.Clear()
        unittest.main(verbosity=2)    
    except:
        print("")