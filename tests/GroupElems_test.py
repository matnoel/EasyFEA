import unittest
import os

class Test_GroupElem(unittest.TestCase):
    
    def setUp(self):
        self.elements = []
    
    def test_creation2D(self):
        from Gmsh_Interface import Mesher
        from Mesh import Mesh

        list_mesh2D = Mesher.Construct_2D_meshes(L=1, h=1, meshSize=0.5)

        for mesh in list_mesh2D:
            
            mesh: Mesh = mesh
            
            mesh.assembly_e
            mesh.columnsScalar_e
            mesh.columnsVector_e
            mesh.columnsScalar_e
            mesh.Get_N_pg("rigi")
            mesh.Get_N_vector_pg("rigi")

if __name__ == '__main__':
    unittest.main(verbosity=2)