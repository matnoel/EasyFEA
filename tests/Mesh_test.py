import matplotlib.pyplot as plt
import unittest
import os
from Materials import _Displacement_Model, _Materiau
import numpy as np
from Mesh import Mesh
import Display

class Test_Mesh(unittest.TestCase):

    def test_Matrix(self):
        from Gmsh_Interface import Mesher
        list_Mesh = Mesher.Construct_2D_meshes()
        for mesh in list_Mesh:
            self.__Construct_B_e_pg(mesh)

    # Verifivation
    def __Construct_B_e_pg(self, mesh: Mesh):

        dim = mesh.dim
        connect = mesh.connect
        listElement = range(mesh.Ne)
        listPg = np.arange(mesh.Get_nPg("rigi"))
        nPe = connect.shape[1]

        # Verification assemblage
        assembly_e_test = np.array([[int(n * dim + d)for n in connect[e] for d in range(dim)] for e in listElement])
        testAssembly = np.testing.assert_array_almost_equal(mesh.assembly_e, assembly_e_test, verbose=False)
        self.assertIsNone(testAssembly)

        # Verification lignes_e 
        lignes_e_test = np.array([[i for i in mesh.assembly_e[e] for j in mesh.assembly_e[e]] for e in listElement])
        testLignes = np.testing.assert_array_almost_equal(lignes_e_test, mesh.linesVector_e, verbose=False)
        self.assertIsNone(testLignes)

        # Verification lignes_e 
        colonnes_e_test = np.array([[j for i in mesh.assembly_e[e] for j in mesh.assembly_e[e]] for e in listElement])
        testColonnes = np.testing.assert_array_almost_equal(colonnes_e_test, mesh.columnsVector_e, verbose=False)
        self.assertIsNone(testColonnes)

        list_B_rigi_e_pg = []

        for e in listElement:
            list_B_rigi_pg = []
            for pg in listPg:
                if dim == 2:
                    B_dep_pg = np.zeros((3, nPe*dim))
                    colonne = 0
                    B_sclaire_e_pg = mesh.Get_dN_e_pg("rigi")
                    dN = B_sclaire_e_pg[e,pg]
                    for n in range(nPe):
                        dNdx = dN[0, n]
                        dNdy = dN[1, n]
                        
                        # B rigi
                        B_dep_pg[0, colonne] = dNdx
                        B_dep_pg[1, colonne+1] = dNdy
                        B_dep_pg[2, colonne] = dNdy; B_dep_pg[2, colonne+1] = dNdx
                        
                        colonne += 2
                    list_B_rigi_pg.append(B_dep_pg)    
                else:
                    B_dep_pg = np.zeros((6, nPe*dim))
                    
                    colonne = 0
                    for n in range(nPe):
                        dNdx = dN[0, n]
                        dNdy = dN[1, n]
                        dNdz = dN[2, n]                        
                        
                        B_dep_pg[0, colonne] = dNdx
                        B_dep_pg[1, colonne+1] = dNdy
                        B_dep_pg[2, colonne+2] = dNdz
                        B_dep_pg[3, colonne] = dNdy; B_dep_pg[3, colonne+1] = dNdx
                        B_dep_pg[4, colonne+1] = dNdz; B_dep_pg[4, colonne+2] = dNdy
                        B_dep_pg[5, colonne] = dNdz; B_dep_pg[5, colonne+2] = dNdx
                        colonne += 3
                    list_B_rigi_pg.append(B_dep_pg)
                
            list_B_rigi_e_pg.append(list_B_rigi_pg)
        
        list_B_rigi_e_pg = _Displacement_Model.KelvinMandel_B_e_pg(dim, np.array(list_B_rigi_e_pg))

        B_rigi_e_pg = mesh.Get_B_e_pg("rigi")

        testB_rigi = np.testing.assert_array_almost_equal(np.array(list_B_rigi_e_pg), B_rigi_e_pg, verbose=False)
        self.assertIsNone(testB_rigi)


if __name__ == '__main__':        
    try:
        import Display
        Display.Clear()
        unittest.main(verbosity=2)
    except:
        print("")