# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA import Mesher, ElemType, MatrixType, Materials, Simulations, np
from EasyFEA.materials._hyperelastic import HyperElastic
from EasyFEA.Geoms import Domain

class TestHyperElastic:

    # def test_D_e_pg(self):

    #     L = 120 #mm
    #     h = 13
    #     meshSize = h/3
        
    #     contour = Domain((0,0), (L, h), meshSize)

    #     mesh = Mesher().Mesh_2D(contour, [], ElemType.TRI3)

    #     PyVista.Plot_Mesh(mesh).show()
        
    #     pass

    def test_Epsilon_e_pg(self):

        L = 120 #mm
        h = 13
        meshSize = h/3
        
        contour = Domain((0,0), (L, h), meshSize)

        # Tests= 2d
        mesh2d = Mesher().Mesh_2D(contour, [], ElemType.TRI3)
        simu2d = Simulations.ElasticSimu(mesh2d, Materials.Elas_Isot(2))

        simu2d.add_dirichlet(mesh2d.Nodes_Conditions(lambda x,y,z: x==0), [0,0], simu2d.Get_dofs())
        simu2d.add_dirichlet(mesh2d.Nodes_Conditions(lambda x,y,z: x==L), [.1], ["x"])
        simu2d.Solve()

        Eps2d_e_pg = simu2d._Calc_Epsilon_e_pg(simu2d.displacement, MatrixType.rigi)
        test2d_e_pg = Eps2d_e_pg - HyperElastic.Compute_Epsilon_e_pg(mesh2d, simu2d.displacement, MatrixType.rigi)

        assert np.linalg.norm(test2d_e_pg) / np.linalg.norm(Eps2d_e_pg) < 1e-12

        # Tests= 2d
        mesh3d = Mesher().Mesh_Extrude(contour, [], [0,0,h], [h/meshSize], ElemType.TETRA4)
        simu3d = Simulations.ElasticSimu(mesh3d, Materials.Elas_Isot(3))

        simu3d.add_dirichlet(mesh3d.Nodes_Conditions(lambda x,y,z: x==0), [0,0,0], simu3d.Get_dofs())
        simu3d.add_dirichlet(mesh3d.Nodes_Conditions(lambda x,y,z: x==L), [.1], ["x"])
        simu3d.Solve()

        Eps3d_e_pg = simu3d._Calc_Epsilon_e_pg(simu3d.displacement, MatrixType.mass)
        test3d_e_pg = Eps3d_e_pg - HyperElastic.Compute_Epsilon_e_pg(mesh3d, simu3d.displacement, MatrixType.mass)

        assert np.linalg.norm(test3d_e_pg) / np.linalg.norm(Eps3d_e_pg) < 1e-12        
        
        pass