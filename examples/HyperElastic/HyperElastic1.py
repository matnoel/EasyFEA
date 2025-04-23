# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA import Display, Folder, Mesher, ElemType, np, Materials, Simulations, PyVista
from EasyFEA.Geoms import Domain

folder = Folder.Dir(__file__)

if __name__ == "__main__":

    Display.Clear()

    L=120
    h=13
    meshSize = h/3
    
    contour = Domain((0,0), (L,h), h/3)

    mesh = Mesher().Mesh_Extrude(contour, [], [0,0,h], [h/meshSize], ElemType.TETRA4, isOrganised=True)
    nodesX0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
    nodesXL = mesh.Nodes_Conditions(lambda x,y,z: x == L)

    matIsot = Materials.Elas_Isot(3)    
    
    K = matIsot.get_bulk()
    K1 = 500
    K2 = 403346.153846154

    # mat = Materials.NeoHookean(3, K)
    # mat = Materials.MooneyRivlin(3, K1, K2)
    mat = Materials.SaintVenantKirchhoff(3, matIsot.get_lambda(), matIsot.get_mu())
    
    simuHyper = Simulations.HyperElasticSimu(mesh, mat, useIterativeSolvers=False)
    
    simuHyper.add_dirichlet(nodesX0, [0,0,0], simuHyper.Get_unknowns())
    simuHyper.add_dirichlet(nodesXL, [-L*.5], ["y"])

    simuHyper.Solve()

    sol = simuHyper.displacement.reshape(-1,3)
    PyVista.Plot(simuHyper, "uy", 1, show_edges=True).show()

    pass