# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA import Display, Folder, Mesher, ElemType, np, Materials, Simulations, MeshIO, PyVista
from EasyFEA.Geoms import Domain

from EasyFEA.materials._hyperelastic import HyperElastic, Project_Kelvin
import EasyFEA.materials._hyperelastic_laws as laws

folder = Folder.Dir(__file__)

if __name__ == "__main__":

    Display.Clear()
    
    # mesh = MeshIO.Medit_to_EasyFEA(Folder.Join(folder, "tetra4.mesh"))
    # PyVista.Plot(mesh, show_grid=True).show()

    L=120
    h=13
    meshSize = h/3
    
    contour = Domain((0,0), (L,h), h/3)

    mesh = Mesher().Mesh_Extrude(contour, [], [0,0,h], [h/meshSize], ElemType.TETRA4)
    nodesX0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
    nodesXL = mesh.Nodes_Conditions(lambda x,y,z: x == L)

    matIsot = Materials.Elas_Isot(3)
    simuIsot = Simulations.ElasticSimu(mesh, matIsot)

    simuIsot.add_dirichlet(nodesX0, [0,0,0], simuIsot.Get_dofs())
    simuIsot.add_dirichlet(nodesXL, [L*1e-6], ["x"])
    u = simuIsot.Solve()
    We = simuIsot.Result("Wdef")

    matrixType = "rigi"

    K = matIsot.get_bulk()
    K1 = 500
    K2 = 403346.153846154

    # mat = laws.NeoHookean(3, K)
    # mat = laws.MooneyRivlin(3, K1, K2)
    mat = laws.SaintVenantKirchhoff(3, matIsot.get_lambda(), matIsot.get_mu())
    # mat = laws.CiarletGeymonat(3, K, K1, K2)

    W_e_pg = mat.Compute_W(mesh, u, matrixType)
    dW_e_pg = mat.Compute_dWde(mesh, u, matrixType)
    d2W_e_pg = mat.Compute_d2Wde(mesh, u, matrixType)

    weightedJacobian_e_pg = mesh.Get_weightedJacobian_e_pg(matrixType)
    Wh = (weightedJacobian_e_pg * W_e_pg).sum()

    error = np.abs(Wh - We)/Wh*100

    from EasyFEA.simulations._hyperelastic import HyperElasticSimu
    simuHyper = HyperElasticSimu(mesh, mat)

    
    simuHyper.add_dirichlet(nodesX0, [0,0,0], simuIsot.Get_dofs())
    simuHyper.add_dirichlet(nodesXL, [L*1e-6], ["y"])

    simuHyper.Solve(maxIter=20)

    sol = simuHyper.displacement.reshape(-1,3)
    PyVista.Plot(simuHyper, sol[:,0]).show()



    pass