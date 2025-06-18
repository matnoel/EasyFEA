# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Elas3
=====

Hydraulic dam subjected to water pressure and its own weight.
"""

from EasyFEA import Display, plt, np, ElemType, Materials, Simulations
from EasyFEA.Geoms import Points

if __name__ == "__main__":

    Display.Clear()

    # Define dimension and mesh size parameters
    dim = 2
    N = 10

    coef = 1e6
    E = 15000 * coef  # Pa (Young's modulus)
    v = 0.25  # Poisson's ratio

    g = 9.81  # m/s^2 (acceleration due to gravity)
    ro = 2400  # kg/m^3 (density)
    w = 1000  # kg/m^3 (density)

    h = 180  # m (thickness)
    thickness = 2 * h

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    contour = Points([(0, 0), (h, 0), (0, h)], h / N)

    if dim == 2:
        mesh = contour.Mesh_2D([], ElemType.TRI6)
        print(f"err area = {np.abs(mesh.area - h**2/2)/mesh.area:.3e}")
    elif dim == 3:
        mesh = contour.Mesh_Extrude([], [0, 0, -thickness], [3], ElemType.PRISM15)
        print(
            f"error volume = {np.abs(mesh.volume - h**2/2 * thickness)/mesh.volume:.3e}"
        )

    nodes_x0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodes_y0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    material = Materials.ElasIsot(dim, E, v, planeStress=False, thickness=thickness)
    simu = Simulations.ElasticSimu(mesh, material)

    simu.add_dirichlet(nodes_y0, [0] * dim, simu.Get_unknowns())
    simu.add_surfLoad(
        nodes_x0, [lambda x, y, z: w * g * (h - y)], ["x"], description="[w*g*(h-y)]"
    )
    simu.add_volumeLoad(mesh.nodes, [-ro * g], ["y"], description="[-ro*g]")

    sol = simu.Solve()
    simu.Save_Iter()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    print(simu)

    Display.Plot_Mesh(simu, h / 10 / np.abs(sol.max()))
    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Result(simu, "Svm", nodeValues=True, coef=1 / coef, ncolors=20)

    plt.show()
