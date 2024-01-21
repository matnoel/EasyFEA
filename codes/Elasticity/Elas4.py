"""I Beam"""

import Folder
import Display
from Interface_Gmsh import Interface_Gmsh, ElemType, Point, PointsList, Circle, Domain, Line, Section
import Simulations
import Materials

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    Display.Clear()    
    # Define material properties
    E = 210000  # MPa (Young's modulus)
    v = 0.3     # Poisson's ratio
    coef = 1

    L = 500 # mm
    hx = 13
    hy = 13
    e = 2
    load = 800 # N

    # --------------------------------------------------------------------------------------------
    # Section
    # --------------------------------------------------------------------------------------------
    
    meshSize = e/2
    elemType = ElemType.HEXA8

    def DoSym(p: Point, n: np.ndarray) -> Point:
        pc = p.copy()
        pc.symmetry(n=n)
        return pc

    p1 = Point(-hx/2,-hy/2)
    p2 = Point(hx/2,-hy/2)
    p3 = Point(hx/2,-hy/2+e)
    p4 = Point(e/2,-hy/2+e, r=e)
    p5 = DoSym(p4,(0,1))
    p6 = DoSym(p3,(0,1))
    p7 = DoSym(p2,(0,1))
    p8 = DoSym(p1,(0,1))
    p9 = DoSym(p6,(1,0))
    p10 = DoSym(p5,(1,0))
    p11 = DoSym(p4,(1,0))
    p12 = DoSym(p3,(1,0))
    section = PointsList([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12], meshSize)    
    meshSection = Interface_Gmsh().Mesh_2D(section)
    
    section.Get_Contour().Plot()
    
    section.rotate(-np.pi/2, direction=(0,1))    

    # --------------------------------------------------------------------------------------------
    # Mesh
    # --------------------------------------------------------------------------------------------

    interf = Interface_Gmsh()
    fact = interf.factory
    surfaces = interf._Surfaces(section, [], elemType)[0]

    # layers = [20]
    layers = [2*L/meshSize/10]

    interf._Extrude(surfaces, [-L/2,0,0], elemType, layers)
    interf._Extrude(surfaces, [L/2,0,0], elemType, layers)
        
    interf._Set_PhysicalGroups()
    interf._Meshing(3, elemType)
    mesh = interf._Construct_Mesh()

    nodes_fixed = mesh.Nodes_Conditions(lambda x,y,z: (x==-L/2) | (x==L/2))
    nodes_load = mesh.Nodes_Line(Line(p7,p8))

    # --------------------------------------------------------------------------------------------
    # Simulation Beam
    # --------------------------------------------------------------------------------------------

    beam = Materials.Beam_Elas_Isot(2, Line(Point(-L/2), Point(L/2), L/10), Section(meshSection), E, v)

    mesh_beam = Interface_Gmsh().Mesh_Beams([beam], ElemType.SEG3)

    beams = Materials.Beam_Structure([beam])
    simu_beam = Simulations.Simu_Beam(mesh_beam, beams)

    simu_beam.add_dirichlet(mesh_beam.Nodes_Conditions(lambda x,y,z: (x==-L/2) | (x==L/2)),
                            [0]*simu_beam.Get_dof_n(), simu_beam.Get_directions())
    simu_beam.add_neumann(mesh_beam.Nodes_Conditions(lambda x,y,z: (x==0)), [-load], ['y'])
    simu_beam.Solve()

    Display.Plot_Result(simu_beam, 'uy')

    u_an = load * L**3 / (192*E*meshSection.Iy)

    uy_1d = np.abs(simu_beam.Result('uy').min())

    print(f"err beam model : {np.abs(u_an-uy_1d)/u_an*100:.2f} %")

    # --------------------------------------------------------------------------------------------
    # Simulation 3D
    # --------------------------------------------------------------------------------------------

    material = Materials.Elas_Isot(3, E, v)
    simu = Simulations.Simu_Displacement(mesh, material)

    simu.add_dirichlet(nodes_fixed, [0]*3, simu.Get_directions())
    simu.add_lineLoad(nodes_load, [-load/hx], ["y"])
    sol = simu.Solve()

    uy_3d = np.abs(simu.Result('uy').min())

    print(f"err 3d model : {np.abs(u_an-uy_3d)/u_an*100:.2f} %")

    # --------------------------------------------------------------------------------------------
    # Results
    # --------------------------------------------------------------------------------------------
    
    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Mesh(simu, hx/2/np.abs(sol).max())
    Display.Plot_Result(simu, "uy", coef=1/coef, nColors=20)
    Display.Plot_Result(simu, "ux", coef=1/coef, nColors=20, plotMesh=True)

    Simulations.Tic.Plot_History(details=False)

    # Interface_Gmsh(True).Save_Simu(simu)

    plt.show()
