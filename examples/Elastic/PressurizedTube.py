"""A cylindrical conduit exposed to uniform pressure."""

from EasyFEA import (Display, np,
                     Mesher, ElemType,
                     Materials, Simulations,
                     PyVista_Interface as pvi)
from EasyFEA.Geoms import Point, Line, Circle, CircleArc, Contour

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    dim = 2
    isSymmetric = True
    openCrack = True

    r = 10
    e = 5

    sig = 5 # bar
    sig *= 1e-1 # 1 bar = 0.1 MPa

    meshSize = e/5
    thickness = 100

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    center = Point()
    if isSymmetric:
        p1 = Point(r,0)
        p2 = Point(e+r,0)
        p3 = Point(0,e+r)
        p4 = Point(0,r)

        line1 = Line(p1, p2, meshSize)
        line2 = CircleArc(p2, p3, center, meshSize=meshSize)
        line3 = Line(p3, p4, meshSize)
        line4 = CircleArc(p4, p1, center, meshSize=meshSize)

        contour = Contour([line1, line2, line3, line4])
        inclusions = []
    else:    
        contour = Circle(center, (r+e)*2, meshSize)
        inclusions = [Circle(center, 2*r, meshSize, isHollow=True)]

    extrude = [0,0,-thickness]

    l = e/4
    p = r+e/2
    alpha = np.pi/3
    pc1 = Point((p-l/2)*np.cos(alpha), (p-l/2)*np.sin(alpha))
    pc2 = Point((p+l/2)*np.cos(alpha), (p+l/2)*np.sin(alpha))    

    if dim == 2:
        crack = Line(pc1, pc2, meshSize/6, isOpen=openCrack)
        mesh = Mesher().Mesh_2D(contour, inclusions, elemType=ElemType.TRI6, cracks=[crack])
    else:
        pc3 = pc2.Copy(); pc3.Translate(*extrude)
        pc4 = pc1.Copy(); pc4.Translate(*extrude)
        l1 = Line(pc1, pc2, meshSize/6, isOpen=openCrack)
        l2 = Line(pc2, pc3, meshSize/6)
        l3 = Line(pc3, pc4, meshSize/6, isOpen=openCrack)
        l4 = Line(pc4, pc1, meshSize/6)
        crack = Contour([l1, l2, l3, l4], isOpen=openCrack)
        mesh = Mesher().Mesh_Extrude(contour, inclusions, extrude, [], ElemType.TETRA4, cracks=[crack])

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    material = Materials.Elas_Isot(dim, E=210000, v=0.3, planeStress=False, thickness=thickness)
    simu = Simulations.ElasticSimu(mesh, material)

    if isSymmetric:
        nodes_x0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
        nodes_y0 = mesh.Nodes_Conditions(lambda x,y,z: y == 0)
        simu.add_dirichlet(nodes_x0, [0], ['x'])
        simu.add_dirichlet(nodes_y0, [0], ['y'])
    
    nodes_load = mesh.Nodes_Cylinder(Circle(center, r*2), extrude)

    if dim == 2 and not isSymmetric:
        sig *= -1

    simu.add_pressureLoad(nodes_load, sig)

    simu.Solve()
    simu.Save_Iter()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    factorDef = r/5 / simu.Result('displacement_norm').max()
    # factorDef = 1
    Display.Plot_Mesh(simu, deformFactor=factorDef)
    Display.Plot_Result(simu, 'ux', ncolors=10, nodeValues=True)
    Display.Plot_Result(simu, 'uy', ncolors=10, nodeValues=True)
    Display.Plot_Result(simu, 'Svm', ncolors=10, nodeValues=True, deformFactor=factorDef, plotMesh=True)

    print(simu)

    pvi.Plot_BoundaryConditions(simu).show()

    Display.plt.show()