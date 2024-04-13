"""Displays stress fields for the plate in compression."""

from EasyFEA import (Display, Tic, plt, np,
                     Mesher, ElemType,
                     Materials, Simulations)
from EasyFEA.Geoms import Point, Domain, Circle

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    dim = 2

    # loading
    SIG = 5 #Pa
    loadInHole = True

    # material
    E=12e9
    v=0.2
    planeStress = True

    # phase field
    split = "Miehe" # ["Bourdin","Amor","Miehe","Stress"]
    regu = "AT2"
    gc = 1.4

    # Geom
    unit = 1e-3
    L=15*unit
    H=30*unit
    h=H/2
    ep=1*unit
    diam=6*unit
    r=diam/2
    l0 = 0.12 *unit*3

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    clD = l0*2
    clC = l0/2

    point = Point()
    domain = Domain(point, Point(x=L, y=H), clD)
    circle = Circle(Point(x=L/2, y=H-h), diam, clC, isHollow=True)
    val = diam*2

    mesher = Mesher(openGmsh=False, verbosity=False)
    if dim == 2:
        mesh = mesher.Mesh_2D(domain, [circle], ElemType.QUAD8)
    else:
        mesh = mesher.Mesh_Extrude(domain, [circle], [0,0,10*unit], [4], ElemType.HEXA8)

    # get nodes
    nodes_y0 = mesh.Nodes_Conditions(lambda x,y,z: y==0)
    nodes_yH = mesh.Nodes_Conditions(lambda x,y,z: y==H)
    node_00 = mesh.Nodes_Conditions(lambda x,y,z: (x==0) & (y==0))
    if dim == 2:
        nodes_circle = mesh.Nodes_Circle(circle)
    else:
        nodes_circle = mesh.Nodes_Cylinder(circle,[0,0,1])
    # get lower nodes
    nodes_circle = nodes_circle[np.where(mesh.coord[nodes_circle,1]<=circle.center.y)]

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    material = Materials.Elas_Isot(dim, E=E, v=v, planeStress=True, thickness=ep)
    pfm = Materials.PhaseField(material, split, regu, gc, l0)

    simu = Simulations.PhaseFieldSimu(mesh, pfm, verbosity=False)

    simu.add_dirichlet(nodes_y0, [0], ["y"])
    simu.add_dirichlet(node_00, [0], ["x"])

    if loadInHole:
        pc = circle.center.coord
        def Eval(x: np.ndarray, y: np.ndarray, z: np.ndarray):
            """Evaluation de la fonction sig cos(theta)^2 vect_n"""
            
            # Angle calculation
            theta = np.arctan((x-pc[0])/(y-pc[1]))

            # Coordinates of Gauss points in matrix form
            coord = np.zeros((x.shape[0],x.shape[1],3))
            coord[:,:,0] = x
            coord[:,:,1] = y
            coord[:,:,2] = 0

            vect = coord - pc
            vectN = np.einsum('npi,np->npi', vect, 1/np.linalg.norm(vect, axis=2))            
            
            loads = SIG * np.einsum('np,npi->npi',np.cos(theta)**2, vectN)

            return loads

        EvalX = lambda x,y,z: Eval(x,y,z)[:,:,0]
        EvalY = lambda x,y,z: Eval(x,y,z)[:,:,1]
        simu.add_surfLoad(nodes_circle, [EvalX, EvalY], ["x","y"], description=r"$\mathbf{q}(\theta) = \sigma \ cos^2(\theta) \ \mathbf{n}(\theta)$")
    else:
        simu.add_surfLoad(nodes_yH, [-SIG], ['y'])

    Display.Plot_BoundaryConditions(simu)

    simu.Solve()
    simu.Save_Iter()

    # ----------------------------------------------
    # PostProcessing
    # ----------------------------------------------
    Display.Plot_Result(simu, "Sxx", nodeValues=True, coef=1/SIG, title=r"$\sigma_{xx}/\sigma$", filename='Sxx', cmap='seismic')
    Display.Plot_Result(simu, "Syy", nodeValues=True, coef=1/SIG, title=r"$\sigma_{yy}/\sigma$", filename='Syy')
    Display.Plot_Result(simu, "Sxy", nodeValues=True, coef=1/SIG, title=r"$\sigma_{xy}/\sigma$", filename='Sxy')
    Display.Plot_Result(simu, "Svm", coef=1/SIG, title=r"$\sigma_{vm}/\sigma$", filename='Svm')

    Display.Plot_Result(simu, "psiP")

    # Display.Plot_Result(simu, "ux")
    # Display.Plot_Result(simu, "uy")
    # vectF = simu.Get_K_C_M_F()[0] @ simu.displacement
    # Display.Plot_Result(simu, vectF.reshape(-1,dim)[:,1])

    Tic.Resume()

    plt.show()