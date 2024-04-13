"""Damage simulation for a L-part."""

from EasyFEA import (Display, Folder, plt, np, Tic,
                     Mesher, ElemType,
                     Materials, Simulations,
                     Paraview_Interface,
                     PyVista_Interface as pvi)
from EasyFEA.Geoms import Point, Points, Domain, Circle

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    solve = True
    test = True
    optimMesh = True

    pltIter = False
    pltLoad = False
    makeMovie = False
    makeParaview = False

    # geom
    dim = 2
    L = 250 # mm
    ep = 100
    l0 = 5

    # material
    E = 2e4 # MPa
    v = 0.18

    # phase field
    split = "Miehe"
    # split = "AnisotStress"
    regu = "AT1"
    Gc = 130 # J/m2
    Gc *= 1000/1e6 #mJ/mm2
    tolConv = 1e-0
    convOption = 2

    # loading
    adaptLoad = True
    # uMax = 1.2 # mm
    uMax = 1 # mm
    inc0 = uMax/200
    inc1 = inc0/2

    # folder
    name = "L_Shape_Benchmark"
    if dim == 3:
        name += '_3D'
    folder = Folder.New_File(name, results=True)

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    nL = L//l0

    if test:
        hC = l0/2
    else:
        hC = 0.5
        # hC = 0.25

    p1 = Point()
    p2 = Point(L,0)
    p3 = Point(L,L)
    p4 = Point(2*L-30,L)
    p5 = Point(2*L,L)
    p6 = Point(2*L,2*L)
    p7 = Point(0,2*L)

    if optimMesh:
        # hauteur zone rafinée
        h = 100
        refineDomain = Domain(Point(0,L-h/3), Point(L+h/3,L+h), hC)
        hD = hC*5
    else:
        refineDomain = None
        hD = hC

    contour = Points([p1,p2,p3,p4,p5,p6,p7], hD)

    circle = Circle(p5, 100)

    if dim == 2:
        mesh = Mesher().Mesh_2D(contour, [], ElemType.TRI3, refineGeoms=[refineDomain])
    else:
        mesh = Mesher().Mesh_Extrude(contour, [], [0,0,-ep], [3], ElemType.HEXA8, refineGeoms=[refineDomain])

    # Display.Plot_Mesh(mesh)
    # Display.Plot_Tags(mesh)
    # from EasyFEA import PyVista_Interface as pvi
    # pvi.Plot_Mesh(mesh).show()

    nodes_y0 = mesh.Nodes_Conditions(lambda x,y,z: y==0)
    nodes_load = mesh.Nodes_Conditions(lambda x,y,z: (y==L) & (x>=2*L-30))
    node3 = mesh.Nodes_Point(p3); node4 = mesh.Nodes_Point(p4)
    nodes_circle = mesh.Nodes_Cylinder(circle, [0,0,ep])
    nodes_edges = mesh.Nodes_Conditions(lambda x,y,z: (x==0) | (y==0))

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    material = Materials.Elas_Isot(dim, E, v, True, ep)
    pfm = Materials.PhaseField(material, split, regu, Gc, l0)

    folderSimu = Folder.PhaseField_Folder(folder, "", pfm.split, pfm.regularization, "CP", tolConv, "", test, optimMesh, nL=nL)

    if solve:

        simu = Simulations.PhaseFieldSimu(mesh, pfm)
        
        dofsY_load = simu.Bc_dofs_nodes(nodes_load, ['y'])
        
        if pltIter:
            axIter = Display.Plot_Result(simu, 'damage')

            axLoad = Display.init_Axes()
            axLoad.set_xlabel('displacement [mm]')
            axLoad.set_ylabel('load [kN]')

        displacement = []
        force = []
        ud = - inc0
        iter = -1

        while ud <= uMax:
            
            # update displacement
            iter += 1
            ud += inc0 if simu.damage.max() < 0.6 else inc1
            
            # update boundary conditions
            simu.Bc_Init()
            simu.add_dirichlet(nodes_circle, [0], ['d'], "damage")
            simu.add_dirichlet(nodes_y0, [0]*dim, simu.Get_dofs())
            simu.add_dirichlet(nodes_load, [ud], ['y'])

            # solve
            u, d, Kglob, convergence = simu.Solve(tolConv, 500, convOption)
            # calc load
            fr = np.sum(Kglob[dofsY_load,:] @ u)

            # save load and displacement
            displacement.append(ud)
            force.append(fr)

            # print iter
            simu.Results_Set_Iteration_Summary(iter, ud, "mm", ud/uMax, True)

            # save iteration
            simu.Save_Iter()

            if pltIter:
                plt.figure(axIter.figure)
                Display.Plot_Result(simu, 'damage', ax=axIter)
                plt.pause(1e-12)

                plt.figure(axLoad.figure)
                axLoad.scatter(ud, fr/1000, c='black')            
                plt.pause(1e-12)

            if not convergence or np.max(d[nodes_edges]) >= 1:
                # stop simulation if damage occurs on edges or convergence has not been reached
                break
        
        # save load and displacement
        displacement = np.asarray(displacement)
        force = np.asarray(force)
        Simulations.Save_Force_Displacement(force, displacement, folderSimu)

        # save the simulation
        simu.Save(folderSimu)

        Tic.Plot_History(folderSimu, True)    

    else:

        simu = Simulations.Load_Simu(folderSimu)
        mesh = simu.mesh

    force, displacement = Simulations.Load_Force_Displacement(folderSimu)

    # ----------------------------------------------
    # PostProcessing
    # ----------------------------------------------
    Display.Plot_BoundaryConditions(simu)

    Display.Plot_Result(simu, 'damage', folder=folderSimu)

    axLoad = Display.init_Axes()
    axLoad.set_xlabel('displacement [mm]')
    axLoad.set_ylabel('load [kN]')
    axLoad.plot(displacement, force/1000, c="blue")
    Display.Save_fig(folderSimu, "forcedep")

    Display.Plot_Iter_Summary(simu, folderSimu)

    if makeMovie:
        depMax = simu.Result("displacement_norm").max()
        deformFactor = L*.1/depMax
        pvi.Movie_simu(simu, 'damage', folderSimu, 'damage.mp4', show_edges=True, deformFactor=deformFactor, clim=(0,1))

    if makeParaview:
        Paraview_Interface.Make_Paraview(simu, folderSimu)

    plt.show()