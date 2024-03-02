import Folder
import PostProcessing
import Display
from Geoms import *
import Materials
from Mesh import Mesh, Calc_projector
from Gmsh_Interface import Mesher, ElemType
import Simulations
from TicTac import Tic
import PyVista_Interface as pvi

import os
import matplotlib.pyplot as plt

if __name__ == '__main__':

    Display.Clear()

    # --------------------------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------------------------
    dim = 2 # Dimension of the problem (2D or 3D)

    # Choosing the part type (you can uncomment one of the parts)
    part = "equerre"
    # part = "lmt"
    # part = "other"

    # Creating a folder to store the results
    folder = Folder.New_File(Folder.Join('Meshes', f'Optim{dim}D'), results=True)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Options for plotting the results
    plotResult = True
    plotError = False
    plotProj = True
    makeMovie = False
    makeParaview = False

    # Scaling coefficient for the optimization process
    coef = 1/10
    # Target error for the optimization process
    treshold = 1/100 if dim == 2 else 0.04
    # Maximum number of iterations for the optimization process
    iterMax = 20

    # Parameters for the geometry
    L = 120  # mm
    h = 13
    b = 13

    P = 800  # N
    lineLoad = P / h  # N/mm
    surfLoad = P / h / b  # N/mm2

    # TODO use .geo to build geometry ?

    # Selecting the element type for the mesh
    if dim == 2:
        elemType = ElemType.TRI3 # TRI3, TRI6, TRI10, QUAD4, QUAD8
    else:
        elemType = ElemType.TETRA4 # TETRA4, TETRA10, HEXA8, HEXA20, PRISM6, PRISM15

    # --------------------------------------------------------------------------------------------
    # Meshing
    # --------------------------------------------------------------------------------------------

    # Define the geometry based on the chosen part type
    if part == "equerre":

        L = 120 #mm
        h = L * 0.3
        b = h

        N = 2
        meshSize = h/N

        pt1 = Point(isOpen=True, r=-10)
        pt2 = Point(x=L)
        pt3 = Point(x=L,y=h)
        pt4 = Point(x=h, y=h, r=10)
        pt5 = Point(x=h, y=L)
        pt6 = Point(y=L)
        pt7 = Point(x=h, y=h)

        points = Points([pt1, pt2, pt3, pt4, pt5, pt6], h/N)

        inclusions = [Circle(Point(x=h/2, y=h*(i+1)), h/4, meshSize, isHollow=True) for i in range(3)]

        inclusions.extend([Domain(Point(x=h,y=h/2-h*0.1), Point(x=h*2.1,y=h/2+h*0.1), meshSize, False)])

    elif part == "lmt":

        L = 120
        h = L * 2/3
        b = h
        r = h/(2+1e-2)
        e = (L - 2*r)/2

        meshSize = h/10

        pt1 = Point()
        pt2 = Point(e,0)
        pt3 = Point(e,r,r=r)
        pt4 = Point(L-e,r,r=r)
        pt5 = Point(L-e,0)
        pt6 = Point(L,0)

        pt7 = Point(L,h)
        pt8 = Point(L-e,h)
        pt9 = Point(L-e,h-r,r=r)
        pt10 = Point(e,h-r,r=r)
        pt11 = Point(e,h)
        pt12 = Point(0,h)

        points = Points([pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9, pt10, pt11, pt12], meshSize)    
        inclusions = []

    else:  
        

        L = 80
        h1 = L/4
        e1 = h1*.1
        h2 = (h1-e1)*.95
        e2 = (h1-e1-h2)/2
        r = h2/4
        l = L/2
        b = h1

        meshSize = r/3

        F = 1e-3 # 5g
        surfLoad = F/(h1-e1)/b # 1g

        pt1 = Point()
        pt2 = Point(L-h1)
        pt3 = pt2 + [e1,-e1]
        pt4 = Point(L,-e1)
        pt5 = pt4 + [0,h1]
        pt6 = Point(h1, h1-e1)
        pt7 = pt6 + [-e1,e1]
        pt8 = pt1 + [0,h1]

        points = Points([pt1,pt2,pt3,pt4,pt5,pt6,pt7,pt8], meshSize)

        p1 = Point(L/2-l/2,e2, r=r)
        p2 = Point(L/2-l/2+2*r,e2, r=r)
        p3 = Point(L/2-l/2+2*r,e2+r)
        p4 = p3.copy(); p4.symmetry((L/2, (h1-e1)/2), (1,0))
        p5 = p2.copy(); p5.symmetry((L/2, (h1-e1)/2), (1,0))
        p6 = p1.copy(); p6.symmetry((L/2, (h1-e1)/2), (1,0))
        p7 = p6.copy(); p7.symmetry((L/2, (h1-e1)/2), (0,1))
        p8 = p5.copy(); p8.symmetry((L/2, (h1-e1)/2), (0,1))
        p9 = p4.copy(); p9.symmetry((L/2, (h1-e1)/2), (0,1))
        p10 = p3.copy(); p10.symmetry((L/2, (h1-e1)/2), (0,1))
        p11 = p2.copy(); p11.symmetry((L/2, (h1-e1)/2), (0,1))
        p12 = p1.copy(); p12.symmetry((L/2, (h1-e1)/2), (0,1))

        inclusion = Points([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12], meshSize, True)
        inclusions = [inclusion]

        # ax = points.Get_Contour().Plot()
        # inclusion.Get_Contour().Plot(ax)

    # Create an instance of the Gmsh interface
    mesher = Mesher()

    def DoMesh(refineGeoms=[]) -> Mesh:
        """Function used for mesh generation"""
        if dim == 2:
            return mesher.Mesh_2D(points, inclusions, elemType, [], refineGeoms)
        else:
            return mesher.Mesh_Extrude(points, inclusions, [0,0,b], [], elemType, [], refineGeoms)

    # Construct the initial mesh
    mesh = DoMesh()

    # --------------------------------------------------------------------------------------------
    # Material and Simulation
    # --------------------------------------------------------------------------------------------
    material = Materials.Elas_Isot(dim, E=210000, v=0.3, thickness=b)
    simu = Simulations.Displacement(mesh, material)
    simu.rho = 8100*1e-9

    def DoSimu(i=0):

        if part in ["equerre","lmt"]:        
            nodes_Fixed = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
            nodes_Load = mesh.Nodes_Conditions(lambda x,y,z: x == L)
        else:
            nodes_Fixed = mesh.Nodes_Conditions(lambda x,y,z: y == -e1)
            nodes_Load = mesh.Nodes_Conditions(lambda x,y,z: y == h1)
        
        simu.Bc_Init()
        simu.add_dirichlet(nodes_Fixed, [0]*dim, simu.Get_directions(), description="Fixed")
        simu.add_surfLoad(nodes_Load, [-surfLoad], ["y"])
        # simu.add_surfLoad(noeuds_en_L, [surfLoad], ["x"])

        simu.Solve()

        simu.Save_Iter()

        # --------------------------------------------------------------------------------------------
        # Calc ZZ1
        # --------------------------------------------------------------------------------------------
        error, error_e = simu._Calc_ZZ1()

        # deformFactor = mesh.coordo.mean()*.1/simu.Result('displacement_norm').max()
        # plotter = pvi.Plot_BoundaryConditions(simu).show()
        # # pvi.Plot(simu, None, deformFactor, show_edges=True, plotter=plotter, opacity=.5).show()

        # --------------------------------------------------------------------------------------------
        # Refine mesh
        # --------------------------------------------------------------------------------------------
        meshSize_n = simu.mesh.Get_New_meshSize_n(error_e, coef)

        if plotError:
            Display.Plot_Result(simu, error_e*100, nodeValues=True, title="error %", plotMesh=True)

        path = mesher.Create_posFile(simu.mesh.coordo, meshSize_n, folder, f"simu{i}")

        return path, error

    # --------------------------------------------------------------------------------------------
    # Optimization
    # --------------------------------------------------------------------------------------------
    path = None
    error = 1
    i = -1
    while error >= treshold and i < iterMax:

        i += 1

        # save previous mesh
        oldMesh = simu.mesh
        oldU_d = simu.displacement.reshape((oldMesh.Nn,-1))
        oldW = simu._Calc_Psi_Elas()

        mesh = DoMesh([path])    

        simu.mesh = mesh

        if i > 0:
            
            os.remove(path)

            if plotProj and i == 1:
                
                # constructs a projector to pass nodes values from the old mesh to the new one
                proj = Calc_projector(oldMesh, mesh)                
                newU_d = np.zeros((mesh.Nn, dim), dtype=float)
                for d in range(dim):
                    newU_d[:,d] = proj @ oldU_d[:,d]

                simu.set_u_n("displacement", newU_d.reshape(-1))
                newW = simu._Calc_Psi_Elas()

                axOld = Display.Plot_Result(oldMesh, oldU_d[:,0], plotMesh=True, title="old")[1]
                axOld.scatter(*mesh.coordo[:,:dim].T, marker='+', c='black', label='new nodes')
                axOld.legend()

                axNew = Display.Plot_Result(simu, "ux", plotMesh=True, title="new")[1]

                # pass
                # plt.close(axOld.figure)
                # plt.close(axNew.figure)


        path, error = DoSimu(i)

        print(f"{i} error = {error*100:.3} %, Wdef = {simu.Result('Wdef'):.3f} mJ")

    if i > 0:
        os.remove(path)

    # --------------------------------------------------------------------------------------------
    # PostProcessing
    # --------------------------------------------------------------------------------------------
    # folder=""
    if plotResult:
        tic = Tic()    
        # Display.Plot_Result(simu, "displacement_norm")
        # Display.Plot_Mesh(simu, deformation=True, folder=folder)
        Display.Plot_Result(simu, "ux", nodeValues=False)        
        Display.Plot_Result(simu, "Svm", plotMesh=True, nodeValues=False)
        Display.Plot_Result(simu, "ZZ1_e", plotMesh=True)
        # Display.Plot_Mesh(mesh, alpha=0, edgecolor='white', ax=plt.gca())
        # Display.Plot_Result(simu, "Svm", deformation=True, nodeValues=False, plotMesh=False, folder=folder)

    if makeParaview:
        PostProcessing.Make_Paraview(folder, simu, nodesResult=["ZZ1_e"])

    if makeMovie:


        def func(plotter, n):

            simu.Set_Iter(n)

            pvi.Plot(simu, 'ZZ1_e', show_edges=True, edge_color='grey', plotter=plotter, clim=(0, error), verticalColobar=False)
            # pvi.Plot_BoundaryConditions(simu, plotter=plotter)

            zz1 = simu._Calc_ZZ1()[0]

            plotter.add_title(f'ZZ1 = {zz1*100:.2f} %')

        pvi.Movie_func(func, len(simu.results), folder, f'{part}.gif')

        # PostProcessing.Make_Movie(folder, "ZZ1_e", simu, plotMesh=True, fps=1, nodeValues=True)

    Tic.Plot_History(details=True)
    plt.show()