import Folder
import PostProcessing
import Display
from Geoms import *
import Materials
from Mesh import Mesh, Calc_projector, Calc_New_meshSize_n
from GmshInterface import Mesher, ElemType
import Simulations
from TicTac import Tic

import os
import matplotlib.pyplot as plt

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    dim = 2 # Dimension of the problem (2D or 3D)

    # Choosing the part type (you can uncomment one of the parts)
    part = "equerre"
    # part = "lmt"
    # part = "other"

    # Creating a folder to store the results
    folder = Folder.New_File(f"OptimMesh{dim}D", results=True)
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
    cible = 1/100 if dim == 2 else 0.04
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
        elemType = ElemType.HEXA8 # TETRA4, TETRA10, HEXA8, HEXA20, PRISM6, PRISM15

    # ----------------------------------------------
    # Meshing
    # ----------------------------------------------

    # Define the geometry based on the chosen part type
    if part == "equerre":

        L = 120 #mm
        h = L * 0.3
        b = h

        N = 5
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
        r = h/(2+1e-3)
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

        meshSize = h/3

        pt1 = Point(0, 0, isOpen=True)
        pt2 = Point(L, 0)
        pt3 = Point(L, h)
        pt4 = Point(0, h)

        points = Points([pt1, pt2, pt3, pt4], meshSize)    

        inclusions = []
        nL = 20
        nH = 3
        cL = L/(2*nL)
        cH = h/(2*nH)
        for i in range(nL):
            x = cL + cL*(2*i)
            for j in range(nH):
                y = cH + cH*(2*j)

                ptd1 = Point(x-cL/2, y-cH/2)
                ptd2 = Point(x+cL/2, y+cH/2)
                
                isHollow = True
                
                if (i+j)//2 % 2 == 1:
                    inclusion = Domain(ptd1, ptd2, meshSize, isHollow=isHollow)
                else:
                    # obj = Domain(ptd1, ptd2, meshSize, isHollow=isHollow)
                    inclusion = Circle(Point(x, y), cH, meshSize, isHollow=isHollow)

                inclusions.append(inclusion)
        inclusions = []

    # Create an instance of the Gmsh interface
    interfaceGmsh = Mesher()

    def DoMesh(refineGeoms=[]) -> Mesh:
        """Function used for mesh generation"""
        if dim == 2:
            return interfaceGmsh.Mesh_2D(points, inclusions, elemType, [], refineGeoms)
        else:
            return interfaceGmsh.Mesh_Extrude(points, inclusions, [0,0,b], [5], elemType, [], refineGeoms)

    # Construct the initial mesh
    mesh = DoMesh()

    # ----------------------------------------------
    # Material and Simulation
    # ----------------------------------------------
    material = Materials.Elas_Isot(dim, E=210000, v=0.3, thickness=b)
    simu = Simulations.Simu_Displacement(mesh, material, verbosity=False)
    simu.rho = 8100*1e-9

    def DoSimu(i=0):    
        
        nodes_x0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
        nodes_xL = mesh.Nodes_Conditions(lambda x,y,z: x == L)
        
        simu.Bc_Init()
        if dim == 2:
            simu.add_dirichlet(nodes_x0, [0, 0], ["x","y"], description="Encastrement")
        elif dim == 3:
            simu.add_dirichlet(nodes_x0, [0, 0, 0], ["x","y","z"], description="Encastrement")

        simu.add_surfLoad(nodes_xL, [-surfLoad], ["y"])
        # simu.add_surfLoad(noeuds_en_L, [surfLoad], ["x"])

        simu.Solve()

        simu.Save_Iter()

        # ----------------------------------------------
        # Calc ZZ1
        # ----------------------------------------------
        error, error_e = simu._Calc_ZZ1()

        # ----------------------------------------------
        # Refine mesh
        # ----------------------------------------------
        meshSize_n = Calc_New_meshSize_n(simu.mesh, error_e, coef)

        if plotError:
            Display.Plot_Result(simu, error_e*100, nodeValues=True, title="error %", plotMesh=True)

        path = interfaceGmsh.Create_posFile(simu.mesh.coordo, meshSize_n, folder, f"simu{i}")

        return path, error

    path = None
    error = 1
    i = -1
    while error >= cible and i < iterMax:

        i += 1

        if i > 0:
            oldMesh = simu.mesh
            oldU = simu.displacement

        mesh = DoMesh([path])    

        simu.mesh = mesh

        if i > 0:
            os.remove(path)

            if plotProj:
                
                proj = Calc_projector(oldMesh, mesh)

                newDofs = simu.Bc_dofs_nodes(mesh.nodes, ["x"])
                oldDofs = simu.Bc_dofs_nodes(oldMesh.nodes, ["x"])
                uproj = np.zeros(mesh.Nn*dim)        
                for d in range(dim):
                    uproj[newDofs+d] = proj @ oldU[oldDofs+d]

                simu.set_u_n("displacement", uproj)

                ax1 = Display.Plot_Result(oldMesh, oldU.reshape(-1,dim)[:,1], plotMesh=True, title="old uy")[1]
                ax2 = Display.Plot_Result(simu, "uy", plotMesh=True, title="uy proj")[1]
                if dim == 2:
                    ax1.scatter(mesh.coordo[:,0], mesh.coordo[:,1], marker='+', c='black')
                else:
                    ax1.scatter(mesh.coordo[:,0], mesh.coordo[:,1], mesh.coordo[:,2], marker='+', c='black')

                pass

                plt.close(ax1.figure)
                plt.close(ax2.figure)


        path, error = DoSimu(i)

        print(f"{i} error = {error*100:.3} %, Wdef = {simu.Result('Wdef'):.3f} mJ")

    if i > 0:
        os.remove(path)

    # ----------------------------------------------
    # PostProcessing
    # ----------------------------------------------
    # folder=""
    if plotResult:
        tic = Tic()    
        # Display.Plot_Result(simu, "displacement_norm")
        # Display.Plot_Mesh(simu, deformation=True, folder=folder)
        Display.Plot_Result(simu, "ux", nodeValues=False)        
        Display.Plot_Result(simu, "Svm", plotMesh=True, nodeValues=False)
        Display.Plot_Result(simu, "ZZ1", plotMesh=True)
        # Display.Plot_Mesh(mesh, alpha=0, edgecolor='white', ax=plt.gca())
        # Display.Plot_Result(simu, "Svm", deformation=True, nodeValues=False, plotMesh=False, folder=folder)

    if makeParaview:
        PostProcessing.Make_Paraview(folder, simu, nodesResult=["ZZ1"])

    if makeMovie:
        PostProcessing.Make_Movie(folder, "ZZ1", simu, plotMesh=True, fps=1, nodeValues=True)

    Tic.Plot_History(details=True)
    plt.show()