"""Code used to perform elastic simulations with FCBA samples"""

import Display
import PyVista_Interface as pvi
from Geoms import Point, Points, Circle, Domain, Line
from Gmsh_Interface import Mesher, ElemType, Mesh, Normalize_vect
import Materials
import Simulations
import Folder
import PostProcessing

import matplotlib.pyplot as plt
import numpy as np

Display.Clear()

folder_FCBA = Folder.New_File("FCBA",results=True)
folder = Folder.Join(folder_FCBA, "Elas")

def DoMesh_FCBA(dim:int, L:float, H:float, D:float, h:float, D2:float, h2:float, t:float, l0:float, test:bool, optimMesh:bool) -> Mesh:

    clC = l0 if test else l0/2
    clD = clC*3 if optimMesh else clC

    if optimMesh:
        refineGeom = Domain(Point(L/2-D, 0, 0), Point(L/2+D, H, t), clC)
    else:
        refineGeom = None

    contour = Domain(Point(), Point(L,H), clD)
    circle = Circle(Point(L/2, H-h), D, clC, True)

    # Hole        
    p1 = Point(L/2, H-55, t/2)
    p2 = Point(L/2+D2/2, p1.y+D2/2, p1.z)
    p3 = Point(p2.x, H, p1.z)
    p4 = p3 - [D2/2]        
    hole = Points([p1,p2,p3,p4])
    axis = Line(p1,p4)
    # hole.Plot_Geoms([contour, circle, hole, axis])

    if dim == 2:
        mesh = Mesher().Mesh_2D(contour, [circle], ElemType.TRI3, refineGeoms=[refineGeom])
        
    elif dim == 3:
        if D2 == 0:
            mesh = Mesher().Mesh_Extrude(contour, [circle], [0,0,t], [4], ElemType.TETRA4, refineGeoms=[refineGeom])
        else:
            mesher = Mesher(False, False)
            fact = mesher._factory

            # Box and cylinder
            surf1 = mesher._Surfaces(contour, [circle])[0]
            vol1 = mesher._Extrude(surf1, [0,0,t])
            # Hole
            surf2 = mesher._Surfaces(hole)[0]
            vol2 = mesher._Revolve(surf2, axis)
            
            fact.cut(vol1, [(ent) for ent in vol2 if ent[0] == 3])

            mesher.Set_meshSize(clD)

            mesher._RefineMesh([refineGeom], clD)

            mesher._Set_PhysicalGroups()

            mesher._Meshing(3, ElemType.TETRA4)

            mesh = mesher._Construct_Mesh()

    return mesh

if __name__  == '__main__':

    # --------------------------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------------------------
    dim = 2

    test = True
    optimMesh = True
    loadInHole = True; pltLoadInHole=False
    makeParaview = False

    # geom
    H = 120 # mm
    L = 90
    D = 10
    h = 35
    D2 = 7
    h2 = 55
    t = 20

    # nL = 50
    # l0 = L/nL
    l0 = 1
    nL = L//l0

    # --------------------------------------------------------------------------------------------
    # Mesh
    # --------------------------------------------------------------------------------------------    
    mesh = DoMesh_FCBA(dim, L, H, D, h, D2, h2, t, l0, test, optimMesh)

    Display.Plot_Mesh(mesh)
    print(mesh)

    # --------------------------------------------------------------------------------------------
    # Material
    # --------------------------------------------------------------------------------------------
    # Properties for test 4
    Gc = 0.075 # mJ/mm2

    psiC = (3*Gc)/(16*l0) 

    El = 15716.16722094732 
    Et = 232.6981580878141
    Gl = 557.3231495541391
    vl = 0.02
    vt = 0.44

    rot = 90 * np.pi/180
    axis_l = np.array([np.cos(rot), np.sin(rot), 0])
    axis_t = np.cross(np.array([0,0,1]), axis_l)

    split = "AnisotStress"
    regu = "AT1"

    comp = Materials.Elas_IsotTrans(dim, El, Et, Gl, vl, vt, axis_l, axis_t, True, t)
    pfm = Materials.PhaseField_Model(comp, split, regu, Gc, l0)

    # --------------------------------------------------------------------------------------------
    # Simulation
    # --------------------------------------------------------------------------------------------
    simu = Simulations.Displacement(mesh, comp)

    nodesLower = mesh.Nodes_Conditions(lambda x,y,z: y==0)

    # --------------------------------------------------------------------------------------------
    # Loading
    # --------------------------------------------------------------------------------------------
    if loadInHole:

        surf = np.pi * D/2 * t
        nodesLoad = mesh.Nodes_Cylinder(Circle(Point(L/2,H-h), D), [0,0,-t])
        nodesLoad = nodesLoad[mesh.coordo[nodesLoad,1] <= H-h]
        # Display.Plot_Nodes(mesh, nodesLoad)

        group = mesh.Get_list_groupElem(dim-1)[0]
        elems = group.Get_Elements_Nodes(nodesLoad)

        aire = np.einsum('ep,p->', group.Get_jacobian_e_pg("mass")[elems], group.Get_weight_pg("mass"))

        if dim == 2:
            aire *= t 

        print(f"errSurf = {np.abs(surf-aire)/surf:.3e}")

        def Eval(x: np.ndarray, y: np.ndarray, z: np.ndarray):
            """Evaluation of the sig cos(theta)^2 vect_n function\n
            x,y,z (ep)"""
            
            # Angle calculation
            theta = np.arctan((x-L/2)/(y-(H-h)))

            # Coordinates of Gauss points in matrix form
            coord = np.zeros((x.shape[0],x.shape[1],3))
            coord[:,:,0] = x
            coord[:,:,1] = y
            coord[:,:,2] = 0

            # Construction of the normal vector
            vect = coord - np.array([L/2, H-h,0])
            vectN = np.einsum('npi,np->npi', vect, 1/np.linalg.norm(vect, axis=2))
            
            # Loading
            loads = f/surf * np.einsum('np,npi->npi',np.cos(theta)**2, vectN)

            return loads

        EvalX = lambda x,y,z: Eval(x,y,z)[:,:,0]
        EvalY = lambda x,y,z: Eval(x,y,z)[:,:,1]

    else:
        surf = t * L
        nodesLoad = mesh.Nodes_Conditions(lambda x,y,z: y==H)

    # --------------------------------------------------------------------------------------------
    # Simulation
    # --------------------------------------------------------------------------------------------
    array_f = np.linspace(0, 4, 10)*1000
    # array_f = np.array([2000])

    list_psiP: list[float] = []

    for f in array_f:

        simu.Bc_Init()
        simu.add_dirichlet(nodesLower, [0]*dim, simu.Get_directions())
        if loadInHole:
            # label = r"$\mathbf{q}(\theta) = \sigma \ sin^2(\theta) \ \mathbf{n}(\theta)$"
            label = ""
            simu.add_surfLoad(nodesLoad, [EvalX, EvalY], ["x","y"],
                            description=label)
        else:
            simu.add_surfLoad(nodesLoad, [-f/surf], ['y'])
        
        # solve and save iteraton
        simu.Solve()
        simu.Save_Iter()

        # Energy calculation
        Epsilon_e_pg = simu._Calc_Epsilon_e_pg(simu.displacement, "mass")
        psiP_e_pg, psiM_e_pg = pfm.Calc_psi_e_pg(Epsilon_e_pg)
        psiP_e = np.max(psiP_e_pg, axis=1)

        list_psiP.append(np.max(psiP_e))

        print(f"f = {f/1000:.3f} kN -> psiP/psiC = {list_psiP[-1]/psiC:.2e}")

    # --------------------------------------------------------------------------------------------
    # Results
    # --------------------------------------------------------------------------------------------

    if pltLoadInHole:
        pvi.Plot_BoundaryConditions(simu).show()    

    if len(list_psiP) > 1:
        axLoad = plt.subplots()[1]
        axLoad.set_xlabel("$f \ [kN]$"); axLoad.set_ylabel("$\psi^+ \ / \ \psi_c$")
        axLoad.grid() 

        array_psiP = np.array(list_psiP)

        axLoad.plot([0,array_f[-1]/1000], [1, 1], zorder=3, c='black')
        axLoad.plot(array_f/1000, array_psiP/psiC, zorder=3, c='blue')
    
        Display.Save_fig(folder, "Load")

    Display.Plot_Mesh(mesh)
    
    if dim == 2 and pltLoadInHole:
        ax = Display.Plot_BoundaryConditions(simu, folder=folder)
        f_v = simu.Get_K_C_M_F()[0] @ simu.displacement
        f_m = f_v.reshape(-1,2)
        f_m *= 1
        nodes = np.concatenate([nodesLoad, nodesLower])
        xn,yn,zn = mesh.coordo[:,0], mesh.coordo[:,1], mesh.coordo[:,2]
        # ax.quiver(xn[nodes], yn[nodes], f_m[nodes,0], f_m[nodes,1], color='red', width=1e-3, scale=1e3)
        ax.quiver(xn, yn, f_m[:,0], f_m[:,1], color='red', width=1e-3, scale=1e4)

    Display.Plot_Result(simu, psiP_e, title="$\psi^+$", nodeValues=False)
    ax = Display.Plot_Result(simu, psiP_e/psiC, nodeValues=True, title="$\psi^+ \ / \ \psi_c$", colorbarIsClose=False)[1]

    # # plot damage nodes
    # elemtsDamage = np.where(psiP_e >= psiC)[0]
    # if elemtsDamage.size > 0:
    #     nodes = list(set(mesh.connect[elemtsDamage].ravel()))
    #     # Display.Plot_Elements(mesh, nodes, alpha=0.2, edgecolor='black', ax=ax)
        
    Display.Save_fig(folder, "psiPpsiC")

    Display.Plot_Result(simu, "Sxx", plotMesh=False)
    Display.Plot_Result(simu, "Syy", plotMesh=False)
    Display.Plot_Result(simu, "Sxy", plotMesh=False)

    print(simu)

    if makeParaview:
        PostProcessing.Make_Paraview(folder, simu)

    plt.show()