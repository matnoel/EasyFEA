import Affichage
from Interface_Gmsh import Interface_Gmsh
from Geom import Point, PointsList, Line, Domain, Circle, normalize_vect
import Materials
import Simulations
import Folder
import PostTraitement

plt = Affichage.plt
np = Affichage.np

pltIter = True
pltLoad = True

makeMovie = False
makeParaview = False

doSimu = True

Affichage.Clear()

dim = 2

name = "NotchedBeam_Benchmark"

if dim == 3:
    name += '_3D'

folder = Folder.New_File(name, results=True)

# ----------------------------------------------
# Config
# ----------------------------------------------

test = True
optimMesh = True

unit = 1e-3; # for mm [Guidault, Allix, Champaney, Cornuault, 2008, CMAME], [Miehe, Welschinger, Hofacker, 2010, IJNME], [Miehe, Hofacker, Welschinger, 2010, CMAME],[Passieux, Rethore, Gravouil, Baietto, 2013, CM]

L = 8*unit # height
L1 = 10*unit
L2 = 9*unit
ep = 1
nw = 0.05*unit # notch width mm
diam = 0.5*unit # hole diameter
 
e1 = 5 # inch
e2 = 1.5

l0 = 0.025*unit # mm [Miehe, Welschinger, Hofacker, 2010, IJNME], [Miehe, Hofacker, Welschinger, 2010, CMAME], [Wu, Nguyen, 2018, JMPS], [Wu, Nguyen, Nguyen, Sutula, Bordas, Sinaie, 2019, AAM]

# l0 = L/100

split = "Miehe"
regu = "AT2"

tolConv = 1e-0
convOption = 2

# ----------------------------------------------
# Mesh
# ----------------------------------------------

e1 *= unit
e2 *= unit

if test:
    hC = l0
else:
    hC = l0/2

nL = L/hC

p0 = Point(0, L)
p1 = Point(-L1, L)
p2 = Point(-L1, 0)
p3 = Point(-L2, 0)

pC1 = Point(-e1-nw/2,0)
pC2 = Point(-e1-nw/2,e2)
pC3 = Point(-e1+nw/2,e2)
pC4 = Point(-e1+nw/2,0)

p4 = Point(L2, 0)
p5 = Point(L1, 0)
p6 = Point(L1, L)

c1 = Circle(Point(-4*unit, 2.75*unit), diam, hC, True)
c2 = Circle(Point(-4*unit, 4.75*unit), diam, hC, True)
c3 = Circle(Point(-4*unit, 6.75*unit), diam, hC, True)

if optimMesh:
    # zone rafinée
    z = e2 /2
    refineDomain = Domain(Point(-e1-z,0), Point(-4*unit+z,L), hC)
    # hD = hC*5
    hD = 8*hC
else:
    refineDomain = None
    hD = 0.1*unit # 8 * hC -> 0.025*unit/2 * 8

contour = PointsList([p0,p1,p2,p3,pC1,pC2,pC3,pC4,p4,p5,p6], hD)

inclusions = [c1, c2, c3]

circlePos1 = Circle(p3, e2)
circlePos2 = Circle(p4, e2)

if dim == 2:
    mesh = Interface_Gmsh().Mesh_2D(contour, inclusions, "TRI3", refineGeom=refineDomain)
    directions = ["x","y"]
else:
    mesh = Interface_Gmsh().Mesh_3D(contour, inclusions, [0,0,ep], 3, "HEXA8", refineGeom=refineDomain)
    directions = ["x","y","z"]

Affichage.Plot_Mesh(mesh)
# Affichage.Plot_Model(mesh)

nodesLoad = mesh.Nodes_Point(p0)
node3 = mesh.Nodes_Point(p3); node4 = mesh.Nodes_Point(p4)
nodesEnca = np.concatenate([node3, node4])

nodesCircle1 = mesh.Nodes_Cylindre(circlePos1, [0,0,ep])
nodesCircle2 = mesh.Nodes_Cylindre(circlePos2, [0,0,ep])
nodesDamage = np.concatenate([nodesCircle1, nodesCircle2])

ddlsY_Load = Simulations.BoundaryCondition.Get_ddls_noeuds(dim, "displacement", nodesLoad, ['y'])

# ----------------------------------------------
# Material
# ----------------------------------------------
# [Ambati, Gerasimov, De Lorenzis, 2015, CM]
E = 20.8e9 # Pa
v = 0.3

Gc = 1e-3 # kN / mm
Gc *= 1000*1000 # 1e3 N / m -> J/m2

comportement = Materials.Elas_Isot(dim, E, v, True, ep)

pfm = Materials.PhaseField_Model(comportement, split, regu, Gc, l0)

folderSimu = Folder.PhaseField_Folder(folder, "", pfm.split, pfm.regularization, "DP", tolConv, "", test, optimMesh, nL=nL)

if doSimu:

    simu = Simulations.Simu_PhaseField(mesh, pfm)

    def Add_Dep(x,y,z):
        """Fonction qui projete le déplacement dans la bonne direction"""
        
        # récupère le déplacement
        dep = simu.Resultats_matrice_displacement()
        # nouvelles coordonées du maillage
        newCoordo = simu.mesh.coordo + dep

        vectBord = newCoordo[node4] - newCoordo[node3]
        vectBord = normalize_vect(vectBord)

        vectDep = np.cross([0,0,1], vectBord)
        vectDep = normalize_vect(vectDep)

        displ = ud * vectDep[0,:2]

        return displ
    
    loadX = lambda x,y,z: Add_Dep(x,y,z)[0]
    loadY = lambda x,y,z: Add_Dep(x,y,z)[1]
    
    if pltIter:
        __, axIter, cb = Affichage.Plot_Result(simu, 'damage')

        axLoad = plt.subplots()[1]
        axLoad.set_xlabel('displacement [mm]')
        axLoad.set_ylabel('load [kN]')

    # [Ambati, Gerasimov, De Lorenzis, 2015, CM]
    # du = 1e-3 mm during the first 200 time steps (up to u = 0.2 mm)
    # du = 1e-4 mm during the last  500 time steps (up to u = 0.25 mm)    
    if test:
        inc0 = 2e-3*unit
        Nt0 = 100        
        inc1 = 2e-4*unit
        Nt1 = 250
    else:
        inc0 = 1e-3*unit
        Nt0 = 200
        inc1 = 1e-4*unit
        Nt1 = 500

    disp1 = np.linspace(0, inc0*Nt0, Nt0)
    start = disp1[-1]
    disp2 = np.linspace(start, start+inc1*Nt1, Nt1)

    displacement = np.unique(np.concatenate([disp1, disp2]))  

    uMax = displacement[-1]    
    load = []

    for iter, ud in enumerate(displacement):

        simu.Bc_Init()
        simu.add_dirichlet(nodesDamage, [0], ['d'], "damage")
        simu.add_dirichlet(nodesEnca, [0]*dim, directions)       
        
        simu.add_dirichlet(nodesLoad, [-ud], ['y'])
        # simu.add_dirichlet(nodesLoad, [loadX, loadY], ['x','y'])

        # Affichage.Plot_BoundaryConditions(simu)        

        u, diam, Kglob, convergence = simu.Solve(tolConv, 500, convOption)

        fr = np.abs(np.sum(Kglob[ddlsY_Load,:] @ u))

        load.append(fr)

        simu.Resultats_Set_Resume_Iteration(iter, ud, "mm", ud/uMax, True)

        simu.Save_Iteration()

        if pltIter:
            plt.figure(axIter.figure)
            cb.remove()
            cb = Affichage.Plot_Result(simu, 'damage', ax=axIter)[2]
            plt.pause(1e-12)

            plt.figure(axLoad.figure)
            axLoad.scatter(ud, fr/1000, c='black')            
            plt.pause(1e-12)

        if not convergence:
            break

    displacement = np.array(displacement)
    load = np.array(load)

    PostTraitement.Save_Load_Displacement(load, displacement, folderSimu)

    simu.Save(folderSimu)

    PostTraitement.Tic.Plot_History(folderSimu, True)

    PostTraitement.Make_Paraview(folderSimu, simu)

else:

    simu = Simulations.Load_Simu(folderSimu)
    mesh = simu.mesh

load, displacement = PostTraitement.Load_Load_Displacement(folderSimu)

# ----------------------------------------------
# PostTraitement
# ----------------------------------------------

Affichage.Plot_Result(simu, 'damage', folder=folderSimu)

axLoad = plt.subplots()[1]
axLoad.set_xlabel('displacement [mm]')
axLoad.set_ylabel('load [kN]')
axLoad.plot(displacement, load/1000, c="blue")
Affichage.Save_fig(folderSimu, "forcedep")

Affichage.Plot_ResumeIter(simu, folderSimu)

if makeMovie:
    depMax = simu.Get_Resultat("amplitude").max()
    facteur = 10*depMax
    PostTraitement.Make_Movie(folderSimu, 'damage', simu, deformation=True, facteurDef=facteur, plotMesh=False)

if makeParaview:
    Affichage.Plot_BoundaryConditions(simu, folderSimu)

plt.show()

pass