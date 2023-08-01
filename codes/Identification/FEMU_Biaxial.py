from scipy.optimize import least_squares
import pandas as pd

from Interface_Gmsh import Interface_Gmsh
from Geom import Point, PointsList, Circle
import Display
import Materials
import Simulations
import Folder

Get_ddls_noeuds = Simulations.BoundaryCondition.Get_dofs_nodes

np = Materials.np
plt = Display.plt

Display.Clear()

folder = Folder.New_File("Identification Biaxial", results=True)

# --------------------------------------
# Config
# --------------------------------------

L = 70 #mm
h = 40
# r = 15 # 14
r = (L-h)/2
ep = 0.5

meshSize = h/20

pltMesh = False

# --------------------------------------
# Mesh
# --------------------------------------

pt1 = Point(-L/2, h/2)
pt2 = Point(-L/2, -h/2)
pt3 = Point(-h/2, -h/2, r=r)
pt4 = Point(-h/2, -L/2)
pt5 = Point(h/2, -L/2)
pt6 = Point(h/2, -h/2, r=r)
pt7 = Point(L/2, -h/2)
pt8 = Point(L/2, h/2)
pt9 = Point(h/2, h/2, r=r)
pt10 = Point(h/2, L/2)
pt11 = Point(-h/2, L/2)
pt12 = Point(-h/2, h/2, r=r)

contour = PointsList([pt1,pt2,pt3,pt4,pt5,pt6,pt7,pt8,pt9,pt10,pt11,pt12], meshSize)

circle = Circle(Point(h/3, h/3), 10, meshSize, False)

mesh = Interface_Gmsh(False, False).Mesh_2D(contour, [circle], "QUAD4")

if pltMesh:
    Display.Plot_Mesh(mesh)

# récupère les noeuds

nodesLeft = mesh.Nodes_Conditions(lambda x,y,z: x==-L/2)
nodesRight = mesh.Nodes_Conditions(lambda x,y,z: x==L/2)
nodesUpper = mesh.Nodes_Conditions(lambda x,y,z: y==L/2)
nodesLower = mesh.Nodes_Conditions(lambda x,y,z: y==-L/2)

# --------------------------------------
# Comportement
# --------------------------------------

tol0 = 1e-6
bSup = np.inf

El = 3000
Et = 6000
Gl = 1500
vl = 0.25
vt = 0.4

axisL = np.array([0,1,0])
axisT = np.array([1,0,0])


dict_param = {
    "El" : El,
    "Gl" : Gl,
    "Et" : Et,
    "vl" : vl
}

El0 = El * 10
Gl0 = Gl * 10
Et0 = Et * 10
vl0 = vl

lb = [tol0]*4
ub = (bSup, bSup, bSup, 0.5-tol0)
bounds = (lb, ub)
x0 = [El0, Gl0, Et0, vl0]

comp = Materials.Elas_IsotTrans(2, El, Et, Gl, vl, vt, axisL, axisT, True, ep)

compIdentif = Materials.Elas_IsotTrans(2, El0, Et0, Gl0, vl0, vt, axisL, axisT, True, ep)

# --------------------------------------
# Simulation
# --------------------------------------

simu = Simulations.Simu_Displacement(mesh, comp)

# dep = 1 # mm
# simu.add_dirichlet(nodesLeft, [-dep], ['x'])
# simu.add_dirichlet(nodesRight, [dep], ['x'])
# simu.add_dirichlet(nodesUpper, [dep], ['y'])
# simu.add_dirichlet(nodesLower, [-dep], ['y'])


fexp = 1 # N
simu.add_lineLoad(nodesLeft, [-fexp/h], ['x'])
simu.add_lineLoad(nodesRight, [fexp/h], ['x'])
simu.add_lineLoad(nodesUpper, [fexp/h], ['y'])
simu.add_lineLoad(nodesLower, [-fexp/h], ['y'])



# Display.Plot_BoundaryConditions(simu)

u_exp = simu.Solve()
simu.Save_Iter()

# Display.Plot_Result(simu, "ux")
# Display.Plot_Result(simu, "uy")

# Display.Plot_Result(simu, "Sxx")
# Display.Plot_Result(simu, "Syy")
# Display.Plot_Result(simu, "Sxy")
# Display.Plot_Result(simu, "Svm")

# ----------------------------------------------
# Identification
# ----------------------------------------------

useRescale = True
perturbations = np.linspace(0, 0.02, 4)
nTirage = 10
tol = 1e-10

Display.Section("Identification")

simuIdentif = Simulations.Simu_Displacement(mesh, compIdentif)

def func(x):
    # Fonction coût

    # x0 = [EL0, GL0, ET0, vL0]
    compIdentif.El = x[0]
    compIdentif.Gl = x[1]
    compIdentif.Et = x[2]
    compIdentif.vl = x[3]        

    simuIdentif.Need_Update()

    u = simuIdentif.Solve()
    
    diff = u - u_exp_bruit
    diff = diff[ddlsInconnues]

    return diff

def Add_Dirichlet(nodes: np.ndarray, directions=["x","y"]):
    """Ajoute les conditions de déplacements"""

    ddls = Get_ddls_noeuds(2, "displacement", nodes, directions)

    nDim = len(directions)

    values = u_exp_bruit[ddls]
    values = np.reshape(values, (-1,nDim))

    valeurs = [values[:,d] for d in range(nDim)]

    simuIdentif.add_dirichlet(nodes, valeurs, directions)
        

# liste de dictionnaire qui va contenir pour les différentes perturbations les
# propriétés identifiées

list_dict_perturbation = []

for perturbation in perturbations:

    print(f"\nperturbation = {perturbation}")

    list_dict_tirage = []

    for tirage in range(nTirage):

        print(f"tirage = {tirage}", end='\r')        

        # bruitage de la solution
        bruit = np.abs(u_exp).max() * (np.random.rand(u_exp.shape[0]) - 1/2) * perturbation
        u_exp_bruit = u_exp + bruit

        compIdentif.El = El0
        compIdentif.Gl = Gl0
        compIdentif.Et = Et0
        compIdentif.vl = vl0            

        simuIdentif.Bc_Init()
        
        simuIdentif.add_lineLoad(nodesRight, [fexp/h], ['x'])
        simuIdentif.add_lineLoad(nodesUpper, [fexp/h], ['y'])

        simuIdentif.add_lineLoad(nodesLeft, [-fexp/h], ['x'])
        # Add_Dirichlet(nodesLeft, ['x','y'])

        # simuIdentif.add_lineLoad(nodesLower, [-fexp/h], ['y'])
        Add_Dirichlet(nodesLower, ['x','y'])        

        ddlsConnues, ddlsInconnues = simuIdentif.Bc_dofs_known_unknow(simuIdentif.problemType)        

        # res = least_squares(func, x0, bounds=bounds, verbose=2, ftol=tol, gtol=tol, xtol=tol, jac='3-point')
        res = least_squares(func, x0, bounds=bounds, verbose=0, ftol=tol, gtol=tol, xtol=tol)

        dict_tirage = {
            "tirage" : tirage
        }

        dict_tirage["El"]=res.x[0]
        dict_tirage["Gl"]=res.x[1]
        dict_tirage["Et"]=res.x[2]
        dict_tirage["vl"]=res.x[3]
            

        list_dict_tirage.append(dict_tirage)

    df_tirage = pd.DataFrame(list_dict_tirage)

    dict_perturbation = {
        "perturbation" : perturbation,
    }

    dict_perturbation["El"] = df_tirage["El"].values
    dict_perturbation["Gl"] = df_tirage["Gl"].values
    dict_perturbation["Et"] = df_tirage["Et"].values
    dict_perturbation["vl"] = df_tirage["vl"].values        

    list_dict_perturbation.append(dict_perturbation)
    
Display.Plot_BoundaryConditions(simuIdentif, folder)

df_pertubation = pd.DataFrame(list_dict_perturbation)

# ----------------------------------------------
# Affichage
# ----------------------------------------------

params = ["El", "Gl", "Et", "vl"]    

borne = 0.95
bInf = 0.5 - (0.95/2)
bSup = 0.5 + (0.95/2)

for param in params:

    axParam = plt.subplots()[1]
    
    paramExp = dict_param[param]

    perturbations = df_pertubation["perturbation"]

    nPertu = perturbations.size
    values = np.zeros((nPertu, nTirage))
    for p in range(nPertu):
        values[p] = df_pertubation[param].values[p]
    
    print(f"{param} = {values.mean()}")

    values *= 1/paramExp

    mean = values.mean(axis=1)
    std = values.std(axis=1)

    paramInf, paramSup = tuple(np.quantile(values, (bInf, bSup), axis=1))

    axParam.plot(perturbations, [1]*nPertu, label=f"{param}_exp", c="black", ls='--')
    axParam.plot(perturbations, mean, label=f"{param}_moy")
    axParam.fill_between(perturbations, paramInf, paramSup, alpha=0.3, label=f"{borne*100} % ({nTirage} tirages)")
    axParam.set_xlabel("perturbations")
    axParam.set_ylabel(fr"${param} \ / \ {param}_{'{exp}'}$")
    axParam.grid()
    axParam.legend(loc="upper left")
    
    Display.Save_fig(folder, "FEMU_"+param, extension='pdf')

    


diff_n = np.reshape(simuIdentif.displacement - u_exp, (mesh.Nn, 2))

# err_n = np.linalg.norm(diff_n, axis=1)/np.linalg.norm(u_exp.reshape((mesh.Nn,2)), axis=1)
err_n = np.linalg.norm(diff_n, axis=1)/np.linalg.norm(u_exp)
# err_n = np.linalg.norm(diff_n, axis=1)

Display.Plot_Result(simuIdentif, err_n, title=r"$\dfrac{\Vert u(p) - u_{exp} \Vert^2}{\Vert u_{exp} \Vert^2}$")

# print(np.linalg.norm(diff_n)/np.linalg.norm(u_exp))


plt.show()