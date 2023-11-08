"""Functions for importing samples data"""

import Folder
import pickle
import pandas as pd
import numpy as np
import Materials
from Interface_Gmsh import Interface_Gmsh, Mesh, ElemType
from Geom import Point, Domain, Circle
import Display


folder = Folder.Get_Path(__file__)

# ----------------------------------------------
# Datas
# ----------------------------------------------

# récupère les courbes forces déplacements
# pathDataFrame = Folder.Join([folder_file, "data_dfEssais.pickle"])
pathDataFrame = Folder.Join([folder, "data_dfEssaisRedim.pickle"])
with open(pathDataFrame, "rb") as file:
    dfLoad = pd.DataFrame(pickle.load(file))

pathDataLoadMax = Folder.Join([folder, "data_df_loadMax.pickle"])
with open(pathDataLoadMax, "rb") as file:
    dfLoadMax = pd.DataFrame(pickle.load(file))

# récupère les proritétés identifiées
pathParams = Folder.Join([folder, "params_Essais.xlsx"])
# pathParams = Folder.Join([folder_file, "params_Essais new.xlsx"])
dfParams = pd.read_excel(pathParams)

def Get_material(idxEssai: int, thickness: float) -> Materials.Elas_IsotTrans:

    El = dfParams["El"][idxEssai]
    Et = dfParams["Et"][idxEssai]
    Gl = dfParams["Gl"][idxEssai]
    # vl = dfParams["vl"][idxEssai]
    vl = 0.02
    vt = 0.44

    rot = 90 * np.pi/180
    axis_l = np.array([np.cos(rot), np.sin(rot), 0])
    axis_t = np.cross(np.array([0,0,1]), axis_l)

    material = Materials.Elas_IsotTrans(2, El, Et, Gl, vl, vt, axis_l, axis_t, True, thickness)

    return material

def Get_loads_informations(idxEssai: int) -> tuple[float, float, float]:

    forces = dfLoad["forces"][idxEssai]
    deplacements = dfLoad["deplacements"][idxEssai]

    f_max = np.max(forces)
    f_crit = dfLoadMax["Load [kN]"][idxEssai]

    return forces, deplacements, f_crit

def DoMesh(L: float, H: float, D: float, l0: float, test: bool, optimMesh: bool) -> Mesh:

    meshSize = l0 if test else l0/2

    if optimMesh:
        epRefine = D
        refineGeom = Domain(Point(L/2-epRefine), Point(L/2+epRefine, H), meshSize)
        meshSize *= 3
    else:
        refineGeom = None

    domain = Domain(Point(), Point(L, H), meshSize)
    circle = Circle(Point(L/2, H/2), D, meshSize)

    mesh = Interface_Gmsh().Mesh_2D(domain, [circle], ElemType.TRI3, refineGeoms=[refineGeom])

    return mesh

def Calc_a_b(forces, deplacements, fmax):
    """Calcul des coefs de f(x) = a x + b"""

    idxElas = np.where((forces <= fmax))[0]
    idx1, idx2 = idxElas[0], idxElas[-1]
    x1, x2 = deplacements[idx1], deplacements[idx2]
    f1, f2 = forces[idx1], forces[idx2]
    vect_ab = np.linalg.inv(np.array([[x1, 1],[x2, 1]])).dot(np.array([f1, f2]))
    a, b = vect_ab[0], vect_ab[1]

    return a, b



if __name__ == "__main__":

    plt = Display.plt

    Display.Clear()

    folderIden = Folder.Join([Folder.New_File("Essais FCBA",results=True), "Identification"])

    pathData = Folder.Join([folderIden, "identification.xlsx"])

    df = pd.read_excel(pathData)

    df = df[(df['solveur']==1)&(df['ftol']==1e-5)]
    df = df.sort_values(by=['Essai'])

    df = df.set_index(np.arange(df.shape[0]))

    # print(df)

    # df.

    axFcrit = plt.subplots()[1]
    axFcrit.bar(df.index, df["f_crit"].values)
    axFcrit.set_xticks(df.index)
    axFcrit.set_xlabel("Samples")
    axFcrit.set_ylabel("Crack initiation forces")
    Display.Save_fig(folderIden, 'crack init essais')
    # axFcrit.tick_params(axis='x', labelrotation = 45)
    # plt.xlim([0, None])
    # plt.ylim([0, y_max])

    axGc = plt.subplots()[1]
    axGc.bar(df.index, df["Gc"].values)
    axGc.set_xticks(df.index)
    # axGc.set_xlabel("Samples", fontsize=14)
    axGc.set_xlabel("Samples")
    axGc.set_ylabel("$G_c \ [mJ \ mm^{-2}]$")
    Display.Save_fig(folderIden, 'Gc essais')

    # errors = [2,6,8,9,11,13,17]
    errors = []

    df.drop(errors, axis=0, inplace=True)

    ax_fit = plt.subplots()[1]
    ax_fit.set_xlabel('$G_c$')
    ax_fit.set_ylabel('Crack initiation forces')

    f_crit = df["f_crit"].values
    Gc = df["Gc"].values
    for i in range(Gc.size):        
        ax_fit.scatter(Gc[i],f_crit[i],c='blue')
        # ax.text(Gc[i],f_crit[i],f'Essai{i}')


    from scipy.optimize import minimize
    J = lambda x: np.linalg.norm(f_crit - (x[0]*Gc + x[1]))

    res = minimize(J, [0,0])
    a, b = tuple(res.x)

    Gc_array = np.linspace(Gc.min(), Gc.max(), 100)
    curve: np.ndarray = a*Gc_array + b
    

    r = np.mean((Gc-Gc.mean())/Gc.std() * (f_crit-f_crit.mean())/f_crit.std())
    # r = np.corrcoef(Gc,f_crit)[0,1]


    ax_fit.plot(Gc_array, curve,c='red')

    ax_fit.text(Gc_array.mean(), curve.mean(), f"{a:.3f} Gc + {b:.3f}, r={r:.3f}", va='top')
    # bbox=dict(boxstyle="square,pad=0.3",alpha=1,color='white')

    Display.Save_fig(folderIden, "corr")


    pass