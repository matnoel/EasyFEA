"""Functions for importing samples data"""

import Folder
import pickle
import pandas as pd
import numpy as np
import Materials
from Interface_Gmsh import Interface_Gmsh, Mesh, ElemType
from Geom import Point, Domain, Circle


folder_file = Folder.Get_Path(__file__)

# ----------------------------------------------
# Datas
# ----------------------------------------------

# récupère les courbes forces déplacements
# pathDataFrame = Folder.Join([folder_file, "data_dfEssais.pickle"])
pathDataFrame = Folder.Join([folder_file, "data_dfEssaisRedim.pickle"])
with open(pathDataFrame, "rb") as file:
    dfLoad = pd.DataFrame(pickle.load(file))

pathDataLoadMax = Folder.Join([folder_file, "data_df_loadMax.pickle"])
with open(pathDataLoadMax, "rb") as file:
    dfLoadMax = pd.DataFrame(pickle.load(file))

# récupère les proritétés identifiées
pathParams = Folder.Join([folder_file, "params_Essais.xlsx"])
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