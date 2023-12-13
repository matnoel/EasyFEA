"""Functions for importing samples data"""

import Folder
import Materials
from Interface_Gmsh import Interface_Gmsh, Mesh, ElemType
from Geom import Point, Domain, Circle
import Display

import pandas as pd
import numpy as np
from dataclasses import dataclass

folder = Folder.Get_Path(__file__)

# ----------------------------------------------
# Datas
# ----------------------------------------------
dfEssais: pd.DataFrame = pd.read_pickle(Folder.Join(folder, "_essais.pickle"))
"""Data DIC
"Essai": le nom de l'essai sous la forme Essai_XX
"rotate": l'angle pour tourner l'image lors de l'importation
"imgScale [mm/px]": facteur d'echelle entre les mm et les pixels
"(X0, Y0)": coordonnées en haut a gauche de l'échantillon
"(X1, Y1)": coordonnées en bas a droite de l'échantillon
"(XC, YC)": coordonnées du centre du perçage
"Forces [kN]": forces de compression mesurées par la machine
"Deplacements [mm]": déplacements de la traverse lors de l'essai
"Forces redim [kN]": forces mesurées sans la phase transitoire
"Deplacements redim [mm]": deplacements mesurées sans la phase transitoire
"images": images associées aux déplacements et forces mesurées
"Force crack [kN]": force de transition entre le comportement elastique et endommagé
"""

# récupère les proritétés identifiées
dfParams = pd.read_excel(Folder.Join(folder, "_params_article.xlsx"))
"""FEMU params [El, Et, Gl, vl]
"Essai": le nom de l'essai sous la forme Essai_XX
"elemType": le type d'element
"meshSize": taille de maille
"lr": longueur de régularisation
"param": mean(list param)
"std param": écart type
"disp param": std/mean
"list param": liste des paramètres identifiés
"""

# dfGc = pd.read_excel(Folder.Join(folder_iden, "identification.xlsx"))
dfGc = pd.read_excel(Folder.Join(folder, "_gc_article.xlsx"))# _gc_article est une copie de Folder.Join(folder_iden, "identification.xlsx") faite le 6 decembre 2023
"""Identified Gc in a pandas dataframe\n
solver -> solver used to minimize:
    (0, least_squares), (1, minimize)
ftol -> converg tolerance:
    1e-1, 1e-3, 1e-5, 1e-12
split -> Phase field split:
    He, Zhang, AnisotStress 
regu -> phase field regularisation:
    AT1, AT2
tolConv -> phase field tol convergence:
    1e-0, 1e-2 1e-3
convOption -> convergence option for phasefield:
    (0, bourdin), (1, energie crack), (2, energie tot)
"""

def Get_material(idxEssai: int, thickness: float, dim=2) -> Materials.Elas_IsotTrans:

    El = dfParams["El"][idxEssai]
    Et = dfParams["Et"][idxEssai]
    Gl = dfParams["Gl"][idxEssai]
    # vl = dfParams["vl"][idxEssai]
    vl = 0.02
    vt = 0.44

    rot = 90 * np.pi/180
    axis_l = np.array([np.cos(rot), np.sin(rot), 0])
    axis_t = np.cross(np.array([0,0,1]), axis_l)

    material = Materials.Elas_IsotTrans(dim, El, Et, Gl, vl, vt, axis_l, axis_t, True, thickness)

    return material

def Get_loads_informations(idxEssai: int) -> tuple[np.ndarray, np.ndarray, float]:
    """return forces, displacements, f_crit"""

    forces = dfEssais["Forces [kN]"][idxEssai]
    displacements = dfEssais["Deplacements [mm]"][idxEssai]
    
    f_crit = dfEssais["Force crack [kN]"][idxEssai]

    return forces, displacements, f_crit

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

def Calc_a_b(forces, deplacements, fmax) -> tuple[float, float]:
    """Calcul des coefs de f(x) = a x + b"""

    idxElas = np.where((forces <= fmax))[0]
    idx1, idx2 = idxElas[0], idxElas[-1]
    x1, x2 = deplacements[idx1], deplacements[idx2]
    f1, f2 = forces[idx1], forces[idx2]
    vect_ab = np.linalg.inv(np.array([[x1, 1],[x2, 1]])).dot(np.array([f1, f2]))
    a, b = vect_ab[0], vect_ab[1]

    return a, b

@dataclass
class Config:
    start: int
    N: int
    test: bool
    split: str # Bourdin, Amor, Miehe, He etc ..   
    regu: str # AT1, AT2
    tolConv: float # tolerance used in staggered scheme
    convOption: int # convergence option
    # (0, bourdin)
    # (1, crack energy)
    # (2, crack + strain energy
    nL: int # the half crack length l0 = L/nL 
    optimMesh: bool

    @property
    def config_name(self) -> str:
        """configuration name"""
        
        config_name = f'{self.split}_'
        config_name += f'{self.regu}_'
        config_name += f'tolConv{self.tolConv:1.0e}_'
        config_name += f'conv{self.convOption}_'
        config_name += f'nL{self.nL}_'
        if self.optimMesh:
            config_name += 'optimMesh'

        return config_name
    
    @property
    def path(self) -> str:
        """configuration path"""
        path = f'{self.start}_{self.N}'
        config = self.config_name

        if self.test:
            path = Folder.Join(path, 'Test', config)
        else:
            path = Folder.Join(path, config)

        return path
    
    @staticmethod
    def Config_From_Path(path: str):

        sep = Folder.os.path.sep

        folders = path.split(sep)
        
        params = folders[-1]

        # config
        split, regu = params.split('_')[:2]
        tolConv = float(params.split('tolConv')[1].split('_')[0])
        convOption = int(params.split('conv')[1].split('_')[0])
        nL = int(params.split('nL')[1].split('_')[0])
        optimMesh = 'optimMesh' in params

        test = 'Test' in folders[-2]

        if test:
            start_N = folders[-3]
        else:
            start_N = folders[-2]

        start, N = [int(s) for s in start_N.split('_')]

        config = Config(start, N, test, split, regu, tolConv, convOption, nL, optimMesh)

        return config