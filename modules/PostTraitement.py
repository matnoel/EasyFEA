import os
from colorama import Fore

import Affichage as Affichage
import Simulations
import Folder
from TicTac import Tic
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from datetime import datetime

# Il est possible de sinspirer de :
# https://www.python-graph-gallery.com/


# =========================================== Load and Displacement ==================================================

def Save_Load_Displacement(load: np.ndarray, displacement: np.ndarray, folder:str):
    "Sauvegarde les valeurs de forces [N] et déplacements [m] dans le dossier"
    
    filename = Folder.Join([folder, "load and displacement.pickle"])

    print(Fore.GREEN + f'\nSauvegarde de :\n  - load and displacement.pickle' + Fore.WHITE)

    values = {
        'load': load,
        'displacement' : displacement
    }

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(filename, "wb") as file:
        pickle.dump(values, file)
    
def Load_Load_Displacement(folder:str, verbosity=False):
    """Charge les forces [N] et déplacements [m]

    Parameters
    ----------
    folder : str
        nom du dossier dans lequel les valeurs de forces et de déplacements sont sauvegardées

    return load, displacement
    """

    filename = Folder.Join([folder, "load and displacement.pickle"])
    assert os.path.exists(filename), Fore.RED + "Le fichier load and displacement.pickle est introuvable" + Fore.WHITE

    with open(filename, 'rb') as file:
        values = pickle.load(file)

    load = np.array(values['load'])
    displacement = np.array(values['displacement'])

    if verbosity:
        print(f'\nChargement de :\n {filename}\n')

    return load, displacement


# =========================================== Animation ==================================================

def Make_Movie(folder: str, option: str, simu: Simulations.Simu, Niter=200, NiterFin=100, deformation=False, plotMesh=False, facteurDef=4, nodeValues=True, fps=30):
    
    resultat = simu.Get_Resultat(option)
    if not (isinstance(resultat, np.ndarray) or isinstance(resultat, list)):
        return

    # Ajoute le caractère de fin
    if nodeValues:
        name = f'{option}_n'
    else:
        name = f'{option}_e'
    
    # Nom de la vidéo dans le dossier ou est communiqué le dossier
    filename = Folder.Join([folder, f'{name}.mp4'])

    if not os.path.exists(folder):
        os.makedirs(folder)

    resultats = simu._results

    N = len(resultats)

    listIter = Make_listIter(NiterMax=N-1, NiterFin=NiterFin, NiterCyble=Niter)
    
    Niter = len(listIter)

    # Met à jour la simulation pour creer la première figure qui sera utilisée pour l'animation
    simu.Update_iter(0)

    # Trace la première figure
    fig, ax, cb = Affichage.Plot_Result(simu, option, plotMesh=plotMesh, deformation=deformation, facteurDef=facteurDef, nodeValues=nodeValues)
    
    # Donne le lien vers ffmpeg.exe

    def Get_ffmpegpath():
        paths = ["D:\\Soft\\ffmpeg\\bin\\ffmpeg.exe",
                 "D:\\Pro\\ffmpeg\\bin\\ffmpeg.exe",
                 "/opt/local/bin/ffmpeg",
                 "/home/m/matnoel/Applications/ffmpeg"]
        
        for p in paths:
            if os.path.exists(p):
                return p
        
        raise Exception("Dossier inexistant")

    ffmpegpath = Get_ffmpegpath()
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath

    
    listTemps = []
    writer = animation.FFMpegWriter(fps)
    with writer.saving(fig, filename, 200):
        tic = Tic()
        for i, iter in enumerate(listIter):
            simu.Update_iter(iter)

            cb.remove()
            
            fig, ax, cb = Affichage.Plot_Result(simu, option, ax=ax, deformation=deformation, plotMesh=plotMesh, facteurDef=facteurDef, nodeValues=nodeValues)

            title1 = ax.get_title()
            ax.set_title(f'{title1} : {iter}/{N-1}')

            plt.pause(1e-12)

            writer.grab_frame()

            tf = tic.Tac("Animation",f"Plot {title1}", False)
            listTemps.append(tf)

            pourcentageEtTempsRestant = _GetPourcentageEtTemps(listIter, listTemps, i)

            print(f"Makemovie {iter}/{N-1} {pourcentageEtTempsRestant}    ", end='\r')

# ========================================== Paraview =================================================

def Make_Paraview(folder: str, simu: Simulations.Simu, Niter=200, details=False):
    """Sauvegarde de la simulation sur paraview

    Parameters
    ----------
    folder : str
        dossier dans lequel on va creer le dossier Paraview
    simu : Simu
        Simulation
    Niter : int, optional
        Nombre d'iteration maximum d'affichage, by default 200
    details: bool, optional
        details de nodesField et elementsField utilisé dans le .vtu
    """
    print('\n')

    vtuFiles=[]

    resultats = simu._results

    NiterMax = len(resultats)-1

    listIter = Make_listIter(NiterMax=NiterMax, NiterFin=100, NiterCyble=Niter)
    
    Niter = len(listIter)

    folder = Folder.Join([folder,"Paraview"])

    if not os.path.exists(folder):
        os.makedirs(folder)

    listTemps = []
    tic = Tic()    

    nodesField, elementsField = simu.Paraview_nodesField_elementsField(details)

    if len(nodesField) == 0 or len(elementsField) == 0:
        print("La simulation ne possède pas de champs de solution à afficher dans paraview")

    for i, iter in enumerate(listIter):

        f = Folder.Join([folder,f'solution_{iter}.vtu'])

        vtuFile = __Make_vtu(simu, iter, f, nodesField=nodesField, elementsField=elementsField)
        
        vtuFiles.append(vtuFile)

        temps = tic.Tac("Paraview","Make vtu", False)
        listTemps.append(temps)

        pourcentageEtTempsRestant = _GetPourcentageEtTemps(listIter, listTemps, i)

        print(f"SaveParaview {iter}/{NiterMax} {pourcentageEtTempsRestant}    ", end='\r')
    print('\n')

    tic = Tic()

    filenamePvd = os.path.join(folder,"simulation")    
    __Make_pvd(filenamePvd, vtuFiles)

    tic.Tac("Paraview","Make pvd", False)

def Make_listIter(NiterMax: int, NiterFin: int, NiterCyble: int) -> np.ndarray:
    """Découpage de la liste ditération 

    Parameters
    ----------
    NiterMax : int
        Nombre ditération de la liste
    NiterFin : int
        Nombre d'itération à la fin
    NiterCyble : int
        Nombre d'itération maximale selectionnée dans la liste

    Returns
    -------
    np.ndarray
        la liste d'itération
    """

    NavantFin = NiterMax-NiterFin
    listIterFin = np.arange(NavantFin+1, NiterMax+1, dtype=int)
    NiterRestant = NiterCyble - NiterFin

    if NiterMax > NiterCyble:
        listIter = np.linspace(0, NavantFin, NiterRestant, endpoint=True, dtype=int)
        listIter = np.append(listIter, listIterFin)
    else:
        listIter = np.arange(NiterMax+1)


    return listIter


def _GetPourcentageEtTemps(listIter: list[int], listTemps: list[float], i: int) -> str:
    """Calul le pourcentage et le temps restant

    Parameters
    ----------
    listIter : list[int]
        liste d'itération
    listTemps : list
        liste de temps en s
    i : int
        iteration dans la boucle

    Returns
    -------
    str
        string contenant le pourcentage et le temps restant
    """
    if listIter[-1] == 0:
        return ""

    pourcentage = listIter[i]/listIter[-1]
    if pourcentage > 0:
        tempsRestant = np.min(listTemps)*(len(listIter)-i-1)
        tempsCoef, unite = Tic.Get_temps_unite(tempsRestant)
        pourcentageEtTempsRestant = f"({int(pourcentage*100):3}%) {np.round(tempsCoef, 3)} {unite}"
    else:
        pourcentageEtTempsRestant = ""

    return pourcentageEtTempsRestant

def __Make_vtu(simu: Simulations.Simu, iter: int, filename: str, nodesField: list[str], elementsField: list[str]):
    """Creer le .vtu qui peut être lu sur paraview
    """

    options = nodesField+elementsField
   
    simu.Update_iter(iter)

    # Verification si la liste de résultats est compatible avec la simulation
    for option in options:
        resultat = simu.Get_Resultat(option)
        if not (isinstance(resultat, np.ndarray) or isinstance(resultat, list)):
            return

    connect = simu.mesh.connect
    coordo = simu.mesh.coordo
    Ne = simu.mesh.Ne
    Nn = simu.mesh.Nn
    nPe = simu.mesh.nPe

    typesParaviewElement = {
        "SEG2" : 3,
        "SEG3" : 21,
        "SEG4" : 35,        
        "TRI3" : 5,
        "TRI6" : 22,
        "TRI10" : 69,
        "TRI15" : 69,
        "QUAD4" : 9,
        "QUAD8" : 23,
        "TETRA4" : 10,
        # "TETRA10" : 24,
        "TETRA10" : 10,
        "HEXA8": 12,
        "PRISM6": 13
    } # regarder https://github.com/Kitware/VTK/blob/master/Common/DataModel/vtkCellType.h

    typeParaviewElement = typesParaviewElement[simu.mesh.elemType]
    
    types = np.ones(Ne, dtype=int)*typeParaviewElement

    node = coordo.reshape(-1)
    """coordonnées des noeuds en lignes"""

    connectivity = connect.reshape(-1)

    offsets = np.arange(nPe,nPe*Ne+1,nPe, dtype=np.int32)-3

    endian_paraview = 'LittleEndian' # 'LittleEndian' 'BigEndian'

    const=4

    def CalcOffset(offset, taille):
        return offset + const + (const*taille)

    with open(filename, "w") as file:
        
        file.write('<?pickle version="1.0" ?>\n')
        
        file.write(f'<VTKFile type="UnstructuredGrid" version="0.1" byte_order="{endian_paraview}">\n')

        file.write('\t<UnstructuredGrid>\n')
        file.write(f'\t\t<Piece NumberOfPoints="{Nn}" NumberOfCells="{Ne}">\n')

        # Valeurs aux noeuds
        file.write('\t\t\t<PointData scalars="scalar"> \n')
        offset=0
        list_valeurs_n=[]
        for resultat_n in nodesField:

            valeurs_n = simu.Get_Resultat(resultat_n, nodeValues=True).reshape(-1)
            list_valeurs_n.append(valeurs_n)

            nombreDeComposantes = int(valeurs_n.size/Nn) # 1 ou 3
            if resultat_n == "matrice_displacement": resultat_n="displacement"
            file.write(f'\t\t\t\t<DataArray type="Float32" Name="{resultat_n}" NumberOfComponents="{nombreDeComposantes}" format="appended" offset="{offset}" />\n')
            offset = CalcOffset(offset, valeurs_n.size)

        file.write('\t\t\t</PointData> \n')

        # Valeurs aux elements
        file.write('\t\t\t<CellData> \n')
        list_valeurs_e=[]
        for resultat_e in elementsField:

            valeurs_e = simu.Get_Resultat(resultat_e, nodeValues=False).reshape(-1)
            list_valeurs_e.append(valeurs_e)

            nombreDeComposantes = int(valeurs_e.size/Ne)
            
            file.write(f'\t\t\t\t<DataArray type="Float32" Name="{resultat_e}" NumberOfComponents="{nombreDeComposantes}" format="appended" offset="{offset}" />\n')
            offset = CalcOffset(offset, valeurs_e.size)
        
        file.write('\t\t\t</CellData> \n')

        # Points
        file.write('\t\t\t<Points>\n')
        file.write(f'\t\t\t\t<DataArray type="Float32" NumberOfComponents="3" format="appended" offset="{offset}" />\n')
        offset = CalcOffset(offset, node.size)
        file.write('\t\t\t</Points>\n')

        # Elements
        file.write('\t\t\t<Cells>\n')
        file.write(f'\t\t\t\t<DataArray type="Int32" Name="connectivity" format="appended" offset="{offset}" />\n')
        offset = CalcOffset(offset, connectivity.size)
        file.write(f'\t\t\t\t<DataArray type="Int32" Name="offsets" format="appended" offset="{offset}" />\n')
        offset = CalcOffset(offset, offsets.size)
        file.write(f'\t\t\t\t<DataArray type="Int8" Name="types" format="appended" offset="{offset}" />\n')
        file.write('\t\t\t</Cells>\n')                    
        
        # END VTK FILE
        file.write('\t\t</Piece>\n')
        file.write('\t</UnstructuredGrid> \n')
        
        # Ajout des valeurs
        file.write('\t<AppendedData encoding="raw"> \n_')

    # Ajoute toutes les valeurs en binaire
    with open(filename, "ab") as file:

        # Valeurs aux noeuds
        for valeurs_n in list_valeurs_n:
            __WriteBinary(const*(valeurs_n.size), "uint32", file)
            __WriteBinary(valeurs_n, "float32", file)

        # Valeurs aux elements
        for valeurs_e in list_valeurs_e:                
            __WriteBinary(const*(valeurs_e.size), "uint32", file)
            __WriteBinary(valeurs_e, "float32", file)

        # Noeuds
        __WriteBinary(const*(node.size), "uint32", file)
        __WriteBinary(node, "float32", file)

        # Connectivity            
        __WriteBinary(const*(connectivity.size), "uint32", file)
        __WriteBinary(connectivity, "int32", file)

        # Offsets
        __WriteBinary(const*Ne, "uint32", file)
        __WriteBinary(offsets+3, "int32", file)

        # Type d'element
        __WriteBinary(types.size, "uint32", file)
        __WriteBinary(types, "int8", file)

    with open(filename, "a") as file:

        # Fin de l'ajout des données
        file.write('\n\t</AppendedData>\n')

        # Fin du vtk
        file.write('</VTKFile> \n')
    
    path = Folder.Get_Path(filename)
    vtuFile = str(filename).replace(path+'\\', '')

    return vtuFile


def __Make_pvd(filename: str, vtuFiles=[]):

    tic = Tic()

    endian_paraview = 'LittleEndian' # 'LittleEndian' 'BigEndian'

    filename = filename+".pvd"

    with open(filename, "w") as file:

        file.write('<?pickle version="1.0" ?>\n')

        file.write(f'<VTKFile type="Collection" version="0.1" byte_order="{endian_paraview}">\n')
        file.write('\t<Collection>\n')
        
        for t, vtuFile in enumerate(vtuFiles):
            file.write(f'\t\t<DataSet timestep="{t}" group="" part="1" file="{vtuFile}"/>\n')
        
        file.write('\t</Collection>\n')
        file.write('</VTKFile>\n')
    
    t = tic.Tac("Paraview","Make pvd", False)

def __WriteBinary(valeur, type: str, file):
        """Convertie en byte

        Args:
            valeur (_type_): valeur a convertir
            type (str): type de conversion 'uint32','float32','int32','int8'
        """            

        if type not in ['uint32','float32','int32','int8']:
            raise Exception("Type non implémenté")

        if type == "uint32":
            valeur = np.uint32(valeur)
        elif type == "float32":
            valeur = np.float32(valeur)
        elif type == "int32":
            valeur = np.int32(valeur)
        elif type == "int8":
            valeur = np.int8(valeur)

        convert = valeur.tobytes()
        
        file.write(convert)