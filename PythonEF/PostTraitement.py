
import os

import Affichage as Affichage
from Simu import Simu
import Dossier as Dossier
from TicTac import Tic
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from datetime import datetime

# Il est possible de sinspirer de :
# https://www.python-graph-gallery.com/


# =========================================== Simulation ==================================================

def Save_Simu(simu: Simu, folder:str):
    "Sauvegarde la simulation et son résumé dans le dossier"

    # TODO Effacer les matrices elements finis construit pour prendre moins de place
    # Il faut vider les matrices dans les groupes d'elements
    try:
        simu.mesh.ResetMatrices()
    except:
        # Cette option n'était pas encore implémentée
        pass
 
    # returns current date and time
    dateEtHeure = datetime.now()
    resume = f"Simulation réalisée le : {dateEtHeure}"
    nomSimu = "simulation.pickle"
    filename = Dossier.Join([folder, nomSimu])
    print(f'\nSauvegarde dans : \n{folder}')
    print(f'  - {nomSimu}')
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Sauvagarde la simulation
    with open(filename, "wb") as file:
        pickle.dump(simu, file)

    # Sauvegarde le résumé de la simulation
    resume += simu.Resume(False)
    nomResume = "résumé.txt"
    print(f'  - {nomResume} \n')
    filenameResume = Dossier.Join([folder, nomResume])

    with open(filenameResume, 'w', encoding='utf8') as file:
        file.write(resume)

def Load_Simu(folder: str, verbosity=False):
    """Charge la simulation depuis le dossier

    Parameters
    ----------
    folder : str
        nom du dossier dans lequel simulation est sauvegardée

    Returns
    -------
    Simu
        simu
    """

    filename = Dossier.Join([folder, "simulation.pickle"])
    assert os.path.exists(filename), "Le fichier simulation.pickle est introuvable"

    with open(filename, 'rb') as file:
        simu = pickle.load(file)

    assert isinstance(simu, Simu)

    if verbosity:
        print(f'\nChargement de :\n{filename}\n')
        simu.mesh.Resume()
        simu.materiau.Resume()
    return simu


# =========================================== Load and Displacement ==================================================

def Save_Load_Displacement(load: np.ndarray, displacement: np.ndarray, folder:str):
    "Sauvegarde les valeurs de forces [N] et déplacements [m] dans le dossier"
    
    filename = Dossier.Join([folder, "load and displacement.pickle"])

    print(f'\nSauvegarde de :\n {filename}\n')

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

    filename = Dossier.Join([folder, "load and displacement.pickle"])
    assert os.path.exists(filename), "Le fichier load and displacement.pickle est introuvable"

    with open(filename, 'rb') as file:
        values = pickle.load(file)

    load = np.array(values['load'])
    displacement = np.array(values['displacement'])

    if verbosity:
        print(f'\nChargement de :\n {filename}\n')

    return load, displacement


# =========================================== Animation ==================================================

def MakeMovie(folder: str, option: str, simu: Simu, Niter=200,
deformation=False, affichageMaillage=False, facteurDef=4, valeursAuxNoeuds=True):
    
    # Verifie que l'option est dispo
    if not simu.VerificationOption(option):
        return

    # Ajoute le caractère de fin
    if valeursAuxNoeuds:
        name = f'{option}_n'
    else:
        name = f'{option}_e'
    
    # Nom de la vidéo dans le dossier ou est communiqué le dossier
    filename = Dossier.Join([folder, f'{name}.mp4'])

    if not os.path.exists(folder):
        os.makedirs(folder)

    results = simu.Get_All_Results()

    N = len(results)

    listIter = __Get_listIter(NiterMax=N, NiterFin=100, NiterCyble=Niter)
    
    Niter = len(listIter)

    # Met à jour la simulation pour creer la première figure qui sera utilisée pour l'animation
    simu.Update_iter(0)

    # Trace la première figure
    fig, ax, cb = Affichage.Plot_Result(simu, option,
    affichageMaillage=affichageMaillage, deformation=deformation, facteurDef=facteurDef)
    
    # Donne le lien vers ffmpeg.exe

    def Get_ffmpegpath():
        paths = ["D:\\Soft\\ffmpeg\\bin\\ffmpeg.exe",
                 "D:\\Pro\\ffmpeg\\bin\\ffmpeg.exe",
                 "/opt/local/bin/ffmpeg"]
        
        for p in paths:
            if os.path.exists(p):
                return p
        
        raise "Dossier inexistant"

    ffmpegpath = Get_ffmpegpath()
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath

    
    listTemps = []
    writer = animation.FFMpegWriter(fps=30)
    with writer.saving(fig, filename, 200):
        tic = Tic()
        for i, iter in enumerate(listIter):
            simu.Update_iter(iter)

            cb.remove()
            
            fig, ax, cb = Affichage.Plot_Result(simu, option, oldfig=fig, oldax=ax,
            deformation=deformation, affichageMaillage=affichageMaillage, facteurDef=facteurDef, valeursAuxNoeuds=True)

            title = ax.get_title()
            ax.set_title(f'{title} : {iter}/{N-1}')

            plt.pause(0.00001)

            writer.grab_frame()

            tf = tic.Tac("Animation",f"Plot {ax.get_title()}", False)
            listTemps.append(tf)

            pourcentageEtTempsRestant = __GetPourcentageEtTemps(listIter, listTemps, i)

            print(f"Makemovie {iter}/{N-1} {pourcentageEtTempsRestant}    ", end='\r')
    
# =========================================== Paraview ==================================================

def Save_Simulation_in_Paraview(folder: str, simu: Simu, Niter=200):
    """Sauvegarde de la simulation sur paraview

    Parameters
    ----------
    folder : str
        dossier dans lequel on va creer le dossier Paraview
    simu : Simu
        Simulation
    Niter : int, optional
        Nombre d'iteration maximum d'affichage, by default 200
    """
    print('\n')

    vtuFiles=[]

    results = simu.Get_All_Results()

    NiterMax = len(results)-1

    listIter = __Get_listIter(NiterMax=NiterMax, NiterFin=100, NiterCyble=Niter)
    
    Niter = len(listIter)

    folder = Dossier.Join([folder,"Paraview"])

    if not os.path.exists(folder):
        os.makedirs(folder)

    listTemps = []
    tic = Tic()

    problemType = simu.problemType
    
    if problemType == "thermal":
        nodesField = ["thermal", "thermalDot"] # "thermalDot"
        elementsField = []
    elif problemType == "damage":
        nodesField =  ["coordoDef","damage"]        
        elementsField=["Stress"] # ["Stress","psiP"]
    elif problemType == "displacement":
        nodesField =  ["coordoDef"]
        if isinstance(simu.speed, np.ndarray):
            nodesField.append("speed")
        if isinstance(simu.accel, np.ndarray):
            nodesField.append("accel")
        elementsField=["Stress"]

    for i, iter in enumerate(listIter):

        f = Dossier.Join([folder,f'solution_{iter}.vtu'])

        vtuFile = __Make_vtu(simu, iter, f, nodesField=nodesField, elementsField=elementsField)
        
        vtuFiles.append(vtuFile)

        temps = tic.Tac("Paraview","Make vtu", False)
        listTemps.append(temps)

        pourcentageEtTempsRestant = __GetPourcentageEtTemps(listIter, listTemps, i)

        print(f"SaveParaview {iter}/{NiterMax} {pourcentageEtTempsRestant}    ", end='\r')
    print('\n')

    tic = Tic()

    filenamePvd = os.path.join(folder,"simulation")    
    __Make_pvd(filenamePvd, vtuFiles)

    tic.Tac("Paraview","Make pvd", False)

def __Get_listIter(NiterMax: int, NiterFin: int, NiterCyble: int) -> np.ndarray:

    NavantFin = NiterMax-NiterFin
    listIterFin = np.arange(NavantFin+1, NiterMax+1, dtype=int)
    NiterRestant = NiterCyble - NiterFin

    if NiterMax > NiterCyble:
        listIter = np.linspace(0, NavantFin, NiterRestant, endpoint=True, dtype=int)
        listIter = np.append(listIter, listIterFin)
    else:
        listIter = np.arange(NiterMax+1)


    return listIter


def __GetPourcentageEtTemps(listIter: list, listTemps: list, i):
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

def __Make_vtu(simu: Simu, iter: int, filename: str,nodesField=["coordoDef","Stress"], elementsField=["Stress","Strain"]):
    """Creer le .vtu qui peut être lu sur paraview
    """

    options = nodesField+elementsField
   
    simu.Update_iter(iter)

    for option in options:
        if not simu.VerificationOption(option):
            return

    connect = simu.mesh.connect
    coordo = simu.mesh.coordo
    Ne = simu.mesh.Ne
    Nn = simu.mesh.Nn
    nPe = simu.mesh.nPe

    typesParaviewElement = {
        "TRI3" : 5,
        "TRI6" : 22,
        "QUAD4" : 9,
        "QUAD8" : 23,
        "TETRA4" : 10,
        "HEXA8": 12,
        "PRISM6": 13
    } # regarder vtkelemtype D:\SOFT\FEMObject\BASIC\MODEL\@ELEMENT\vtkelemtype.m

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

            valeurs_n = simu.Get_Resultat(resultat_n, valeursAuxNoeuds=True).reshape(-1)
            list_valeurs_n.append(valeurs_n)

            nombreDeComposantes = int(valeurs_n.size/Nn) # 1 ou 3
            if resultat_n == "coordoDef": resultat_n="displacement"
            file.write(f'\t\t\t\t<DataArray type="Float32" Name="{resultat_n}" NumberOfComponents="{nombreDeComposantes}" format="appended" offset="{offset}" />\n')
            offset = CalcOffset(offset, valeurs_n.size)

        file.write('\t\t\t</PointData> \n')

        # Valeurs aux elements
        file.write('\t\t\t<CellData> \n')
        list_valeurs_e=[]
        for resultat_e in elementsField:

            valeurs_e = simu.Get_Resultat(resultat_e, valeursAuxNoeuds=False).reshape(-1)
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
    
    path = Dossier.GetPath(filename)
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
            raise "Pas dans les options"

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


def Save_fig(folder:str, title: str,transparent=False, extension='png'):

    if folder == "": return

    for char in ['NUL', '\ ', ',', '/',':','*', '?', '<','>','|']: title = title.replace(char, '')

    nom = Dossier.Join([folder, title+'.'+extension])

    if not os.path.exists(folder):
        os.makedirs(folder)

    # plt.savefig(nom, dpi=200)
    plt.savefig(nom, dpi=500, transparent=transparent,bbox_inches='tight')