"""Post-processing module, used to create video or Paraview files."""

import os
from colorama import Fore

import Display as Display
import Simulations
import Folder
from TicTac import Tic
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from datetime import datetime

# It is possible to draw inspiration from :
# https://www.python-graph-gallery.com/

# =========================================== Load and Displacement ==================================================

def Save_Load_Displacement(load: np.ndarray, displacement: np.ndarray, folder:str):
    """Save the values of load and displacements in the folder"""
    
    folder_PythonEF = Folder.Get_Path(Folder.Get_Path())
    filename = Folder.Join(folder, "load and displacement.pickle")

    if not os.path.exists(folder):
        os.makedirs(folder)

    values = {
        'load': load,
        'displacement' : displacement
    }

    with open(filename, "wb") as file:
        pickle.dump(values, file)
    print(Fore.GREEN + f'{filename.replace(folder_PythonEF,"")} (saved)' + Fore.WHITE)
    
def Load_Load_Displacement(folder:str, verbosity=False):
    """Load forces and displacements

    Parameters
    ----------
    folder : str
        name of the folder in which the force and displacement values are saved

    return load, displacement
    """

    folder_PythonEF = Folder.Get_Path(Folder.Get_Path())

    filename = Folder.Join(folder, "load and displacement.pickle")
    error = f"{filename.replace(folder_PythonEF,'')} does not exist"
    assert Folder.Exists(filename), Fore.RED + error + Fore.WHITE

    with open(filename, 'rb') as file:
        values = pickle.load(file)

    load = np.array(values['load'])
    displacement = np.array(values['displacement'])

    if verbosity:
        print(f'\nLoading of:\n {filename}\n')

    return load, displacement


# =========================================== Animation ==================================================

def Get_ffmpegpath() -> str:
    """Return the path to ffmpeg."""

    paths = ["D:\\Soft\\ffmpeg\\bin\\ffmpeg.exe",
                "D:\\Pro\\ffmpeg\\bin\\ffmpeg.exe",
                "/opt/local/bin/ffmpeg",
                "/home/m/matnoel/Applications/ffmpeg"]
    
    for p in paths:
        if os.path.exists(p):
            return p
    
    raise Exception("Folder does not exist")

def Make_Movie(folder: str, option: str, simu: Simulations._Simu, Niter=200, NiterFin=100, deformation=False, plotMesh=False, factorDef=0.0, nodeValues=True, fps=30) -> None:
    """Make a movie from a simulation

    Parameters
    ----------
    folder : str
        save folder
    option : str
        result to display
    simu : Simulations._Simu
        simulation
    Niter : int, optional
        Maximum number of display iterations, by default 200
    NiterFin : int, optional
        Numbre or iterations before the end, by default 100
    deformation : bool, optional
        Displays the deformation, by default False
    plotMesh : bool, optional
        Plot the mesh, by default False
    factorDef : int, optional
        deformation factor, by default 0.0
    nodeValues : bool, optional
        displays result to nodes otherwise displays it to elements, by default True
    fps : int, optional
        frames per second, by default 30
    """
    
    resultat = simu.Result(option)
    if not (isinstance(resultat, np.ndarray) or isinstance(resultat, list)):
        return

    # Add the end character
    if nodeValues:
        name = f'{option}_n'
    else:
        name = f'{option}_e'
    
    # Name of the video in the folder where the folder is communicated
    filename = Folder.Join(folder, f'{name}.mp4')

    if not os.path.exists(folder):
        os.makedirs(folder)

    resultats = simu.results

    N = len(resultats)

    listIter = Make_listIter(NiterMax=N-1, NiterFin=NiterFin, NiterCyble=Niter)
    
    Niter = len(listIter)

    # Update the simulation to create the first figure that will be used for the animation
    simu.Set_Iter(0)

    # Display the first figure
    fig, ax, cb = Display.Plot_Result(simu, option, plotMesh=plotMesh, deformFactor=factorDef, nodeValues=nodeValues)
    
    # Give the link to ffmpeg.exe
    ffmpegpath = Get_ffmpegpath()
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath
    
    listTemps = []
    writer = animation.FFMpegWriter(fps)
    with writer.saving(fig, filename, 200):
        tic = Tic()
        for i, iter in enumerate(listIter):
            simu.Set_Iter(iter)

            cb.remove()
            
            fig, ax, cb = Display.Plot_Result(simu, option, ax=ax, plotMesh=plotMesh, deformFactor=factorDef, nodeValues=nodeValues)

            title1 = ax.get_title()
            ax.set_title(f'{title1} : {iter}/{N-1}')

            plt.pause(1e-12)

            writer.grab_frame()

            tf = tic.Tac("Animation",f"Plot {title1}", False)
            listTemps.append(tf)

            pourcentageEtTempsRestant = _RemainingTime(listIter, listTemps, i)

            print(f"Make_Movie {iter}/{N-1} {pourcentageEtTempsRestant}    ", end='\r')

# ========================================== Paraview =================================================

def Make_Paraview(folder: str, simu: Simulations._Simu, Niter=200, details=False, nodesResult=[], elementsResult=[]):
    """Saving the simulation on paraview

    Parameters
    ----------
    folder: str
        folder in which we will create the Paraview folder
    simulation : Sim
        Simulation
    Niter : int, optional
        Maximum number of display iterations, by default 200
    details: bool, optional
        details of nodesField and elementsField used in the .vtu
    nodesResult: list, optional
        Additional nodesField, by default []
    elementsResult: list, optional
        Additional elemsField, by default []
    """
    print('\n')

    vtuFiles=[]

    resultats = simu.results

    NiterMax = len(resultats)-1

    listIter = Make_listIter(NiterMax=NiterMax, NiterFin=100, NiterCyble=Niter)
    
    Niter = len(listIter)

    folder = Folder.Join(folder,"Paraview")

    if not os.path.exists(folder):
        os.makedirs(folder)

    listTemps = []
    tic = Tic()    

    nodesField, elementsField = simu.Results_nodesField_elementsField(details)

    checkNodesField = [n for n in nodesResult if simu._Results_Check_Available(n)]
    checkElemsField = [n for n in elementsResult if simu._Results_Check_Available(n)]

    nodesField.extend(checkNodesField)
    elementsField.extend(checkElemsField)

    if len(nodesField) == 0 and len(elementsField) == 0:
        print("La simulation ne possède pas de champs de solution à afficher dans paraview")

    for i, iter in enumerate(listIter):

        f = Folder.Join(folder,f'solution_{iter}.vtu')

        vtuFile = __Make_vtu(simu, iter, f, nodesField=nodesField, elementsField=elementsField)
        
        # vtuFiles.append(vtuFile)
        vtuFiles.append(f'solution_{iter}.vtu')

        temps = tic.Tac("Paraview","Make vtu", False)
        listTemps.append(temps)

        pourcentageEtTempsRestant = _RemainingTime(listIter, listTemps, i)

        print(f"SaveParaview {iter}/{NiterMax} {pourcentageEtTempsRestant}    ", end='\r')
    print('\n')

    tic = Tic()

    filenamePvd = os.path.join(folder,"simulation")    
    __Make_pvd(filenamePvd, vtuFiles)

    tic.Tac("Paraview","Make pvd", False)

def Make_listIter(NiterMax: int, NiterFin: int, NiterCyble: int) -> np.ndarray:
    """Cutting the iteration list 

    Parameters
    ----------
    NiterMax : int
        Number of list diteration
    NiterFin : int
        Number of iterations to end
    NiterCyble : int
        Maximum number of iterations selected in the list

    Returns
    -------
    np.ndarray
        iteration list
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

def _RemainingTime(listIter: list[int], listTemps: list[float], i: int) -> str:
    """Calculation of remaining time

    Parameters
    ----------
    listIter : list[int]        
        iteration list
    listTime: list
        list from time to time
    i: int
        loop iteration

    Returns
    -------
    str
        string containing the percentage and the remaining time
    """
    if listIter[-1] == 0:
        return ""

    pourcentage = listIter[i]/listIter[-1]
    if pourcentage > 0:
        tempsRestant = np.min(listTemps)*(len(listIter)-i-1)
        tempsCoef, unite = Tic.Get_time_unity(tempsRestant)
        pourcentageEtTempsRestant = f"({int(pourcentage*100):3}%) {np.round(tempsCoef, 3)} {unite}"
    else:
        pourcentageEtTempsRestant = ""

    return pourcentageEtTempsRestant

def __Make_vtu(simu: Simulations._Simu, iter: int, filename: str, nodesField: list[str], elementsField: list[str]):
    """Create the .vtu which can be read on paraview
    """

    options = nodesField+elementsField
   
    simu.Set_Iter(iter)

    # Verification if the results list is compatible with the simulation
    for option in options:
        resultat = simu.Result(option)
        if not (isinstance(resultat, np.ndarray) or isinstance(resultat, list)):
            return

    connect = simu.mesh.connect
    coordo = simu.mesh.coordo
    Ne = simu.mesh.Ne
    Nn = simu.mesh.Nn
    nPe = simu.mesh.groupElem.nPe

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
        "TETRA10" : 24,
        "TETRA10" : 10,
        "HEXA8": 12,
        # "HEXA20": 29,
        "HEXA20": 12,
        "PRISM6": 13,
        # "PRISM15": 15
        "PRISM15": 13
    } # regarder https://github.com/Kitware/VTK/blob/master/Common/DataModel/vtkCellType.h    

    typeParaviewElement = typesParaviewElement[simu.mesh.elemType]
    
    types = np.ones(Ne, dtype=int)*typeParaviewElement

    node = coordo.reshape(-1)
    """coordinates of nodes in lines"""

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

        # Values at nodes
        file.write('\t\t\t<PointData scalars="scalar"> \n')
        offset=0
        list_valeurs_n=[]
        for resultat_n in nodesField:

            valeurs_n = simu.Result(resultat_n, nodeValues=True).reshape(-1)
            list_valeurs_n.append(valeurs_n)

            nombreDeComposantes = int(valeurs_n.size/Nn) # 1 ou 3
            if resultat_n == "matrix_displacement": resultat_n="displacement"
            file.write(f'\t\t\t\t<DataArray type="Float32" Name="{resultat_n}" NumberOfComponents="{nombreDeComposantes}" format="appended" offset="{offset}" />\n')
            offset = CalcOffset(offset, valeurs_n.size)

        file.write('\t\t\t</PointData> \n')

        # Element values
        file.write('\t\t\t<CellData> \n')
        list_valeurs_e=[]
        for resultat_e in elementsField:

            valeurs_e = simu.Result(resultat_e, nodeValues=False).reshape(-1)
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
        
        # Adding values
        file.write('\t<AppendedData encoding="raw"> \n_')

    # Add all values in binary
    with open(filename, "ab") as file:

        # Nodes values
        for valeurs_n in list_valeurs_n:
            __WriteBinary(const*(valeurs_n.size), "uint32", file)
            __WriteBinary(valeurs_n, "float32", file)

        # Elements values
        for valeurs_e in list_valeurs_e:                
            __WriteBinary(const*(valeurs_e.size), "uint32", file)
            __WriteBinary(valeurs_e, "float32", file)

        # Nodes
        __WriteBinary(const*(node.size), "uint32", file)
        __WriteBinary(node, "float32", file)

        # Connectivity            
        __WriteBinary(const*(connectivity.size), "uint32", file)
        __WriteBinary(connectivity, "int32", file)

        # Offsets
        __WriteBinary(const*Ne, "uint32", file)
        __WriteBinary(offsets+3, "int32", file)

        # Element tyoes
        __WriteBinary(types.size, "uint32", file)
        __WriteBinary(types, "int8", file)

    with open(filename, "a") as file:

        # End of adding data
        file.write('\n\t</AppendedData>\n')

        # End of vtk
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
        """Convert to Binary"""

        if type not in ['uint32','float32','int32','int8']:
            raise Exception("Type not implemented")

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