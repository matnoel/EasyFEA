
import Affichage
from Simu import Simu
import Dossier
from TicTac import TicTac
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def Save_Simulations_in_Paraview(filename: str, simu: Simu, uglob_t: list, damage_t=[]):
    print('\n')

    vtuFiles=[]
    N = len(uglob_t)

    folder = Dossier.GetPath(filename)
    folder = Dossier.Append([folder,"Paraview"])

    Nn = simu.mesh.Nn
    dim = simu.mesh.dim

    for t, uglob in enumerate(uglob_t):

        assert Nn == damage_t[t].size, "Erreur"
        assert Nn*dim == uglob.size, "Erreur"

        simu.Update(uglob, damage_t[t])

        f = f'{folder}\\solution_{t}'

        if len(damage_t) > 0:
            vtuFile = SaveParaview(simu, f, nodesField=["displacement","damage"], elementsField=["Stress"])
        else:
            vtuFile = SaveParaview(simu, f, nodesField=["displacement"], elementsField=["Stress"])
        
        vtuFiles.append(vtuFile)

        print(f"SaveParaview {t+1}/{N}", end='\r')
    
    print('\n')
    filenamePvd = f'{folder}\\solution'
    MakePvd(filenamePvd, vtuFiles)


def MakeMovie(filename: str, option: str, simu: Simu, uglob_t: list,
    damage_t=[], deformation=False, affichageMaillage=False, facteurDef=4, valeursAuxNoeuds=True):
    
    # Verifie que l'option est dispo
    if not simu.VerificationOption(option):
        return
    
    # Verifie que si on demande d'afficher l'endommagement l'endommagement est donné
    if option  == "damage" and len(damage_t) == 0:
        raise "Impossible d'afficher car damage_t n'est pas renseigné"

    # Ajoute le caractère de fin
    if valeursAuxNoeuds:
        name = f'{option}_n'
    else:
        name = f'{option}_e'
    
    # Nom de la vidéo dans le dossier ou est communiqué le dossier
    filename = f'{Dossier.GetPath(filename)}\\{name}.mp4'

    # Nombre d'iteration à afficher
    N = len(uglob_t)

    # Met à jour le matériau pour creer la première figure qui sera utilisée pour l'animation
    simu.Update(uglob=uglob_t[0])
    if len(damage_t) > 0:
        simu.Update(damage=damage_t[0])

    # Trace la première figure
    fig, ax, cb = Affichage.Plot_Result(simu, option,
    affichageMaillage=affichageMaillage, deformation=deformation, facteurDef=facteurDef)
    
    # Donne le lien vers ffmpeg.exe
    # ffmpegpath = "F:\\Pro\\ffmpeg\\bin\\ffmpeg.exe"
    ffmpegpath = "D:\\Soft\\ffmpeg\\bin\\ffmpeg.exe"
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath

    writer = animation.FFMpegWriter(fps=30)
    with writer.saving(fig, filename, 200):
    
        tic = TicTac()
        for t, damage in enumerate(damage_t):
            simu.Update(damage=damage)

            cb.remove()
            
            fig, ax, cb = Affichage.Plot_Result(simu, option, oldfig=fig, oldax=ax,
            deformation=False, affichageMaillage=False, facteurDef=4, valeursAuxNoeuds=True)

            title = ax.get_title()
            ax.set_title(f'{title} : {t+1}/{N}')

            plt.pause(0.00001)

            writer.grab_frame()

            

            tf = tic.Tac("Post Traitement","Plot", False)
            print(f'Plot {ax.get_title()} in {np.round(tf,3)}', end='\r')
    
# =========================================== Paraview ==================================================

def SaveParaview(simu: Simu, filename: str,nodesField=["displacement","Stress"], elementsField=["Stress","Strain"]):
    """Creer le .vtu qui peut être lu sur paraview
    """
    
    if not simu.VerificationOption("displacement"):
        return
    
    # resultats_e=["Svm","Evm"]
    options = nodesField+elementsField
    # options = np.array([options_n, options_e], dtype=str).reshape(-1)
    for option in options:
        if not simu.VerificationOption(option):
            return
    
    tic = TicTac()

    filename = filename+".vtu"

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
        "TETRA4" : 10
    } # regarder vtkelemtype

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
        
        file.write('<?xml version="1.0" ?>\n')
        
        file.write(f'<VTKFile type="UnstructuredGrid" version="0.1" byte_order="{endian_paraview}">\n')

        file.write('\t<UnstructuredGrid>\n')
        file.write(f'\t\t<Piece NumberOfPoints="{Nn}" NumberOfCells="{Ne}">\n')

        # Valeurs aux noeuds
        file.write('\t\t\t<PointData scalars="scalar"> \n')
        offset=0
        list_valeurs_n=[]
        for resultat_n in nodesField:

            valeurs_n = simu.GetResultat(resultat_n, valeursAuxNoeuds=True).reshape(-1)
            list_valeurs_n.append(valeurs_n)

            nombreDeComposantes = int(valeurs_n.size/Nn) # 1 ou 3
            file.write(f'\t\t\t\t<DataArray type="Float32" Name="{resultat_n}" NumberOfComponents="{nombreDeComposantes}" format="appended" offset="{offset}" />\n')
            offset = CalcOffset(offset, valeurs_n.size)

        file.write('\t\t\t</PointData> \n')

        # Valeurs aux elements
        file.write('\t\t\t<CellData> \n')
        list_valeurs_e=[]
        for resultat_e in elementsField:

            valeurs_e = simu.GetResultat(resultat_e, valeursAuxNoeuds=False).reshape(-1)
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
    
    tParaview = tic.Tac("Paraview","SaveParaview", False)
    
    path = Dossier.GetPath(filename)
    vtuFile = str(filename).replace(path+'\\', '')

    return vtuFile


def MakePvd(filename: str, vtuFiles=[]):

    tic = TicTac()

    endian_paraview = 'LittleEndian' # 'LittleEndian' 'BigEndian'

    filename = filename+".pvd"

    with open(filename, "w") as file:

        file.write('<?xml version="1.0" ?>\n')

        file.write(f'<VTKFile type="Collection" version="0.1" byte_order="{endian_paraview}">\n')
        file.write('\t<Collection>\n')
        
        for t, vtuFile in enumerate(vtuFiles):
            file.write(f'\t\t<DataSet timestep="{t}" group="" part="1" file="{vtuFile}"/>\n')
        
        file.write('\t</Collection>\n')
        file.write('</VTKFile>\n')
    
    t = tic.Tac("Paraview","Paraview", False)

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

    