
from Simu import Simu
import Dossier
from Mesh import Mesh
from TicTac import TicTac
import numpy as np

def SaveParaview(simu: Simu, filename: str,nodesField=["deplacement","Stress"], elementsField=["Stress","Strain"]):
    """Creer le .vtu qui peut être lu sur paraview
    """
    
    if not simu.VerificationOptions("Uglob"):
        return
    
    # resultats_e=["Svm","Evm"]
    options = nodesField+elementsField
    # options = np.array([options_n, options_e], dtype=str).reshape(-1)
    for option in options:
        if not simu.VerificationOptions(option):
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
    
    tParaview = tic.Tac("Post Traitement","SaveParaview", True)

    path = Dossier.GetPath(filename)
    vtuFile = str(filename).replace(path+'\\', '')

    return vtuFile


def MakePvd(filename: str, vtuFiles=[]):

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

        if isinstance(valeur, np.ndarray):
            convert = valeur.tobytes()
        else:
            # convert = np.byte(valeur)
            convert = valeur.tobytes()
            # convert = sys
            # convert = bytes(valeur, 'uint32')
            # convert = valeur.to_bytes(valeur)
        
        file.write(convert)

    