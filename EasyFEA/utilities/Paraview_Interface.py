# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""This module allows you to save a simulation's results on Paraview (https://www.paraview.org/)."""

import numpy as np

# utilities
from . import Display, Folder, Tic
from .PyVista_Interface import DICT_VTK_INDEXES, DICT_CELL_TYPES

# ----------------------------------------------
# Paraview
# ----------------------------------------------
def Make_Paraview(simu, folder: str, N=200, details=False, nodesField=[], elementsField=[]):
    """Generates the paraview (*.pvd and *.pvu files).

    Parameters
    ----------
    simulation : _Simu
        Simulation
    folder: str
        folder in which we will create the Paraview folder
    N : int, optional
        Maximal number of iterations displayed, by default 200
    details: bool, optional
        details of nodesField and elementsField used in the *.vtu
    nodesField: list, optional
        Additional nodesField, by default []
    elementsField: list, optional
        Additional elementsField, by default []
    """
    print('\n')

    vtuFiles: list[str] = []

    simu = Display._Init_obj(simu)[0]

    results = simu.results

    Niter = len(results)
    step = np.max([1, Niter//N])
    iterations: np.ndarray = np.arange(0, Niter, step)

    folder = Folder.Join(folder,"Paraview")

    if not Folder.Exists(folder):
        Folder.os.makedirs(folder)

    times = []
    tic = Tic()

    additionalNodesField = nodesField
    additionalElementsField = elementsField

    nodesField, elementsField = simu.Results_nodesField_elementsField(details)

    [nodesField.append(n) for n in additionalNodesField
     if simu._Results_Check_Available(n) and n not in nodesField]
    [elementsField.append(e) for e in additionalElementsField
     if simu._Results_Check_Available(e) and e not in elementsField]

    if len(nodesField) == 0 and len(elementsField) == 0:
        Display.MyPrintError("The simulation has no solution fields to display in paraview.")

    # activate the first iteration
    simu.Set_Iter(0, resetAll=True)

    for i, iter in enumerate(iterations):

        f = Folder.Join(folder,f'solution_{iter}.vtu')

        __Make_vtu(simu, iter, f, nodesField=nodesField, elementsField=elementsField)
        
        # vtuFiles.append(vtuFile)
        vtuFiles.append(f'solution_{iter}.vtu')
        
        times.append(tic.Tac("Paraview","Make vtu", False))

        rmTime = Tic.Get_Remaining_Time(i, iterations.size-1, times[-1])

        Display.MyPrint(f"SaveParaview {i}/{iterations.size-1} {rmTime}     ", end='\r')
    
    print('\n')

    tic = Tic()

    filenamePvd = Folder.os.path.join(folder,"simulation")    
    __Make_pvd(filenamePvd, vtuFiles)

    tic.Tac("Paraview","Make pvd", False)

# ----------------------------------------------
# Functions
# ----------------------------------------------
def __Make_vtu(simu, iter: int, filename: str, nodesField: list[str], elementsField: list[str]):
    """Generates the *.vtu files in binary format.
    """

    simu = Display._Init_obj(simu)[0]

    options = nodesField+elementsField
   
    simu.Set_Iter(iter)

    # check the compatibility of the available results
    for option in options:
        resultat = simu.Result(option)
        if not (isinstance(resultat, np.ndarray) or isinstance(resultat, list)):
            return

    connect = simu.mesh.connect
    
    # reorder gmsh idx to vtk indexes
    if simu.mesh.elemType in DICT_VTK_INDEXES.keys():
        vtkIndexes = DICT_VTK_INDEXES[simu.mesh.elemType]
    else:
        vtkIndexes = np.arange(simu.mesh.nPe)    
    connect = connect[:, vtkIndexes]

    coord = simu.mesh.coord
    Ne = simu.mesh.Ne
    Nn = simu.mesh.Nn
    nPe = simu.mesh.groupElem.nPe    

    # TODO: The display of PRISM18 elements does not seem to work correctly in ParaView.
    # The indices and VTK type seem to be correct because the display in PyVista works fine.
    # The values on the faces seem strange.
    paraviewType = DICT_CELL_TYPES[simu.mesh.elemType][1]
    
    types = np.ones(Ne, dtype=int)*paraviewType

    # coordinates as a vector (e.g (x1, y1, y2,..., xn, yn, yn))
    nodes = coord.ravel()
    # connect as a vector (e.g (n1^1, n2^1, n3^1, ..., n1^e, n2^e, n3^e))
    connectivity = connect.ravel()

    offsets = np.arange(nPe,nPe*Ne+1,nPe, dtype=np.int32)-3

    endian_paraview = 'LittleEndian' # 'LittleEndian' 'BigEndian'

    bitSize = 4 # bit size
    CalcOffset = lambda offset, size: offset + bitSize + (bitSize*size)

    with open(filename, "w") as file:
        
        # Specify the mesh
        file.write('<?pickle version="1.0" ?>\n')
        
        file.write(f'<VTKFile type="UnstructuredGrid" version="0.1" byte_order="{endian_paraview}">\n')

        file.write('\t<UnstructuredGrid>\n')
        file.write(f'\t\t<Piece NumberOfPoints="{Nn}" NumberOfCells="{Ne}">\n')

        # Specify the nodes values
        file.write('\t\t\t<PointData scalars="scalar"> \n')
        offset=0
        list_values_n: list[np.ndarray] = [] # list of nodes values
        for result_n in nodesField:

            values_n = simu.Result(result_n, nodeValues=True).ravel()
            list_values_n.append(values_n)

            dof_n = values_n.size // Nn # 1 ou 3
            if result_n == "displacement_matrix": result_n="displacement"
            file.write(f'\t\t\t\t<DataArray type="Float32" Name="{result_n}" NumberOfComponents="{dof_n}" format="appended" offset="{offset}" />\n')
            offset = CalcOffset(offset, values_n.size)

        file.write('\t\t\t</PointData> \n')

        # Specify the elements values
        file.write('\t\t\t<CellData> \n')
        list_values_e: list[np.ndarray] = []
        for result_e in elementsField:

            values_e = simu.Result(result_e, nodeValues=False).ravel()
            list_values_e.append(values_e)

            dof_e = values_e.size // Ne
            
            file.write(f'\t\t\t\t<DataArray type="Float32" Name="{result_e}" NumberOfComponents="{dof_e}" format="appended" offset="{offset}" />\n')
            offset = CalcOffset(offset, values_e.size)
        
        file.write('\t\t\t</CellData> \n')

        # Points / Nodes coordinates
        file.write('\t\t\t<Points>\n')
        file.write(f'\t\t\t\t<DataArray type="Float32" NumberOfComponents="3" format="appended" offset="{offset}" />\n')
        offset = CalcOffset(offset, nodes.size)
        file.write('\t\t\t</Points>\n')

        # Elements -> Connectivity matrix
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
        for values_n in list_values_n:
            __WriteBinary(bitSize*(values_n.size), "uint32", file)
            __WriteBinary(values_n, "float32", file)

        # Elements values
        for values_e in list_values_e:                
            __WriteBinary(bitSize*(values_e.size), "uint32", file)
            __WriteBinary(values_e, "float32", file)

        # Nodes
        __WriteBinary(bitSize*(nodes.size), "uint32", file)
        __WriteBinary(nodes, "float32", file)

        # Connectivity            
        __WriteBinary(bitSize*(connectivity.size), "uint32", file)
        __WriteBinary(connectivity, "int32", file)

        # Offsets
        __WriteBinary(bitSize*Ne, "uint32", file)
        __WriteBinary(offsets+3, "int32", file)

        # Element types
        __WriteBinary(types.size, "uint32", file)
        __WriteBinary(types, "int8", file)

    with open(filename, "a") as file:

        # End of adding data
        file.write('\n\t</AppendedData>\n')

        # End of vtk
        file.write('</VTKFile> \n')
    
    path = Folder.Dir(filename)
    vtuFile = str(filename).replace(path+'\\', '')

    return vtuFile

def __Make_pvd(filename: str, vtuFiles=[]):
    """Makes *.pvd file to link the *.vtu files."""

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

def __WriteBinary(value, type: str, file):
    """Converts value (int of array) to Binary"""

    if type not in ['uint32','float32','int32','int8']:
        raise Exception("Type not implemented")

    if type == "uint32":
        value = np.uint32(value)
    elif type == "float32":
        value = np.float32(value)
    elif type == "int32":
        value = np.int32(value)
    elif type == "int8":
        value = np.int8(value)

    convert = value.tobytes()
    
    file.write(convert)