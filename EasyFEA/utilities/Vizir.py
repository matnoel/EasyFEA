# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Module providing functions used to save FEM-solutions for vizir (https://pyamg.saclay.inria.fr/vizir4.html)."""

from typing import Union
import numpy as np
import io

from ..utilities import Folder, MeshIO
from ..simulations._simu import _Simu, _Init_obj, _Get_values
from ..fem._group_elems import _GroupElem
from ..fem._mesh import Mesh
from ..geoms._utils import (
    _Get_BaryCentric_Coordinates_In_Triangle,
    _Get_BaryCentric_Coordinates_In_Tetrahedron,
    _Get_BaryCentric_Coordinates_In_Segment
)

def __Get_vizir_solution_key(groupElem: _GroupElem) -> str:

    elemType = groupElem.elemType

    if elemType.startswith("SEG"):
        keyword = "HOSolAtEdgesP"
    elif elemType.startswith("HEXA"):
        keyword = "HOSolAtHexahedraQ"
    elif elemType.startswith("PRISM"):
        keyword = "HOSolAtPrismsP"
    elif elemType.startswith("QUAD"):
        keyword = "HOSolAtQuadrilateralsQ"
    elif elemType.startswith("TETRA"):
        keyword = "HOSolAtTetrahedraP"
    elif elemType.startswith("TRI"):
        keyword = "HOSolAtTrianglesP"
    else:
        raise TypeError("Unknown element type")

    return keyword

def _Get_BaryCentric_Coordinates(groupElem: _GroupElem) -> np.ndarray:

    elemType = groupElem.elemType
    local_coords = groupElem.Get_Local_Coords()
    vertices_coords = local_coords[:groupElem.nbCorners]

    if elemType.startswith("SEG"):
        coordinates = _Get_BaryCentric_Coordinates_In_Segment(vertices_coords, local_coords)
    # elif elemType.startswith("PRISM"):
    #     coordinates = None
    elif elemType.startswith("TETRA"):
        coordinates = _Get_BaryCentric_Coordinates_In_Tetrahedron(vertices_coords, local_coords)
    elif elemType.startswith("TRI"):
        coordinates = _Get_BaryCentric_Coordinates_In_Triangle(vertices_coords, local_coords)
    else:
        raise TypeError("Unknown element type")
    
    return coordinates

def __Get_NodesPositions(groupElem: _GroupElem) -> np.ndarray:

    elemType = groupElem.elemType
    local_coords = groupElem.Get_Local_Coords()

    if elemType.startswith(("SEG", "PRISM", "TETRA", "TRI")):
        nodes_positions = _Get_BaryCentric_Coordinates(groupElem)
    else:
        nodes_positions = local_coords.astype(float)
        nodes_positions -= groupElem.origin
        nodes_positions /= nodes_positions.max()

    return nodes_positions

def __Write_RefGeomElt(file: io.TextIOWrapper, groupElem: _GroupElem, solutionOrder: int) -> None:

    # get keyword
    keyword = __Get_vizir_solution_key(groupElem)

    # write ref geom element
    file.write(f"{keyword}{solutionOrder}NodesPositions\n{groupElem.nPe}\n")
    nodesPositions = __Get_NodesPositions(groupElem)
    np.savetxt(file, nodesPositions)

def __Write_Solution(file: io.TextIOWrapper, groupElem: _GroupElem, dofsValues: np.ndarray, dof_n: int , solutionOrder: int) -> None:
    
    # get dofsValues as a (Ne, nPe, dof_n) array
    assembly_e = groupElem.Get_assembly_e(dof_n)
    dofsValues_e = dofsValues[assembly_e].reshape(groupElem.Ne, groupElem.nPe, -1)

    if dof_n == 1:
        # scalar case
        solutionDim = 1
    elif 1 < dof_n <= 3:
        # vector case
        solutionDim = 2
        zeros_e = np.zeros((groupElem.Ne, groupElem.nPe, 1))
        for _ in range(3 - dof_n):
            dofsValues_e = np.concatenate((dofsValues_e, zeros_e), axis=-1)
    else:
        raise Exception("Symmetric/non-symmetric matrices are not yet implemented.")
    dofsValues_e = dofsValues_e.reshape(groupElem.Ne, -1)

    # write solution
    keyword = __Get_vizir_solution_key(groupElem)
    file.write(f"\n{keyword}{solutionOrder}\n{groupElem.Ne}\n")
    file.write(f"1 {solutionDim}\n")
    file.write(f"{groupElem.order} {groupElem.nPe}\n")
    
    # write solution array
    np.savetxt(file, dofsValues_e)
    file.write("\n")

def Save_result(obj: Union[_Simu, Mesh], result: Union[str,np.ndarray], folder: str, filename: str, deformFactor:float=1.0, solutionOrder:int=None):

    # get mesh and simu fro obj
    simu, mesh, coord, _ = _Init_obj(obj, deformFactor)
    mesh.coordGlob = coord    
    
    # save the mesh in Medit format
    meshFile = MeshIO.EasyFEA_to_Medit(mesh, folder, "mesh")

    # initsolution name
    solutionFile = Folder.Join(folder, f"{filename}.sol", mkdir=True)

    # get solution
    # assume nodesValues is True
    dofsValues = _Get_values(simu, mesh, result, nodeValues=True)
    dof_n = dofsValues.size // mesh.Nn
    
    # get solution order
    if solutionOrder is None:
        solutionOrder = mesh.groupElem.order

    list_groupElem = mesh.Get_list_groupElem()
    list_groupElem.extend(mesh.Get_list_groupElem(mesh.dim - 1))

    with open(solutionFile, "w") as f:

        # write first lines
        f.write("MeshVersionFormatted 2\n")
        f.write(f"Dimension 3\n\n") # the mesh is always in a 3d space

        for groupElem in list_groupElem:

            __Write_RefGeomElt(f, groupElem, solutionOrder)

            __Write_Solution(f, groupElem, dofsValues, dof_n, solutionOrder)
        
        

    print(f"\nvizir4 -in {meshFile} -sol {solutionFile}\n")

    pass