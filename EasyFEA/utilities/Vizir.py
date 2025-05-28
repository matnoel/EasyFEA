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

def __Write_Solution(file: io.TextIOWrapper, groupElem: _GroupElem, dofsValues: np.ndarray, dof_n: int , resultOrder: int) -> None:
    
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
    file.write(f"\n{keyword}{resultOrder}\n{groupElem.Ne}\n")
    file.write(f"1 {solutionDim}\n")
    file.write(f"{groupElem.order} {groupElem.nPe}\n")
    
    # write solution array
    np.savetxt(file, dofsValues_e)
    file.write("\n")

def __Write_solution_file(simu: Union[_Simu, None], mesh: Mesh, result: Union[str, np.ndarray], resultOrder: int, folder: str, filename:str) -> str:

    # get solution
    # assume nodesValues is True
    dofsValues = _Get_values(simu, mesh, result, nodeValues=True)
    dof_n = dofsValues.size // mesh.Nn

    list_groupElem = mesh.Get_list_groupElem()
    list_groupElem.extend(mesh.Get_list_groupElem(mesh.dim - 1))

    # init solution file
    solutionFile = Folder.Join(folder, f"{filename}.sol", mkdir=True)    

    with open(solutionFile, "w") as f:

        # write first lines
        f.write("MeshVersionFormatted 2\n")
        f.write(f"Dimension 3\n\n") # the mesh is always in a 3d space

        for groupElem in list_groupElem:

            __Write_RefGeomElt(f, groupElem, resultOrder)

            __Write_Solution(f, groupElem, dofsValues, dof_n, resultOrder)

        f.write("End\n")

    return solutionFile
    

def Save_result(obj: Union[_Simu, Mesh], result: Union[str,np.ndarray], folder: str, filename: str, deformFactor:float=1.0, resultOrder:int=None):

    # get mesh and simu fro obj
    simu, mesh, coord, _ = _Init_obj(obj, deformFactor)
    mesh.coordGlob = coord    
    
    # save the mesh in Medit format
    meshFile = MeshIO.EasyFEA_to_Medit(mesh, folder, "mesh")
    
    # get solution order
    if resultOrder is None:
        resultOrder = mesh.groupElem.order

    if simu is not None:
        Niter = simu.Niter

        solsFile = Folder.Join(folder, filename+".sols", mkdir=True)

        meshFiles: list[str] = []
        solutionFiles: list[str] = []

        with open(solsFile, "w") as f:

            for iter in range(Niter):
                simu.Set_Iter(iter)
                filename_with_iter = f"{filename}.{iter}"
                solutionFile = __Write_solution_file(simu, mesh, result, resultOrder, folder, filename_with_iter)

                meshFiles.append(meshFile)
                solutionFiles.append(solutionFile)
                
                f.write(f"{solutionFile}\n")
        

        vizirMovie = Folder.Join(folder, f"{filename}.movie", mkdir=True)
        
        with open(vizirMovie, "w") as f:

            for meshFile, solutionFile in zip(meshFiles, solutionFiles):

                f.write(f"{meshFile}\t{solutionFile}\n")
        
            # within the folder
            # vizir4 -movie filename.movie

        return f"vizir4 -in {meshFile} -sols {solsFile}"


    else:
        solutionFile = __Write_solution_file(simu, mesh, result, resultOrder, folder, filename)
    
        return f"vizir4 -in {meshFile} -sol {solutionFile}"
    
# def Make_movie(folder: str, )