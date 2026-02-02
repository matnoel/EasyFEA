# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module providing an interface with Graphics Library Transmission Format (GLTF) using pygltflib (https://pypi.org/project/pygltflib/)."""

from __future__ import annotations
from typing import TYPE_CHECKING, Union
from enum import Enum
import struct

import numpy as np


from ._requires import Create_requires_decorator

from . import Folder
from .MeshIO import Surface_reconstruction


if TYPE_CHECKING:
    from ..FEM._mesh import Mesh

try:
    from pygltflib import (
        GLTF2,
        Scene,
        Node,
        Mesh,
        Primitive,
        Buffer,
        BufferView,
        Accessor,
    )
    from pygltflib import Animation, AnimationSampler, AnimationChannel

    class Type(str, Enum):
        # https://github.com/KhronosGroup/glTF/blob/main/specification/2.0/Specification.adoc#3622-accessor-data-types
        SCALAR = "SCALAR"
        VEC2 = "VEC2"
        VEC3 = "VEC3"
        VEC4 = "VEC4"
        MAT2 = "MAT2"
        MAT3 = "MAT3"
        MAT4 = "MAT4"

    class Component(int, Enum):
        # https://github.com/KhronosGroup/glTF/blob/main/specification/2.0/Specification.adoc#3622-accessor-data-types
        # SIGNED_BYTE = 5120
        # UNSIGNED_BYTE = 5121
        SIGNED_SHORT = 5122
        UNSIGNED_SHORT = 5123
        UNSIGNED_INT = 5125
        FLOAT = 5126

    class StructFormat(str, Enum):
        # https://docs.python.org/3/library/struct.html#format-characters
        # SIGNED_BYTE = 5120
        # UNSIGNED_BYTE = 5121
        SIGNED_SHORT = "h"
        UNSIGNED_SHORT = "H"
        UNSIGNED_INT = "I"
        FLOAT = "f"

    class Data:

        __offset = 0
        __NbufferViews = 0
        __Naccessors = 0

        def __init__(
            self,
            data: Union[np.ndarray, list[np.ndarray]],
            count: int,
            type: Type,
            component: Component,
        ):
            self._data = data
            self._count = count
            self._type = type
            self._component = component

            # in each case:
            # - get buffer data
            # - get byte length
            # - get buffer views
            # - update offset
            # - get buffer views index

            if isinstance(data, np.ndarray):

                bufferData = self.__get_buffer_data(data)
                byteLength = len(bufferData)

                bufferViews = [
                    BufferView(
                        buffer=0, byteOffset=Data.__offset, byteLength=byteLength
                    )
                ]
                Data.__offset += byteLength

                self._bufferViews_index = [Data.__NbufferViews]

            elif isinstance(data, list):
                bufferData = b"".join(
                    self.__get_buffer_data(np.asarray(value)) for value in data
                )
                byteLength = len(self.__get_buffer_data(np.asarray(data[0])))

                Ndata = len(data)

                offsets = [Data.__offset + i * byteLength for i in range(Ndata + 1)]

                bufferViews = [
                    BufferView(buffer=0, byteOffset=offset, byteLength=byteLength)
                    for offset in offsets
                ]
                Data.__offset += offsets[-1]

                self._bufferViews_index = [
                    Data.__NbufferViews + i for i in range(Ndata)
                ]

            else:
                raise TypeError

            # update NbufferViews
            Data.__NbufferViews = self._bufferViews_index[-1] + 1

            self._bufferData = bufferData
            self._bufferViews: list[BufferView] = bufferViews

            # get accessors (linked to buffer views)
            self._accessors: list[Accessor] = [
                Accessor(
                    bufferView=bufferView,
                    componentType=self._component.value,
                    count=self._count,
                    type=self._type.value,
                )
                for bufferView in self._bufferViews_index
            ]
            # get accessors index
            self._accessors_index = [
                Data.__Naccessors + i for i in range(len(self._accessors))
            ]
            # update Naccessors
            Data.__Naccessors = self._accessors_index[-1] + 1

        def __get_buffer_data(self, data: np.ndarray):

            # return data.tobytes() does not work !
            # return data.reshape(self._count, -1).tobytes() does not work !

            structType = getattr(StructFormat, self._component.name).value

            assert isinstance(data, np.ndarray)

            if data.ndim == 1:
                bufferData = b"".join(struct.pack(structType, value) for value in data)
            elif data.ndim == 2:
                types = structType * data.shape[1]
                bufferData = b"".join(struct.pack(types, *value) for value in data)
            else:
                raise ValueError

            return bufferData

except ImportError:
    pass
requires_pygltflib = Create_requires_decorator("pygltflib")


@requires_pygltflib
def Save_mesh_to_glb(
    mesh: "Mesh",
    folder: str,
    filename: str = "mesh",
    list_displacementMatrix: list[np.ndarray] = [],
    fps: int = 30,
    useSurfaceReconstruction: bool = True,
) -> str:
    """Save the mesh to a glb file

    Parameters
    ----------
    mesh : Mesh
        The mesh
    folder : str
        The directory where the results will be saved.
    filename : str, optional
        The name of the solution file, by default "mesh"
    list_displacementMatrix : list[np.ndarray], optional
        List of displacement matrix, by default []
    fps : int, optional
        Frames per second, by default 30
    useSurfaceReconstruction : bool, optional
        Ensure that surfaces are facing outward, by default True

    Returns
    -------
    str
        The path to the created glb file.
    """

    assert mesh.dim >= 2

    if useSurfaceReconstruction:
        # ensure that surfaces are facing outward
        mesh = Surface_reconstruction(mesh)

    data_coord = Data(mesh.coord, mesh.Nn, Type.VEC3, Component.FLOAT)

    # get triangles connectivity
    triangles = np.concatenate(
        [
            groupElem.connect[:, groupElem.triangles].reshape(-1, 3)
            for groupElem in mesh.Get_list_groupElem(2)
        ],
        axis=0,
    )
    data_triangles = Data(
        triangles.ravel(), triangles.size, Type.SCALAR, Component.UNSIGNED_INT
    )

    list_data: list[Data] = [data_coord, data_triangles]

    numFrames = len(list_displacementMatrix)

    if numFrames > 0:
        numTargets = numFrames

        times = np.array([i / fps for i in range(numFrames)], dtype=float)
        data_times = Data(times, times.size, Type.SCALAR, Component.FLOAT)

        weightsValues = np.eye(numFrames, numTargets)
        weightsValues = np.concat((np.zeros((1, numTargets)), weightsValues), axis=0)
        data_weights = Data(
            weightsValues.ravel(), numFrames * numTargets, Type.SCALAR, Component.FLOAT
        )  # KEEP numFrames * numTargets
        data_list_displacementMatrix = Data(
            list_displacementMatrix, mesh.Nn, Type.VEC3, Component.FLOAT
        )

        list_data.extend([data_times, data_weights, data_list_displacementMatrix])

    # create gltf object
    gltf = GLTF2()
    bufferData = list_data[0]._bufferData
    for data in list_data[1:]:
        bufferData += data._bufferData
    gltf.buffers.append(Buffer(byteLength=len(bufferData)))
    gltf.set_binary_blob(bufferData)

    for data in list_data:
        gltf.bufferViews.extend(data._bufferViews)
        gltf.accessors.extend(data._accessors)

    targets = (
        [{"POSITION": index} for index in data_list_displacementMatrix._accessors_index]
        if numFrames > 0
        else []
    )

    # meshe
    gltf.meshes.append(
        Mesh(
            primitives=[
                Primitive(
                    attributes={"POSITION": data_coord._accessors_index[0]},
                    targets=targets,
                    indices=data_triangles._accessors_index[0],
                )
            ]
        )
    )

    # nodes + scence
    gltf.nodes.append(Node(mesh=0))
    gltf.scenes.append(Scene(nodes=[0]))
    gltf.scene = 0

    # animation

    if numFrames > 0:

        # animation objects
        sampler = AnimationSampler(
            input=data_times._accessors_index[0],  # times accessor index
            output=data_weights._accessors_index[0],  # weightValues accessor index
            interpolation="LINEAR",
        )

        channel = AnimationChannel(sampler=0, target={"node": 0, "path": "weights"})

        anim = Animation(samplers=[sampler], channels=[channel])
        gltf.animations.append(anim)

    # save
    filename = Folder.Join(folder, f"{filename}.glb")
    gltf.save_binary(filename)

    return filename
