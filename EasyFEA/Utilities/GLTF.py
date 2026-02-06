# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module providing an interface with Graphics Library Transmission Format (GLTF) using pygltflib (https://pypi.org/project/pygltflib/)."""
# https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#geometry

from __future__ import annotations
from typing import TYPE_CHECKING, Union
from enum import Enum
import struct

import numpy as np

from ._requires import Create_requires_decorator
from ..Simulations._simu import _Init_obj

from . import Folder, Display
from .MeshIO import Surface_reconstruction


if TYPE_CHECKING:
    from ..FEM._mesh import Mesh
    from ..Simulations import _Simu

try:
    import pygltflib

    class Type(str, Enum):
        # https://github.com/KhronosGroup/glTF/blob/main/specification/2.0/Specification.adoc#3622-accessor-data-types
        SCALAR = pygltflib.SCALAR
        VEC2 = pygltflib.VEC2
        VEC3 = pygltflib.VEC3
        VEC4 = pygltflib.VEC4
        MAT2 = pygltflib.MAT2
        MAT3 = pygltflib.MAT3
        MAT4 = pygltflib.MAT4

    class Component(int, Enum):
        # https://github.com/KhronosGroup/glTF/blob/main/specification/2.0/Specification.adoc#3622-accessor-data-types
        SIGNED_BYTE = 5120
        UNSIGNED_BYTE = 5121
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

    class Target(int, Enum):
        ARRAY_BUFFER = pygltflib.ARRAY_BUFFER
        ELEMENT_ARRAY_BUFFER = pygltflib.ELEMENT_ARRAY_BUFFER

except ImportError:
    pass
requires_pygltflib = Create_requires_decorator("pygltflib")


# --------------------------------------------
# Utilities for GLTF and USD
# --------------------------------------------


def _get_list_nodesValues(list_nodesValues_n: list[np.ndarray]):

    assert isinstance(list_nodesValues_n, list)
    assert isinstance(list_nodesValues_n[0], np.ndarray)

    ndim = list_nodesValues_n[0].ndim
    if ndim == 1:
        list_nodesValues = list_nodesValues_n
    elif ndim == 2:
        list_nodesValues = [
            np.linalg.norm(nodesValues_n, axis=1)
            for nodesValues_n in list_nodesValues_n
        ]
    else:
        raise ValueError(f"Must have 1 or 2 dimensions, got {ndim}.")

    return list_nodesValues


def _get_colors_for_values(
    values: np.ndarray, vMax: float = None, vMin: float = None
) -> np.ndarray:

    assert isinstance(values, np.ndarray)
    assert values.ndim == 1

    # Normalize values between 0 and 1
    vMin = values.min() if vMin is None else vMin
    vMax = values.max() if vMax is None else vMax
    if vMax > vMin:
        normalizedValues = (values - vMin) / (vMax - vMin)
    else:
        normalizedValues = np.zeros_like(values)

    colors = np.zeros((values.size, 3))
    colors[:, 0] = normalizedValues  # red
    colors[:, 1] = 1 - np.abs(normalizedValues - 0.5) * 2  # green
    colors[:, 2] = 1 - normalizedValues  # blue

    return colors


# --------------------------------------------
# Classes and functions
# --------------------------------------------


class Data:
    """Data class used to generate gltf data and save offset, bufferViews and accesors."""

    __offset = 0
    __NbufferViews = 0
    __Naccessors = 0

    __list_data: list[Data] = []

    def __init__(
        self,
        data: Union[np.ndarray, list[np.ndarray]],
        count: int,
        type: Type,
        component: Component,
        target: Target = None,
    ):
        """Generates gltf data.

        Parameters
        ----------
        data : Union[np.ndarray, list[np.ndarray]]
            data to save
        count : int
            Count/size of data.
        type : Type
            Data type (e.g. SCALAR, VEC2, VEC3, MAT2, ...)
        component : Component
            Component type (e.g. UNSIGNED_INT, FLOAT, ...)
        target : Target, optional
            target (e.g. ARRAY_BUFFER or ELEMENT_ARRAY_BUFFER), by default None
        """
        self._data = data
        self._count = count
        self._type = type
        self._component = component
        self._target = target

        Data.__list_data.append(self)

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
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=Data.__offset,
                    byteLength=byteLength,
                    target=target,
                )
            ]
            Data.__offset += byteLength

            self._bufferViews_index = [Data.__NbufferViews]

            list_min, list_max = list(map(list, zip(self.__get_list_min_max(data))))

        elif isinstance(data, list):
            bufferData = b"".join(
                self.__get_buffer_data(np.asarray(value)) for value in data
            )
            byteLength = len(self.__get_buffer_data(np.asarray(data[0])))

            Ndata = len(data)

            offsets = [Data.__offset + i * byteLength for i in range(Ndata + 1)]

            bufferViews = [
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=offset,
                    byteLength=byteLength,
                    target=target,
                )
                for offset in offsets
            ]
            Data.__offset += offsets[-1]

            self._bufferViews_index = [Data.__NbufferViews + i for i in range(Ndata)]

            list_min, list_max = list(
                map(list, zip(*[self.__get_list_min_max(values) for values in data]))
            )

        else:
            raise TypeError

        # update NbufferViews
        Data.__NbufferViews = self._bufferViews_index[-1] + 1

        self._bufferData = bufferData
        self._bufferViews: list[pygltflib.BufferView] = bufferViews

        # get accessors (linked to buffer views)
        self._accessors: list[pygltflib.Accessor] = [
            pygltflib.Accessor(
                bufferView=bufferView,
                componentType=self._component.value,
                count=self._count,
                type=self._type.value,
                max=list_max[i],
                min=list_min[i],
            )
            for i, bufferView in enumerate(self._bufferViews_index)
        ]
        # get accessors index
        self._accessors_index = [
            Data.__Naccessors + i for i in range(len(self._accessors))
        ]
        # update Naccessors
        Data.__Naccessors = self._accessors_index[-1] + 1

    @classmethod
    def _Get_list_data(cls) -> list[Data]:
        """Returns list of Data."""
        return cls.__list_data

    @classmethod
    def _Clear_list_data(cls) -> None:
        """Resets list of Data."""
        cls.__list_data = []

    def __get_list_min_max(self, data: np.ndarray):
        """Returns list_min, list_max"""

        assert isinstance(data, np.ndarray)

        usedType = float if self._component is Component.FLOAT else int

        if data.ndim == 1:
            list_max = [usedType(data.max(0))]
            list_min = [usedType(data.min(0))]
        elif data.ndim == 2:
            list_max = [usedType(value) for value in data.max(0)]
            list_min = [usedType(value) for value in data.min(0)]
        else:
            raise ValueError

        return list_min, list_max

    def __get_buffer_data(self, data: np.ndarray):
        """Returns buffer data with struct.pack."""

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


@requires_pygltflib
def Save_simu(
    simu: "_Simu",
    result: str,
    folder: str,
    filename: str = None,
    N: int = 200,
    deformFactor=1.0,
    fps: int = 30,
) -> None:
    """Saves the simulation as glb file.

    Parameters
    ----------
    simu : _Simu
        simulation
    result : str
        result that you want to plot
    folder : str
        folder where you want to save the video
    filename : str, optional
        filename of the glb file, by default result.glb
    N : int, optional
        Maximal number of iterations displayed, by default 200
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 0.0
    fps : int, optional
        Frames per second, by default 30

    Returns
    -------
    str
        The path to the created glb file.
    """

    simu, mesh, _, _ = _Init_obj(simu)  # type: ignore [assignment]

    if simu is None:
        Display.MyPrintError("Must give a simulation.")
        return

    Niter = len(simu.results)
    N = np.min([Niter, N])
    iterations = np.linspace(0, Niter - 1, N, endpoint=True, dtype=int)

    # activates the first iteration
    simu.Set_Iter(0, resetAll=True)

    # init list
    list_displacementMatrix: list[np.ndarray] = [None] * N
    list_nodesValues_n: list[np.ndarray] = [None] * N

    # get values
    for i, iter in enumerate(iterations):
        simu.Set_Iter(iter)
        list_displacementMatrix[i] = deformFactor * simu.Results_displacement_matrix()
        list_nodesValues_n[i] = simu.Result(result)

    if filename is None:
        filename = result

    return Save_mesh(
        mesh=mesh,
        folder=folder,
        filename=filename,
        list_displacementMatrix=list_displacementMatrix,
        list_nodesValues_n=list_nodesValues_n,
        useSurfaceReconstruction=True,
        fps=fps,
    )


@requires_pygltflib
def Save_mesh(
    mesh: "Mesh",
    folder: str,
    filename: str = "mesh",
    list_displacementMatrix: list[np.ndarray] = [],
    list_nodesValues_n: list[np.ndarray] = [],
    useSurfaceReconstruction: bool = True,
    fps: int = 30,
) -> str:
    """Saves the mesh as glb file.

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
    useSurfaceReconstruction : bool, optional
        Ensure that surfaces are facing outward, by default True
    fps : int, optional
        Frames per second, by default 30

    Returns
    -------
    str
        The path to the created glb file.
    """

    assert mesh.dim >= 2

    if useSurfaceReconstruction and mesh.dim == 3:
        # ensure that surfaces are facing outward
        mesh = Surface_reconstruction(mesh)

    if len(list_displacementMatrix) == 0:
        list_displacementMatrix = [np.zeros_like(mesh.coord)]

    # check list length
    Ndisplacement = len(list_displacementMatrix)
    Nvalues = len(list_nodesValues_n)
    if Ndisplacement > 0 and Nvalues > 0:
        assert Ndisplacement == Nvalues, (
            f"list_displacementMatrix and list_nodesValues_n must have the same length. "
            f"Got {Ndisplacement} and {Nvalues}"
        )

    # get list of nodes values
    if Nvalues > 0:
        list_nodesValues = _get_list_nodesValues(list_nodesValues_n)
        vMax, vMin = np.max(list_nodesValues), np.min(list_nodesValues)

    # get default colors
    defaultColors = np.ones((mesh.Nn, 3)) * 0.5  # Default grey (normalized 0-1)
    data_defaultColors = Data(
        defaultColors, mesh.Nn, Type.VEC3, Component.FLOAT, Target.ARRAY_BUFFER
    )

    # init list of gltf mesh
    list_gltfMeshes: list[pygltflib.Mesh] = [None] * Ndisplacement

    # animation
    # https://gltf-tutorial.readthedocs.io/en/latest/gltfTutorial_007_Animations/#step
    times = np.array([i / fps for i in range(Ndisplacement)], dtype=float)
    data_times = Data(times, times.size, Type.SCALAR, Component.FLOAT)
    samplers: list[pygltflib.Sampler] = []
    channels: list[pygltflib.AnimationChannel] = []

    for i, displacementMatrix in enumerate(list_displacementMatrix):

        # get triangles connectivity
        if i == 0:
            triangles = np.concatenate(
                [
                    groupElem.connect[:, groupElem.triangles].reshape(-1, 3)
                    for groupElem in mesh.Get_list_groupElem(2)
                ],
                axis=0,
            )
            data_triangles = Data(
                triangles.ravel(),
                triangles.size,
                Type.SCALAR,
                Component.UNSIGNED_INT,
                Target.ELEMENT_ARRAY_BUFFER,
            )

        # get mesh coordinates
        coords = mesh.coord + displacementMatrix
        data_coord0 = Data(
            coords,
            mesh.Nn,
            Type.VEC3,
            Component.FLOAT,
            Target.ARRAY_BUFFER,
        )

        # get colors
        if Nvalues > 0:
            colors = _get_colors_for_values(list_nodesValues[i], vMax=vMax, vMin=vMin)
            data_colors = Data(
                colors, mesh.Nn, Type.VEC3, Component.FLOAT, Target.ARRAY_BUFFER
            )
        else:
            data_colors = data_defaultColors

        # mesh
        gltfMesh = pygltflib.Mesh(
            primitives=[
                pygltflib.Primitive(
                    attributes={
                        "POSITION": data_coord0._accessors_index[0],
                        "COLOR_0": data_colors._accessors_index[0],
                    },
                    indices=data_triangles._accessors_index[0],
                )
            ]
        )
        list_gltfMeshes[i] = gltfMesh

        # animation
        option = "scale"
        scales = np.zeros((Ndisplacement, 3), dtype=float)
        scales[i] = 1.0

        # option = "translation"
        # scales = np.ones((Ndisplacement, 3), dtype=float) * 4
        # scales[i] = 0.0

        data_scales = Data(scales, Ndisplacement, Type.VEC3, Component.FLOAT)

        # option = "weights" # raise ANIMATION_CHANNEL_TARGET_NODE_WEIGHTS_NO_MORPHS error
        # scales = np.eye(Ndisplacement)
        # data_scales = Data(scales.ravel(), scales.size, Type.SCALAR, Component.FLOAT)

        samplers.append(
            pygltflib.AnimationSampler(
                input=data_times._accessors_index[0],
                output=data_scales._accessors_index[0],
                interpolation=pygltflib.ANIM_STEP,
                # interpolation=pygltflib.ANIM_LINEAR,
                # interpolation=pygltflib.ANIM_CUBICSPLINE,
            )
        )

        channels.append(
            pygltflib.AnimationChannel(sampler=i, target={"node": i, "path": option})
        )

    # create gltf object
    gltf = pygltflib.GLTF2()
    bufferData = b"".join(data._bufferData for data in Data._Get_list_data())
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(bufferData)))
    gltf.set_binary_blob(bufferData)

    # add buffer views and accessors
    for data in Data._Get_list_data():
        gltf.bufferViews.extend(data._bufferViews)
        gltf.accessors.extend(data._accessors)

    # add mesh
    gltf.meshes = list_gltfMeshes

    # add nodes + scence
    gltf.nodes.extend([pygltflib.Node(mesh=i) for i in range(Ndisplacement)])
    gltf.scenes.append(pygltflib.Scene(nodes=list(range(Ndisplacement))))
    gltf.scene = 0

    # animation
    anim = pygltflib.Animation(samplers=samplers, channels=channels)
    gltf.animations.append(anim)

    # save
    filename = Folder.Join(folder, f"{filename}.glb")
    gltf.save_binary(filename)

    return filename
