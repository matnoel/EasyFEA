# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module providing an interface with Graphics Library Transmission Format (GLTF) using pygltflib (https://pypi.org/project/pygltflib/)."""
# https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#geometry

from __future__ import annotations
from typing import TYPE_CHECKING, Union
from enum import Enum
import struct
import textwrap
from pathlib import Path
import http.server
import threading
import webbrowser

import numpy as np

from ._requires import Create_requires_decorator
from ..Simulations._simu import _Init_obj

from . import Folder, Display


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
        raise NotImplementedError(f"Not implemented for ndim={ndim}.")

    return list_nodesValues


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
    def _Clear_All(cls) -> None:
        """Resets all class data."""
        cls.__offset = 0
        cls.__NbufferViews = 0
        cls.__Naccessors = 0
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
    results: str,
    folder: str,
    N: int = 200,
    deformFactor=1.0,
    plotMesh=False,
    fps: int = 30,
    openWebBrowser=False,
) -> None:
    """Saves the simulation as glb file.

    Parameters
    ----------
    simu : _Simu
        simulation
    results : list[str]
        result that you want to plot
    folder : str
        folder where you want to save the video
    N : int, optional
        Maximal number of iterations displayed, by default 200
    deformFactor : float, optional
        Factor used to display the deformed solution (0 means no deformations), default 1.0
    plotMesh : bool, optional
        displays mesh, by default False
    fps : int, optional
        Frames per second, by default 30
    openWebBrowser : bool, optional
        open in the generated files in your web browser, by default False

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

    for result in results:

        # init list
        list_displacementMatrix: list[np.ndarray] = [None] * N
        list_nodesValues_n: list[np.ndarray] = [None] * N
        list_mesh: list[Mesh] = [None] * N

        # activates the first iteration
        simu.Set_Iter(0, resetAll=True)

        # get values
        for i, iter in enumerate(iterations):
            simu.Set_Iter(iter)
            list_mesh[i] = simu.mesh
            list_displacementMatrix[i] = (
                deformFactor * simu.Results_displacement_matrix()
            )
            list_nodesValues_n[i] = simu.Result(result).reshape(simu.mesh.Nn, -1)

        dof_n = list_nodesValues_n[0].shape[1]

        if dof_n == 1:
            unknowns = [""]
        else:
            unknowns = [f"_{d}" for d in range(dof_n)]

        # save each dofs
        for d in range(dof_n):
            Save_mesh(
                mesh=list_mesh,
                folder=folder,
                filename=f"{result}{unknowns[d]}",
                list_displacementMatrix=list_displacementMatrix,
                list_nodesValues_n=[
                    nodesValues_n[:, d] for nodesValues_n in list_nodesValues_n
                ],
                plotMesh=plotMesh,
                fps=fps,
            )

        if dof_n > 1:
            Save_mesh(
                mesh=list_mesh,
                folder=folder,
                filename=f"{result}_norm",
                list_displacementMatrix=list_displacementMatrix,
                list_nodesValues_n=list_nodesValues_n,
                plotMesh=plotMesh,
                fps=fps,
            )

    if openWebBrowser:
        Open(folder)


@requires_pygltflib
def Save_mesh(
    mesh: "Mesh",
    folder: str,
    filename: str = "mesh",
    list_displacementMatrix: list[np.ndarray] = [],
    list_nodesValues_n: list[np.ndarray] = [],
    plotMesh=False,
    cmap="jet",
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
    plotMesh : bool, optional
        displays mesh, by default False
    cmap: str, optional
        the color map used near the figure, by default "jet" \n
        ["jet", "seismic", "binary", "viridis"] -> https://matplotlib.org/stable/tutorials/colors/colormaps.html
    fps : int, optional
        Frames per second, by default 30

    Returns
    -------
    str
        The path to the created glb file.
    """

    updatedMesh = isinstance(mesh, list)
    if updatedMesh:
        list_mesh = mesh
        mesh = list_mesh[0]
    else:
        list_mesh = [mesh]

    assert mesh.dim >= 2

    # check list length
    Ndisplacement = len(list_displacementMatrix)
    Nvalues = len(list_nodesValues_n)
    if Ndisplacement > 0 and Nvalues > 0:
        assert Ndisplacement == Nvalues, (
            f"list_displacementMatrix and list_nodesValues_n must have the same length. "
            f"Got {Ndisplacement} and {Nvalues}"
        )
        if updatedMesh:
            assert Ndisplacement == len(list_mesh)
    Niter = int(np.max([Ndisplacement, Nvalues, 1]))

    # get list of nodes values
    if Nvalues > 0:
        list_nodesValues = _get_list_nodesValues(list_nodesValues_n)
        if updatedMesh:
            vMin = np.min([np.min(nodeValues) for nodeValues in list_nodesValues])
            vMax = np.max([np.max(nodeValues) for nodeValues in list_nodesValues])
        else:
            vMin, vMax = np.min(list_nodesValues), np.max(list_nodesValues)

        Display._Save_colorbar(
            vMin=vMin,
            vMax=vMax,
            folder=folder,
            filename=f"colorbar_{filename}",
            cmap=cmap,
        )

    Data._Clear_All()

    # init list of gltf mesh
    list_gltfMeshes: list[pygltflib.Mesh] = [None] * Niter

    # animation
    if Niter > 1:
        # https://gltf-tutorial.readthedocs.io/en/latest/gltfTutorial_007_Animations/#step
        times = np.array([i / fps for i in range(Niter)], dtype=float)
        data_times = Data(times, times.size, Type.SCALAR, Component.FLOAT)
        samplers: list[pygltflib.Sampler] = []
        channels: list[pygltflib.AnimationChannel] = []

    for i in range(Niter):

        if updatedMesh:
            mesh = list_mesh[i]

        # get triangles connectivity
        if updatedMesh or i == 0:
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

            # get default colors
            defaultColors = np.ones((mesh.Nn, 3)) * 0.5  # Default grey (normalized 0-1)
            data_defaultColors = Data(
                defaultColors, mesh.Nn, Type.VEC3, Component.FLOAT, Target.ARRAY_BUFFER
            )

            if plotMesh:
                # get list_lines
                list_lines: list[np.ndarray] = []
                for groupElem in mesh.Get_list_groupElem(2):
                    segments = groupElem.segments
                    if segments.shape[1] > 2:
                        repeats = [2] * segments.shape[1]
                        repeats[0] = 1
                        repeats[-1] = 1
                        segments = np.repeat(segments, repeats, axis=1)
                    lines = groupElem.connect[:, segments].reshape(-1, 2)
                    list_lines.append(lines)

                # get lines
                lines = np.concatenate(list_lines, axis=0)
                data_lines = Data(
                    lines.ravel(),
                    lines.size,
                    Type.SCALAR,
                    Component.UNSIGNED_INT,
                    Target.ELEMENT_ARRAY_BUFFER,
                )

                # get default colors
                blackColors = np.zeros((mesh.Nn, 3))
                data_blackColors = Data(
                    blackColors,
                    mesh.Nn,
                    Type.VEC3,
                    Component.FLOAT,
                    Target.ARRAY_BUFFER,
                )

        # get mesh coordinates
        coords = mesh.coord
        if Ndisplacement > 0:
            coords += list_displacementMatrix[i]
        data_coords = Data(
            coords,
            mesh.Nn,
            Type.VEC3,
            Component.FLOAT,
            Target.ARRAY_BUFFER,
        )

        # get colors
        if Nvalues > 0:
            colors = Display._Get_colors_for_values(
                list_nodesValues[i], vMax=vMax, vMin=vMin, cmap=cmap
            )
            data_colors = Data(
                colors, mesh.Nn, Type.VEC3, Component.FLOAT, Target.ARRAY_BUFFER
            )
        else:
            data_colors = data_defaultColors

        # primitives
        primitives = [
            pygltflib.Primitive(
                attributes={
                    "POSITION": data_coords._accessors_index[0],
                    "COLOR_0": data_colors._accessors_index[0],
                },
                indices=data_triangles._accessors_index[0],
                material=0,
            )
        ]
        if plotMesh:
            primitives.append(
                pygltflib.Primitive(
                    attributes={
                        "POSITION": data_coords._accessors_index[0],
                        "COLOR_0": data_blackColors._accessors_index[0],
                    },
                    indices=data_lines._accessors_index[0],
                    mode=1,  # line mode
                    material=0,
                )
            )

        # mesh
        gltfMesh = pygltflib.Mesh(primitives=primitives)
        list_gltfMeshes[i] = gltfMesh

        # animation
        if Niter > 1:
            option = "scale"
            scales = np.zeros((Niter, 3), dtype=float)
            scales[i] = 1.0

            # option = "translation"
            # scales = np.ones((Niter, 3), dtype=float) * 4
            # scales[i] = 0.0

            data_scales = Data(scales, Niter, Type.VEC3, Component.FLOAT)

            # option = "weights" # raise ANIMATION_CHANNEL_TARGET_NODE_WEIGHTS_NO_MORPHS error
            # scales = np.eye(Niter)
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
                pygltflib.AnimationChannel(
                    sampler=i, target={"node": i, "path": option}
                )
            )

    # create gltf object
    gltf = pygltflib.GLTF2()
    bufferData = b"".join(data._bufferData for data in Data._Get_list_data())
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(bufferData)))
    gltf.set_binary_blob(bufferData)

    # material
    material = pygltflib.Material()
    material.doubleSided = True  # visible surface on both sides
    gltf.materials.append(material)

    # add buffer views and accessors
    for data in Data._Get_list_data():
        gltf.bufferViews.extend(data._bufferViews)
        gltf.accessors.extend(data._accessors)

    # add mesh
    gltf.meshes = list_gltfMeshes

    # add nodes + scence
    gltf.nodes.extend([pygltflib.Node(mesh=i) for i in range(Niter)])
    gltf.scenes.append(pygltflib.Scene(nodes=list(range(Niter))))
    gltf.scene = 0

    # animation
    if Niter > 1:
        anim = pygltflib.Animation(samplers=samplers, channels=channels)
        gltf.animations.append(anim)

    # save
    filename = Folder.Join(folder, f"{filename}.glb", mkdir=True)
    gltf.save_binary(filename)

    return filename


def _write_file(file: str, content: str) -> str:
    with open(file, "w") as f:
        f.write(textwrap.dedent(content).strip())
    return file


def _Create_modelViewer_folder(
    folder: str,
    useAnimation: bool = False,
    useColorbar: bool = False,
    useModelSelector: bool = False,
):

    assert Folder.Exists(folder)

    # -------------------- create reset.css --------------------
    _write_file(
        Folder.Join(folder, "reset.css"),
        """
        :not(:defined) > * {
            display: none;
        }

        html {
            height: 100%;
        }

        body {
            margin: 0;
            padding: 0;
            height: 100vh;
            overflow: hidden;
            background-color: rgb(255, 255, 255);
        }
        """,
    )

    # -------------------- create mode-viewer.css --------------------
    _write_file(
        Folder.Join(folder, "model-viewer.css"),
        """
        model-viewer {
            width: 100%;
            height: 100%;
            position: relative;
            background-color: rgb(255, 255, 255);
        }

        @keyframes circle {
            from { transform: translateX(-50%) rotate(0deg) translateX(50px) rotate(0deg); }
            to { transform: translateX(-50%) rotate(360deg) translateX(50px) rotate(-360deg); }
        }

        @keyframes elongate {
            from { transform: translateX(100px); }
            to { transform: translateX(-100px); }
        }

        model-viewer > #ar-prompt {
            position: absolute;
            left: 50%;
            bottom: 60px;
            animation: elongate 2s infinite ease-in-out alternate;
            display: none;
        }

        model-viewer[ar-status="session-started"] > #ar-prompt {
            display: block;
        }

        model-viewer > #ar-prompt > img {
            animation: circle 4s linear infinite;
        }
        """,
    )

    if useAnimation:
        # -------------------- create animation.css --------------------
        _write_file(
            Folder.Join(folder, "animation.css"),
            """
            #autoplay-toggle {
                position: fixed;
                bottom: 5%;
                left: 5%;
                z-index: 10000;
                padding: 10px 20px;
                font-size: 16px;
                transform: translateY(-50%);
                cursor: pointer;
                background: rgba(255,255,255,0.9);
                border: 1px solid #ccc;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }

            #animation-slider {
                position: fixed;
                bottom: 5%;
                left: 50%;
                width: 40%;
                transform: translate(-50%, -50%);
                z-index: 10000;
                display: none;
            }
            """,
        )

        # -------------------- create animation.js --------------------
        _write_file(
            Folder.Join(folder, "animation.js"),
            """
            const modelSelector = document.getElementById('model-selector');
            const viewer = document.getElementById('model-viewer');
            const toggleBtn = document.getElementById('autoplay-toggle');
            const slider = document.getElementById('animation-slider');

            let autoplayOn = true;
            let sliderValue = 0;

            // ---------- Autoplay toggle ----------
            toggleBtn.addEventListener('click', () => {
                autoplayOn = !autoplayOn;

                if (autoplayOn) {
                    viewer.setAttribute('autoplay', '');
                    viewer.removeAttribute('animation-controls');
                    viewer.play();

                    slider.style.display = 'none';
                    toggleBtn.textContent = '⏸ Autoplay ON';
                } else {
                    viewer.removeAttribute('autoplay');
                    viewer.setAttribute('animation-controls', '');
                    viewer.pause();

                    slider.style.display = 'block';
                    toggleBtn.textContent = '▶ Autoplay OFF';
                }

                // save slider value
                if (viewer.duration) {
                    sliderValue = viewer.currentTime / viewer.duration;
                    slider.value = sliderValue;
                }
            });

            // ---------- Slider controls animation ----------
            slider.addEventListener('input', (e) => {
                sliderValue = parseFloat(e.target.value);
                if (viewer.duration) {
                    viewer.currentTime = sliderValue * viewer.duration;
                }
            });

            // ---------- Update slider at model load ----------
            viewer.addEventListener('load', () => {
                if (viewer.duration) {
                    viewer.currentTime = sliderValue * viewer.duration;
                }
            });
            """,
        )

    if useColorbar:
        # -------------------- create colorbar.css --------------------
        _write_file(
            Folder.Join(folder, "colorbar.css"),
            """
            .colorbar-overlay {
                position: fixed;
                right: 20px;
                top: 50%;
                transform: translateY(-50%);
                z-index: 9999;
                pointer-events: none;
            }

            .colorbar-overlay img {
                height: 400px;
                width: auto;
                display: block;
                filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
            }
            """,
        )

    if useModelSelector:
        # -------------------- create model-selector.css --------------------
        _write_file(
            Folder.Join(folder, "model-selector.css"),
            """
            #model-selector {
                position: fixed;
                bottom: 5%;
                right: 5%;
                z-index: 10000;
                padding: 10px 20px;
                font-size: 16px;
                transform: translateY(-50%);
                cursor: pointer;
                min-width: 150px;
                background: rgba(255,255,255,0.9);
                border: 1px solid #ccc;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }
            """,
        )

        # -------------------- create model-selector.js --------------------
        _write_file(
            Folder.Join(folder, "model-selector.js"),
            """
            document.addEventListener('DOMContentLoaded', function() {
                const modelSelector = document.getElementById('model-selector');
                const modelViewer = document.getElementById('model-viewer');
                const colorbarImg = document.getElementById('colorbar');

                modelSelector.addEventListener('change', function(event) {
                    const selectedOption = event.target.options[event.target.selectedIndex];

                    modelViewer.src = selectedOption.value;
                    colorbarImg.src = selectedOption.dataset.colorbar;
                });
            });
            """,
        )


def Create_html(
    path: str,
    modelViewerDir: str = None,
    defaultResult: str = None,
    allowModelSelectorButton=True,
    allowAninationButton=True,
    allowColorbar=True,
):

    isDir = Folder.os.path.isdir(path)

    folder = path if isDir else Folder.Dir(path)
    filename = Path(path).stem

    # ---------- get glb files ----------
    extensions = (".glb", "gltf")
    list_glbFile = [
        file for file in Folder.os.listdir(folder) if file.endswith(extensions)
    ]
    list_glbFile.sort()
    if len(list_glbFile) == 0:
        raise FileExistsError(
            f"No glb or gltf files were detected in the {folder} folder."
        )
    list_glb = [
        pygltflib.GLTF2().load(Folder.Join(folder, file)) for file in list_glbFile
    ]
    NglbFile = len(list_glb)
    useModelSelector = allowModelSelectorButton and isDir and NglbFile > 1

    # get default glb
    if defaultResult is None:
        defaultIndex = 0 if isDir else list_glbFile.index(Path(path).name)
    else:
        defaultIndex = [
            i
            for i, glbFile in enumerate(list_glbFile)
            if glbFile.startswith(defaultResult)
        ][0]

    # check if there is animation
    useAnination = allowAninationButton and (
        np.any([len(glb.animations) > 0 for glb in list_glb])
        if isDir
        else len(list_glb[defaultIndex].animations) > 0
    )

    # ---------- get colorbars ----------
    list_colorbar = [
        file for file in Folder.os.listdir(folder) if file.startswith("colorbar")
    ]
    list_colorbar.sort()
    useColorbar = allowColorbar and len(list_colorbar) == NglbFile

    # ---------- get model viewer directory ----------
    if modelViewerDir is None:
        modelViewerDir = Folder.Join(folder, "model-viewer")
        if not Folder.Exists(modelViewerDir):
            Folder.os.makedirs(modelViewerDir)
        _Create_modelViewer_folder(
            modelViewerDir,
            useAnimation=useAnination,
            useColorbar=useColorbar,
            useModelSelector=useModelSelector,
        )
    else:
        assert Folder.Exists(modelViewerDir)

    htmlFile = Folder.Join(folder, f"{filename}.html")
    # get relative path to model-viewer editor from html file
    relativModelViewerDir = Folder.os.path.relpath(
        modelViewerDir, start=Folder.Dir(htmlFile)
    )

    # ---------- start html with head contents ----------
    content = f"""
    <!doctype html>
<html lang="en">
    <head>
        <title>&lt;model-viewer&gt; with colorbar</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="{relativModelViewerDir}/reset.css">
        <link rel="stylesheet" href="{relativModelViewerDir}/model-viewer.css">
    """

    if useModelSelector:
        content += f"""\t<link rel="stylesheet" href="{relativModelViewerDir}/model-selector.css">"""

    if useAnination:
        content += f"""\t<link rel="stylesheet" href="{relativModelViewerDir}/animation.css">"""

    if useColorbar:
        content += (
            f"""\t<link rel="stylesheet" href="{relativModelViewerDir}/colorbar.css">"""
        )

    # ---------- start body with model-viewer ----------
    content += f"""
    </head>
    <body>

        <!-- model-viewer -->
        <model-viewer
            id="model-viewer"
            src="{list_glbFile[defaultIndex]}"
            camera-controls 
            tone-mapping="neutral"
            shadow-intensity="1"
            autoplay
            animation-name="*">
        </model-viewer>
        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js"></script>
    """

    # ---------- add model-selector ----------
    if useModelSelector:
        content += """\n\t\t<!-- model-selector -->\n\t\t<select id="model-selector">"""

        for i, glbFile in enumerate(list_glbFile):
            name = glbFile.split(".")[0]
            tabs = "\t" * 3
            selected = "selected" if i == defaultIndex else ""
            if useColorbar:
                colorbar = list_colorbar[i]
                content += f"""\n{tabs}<option value="{glbFile}" data-colorbar="{colorbar}" {selected}>{name}</option>"""
            else:
                content += (
                    f"""\n{tabs}<option value="{glbFile}" {selected}>{name}</option>"""
                )

        content += f"""
        </select>
        <script src="{relativModelViewerDir}/model-selector.js"></script>
        """

    # ---------- add colorbar ----------
    if useColorbar:
        content += f"""
        <!-- colorbar -->
        <div class="colorbar-overlay img">
            <img id="colorbar" src="{list_colorbar[defaultIndex]}" alt="Colorbar">
        </div>
        """
        pass

    # ---------- add animation ----------
    if useAnination:
        content += f"""
        <!-- animations -->
        <button id="autoplay-toggle">⏸ Autoplay ON</button>
        <input
            id="animation-slider"
            type="range"
            min="0"
            max="1"
            step="0.001"
            value="0"
            style="display: none;"
        >
        <script src="{relativModelViewerDir}/animation.js"></script>
        """

    # ---------- end body and html file ----------
    content += """
    </body>
</html>
    """

    # write the html file
    _write_file(htmlFile, content)

    return htmlFile, modelViewerDir


def Open(path: str):
    """
    Opens the specified file or directory in the default web browser.

    Parameters
    ----------
    path : str
        Path to a glb/gltf file or a directory containing glb/gltf files.
    """

    htmlFile, modelViewerDir = Create_html(path)

    # define http root
    httpRoot = Folder.os.path.commonpath([htmlFile, modelViewerDir])
    Folder.os.chdir(httpRoot)

    # Create local http server
    server = http.server.ThreadingHTTPServer(
        ("localhost", 0),  # local server with available port
        http.server.SimpleHTTPRequestHandler,  # Convert a file system folder into a minimal HTTP server.
    )
    port = server.server_address[1]

    # The thread allows: active server + continuous script
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    # daemon=True means: if the main program terminates the server dies automatically
    thread.start()

    # open in web browser
    relativHtmlFile = Folder.os.path.relpath(htmlFile, start=httpRoot)
    webbrowser.open(f"http://localhost:{port}/{relativHtmlFile}")

    thread.join()  # prevents the server from shutting down at the end of the script
