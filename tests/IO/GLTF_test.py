# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import os
import struct

import numpy as np
import pytest
from pygltflib import GLTF2
from pygltflib.validator import validate

from EasyFEA import ElemType, Mesh, Simulations, IO
from EasyFEA.IO.GLTF import Data, Type, Component, StructFormat, Target

from Simulations._simus import LOADS
from ._meshes import frames_of

# validate() warns that it is provisional on every call; any other warning still surfaces
pytestmark = pytest.mark.filterwarnings(
    "ignore:pygltf.utils.validator is a provisional function:UserWarning"
)


def read_accessor(gltf: GLTF2, index: int) -> np.ndarray:
    accessor = gltf.accessors[index]
    bufferView = gltf.bufferViews[accessor.bufferView]
    data = gltf.binary_blob()[
        bufferView.byteOffset : bufferView.byteOffset + bufferView.byteLength
    ]
    dtype = np.float32 if accessor.componentType == Component.FLOAT else np.uint32
    return np.frombuffer(data, dtype=dtype).reshape(accessor.count, -1)


def test_Type_spells_the_gltf_accessor_types():
    assert [t.value for t in Type] == [
        "SCALAR",
        "VEC2",
        "VEC3",
        "VEC4",
        "MAT2",
        "MAT3",
        "MAT4",
    ]


def test_Component_holds_the_gltf_component_type_codes():
    assert Component.FLOAT == 5126
    assert Component.UNSIGNED_INT == 5125
    assert Component.SIGNED_BYTE == 5120


def test_StructFormat_packs_each_component_to_its_gltf_byte_size():
    assert struct.calcsize(StructFormat.SIGNED_SHORT.value) == 2
    assert struct.calcsize(StructFormat.UNSIGNED_INT.value) == 4
    assert struct.calcsize(StructFormat.FLOAT.value) == 4


def test_Target_holds_the_gltf_buffer_targets():
    assert Target.ARRAY_BUFFER == 34962
    assert Target.ELEMENT_ARRAY_BUFFER == 34963


def test_get_list_nodesValues_keeps_scalars_and_takes_the_norm_of_vectors():
    vectors = [np.array([[3.0, 4.0], [0.0, 1.0]])]
    scalars = [np.array([1.0, 2.0])]

    assert IO.GLTF._get_list_nodesValues(scalars) is scalars
    np.testing.assert_allclose(IO.GLTF._get_list_nodesValues(vectors)[0], [5, 1])
    with pytest.raises(NotImplementedError):
        IO.GLTF._get_list_nodesValues([np.zeros((2, 2, 2))])


def test_Data_packs_consecutive_buffer_views():
    Data._Clear_All()

    first = Data(np.arange(3.0), 3, Type.SCALAR, Component.FLOAT)
    second = Data(np.ones((2, 3)), 2, Type.VEC3, Component.FLOAT)

    assert [(v.byteOffset, v.byteLength) for v in first._bufferViews] == [(0, 12)]
    assert [(v.byteOffset, v.byteLength) for v in second._bufferViews] == [(12, 24)]
    assert first._accessors_index == [0]
    assert second._accessors_index == [1]
    assert second._accessors[0].max == [1.0, 1.0, 1.0]


def test_Data_gives_a_list_one_buffer_view_per_array():
    Data._Clear_All()

    frames = Data([np.arange(3.0), np.arange(3.0) + 1], 3, Type.SCALAR, Component.FLOAT)
    after = Data(np.arange(3.0), 3, Type.SCALAR, Component.FLOAT)

    assert [(v.byteOffset, v.byteLength) for v in frames._bufferViews] == [
        (0, 12),
        (12, 12),
    ]
    assert len(frames._bufferData) == 24
    assert [(v.byteOffset, v.byteLength) for v in after._bufferViews] == [(24, 12)]
    assert after._bufferViews_index == [2]


def test_Data__Get_list_data_lists_every_data_since_the_last_clear():
    Data._Clear_All()

    first = Data(np.arange(3.0), 3, Type.SCALAR, Component.FLOAT)
    second = Data(np.arange(3.0), 3, Type.SCALAR, Component.FLOAT)

    assert Data._Get_list_data() == [first, second]


def test_Data__Clear_All_restarts_offsets_and_indices():
    Data(np.arange(3.0), 3, Type.SCALAR, Component.FLOAT)

    Data._Clear_All()
    data = Data(np.arange(3.0), 3, Type.SCALAR, Component.FLOAT)

    assert Data._Get_list_data() == [data]
    assert data._bufferViews[0].byteOffset == 0
    assert data._accessors_index == [0]


def test_Save_mesh_writes_a_valid_glb_with_the_mesh_nodes(mesh_of, tmp_path, subtests):
    for elemType in ElemType.Get_2D() + ElemType.Get_3D():
        mesh = mesh_of(elemType)
        for plotMesh in (False, True):
            with subtests.test(elemType=str(elemType), plotMesh=plotMesh):
                path = IO.GLTF.Save_mesh(mesh, str(tmp_path), "mesh", plotMesh=plotMesh)

                gltf = GLTF2().load(path)
                validate(gltf)
                primitives = gltf.meshes[0].primitives
                assert len(primitives) == (2 if plotMesh else 1)
                positions = read_accessor(gltf, primitives[0].attributes.POSITION)
                np.testing.assert_allclose(positions, mesh.coord, rtol=1e-6)
                assert gltf.animations == []


def test_Save_mesh_animates_one_node_per_frame_at_its_displaced_position(
    make_mesh_2D, tmp_path
):
    mesh = make_mesh_2D()
    frames = frames_of(mesh)

    path = IO.GLTF.Save_mesh(
        mesh,
        str(tmp_path),
        "mesh",
        list_displacementMatrix=frames,
        list_nodesValues_n=frames,
    )

    gltf = GLTF2().load(path)
    validate(gltf)
    assert len(gltf.nodes) == len(frames)
    assert len(gltf.animations[0].channels) == len(frames)
    for gltfMesh, frame in zip(gltf.meshes, frames):
        positions = read_accessor(gltf, gltfMesh.primitives[0].attributes.POSITION)
        np.testing.assert_allclose(positions, mesh.coord + frame, rtol=1e-6)
    assert (tmp_path / "colorbar_mesh.png").exists()


def test_Save_mesh_takes_one_mesh_per_frame(make_mesh_2D, tmp_path):
    meshes = [make_mesh_2D(meshSize=meshSize) for meshSize in (1 / 2, 1 / 3)]
    values = [mesh.coord[:, 0] for mesh in meshes]

    path = IO.GLTF.Save_mesh(meshes, str(tmp_path), "mesh", list_nodesValues_n=values)

    gltf = GLTF2().load(path)
    validate(gltf)
    for gltfMesh, mesh in zip(gltf.meshes, meshes):
        positions = read_accessor(gltf, gltfMesh.primitives[0].attributes.POSITION)
        np.testing.assert_allclose(positions, mesh.coord, rtol=1e-6)


def test_Save_mesh_is_binary_only(make_mesh_2D, tmp_path):
    mesh = make_mesh_2D()

    with pytest.raises(ValueError):
        IO.GLTF.Save_mesh(mesh, str(tmp_path), "text", useBinary=False)


def test_Save_simu_writes_one_glb_per_component_and_the_norm(
    bar_simus: list[Simulations.Elastic], tmp_path, subtests
):
    for simu in bar_simus:
        with subtests.test(dim=simu.mesh.dim):
            folder = tmp_path / str(simu.mesh.dim)

            saved = IO.GLTF.Save_simu(simu, str(folder), results=["uy", "displacement"])

            assert saved == str(folder)
            components = [f"displacement_{d}" for d in range(simu.mesh.dim)]
            glbFiles = sorted(f for f in os.listdir(folder) if f.endswith(".glb"))
            assert glbFiles == sorted(
                f"{name}.glb" for name in ["uy", *components, "displacement_norm"]
            )
            gltf = GLTF2().load(str(folder / "uy.glb"))
            validate(gltf)
            # the first iteration is repeated at the end, so the animation loops back
            assert len(gltf.nodes) == len(LOADS) + 1
            for i in range(len(LOADS)):
                simu.Set_Iter(i)
                positions = read_accessor(
                    gltf, gltf.meshes[i].primitives[0].attributes.POSITION
                )
                expected = simu.mesh.coord + simu.Results_displacement_matrix()
                np.testing.assert_allclose(positions, expected, rtol=1e-6, atol=1e-9)


def test_Save_simu_needs_a_simulation(make_mesh_2D, tmp_path):
    mesh = make_mesh_2D()

    assert IO.GLTF.Save_simu(mesh, str(tmp_path), results=["uy"]) is None  # type: ignore [arg-type]


def test_write_file_dedents_and_strips(tmp_path):
    path = str(tmp_path / "file.txt")

    assert IO.GLTF._write_file(path, "\n    a\n      b\n    ") == path
    with open(path, encoding="utf-8") as file:
        assert file.read() == "a\n  b"


def test_Create_modelViewer_folder_writes_the_files_of_each_option(tmp_path, subtests):
    cases = {
        "base": ({}, []),
        "animation": ({"useAnimation": True}, ["animation.css", "animation.js"]),
        "colorbar": ({"useColorbar": True}, ["colorbar.css"]),
        "modelSelector": (
            {"useModelSelector": True},
            ["model-selector.css", "model-selector.js"],
        ),
    }
    for name, (options, files) in cases.items():
        with subtests.test(option=name):
            folder = tmp_path / name
            folder.mkdir()

            IO.GLTF._Create_modelViewer_folder(str(folder), **options)

            assert sorted(os.listdir(folder)) == sorted(
                ["reset.css", "model-viewer.css", *files]
            )


def save_two_results(mesh: Mesh, folder: str) -> None:
    frames = frames_of(mesh)
    for name in ("ux", "uy"):
        IO.GLTF.Save_mesh(
            mesh,
            folder,
            name,
            list_displacementMatrix=frames,
            list_nodesValues_n=frames,
        )


def test_Create_html_on_a_folder_selects_between_its_glb_files(make_mesh_2D, tmp_path):
    save_two_results(make_mesh_2D(), str(tmp_path))

    htmlFile, modelViewerDir = IO.GLTF.Create_html(str(tmp_path), defaultResult="uy")

    with open(htmlFile, encoding="utf-8") as file:
        html = file.read()
    assert htmlFile == str(tmp_path / f"{tmp_path.name}.html")
    assert 'src="uy.glb"' in html
    assert '<option value="ux.glb" data-colorbar="colorbar_ux.png" >' in html
    assert sorted(os.listdir(modelViewerDir)) == sorted(
        [
            "reset.css",
            "model-viewer.css",
            "animation.css",
            "animation.js",
            "colorbar.css",
            "model-selector.css",
            "model-selector.js",
        ]
    )


def test_Create_html_on_a_file_shows_that_file_alone(make_mesh_2D, tmp_path):
    mesh = make_mesh_2D()
    IO.GLTF.Save_mesh(mesh, str(tmp_path), "a")
    path = IO.GLTF.Save_mesh(mesh, str(tmp_path), "b")

    htmlFile, _ = IO.GLTF.Create_html(path)

    with open(htmlFile, encoding="utf-8") as file:
        html = file.read()
    assert htmlFile == str(tmp_path / "b.html")
    assert 'src="b.glb"' in html
    assert "model-selector" not in html
    assert "animation.js" not in html


def test_Create_html_needs_a_glb_file(tmp_path):
    with pytest.raises(FileExistsError):
        IO.GLTF.Create_html(str(tmp_path))


def test_Open_serves_the_html_and_opens_it_in_the_browser(
    make_mesh_2D, tmp_path, monkeypatch
):
    path = IO.GLTF.Save_mesh(make_mesh_2D(), str(tmp_path), "mesh")
    opened: list[str] = []

    class Server:
        server_address = ("localhost", 8123)

        def __init__(self, *args):
            pass

        def serve_forever(self):
            pass

    class Thread:
        def __init__(self, target, daemon):
            pass

        def start(self):
            pass

        def join(self):
            pass

    monkeypatch.chdir(tmp_path)  # Open moves to the served root
    monkeypatch.setattr(IO.GLTF.http.server, "ThreadingHTTPServer", Server)
    monkeypatch.setattr(IO.GLTF.threading, "Thread", Thread)
    monkeypatch.setattr(IO.GLTF.webbrowser, "open", opened.append)

    IO.GLTF.Open(path)

    assert opened == ["http://localhost:8123/mesh.html"]
