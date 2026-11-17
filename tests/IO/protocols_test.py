# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA import IO, Simulations

from _signatures import assert_spelled_as
from Simulations._simus import assert_same_state, live_state, moved_on

SAVERS = [IO.Gmsh, IO.Medit, IO.Ensight, IO.USD, IO.GLTF]
LOADERS = [IO.Gmsh, IO.Medit, IO.Ensight]
SIMU_SAVERS = [IO.Gmsh, IO.Paraview, IO.Vizir, IO.USD, IO.GLTF]


def test_MeshSaver_is_implemented_by_every_saving_format(subtests):
    for module in SAVERS:
        with subtests.test(module=module.__name__):
            assert callable(module.Save_mesh)


def test_MeshLoader_is_implemented_by_every_loading_format(subtests):
    for module in LOADERS:
        with subtests.test(module=module.__name__):
            assert callable(module.Load_mesh)


def test_SimuSaver_is_implemented_by_every_simulation_format(subtests):
    for module in SIMU_SAVERS:
        with subtests.test(module=module.__name__):
            assert callable(module.Save_simu)


def test_MeshSaver_Save_mesh_is_spelled_alike_by_every_format(subtests):
    for module in SAVERS:
        with subtests.test(module=module.__name__):
            assert_spelled_as(IO.MeshSaver.Save_mesh, module.Save_mesh)


def test_MeshLoader_Load_mesh_is_spelled_alike_by_every_format(subtests):
    for module in LOADERS:
        with subtests.test(module=module.__name__):
            assert_spelled_as(IO.MeshLoader.Load_mesh, module.Load_mesh)


def test_SimuSaver_Save_simu_is_spelled_alike_by_every_format(subtests):
    for module in SIMU_SAVERS:
        with subtests.test(module=module.__name__):
            assert_spelled_as(IO.SimuSaver.Save_simu, module.Save_simu)


def test_SimuSaver_Save_simu_leaves_the_simulation_where_it_is(
    bar_simu: Simulations.Elastic, tmp_path, subtests
):
    simu = moved_on(bar_simu)
    live = live_state(simu)
    for module in SIMU_SAVERS:
        with subtests.test(module=module.__name__):
            options = {"types": [1]} if module is IO.Vizir else {}
            folder = str(tmp_path / module.__name__)

            module.Save_simu(simu, folder, results=["ux"], **options)

            assert_same_state(live_state(simu), live)
