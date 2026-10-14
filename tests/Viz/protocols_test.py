# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA import Matplotlib, PyVista
from EasyFEA.Viz import Viewer

from _signatures import assert_spelled_as

VIEWERS = (Matplotlib, PyVista)


def test_Viewer_is_implemented_by_both_viewers(subtests):
    members = [name for name in vars(Viewer) if not name.startswith("_")]
    for module in VIEWERS:
        with subtests.test(module=module.__name__):
            assert all(callable(getattr(module, name, None)) for name in members)


def test_Viewer_Plot_is_spelled_alike_by_both_viewers(subtests):
    for module in VIEWERS:
        with subtests.test(module=module.__name__):
            assert_spelled_as(Viewer.Plot, module.Plot)


def test_Viewer_Plot_Mesh_is_spelled_alike_by_both_viewers(subtests):
    for module in VIEWERS:
        with subtests.test(module=module.__name__):
            assert_spelled_as(Viewer.Plot_Mesh, module.Plot_Mesh)


def test_Viewer_Plot_Nodes_is_spelled_alike_by_both_viewers(subtests):
    for module in VIEWERS:
        with subtests.test(module=module.__name__):
            assert_spelled_as(Viewer.Plot_Nodes, module.Plot_Nodes)


def test_Viewer_Plot_Elements_is_spelled_alike_by_both_viewers(subtests):
    for module in VIEWERS:
        with subtests.test(module=module.__name__):
            assert_spelled_as(Viewer.Plot_Elements, module.Plot_Elements)


def test_Viewer_Plot_BoundaryConditions_is_spelled_alike_by_both_viewers(subtests):
    for module in VIEWERS:
        with subtests.test(module=module.__name__):
            assert_spelled_as(
                Viewer.Plot_BoundaryConditions, module.Plot_BoundaryConditions
            )


def test_Viewer_Plot_Tags_is_spelled_alike_by_both_viewers(subtests):
    for module in VIEWERS:
        with subtests.test(module=module.__name__):
            assert_spelled_as(Viewer.Plot_Tags, module.Plot_Tags)


def test_Viewer_Plot_Geoms_is_spelled_alike_by_both_viewers(subtests):
    for module in VIEWERS:
        with subtests.test(module=module.__name__):
            assert_spelled_as(Viewer.Plot_Geoms, module.Plot_Geoms)


def test_Viewer_Movie_simu_is_spelled_alike_by_both_viewers(subtests):
    for module in VIEWERS:
        with subtests.test(module=module.__name__):
            assert_spelled_as(Viewer.Movie_simu, module.Movie_simu)


def test_Viewer_Movie_func_is_spelled_alike_by_both_viewers(subtests):
    for module in VIEWERS:
        with subtests.test(module=module.__name__):
            assert_spelled_as(Viewer.Movie_func, module.Movie_func)
