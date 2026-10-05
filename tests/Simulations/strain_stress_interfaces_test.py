# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""``_Calc_Epsilon`` / ``_Calc_Sigma`` (and the hyperelastic pair): an ``FeArray`` on one group, ``{groupElem: FeArray}`` on several, both plottable."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
import pyvista  # noqa: E402

from EasyFEA import (  # noqa: E402
    ElemType,
    Matplotlib,
    Mesh,
    Mesher,
    Models,
    PyVista,
    Simulations,
)
from EasyFEA.FEM import FeArray, MatrixType  # noqa: E402
from EasyFEA.Geoms import Domain, Line, Point  # noqa: E402
from EasyFEA.Models.InElastic import Plasticity  # noqa: E402
from EasyFEA.Models.InElastic.IsotropicHardening import Linear  # noqa: E402
from EasyFEA.Models.InElastic.Yield import VonMises  # noqa: E402

pyvista.OFF_SCREEN = True

L, H = 120.0, 13.0
E, v = 210e3, 0.3


def _mesh(mixed: bool) -> Mesh:
    domain = Domain(Point(0, 0), Point(L, H), H / 2)
    if mixed:
        mesh = domain.Mesh_2D([], ElemType.QUAD4)
        assert len(mesh.Get_list_groupElem()) == 2, "the mesh mixes QUAD4 and TRI3"
    else:
        mesh = domain.Mesh_2D([], ElemType.TRI3)
        assert len(mesh.Get_list_groupElem()) == 1
    return mesh


def _pull(simu, mesh: Mesh, u: float = 0.01):
    simu.add_dirichlet(
        mesh.Nodes_Conditions(lambda x, y, z: x == 0), [0, 0], ["x", "y"]
    )
    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x, y, z: x == L), [u], ["x"])
    simu.Solve()
    return simu


def _elastic(mesh: Mesh):
    return _pull(Simulations.Elastic(mesh, Models.Elastic.Isotropic(2, E, v)), mesh)


def _phasefield(mesh: Mesh):
    pfm = Models.PhaseField(Models.Elastic.Isotropic(2, E, v), "Miehe", "AT2", 1, 1)
    simu = Simulations.PhaseField(mesh, pfm)
    simu.add_dirichlet(
        mesh.Nodes_Conditions(lambda x, y, z: x == 0), [0, 0], ["x", "y"]
    )
    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x, y, z: x == L), [0.01], ["x"])
    simu.Solve(1e-1, maxIter=2)
    return simu


def _inelastic(mesh: Mesh):
    behavior = Plasticity(Models.Elastic.Isotropic(2, E, v), VonMises(250), Linear(1e3))
    return _pull(Simulations.InElastic(mesh, behavior), mesh)


def _hyperelastic(mesh: Mesh):
    material = Models.HyperElastic.NeoHookean(2, K=5.0e4)
    return _pull(Simulations.HyperElastic(mesh, material, verbosity=False), mesh)


INTERFACES = [
    (_elastic, "_Calc_Epsilon"),
    (_elastic, "_Calc_Sigma"),
    (_phasefield, "_Calc_Epsilon"),
    (_phasefield, "_Calc_Sigma"),
    (_inelastic, "_Calc_Epsilon"),
    (_inelastic, "_Calc_Sigma"),
    (_hyperelastic, "_Calc_GreenLagrange"),
    (_hyperelastic, "_Calc_SecondPiolaKirchhoff"),
]


def _plot(simu, values) -> None:
    Matplotlib.Plot(simu, values, nodeValues=False)
    plt.close("all")
    PyVista.Plot(simu, values, nodeValues=False).close()


@pytest.mark.parametrize("mixed", [False, True], ids=["one group", "mixed"])
@pytest.mark.parametrize(
    "build, interface", INTERFACES, ids=[f"{b.__name__}{i}" for b, i in INTERFACES]
)
def test_interface_is_per_group_and_plots(build, interface: str, mixed: bool):
    mesh = _mesh(mixed)
    simu = build(mesh)
    field = getattr(simu, interface)()

    if mixed:
        assert isinstance(field, dict)
        assert list(field) == mesh.Get_list_groupElem()
        for groupElem, field_e_pg in field.items():
            assert isinstance(field_e_pg, FeArray)
            assert field_e_pg.shape[:2] == (
                groupElem.Ne,
                groupElem.Get_gauss(MatrixType.rigi).nPg,
            )
        values = {groupElem: f[..., 0] for groupElem, f in field.items()}
    else:
        assert isinstance(field, FeArray)
        assert field.shape[0] == mesh.Ne
        values = field[..., 0]

    _plot(simu, values)


@pytest.mark.parametrize("interface", ["_Calc_Epsilon", "_Calc_Sigma"])
def test_beam_interface_plots(interface: str):
    section = Domain(Point(-5, -5), Point(5, 5)).Mesh_2D([], ElemType.QUAD4)
    beam = Models.Beam.Isotropic(2, Line(Point(), Point(L), L / 10), section, E, v)
    structure = Models.Beam.BeamStructure([beam])
    mesh = Mesher().Mesh_Beams([beam], ElemType.SEG2)
    simu = Simulations.Beam(mesh, structure, verbosity=False)
    simu.add_dirichlet(mesh.Nodes_Point(Point()), [0, 0, 0], ["x", "y", "rz"])
    simu.add_neumann(mesh.Nodes_Point(Point(L)), [-800], ["y"])
    simu.Solve()

    field = getattr(simu, interface)()
    assert isinstance(field, FeArray)
    assert field.shape[:2] == (mesh.Ne, mesh.groupElem.Get_gauss(MatrixType.beam).nPg)
    _plot(simu, field[..., 0])
