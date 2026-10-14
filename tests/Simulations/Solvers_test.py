# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import importlib.util

import pytest

from EasyFEA import AlgoType, ElemType, Models, Simulations, SolverType
from EasyFEA.Geoms import Domain

from _numerics import rel_error
from . import _reference

T0, TL = 10.0, 40.0


def test_AlgoType_prints_its_name():
    assert [str(algo) for algo in AlgoType] == [algo.value for algo in AlgoType]


def test_AlgoType_Get_Hyperbolic_Types_lists_every_second_order_scheme():
    hyperbolic = AlgoType.Get_Hyperbolic_Types()

    assert set(hyperbolic) == set(AlgoType) - {AlgoType.elliptic, AlgoType.parabolic}


def test_AlgoType_Get_Hyperbolic_and_Parabolic_Types_adds_parabolic():
    timeDependent = AlgoType.Get_Hyperbolic_and_Parabolic_Types()

    assert set(timeDependent) == set(AlgoType) - {AlgoType.elliptic}


HAS_PYPARDISO = importlib.util.find_spec("pypardiso") is not None
LINEAR_SOLVERS = [
    SolverType.petsc,
    SolverType.scipy,
    SolverType.cg,
    SolverType.bicg,
    SolverType.gmres,
    SolverType.lgmres,
]
if HAS_PYPARDISO:
    LINEAR_SOLVERS.append(SolverType.pypardiso)
DIRECT = {SolverType.pypardiso, SolverType.scipy}


def test_SolverType_every_linear_solver_solves_the_heat_slab(subtests):
    """Direct solvers to round-off, iterative ones to their default tolerance. On 4 QUAD4."""
    domain = Domain((0, 0), (2, 0.5), 0.5)
    mesh = domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == 2)
    expected = _reference.slab_temperature(mesh.coord[:, 0], 2, T0, TL)
    for solver in LINEAR_SOLVERS:
        with subtests.test(solver=str(solver)):
            simu = Simulations.Thermal(mesh, Models.Thermal(1.0, 1.0))
            simu.solver = solver
            simu.add_dirichlet(nodesX0, [T0], ["t"])
            simu.add_dirichlet(nodesXL, [TL], ["t"])

            simu.Solve()

            tol = 1e-12 if solver in DIRECT else 1e-4
            assert rel_error(simu.thermal, expected) < tol


def test_SolverType_lsq_linear_bounds_the_damage(unit_square):
    """The damage of a pulled bar is still the homogeneous one, solved within its bounds."""
    mesh = unit_square(0.5)
    E, Gc, l0, strain = 210e3, 2.7, 0.1, 0.02
    pfm = Models.PhaseField(
        Models.Elastic.Isotropic(2, E=E, v=0.3, planeStress=True),
        "Bourdin",
        "AT2",
        Gc,
        l0,
        solver=Models.PhaseField.SolverType.BoundConstrain,
    )
    simu = Simulations.PhaseField(mesh, pfm)
    # direct: the default cg + gamg stops at ~1e-5
    simu._Solver_Set_PETSc4Py_Options(
        "preonly", "lu", "petsc", simu.ProblemTypes.elastic
    )
    _reference.pull_bar(simu, strain)

    simu.Solve()
    simu.Solve()

    expected = _reference.homogeneous_damage(E * strain**2 / 2, Gc, l0, "AT2")
    assert rel_error(simu.damage, expected) < 1e-9
