# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""`examples/WeakForms` against their closed form or the dedicated simulation."""

import numpy as np
import pytest

from EasyFEA import ElemType, Mesh, Models, Simulations, SolverType
from EasyFEA.FEM import BiLinearForm, FeArray, Field, LinearForm, Sym_Grad, Trace
from EasyFEA.Geoms import Domain

from _numerics import rel_error
from ._simus import laplacian_form

L, H = 120.0, 13.0
E, v = 210000.0, 0.3


def _poisson(mesh: Mesh, linearForm: LinearForm) -> Simulations.WeakForms:
    """-Δu = f on the square, u = 0 on its edges."""
    field = Field(mesh.groupElem, 1)
    weakForms = Models.WeakForms(field, computeK=laplacian_form(), computeF=linearForm)
    simu = Simulations.WeakForms(mesh, weakForms)
    simu.solver = SolverType.scipy
    nodesEdges = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])
    simu.add_dirichlet(nodesEdges, [0], ["u"])
    simu.Solve()
    return simu


def test_WeakForms_poisson1_is_Thermal_under_a_unit_source(unit_square):
    mesh = unit_square(1 / 8, ElemType.TRI3)

    @LinearForm
    def unit(v: Field):
        return 1.0 * v

    simu = _poisson(mesh, unit)

    thermal = Simulations.Thermal(mesh, Models.Thermal(k=1.0))
    thermal.solver = SolverType.scipy
    nodesEdges = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])
    thermal.add_dirichlet(nodesEdges, [0], ["t"])
    thermal.add_volumeLoad(mesh.nodes, [1.0], ["t"])
    thermal.Solve()
    assert np.abs(simu.u).max() > 0
    assert rel_error(simu.u, thermal.thermal) < 1e-12


def _poisson2_error(mesh: Mesh) -> float:
    """Nodal error against sin πx sin πy / 2π²."""

    @LinearForm
    def source(v: Field):
        x, y, _ = v.Get_coords()
        return np.sin(np.pi * x) * np.sin(np.pi * y) * v

    simu = _poisson(mesh, source)

    x, y, _ = mesh.coord.T
    exact = np.sin(np.pi * x) * np.sin(np.pi * y) / (2 * np.pi**2)
    return rel_error(simu.u, exact)


def test_WeakForms_poisson2_converges_at_the_element_order(subtests, unit_square):
    """Nodal values converge at least as h^(order + 1)."""
    for elemType, order in ((ElemType.TRI3, 1), (ElemType.TRI6, 2)):
        with subtests.test(elemType=str(elemType)):
            coarse = _poisson2_error(unit_square(1 / 4, elemType))
            fine = _poisson2_error(unit_square(1 / 8, elemType))

            assert np.log2(coarse / fine) > order + 1 - 0.1


def _elastic_forms(mesh: Mesh, rho: float = 0.0, damping: float = 0.0):
    """The linear elastic K, and with `rho` the mass M and Rayleigh C = damping·(K + M)."""
    material = Models.Elastic.Isotropic(2, E, v, planeStress=True)
    lmbda, mu = material.get_lambda(), material.get_mu()
    field = Field(mesh.groupElem, 2)

    def S(u: Field) -> FeArray:
        Eps = Sym_Grad(u)
        return 2 * mu * Eps + lmbda * Trace(Eps) * np.eye(2)

    @BiLinearForm
    def computeK(u: Field, v: Field):
        return S(u).ddot(Sym_Grad(v))

    @BiLinearForm
    def computeM(u: Field, v: Field):
        return rho * u.dot(v)

    @BiLinearForm
    def computeC(u: Field, v: Field):
        return damping * (computeK(u, v) + computeM(u, v))

    if rho == 0.0:
        return Models.WeakForms(field, computeK), material
    return Models.WeakForms(field, computeK, computeC, computeM), material


def _cantilever_mesh(elemType: ElemType) -> Mesh:
    domain = Domain((0, 0), (L, H), H / 3)
    return domain.Mesh_2D([], elemType, isOrganised=True)


def _bend(simu: Simulations._Simu, mesh: Mesh) -> None:
    simu.solver = SolverType.scipy
    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)
    simu.add_dirichlet(nodesX0, [0, 0], simu.Get_unknowns())
    simu.add_surfLoad(nodesXL, [-800 / H], ["y"])


def test_WeakForms_linear_elasticity1_is_Elastic():
    mesh = _cantilever_mesh(ElemType.QUAD9)
    weakForms, material = _elastic_forms(mesh)
    simu = Simulations.WeakForms(mesh, weakForms)
    elastic = Simulations.Elastic(mesh, material)

    for s in (simu, elastic):
        _bend(s, mesh)
        s.Solve()

    assert rel_error(simu.u, elastic.displacement) < 1e-10


def _dynamic_pair() -> tuple[Simulations.WeakForms, Simulations.Elastic]:
    """`LinearElasticity2` and its Elastic twin, bent statically then released under Newmark."""
    rho, damping, dt = 8.1e-6, 1e-3, 0.01
    mesh = _cantilever_mesh(ElemType.QUAD4)
    weakForms, material = _elastic_forms(mesh, rho, damping)
    simu = Simulations.WeakForms(mesh, weakForms)
    elastic = Simulations.Elastic(mesh, material)
    elastic.rho = rho
    elastic.Set_Rayleigh_Damping_Coefs(coefM=damping, coefK=damping)
    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    for s in (simu, elastic):
        _bend(s, mesh)
        s.Solve()
        s.Bc_Init()
        s.Solver_Set_Hyperbolic_Algorithm(dt)
        s.add_dirichlet(nodesX0, [0, 0], ["x", "y"])
        s.Solve()
        s.Solve()
    return simu, elastic


def test_WeakForms_linear_elasticity2_is_Elastic_in_dynamics():
    simu, elastic = _dynamic_pair()

    assert rel_error(simu.u, elastic.displacement) < 1e-10


def test_WeakForms_weakForms_is_the_given_model(unit_square):
    mesh = unit_square(1 / 2, ElemType.TRI3)
    weakForms = Models.WeakForms(Field(mesh.groupElem, 1), laplacian_form())

    assert Simulations.WeakForms(mesh, weakForms).weakForms is weakForms


def test_WeakForms_rejects_a_model_that_is_not_weak_forms(unit_square):
    mesh = unit_square(1 / 2, ElemType.TRI3)

    with pytest.raises(AssertionError, match="weak"):
        Simulations.WeakForms(mesh, Models.Thermal(1.0))


def test_WeakForms_u_is_the_nodal_solution():
    simu, elastic = _dynamic_pair()

    assert simu.u.shape == (simu.mesh.Nn * 2,)
    assert np.array_equal(simu.u, simu.Result("u"))


def test_WeakForms_v_is_the_velocity():
    simu, elastic = _dynamic_pair()

    assert np.abs(simu.v).max() > 0
    assert rel_error(simu.v, elastic.speed) < 1e-8


def test_WeakForms_a_is_the_acceleration():
    simu, elastic = _dynamic_pair()

    assert rel_error(simu.a, elastic.accel) < 1e-8


def test_WeakForms_components_split_their_field(subtests):
    simu, _ = _dynamic_pair()
    for field in ("u", "v", "a"):
        with subtests.test(field=field):
            values = getattr(simu, field).reshape(-1, 2)

            assert np.array_equal(simu.Result(f"{field}x"), values[:, 0])
            assert np.array_equal(simu.Result(f"{field}y"), values[:, 1])


def test_WeakForms_nonlinear_solve_of_a_linear_form_is_the_linear_solve():
    mesh = _cantilever_mesh(ElemType.QUAD4)
    weakForms, _ = _elastic_forms(mesh)
    linear = Simulations.WeakForms(mesh, weakForms)
    nonlinear = Simulations.WeakForms(mesh, weakForms, isNonLinear=True, tolConv=1e-8)

    for s in (linear, nonlinear):
        _bend(s, mesh)
        s.Solve()

    assert nonlinear.isNonLinear
    assert rel_error(nonlinear.u, linear.u) < 1e-10
