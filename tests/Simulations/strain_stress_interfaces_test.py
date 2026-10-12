# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""``_Calc_Epsilon`` / ``_Calc_Sigma`` (and the hyperelastic pair): an ``FeArray`` on one group, ``{groupElem: FeArray}`` on several, holding the pulled bar's uniaxial state."""

from typing import Callable, NamedTuple

import numpy as np
import pytest

from EasyFEA import ElemType, Mesh, Mesher, Models, Simulations, SolverType
from EasyFEA.FEM import FeArray, MatrixType
from EasyFEA.Geoms import Domain, Line, Point
from EasyFEA.Models.InElastic import Plasticity
from EasyFEA.Models.InElastic.IsotropicHardening import Linear
from EasyFEA.Models.InElastic.Yield import VonMises

from _numerics import rel_error
from . import _reference

L, H = 120.0, 13.0
E, v = 210e3, 0.3
STRAIN = 1e-4  # below the plastic yield strain 250 / E


def _tri3() -> Mesh:
    domain = Domain(Point(0, 0), Point(L, H), H / 2)
    return domain.Mesh_2D([], ElemType.TRI3)


def _quad4_tri3() -> Mesh:
    domain = Domain(Point(0, 0), Point(L, H), H / 2)
    return domain.Mesh_2D([], ElemType.QUAD4)


def _hexa8_prism6() -> Mesh:
    domain = Domain(Point(0, 0), Point(L, H), H / 2)
    return domain.Mesh_Extrude([], [0, 0, H], [2], ElemType.HEXA8)


MESHES: dict[str, Callable[[], Mesh]] = {
    "TRI3": _tri3,
    "QUAD4+TRI3": _quad4_tri3,
    "HEXA8+PRISM6": _hexa8_prism6,
}


def _elastic(dim: int) -> Models.Elastic.Isotropic:
    return Models.Elastic.Isotropic(dim, E, v, planeStress=True)


def _phasefield(mesh: Mesh) -> Simulations.PhaseField:
    pfm = Models.PhaseField(
        _elastic(mesh.dim),
        "Amor",
        "AT2",
        1,
        1,
    )
    simu = Simulations.PhaseField(mesh, pfm)
    # direct: the default cg + gamg stops at ~1e-5
    simu._Solver_Set_PETSc4Py_Options(
        "preonly", "lu", "petsc", simu.ProblemTypes.elastic
    )
    return simu


def _direct(simu: Simulations._Simu) -> Simulations._Simu:
    simu.solver = SolverType.scipy  # petsc's iterative default stops at ~1e-6
    return simu


def _inelastic(mesh: Mesh) -> Simulations.InElastic:
    behavior = Plasticity(_elastic(mesh.dim), VonMises(250), Linear(1e3))
    return _direct(Simulations.InElastic(mesh, behavior))


def _hyperelastic(mesh: Mesh) -> Simulations.HyperElastic:
    """Saint-Venant-Kirchhoff: the bar reaches a homogeneous state with Syy = 0."""
    material = Models.HyperElastic.SaintVenantKirchhoff(mesh.dim, lmbda=1e5, mu=8e4)
    return _direct(Simulations.HyperElastic(mesh, material))


class Kind(NamedTuple):
    """A simulation, its strain and stress interfaces, and the bar's exact strain and stress (None: no closed form) along x."""

    build: Callable[[Mesh], Simulations._Simu]
    strain: str
    stress: str
    Exx: float
    Sxx: float | None


KINDS = {
    "elastic": Kind(
        lambda mesh: _direct(Simulations.Elastic(mesh, _elastic(mesh.dim))),
        "_Calc_Epsilon",
        "_Calc_Sigma",
        STRAIN,
        E * STRAIN,
    ),
    "phasefield": Kind(
        _phasefield,
        "_Calc_Epsilon",
        "_Calc_Sigma",
        STRAIN,
        E * STRAIN,
    ),
    "inelastic": Kind(
        _inelastic,
        "_Calc_Epsilon",
        "_Calc_Sigma",
        STRAIN,
        E * STRAIN,
    ),
    "hyperelastic": Kind(
        _hyperelastic,
        "_Calc_GreenLagrange",
        "_Calc_SecondPiolaKirchhoff",
        STRAIN + STRAIN**2 / 2,
        None,
    ),
}


def _pulled(kind: Kind, mesh: Mesh) -> Simulations._Simu:
    simu = kind.build(mesh)
    _reference.pull_bar(simu, STRAIN * L)
    simu.Solve()
    return simu


def _component(field, mesh: Mesh, index: int) -> np.ndarray:
    """Component `index` at every Gauss point, after checking the field is one FeArray per group."""
    groups = mesh.Get_list_groupElem()
    if len(groups) == 1:
        assert isinstance(field, FeArray)
        field = {groups[0]: field}
    assert isinstance(field, dict)
    assert list(field) == groups
    for groupElem, field_e_pg in field.items():
        assert isinstance(field_e_pg, FeArray)
        nPg = groupElem.Get_gauss(MatrixType.rigi).nPg
        assert field_e_pg.shape[:2] == (groupElem.Ne, nPg)
    return np.concatenate([np.ravel(f[..., index]) for f in field.values()])


@pytest.mark.parametrize("meshName", MESHES)
@pytest.mark.parametrize("kindName", KINDS)
def test_strain_is_per_group_and_matches_the_bar(kindName: str, meshName: str):
    kind, mesh = KINDS[kindName], MESHES[meshName]()
    simu = _pulled(kind, mesh)

    Exx = _component(getattr(simu, kind.strain)(), mesh, 0)

    assert rel_error(Exx, kind.Exx) < 1e-10


@pytest.mark.parametrize("meshName", MESHES)
@pytest.mark.parametrize("kindName", KINDS)
def test_stress_is_per_group_and_uniaxial(kindName: str, meshName: str):
    kind, mesh = KINDS[kindName], MESHES[meshName]()
    simu = _pulled(kind, mesh)
    stress = getattr(simu, kind.stress)()

    Sxx = _component(stress, mesh, 0)
    Syy = _component(stress, mesh, 1)

    assert rel_error(Syy, 0.0, scale=np.abs(Sxx).max()) < 1e-10
    if kind.Sxx is not None:
        assert rel_error(Sxx, kind.Sxx) < 1e-10


def _cantilever() -> tuple[Simulations.Beam, np.ndarray]:
    """SEG2 cantilever with a tip load, and its exact moment at the element centers."""
    domain = Domain(Point(-5, -5), Point(5, 5))
    section = domain.Mesh_2D([], ElemType.QUAD4)
    line = Line(Point(), Point(L), L / 10)
    beam = Models.Beam.Isotropic(
        2,
        line,
        section,
        E,
        v,
    )
    mesh = Mesher().Mesh_Beams([beam], elemType=ElemType.SEG2)
    simu = Simulations.Beam(mesh, Models.Beam.BeamStructure([beam]))
    nodesClamped = mesh.Nodes_Point(Point())
    nodesTip = mesh.Nodes_Point(Point(L))
    simu.add_dirichlet(nodesClamped, [0, 0, 0], ["x", "y", "rz"])
    simu.add_neumann(nodesTip, [-800], ["y"])
    simu.Solve()
    x_e = mesh.coord[mesh.connect, 0].mean(1)
    Mz = _reference.cantilever_tip(L, -800, E * beam.Iz).Mz(x_e)
    return simu, Mz


def _element_means(simu: Simulations.Beam, field, index: int) -> np.ndarray:
    assert isinstance(field, FeArray)
    nPg = simu.mesh.groupElem.Get_gauss(MatrixType.beam).nPg
    assert field.shape[:2] == (simu.mesh.Ne, nPg)
    return np.asarray(field[..., index]).mean(1)


def test_beam_curvature_is_the_moment_over_EI():
    simu, Mz = _cantilever()
    EI = E * _reference.rect_Iz(10, 10)

    curvature = _element_means(simu, simu._Calc_Epsilon(), 1)

    assert rel_error(curvature, Mz / EI) < 1e-10


def test_beam_bending_stress_follows_the_moment():
    """Euler-Bernoulli: Sxx proportional to Mz, Syy = Sxy = 0."""
    simu, Mz = _cantilever()
    sigma = simu._Calc_Sigma()

    Sxx = _element_means(simu, sigma, 0)

    assert rel_error(Sxx / Mz, (Sxx / Mz).mean()) < 1e-10
    assert np.abs(np.asarray(sigma[..., 1:])).max() == 0
