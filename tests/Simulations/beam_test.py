# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import NamedTuple

import numpy as np
import pytest

from EasyFEA import ElemType, Mesh, Mesher, Models, Simulations
from EasyFEA.Geoms import Circle, Domain, Line, Point

from _numerics import rel_error
from . import _reference

# Floor is round-off in high-degree shape functions: SEG2 ~1e-13 ... SEG5 ~3e-9.
TOL = 1e-7
ELEM_TYPES = ElemType.Get_1D()

E, v = 210000.0, 0.3
B = H = 13.0
F, Q = -800.0, -10.0  # point and distributed loads in y, downward
EI = E * _reference.rect_Iz(B, H)
KAPPA = 5 / 6
KGA = KAPPA * E / (2 * (1 + v)) * B * H


def _section(b: float = B, h: float = H, elemType: ElemType = ElemType.TRI3) -> Mesh:
    domain = Domain(Point(-b / 2, -h / 2), Point(b / 2, h / 2))
    return domain.Mesh_2D([], elemType)


def _beam(
    elemType: ElemType,
    beamDim: int,
    length: float,
    loadPoint: float | None = None,
    useTimoshenko: bool = False,
) -> Simulations.Beam:
    """A straight beam along x in `length / 10` elements, with a node at `loadPoint`."""
    line = Line(Point(), Point(x=length), length / 10)
    beam = Models.Beam.Isotropic(beamDim, line, _section(), E, v)
    beam._ky = beam._kz = KAPPA
    additionalPoints = [] if loadPoint is None else [Point(x=loadPoint)]
    mesh = Mesher().Mesh_Beams(
        [beam], additionalPoints=additionalPoints, elemType=elemType
    )
    structure = Models.Beam.BeamStructure([beam])
    return Simulations.Beam(mesh, structure, useTimoshenko=useTimoshenko)


def _clamp(simu: Simulations.Beam, x: float) -> None:
    nodesClamped = simu.mesh.Nodes_Point(Point(x=x))
    simu.add_dirichlet(nodesClamped, [0] * simu.Get_dof_n(), simu.Get_unknowns())


def _pin_and_roll(simu: Simulations.Beam, length: float) -> None:
    """Pinned at x=0, roller at x=`length`; in 3D, torsion held at the pin."""
    nodesPinned = simu.mesh.Nodes_Point(Point())
    nodesRoller = simu.mesh.Nodes_Point(Point(x=length))
    if "z" not in simu.Get_unknowns():
        simu.add_dirichlet(nodesPinned, [0, 0], ["x", "y"])
        simu.add_dirichlet(nodesRoller, [0], ["y"])
    else:
        simu.add_dirichlet(nodesPinned, [0, 0, 0, 0], ["x", "y", "z", "rx"])
        simu.add_dirichlet(nodesRoller, [0, 0], ["y", "z"])


class Case(NamedTuple):
    """A loaded beam: its closed form, supports, and the point load position (None: distributed)."""

    fields: _reference.BeamFields
    supports: str
    loadPoint: float | None


def _cases(length: float) -> dict[str, Case]:
    a, b = length / 2, 0.8 * length
    return {
        "cantilever tip": Case(
            _reference.cantilever_tip(length, F, EI), "cantilever", length
        ),
        "cantilever distributed": Case(
            _reference.cantilever_distributed(length, Q, EI), "cantilever", None
        ),
        "simply supported symmetric": Case(
            _reference.simply_supported(length, a, F, EI), "simply supported", a
        ),
        "simply supported eccentric": Case(
            _reference.simply_supported(length, b, F, EI), "simply supported", b
        ),
        "simply supported distributed": Case(
            _reference.simply_supported_distributed(length, Q, EI),
            "simply supported",
            None,
        ),
        "fixed-fixed symmetric": Case(
            _reference.fixed_fixed(length, a, F, EI), "fixed-fixed", a
        ),
        "fixed-fixed eccentric": Case(
            _reference.fixed_fixed(length, b, F, EI), "fixed-fixed", b
        ),
    }


def _solve(
    case: Case,
    elemType: ElemType,
    beamDim: int,
    length: float,
    useTimoshenko: bool = False,
) -> Simulations.Beam:
    simu = _beam(elemType, beamDim, length, case.loadPoint, useTimoshenko)
    if case.supports == "cantilever":
        _clamp(simu, 0.0)
    elif case.supports == "simply supported":
        _pin_and_roll(simu, length)
    else:
        _clamp(simu, 0.0)
        _clamp(simu, length)
    if case.loadPoint is None:
        simu.add_lineLoad(simu.mesh.nodes, [Q], ["y"])
    else:
        nodesLoaded = simu.mesh.Nodes_Point(Point(x=case.loadPoint))
        simu.add_neumann(nodesLoaded, [F], ["y"])
    simu.Solve()
    return simu


def _x_nodes_and_centers(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    x_n = mesh.coord[:, 0]
    return x_n, x_n[mesh.connect].mean(1)


L_EB = 120.0
EB_CASES = _cases(L_EB)


@pytest.mark.parametrize("beamDim", [2, 3])
@pytest.mark.parametrize("elemType", ELEM_TYPES)
@pytest.mark.parametrize("case", EB_CASES.values(), ids=EB_CASES.keys())
def test_euler_bernoulli_beam_matches_its_closed_form(
    case: Case, elemType: ElemType, beamDim: int
):
    """Nodal uy and rz are exact; so are the element shear forces and, under a point load, the piecewise-linear moment."""
    simu = _solve(case, elemType, beamDim, L_EB)
    x_n, x_e = _x_nodes_and_centers(simu.mesh)
    exact = case.fields

    assert rel_error(simu.Result("uy"), exact.uy(x_n)) < TOL
    assert rel_error(simu.Result("rz"), exact.rz(x_n)) < TOL
    assert rel_error(simu.Result("Ty", nodeValues=False), exact.Ty(x_e)) < TOL
    if case.loadPoint is not None:
        assert rel_error(simu.Result("Mz", nodeValues=False), exact.Mz(x_e)) < TOL


# Stocky (L/h ≈ 1.5), so the shear deflection is ~33 % of the bending one. Selective
# reduced integration: exact once the Lagrange degree reaches that of uy (cubic for a
# tip load, quartic when distributed), O(h²) on SEG2.
L_TIMOSHENKO = 20.0
TIMOSHENKO_CASES = {
    name: _cases(L_TIMOSHENKO)[name]
    for name in (
        "cantilever tip",
        "cantilever distributed",
        "simply supported distributed",
    )
}
TOLS_TIP = {"SEG2": 2e-3, "SEG3": 1e-11, "SEG4": 1e-11, "SEG5": 1e-11}
TOLS_DISTRIBUTED = {"SEG2": 1e-2, "SEG3": 1e-5, "SEG4": 1e-6, "SEG5": 1e-11}


@pytest.mark.parametrize("beamDim", [2, 3])
@pytest.mark.parametrize("elemType", ELEM_TYPES)
@pytest.mark.parametrize("case", TIMOSHENKO_CASES.values(), ids=TIMOSHENKO_CASES.keys())
def test_timoshenko_beam_adds_the_shear_deflection(
    case: Case, elemType: ElemType, beamDim: int
):
    simu = _solve(case, elemType, beamDim, L_TIMOSHENKO, useTimoshenko=True)
    x_n, x_e = _x_nodes_and_centers(simu.mesh)
    exact = _reference.timoshenko(case.fields, KGA)
    tols = TOLS_TIP if case.loadPoint is not None else TOLS_DISTRIBUTED
    tol = tols[str(elemType)]

    assert rel_error(simu.Result("uy"), exact.uy(x_n)) < tol
    if case.supports == "cantilever":
        assert rel_error(simu.Result("rz"), exact.rz(x_n)) < tol
    assert rel_error(simu.Result("Ty", nodeValues=False), exact.Ty(x_e)) < tol
    if case.loadPoint is not None:
        assert rel_error(simu.Result("Mz", nodeValues=False), exact.Mz(x_e)) < TOL


def test_slender_timoshenko_cantilever_on_a_tri6_section():
    """As `examples/Beam/Beam2.py`: shear correction computed from the TRI6 section, SEG3 exact for the cubic deflection."""
    line = Line(Point(), Point(x=L_EB), L_EB / 10)
    section = _section(elemType=ElemType.TRI6)
    beam = Models.Beam.Isotropic(2, line, section, E, v)
    kGA = beam._ky * E / (2 * (1 + v)) * B * H
    mesh = Mesher().Mesh_Beams([beam], elemType=ElemType.SEG3)
    simu = Simulations.Beam(mesh, Models.Beam.BeamStructure([beam]), useTimoshenko=True)
    _clamp(simu, 0.0)
    nodesTip = mesh.Nodes_Point(Point(x=L_EB))
    simu.add_neumann(nodesTip, [F], ["y"])
    simu.Solve()

    x_n, x_e = _x_nodes_and_centers(mesh)
    exact = _reference.timoshenko(_reference.cantilever_tip(L_EB, F, EI), kGA)
    assert rel_error(simu.Result("uy"), exact.uy(x_n)) < 1e-9
    assert rel_error(simu.Result("rz"), exact.rz(x_n)) < 1e-9
    assert rel_error(simu.Result("Mz", nodeValues=False), exact.Mz(x_e)) < TOL
    assert rel_error(simu.Result("Ty", nodeValues=False), exact.Ty(x_e)) < TOL


@pytest.mark.parametrize("beamDim", [1, 2, 3])
@pytest.mark.parametrize("elemType", ELEM_TYPES)
def test_axial_bar_under_tip_force_and_own_weight(elemType: ElemType, beamDim: int):
    length, b, h = 10.0, 0.1, 0.1
    E_steel, rho, g = 200e9, 7800.0, 10.0
    P = 5000.0
    q = rho * g * b * h
    line = Line(Point(), Point(x=length), length / 10)
    beam = Models.Beam.Isotropic(beamDim, line, _section(b, h), E_steel, v)
    mesh = Mesher().Mesh_Beams([beam], elemType=elemType)
    simu = Simulations.Beam(mesh, Models.Beam.BeamStructure([beam]))
    simu.rho = rho
    _clamp(simu, 0.0)
    nodesLine = mesh.Nodes_Line(line)
    nodesTip = mesh.Nodes_Point(Point(x=length))
    simu.add_lineLoad(nodesLine, [q], ["x"])
    simu.add_neumann(nodesTip, [P], ["x"])
    simu.Solve()

    x_n, x_e = _x_nodes_and_centers(mesh)
    exact = _reference.axial_bar(length, P, q, E_steel * b * h)
    assert rel_error(simu.mass, length * b * h * rho) < 1e-12
    assert rel_error(simu.Result("ux"), exact.ux(x_n)) < TOL
    assert rel_error(simu.Result("N", nodeValues=False), exact.N(x_e)) < TOL


@pytest.mark.parametrize("useTimoshenko", [False, True], ids=["EB", "Timoshenko"])
@pytest.mark.parametrize("elemType", ELEM_TYPES)
def test_welded_l_frame_tip_displacement(elemType: ElemType, useTimoshenko: bool):
    """Clamped at the origin, welded at (L, 0), pushed in +x at (L, L).

    The horizontal beam carries a constant moment -F·L; the vertical one bends as a cantilever on a base rotating by -F·L²/EI. At the tip: ux = F·L/EA + 4·F·L³/(3·EI) (+ F·L/kGA), |uy| = F·L³/(2·EI).
    """
    length, b, h = 10.0, 0.5, 0.5
    E_steel, force = 210e9, 1000.0
    A, EI_frame = b * h, E_steel * _reference.rect_Iz(b, h)
    kGA = KAPPA * E_steel / (2 * (1 + v)) * A
    section = _section(b, h)
    line1 = Line(Point(), Point(length, 0), length / 10)
    line2 = Line(Point(length, 0), Point(length, length), length / 10)
    beams = [
        Models.Beam.Isotropic(2, line, section, E_steel, v) for line in (line1, line2)
    ]
    for beam in beams:
        beam._ky = KAPPA
    mesh = Mesher().Mesh_Beams(beams, elemType=elemType)
    simu = Simulations.Beam(
        mesh, Models.Beam.BeamStructure(beams), useTimoshenko=useTimoshenko
    )
    nodesClamped = mesh.Nodes_Point(Point())
    nodesCorner = mesh.Nodes_Point(Point(length, 0))
    nodesTip = mesh.Nodes_Point(Point(length, length))
    simu.add_dirichlet(nodesClamped, [0, 0, 0], ["x", "y", "rz"])
    simu.add_connection_fixed(nodesCorner)
    simu.add_neumann(nodesTip, [force], ["x"])
    simu.Solve()

    shear = force * length / kGA if useTimoshenko else 0.0
    ux = force * length / (E_steel * A) + 4 * force * length**3 / (3 * EI_frame)
    uy = force * length**3 / (2 * EI_frame)
    # SEG2's linear deflection converges in O(h²); SEG3+ are exact
    tol = 1e-3 if elemType == "SEG2" else 1e-8
    assert rel_error(simu.Result("ux")[nodesTip], ux + shear) < tol
    assert rel_error(np.abs(simu.Result("uy")[nodesTip]), uy) < tol


# Saint-Venant shear correction from the section: TRI6 captures the cubic warping
# function to O(h³), ~1e-5 at these sizes.


@pytest.mark.parametrize(
    "b, h",
    [(10.0, 10.0), (10.0, 30.0), (30.0, 10.0), (2.0, 20.0)],
    ids=["square", "tall", "wide", "thin-tall"],
)
@pytest.mark.parametrize("axis", ["y", "z"])
def test_rectangle_shear_correction_is_five_sixths(b: float, h: float, axis: str):
    domain = Domain((-b / 2, -h / 2), (b / 2, h / 2), meshSize=min(b, h) / 10)
    section = domain.Mesh_2D([], ElemType.TRI6)
    beam = Models.Beam.Isotropic(2, Line((0, 0), (10, 0), 1), section, 210e9, 0.3)

    assert rel_error(beam._Get_shear_correction_factor(axis), 5 / 6) < 1e-4


@pytest.mark.parametrize("diam", [5.0, 10.0, 20.0])
@pytest.mark.parametrize("axis", ["y", "z"])
def test_circle_shear_correction_is_six_sevenths(diam: float, axis: str):
    circle = Circle(Point(), diam=diam, meshSize=diam / 30)
    section = circle.Mesh_2D([], ElemType.TRI6)
    beam = Models.Beam.Isotropic(2, Line((0, 0), (10, 0), 1), section, 210e9, 0.3)

    assert rel_error(beam._Get_shear_correction_factor(axis), 6 / 7) < 1e-4


def test_shear_correction_converges_with_the_section_mesh():
    line = Line(Point(), Point(x=10), 1)
    errors = []
    for meshSize in [2.0, 1.0, 0.5, 0.25]:
        domain = Domain(Point(-5, -5), Point(5, 5), meshSize=meshSize)
        beam = Models.Beam.Isotropic(2, line, domain.Mesh_2D(), 210e9, 0.3)
        errors.append(rel_error(beam._Get_shear_correction_factor("y"), 5 / 6))

    assert all(fine <= coarse + 1e-12 for fine, coarse in zip(errors[1:], errors))
    assert errors[-1] <= 5e-4


def test_shear_correction_rejects_the_beam_axis():
    beam = Models.Beam.Isotropic(
        2, Line(Point(), Point(x=10), 1), _section(10.0, 10.0), 210e9, 0.3
    )
    with pytest.raises(ValueError):
        beam._Get_shear_correction_factor("x")


CHANGES = {
    "E": lambda beam, section: setattr(beam, "E", 2 * beam.E),
    "v": lambda beam, section: setattr(beam, "v", 0.4),
    "section": lambda beam, section: setattr(beam, "section", section),
}


@pytest.mark.parametrize("change", CHANGES.values(), ids=CHANGES.keys())
@pytest.mark.parametrize("which", [0, 1], ids=["first beam", "second beam"])
def test_a_beam_change_flags_the_simulation_for_reassembly(change, which: int):
    section = _section(0.01, 0.01)
    rotated = section.copy()
    rotated.Rotate(30, rotated.center)
    beams = [
        Models.Beam.Isotropic(2, Line(Point(), Point(5)), section, 210e9, v=0.1),
        Models.Beam.Isotropic(2, Line(Point(5), Point(10)), section, 210e9, v=0.1),
    ]
    mesh = Mesher().Mesh_Beams(beams)
    simu = Simulations.Beam(mesh, Models.Beam.BeamStructure(beams))
    simu.Get_K_C_M_F()
    assert not simu.needUpdate

    change(beams[which], rotated)

    assert simu.needUpdate


@pytest.mark.parametrize("E, v", [(np.full(3, 210e9), 0.3), (210e9, np.full(3, 0.3))])
def test_beam_parameters_are_scalar(E, v):
    line = Line(Point(), Point(x=1.0), 0.1)

    with pytest.raises(AssertionError, match="scalar"):
        Models.Beam.Isotropic(2, line, _section(0.1, 0.1), E, v)
