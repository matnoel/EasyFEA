# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Callable, NamedTuple

import numpy as np
import pytest

from EasyFEA import ElemType, Mesh, Mesher, Models, Simulations
from EasyFEA.FEM import FeArray, MatrixType
from EasyFEA.Geoms import Domain, Line, Point

from _numerics import rel_error
from . import _reference

# Floor is round-off in high-degree shape functions: SEG2 ~1e-13 ... SEG5 ~3e-9.
TOL = 1e-7

E, v = 210000.0, 0.3
B = H = 13.0
F, Q = -800.0, -10.0  # point and distributed loads in y, downward
EI = E * _reference.rect_Iz(B, H)
KAPPA = 5 / 6


def _kGA(E: float, A: float, kappa: float = KAPPA) -> float:
    return kappa * E / (2 * (1 + v)) * A


KGA = _kGA(E, B * H)


def _section(b: float = B, h: float = H, elemType: ElemType = ElemType.TRI3) -> Mesh:
    domain = Domain((-b / 2, -h / 2), (b / 2, h / 2))
    return domain.Mesh_2D([], elemType)


def _beam(
    elemType: ElemType,
    beamDim: int,
    length: float,
    loadPoint: float | None = None,
    useTimoshenko: bool = False,
) -> Simulations.Beam:
    """A straight beam along x in `length / 10` elements, with a node at `loadPoint`."""
    line = Line((0, 0), (length, 0), length / 10)
    beam = Models.Beam.Isotropic(
        beamDim,
        line,
        _section(),
        E,
        v,
    )
    beam._ky = beam._kz = KAPPA
    additionalPoints = [] if loadPoint is None else [Point(x=loadPoint)]
    mesh = Mesher().Mesh_Beams(
        [beam], additionalPoints=additionalPoints, elemType=elemType
    )
    structure = Models.Beam.BeamStructure([beam])
    return Simulations.Beam(mesh, structure, useTimoshenko=useTimoshenko)


def _clamp(simu: Simulations.Beam, x: float) -> None:
    nodesClamped = simu.mesh.Nodes_Point((x, 0))
    simu.add_dirichlet(nodesClamped, [0] * simu.Get_dof_n(), simu.Get_unknowns())


def _cantilever(simu: Simulations.Beam, length: float) -> None:
    _clamp(simu, 0.0)


def _fixed_fixed(simu: Simulations.Beam, length: float) -> None:
    _clamp(simu, 0.0)
    _clamp(simu, length)


def _simply_supported(simu: Simulations.Beam, length: float) -> None:
    """Pinned at x=0, roller at x=`length`; in 3D, torsion held at the pin."""
    nodesPinned = simu.mesh.Nodes_Point((0, 0))
    nodesRoller = simu.mesh.Nodes_Point((length, 0))
    if "z" not in simu.Get_unknowns():
        simu.add_dirichlet(nodesPinned, [0, 0], ["x", "y"])
        simu.add_dirichlet(nodesRoller, [0], ["y"])
    else:
        simu.add_dirichlet(nodesPinned, [0, 0, 0, 0], ["x", "y", "z", "rx"])
        simu.add_dirichlet(nodesRoller, [0, 0], ["y", "z"])


class Case(NamedTuple):
    """A loaded beam: its closed form, supports, and the point load position (None: distributed)."""

    fields: _reference.BeamFields
    supports: Callable[[Simulations.Beam, float], None]
    loadPoint: float | None


def _cases(length: float) -> dict[str, Case]:
    a, b = length / 2, 0.8 * length
    return {
        "cantilever tip": Case(
            _reference.cantilever_tip(length, F, EI), _cantilever, length
        ),
        "cantilever distributed": Case(
            _reference.cantilever_distributed(length, Q, EI), _cantilever, None
        ),
        "simply supported symmetric": Case(
            _reference.simply_supported(length, a, F, EI), _simply_supported, a
        ),
        "simply supported eccentric": Case(
            _reference.simply_supported(length, b, F, EI), _simply_supported, b
        ),
        "simply supported distributed": Case(
            _reference.simply_supported_distributed(length, Q, EI),
            _simply_supported,
            None,
        ),
        "fixed-fixed symmetric": Case(
            _reference.fixed_fixed(length, a, F, EI), _fixed_fixed, a
        ),
        "fixed-fixed eccentric": Case(
            _reference.fixed_fixed(length, b, F, EI), _fixed_fixed, b
        ),
    }


def _solve(
    case: Case,
    elemType: ElemType,
    beamDim: int,
    length: float,
    useTimoshenko: bool = False,
) -> Simulations.Beam:
    simu = _beam(
        elemType,
        beamDim,
        length,
        case.loadPoint,
        useTimoshenko,
    )
    case.supports(simu, length)
    if case.loadPoint is None:
        simu.add_lineLoad(simu.mesh.nodes, [Q], ["y"])
    else:
        nodesLoaded = simu.mesh.Nodes_Point((case.loadPoint, 0))
        simu.add_neumann(nodesLoaded, [F], ["y"])
    simu.Solve()
    return simu


def _x_nodes_and_centers(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    x_n = mesh.coord[:, 0]
    return x_n, x_n[mesh.connect].mean(1)


L_EB = 120.0
EB_CASES = _cases(L_EB)


def test_Beam_euler_bernoulli_matches_its_closed_form(subtests):
    """Nodal uy and rz are exact; so are the element shear forces and, under a point load, the piecewise-linear moment."""
    for name, case in EB_CASES.items():
        for elemType in ElemType.Get_1D():
            for beamDim in (2, 3):
                with subtests.test(case=name, elemType=str(elemType), beamDim=beamDim):
                    simu = _solve(case, elemType, beamDim, L_EB)
                    x_n, x_e = _x_nodes_and_centers(simu.mesh)
                    exact = case.fields

                    assert rel_error(simu.Result("uy"), exact.uy(x_n)) < TOL
                    assert rel_error(simu.Result("rz"), exact.rz(x_n)) < TOL
                    Ty = simu.Result("Ty", nodeValues=False)
                    assert rel_error(Ty, exact.Ty(x_e)) < TOL
                    if case.loadPoint is not None:
                        Mz = simu.Result("Mz", nodeValues=False)
                        assert rel_error(Mz, exact.Mz(x_e)) < TOL


# Stocky (L/h ≈ 1.5): shear is ~33 % of the deflection. Exact once the degree reaches uy's.
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
TOLS_DISTRIBUTED = {"SEG2": 2e-2, "SEG3": 1e-5, "SEG4": 1e-6, "SEG5": 1e-11}


def test_Beam_timoshenko_adds_the_shear_deflection(subtests):
    for name, case in TIMOSHENKO_CASES.items():
        exact = _reference.timoshenko(case.fields, KGA)
        tols = TOLS_TIP if case.loadPoint is not None else TOLS_DISTRIBUTED
        for elemType in ElemType.Get_1D():
            tol = tols[str(elemType)]
            for beamDim in (2, 3):
                with subtests.test(case=name, elemType=str(elemType), beamDim=beamDim):
                    simu = _solve(
                        case,
                        elemType,
                        beamDim,
                        L_TIMOSHENKO,
                        useTimoshenko=True,
                    )
                    x_n, x_e = _x_nodes_and_centers(simu.mesh)

                    assert rel_error(simu.Result("uy"), exact.uy(x_n)) < tol
                    assert rel_error(simu.Result("rz"), exact.rz(x_n)) < tol
                    Ty = simu.Result("Ty", nodeValues=False)
                    assert rel_error(Ty, exact.Ty(x_e)) < tol
                    if case.loadPoint is not None:
                        Mz = simu.Result("Mz", nodeValues=False)
                        assert rel_error(Mz, exact.Mz(x_e)) < TOL


def test_Beam_slender_timoshenko_cantilever_on_a_tri6_section():
    """As `examples/Beam/Beam2.py`: shear correction computed from the TRI6 section, SEG3 exact for the cubic deflection."""
    line = Line((0, 0), (L_EB, 0), L_EB / 10)
    section = _section(elemType=ElemType.TRI6)
    beam = Models.Beam.Isotropic(
        2,
        line,
        section,
        E,
        v,
    )
    kGA = _kGA(E, B * H, beam._ky)
    mesh = Mesher().Mesh_Beams([beam], elemType=ElemType.SEG3)
    simu = Simulations.Beam(mesh, Models.Beam.BeamStructure([beam]), useTimoshenko=True)
    _clamp(simu, 0.0)
    nodesTip = mesh.Nodes_Point((L_EB, 0))
    simu.add_neumann(nodesTip, [F], ["y"])
    simu.Solve()

    x_n, x_e = _x_nodes_and_centers(mesh)
    exact = _reference.timoshenko(_reference.cantilever_tip(L_EB, F, EI), kGA)
    assert rel_error(simu.Result("uy"), exact.uy(x_n)) < 1e-9
    assert rel_error(simu.Result("rz"), exact.rz(x_n)) < 1e-9
    assert rel_error(simu.Result("Mz", nodeValues=False), exact.Mz(x_e)) < TOL
    assert rel_error(simu.Result("Ty", nodeValues=False), exact.Ty(x_e)) < TOL


def test_Beam_axial_bar_under_tip_force_and_own_weight(subtests):
    length, b, h = 10.0, 0.1, 0.1
    E_steel, rho, g = 200e9, 7800.0, 10.0
    P = 5000.0
    q = rho * g * b * h
    line = Line((0, 0), (length, 0), length / 10)
    exact = _reference.axial_bar(length, P, q, E_steel * b * h)
    for elemType in ElemType.Get_1D():
        for beamDim in (1, 2, 3):
            with subtests.test(elemType=str(elemType), beamDim=beamDim):
                beam = Models.Beam.Isotropic(
                    beamDim,
                    line,
                    _section(b, h),
                    E_steel,
                    v,
                )
                mesh = Mesher().Mesh_Beams([beam], elemType=elemType)
                simu = Simulations.Beam(mesh, Models.Beam.BeamStructure([beam]))
                simu.rho = rho
                _clamp(simu, 0.0)
                nodesLine = mesh.Nodes_Line(line)
                nodesTip = mesh.Nodes_Point((length, 0))
                simu.add_lineLoad(nodesLine, [q], ["x"])
                simu.add_neumann(nodesTip, [P], ["x"])
                simu.Solve()

                x_n, x_e = _x_nodes_and_centers(mesh)
                assert rel_error(simu.mass, length * b * h * rho) < 1e-12
                assert rel_error(simu.Result("ux"), exact.ux(x_n)) < TOL
                N = simu.Result("N", nodeValues=False)
                assert rel_error(N, exact.N(x_e)) < TOL


IY = _reference.rect_Iz(H, B)
STRAIN_FORCE_STIFFNESS = [
    (1, "ux'", "N", E * B * H),
    (2, "ux'", "N", E * B * H),
    (2, "rz'", "Mz", EI),
    (3, "ux'", "N", E * B * H),
    (3, "rx'", "Mx", E / (2 * (1 + v)) * (_reference.rect_Iz(B, H) + IY)),
    (3, "ry'", "My", E * IY),
    (3, "rz'", "Mz", EI),
]


def test_Beam__Calc_Epsilon_is_the_internal_force_over_the_stiffness(subtests):
    solved: dict[int, Simulations.Beam] = {}
    for beamDim in (1, 2, 3):
        simu = _beam(ElemType.SEG3, beamDim, L_EB)
        _cantilever(simu, L_EB)
        nodesTip = simu.mesh.Nodes_Point((L_EB, 0))
        simu.add_neumann(nodesTip, [F] * simu.Get_dof_n(), simu.Get_unknowns())
        simu.Solve()
        solved[beamDim] = simu
    for beamDim, strain, force, stiffness in STRAIN_FORCE_STIFFNESS:
        with subtests.test(beamDim=beamDim, strain=strain):
            simu = solved[beamDim]

            expected = simu.Result(force, nodeValues=False) / stiffness

            assert np.abs(expected).max() > 0
            assert rel_error(simu.Result(strain, nodeValues=False), expected) < TOL


def _weld_an_l_frame(elemType: ElemType, useTimoshenko: bool) -> None:
    """Solves the L-frame and checks its tip against the closed form."""
    length, b, h = 10.0, 0.5, 0.5
    E_steel, force = 210e9, 1000.0
    A, EI_frame = b * h, E_steel * _reference.rect_Iz(b, h)
    kGA = _kGA(E_steel, A)
    section = _section(b, h)
    line1 = Line((0, 0), (length, 0), length / 10)
    line2 = Line((length, 0), (length, length), length / 10)
    beams = [
        Models.Beam.Isotropic(
            2,
            line,
            section,
            E_steel,
            v,
        )
        for line in (line1, line2)
    ]
    for beam in beams:
        beam._ky = KAPPA
    mesh = Mesher().Mesh_Beams(beams, elemType=elemType)
    simu = Simulations.Beam(
        mesh, Models.Beam.BeamStructure(beams), useTimoshenko=useTimoshenko
    )
    nodesClamped = mesh.Nodes_Point((0, 0))
    nodesCorner = mesh.Nodes_Point((length, 0))
    nodesTip = mesh.Nodes_Point((length, length))
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


def test_Beam_add_connection_fixed_welds_an_l_frame(subtests):
    """Clamped at the origin, welded at (L, 0), pushed in +x at (L, L): the vertical leg bends on a base rotated by the horizontal one."""
    for elemType in ElemType.Get_1D():
        for useTimoshenko in (False, True):
            with subtests.test(elemType=str(elemType), useTimoshenko=useTimoshenko):
                _weld_an_l_frame(elemType, useTimoshenko)


CHANGES = {
    "E": lambda beam, section: setattr(beam, "E", 2 * beam.E),
    "v": lambda beam, section: setattr(beam, "v", 0.4),
    "section": lambda beam, section: setattr(beam, "section", section),
}


def test_Beam_structure_change_flags_reassembly(
    subtests, assert_change_flags_reassembly
):
    section = _section(0.01, 0.01)
    rotated = section.copy()
    rotated.Rotate(30, rotated.center)
    lines = [Line((0, 0), (5, 0)), Line((5, 0), (10, 0))]
    for name, change in CHANGES.items():
        for which in (0, 1):
            with subtests.test(change=name, beam=which):
                beams = [
                    Models.Beam.Isotropic(
                        2,
                        line,
                        section,
                        210e9,
                        v=0.1,
                    )
                    for line in lines
                ]
                mesh = Mesher().Mesh_Beams(beams)
                simu = Simulations.Beam(mesh, Models.Beam.BeamStructure(beams))

                assert_change_flags_reassembly(
                    simu, lambda: change(beams[which], rotated)
                )


def test_Beam_accepts_a_beam_or_a_list_of_beams():
    line = Line((0, 0), (L_EB, 0), L_EB / 10)
    beam = Models.Beam.Isotropic(2, line, _section(), E, v)
    mesh = Mesher().Mesh_Beams([beam])

    for structure in (beam, [beam]):
        simu = Simulations.Beam(mesh, structure)

        assert isinstance(simu.structure, Models.Beam.BeamStructure)
        assert simu.structure.beams == [beam]


LINE_LOADS = [(1, "x", False), (2, "y", False), (3, "z", False)]
LINE_LOADS += [(2, "y", True), (3, "z", True)]  # no shear in 1D


def test_Beam_add_lineLoad_sums_to_the_load_times_the_length(subtests):
    for beamDim, unknown, useTimoshenko in LINE_LOADS:
        with subtests.test(beamDim=beamDim, useTimoshenko=useTimoshenko):
            simu = _beam(ElemType.SEG3, beamDim, L_EB, useTimoshenko=useTimoshenko)

            simu.add_lineLoad(simu.mesh.nodes, [Q], [unknown])

            index = simu.Get_unknowns().index(unknown)
            total = simu.Bc_vector_Neumann()[index :: simu.Get_dof_n()].sum()
            assert rel_error(total, Q * L_EB) < 1e-12


def test_Beam_rejects_surface_and_volume_loads():
    simu = _beam(ElemType.SEG2, 2, L_EB)
    nodes = simu.mesh.nodes

    simu.add_surfLoad(nodes, [1.0], ["y"])
    simu.add_volumeLoad(nodes, [1.0], ["y"])

    assert simu.Bc_Neuman == []


def test_Beam_nodal_forces_are_the_load_and_the_reaction():
    simu = _solve(EB_CASES["cantilever tip"], ElemType.SEG3, 2, L_EB)
    nodesClamped = simu.mesh.Nodes_Point((0, 0))
    nodesTip = simu.mesh.Nodes_Point((L_EB, 0))

    fy = simu.Result("fy")

    assert rel_error(fy[nodesTip], F) < 1e-9
    assert rel_error(fy[nodesClamped], -F) < 1e-9
    assert rel_error(simu.Result("cz")[nodesClamped], -F * L_EB) < 1e-9


def test_Beam_iteration_summary_reports_the_extremes(subtests):
    for beamDim in (1, 2, 3):
        with subtests.test(beamDim=beamDim):
            simu = _beam(ElemType.SEG2, beamDim, L_EB)
            _cantilever(simu, L_EB)
            nodesTip = simu.mesh.Nodes_Point((L_EB, 0))
            simu.add_neumann(nodesTip, [F], ["x"])
            simu.Solve()

            summary = simu.Results_Get_Iteration_Summary()

            assert f"Ux min = {F * L_EB / (E * B * H):.2e}" in summary
            assert ("Uy max" in summary) == (beamDim > 1)
            assert ("Uz max" in summary) == (beamDim == 3)


def test_Beam_structure_is_the_given_one():
    simu = _beam(ElemType.SEG2, 2, L_EB)

    assert simu.structure.nBeam == 1
    assert simu.Get_dof_n() == simu.structure.dof_n == 3


def test_Beam_displacement_holds_every_dof_of_every_node(subtests):
    for beamDim in (1, 2, 3):
        with subtests.test(beamDim=beamDim):
            simu = _beam(ElemType.SEG2, beamDim, L_EB)
            u = np.linspace(0, 1, simu.mesh.Nn * simu.Get_dof_n())
            simu._Set_solutions(simu.problemType, u)

            displacement = simu.displacement.reshape(simu.mesh.Nn, -1)

            assert displacement.shape[1] == simu.Get_dof_n()
            assert np.array_equal(displacement[:, 0], simu.Result("ux"))


def _two_beams(beamDim: int, a: float = L_EB / 2) -> Simulations.Beam:
    """Two collinear beams meeting at x=a, clamped at x=0 and x=2a, loaded by F at x=a."""
    lines = [Line((0, 0), (a, 0), a / 10), Line((a, 0), (2 * a, 0), a / 10)]
    beams = [Models.Beam.Isotropic(beamDim, line, _section(), E, v) for line in lines]
    mesh = Mesher().Mesh_Beams(beams, elemType=ElemType.SEG3)
    simu = Simulations.Beam(mesh, Models.Beam.BeamStructure(beams))
    _clamp(simu, 0.0)
    _clamp(simu, 2 * a)
    return simu


def _load_the_joint(simu: Simulations.Beam, a: float = L_EB / 2) -> np.ndarray:
    nodesJoint = simu.mesh.Nodes_Point((a, 0))
    assert nodesJoint.size == 2
    simu.add_neumann(nodesJoint, [F], ["y"])
    simu.Solve()
    return nodesJoint


def test_Beam_add_connection_hinged_lets_the_joint_rotate(subtests):
    """Each half is a cantilever under F/2, so the two sides turn opposite ways."""
    a = L_EB / 2
    half = _reference.cantilever_tip(a, F / 2, EI)
    for beamDim, free in ((2, [""]), (3, ["rz"])):
        with subtests.test(beamDim=beamDim):
            simu = _two_beams(beamDim)
            nodesJoint = simu.mesh.Nodes_Point((a, 0))

            simu.add_connection_hinged(nodesJoint, free)
            _load_the_joint(simu)

            assert rel_error(simu.Result("uy")[nodesJoint], half.uy(a)) < 1e-9
            rz = simu.Result("rz")[nodesJoint]
            assert rel_error(rz, half.rz(a) * np.array([1, -1])) < 1e-9


def test_Beam_add_connection_hinged_in_3d_shares_the_rotations_not_freed():
    a = L_EB / 2
    simu = _two_beams(3)
    nodesJoint = simu.mesh.Nodes_Point((a, 0))

    simu.add_connection_hinged(nodesJoint)
    _load_the_joint(simu)

    welded = _reference.fixed_fixed(2 * a, a, F, EI)
    assert rel_error(simu.Result("uy")[nodesJoint], welded.uy(a)) < 1e-9


def test_Beam_add_connection_hinged_in_1d_adds_nothing():
    simu = _two_beams(1)
    nodesJoint = simu.mesh.Nodes_Point((L_EB / 2, 0))

    simu.add_connection_hinged(nodesJoint)

    assert simu.Bc_Lagrange == []


def test_Beam_add_connection_fixed_welds_two_collinear_beams():
    a = L_EB / 2
    simu = _two_beams(2)
    nodesJoint = simu.mesh.Nodes_Point((a, 0))

    simu.add_connection_fixed(nodesJoint)
    _load_the_joint(simu)

    welded = _reference.fixed_fixed(2 * a, a, F, EI)
    assert rel_error(simu.Result("uy")[nodesJoint], welded.uy(a)) < 1e-9
    assert rel_error(simu.Result("rz")[nodesJoint], 0.0, scale=1e-3) < 1e-9


def test_Beam_add_connection_ties_only_the_given_unknowns():
    """Tied in y alone, the joint shares its deflection but not its rotation, as a hinge."""
    a = L_EB / 2
    simu = _two_beams(2)
    nodesJoint = simu.mesh.Nodes_Point((a, 0))

    simu.add_connection(nodesJoint, ["x", "y"], "tied")
    _load_the_joint(simu)

    assert len(simu.Bc_Lagrange) == 2
    assert simu.Bc_Display[-1].description == "tied"
    half = _reference.cantilever_tip(a, F / 2, EI)
    assert rel_error(simu.Result("uy")[nodesJoint], half.uy(a)) < 1e-9


def test_Beam_add_connection_on_one_node_holds_it():
    simu = _beam(ElemType.SEG2, 2, L_EB)
    nodesTip = simu.mesh.Nodes_Point((L_EB, 0))

    simu.add_connection(nodesTip, ["y"], "held")

    assert simu.Bc_Lagrange == []
    assert np.array_equal(simu.Bc_dofs_Dirichlet(), simu.Bc_dofs_nodes(nodesTip, ["y"]))


INDICES = {
    1: {"ux": 0, "fx": 0, "N": 0},
    2: {"ux": 0, "uy": 1, "rz": 2, "fy": 1, "cz": 2, "N": 0, "Mz": 1},
    3: {
        "ux": 0,
        "uy": 1,
        "uz": 2,
        "rx": 3,
        "ry": 4,
        "rz": 5,
        "N": 0,
        "Mx": 1,
        "My": 2,
        "Mz": 3,
    },
}


def test_Beam__indexResult_gives_the_column_of_a_result(subtests):
    for beamDim, indices in INDICES.items():
        with subtests.test(beamDim=beamDim):
            simu = _beam(ElemType.SEG2, beamDim, L_EB)

            columns = {result: simu._indexResult(result) for result in indices}

            assert columns == indices
            with pytest.raises(ValueError):
                simu._indexResult("Ty")


def test_Beam__indexResult_gives_the_shear_columns_under_timoshenko():
    simu = _beam(ElemType.SEG3, 3, L_EB, useTimoshenko=True)

    assert (simu._indexResult("Ty"), simu._indexResult("Tz")) == (4, 5)


def _tip_loaded_cantilever() -> tuple[Simulations.Beam, np.ndarray]:
    """SEG2 cantilever with a tip load, and its exact moment at the element centers."""
    simu = _beam(ElemType.SEG2, 2, L_EB)
    _cantilever(simu, L_EB)
    nodesTip = simu.mesh.Nodes_Point((L_EB, 0))
    simu.add_neumann(nodesTip, [F], ["y"])
    simu.Solve()
    _, x_e = _x_nodes_and_centers(simu.mesh)
    return simu, _reference.cantilever_tip(L_EB, F, EI).Mz(x_e)


def _element_means(simu: Simulations.Beam, field, index: int) -> np.ndarray:
    assert isinstance(field, FeArray)
    nPg = simu.mesh.groupElem.Get_gauss(MatrixType.beam).nPg
    assert field.shape[:2] == (simu.mesh.Ne, nPg)
    return np.asarray(field[..., index]).mean(1)


def test_Beam__Calc_Epsilon_curvature_is_the_moment_over_EI():
    simu, Mz = _tip_loaded_cantilever()

    curvature = _element_means(simu, simu._Calc_Epsilon(), 1)

    assert rel_error(curvature, Mz / EI) < 1e-10


def test_Beam__Calc_Sigma_bending_stress_follows_the_moment():
    """Euler-Bernoulli: Sxx proportional to Mz, Sxy = 0."""
    simu, Mz = _tip_loaded_cantilever()
    sigma = simu._Calc_Sigma()

    Sxx = _element_means(simu, sigma, 0)

    assert rel_error(Sxx / Mz, (Sxx / Mz).mean()) < 1e-10
    assert np.abs(np.asarray(sigma[..., 1:])).max() == 0
