# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np
import pytest

from EasyFEA.FEM import ElemType, Kinematics, MatrixType, Mesher
from EasyFEA.Geoms import Circle, Domain, Line
from EasyFEA.Models.Beam import BeamStructure, Isotropic

# rule 7 exception: beam element groups are built only by Simulations.Beam
from EasyFEA import Simulations

from _numerics import rel_error

TOL = 1e-12
B, H, L = 0.1, 0.2, 2.0
E, v = 210.0, 0.3


def _rectangle(x0: float = 0.0):
    """B along x, H along y, offset by `x0` to check the centering."""
    rectangle = Domain((x0, 0), (x0 + B, H), B / 4)
    return rectangle.Mesh_2D([], ElemType.QUAD4, isOrganised=True)


def _beam(dim: int = 3, line: Line | None = None, section=None, **kwargs) -> Isotropic:
    if line is None:
        line = Line((0, 0), (L, 0), L / 4)
    if section is None:
        section = _rectangle()
    return Isotropic(
        dim,
        line,
        section,
        E,
        v,
        **kwargs,
    )


def _structure(dim: int, useTimoshenko: bool = False, beams=None):
    """(structure, groupElem, x): beams along x, meshed with SEG3."""
    beams = [_beam(dim)] if beams is None else beams
    mesh = Mesher().Mesh_Beams(beams, elemType=ElemType.SEG3)
    structure = BeamStructure(beams)
    simu = Simulations.Beam(mesh, structure, useTimoshenko=useTimoshenko)
    return structure, simu.mesh.groupElem, simu.mesh.coord[:, 0]


def _axial(structure, x, a: float) -> np.ndarray:
    """Nodal dofs of u = a x, nothing else."""
    values = np.zeros((x.size, structure.dof_n))
    values[:, 0] = a * x
    return values.ravel()


# ----------------------------------------------
# _Beam
# ----------------------------------------------


def test_Beam_prints_its_name_and_section_properties():
    beam = _beam()
    text = str(beam)

    assert text.startswith(beam.name)
    assert all(f"\n{name} = " in text for name in ("area", "Iz", "Iy", "J"))


def test_Beam_area_is_B_H():
    assert np.isclose(_beam().area, B * H)


def test_Beam_Iy_is_H_B3_over_12():
    assert np.isclose(_beam().Iy, H * B**3 / 12)


def test_Beam_Iz_is_B_H3_over_12():
    assert np.isclose(_beam().Iz, B * H**3 / 12)


def test_Beam_J_is_Iy_plus_Iz():
    beam = _beam()

    assert np.isclose(beam.J, beam.Iy + beam.Iz)


def test_Beam_line_is_the_neutral_fiber():
    line = Line((0, 0), (L, 0), L / 4)

    assert _beam(line=line).line is line


def test_Beam_section_is_centered_on_its_centroid():
    beam = _beam(section=_rectangle(x0=5.0))

    assert np.allclose(beam.section.center, 0.0)
    assert np.isclose(beam.Iy, H * B**3 / 12)


def test_Beam_section_must_lie_in_the_xy_plane():
    rectangle = Domain((0, 0), (B, H), B / 2)
    section = rectangle.Mesh_Extrude([], [0, 0, B], [1], ElemType.HEXA8)
    with pytest.raises(AssertionError, match="plane"):
        _beam(section=section)


def test_Beam_xAxis_is_the_line_direction():
    diagonal = Line((0, 0), (1, 1), 0.5)
    beam = _beam(line=diagonal)

    assert np.allclose(beam.xAxis, np.array([1, 1, 0]) / np.sqrt(2))


def test_Beam_yAxis_is_made_unit_and_orthogonal_to_the_fiber():
    assert np.allclose(_beam(yAxis=(0, 1, 0)).yAxis, (0, 1, 0))
    assert np.allclose(_beam(yAxis=(1, 1, 0)).yAxis, (0, 1, 0))
    assert np.allclose(_beam(yAxis=(0, 0, 3)).yAxis, (0, 0, 1))


def test_Beam_yAxis_along_the_fiber_falls_back_to_z_cross_x():
    assert np.allclose(_beam(yAxis=(1, 0, 0)).yAxis, (0, 1, 0))


def test_Beam_name_is_unique():
    first, second = _beam(), _beam()

    assert first.name.startswith("beam") and first.name != second.name


DOF_N = {1: 1, 2: 3, 3: 6}
"""Beam dofs per node, per dimension."""


def test_Beam_dof_n_per_dimension():
    assert {dim: _beam(dim).dof_n for dim in DOF_N} == DOF_N


def _D(dim: int, useTimoshenko: bool) -> list[float]:
    A, Iy, Iz = B * H, H * B**3 / 12, B * H**3 / 12
    mu = E / (2 * (1 + v))
    return {
        (1, False): [E * A],
        (2, False): [E * A, E * Iz],
        (2, True): [E * A, E * Iz, mu * 5 / 6 * A],
        (3, False): [E * A, mu * (Iy + Iz), E * Iy, E * Iz],
        (3, True): [
            E * A,
            mu * (Iy + Iz),
            E * Iy,
            E * Iz,
            5 / 6 * mu * A,
            5 / 6 * mu * A,
        ],
    }[(dim, useTimoshenko)]


BEAMS = [(1, False), (2, False), (2, True), (3, False), (3, True)]
"""(dim, useTimoshenko) of every beam kinematics."""


def test_Beam_Get_D_is_the_section_stiffness(subtests):
    for dim, useTimoshenko in BEAMS:
        with subtests.test(dim=dim, useTimoshenko=useTimoshenko):
            beam = _beam(dim)
            beam._ky = beam._kz = 5 / 6
            D = np.diag(_D(dim, useTimoshenko))
            assert rel_error(beam.Get_D(useTimoshenko), D) < TOL


def test_Beam_Get_M_is_the_section_area_on_translations(subtests):
    for dim, diagonal in [(1, [1]), (2, [1, 1, 0]), (3, [1, 1, 1, 0, 0, 0])]:
        with subtests.test(dim=dim):
            assert rel_error(_beam(dim).Get_M(), B * H * np.diag(diagonal)) < TOL


def test_Beam__Calc_P_has_the_beam_axes_as_columns():
    diagonal = Line((0, 0), (1, 1), 0.5)
    beam = _beam(line=diagonal, yAxis=(0, 0, 1))
    P = beam._Calc_P()

    assert rel_error(P.T @ P, np.eye(3)) < TOL
    assert np.allclose(P[:, 0], beam.xAxis) and np.allclose(P[:, 1], beam.yAxis)
    assert np.allclose(P[:, 2], np.cross(beam.xAxis, beam.yAxis))


def test_Beam__Get_shear_correction_factor_of_a_rectangle_is_5_6(subtests):
    rectangle = Domain((0, 0), (B, H), B / 10)
    section = rectangle.Mesh_2D([], ElemType.QUAD8, isOrganised=True)
    beam = _beam(section=section)
    for axis in ("y", "z"):
        with subtests.test(axis=axis):
            assert abs(beam._Get_shear_correction_factor(axis) - 5 / 6) < 1e-2


def test_Beam__Get_shear_correction_factor_of_a_circle_is_6_7():
    circle = Circle((0, 0), 0.2, 0.02, isFilled=True)
    section = circle.Mesh_2D([], ElemType.TRI6)

    assert abs(_beam(section=section)._Get_shear_correction_factor("y") - 6 / 7) < 1e-2


def test_Beam__Get_shear_correction_factor_rejects_another_axis():
    with pytest.raises(ValueError, match="axis must be"):
        _beam()._Get_shear_correction_factor("x")


# ----------------------------------------------
# Isotropic
# ----------------------------------------------


def test_Isotropic_mu_is_E_over_2_1_plus_v():
    assert np.isclose(_beam().mu, E / (2 * (1 + v)))


def test_Isotropic_rejects_unphysical_moduli(subtests):
    line = Line((0, 0), (L, 0), L / 4)
    section = _rectangle()
    for E_, v_ in [(-1.0, 0.3), (1.0, 0.5), (1.0, -1.0)]:
        with subtests.test(E=E_, v=v_):
            with pytest.raises(AssertionError):
                Isotropic(
                    3,
                    line,
                    section,
                    E_,
                    v_,
                )


def test_Isotropic_rejects_heterogeneous_moduli(subtests):
    line = Line((0, 0), (L, 0), L / 4)
    section = _rectangle()
    for name, E_, v_ in [("E", np.full(3, E), v), ("v", E, np.full(3, v))]:
        with subtests.test(field=name):
            with pytest.raises(AssertionError, match="scalar"):
                Isotropic(
                    2,
                    line,
                    section,
                    E_,
                    v_,
                )


# ----------------------------------------------
# BeamStructure
# ----------------------------------------------


def _two_beams(dim: int) -> list[Isotropic]:
    """Two beams in a row along x, the second twice as wide."""
    wideRectangle = Domain((0, 0), (2 * B, H), B / 4)
    wide = wideRectangle.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    firstLine = Line((0, 0), (L, 0), L / 4)
    secondLine = Line((L, 0), (2 * L, 0), L / 4)
    return [_beam(dim, line=firstLine), _beam(dim, line=secondLine, section=wide)]


def test_BeamStructure_beams_are_the_given_ones():
    beams = _two_beams(2)

    assert BeamStructure(beams).beams == beams


def test_BeamStructure_nBeam_counts_them():
    assert BeamStructure(_two_beams(2)).nBeam == 2


def test_BeamStructure_areas_per_beam():
    assert np.allclose(BeamStructure(_two_beams(2)).areas, [B * H, 2 * B * H])


def test_BeamStructure_dof_n_is_the_beams_one():
    dof_n = {dim: BeamStructure([_beam(dim)]).dof_n for dim in DOF_N}
    assert dof_n == DOF_N


def test_BeamStructure_rejects_mixed_dimensions():
    with pytest.raises(AssertionError, match="identical beams dimensions"):
        BeamStructure([_beam(2), _beam(3)])


def test_BeamStructure_Calc_D_e_pg_gives_each_element_its_beam_D(subtests):
    beams = _two_beams(2)
    for useTimoshenko in (False, True):
        structure, groupElem, _ = _structure(2, useTimoshenko, beams)
        D_e_pg = structure.Calc_D_e_pg(groupElem)
        for index, beam in enumerate(beams):
            with subtests.test(useTimoshenko=useTimoshenko, beam=index):
                elems = groupElem.Get_Elements_Tag(beam.name)
                D = np.broadcast_to(beam.Get_D(useTimoshenko), D_e_pg[elems].shape)
                assert rel_error(D_e_pg[elems], D) < TOL


def test_BeamStructure_Calc_M_e_pg_gives_each_element_its_beam_M(subtests):
    beams = _two_beams(3)
    structure, groupElem, _ = _structure(3, beams=beams)
    M_e_pg = structure.Calc_M_e_pg(groupElem)
    for index, beam in enumerate(beams):
        with subtests.test(beam=index):
            elems = groupElem.Get_Elements_Tag(beam.name)
            M = np.broadcast_to(beam.Get_M(), M_e_pg[elems].shape)
            assert rel_error(M_e_pg[elems], M) < TOL


AXIAL = 1e-3


def _stretched(dim: int, useTimoshenko: bool):
    structure, groupElem, x = _structure(dim, useTimoshenko)
    kinematics = Kinematics(groupElem, _axial(structure, x, AXIAL), MatrixType.beam)
    return structure, kinematics


def test_BeamStructure_Compute_Epsilon_of_a_stretch_is_axial(subtests):
    for dim, useTimoshenko in BEAMS:
        with subtests.test(dim=dim, useTimoshenko=useTimoshenko):
            structure, kinematics = _stretched(dim, useTimoshenko)
            eps = structure.Compute_Epsilon(kinematics)
            expected = np.zeros(eps.shape[-1])
            expected[0] = AXIAL

            assert rel_error(eps, np.broadcast_to(expected, eps.shape), AXIAL) < 1e-10


def test_BeamStructure_Compute_InternalForces_of_a_stretch_is_E_A_eps(subtests):
    for dim, useTimoshenko in BEAMS:
        with subtests.test(dim=dim, useTimoshenko=useTimoshenko):
            structure, kinematics = _stretched(dim, useTimoshenko)
            forces = structure.Compute_InternalForces(kinematics)
            expected = np.zeros(forces.shape[-1])
            expected[0] = E * B * H * AXIAL

            expected = np.broadcast_to(expected, forces.shape)
            assert rel_error(forces, expected, E * B * H * AXIAL) < 1e-10


def test_BeamStructure_Compute_Sigma_of_a_stretch_is_E_eps(subtests):
    for dim, nS in [(1, 1), (2, 3), (3, 6)]:
        with subtests.test(dim=dim):
            structure, kinematics = _stretched(dim, False)
            sigma = structure.Compute_Sigma(kinematics)
            expected = np.zeros(nS)
            expected[0] = E * AXIAL

            expected = np.broadcast_to(expected, sigma.shape)
            assert rel_error(sigma, expected, E * AXIAL) < 1e-10


def test_BeamStructure_Get_axis_e_gives_each_element_its_beam_axes(subtests):
    horizontal = Line((0, 0), (L, 0), L / 4)
    vertical = Line((L, 0), (L, L), L / 4)
    beams = [_beam(3, line=horizontal), _beam(3, line=vertical, yAxis=(-1, 0, 0))]
    mesh = Mesher().Mesh_Beams(beams, elemType=ElemType.SEG2)
    xAxis_e, yAxis_e = BeamStructure(beams).Get_axis_e(mesh.groupElem)
    for index, beam in enumerate(beams):
        with subtests.test(beam=index):
            elems = mesh.groupElem.Get_Elements_Tag(beam.name)
            assert np.allclose(xAxis_e[elems], beam.xAxis)
            assert np.allclose(yAxis_e[elems], beam.yAxis)


def test_BeamStructure_Get_axis_e_is_none_off_a_line_mesh():
    assert BeamStructure([_beam(3)]).Get_axis_e(_rectangle().groupElem) is None
