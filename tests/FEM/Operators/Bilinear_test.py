# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Energies of affine fields are exact: uᵀ K u = measure · (closed-form energy density)."""

import functools

import pytest
import numpy as np

from EasyFEA.FEM import ElemType, Mesher, Operators
from EasyFEA.Geoms import Domain, Line

# beam element groups and their BeamStructure only exist through Simulations.Beam (rule 7 exception)
from EasyFEA import Models, Simulations

from _numerics import G3, affine, isotropic_stiffness, mandel, quadratic_form, rel_error
from .._reference import MEASURES, MeshOf
from ._reference import assert_restricts

Bilinear = Operators.Bilinear

TYPES = [ElemType.TRI3, ElemType.TETRA4]


# ----------------------------------------------------------------------------
# continuum
# ----------------------------------------------------------------------------


def test_GradUGradV_energy_of_an_affine_field(mesh_of: MeshOf, subtests):
    """f = a·x: fᵀ K f = coef · |a|² · measure, and constants are in the kernel."""
    for elemType in TYPES:
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType)
            groupElem = mesh.groupElem
            a = np.array([1.0, -2.0, 0.5])[: groupElem.dim]

            K_e = Bilinear.GradUGradV(groupElem, 3.0)

            f = affine(mesh.coord, a)
            expected = 3.0 * a @ a * MEASURES[groupElem.dim]
            assert rel_error(quadratic_form(groupElem, K_e, f), expected) < 1e-10
            assert np.abs(K_e.sum(-1)).max() < 1e-10 * np.abs(K_e).max()


def test_GradUGradV_restricted_to_elements(mesh_of: MeshOf, subtests):
    for elemType in TYPES:
        with subtests.test(elemType=str(elemType)):
            groupElem = mesh_of(elemType).groupElem
            elements = np.array([0, groupElem.Ne - 1])

            assert_restricts(
                lambda e: Bilinear.GradUGradV(groupElem, elements=e), elements
            )


def test_UV_mass_sums_to_dof_n_times_the_measure(mesh_of: MeshOf, subtests):
    for elemType in TYPES:
        for dof_n in [1, 3]:
            with subtests.test(elemType=str(elemType), dof_n=dof_n):
                groupElem = mesh_of(elemType).groupElem

                M_e = Bilinear.UV(groupElem, 2.0, dof_n)

                assert M_e.shape[-1] == groupElem.nPe * dof_n
                expected = 2.0 * dof_n * MEASURES[groupElem.dim]
                assert rel_error(M_e.sum(), expected) < 1e-12


def test_UV_per_element_coefficient_and_restriction(mesh_of: MeshOf, subtests):
    for elemType in TYPES:
        with subtests.test(elemType=str(elemType)):
            groupElem = mesh_of(elemType).groupElem
            coef_e = np.arange(groupElem.Ne, dtype=float)
            elements = np.arange(0, groupElem.Ne, 3)

            M_e = Bilinear.UV(groupElem, coef_e, elements=elements)

            volume_e = groupElem.Integrate_e()
            expected = coef_e[elements] * volume_e[elements]
            assert rel_error(M_e.sum((1, 2))[elements], expected) < 1e-12
            assert not np.any(np.delete(M_e, elements, axis=0))


def test_dof_n_repeats_the_scalar_block(mesh_of: MeshOf, subtests):
    """dof_n = 3: the scalar block on each component, zero between components."""
    groupElem = mesh_of(ElemType.TETRA4).groupElem
    for operator in [Bilinear.UV, Bilinear.GradUGradV]:
        with subtests.test(operator=operator.__name__):
            scalar = operator(groupElem)
            vector = operator(groupElem, dof_n=3)
            for c in range(3):
                for d in range(3):
                    expected = scalar if c == d else np.zeros_like(scalar)
                    assert np.array_equal(vector[:, c::3, d::3], expected)


def test_UV_mass_of_an_affine_field_on_quadratic_elements(mesh_of: MeshOf, subtests):
    """f = x − 2y + z/2: fᵀ M f = ∫ f² dΩ."""
    for elemType in [ElemType.TETRA10, ElemType.HEXA27, ElemType.PRISM18]:
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType)
            groupElem = mesh.groupElem
            f = affine(mesh.coord, [1.0, -2.0, 0.5])

            M_e = Bilinear.UV(groupElem)

            expected = groupElem.Integrate_e(
                lambda x, y, z: (x - 2 * y + 0.5 * z) ** 2
            ).sum()
            assert rel_error(quadratic_form(groupElem, M_e, f), expected) < 1e-12


def test_LinearizedElasticity_energy_of_an_affine_displacement(
    mesh_of: MeshOf, subtests
):
    """u = G x: uᵀ K u = measure · ε : C : ε."""
    for elemType in TYPES:
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType)
            groupElem = mesh.groupElem
            dim = groupElem.dim
            G = G3[:dim, :dim]
            C = isotropic_stiffness(dim)
            u = affine(mesh.coord, G)

            K_e = Bilinear.LinearizedElasticity(groupElem, C)

            eps = mandel(G)
            energy = quadratic_form(groupElem, K_e, u)
            assert rel_error(energy, MEASURES[dim] * eps @ C @ eps) < 1e-10


def test_LinearizedElasticity_restricted_to_elements(mesh_of: MeshOf, subtests):
    for elemType in TYPES:
        with subtests.test(elemType=str(elemType)):
            groupElem = mesh_of(elemType).groupElem
            C = isotropic_stiffness(groupElem.dim)

            assert_restricts(
                lambda e: Bilinear.LinearizedElasticity(groupElem, C, e), np.array([0])
            )


def test_MassAlongNormal_weighs_only_the_normal_component():
    """A flat face in the xy plane: u = e_z carries the area, u = e_x nothing."""
    contour = Domain((0, 0), (2, 1), 0.5)
    mesh = contour.Mesh_2D([], ElemType.TRI3, isOrganised=True)
    groupElem = mesh.groupElem
    ez = np.tile([0.0, 0, 1], mesh.Nn)
    ex = np.tile([1.0, 0, 0], mesh.Nn)

    K_e = Bilinear.MassAlongNormal(groupElem, 3.0)

    assert rel_error(quadratic_form(groupElem, K_e, ez), 3.0 * 2.0) < 1e-12
    assert quadratic_form(groupElem, K_e, ex) == 0
    assert_restricts(
        lambda e: Bilinear.MassAlongNormal(groupElem, 3.0, e), np.array([0])
    )


def test_GradU_A_GradV_energy_of_an_affine_field(mesh_of: MeshOf, subtests):
    """f = a·x: fᵀ K f = coef · aᵀ A a · measure."""
    A3 = np.array([[2.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 3.0]])
    for elemType in TYPES:
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType)
            groupElem = mesh.groupElem
            dim = groupElem.dim
            a = np.array([1.0, -2.0, 0.5])[:dim]
            A = A3[:dim, :dim]

            K_e = Bilinear.GradU_A_GradV(groupElem, A, 2.0)

            f = affine(mesh.coord, a)
            expected = 2.0 * a @ A @ a * MEASURES[dim]
            assert rel_error(quadratic_form(groupElem, K_e, f), expected) < 1e-10


# ----------------------------------------------------------------------------
# beams
# ----------------------------------------------------------------------------

L = 2.0
E, v = 210.0, 0.3


class Beam:
    """A straight beam along x of length L, its group of beam elements and structure."""

    def __init__(
        self, dim: int, useTimoshenko: bool, elemType: ElemType = ElemType.SEG3
    ):
        contour = Domain((-0.05, -0.1), (0.05, 0.1))
        section = contour.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
        line = Line((0, 0), (L, 0), L / 4)
        self.beam = Models.Beam.Isotropic(dim, line, section, E, v)
        self.beam._ky = self.beam._kz = 5 / 6
        mesh = Mesher().Mesh_Beams([self.beam], elemType=elemType)
        self.structure = Models.Beam.BeamStructure([self.beam])
        simu = Simulations.Beam(mesh, self.structure, useTimoshenko=useTimoshenko)
        self.groupElem = simu.mesh.groupElem
        self.dof_n = self.structure.dof_n
        self.x = simu.mesh.coord[:, 0]

    def field(self, **components) -> np.ndarray:
        """Nodal dofs, each named component a function of x."""
        names = {1: ["u"], 2: ["u", "v", "rz"], 3: ["u", "v", "w", "rx", "ry", "rz"]}
        names = names[self.structure.dim]
        values = np.zeros((self.x.size, self.dof_n))
        for name, func in components.items():
            values[:, names.index(name)] = func(self.x)
        return values.ravel()

    def energy(self, K_e: np.ndarray, u: np.ndarray) -> float:
        return quadratic_form(self.groupElem, K_e, u)


BEAMS = {
    "1D": (1, False),
    "2D Euler-Bernoulli": (2, False),
    "2D Timoshenko": (2, True),
    "3D Euler-Bernoulli": (3, False),
    "3D Timoshenko": (3, True),
}


@functools.cache
def _beam(name: str) -> Beam:
    return Beam(*BEAMS[name])


def _bending(
    beam: Beam, a: float = 0.01, kappa: float = 0.02
) -> tuple[np.ndarray, float]:
    """Stretch u = a x and uniform curvature v = κx²/2, rz = κx, shear free: its energy EA a² L + EI κ² L."""
    model = beam.beam
    if beam.structure.dim == 1:
        return beam.field(u=lambda x: a * x), E * model.area * a**2 * L
    u = beam.field(
        u=lambda x: a * x, v=lambda x: kappa * x**2 / 2, rz=lambda x: kappa * x
    )
    return u, (E * model.area * a**2 + E * model.Iz * kappa**2) * L


def _shear(beam: Beam, gamma: float = 0.01) -> tuple[np.ndarray, float]:
    """v = γ x without rotation: uniform shear, its energy kGA γ² L."""
    model = beam.beam
    return (
        beam.field(v=lambda x: gamma * x),
        model._ky * model.mu * model.area * gamma**2 * L,
    )


def test_BeamBending_energy_of_stretch_and_uniform_curvature(subtests):
    for name in BEAMS:
        with subtests.test(beam=name):
            beam = _beam(name)
            u, expected = _bending(beam)

            K_e = Bilinear.BeamBending(beam.groupElem, beam.structure)

            assert rel_error(beam.energy(K_e, u), expected) < 1e-9


def test_BeamBending_needs_beam_elements():
    beam = _beam("3D Timoshenko")
    contour = Domain((0, 0), (1, 1))
    groupElem = contour.Mesh_2D().groupElem

    with pytest.raises(AssertionError):
        Bilinear.BeamBending(groupElem, beam.structure)


def test_BeamShear_energy_of_a_uniform_shear(subtests):
    """Timoshenko only: Euler-Bernoulli beams and bars carry no shear energy."""
    for name in BEAMS:
        with subtests.test(beam=name):
            beam = _beam(name)
            K_e = Bilinear.BeamShear(beam.groupElem, beam.structure)

            dim, useTimoshenko = BEAMS[name]
            if dim == 1 or not useTimoshenko:
                assert not np.any(K_e)
                continue
            u, expected = _shear(beam)
            assert rel_error(beam.energy(K_e, u), expected) < 1e-9
            assert beam.energy(K_e, _bending(beam)[0]) < 1e-12 * expected


def test_BeamStiffness_is_bending_plus_shear(subtests):
    for name in BEAMS:
        with subtests.test(beam=name):
            beam = _beam(name)
            K_e = Bilinear.BeamStiffness(beam.groupElem, beam.structure)

            expected = Bilinear.BeamBending(beam.groupElem, beam.structure)
            expected += Bilinear.BeamShear(beam.groupElem, beam.structure)
            assert rel_error(K_e, expected) < 1e-12
            u, energy = _bending(beam)
            assert rel_error(beam.energy(K_e, u), energy) < 1e-9


def test_BeamStiffness_leaves_rigid_motions_free(subtests):
    for name in BEAMS:
        with subtests.test(beam=name):
            beam = _beam(name)
            translation = beam.field(u=lambda x: 0 * x + 1)
            if beam.structure.dim > 1:
                translation += beam.field(v=lambda x: 0 * x + 1)
                rotation = beam.field(v=lambda x: 0.1 * x, rz=lambda x: 0 * x + 0.1)
            else:
                rotation = translation

            K_e = Bilinear.BeamStiffness(beam.groupElem, beam.structure)

            scale = np.abs(K_e).max()
            assert abs(beam.energy(K_e, translation)) < 1e-10 * scale
            assert abs(beam.energy(K_e, rotation)) < 1e-10 * scale


def test_BeamMass_of_a_translation_is_the_beam_mass(subtests):
    for name in BEAMS:
        with subtests.test(beam=name):
            beam = _beam(name)
            rho = 7.8
            translation = beam.field(u=lambda x: 0 * x + 1)

            M_e = Bilinear.BeamMass(beam.groupElem, beam.structure, rho)

            assert (
                rel_error(beam.energy(M_e, translation), rho * beam.beam.area * L)
                < 1e-12
            )
            if beam.structure.dim > 1:
                lateral = beam.field(v=lambda x: 0 * x + 1)
                assert (
                    rel_error(beam.energy(M_e, lateral), rho * beam.beam.area * L)
                    < 1e-12
                )
