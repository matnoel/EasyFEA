# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Every strain measure of an affine displacement u = G x is exact at every Gauss point: F = I + G."""

import pytest
import numpy as np

from EasyFEA.FEM import ElemType, MatrixType, Mesh, Kinematics
from EasyFEA.Geoms import Domain, Line

from _numerics import G3, H3, affine, mandel, rel_error, unmandel

T1 = np.array([1.0, 2.0, 2.0]) / 3
T2 = np.array([2.0, -2.0, 1.0]) / 3


def _mesh(dim: int) -> Mesh:
    if dim == 1:
        line = Line((0, 0), (2, 0), 0.5)
        return line.Mesh_1D(ElemType.SEG3)
    contour = Domain((0, 0), (2, 1), 0.5)
    if dim == 2:
        return contour.Mesh_2D([], ElemType.QUAD8, isOrganised=True)
    return contour.Mesh_Extrude([], (0, 0, 1), [2], ElemType.HEXA20, isOrganised=True)


def _pad(A: np.ndarray, dim: int, diagonal: float = 0.0) -> np.ndarray:
    """`A[:dim, :dim]` in a 3 × 3 whose other diagonal entries are `diagonal`."""
    padded = diagonal * np.eye(3)
    padded[:dim, :dim] = A[:dim, :dim]
    return padded


def _mandel_matrix(dim: int, func) -> np.ndarray:
    """Matrix of the linear map `func` on symmetric 3 × 3 tensors, in Kelvin–Mandel components."""
    n = {1: 1, 2: 3, 3: 6}[dim]
    basis = []
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        basis.append(unmandel(e))
    return np.array([mandel(func(B), dim) for B in basis]).T


class Case:
    """Kinematics of u = G x on a `dim` mesh, with its closed-form F and C."""

    def __init__(self, dim: int):
        self.dim = dim
        self.mesh = _mesh(dim)
        self.kinematics = Kinematics(
            self.mesh.groupElem, affine(self.mesh.coord, G3[:dim, :dim])
        )
        self.G = _pad(G3, dim)
        self.F = np.eye(3) + self.G
        self.C = self.F.T @ self.F

    def assert_everywhere(self, values, expected) -> None:
        values = np.asarray(values)
        expected = np.broadcast_to(expected, values.shape)
        assert rel_error(values, expected, scale=1.0) < 1e-10


@pytest.fixture(scope="module")
def cases() -> list[Case]:
    """One case per dimension."""
    return [Case(dim) for dim in [1, 2, 3]]


# ----------------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------------


def test_Kinematics_rejects_a_displacement_of_the_wrong_size():
    mesh = _mesh(2)

    with pytest.raises(AssertionError):
        Kinematics(mesh.groupElem, np.zeros(2 * mesh.Nn + 1))


def test_Kinematics__CheckFormat_needs_a_group_of_elements():
    mesh = _mesh(2)

    Kinematics._CheckFormat(mesh.groupElem, np.zeros(2 * mesh.Nn))
    with pytest.raises(AssertionError):
        Kinematics._CheckFormat(mesh, np.zeros(2 * mesh.Nn))  # type: ignore [arg-type]


def test_Kinematics_groupElem_is_the_given_group(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            assert case.kinematics.groupElem is case.mesh.groupElem


def test_Kinematics_displacement_is_the_given_field(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            assert np.array_equal(
                case.kinematics.displacement,
                affine(case.mesh.coord, G3[: case.dim, : case.dim]),
            )


def test_Kinematics_matrixType_is_read_only(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            assert case.kinematics.matrixType == MatrixType.rigi
            with pytest.raises(AttributeError):
                case.kinematics.matrixType = MatrixType.mass  # type: ignore[misc]


def test_Kinematics_displacement_e_pg_repeats_the_element_dofs_and_is_not_cached(
    cases: list[Case], subtests
):
    for case in cases:
        with subtests.test(dim=case.dim):
            kinematics = case.kinematics
            groupElem = kinematics.groupElem
            nPg = groupElem.Get_gauss(kinematics.matrixType).nPg

            u_e_pg = kinematics.displacement_e_pg

            assert u_e_pg.shape == (groupElem.Ne, nPg, groupElem.nPe * case.dim)
            u_e = kinematics.displacement[groupElem.Get_assembly_e(case.dim)]
            assert np.array_equal(np.asarray(u_e_pg)[:, -1], u_e)
            assert kinematics.displacement_e_pg is not u_e_pg


def test_Kinematics__GetDims_reads_the_displacement_dimension(
    cases: list[Case], subtests
):
    for case in cases:
        with subtests.test(dim=case.dim):
            groupElem = case.mesh.groupElem
            nPg = groupElem.Get_gauss(MatrixType.rigi).nPg

            assert case.kinematics._GetDims() == (groupElem.Ne, nPg, case.dim)


def test_Kinematics_dim_is_the_displacement_dimension(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            assert case.kinematics.dim == case.dim


# ----------------------------------------------------------------------------
# strain measures
# ----------------------------------------------------------------------------


def test_Kinematics_Compute_GradU_is_the_gradient_padded_to_3x3(
    cases: list[Case], subtests
):
    for case in cases:
        with subtests.test(dim=case.dim):
            case.assert_everywhere(case.kinematics.Compute_GradU(), case.G)


def test_Kinematics_Compute_GradU_is_computed_once(monkeypatch):
    mesh = _mesh(2)
    groupElem = mesh.groupElem
    kinematics = Kinematics(groupElem, affine(mesh.coord, G3[:2, :2]))
    calls = []
    original = groupElem.Get_Gradient_e_pg

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(groupElem, "Get_Gradient_e_pg", counting)

    gradU = kinematics.Compute_GradU()
    kinematics.Compute_F()
    kinematics.Compute_GreenLagrange()

    assert kinematics.Compute_GradU() is gradU
    assert len(calls) == 1


def test_Kinematics_Compute_F_is_identity_plus_the_gradient(
    cases: list[Case], subtests
):
    for case in cases:
        with subtests.test(dim=case.dim):
            case.assert_everywhere(case.kinematics.Compute_F(), case.F)


def test_Kinematics_Compute_J_is_the_determinant_of_F(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            case.assert_everywhere(case.kinematics.Compute_J(), np.linalg.det(case.F))


def test_Kinematics_Compute_C_is_FtF(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            case.assert_everywhere(case.kinematics.Compute_C(), case.C)


def test_Kinematics__Compute_C_lists_the_components_row_by_row(
    cases: list[Case], subtests
):
    for case in cases:
        with subtests.test(dim=case.dim):
            components = case.kinematics._Compute_C()

            assert len(components) == 9
            for component, expected in zip(components, case.C.ravel()):
                case.assert_everywhere(component, expected)


def test_Kinematics_Compute_GreenLagrange_is_half_C_minus_I(
    cases: list[Case], subtests
):
    for case in cases:
        with subtests.test(dim=case.dim):
            E = case.kinematics.Compute_GreenLagrange()

            case.assert_everywhere(E, (case.C - np.eye(3)) / 2)


def test_Kinematics_Compute_Epsilon_is_the_mandel_symmetric_gradient(
    cases: list[Case], subtests
):
    for case in cases:
        with subtests.test(dim=case.dim):
            if case.dim == 1:
                pytest.skip("no B operator in 1D")

            epsilon = case.kinematics.Compute_Epsilon()

            case.assert_everywhere(epsilon, mandel(case.G, case.dim))
            assert case.kinematics.Compute_Epsilon() is epsilon


def test_Kinematics_Compute_Edot_vec_is_sym_Ft_gradv(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            if case.dim == 1:
                pytest.skip("De is 2D or 3D")
            H = _pad(H3, case.dim)
            velocity = affine(case.mesh.coord, H3[: case.dim, : case.dim])

            Edot = case.kinematics.Compute_Edot_vec(velocity)

            case.assert_everywhere(Edot, mandel(case.F.T @ H, case.dim))


def test_Kinematics_Compute_De_maps_flat_gradv_to_sym_Ft_gradv(
    cases: list[Case], subtests
):
    for case in cases:
        with subtests.test(dim=case.dim):
            if case.dim == 1:
                pytest.skip("De is 2D or 3D")
            dim = case.dim
            H = _pad(H3, dim)

            De = case.kinematics.Compute_De()

            values = np.asarray(De) @ H[:dim, :dim].ravel()
            case.assert_everywhere(values, mandel(case.F.T @ H, dim))


def test_Kinematics_Compute_Deta_maps_flat_graddu_to_sym_gradvt_graddu(
    cases: list[Case], subtests
):
    """∂Ė/∂(∇u)·∇δu = sym(∇vᵀ ∇δu) at fixed velocity."""
    for case in cases:
        with subtests.test(dim=case.dim):
            if case.dim == 1:
                pytest.skip("Deta is 2D or 3D")
            dim = case.dim
            H = _pad(H3, dim)
            K = _pad(G3.T, dim)

            Deta = case.kinematics.Compute_Deta(affine(case.mesh.coord, H3[:dim, :dim]))

            values = np.asarray(Deta) @ K[:dim, :dim].ravel()
            case.assert_everywhere(values, mandel(H.T @ K, dim))


# ----------------------------------------------------------------------------
# isotropic invariants
# ----------------------------------------------------------------------------


def _cofactor(C: np.ndarray) -> np.ndarray:
    return np.linalg.det(C) * np.linalg.inv(C).T


def test_Kinematics_Compute_I1_is_the_trace_of_C(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            case.assert_everywhere(case.kinematics.Compute_I1(), np.trace(case.C))


def test_Kinematics_Compute_dI1dC_is_the_identity(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            case.assert_everywhere(
                case.kinematics.Compute_dI1dC(), mandel(np.eye(3), case.dim)
            )


def test_Kinematics_Compute_d2I1dC_is_zero(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            d2I1dC = case.kinematics.Compute_d2I1dC()

            assert d2I1dC.shape[-1] == {1: 1, 2: 3, 3: 6}[case.dim]
            assert not np.any(d2I1dC)


def test_Kinematics_Compute_I2_is_the_second_invariant(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            C = case.C
            I2 = (np.trace(C) ** 2 - np.trace(C @ C)) / 2

            case.assert_everywhere(case.kinematics.Compute_I2(), I2)


def test_Kinematics_Compute_dI2dC_is_I1_I_minus_C(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            C = case.C

            dI2dC = case.kinematics.Compute_dI2dC()

            case.assert_everywhere(dI2dC, mandel(np.trace(C) * np.eye(3) - C, case.dim))


def test_Kinematics_Compute_d2I2dC_differentiates_I1_I_minus_C(
    cases: list[Case], subtests
):
    for case in cases:
        with subtests.test(dim=case.dim):
            expected = _mandel_matrix(
                case.dim, lambda dC: np.trace(dC) * np.eye(3) - dC
            )

            case.assert_everywhere(case.kinematics.Compute_d2I2dC(), expected)


def test_Kinematics_Compute_I3_is_the_determinant_of_C(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            case.assert_everywhere(case.kinematics.Compute_I3(), np.linalg.det(case.C))


def test_Kinematics_Compute_dI3dC_is_the_cofactor_of_C(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            case.assert_everywhere(
                case.kinematics.Compute_dI3dC(), mandel(_cofactor(case.C), case.dim)
            )


def test_Kinematics_Compute_d2I3dC_differentiates_the_cofactor(
    cases: list[Case], subtests
):
    """The cofactor is quadratic in C: a central difference is exact."""
    for case in cases:
        with subtests.test(dim=case.dim):
            C, h = case.C, 1e-3

            def dcof(dC: np.ndarray) -> np.ndarray:
                return (_cofactor(C + h * dC) - _cofactor(C - h * dC)) / (2 * h)

            expected = _mandel_matrix(case.dim, dcof)

            case.assert_everywhere(case.kinematics.Compute_d2I3dC(), expected)


# ----------------------------------------------------------------------------
# anisotropic invariants
# ----------------------------------------------------------------------------


def _in_dim(T: np.ndarray, dim: int) -> np.ndarray:
    """T with its components above `dim` zeroed."""
    T = T.copy()
    T[dim:] = 0
    return T


def test_Kinematics__Get_normalized_components_zeroes_the_components_above_dim(
    cases: list[Case], subtests
):
    for case in cases:
        with subtests.test(dim=case.dim):
            components = case.kinematics._Get_normalized_components(T1)

            for component, expected in zip(components, _in_dim(T1, case.dim)):
                case.assert_everywhere(component, expected)


def test_Kinematics__Compute_Anisotropic_Invariants_is_T1_C_T2(
    cases: list[Case], subtests
):
    for case in cases:
        with subtests.test(dim=case.dim):
            t1, t2 = _in_dim(T1, case.dim), _in_dim(T2, case.dim)

            value = case.kinematics._Compute_Anisotropic_Invariants(T1, T2)

            case.assert_everywhere(value, t1 @ case.C @ t2)


def test_Kinematics__Compute_Anisotropic_Invariants_First_Derivatives_is_sym_T1_T2(
    cases: list[Case], subtests
):
    for case in cases:
        with subtests.test(dim=case.dim):
            t1, t2 = _in_dim(T1, case.dim), _in_dim(T2, case.dim)

            values = case.kinematics._Compute_Anisotropic_Invariants_First_Derivatives(
                T1, T2
            )

            case.assert_everywhere(values, mandel(np.outer(t1, t2), case.dim))


def test_Kinematics_Compute_I4_is_T_C_T(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            t = _in_dim(T1, case.dim)

            case.assert_everywhere(case.kinematics.Compute_I4(T1), t @ case.C @ t)


def test_Kinematics_Compute_I4_takes_a_direction_per_element(
    cases: list[Case], subtests
):
    for case in cases:
        with subtests.test(dim=case.dim):
            Ne = case.mesh.groupElem.Ne
            t = _in_dim(T1, case.dim)

            I4 = case.kinematics.Compute_I4(np.tile(T1, (Ne, 1)))

            case.assert_everywhere(I4, t @ case.C @ t)


def test_Kinematics_Compute_dI4dC_is_T_outer_T(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            t = _in_dim(T1, case.dim)

            case.assert_everywhere(
                case.kinematics.Compute_dI4dC(T1), mandel(np.outer(t, t), case.dim)
            )


def test_Kinematics_Compute_d2I4dC_is_zero(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            assert not np.any(case.kinematics.Compute_d2I4dC())


def test_Kinematics_Compute_I6_is_T_C_T(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            t = _in_dim(T2, case.dim)

            case.assert_everywhere(case.kinematics.Compute_I6(T2), t @ case.C @ t)


def test_Kinematics_Compute_dI6dC_is_T_outer_T(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            t = _in_dim(T2, case.dim)

            case.assert_everywhere(
                case.kinematics.Compute_dI6dC(T2), mandel(np.outer(t, t), case.dim)
            )


def test_Kinematics_Compute_d2I6dC_is_zero(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            assert not np.any(case.kinematics.Compute_d2I6dC())


def test_Kinematics_Compute_I8_is_T1_C_T2(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            t1, t2 = _in_dim(T1, case.dim), _in_dim(T2, case.dim)

            case.assert_everywhere(case.kinematics.Compute_I8(T1, T2), t1 @ case.C @ t2)


def test_Kinematics_Compute_dI8dC_is_sym_T1_outer_T2(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            t1, t2 = _in_dim(T1, case.dim), _in_dim(T2, case.dim)

            dI8dC = case.kinematics.Compute_dI8dC(T1, T2)

            case.assert_everywhere(dI8dC, mandel(np.outer(t1, t2), case.dim))


def test_Kinematics_Compute_d2I8dC_is_zero(cases: list[Case], subtests):
    for case in cases:
        with subtests.test(dim=case.dim):
            assert not np.any(case.kinematics.Compute_d2I8dC())
