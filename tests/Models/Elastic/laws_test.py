# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest
import numpy as np

from EasyFEA.Models.Elastic import (
    _Elastic,
    Isotropic,
    TransverselyIsotropic,
    Orthotropic,
    Anisotropic,
)
from EasyFEA.FEM import Apply_Pmat, Get_Pmat

from _numerics import rel_error
from ..conftest import affine_kinematics


def _voigt_scale(dim: int) -> np.ndarray:
    """Kelvin = Voigt · outer(w, w): √2 on the shear rows."""
    n = 3 if dim == 2 else 6
    return np.array([1.0] * dim + [np.sqrt(2)] * (n - dim))


def _kelvin_of(C_voigt: np.ndarray) -> np.ndarray:
    w = _voigt_scale(2 if C_voigt.shape[-1] == 3 else 3)
    return C_voigt * np.outer(w, w)


NE = 4
TI = dict(El=11580, Et=500, Gl=450, vl=0.02, vt=0.44)

C_VOIGT_2D = np.array([[60, 20, 0], [20, 120, 0], [0, 0, 30]])
C_VOIGT_3D = np.array(
    [
        [60, 20, 10, 0, 0, 0],
        [20, 120, 80, 0, 0, 0],
        [10, 80, 300, 0, 0, 0],
        [0, 0, 0, 400, 0, 0],
        [0, 0, 0, 0, 500, 0],
        [0, 0, 0, 0, 0, 600],
    ]
)
ALIGNED = (np.array([1, 0, 0]), np.array([0, 1, 0]))
_a = np.deg2rad(30)
ROTATED = (
    np.array([np.cos(_a), np.sin(_a), 0]),
    np.array([-np.sin(_a), np.cos(_a), 0]),
)

ANISOTROPIC = {
    "2D": lambda: Anisotropic(2, C_VOIGT_2D, True),
    "2D-rotated": lambda: Anisotropic(
        2, Apply_Pmat(Get_Pmat(*ROTATED), _kelvin_of(C_VOIGT_3D)), False
    ),
    "3D-aligned": lambda: Anisotropic(
        3, Apply_Pmat(Get_Pmat(*ALIGNED), _kelvin_of(C_VOIGT_3D)), False
    ),
    "3D-rotated": lambda: Anisotropic(
        3, Apply_Pmat(Get_Pmat(*ROTATED), _kelvin_of(C_VOIGT_3D)), False
    ),
}


def test_Anisotropic_C_is_symmetric(subtests):
    for name, material in ANISOTROPIC.items():
        with subtests.test(material=name):
            C = material().C
            assert rel_error(C.T, C) <= 1e-12


def _ti_l_along_x(m: TransverselyIsotropic):
    return [
        [m.El + 4 * m.vl**2 * m.kt, 2 * m.kt * m.vl, 0],
        [2 * m.kt * m.vl, m.kt + m.Gt, 0],
        [0, 0, 2 * m.Gl],
    ]


def _ti_l_along_y(m: TransverselyIsotropic):
    return [
        [m.kt + m.Gt, 2 * m.kt * m.vl, 0],
        [2 * m.kt * m.vl, m.El + 4 * m.vl**2 * m.kt, 0],
        [0, 0, 2 * m.Gl],
    ]


def _ti_l_out_of_plane(m: TransverselyIsotropic):
    return [
        [m.kt + m.Gt, m.kt - m.Gt, 0],
        [m.kt - m.Gt, m.kt + m.Gt, 0],
        [0, 0, 2 * m.Gt],
    ]


def test_TransverselyIsotropic_plane_strain_C_is_the_closed_form(subtests):
    for axis_l, axis_t, closed_form in [
        ([1, 0, 0], [0, 1, 0], _ti_l_along_x),
        ([0, 1, 0], [1, 0, 0], _ti_l_along_y),
        ([0, 0, 1], [1, 0, 0], _ti_l_out_of_plane),
    ]:
        with subtests.test(closed_form=closed_form.__name__):
            material = TransverselyIsotropic(
                2,
                **TI,
                planeStress=False,
                axis_l=axis_l,
                axis_t=axis_t,
            )
            assert rel_error(material.C, np.array(closed_form(material))) <= 1e-12


def test_Orthotropic_with_equal_moduli_is_isotropic():
    E, v = TI["El"], TI["vl"]
    isot = Isotropic(3, E=E, v=v)
    mu = isot.get_mu()
    ortho = Orthotropic(
        3,
        E1=E,
        E2=E,
        E3=E,
        G23=mu,
        G13=mu,
        G12=mu,
        v23=v,
        v13=v,
        v12=v,
        axis_1=[0, 0, 1],
        axis_2=[0, 1, 0],
    )
    assert rel_error(ortho.S, isot.S) <= 1e-12


def test_Orthotropic_with_TI_moduli_is_TI(subtests):
    El, Et, Gl, vl, vt = (TI[k] for k in ("El", "Et", "Gl", "vl", "vt"))
    for dim, axis_l, axis_t in [(3, [1, 0, 0], [0, 1, 0]), (2, [0, 1, 0], [1, 0, 0])]:
        with subtests.test(dim=dim):
            ti = TransverselyIsotropic(
                dim,
                **TI,
                planeStress=False,
                axis_l=axis_l,
                axis_t=axis_t,
            )
            ortho = Orthotropic(
                dim,
                E1=El,
                E2=Et,
                E3=Et,
                G23=ti.Gt,
                G13=Gl,
                G12=Gl,
                v23=vt,
                v13=vl,
                v12=vl,
                planeStress=False,
                axis_1=axis_l,
                axis_2=axis_t,
            )
            assert rel_error(ortho.C, ti.C) <= 1e-12
            assert rel_error(ortho.S, ti.S) <= 1e-12


def test_Anisotropic_from_voigt_is_the_kelvin_C(subtests):
    for dim in (2, 3):
        with subtests.test(dim=dim):
            C_kelvin = Isotropic(dim, E=210e3, v=0.3).C
            kelvinScale = np.ones(C_kelvin.shape[0])
            kelvinScale[dim:] = np.sqrt(2)
            C_voigt = C_kelvin / np.outer(kelvinScale, kelvinScale)

            aniso = Anisotropic(dim, C_voigt, True)

            np.testing.assert_allclose(aniso.C, C_kelvin, atol=1e-6)


def test_Anisotropic_has_no_Walpole_Decomposition():
    aniso = Anisotropic(2, Isotropic(2).C, False)

    with pytest.raises(NotImplementedError):
        aniso.Walpole_Decomposition()


SHAPES = [(), (11,), (11, 4)]


def test_Elastic_Get_sqrt_C_S_square_back(subtests):
    """Whatever the leading axes of C, (11, 4) being properties per Gauss point."""
    for shape in SHAPES:
        with subtests.test(shape=shape):
            n = int(np.prod(shape))
            E = 210e9 if shape == () else np.linspace(200e9, 220e9, n).reshape(shape)
            material = Isotropic(3, E=E, v=0.3)

            sqrtC, sqrtS = material.Get_sqrt_C_S()
            C, S = material.C, material.S

            assert sqrtC.shape == shape + (6, 6)
            assert np.linalg.norm(sqrtC @ sqrtC - C) / np.linalg.norm(C) < 1e-12
            assert np.linalg.norm(sqrtS @ sqrtS - S) / np.linalg.norm(S) < 1e-12
            assert np.linalg.norm(sqrtC @ sqrtS - np.eye(6)) < 1e-12


def test_Elastic_str_prints_a_field_as_its_shape_and_range(subtests):
    laws = {
        "Isotropic": lambda E: Isotropic(2, E=E, v=0.3),
        "TransverselyIsotropic": lambda E: TransverselyIsotropic(
            2, El=E, Et=800, Gl=500, vl=0.3, vt=0.4
        ),
        "Orthotropic": lambda E: Orthotropic(
            3,
            E,
            800,
            500,
            200,
            300,
            400,
            0.3,
            0.25,
            0.4,
        ),
    }
    for name, law in laws.items():
        for shape in SHAPES:
            with subtests.test(law=name, shape=shape):
                E = np.full(shape, 210e3)
                expected = (
                    "2.1e+05" if shape == () else f"{shape} in [2.1e+05, 2.1e+05]"
                )
                assert f"= {expected}\n" in str(law(E))


def test_Orthotropic_str_prints_both_axes():
    material = Orthotropic(
        3,
        1,
        2,
        3,
        4,
        5,
        6,
        0.1,
        0.2,
        0.3,
        axis_1=(1, 0, 0),
        axis_2=(0, 1, 0),
    )
    assert "axis_2 = [0. 1. 0.]" in str(material)


def _Rotation(theta, phi) -> np.ndarray:
    """Rz(theta) @ Rx(phi), with leading axes of theta."""
    theta, phi = np.broadcast_arrays(theta, phi)
    c, s = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    zero, one = np.zeros_like(theta), np.ones_like(theta)
    Rz = np.stack([c, -s, zero, s, c, zero, zero, zero, one], -1)
    Rx = np.stack([one, zero, zero, zero, cp, -sp, zero, sp, cp], -1)
    shape = theta.shape + (3, 3)
    return Rz.reshape(shape) @ Rx.reshape(shape)


def _Axes(axes: str) -> tuple[np.ndarray, np.ndarray]:
    if axes == "none":
        R = np.eye(3)
    elif axes == "in-plane":
        R = _Rotation(0.4, 0.0)
    elif axes == "out-of-plane":
        R = _Rotation(0.4, 0.7)
    elif axes == "(Ne,3)":
        R = _Rotation(np.linspace(0, np.pi, NE), np.linspace(0, 1, NE))
    else:
        raise ValueError(axes)
    return R[..., 0], R[..., 1]


def _Law(law: type, dim: int, planeStress: bool, axes: str, E) -> _Elastic:
    if law is Isotropic:
        return Isotropic(dim, E=210e3 * E, v=0.3, planeStress=planeStress)
    axis1, axis2 = _Axes(axes)
    if law is TransverselyIsotropic:
        return TransverselyIsotropic(
            dim,
            **TI | {"El": TI["El"] * E},
            axis_l=axis1,
            axis_t=axis2,
            planeStress=planeStress,
        )
    return Orthotropic(
        dim,
        11580 * E,
        800,
        500,
        200,
        450,
        400,
        0.3,
        0.02,
        0.03,
        axis_1=axis1,
        axis_2=axis2,
        planeStress=planeStress,
    )


def _Cases(dims=(2, 3)):
    """(id, law, dim, planeStress, axes) over every law, hypothesis and axes."""
    for law in (Isotropic, TransverselyIsotropic, Orthotropic):
        for dim in dims:
            for planeStress in (False, True) if dim == 2 else (False,):
                axesList = ("none",)
                if law is not Isotropic:
                    axesList = ("none", "in-plane", "out-of-plane", "(Ne,3)")
                for axes in axesList:
                    hypothesis = "stress" if planeStress else "strain"
                    name = f"{law.__name__}-{dim}D-{hypothesis}-{axes}"
                    yield name, law, dim, planeStress, axes


MODULI = {"uniform": 1.0, "(Ne,)": np.linspace(1, 2, NE)}


def test_Elastic__Get_C_3D_rebuilds_the_law_as_anisotropic(subtests):
    w = _voigt_scale(3)
    for name, law, dim, planeStress, axes in _Cases():
        with subtests.test(case=name):
            for E in MODULI.values():
                material = _Law(
                    law,
                    dim,
                    planeStress,
                    axes,
                    E,
                )
                aniso = Anisotropic(dim, material._Get_C_3D() / np.outer(w, w), True)
                aniso.planeStress = material.planeStress

                assert rel_error(aniso.C, material.C) <= 1e-12
                assert rel_error(aniso.S, material.S) <= 1e-12


def test_Elastic_C_in_2d_rebuilds_the_law_as_anisotropic(subtests):
    w = _voigt_scale(2)
    for name, law, dim, planeStress, axes in _Cases((2,)):
        with subtests.test(case=name):
            for E in MODULI.values():
                material = _Law(
                    law,
                    dim,
                    planeStress,
                    axes,
                    E,
                )
                aniso = Anisotropic(2, material.C / np.outer(w, w), True)

                assert rel_error(aniso.C, material.C) <= 1e-12
                assert rel_error(aniso.S, material.S) <= 1e-12


def test_Elastic_C_cannot_be_assigned():
    aniso = Anisotropic(3, Isotropic(3).C, False)
    with pytest.raises(AttributeError):
        aniso.C = Isotropic(3).C


def test_Anisotropic_6x6_C_takes_plane_stress_in_2d():
    aniso = Anisotropic(2, Isotropic(3).C, False)
    assert rel_error(aniso.C, Isotropic(2, planeStress=False).C) <= 1e-12
    aniso.planeStress = True
    assert rel_error(aniso.C, Isotropic(2, planeStress=True).C) <= 1e-12


def test_Anisotropic_3x3_C_is_2d_only():
    C = Isotropic(2, planeStress=False).C
    with pytest.raises(ValueError, match="2D"):
        Anisotropic(3, C, False).C
    aniso = Anisotropic(2, C, False)
    aniso.planeStress = True
    with pytest.raises(ValueError, match="plane stress"):
        aniso.C
    aniso.planeStress = False
    with pytest.raises(ValueError, match="3D"):
        aniso._Get_C_3D()


def test_Elastic__Axes_of_different_shapes_raise():
    axis1, axis2 = _Axes("(Ne,3)")
    with pytest.raises(ValueError, match="shape"):
        TransverselyIsotropic(3, **TI, axis_l=axis1[0], axis_t=axis2).C


HYPOTHESES = [(2, False), (2, True), (3, False)]
"""(dim, planeStress) of every hypothesis."""


def test_TransverselyIsotropic_with_isotropic_moduli_is_isotropic_under_any_axes(
    subtests,
):
    E, v = 210e3, 0.3
    for dim, planeStress in HYPOTHESES:
        iso = Isotropic(dim, E=E, v=v, planeStress=planeStress)
        for axes in ("none", "in-plane", "out-of-plane", "(Ne,3)"):
            with subtests.test(dim=dim, planeStress=planeStress, axes=axes):
                axis_l, axis_t = _Axes(axes)
                ti = TransverselyIsotropic(
                    dim,
                    El=E,
                    Et=E,
                    Gl=E / (2 * (1 + v)),
                    vl=v,
                    vt=v,
                    axis_l=axis_l,
                    axis_t=axis_t,
                    planeStress=planeStress,
                )
                C = np.broadcast_to(iso.C, ti.C.shape)
                assert rel_error(ti.C, C) <= 1e-12


def test_Elastic_Walpole_Decomposition_sums_to_C_at_every_point(subtests):
    for law in (TransverselyIsotropic, Orthotropic):
        with subtests.test(law=law.__name__):
            material = _Law(
                law,
                3,
                False,
                "(Ne,3)",
                1.0,
            )
            ci, Ei = material.Walpole_Decomposition()
            assert Ei.shape == (len(ci), NE, 6, 6)
            C = material._Get_C_3D()
            assert rel_error(np.einsum("k,k...->...", ci, Ei), C) <= 1e-12


def test_Elastic_Walpole_Decomposition_with_heterogeneous_moduli(subtests):
    for law in (TransverselyIsotropic, Orthotropic):
        with subtests.test(law=law.__name__):
            material = _Law(
                law,
                3,
                False,
                "none",
                np.linspace(1, 2, NE),
            )
            ci, Ei = material.Walpole_Decomposition()
            C = material._Get_C_3D()
            assert rel_error(np.tensordot(ci, Ei, axes=(0, 0)), C) <= 1e-12


def test_Elastic__Axes_per_gauss_point():
    axis1, axis2 = _Axes("(Ne,3)")
    nPg = 3
    axis1_p, axis2_p = [np.repeat(a[:, np.newaxis], nPg, 1) for a in (axis1, axis2)]
    ti = TransverselyIsotropic(3, **TI, axis_l=axis1, axis_t=axis2)
    ti_p = TransverselyIsotropic(3, **TI, axis_l=axis1_p, axis_t=axis2_p)
    assert rel_error(ti_p.C, np.repeat(ti.C[:, np.newaxis], nPg, 1)) <= 1e-12


def test_Elastic__Axes_not_perpendicular_raise():
    with pytest.raises(ValueError, match="perpendicular"):
        TransverselyIsotropic(3, **TI, axis_l=(1, 0, 0), axis_t=(1, 1, 0)).C


def test_TransverselyIsotropic_setting_an_axis_updates_C():
    ti = TransverselyIsotropic(3, **TI)
    ti_r = TransverselyIsotropic(3, **TI, axis_l=(0, 1, 0), axis_t=(1, 0, 0))
    ti.C
    ti.axis_l, ti.axis_t = np.array([0.0, 1, 0]), np.array([1.0, 0, 0])
    assert rel_error(ti.C, ti_r.C) <= 1e-12


def test_Orthotropic_bounds_raise():
    material = Orthotropic(3, 1, 100, 1, 1, 1, 1, 0.3, 0.3, 0.3)
    with pytest.raises(ValueError, match="v12"):
        material.C


def test_Elastic_Available_Laws_lists_every_law():
    assert _Elastic.Available_Laws() == [
        Isotropic,
        TransverselyIsotropic,
        Orthotropic,
        Anisotropic,
    ]


def test_Isotropic_C_is_the_closed_form(subtests):
    E, v = 210000.0, 0.3
    l3, mu = E * v / ((1 + v) * (1 - 2 * v)), E / (2 * (1 + v))
    for dim, planeStress in HYPOTHESES:
        with subtests.test(dim=dim, planeStress=planeStress):
            if dim == 3:
                ref = np.diag([0, 0, 0, 2 * mu, 2 * mu, 2 * mu])
                ref[:3, :3] = l3 + 2 * mu * np.eye(3)
            elif planeStress:
                ref = E / (1 - v**2) * np.array([[1, v, 0], [v, 1, 0], [0, 0, 1 - v]])
            else:
                ref = np.array(
                    [[l3 + 2 * mu, l3, 0], [l3, l3 + 2 * mu, 0], [0, 0, 2 * mu]]
                )
            for field in (E, np.full(4, E)):
                C = Isotropic(dim, field, v, planeStress).C
                expected = np.broadcast_to(ref, (*np.shape(field), *ref.shape))
                assert rel_error(C, expected) <= 1e-12


def test_Elastic_simplification_names_the_hypothesis():
    assert Isotropic(3, planeStress=False).simplification == "3D"
    assert Isotropic(2, planeStress=False).simplification == "Plane Strain"
    assert Isotropic(2, planeStress=True).simplification == "Plane Stress"


def test_Elastic__Material_C_is_C_in_the_material_frame(subtests):
    for law in (TransverselyIsotropic, Orthotropic):
        with subtests.test(law=law.__name__):
            material = _Law(
                law,
                3,
                False,
                "out-of-plane",
                1.0,
            )
            aligned = _Law(
                law,
                3,
                False,
                "none",
                1.0,
            )

            assert rel_error(material._Material_C(), aligned.C) <= 1e-12


def test_Elastic_S_is_the_inverse_of_C(subtests):
    for dim, planeStress in HYPOTHESES:
        with subtests.test(dim=dim, planeStress=planeStress):
            material = _Law(
                Orthotropic,
                dim,
                planeStress,
                "out-of-plane",
                1.0,
            )
            identity = np.eye(material.C.shape[-1])
            assert rel_error(material.C @ material.S, identity) <= 1e-12


def test_Elastic_Compute_Sigma_is_C_epsilon(subtests):
    for dim in (2, 3):
        with subtests.test(dim=dim):
            affine = affine_kinematics(dim)
            material = Isotropic(dim, planeStress=False)
            sigma = material.Compute_Sigma(affine.kinematics)
            expected = np.broadcast_to(material.C @ affine.epsilon, sigma.shape)
            assert rel_error(sigma, expected) <= 1e-12


def test_Elastic_Compute_Psi_is_half_sigma_epsilon(subtests):
    for dim in (2, 3):
        with subtests.test(dim=dim):
            affine = affine_kinematics(dim)
            material = Isotropic(dim, planeStress=False)
            psi = material.Compute_Psi(affine.kinematics)
            expected = 0.5 * affine.epsilon @ material.C @ affine.epsilon
            assert rel_error(psi, np.full(psi.shape, expected)) <= 1e-12


def test_Elastic__Unit_axes_are_an_orthonormal_right_handed_frame():
    axis1, axis2 = _Axes("(Ne,3)")
    e1, e2, e3 = TransverselyIsotropic(
        3, **TI, axis_l=3 * axis1, axis_t=axis2
    )._Unit_axes()

    assert e1.shape == (NE, 1, 3)
    assert rel_error(np.asarray(e1)[:, 0], axis1) <= 1e-12
    assert rel_error(np.einsum("...i,...i", e1, e2), 0.0, 1.0) <= 1e-12
    assert rel_error(np.asarray(e3), np.cross(e1, e2), 1.0) <= 1e-12


def test_Elastic__Walpole_rejects_moduli_that_do_not_sum_to_C():
    material = _Law(
        TransverselyIsotropic,
        3,
        False,
        "none",
        1.0,
    )
    ci, Ei = material.Walpole_Decomposition()

    assert (
        rel_error(
            np.tensordot(*material._Walpole(list(ci), list(Ei)), axes=1),
            material._Get_C_3D(),
        )
        <= 1e-12
    )
    with pytest.raises(AssertionError):
        material._Walpole(list(2 * ci), list(Ei))


E_V = 210000.0, 0.3


def test_Isotropic_get_lambda_is_the_closed_form():
    E, v = E_V
    assert np.isclose(Isotropic(3, E, v).get_lambda(), E * v / ((1 + v) * (1 - 2 * v)))


def test_Isotropic_get_mu_is_the_closed_form():
    E, v = E_V
    assert np.isclose(Isotropic(3, E, v).get_mu(), E / (2 * (1 + v)))


def test_Isotropic_get_bulk_is_the_closed_form():
    E, v = E_V
    assert np.isclose(Isotropic(3, E, v).get_bulk(), E / (3 * (1 - 2 * v)))


def test_Isotropic_get_lambda_is_reduced_under_plane_stress():
    E, v = 210000.0, 0.3
    material = Isotropic(2, E=E, v=v, planeStress=True)

    assert np.isclose(material.get_lambda(), E * v / (1 - v**2))
    assert np.isclose(material.get_lambda(3), E * v / ((1 + v) * (1 - 2 * v)))


def test_Isotropic_get_bulk_in_2d_is_lambda_plus_mu():
    material = Isotropic(2, planeStress=False)

    assert np.isclose(material.get_bulk(), material.get_lambda() + material.get_mu())


def test_TransverselyIsotropic_Gt_and_kt_are_the_closed_forms():
    El, Et, vl, vt = (TI[k] for k in ("El", "Et", "vl", "vt"))
    ti = TransverselyIsotropic(3, **TI)

    assert np.isclose(ti.Gt, Et / (2 * (1 + vt)))
    assert np.isclose(ti.kt, El * Et / (2 * (1 - vt) * El - 4 * vl**2 * Et))


def test_TransverselyIsotropic_kt_is_the_plane_strain_bulk_of_the_transverse_plane():
    """Isotropic moduli: kt = lambda + mu."""
    iso = Isotropic(3)
    E, v = iso.E, iso.v
    ti = TransverselyIsotropic(3, El=E, Et=E, Gl=iso.get_mu(), vl=v, vt=v)

    assert np.isclose(ti.kt, iso.get_lambda() + iso.get_mu())


def test_Anisotropic_Set_C_replaces_the_stiffness():
    aniso = Anisotropic(3, Isotropic(3).C, False)
    aniso.C
    aniso.Set_C(2 * Isotropic(3).C, False)

    assert rel_error(aniso.C, 2 * Isotropic(3).C) <= 1e-12


def test_Anisotropic_Set_C_rejects_a_bad_matrix():
    with pytest.raises(ValueError, match="C must be a"):
        Anisotropic(3, np.eye(4), False)
    with pytest.raises(ValueError, match="symmetric"):
        Anisotropic(3, np.triu(np.ones((6, 6))), False)
