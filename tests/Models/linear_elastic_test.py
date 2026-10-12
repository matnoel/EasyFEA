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
from EasyFEA.FEM import _kelvin_mandel
from EasyFEA.FEM._kelvin_mandel import Get_Pmat, Apply_Pmat

from _numerics import rel_error


def _Rotate_2D(C_voigt2D: np.ndarray, axis1: np.ndarray, axis2: np.ndarray):
    """In-plane rotation of a 2D Voigt C, through its 3D Kelvin–Mandel lift."""
    idx = _kelvin_mandel.IDX[2]
    C = np.zeros((6, 6))
    C[np.ix_(idx, idx)] = _kelvin_mandel.From_Voigt(C_voigt2D)
    C = Apply_Pmat(Get_Pmat(axis1, axis2), C)
    w = _kelvin_mandel.Weights(2)
    return _kelvin_mandel.Reduce_matrix(C, 2) / np.outer(w, w)


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
_a = 30 * np.pi / 130
ROTATED = (
    np.array([np.cos(_a), np.sin(_a), 0]),
    np.array([-np.sin(_a), np.cos(_a), 0]),
)

ANISOTROPIC = {
    "2D": lambda: Anisotropic(2, C_VOIGT_2D, True),
    "2D-aligned": lambda: Anisotropic(2, _Rotate_2D(C_VOIGT_2D, *ALIGNED), True),
    "2D-rotated": lambda: Anisotropic(2, _Rotate_2D(C_VOIGT_2D, *ROTATED), True),
    "3D-aligned": lambda: Anisotropic(
        3, Apply_Pmat(Get_Pmat(*ALIGNED), _kelvin_mandel.From_Voigt(C_VOIGT_3D)), False
    ),
    "3D-rotated": lambda: Anisotropic(
        3, Apply_Pmat(Get_Pmat(*ROTATED), _kelvin_mandel.From_Voigt(C_VOIGT_3D)), False
    ),
}


@pytest.mark.parametrize("material", ANISOTROPIC.values(), ids=ANISOTROPIC)
def test_anisotropic_C_is_symmetric(material):
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


@pytest.mark.parametrize(
    "axis_l, axis_t, closed_form",
    [
        ([1, 0, 0], [0, 1, 0], _ti_l_along_x),
        ([0, 1, 0], [1, 0, 0], _ti_l_along_y),
        ([0, 0, 1], [1, 0, 0], _ti_l_out_of_plane),
    ],
    ids=["l_along_x", "l_along_y", "l_out_of_plane"],
)
def test_TI_plane_strain_C_is_the_closed_form(axis_l, axis_t, closed_form):
    material = TransverselyIsotropic(
        2, **TI, planeStress=False, axis_l=axis_l, axis_t=axis_t
    )
    assert rel_error(material.C, np.array(closed_form(material))) <= 1e-12


def test_orthotropic_with_equal_moduli_is_isotropic():
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


@pytest.mark.parametrize(
    "dim, axis_l, axis_t", [(3, [1, 0, 0], [0, 1, 0]), (2, [0, 1, 0], [1, 0, 0])]
)
def test_orthotropic_with_TI_moduli_is_TI(dim, axis_l, axis_t):
    El, Et, Gl, vl, vt = (TI[k] for k in ("El", "Et", "Gl", "vl", "vt"))
    ti = TransverselyIsotropic(
        dim, **TI, planeStress=False, axis_l=axis_l, axis_t=axis_t
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


class TestLinearElastic:
    @pytest.mark.parametrize("dim", [2, 3])
    def test_anisotropic_from_voigt_is_the_kelvin_C(self, dim: int):
        C_kelvin = Isotropic(dim, E=210e3, v=0.3).C
        kelvinScale = np.ones(C_kelvin.shape[0])
        kelvinScale[dim:] = np.sqrt(2)
        C_voigt = C_kelvin / np.outer(kelvinScale, kelvinScale)

        aniso = Anisotropic(dim, C_voigt, True)

        np.testing.assert_allclose(aniso.C, C_kelvin, atol=1e-6)

    def test_Anisot_has_no_Walpole_Decomposition(self):
        aniso = Anisotropic(2, Isotropic(2).C, False)

        with pytest.raises(NotImplementedError):
            aniso.Walpole_Decomposition()

    @pytest.mark.parametrize("shape", [(), (11,), (11, 4)])
    def test_sqrt_C_and_S_square_back(self, shape: tuple):
        """Whatever the leading axes of C, (11, 4) being properties per Gauss point."""
        n = int(np.prod(shape))
        E = 210e9 if shape == () else np.linspace(200e9, 220e9, n).reshape(shape)
        material = Isotropic(3, E=E, v=0.3)

        sqrtC, sqrtS = material.Get_sqrt_C_S()
        C, S = material.C, material.S

        assert sqrtC.shape == shape + (6, 6)
        assert np.linalg.norm(sqrtC @ sqrtC - C) / np.linalg.norm(C) < 1e-12
        assert np.linalg.norm(sqrtS @ sqrtS - S) / np.linalg.norm(S) < 1e-12
        assert np.linalg.norm(sqrtC @ sqrtS - np.eye(6)) < 1e-12

    @pytest.mark.parametrize(
        "law",
        [
            lambda E: Isotropic(2, E=E, v=0.3),
            lambda E: TransverselyIsotropic(2, El=E, Et=800, Gl=500, vl=0.3, vt=0.4),
            lambda E: Orthotropic(
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
        ],
        ids=["Isotropic", "TransverselyIsotropic", "Orthotropic"],
    )
    @pytest.mark.parametrize("shape", [(), (11,), (11, 4)])
    def test_str_prints_a_field_as_its_shape_and_range(self, law, shape: tuple):
        E = np.full(shape, 210e3)
        expected = "2.1e+05" if shape == () else f"{shape} in [2.1e+05, 2.1e+05]"
        assert f"= {expected}\n" in str(law(E))

    def test_str_prints_both_axes(self):
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


MODULI = {
    "scalar": 1.0,
    "(Ne,)": np.ones(NE),
    "(Ne,nPg)": np.ones((NE, 3)),
    "(Ne,)-varying": np.linspace(1, 1.001, NE),
}


def _Pmat_axes(dim: int, shape: str) -> tuple[np.ndarray, np.ndarray]:
    """In-plane axes: fixed, one per element, or one per element and Gauss point."""
    if shape == "uniform":
        return np.eye(3)[0, :dim], np.eye(3)[1, :dim]
    angles = np.linspace(0, np.pi, NE)
    axis1 = np.zeros((NE, dim))
    axis2 = np.zeros((NE, dim))
    axis1[:, 0], axis1[:, 1] = np.cos(angles), np.sin(angles)
    axis2[:, 0], axis2[:, 1] = -np.sin(angles), np.cos(angles)
    if shape == "(Ne,nPg)":
        axis1, axis2 = (np.repeat(a[:, np.newaxis], 3, 1) for a in (axis1, axis2))
    return axis1, axis2


AXES = ["uniform", "(Ne,)", "(Ne,nPg)"]


def _Same_leading_axes(a: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
    """Broadcasts two (..., n, n) arrays whose leading axes are (), (Ne,) or (Ne, nPg)."""
    if a.ndim == 3 and b.ndim == 4:
        a = a[:, np.newaxis]
    if b.ndim == 3 and a.ndim == 4:
        b = b[:, np.newaxis]
    return np.broadcast_arrays(a, b)


def _Trace(A: np.ndarray) -> np.ndarray:
    return np.trace(A, axis1=-2, axis2=-1)


@pytest.fixture(params=["C", "S"])
def ti_tensor(request, dim: int, moduli: str) -> np.ndarray:
    """C or S of a TI material whose moduli are scalar, per element or per Gauss point."""
    c = MODULI[moduli]
    material = TransverselyIsotropic(dim, **{k: c * v for k, v in TI.items()})
    return getattr(material, request.param)


@pytest.mark.parametrize("moduli", MODULI)
@pytest.mark.parametrize("axes", AXES)
@pytest.mark.parametrize("dim", [2, 3])
def test_rotation_to_global_preserves_trace_and_det(ti_tensor, dim, axes):
    rotated = Apply_Pmat(Get_Pmat(*_Pmat_axes(dim, axes)), ti_tensor)
    rotated, local = _Same_leading_axes(rotated, ti_tensor)
    assert rel_error(_Trace(rotated), _Trace(local)) <= 1e-12
    assert rel_error(np.linalg.det(rotated), np.linalg.det(local)) <= 1e-12


@pytest.mark.parametrize("moduli", MODULI)
@pytest.mark.parametrize("axes", AXES)
@pytest.mark.parametrize("dim", [2, 3])
def test_rotation_back_to_material_axes_is_identity(ti_tensor, dim, axes):
    Pmat = Get_Pmat(*_Pmat_axes(dim, axes))
    back = Apply_Pmat(Pmat, Apply_Pmat(Pmat, ti_tensor), toGlobal=False)
    back, local = _Same_leading_axes(back, ti_tensor)
    assert rel_error(back, local) <= 1e-12


@pytest.mark.parametrize("axes", AXES)
@pytest.mark.parametrize("dim", [2, 3])
def test_stress_and_strain_rotations_are_inverse_transposes(dim, axes):
    Ps, Pe = Get_Pmat(*_Pmat_axes(dim, axes), False)
    assert rel_error(np.linalg.inv(Ps), np.swapaxes(Pe, -1, -2)) <= 1e-12
    assert rel_error(np.linalg.inv(Pe), np.swapaxes(Ps, -1, -2)) <= 1e-12


# ----------------------------------------------
# Every law against Anisotropic built from its 3D C
# ----------------------------------------------


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
            El=11580 * E,
            Et=500,
            Gl=450,
            vl=0.02,
            vt=0.44,
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
    for law in (Isotropic, TransverselyIsotropic, Orthotropic):
        for dim in dims:
            for planeStress in (False, True) if dim == 2 else (False,):
                axesList = ("none",)
                if law is not Isotropic:
                    axesList = ("none", "in-plane", "out-of-plane", "(Ne,3)")
                for axes in axesList:
                    for E in ("uniform", "(Ne,)"):
                        yield pytest.param(
                            law,
                            dim,
                            planeStress,
                            axes,
                            E,
                            id=f"{law.__name__}-{dim}D-{'stress' if planeStress else 'strain'}-{axes}-{E}",
                        )


class TestElasticPipeline:

    @pytest.mark.parametrize("law, dim, planeStress, axes, E", list(_Cases()))
    def test_law_is_anisotropic_of_its_3d_C(self, law, dim, planeStress, axes, E):
        E = 1.0 if E == "uniform" else np.linspace(1, 2, NE)
        material = _Law(law, dim, planeStress, axes, E)

        w = _kelvin_mandel.Weights(3)
        aniso = Anisotropic(dim, material._Get_C_3D() / np.outer(w, w), True)
        aniso.planeStress = material.planeStress

        assert rel_error(aniso.C, material.C) <= 1e-12
        assert rel_error(aniso.S, material.S) <= 1e-12

    @pytest.mark.parametrize("law, dim, planeStress, axes, E", list(_Cases((2,))))
    def test_2d_law_is_anisotropic_of_its_2d_C(self, law, dim, planeStress, axes, E):
        E = 1.0 if E == "uniform" else np.linspace(1, 2, NE)
        material = _Law(law, dim, planeStress, axes, E)

        w = _kelvin_mandel.Weights(2)
        aniso = Anisotropic(2, material.C / np.outer(w, w), True)

        assert rel_error(aniso.C, material.C) <= 1e-12
        assert rel_error(aniso.S, material.S) <= 1e-12

    def test_a_C_cannot_be_assigned(self):
        aniso = Anisotropic(3, Isotropic(3).C, False)
        with pytest.raises(AttributeError):
            aniso.C = Isotropic(3).C

    def test_a_6x6_C_takes_plane_stress_in_2d(self):
        aniso = Anisotropic(2, Isotropic(3).C, False)
        assert rel_error(aniso.C, Isotropic(2, planeStress=False).C) <= 1e-12
        aniso.planeStress = True
        assert rel_error(aniso.C, Isotropic(2, planeStress=True).C) <= 1e-12

    def test_a_3x3_C_is_2d_only(self):
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

    def test_axes_of_different_shapes_raise(self):
        axis1, axis2 = _Axes("(Ne,3)")
        with pytest.raises(ValueError, match="shape"):
            TransverselyIsotropic(3, 11580, 500, 450, 0.02, 0.44, axis1[0], axis2).C

    @pytest.mark.parametrize("axes", ["none", "in-plane", "out-of-plane", "(Ne,3)"])
    @pytest.mark.parametrize("dim, planeStress", [(2, False), (2, True), (3, False)])
    def test_isotropic_is_a_TI_under_any_axes(self, axes, dim, planeStress):
        E, v = 210e3, 0.3
        axis_l, axis_t = _Axes(axes)
        iso = Isotropic(dim, E=E, v=v, planeStress=planeStress)
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

    @pytest.mark.parametrize("law", [TransverselyIsotropic, Orthotropic])
    def test_walpole_sums_to_C_at_every_point(self, law):
        material = _Law(law, 3, False, "(Ne,3)", 1.0)
        ci, Ei = material.Walpole_Decomposition()
        assert Ei.shape == (len(ci), NE, 6, 6)
        assert (
            rel_error(np.einsum("k,k...->...", ci, Ei), material._Get_C_3D()) <= 1e-12
        )

    @pytest.mark.parametrize("law", [TransverselyIsotropic, Orthotropic])
    def test_walpole_with_heterogeneous_moduli(self, law):
        material = _Law(law, 3, False, "none", np.linspace(1, 2, NE))
        ci, Ei = material.Walpole_Decomposition()
        assert (
            rel_error(np.tensordot(ci, Ei, axes=(0, 0)), material._Get_C_3D()) <= 1e-12
        )

    def test_axes_per_gauss_point(self):
        axis1, axis2 = _Axes("(Ne,3)")
        nPg = 3
        axis1_p, axis2_p = [np.repeat(a[:, np.newaxis], nPg, 1) for a in (axis1, axis2)]
        ti = TransverselyIsotropic(3, 11580, 500, 450, 0.02, 0.44, axis1, axis2)
        ti_p = TransverselyIsotropic(3, 11580, 500, 450, 0.02, 0.44, axis1_p, axis2_p)
        assert rel_error(ti_p.C, np.repeat(ti.C[:, np.newaxis], nPg, 1)) <= 1e-12

    def test_axes_not_perpendicular_raise(self):
        with pytest.raises(ValueError, match="perpendicular"):
            TransverselyIsotropic(
                3, 11580, 500, 450, 0.02, 0.44, (1, 0, 0), (1, 1, 0)
            ).C

    def test_setting_an_axis_updates_C(self):
        ti = TransverselyIsotropic(3, 11580, 500, 450, 0.02, 0.44)
        ti_r = TransverselyIsotropic(
            3, 11580, 500, 450, 0.02, 0.44, (0, 1, 0), (1, 0, 0)
        )
        ti.C
        ti.axis_l, ti.axis_t = np.array([0.0, 1, 0]), np.array([1.0, 0, 0])
        assert rel_error(ti.C, ti_r.C) <= 1e-12

    def test_orthotropic_bounds_raise(self):
        material = Orthotropic(3, 1, 100, 1, 1, 1, 1, 0.3, 0.3, 0.3)
        with pytest.raises(ValueError, match="v12"):
            material.C

    def test_available_laws_list_orthotropic(self):
        assert Orthotropic in _Elastic.Available_Laws()


def test_kelvin_components_remove_sqrt2():
    values = _kelvin_mandel.Components(np.array([1.0, 2.0, 3.0 * _kelvin_mandel.R2]))
    assert values == {"xx": 1.0, "yy": 2.0, "xy": pytest.approx(3.0)}
    with pytest.raises(ValueError):
        _kelvin_mandel.Components(np.zeros(4))


@pytest.mark.parametrize("E", [210000.0, np.full(4, 210000.0)])
@pytest.mark.parametrize("dim, planeStress", [(3, False), (2, False), (2, True)])
def test_isotropic_C_is_the_closed_form(dim, planeStress, E):
    v = 0.3
    E0 = 210000.0
    l3, mu = E0 * v / ((1 + v) * (1 - 2 * v)), E0 / (2 * (1 + v))
    if dim == 3:
        ref = np.diag([0, 0, 0, 2 * mu, 2 * mu, 2 * mu])
        ref[:3, :3] = l3 + 2 * mu * np.eye(3)
    elif planeStress:
        ref = E0 / (1 - v**2) * np.array([[1, v, 0], [v, 1, 0], [0, 0, 1 - v]])
    else:
        ref = np.array([[l3 + 2 * mu, l3, 0], [l3, l3 + 2 * mu, 0], [0, 0, 2 * mu]])
    assert (
        rel_error(
            Isotropic(dim, E, v, planeStress).C,
            np.broadcast_to(ref, (*np.shape(E), *ref.shape)),
        )
        <= 1e-12
    )
