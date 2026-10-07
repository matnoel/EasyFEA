# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest
import numpy as np

# materials
from EasyFEA.Models.Elastic._laws import (
    _Elastic,
    Isotropic,
    TransverselyIsotropic,
    Orthotropic,
    Anisotropic,
)
from EasyFEA.FEM import _kelvin_mandel as kelvin_mandel
from EasyFEA.FEM._kelvin_mandel import Get_Pmat, Apply_Pmat


def _Rotate_2D(C_voigt2D: np.ndarray, axis1: np.ndarray, axis2: np.ndarray):
    """In-plane rotation of a 2D Voigt C, through its 3D Kelvin–Mandel lift."""
    idx = kelvin_mandel.IDX[2]
    C = np.zeros((6, 6))
    C[np.ix_(idx, idx)] = kelvin_mandel.From_Voigt(C_voigt2D)
    C = Apply_Pmat(Get_Pmat(axis1, axis2), C)
    w = kelvin_mandel.Weights(2)
    return kelvin_mandel.Reduce(C, 2) / np.outer(w, w)


@pytest.fixture
def setup_elastic_materials() -> list[_Elastic]:

    elasticMaterials: list[_Elastic] = []

    for comp in _Elastic.Available_Laws():
        if comp == Isotropic:
            elasticMaterials.append(Isotropic(2, E=210e9, v=0.3, planeStress=True))
            elasticMaterials.append(Isotropic(2, E=210e9, v=0.3, planeStress=False))
            elasticMaterials.append(Isotropic(3, E=210e9, v=0.3))
        elif comp == TransverselyIsotropic:
            c = np.sqrt(2) / 2
            elasticMaterials.append(
                TransverselyIsotropic(
                    3,
                    El=11580,
                    Et=500,
                    Gl=450,
                    vl=0.02,
                    vt=0.44,
                    axis_l=[c, c, 0],
                    axis_t=[c, -c, 0],
                )
            )
            elasticMaterials.append(
                TransverselyIsotropic(
                    3,
                    El=11580,
                    Et=500,
                    Gl=450,
                    vl=0.02,
                    vt=0.44,
                    axis_l=[0, 1, 0],
                    axis_t=[1, 0, 0],
                )
            )
            elasticMaterials.append(
                TransverselyIsotropic(
                    2, El=11580, Et=500, Gl=450, vl=0.02, vt=0.44, planeStress=True
                )
            )
            elasticMaterials.append(
                TransverselyIsotropic(
                    2, El=11580, Et=500, Gl=450, vl=0.02, vt=0.44, planeStress=False
                )
            )

        elif comp == Anisotropic:
            C_voigt2D = np.array([[60, 20, 0], [20, 120, 0], [0, 0, 30]])

            axis1_1 = np.array([1, 0, 0])
            axis2_1 = np.array([0, 1, 0])

            tetha = 30 * np.pi / 130
            axis1_2 = np.array([np.cos(tetha), np.sin(tetha), 0])
            axis2_2 = np.array([-np.sin(tetha), np.cos(tetha), 0])

            for axis1, axis2 in [(axis1_1, axis2_1), (axis1_2, axis2_2)]:
                C = _Rotate_2D(C_voigt2D, axis1, axis2)
                elasticMaterials.append(Anisotropic(2, C, True))
                elasticMaterials.append(Anisotropic(2, C, False))

    return elasticMaterials


class TestLinearElastic:

    def test_Elastic_Isot(self, setup_elastic_materials):

        for mat in setup_elastic_materials:
            assert isinstance(mat, _Elastic)
            if isinstance(mat, Isotropic):
                E = mat.E
                v = mat.v
                if mat.dim == 2:
                    if mat.planeStress:
                        C_voigt = (
                            E
                            / (1 - v**2)
                            * np.array([[1, v, 0], [v, 1, 0], [0, 0, (1 - v) / 2]])
                        )
                    else:
                        C_voigt = (
                            E
                            / ((1 + v) * (1 - 2 * v))
                            * np.array(
                                [[1 - v, v, 0], [v, 1 - v, 0], [0, 0, (1 - 2 * v) / 2]]
                            )
                        )
                else:
                    C_voigt = (
                        E
                        / ((1 + v) * (1 - 2 * v))
                        * np.array(
                            [
                                [1 - v, v, v, 0, 0, 0],
                                [v, 1 - v, v, 0, 0, 0],
                                [v, v, 1 - v, 0, 0, 0],
                                [0, 0, 0, (1 - 2 * v) / 2, 0, 0],
                                [0, 0, 0, 0, (1 - 2 * v) / 2, 0],
                                [0, 0, 0, 0, 0, (1 - 2 * v) / 2],
                            ]
                        )
                    )

                c = kelvin_mandel.From_Voigt(C_voigt)

                test_C = np.linalg.norm(c - mat.C) / np.linalg.norm(c)
                assert test_C < 1e-12

    def test_ElasticAnisot(self):

        C_voigt2D = np.array([[60, 20, 0], [20, 120, 0], [0, 0, 30]])

        C_voigt3D = np.array(
            [
                [60, 20, 10, 0, 0, 0],
                [20, 120, 80, 0, 0, 0],
                [10, 80, 300, 0, 0, 0],
                [0, 0, 0, 400, 0, 0],
                [0, 0, 0, 0, 500, 0],
                [0, 0, 0, 0, 0, 600],
            ]
        )

        axis1_1 = np.array([1, 0, 0])
        axis2_1 = np.array([0, 1, 0])

        a = 30 * np.pi / 130
        axis1_2 = np.array([np.cos(a), np.sin(a), 0])
        axis2_2 = np.array([-np.sin(a), np.cos(a), 0])

        mat_2D_1 = Anisotropic(2, _Rotate_2D(C_voigt2D, axis1_1, axis2_1), True)

        mat_2D_2 = Anisotropic(2, _Rotate_2D(C_voigt2D, axis1_2, axis2_2), True)

        mat_2D_3 = Anisotropic(2, C_voigt2D, True)

        C_3D = kelvin_mandel.From_Voigt(C_voigt3D)
        mat_3D_1 = Anisotropic(3, Apply_Pmat(Get_Pmat(axis1_1, axis2_1), C_3D), False)
        mat_3D_2 = Anisotropic(3, Apply_Pmat(Get_Pmat(axis1_2, axis2_2), C_3D), False)

        listComp = [mat_2D_1, mat_2D_2, mat_2D_3, mat_3D_1, mat_3D_2]

        for comp in listComp:
            matC = comp.C
            test_Symetry = np.linalg.norm(matC.T - matC)
            assert test_Symetry <= 1e-12

    @pytest.mark.parametrize("dim", [2, 3])
    def test_Anisot_from_voigt(self, dim: int):
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

    def test_Elastic_IsotTrans(self):

        El = 11580
        Et = 500
        Gl = 450
        vl = 0.02
        vt = 0.44

        # material_cM = np.array([[El+4*vl**2*kt, 2*kt*vl, 2*kt*vl, 0, 0, 0],
        #               [2*kt*vl, kt+Gt, kt-Gt, 0, 0, 0],
        #               [2*kt*vl, kt-Gt, kt+Gt, 0, 0, 0],
        #               [0, 0, 0, 2*Gt, 0, 0],
        #               [0, 0, 0, 0, 2*Gl, 0],
        #               [0, 0, 0, 0, 0, 2*Gl]])

        # axis_l = [1, 0, 0] et axis_t = [0, 1, 0]
        mat1 = TransverselyIsotropic(
            2,
            El=El,
            Et=Et,
            Gl=Gl,
            vl=vl,
            vt=vt,
            planeStress=False,
            axis_l=np.array([1, 0, 0]),
            axis_t=np.array([0, 1, 0]),
        )

        Gt = mat1.Gt
        kt = mat1.kt

        c1 = np.array(
            [
                [El + 4 * vl**2 * kt, 2 * kt * vl, 0],
                [2 * kt * vl, kt + Gt, 0],
                [0, 0, 2 * Gl],
            ]
        )

        test_c1 = np.linalg.norm(c1 - mat1.C) / np.linalg.norm(c1)
        assert test_c1 < 1e-12

        # axis_l = [0, 1, 0] et axis_t = [1, 0, 0]
        mat2 = TransverselyIsotropic(
            2,
            El=El,
            Et=Et,
            Gl=Gl,
            vl=vl,
            vt=vt,
            planeStress=False,
            axis_l=np.array([0, 1, 0]),
            axis_t=np.array([1, 0, 0]),
        )

        c2 = np.array(
            [
                [kt + Gt, 2 * kt * vl, 0],
                [2 * kt * vl, El + 4 * vl**2 * kt, 0],
                [0, 0, 2 * Gl],
            ]
        )

        test_c2 = np.linalg.norm(c2 - mat2.C) / np.linalg.norm(c2)
        assert test_c2 < 1e-12

        # axis_l = [0, 0, 1] et axis_t = [1, 0, 0]
        mat = TransverselyIsotropic(
            2,
            El=El,
            Et=Et,
            Gl=Gl,
            vl=vl,
            vt=vt,
            planeStress=False,
            axis_l=[0, 0, 1],
            axis_t=[1, 0, 0],
        )

        c3 = np.array([[kt + Gt, kt - Gt, 0], [kt - Gt, kt + Gt, 0], [0, 0, 2 * Gt]])

        test_c3 = np.linalg.norm(c3 - mat.C) / np.linalg.norm(c3)
        assert test_c3 < 1e-12

        mat.Walpole_Decomposition()

    def test_Elastic_Orthotropic(self):

        El = 11580
        Et = 500
        Gl = 450
        vl = 0.02
        vt = 0.44

        # axis_l = [0, 0, 1] et axis_t = [0, 1, 0]
        mat0_isot = Isotropic(3, E=El, v=vl, planeStress=False)
        mu = mat0_isot.get_mu()

        mat0_ortho = Orthotropic(
            3,
            E1=El,
            E2=El,
            E3=El,
            G23=mu,
            G13=mu,
            G12=mu,
            v23=vl,
            v13=vl,
            v12=vl,
            planeStress=False,
            axis_1=[0, 0, 1],
            axis_2=[0, 1, 0],
        )

        test_c0 = np.linalg.norm(mat0_ortho.S - mat0_isot.S) / np.linalg.norm(
            mat0_isot.S
        )
        assert test_c0 < 1e-12

        # material_sM = np.array(
        #     [
        #         [1 / El, -vl / El, -vl / El, 0, 0, 0],
        #         [-vl / El, 1 / Et, -vt / Et, 0, 0, 0],
        #         [-vl / El, -vt / Et, 1 / Et, 0, 0, 0],
        #         [0, 0, 0, 1 / (2 * Gt), 0, 0],
        #         [0, 0, 0, 0, 1 / (2 * Gl), 0],
        #         [0, 0, 0, 0, 0, 1 / (2 * Gl)],
        #     ]
        # )

        # axis_l = [1, 0, 0] et axis_t = [0, 1, 0]
        mat1_isotTrans = TransverselyIsotropic(
            3,
            El=El,
            Et=Et,
            Gl=Gl,
            vl=vl,
            vt=vt,
            planeStress=False,
            axis_l=[1, 0, 0],
            axis_t=[0, 1, 0],
        )
        Gt = mat1_isotTrans.Gt

        mat1_ortho = Orthotropic(
            3,
            E1=El,
            E2=Et,
            E3=Et,
            G23=Gt,
            G13=Gl,
            G12=Gl,
            v23=vt,
            v13=vl,
            v12=vl,
            planeStress=False,
            axis_1=[1, 0, 0],
            axis_2=[0, 1, 0],
        )

        test_c1 = np.linalg.norm(mat1_ortho.S - mat1_isotTrans.S) / np.linalg.norm(
            mat1_isotTrans.S
        )
        assert test_c1 < 1e-12

        # axis_l = [0, 1, 0] et axis_t = [1, 0, 0]
        mat2_isotTrans = TransverselyIsotropic(
            2,
            El=El,
            Et=Et,
            Gl=Gl,
            vl=vl,
            vt=vt,
            planeStress=False,
            axis_l=[0, 1, 0],
            axis_t=[1, 0, 0],
        )
        Gt = mat2_isotTrans.Gt

        mat2_ortho = Orthotropic(
            2,
            E1=El,
            E2=Et,
            E3=Et,
            G23=Gt,
            G13=Gl,
            G12=Gl,
            v23=vt,
            v13=vl,
            v12=vl,
            planeStress=False,
            axis_1=[0, 1, 0],
            axis_2=[1, 0, 0],
        )

        test_c2 = np.linalg.norm(mat2_ortho.C - mat2_isotTrans.C) / np.linalg.norm(
            mat2_isotTrans.C
        )
        assert test_c2 < 1e-12

        mat2_ortho.Walpole_Decomposition()

    @pytest.mark.parametrize("shape", [(), (11,), (11, 4)])
    def test_sqrt_C_S(self, shape: tuple):
        """The square root squares back, however many leading axes C carries.

        (11, 4) is material properties given per Gauss point, which the old implementation
        refused outright.
        """
        n = int(np.prod(shape))
        E = 210e9 if shape == () else np.linspace(200e9, 220e9, n).reshape(shape)
        material = Isotropic(3, E=E, v=0.3)

        sqrtC, sqrtS = material.Get_sqrt_C_S()
        C, S = material.C, material.S

        assert sqrtC.shape == shape + (6, 6)
        assert np.linalg.norm(sqrtC @ sqrtC - C) / np.linalg.norm(C) < 1e-12
        assert np.linalg.norm(sqrtS @ sqrtS - S) / np.linalg.norm(S) < 1e-12
        assert np.linalg.norm(sqrtC @ sqrtS - np.eye(6)) < 1e-12

    @pytest.mark.parametrize("shape", [(), (11,), (11, 4)])
    def test_str_heterogeneous(self, shape: tuple):
        """Printing a material works whatever the shape of its parameters; a field prints as its shape and range."""
        E = np.full(shape, 210e3)
        expected = "2.1e+05" if shape == () else f"{shape} in [2.1e+05, 2.1e+05]"
        for material in (
            Isotropic(2, E=E, v=0.3),
            TransverselyIsotropic(2, El=E, Et=800, Gl=500, vl=0.3, vt=0.4),
            Orthotropic(
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
        ):
            assert f"= {expected}\n" in str(material)

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

    def test_getPmat(self):

        Ne = 10
        p = 3

        _ = 1
        _e = np.ones((Ne))
        _e_pg = np.ones((Ne, p))
        _e2 = np.linspace(1, 1.001, Ne)

        El = 15716.16722094732
        Et = 232.6981580878141
        Gl = 557.3231495541391
        vl = 0.02
        vt = 0.44

        for dim in [2, 3]:

            axis1 = np.array([1, 0, 0])[:dim]
            axis2 = np.array([0, 1, 0])[:dim]

            angles = np.linspace(0, np.pi, Ne)
            x1, y1 = np.cos(angles), np.sin(angles)
            x2, y2 = -np.sin(angles), np.cos(angles)
            axis1_e = np.zeros((Ne, dim))
            axis1_e[:, 0] = x1
            axis1_e[:, 1] = y1
            axis2_e = np.zeros((Ne, dim))
            axis2_e[:, 0] = x2
            axis2_e[:, 1] = y2

            axis1_e_p = axis1_e[:, np.newaxis].repeat(p, 1)
            axis2_e_p = axis2_e[:, np.newaxis].repeat(p, 1)

            for c in [_, _e, _e_pg, _e2]:

                mat = TransverselyIsotropic(dim, El * c, Et * c, Gl * c, vl * c, vt * c)
                C = mat.C
                S = mat.S

                for ax1, ax2 in [
                    (axis1, axis2),
                    (axis1_e, axis2_e),
                    (axis1_e_p, axis2_e_p),
                ]:
                    Pmat = Get_Pmat(ax1, ax2)

                    # checks mat to global coord
                    Cglob = Apply_Pmat(Pmat, C)
                    Sglob = Apply_Pmat(Pmat, S)
                    self.__check_invariants(Cglob, C)
                    self.__check_invariants(Sglob, S)

                    # checks global to mat coord
                    Cmat = Apply_Pmat(Pmat, Cglob, toGlobal=False)
                    Smat = Apply_Pmat(Pmat, Sglob, toGlobal=False)
                    self.__check_invariants(Cmat, C, True)
                    self.__check_invariants(Smat, S, True)

                    # checks Ps, Pe
                    Ps, Pe = Get_Pmat(ax1, ax2, False)
                    transp = np.arange(Ps.ndim)
                    transp[-1], transp[-2] = transp[-2], transp[-1]
                    # checks inv(Ps) = Pe'
                    testPs = np.linalg.norm(
                        np.linalg.inv(Ps) - Pe.transpose(transp)
                    ) / np.linalg.norm(Pe.transpose(transp))
                    assert testPs <= 1e-12, f"inv(Ps) != Pe' -> {testPs:.3e}"
                    # checks inv(Pe) = Ps'
                    testPe = np.linalg.norm(
                        np.linalg.inv(Pe) - Ps.transpose(transp)
                    ) / np.linalg.norm(Ps.transpose(transp))
                    assert testPe <= 1e-12, f"inv(Pe) = Ps' -> {testPe:.3e}"

    def __check_invariants(self, mat1: np.ndarray, mat2: np.ndarray, checkSame=False):

        tol = 1e-12

        shape1, dim1 = mat1.shape, mat1.ndim
        shape2, dim2 = mat2.shape, mat2.ndim

        if dim1 > dim2:
            pass
            if dim2 == 3:
                mat2 = mat2[:, np.newaxis].repeat(shape1[1], 1)
            elif dim2 == 4:
                mat2 = mat2[np.newaxis, np.newaxis].repeat(shape1[0], 0)
                mat2 = mat2.repeat(shape1[1], 1)
        elif dim2 > dim1:
            pass
            if dim1 == 3:
                mat1 = mat1[:, np.newaxis].repeat(shape2[1], 1)
            elif dim1 == 4:
                mat1 = mat1[np.newaxis, np.newaxis].repeat(shape2[0], 0)
                mat1 = mat1.repeat(shape2[1], 1)

        tr1 = np.trace(mat1, axis1=-2, axis2=-1)
        tr2 = np.trace(mat2, axis1=-2, axis2=-1)
        trErr = (tr1 - tr2) / tr2
        test_trace = np.linalg.norm(trErr)
        assert (
            test_trace <= tol
        ), f"The trace is not preserved during the process (test_trace = {test_trace:.3e})"

        det1 = np.linalg.det(mat1)
        det2 = np.linalg.det(mat2)
        detErr = (det1 - det2) / det2
        test_det = np.linalg.norm(detErr)
        assert (
            test_det <= tol
        ), f"The determinant is not preserved during the process (test_det = {test_det:.3e})"

        if checkSame:
            matErr = mat1 - mat2
            test_mat = np.linalg.norm(matErr) / np.linalg.norm(mat2)
            assert test_mat <= tol, "mat1 != mat2"


# ----------------------------------------------
# Every law against Anisotropic built from its 3D C
# ----------------------------------------------

NE = 4


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


def _Assert_close(actual: np.ndarray, expected: np.ndarray):
    assert actual.shape == expected.shape
    assert np.abs(actual - expected).max() <= 1e-12 * np.abs(expected).max()


class TestElasticPipeline:

    @pytest.mark.parametrize("law, dim, planeStress, axes, E", list(_Cases()))
    def test_law_is_anisotropic_of_its_3d_C(self, law, dim, planeStress, axes, E):
        E = 1.0 if E == "uniform" else np.linspace(1, 2, NE)
        material = _Law(law, dim, planeStress, axes, E)

        w = kelvin_mandel.Weights(3)
        aniso = Anisotropic(dim, material._Get_C_3D() / np.outer(w, w), True)
        aniso.planeStress = material.planeStress

        _Assert_close(aniso.C, material.C)
        _Assert_close(aniso.S, material.S)

    @pytest.mark.parametrize("law, dim, planeStress, axes, E", list(_Cases((2,))))
    def test_2d_law_is_anisotropic_of_its_2d_C(self, law, dim, planeStress, axes, E):
        E = 1.0 if E == "uniform" else np.linspace(1, 2, NE)
        material = _Law(law, dim, planeStress, axes, E)

        w = kelvin_mandel.Weights(2)
        aniso = Anisotropic(2, material.C / np.outer(w, w), True)

        _Assert_close(aniso.C, material.C)
        _Assert_close(aniso.S, material.S)

    def test_a_C_cannot_be_assigned(self):
        aniso = Anisotropic(3, Isotropic(3).C, False)
        with pytest.raises(AttributeError):
            aniso.C = Isotropic(3).C

    def test_a_6x6_C_takes_plane_stress_in_2d(self):
        aniso = Anisotropic(2, Isotropic(3).C, False)
        _Assert_close(aniso.C, Isotropic(2, planeStress=False).C)
        aniso.planeStress = True
        _Assert_close(aniso.C, Isotropic(2, planeStress=True).C)

    def test_a_3x3_C_is_2d_only(self):
        C = Isotropic(2, planeStress=False).C
        with pytest.raises(ValueError, match="2D"):
            Anisotropic(3, C, False)
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
        _Assert_close(ti.C, C)

    @pytest.mark.parametrize("law", [TransverselyIsotropic, Orthotropic])
    def test_walpole_sums_to_C_at_every_point(self, law):
        material = _Law(law, 3, False, "(Ne,3)", 1.0)
        ci, Ei = material.Walpole_Decomposition()
        assert Ei.shape == (len(ci), NE, 6, 6)
        _Assert_close(np.einsum("k,k...->...", ci, Ei), material._Get_C_3D())

    @pytest.mark.parametrize("law", [TransverselyIsotropic, Orthotropic])
    def test_walpole_with_heterogeneous_moduli(self, law):
        material = _Law(law, 3, False, "none", np.linspace(1, 2, NE))
        ci, Ei = material.Walpole_Decomposition()
        _Assert_close(np.tensordot(ci, Ei, axes=(0, 0)), material._Get_C_3D())

    def test_axes_per_gauss_point(self):
        axis1, axis2 = _Axes("(Ne,3)")
        nPg = 3
        axis1_p, axis2_p = [np.repeat(a[:, np.newaxis], nPg, 1) for a in (axis1, axis2)]
        ti = TransverselyIsotropic(3, 11580, 500, 450, 0.02, 0.44, axis1, axis2)
        ti_p = TransverselyIsotropic(3, 11580, 500, 450, 0.02, 0.44, axis1_p, axis2_p)
        _Assert_close(ti_p.C, np.repeat(ti.C[:, np.newaxis], nPg, 1))

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
        _Assert_close(ti.C, ti_r.C)

    def test_orthotropic_bounds_raise(self):
        material = Orthotropic(3, 1, 100, 1, 1, 1, 1, 0.3, 0.3, 0.3)
        with pytest.raises(ValueError, match="v12"):
            material.C

    def test_available_laws_list_orthotropic(self):
        assert Orthotropic in _Elastic.Available_Laws()
