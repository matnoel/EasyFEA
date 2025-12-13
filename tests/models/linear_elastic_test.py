# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import np

# materials
from EasyFEA.Models import _Elas, ElasIsot, ElasIsotTrans, ElasAnisot, ElasOrthotropic
from EasyFEA.models import (
    Get_Pmat,
    Apply_Pmat,
    KelvinMandel_Matrix,
    Reshape_variable,
)


@pytest.fixture
def setup_elastic_materials() -> list[_Elas]:

    elasticMaterials: list[_Elas] = []

    for comp in _Elas.Available_Laws():
        if comp == ElasIsot:
            elasticMaterials.append(ElasIsot(2, E=210e9, v=0.3, planeStress=True))
            elasticMaterials.append(ElasIsot(2, E=210e9, v=0.3, planeStress=False))
            elasticMaterials.append(ElasIsot(3, E=210e9, v=0.3))
        elif comp == ElasIsotTrans:
            c = np.sqrt(2) / 2
            elasticMaterials.append(
                ElasIsotTrans(
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
                ElasIsotTrans(
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
                ElasIsotTrans(
                    2, El=11580, Et=500, Gl=450, vl=0.02, vt=0.44, planeStress=True
                )
            )
            elasticMaterials.append(
                ElasIsotTrans(
                    2, El=11580, Et=500, Gl=450, vl=0.02, vt=0.44, planeStress=False
                )
            )

        elif comp == ElasAnisot:
            C_voigt2D = np.array([[60, 20, 0], [20, 120, 0], [0, 0, 30]])

            axis1_1 = np.array([1, 0, 0])
            axis2_1 = np.array([0, 1, 0])

            tetha = 30 * np.pi / 130
            axis1_2 = np.array([np.cos(tetha), np.sin(tetha), 0])
            axis2_2 = np.array([-np.sin(tetha), np.cos(tetha), 0])

            elasticMaterials.append(ElasAnisot(2, C_voigt2D, True, axis1_1, axis2_1))
            elasticMaterials.append(ElasAnisot(2, C_voigt2D, False, axis1_1, axis2_1))
            elasticMaterials.append(ElasAnisot(2, C_voigt2D, True, axis1_2, axis2_2))
            elasticMaterials.append(ElasAnisot(2, C_voigt2D, False, axis1_2, axis2_2))

    return elasticMaterials


class TestLinearElastic:

    def test_Elas_Isot(self, setup_elastic_materials):

        for mat in setup_elastic_materials:
            assert isinstance(mat, _Elas)
            if isinstance(mat, ElasIsot):
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

                c = KelvinMandel_Matrix(mat.dim, C_voigt)

                test_C = np.linalg.norm(c - mat.C) / np.linalg.norm(c)
                assert test_C < 1e-12

    def test_ElasAnisot(self):

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

        mat_2D_1 = ElasAnisot(2, C_voigt2D, True, axis1_1, axis2_1)

        mat_2D_2 = ElasAnisot(2, C_voigt2D, True, axis1_2, axis2_2)

        mat_2D_3 = ElasAnisot(2, C_voigt2D, True)

        mat_3D_1 = ElasAnisot(3, C_voigt3D, True, axis1_1, axis2_1)
        mat_3D_2 = ElasAnisot(3, C_voigt3D, True, axis1_2, axis2_2)

        listComp = [mat_2D_1, mat_2D_2, mat_2D_3, mat_3D_1, mat_3D_2]

        for comp in listComp:
            matC = comp.C
            test_Symetry = np.linalg.norm(matC.T - matC)
            assert test_Symetry <= 1e-12

    def test_Elas_IsotTrans(self):

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
        mat1 = ElasIsotTrans(
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
        mat2 = ElasIsotTrans(
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
        mat = ElasIsotTrans(
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

    def test_Elas_Orthotropic(self):

        El = 11580
        Et = 500
        Gl = 450
        vl = 0.02
        vt = 0.44

        # axis_l = [0, 0, 1] et axis_t = [0, 1, 0]
        mat0_isot = ElasIsot(3, E=El, v=vl, planeStress=False)
        mu = mat0_isot.get_mu()

        mat0_ortho = ElasOrthotropic(
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
        mat1_isotTrans = ElasIsotTrans(
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

        mat1_ortho = ElasOrthotropic(
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
        mat2_isotTrans = ElasIsotTrans(
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

        mat2_ortho = ElasOrthotropic(
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

                mat = ElasIsotTrans(dim, El * c, Et * c, Gl * c, vl * c, vt * c)
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
