# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA import Mesher, ElemType, MatrixType, Materials, Simulations, np
from EasyFEA.materials._hyperelastic import HyperElastic
from EasyFEA.Geoms import Domain
from EasyFEA.fem._linalg import Trace, Det, Inv, TensorProd
from EasyFEA.materials._utils import Project_Kelvin, FeArray


def Get_2d_simulations(ud=1e-6) -> list[Simulations.ElasticSimu]:

    L = 120  # mm
    h = 13
    meshSize = h / 1

    contour = Domain((0, 0), (L, h), meshSize)

    simulations = []

    for elemType in ElemType.Get_2D():

        mesh = Mesher().Mesh_2D(contour, [], ElemType.TRI3)
        simu = Simulations.ElasticSimu(mesh, Materials.Elas_Isot(2))

        simu.add_dirichlet(
            mesh.Nodes_Conditions(lambda x, y, z: x == 0), [0, 0], simu.Get_unknowns()
        )
        simu.add_dirichlet(mesh.Nodes_Conditions(lambda x, y, z: x == L), [ud], ["x"])
        simu.Solve()

        simulations.append(simu)

    return simulations


def Get_3d_simulations(ud=1e-6) -> list[Simulations.ElasticSimu]:

    L = 120  # mm
    h = 13
    meshSize = h / 1

    contour = Domain((0, 0), (L, h), meshSize)

    simulations = []

    for elemType in ElemType.Get_3D():

        mesh = Mesher().Mesh_Extrude(contour, [], [0, 0, h], [h / meshSize], elemType)
        simu = Simulations.ElasticSimu(mesh, Materials.Elas_Isot(3))

        simu.add_dirichlet(
            mesh.Nodes_Conditions(lambda x, y, z: x == 0),
            [0, 0, 0],
            simu.Get_unknowns(),
        )
        simu.add_dirichlet(mesh.Nodes_Conditions(lambda x, y, z: x == L), [ud], ["x"])
        simu.Solve()

        simulations.append(simu)

    return simulations


def Get_C_components(simu: Simulations.ElasticSimu, matrixType=MatrixType.rigi):

    return HyperElastic._Compute_C(simu.mesh, simu.displacement, matrixType)


class TestHyperElastic:

    # def test_D_e_pg(self):

    #     L = 120 #mm
    #     h = 13
    #     meshSize = h/3

    #     contour = Domain((0,0), (L, h), meshSize)

    #     mesh = Mesher().Mesh_2D(contour, [], ElemType.TRI3)

    #     PyVista.Plot_Mesh(mesh).show()

    #     pass

    def test_Epsilon(self):

        # 2d simulation

        for simu2d in Get_2d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                Eps2d_e_pg = simu2d._Calc_Epsilon_e_pg(simu2d.displacement, matrixType)
                test2d_e_pg = Eps2d_e_pg - HyperElastic.Compute_Epsilon(
                    simu2d.mesh, simu2d.displacement, matrixType
                )

                assert np.linalg.norm(test2d_e_pg) / np.linalg.norm(Eps2d_e_pg) < 1e-12

        # 3d simulation

        for simu3d in Get_2d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                Eps3d_e_pg = simu3d._Calc_Epsilon_e_pg(
                    simu3d.displacement, MatrixType.mass
                )
                test3d_e_pg = Eps3d_e_pg - HyperElastic.Compute_Epsilon(
                    simu3d.mesh, simu3d.displacement, MatrixType.mass
                )

                assert np.linalg.norm(test3d_e_pg) / np.linalg.norm(Eps3d_e_pg) < 1e-12

    def test_C(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                cxx, cxy, cxz, cyx, cyy, cyz, czx, czy, czz = Get_C_components(
                    simu, matrixType
                )

                assert np.linalg.norm(cxy - cyx) / np.linalg.norm(cxy) < 1e-12

                assert np.linalg.norm(cxz - czx) / np.linalg.norm(cxz) < 1e-12

                assert np.linalg.norm(cyz - czy) / np.linalg.norm(cyz) < 1e-12

    def test_Epsilon_vs_GreenLagrange(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                u = simu.displacement
                Epsilon_e_pg = simu._Calc_Epsilon_e_pg(u, matrixType)

                e_e_pg = Project_Kelvin(
                    HyperElastic.Compute_GreenLagrange(simu.mesh, u, matrixType), 2
                )

                diff_eps = e_e_pg - Epsilon_e_pg

                assert np.linalg.norm(diff_eps) < 1e-12

    # --------------------------------------------------------------------------
    # I1
    # --------------------------------------------------------------------------

    def test_I1(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                C_e_pg = HyperElastic.Compute_C(
                    simu.mesh, simu.displacement, matrixType
                )

                I1 = HyperElastic.Compute_I1(simu.mesh, simu.displacement, matrixType)

                assert np.linalg.norm(I1 - Trace(C_e_pg)) / np.linalg.norm(I1) < 1e-12

    def test_dI1dC(self):

        dI1dC = HyperElastic.Compute_dI1dC()

        dI1dC_v = Project_Kelvin(np.eye(3), 2)

        assert np.linalg.norm(dI1dC - dI1dC_v) / np.linalg.norm(dI1dC) < 1e-12

    def test_d2I1dC(self):

        d2I1dC = HyperElastic.Compute_d2I1dC()

        d2I1dC_v = np.zeros((6, 6))

        assert np.linalg.norm(d2I1dC - d2I1dC_v) < 1e-12

    # --------------------------------------------------------------------------
    # I2
    # --------------------------------------------------------------------------

    def test_I2(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                C_e_pg = HyperElastic.Compute_C(
                    simu.mesh, simu.displacement, matrixType
                )

                I2 = HyperElastic.Compute_I2(simu.mesh, simu.displacement, matrixType)

                I2_v = 1 / 2 * (Trace(C_e_pg) ** 2 - Trace(C_e_pg @ C_e_pg))

                assert np.linalg.norm(I2 - I2_v) / np.linalg.norm(I2) < 1e-12

    def test_dI2dC(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                dI2dC = HyperElastic.Compute_dI2dC(
                    simu.mesh, simu.displacement, matrixType
                )

                C_e_pg = HyperElastic.Compute_C(
                    simu.mesh, simu.displacement, matrixType
                )

                mesh, u = simu.mesh, simu.displacement
                I1_e_pg = HyperElastic.Compute_I1(mesh, u, matrixType)
                C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)

                # I1 * Id - C
                dI2dC_v = (I1_e_pg * np.eye(3)) - C_e_pg
                dI2dC_v = Project_Kelvin(dI2dC_v, 2)

                assert np.linalg.norm(dI2dC - dI2dC_v) / np.linalg.norm(dI2dC) < 1e-12

    def test_d2I2dC(self):

        d2I2dC = HyperElastic.Compute_d2I2dC()

        vect1 = np.array([1, 1, 1, 0, 0, 0])
        Id_order2 = TensorProd(vect1, vect1)

        # Id_order4 = np.eye(6)
        # same as
        vect2 = np.eye(3)
        Id_order4 = FeArray.asfearray(
            Project_Kelvin(TensorProd(vect2, vect2, True)), True
        )

        d2I2dC_v = Id_order2 - Id_order4

        assert np.linalg.norm(d2I2dC - d2I2dC_v) / np.linalg.norm(d2I2dC) < 1e-12

    # --------------------------------------------------------------------------
    # I3
    # --------------------------------------------------------------------------

    def test_I3(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                C_e_pg = HyperElastic.Compute_C(
                    simu.mesh, simu.displacement, matrixType
                )

                I3 = HyperElastic.Compute_I3(simu.mesh, simu.displacement, matrixType)

                assert np.linalg.norm(I3 - Det(C_e_pg)) / np.linalg.norm(I3) < 1e-12

    def test_dI3dC(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                dI3dC = HyperElastic.Compute_dI3dC(
                    simu.mesh, simu.displacement, matrixType
                )

                mesh, u = simu.mesh, simu.displacement
                I3_e_pg = HyperElastic.Compute_I3(mesh, u, matrixType)
                C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)

                dI3dC_v = I3_e_pg * Inv(C_e_pg)
                dI3dC_v = Project_Kelvin(dI3dC_v, 2)

                assert np.linalg.norm(dI3dC - dI3dC_v) / np.linalg.norm(dI3dC) < 1e-12

    def test_d2I3dC(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                mesh, u = simu.mesh, simu.displacement

                d2I3dC = HyperElastic.Compute_d2I3dC(mesh, u, matrixType)

                C_e_pg = HyperElastic.Compute_C(mesh, u, matrixType)
                invC_e_pg = Inv(C_e_pg)
                I3_e_pg = HyperElastic.Compute_I3(mesh, u, matrixType)

                p1_e_pg = np.einsum(
                    "...ij,...kl->...ijkl", I3_e_pg * invC_e_pg, invC_e_pg
                )
                p2_e_pg = I3_e_pg * TensorProd(invC_e_pg, invC_e_pg, True, 2)

                d2I3dC_v = Project_Kelvin(p1_e_pg - p2_e_pg, orderA=4)

                assert (
                    np.linalg.norm(d2I3dC - d2I3dC_v) / np.linalg.norm(d2I3dC) < 1e-12
                )

    # --------------------------------------------------------------------------
    # I4
    # --------------------------------------------------------------------------

    def test_I4(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                C_e_pg = HyperElastic.Compute_C(
                    simu.mesh, simu.displacement, matrixType
                )

                T = np.array([0, 1, 0])

                I4 = HyperElastic.Compute_I4(
                    simu.mesh, simu.displacement, T, matrixType
                )

                I4_v = FeArray.asfearray(
                    np.einsum("...i,...ij,...j->...", T, C_e_pg, T, optimize="optimal")
                )

                assert np.linalg.norm(I4 - I4_v) / np.linalg.norm(I4_v) < 1e-12

    def test_dI4dC(self):

        T = np.array([0, 1, 0])

        dI4dC = HyperElastic.Compute_dI4dC(T)

        dI4dC_v = Project_Kelvin(TensorProd(T, T), 2)

        assert np.linalg.norm(dI4dC - dI4dC_v) / np.linalg.norm(dI4dC_v) < 1e-12

    def test_d2I4dC(self):

        d2I4dC = HyperElastic.Compute_d2I4dC()

        d2I4dC_v = np.zeros((6, 6))

        assert np.linalg.norm(d2I4dC - d2I4dC_v) < 1e-12
