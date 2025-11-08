# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA import Mesher, ElemType, MatrixType, Models, Simulations, np
from EasyFEA.models._hyperelastic import HyperElasticState
from EasyFEA.Geoms import Domain
from EasyFEA.fem._linalg import Trace, Det, Inv, TensorProd
from EasyFEA.models._utils import Project_Kelvin, FeArray


def Get_2d_simulations(ud=1e-6) -> list[Simulations.ElasticSimu]:

    L = 120  # mm
    h = 13
    meshSize = h / 1

    contour = Domain((0, 0), (L, h), meshSize)

    simulations = []

    for elemType in ElemType.Get_2D():

        mesh = Mesher().Mesh_2D(contour, [], ElemType.TRI3)
        simu = Simulations.ElasticSimu(mesh, Models.ElasIsot(2))

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
        simu = Simulations.ElasticSimu(mesh, Models.ElasIsot(3))

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

    return HyperElasticState(simu.mesh, simu.displacement, matrixType)._Compute_C()


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
                test2d_e_pg = (
                    Eps2d_e_pg
                    - HyperElasticState(
                        simu2d.mesh, simu2d.displacement, matrixType
                    ).Compute_Epsilon()
                )

                assert np.linalg.norm(test2d_e_pg) / np.linalg.norm(Eps2d_e_pg) < 1e-12

        # 3d simulation

        for simu3d in Get_2d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                Eps3d_e_pg = simu3d._Calc_Epsilon_e_pg(
                    simu3d.displacement, MatrixType.mass
                )
                test3d_e_pg = (
                    Eps3d_e_pg
                    - HyperElasticState(
                        simu3d.mesh, simu3d.displacement, MatrixType.mass
                    ).Compute_Epsilon()
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
                    HyperElasticState(simu.mesh, u, matrixType).Compute_GreenLagrange(),
                    2,
                )

                diff_eps = e_e_pg - Epsilon_e_pg

                assert np.linalg.norm(diff_eps) < 1e-12

    # --------------------------------------------------------------------------
    # I1
    # --------------------------------------------------------------------------

    def test_I1(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                hyperElasticState = HyperElasticState(
                    simu.mesh, simu.displacement, matrixType
                )

                C_e_pg = hyperElasticState.Compute_C()
                I1 = hyperElasticState.Compute_I1()

                assert np.linalg.norm(I1 - Trace(C_e_pg)) / np.linalg.norm(I1) < 1e-12

    def test_dI1dC(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                hyperElasticState = HyperElasticState(
                    simu.mesh, simu.displacement, matrixType
                )

                dI1dC = hyperElasticState.Compute_dI1dC()

                dI1dC_v = Project_Kelvin(np.eye(3), 2)

                assert np.linalg.norm(dI1dC - dI1dC_v) / np.linalg.norm(dI1dC) < 1e-12

    def test_d2I1dC(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                hyperElasticState = HyperElasticState(
                    simu.mesh, simu.displacement, matrixType
                )

                d2I1dC = hyperElasticState.Compute_d2I1dC()

                d2I1dC_v = np.zeros((6, 6))

                assert np.linalg.norm(d2I1dC - d2I1dC_v) < 1e-12

    # --------------------------------------------------------------------------
    # I2
    # --------------------------------------------------------------------------

    def test_I2(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                hyperElasticState = HyperElasticState(
                    simu.mesh, simu.displacement, matrixType
                )

                C_e_pg = hyperElasticState.Compute_C()

                I2 = hyperElasticState.Compute_I2()

                I2_v = 1 / 2 * (Trace(C_e_pg) ** 2 - Trace(C_e_pg @ C_e_pg))

                assert np.linalg.norm(I2 - I2_v) / np.linalg.norm(I2) < 1e-12

    def test_dI2dC(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                hyperElasticState = HyperElasticState(
                    simu.mesh, simu.displacement, matrixType
                )

                dI2dC = hyperElasticState.Compute_dI2dC()

                C_e_pg = hyperElasticState.Compute_C()

                I1_e_pg = hyperElasticState.Compute_I1()
                C_e_pg = hyperElasticState.Compute_C()

                # I1 * Id - C
                dI2dC_v = (I1_e_pg * np.eye(3)) - C_e_pg
                dI2dC_v = Project_Kelvin(dI2dC_v, 2)

                assert np.linalg.norm(dI2dC - dI2dC_v) / np.linalg.norm(dI2dC) < 1e-12

    def test_d2I2dC(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                hyperElasticState = HyperElasticState(
                    simu.mesh, simu.displacement, matrixType
                )

                d2I2dC = hyperElasticState.Compute_d2I2dC()

                vect1 = np.array([1, 1, 1, 0, 0, 0])
                Id_order2 = TensorProd(vect1, vect1)

                # Id_order4 = np.eye(6)
                # same as
                vect2 = np.eye(3)
                Id_order4 = FeArray.asfearray(
                    Project_Kelvin(TensorProd(vect2, vect2, True)), True
                )

                d2I2dC_v = Id_order2 - Id_order4

                assert (
                    np.linalg.norm(d2I2dC - d2I2dC_v) / np.linalg.norm(d2I2dC) < 1e-12
                )

    # --------------------------------------------------------------------------
    # I3
    # --------------------------------------------------------------------------

    def test_I3(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                hyperElasticState = HyperElasticState(
                    simu.mesh, simu.displacement, matrixType
                )

                C_e_pg = hyperElasticState.Compute_C()

                I3 = hyperElasticState.Compute_I3()

                assert np.linalg.norm(I3 - Det(C_e_pg)) / np.linalg.norm(I3) < 1e-12

    def test_dI3dC(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                hyperElasticState = HyperElasticState(
                    simu.mesh, simu.displacement, matrixType
                )

                dI3dC = hyperElasticState.Compute_dI3dC()

                I3_e_pg = hyperElasticState.Compute_I3()
                C_e_pg = hyperElasticState.Compute_C()

                dI3dC_v = I3_e_pg * Inv(C_e_pg)
                dI3dC_v = Project_Kelvin(dI3dC_v, 2)

                assert np.linalg.norm(dI3dC - dI3dC_v) / np.linalg.norm(dI3dC) < 1e-12

    def test_d2I3dC(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                hyperElasticState = HyperElasticState(
                    simu.mesh, simu.displacement, matrixType
                )

                d2I3dC = hyperElasticState.Compute_d2I3dC()

                C_e_pg = hyperElasticState.Compute_C()
                invC_e_pg = Inv(C_e_pg)
                I3_e_pg = hyperElasticState.Compute_I3()

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

                hyperElasticState = HyperElasticState(
                    simu.mesh, simu.displacement, matrixType
                )

                C_e_pg = hyperElasticState.Compute_C()

                T = np.array([0, 1, 0])

                I4 = hyperElasticState.Compute_I4(T)

                I4_v = FeArray.asfearray(
                    np.einsum("...i,...ij,...j->...", T, C_e_pg, T, optimize="optimal")
                )

                assert np.linalg.norm(I4 - I4_v) / np.linalg.norm(I4_v) < 1e-12

    def __anisotropic_invariants_first_derivatives(T1: np.ndarray, T2: np.ndarray):

        T1 = T1.astype(float) / np.linalg.norm(T1)
        T2 = T2.astype(float) / np.linalg.norm(T2)

        first_derivatives = np.zeros(6)

        compute = lambda mat: np.einsum("...i,...ij,...j", T1, np.array(mat), T2)

        # fmt: off
        first_derivatives[0] = compute([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        first_derivatives[1] = compute([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        first_derivatives[2] = compute([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1]
        ])
        first_derivatives[3] = np.sqrt(2)/2 * compute([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])
        first_derivatives[4] = np.sqrt(2)/2 * compute([
            [0, 0, 1],
            [0, 0, 0],
            [1, 0, 0]
        ])
        first_derivatives[5] = np.sqrt(2)/2 * compute([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ])
        # fmt: on

        return first_derivatives

    def test_dI4dC(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                hyperElasticState = HyperElasticState(
                    simu.mesh, simu.displacement, matrixType
                )

                T = np.array([0, 1, 0])

                dI4dC = hyperElasticState.Compute_dI4dC(T)

                dI4dC_v = TestHyperElastic.__anisotropic_invariants_first_derivatives(
                    T, T
                )

                assert np.linalg.norm(dI4dC - dI4dC_v) / np.linalg.norm(dI4dC_v) < 1e-12

    def test_d2I4dC(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                hyperElasticState = HyperElasticState(
                    simu.mesh, simu.displacement, matrixType
                )

                d2I4dC = hyperElasticState.Compute_d2I4dC()

                d2I4dC_v = np.zeros((6, 6))

                assert np.linalg.norm(d2I4dC - d2I4dC_v) < 1e-12

    # --------------------------------------------------------------------------
    # I6
    # --------------------------------------------------------------------------

    def test_I6(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                hyperElasticState = HyperElasticState(
                    simu.mesh, simu.displacement, matrixType
                )

                C_e_pg = hyperElasticState.Compute_C()

                T = np.array([1, 1, 0])
                T = T.astype(float) / np.linalg.norm(T)

                I6 = hyperElasticState.Compute_I6(T)

                I6_v = FeArray.asfearray(
                    np.einsum("...i,...ij,...j->...", T, C_e_pg, T, optimize="optimal")
                )

                assert np.linalg.norm(I6 - I6_v) / np.linalg.norm(I6_v) < 1e-12

    def test_dI6dC(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                hyperElasticState = HyperElasticState(
                    simu.mesh, simu.displacement, matrixType
                )

                T = np.array([1, 1, 0])

                dI6dC = hyperElasticState.Compute_dI6dC(T)

                dI6dC_v = TestHyperElastic.__anisotropic_invariants_first_derivatives(
                    T, T
                )

                assert np.linalg.norm(dI6dC - dI6dC_v) / np.linalg.norm(dI6dC_v) < 1e-12

    def test_d2I6dC(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                hyperElasticState = HyperElasticState(
                    simu.mesh, simu.displacement, matrixType
                )

                d2I6dC = hyperElasticState.Compute_d2I6dC()

                d2I6dC_v = np.zeros((6, 6))

                assert np.linalg.norm(d2I6dC - d2I6dC_v) < 1e-12

    # --------------------------------------------------------------------------
    # I8
    # --------------------------------------------------------------------------

    def test_I8(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                hyperElasticState = HyperElasticState(
                    simu.mesh, simu.displacement, matrixType
                )

                C_e_pg = hyperElasticState.Compute_C()

                T1 = np.array([1, 1, 0])
                T1 = T1.astype(float) / np.linalg.norm(T1)
                T2 = np.array([0, 1, 0])

                I8 = hyperElasticState.Compute_I8(T1, T2)

                I8_v = FeArray.asfearray(
                    np.einsum(
                        "...i,...ij,...j->...", T1, C_e_pg, T2, optimize="optimal"
                    )
                )

                assert np.linalg.norm(I8 - I8_v) / np.linalg.norm(I8_v) < 1e-12

    def test_dI8dC(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                hyperElasticState = HyperElasticState(
                    simu.mesh, simu.displacement, matrixType
                )

                T1 = np.array([1, 1, 0])
                T2 = np.array([0, 1, 0])

                dI8dC = hyperElasticState.Compute_dI8dC(T1, T2)

                dI8dC_v = TestHyperElastic.__anisotropic_invariants_first_derivatives(
                    T1, T2
                )

                assert np.linalg.norm(dI8dC - dI8dC_v) / np.linalg.norm(dI8dC_v) < 1e-12

    def test_d2I8dC(self):

        for simu in Get_3d_simulations():

            for matrixType in [MatrixType.rigi, MatrixType.mass]:

                hyperElasticState = HyperElasticState(
                    simu.mesh, simu.displacement, matrixType
                )

                d2I8dC = hyperElasticState.Compute_d2I8dC()

                d2I8dC_v = np.zeros((6, 6))

                assert np.linalg.norm(d2I8dC - d2I8dC_v) < 1e-12
