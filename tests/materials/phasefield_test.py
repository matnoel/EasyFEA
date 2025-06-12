# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import Geoms, Mesher, Simulations, np

# materials
from EasyFEA.Materials import _Elas, Elas_Isot, Elas_IsotTrans, Elas_Anisot, PhaseField
from EasyFEA.materials import Reshape_variable
from EasyFEA.fem._linalg import Norm
from EasyFEA.fem import FeArray


from .linear_elastic_test import setup_elastic_materials


@pytest.fixture
def setup_pfm_materials(setup_elastic_materials) -> list[PhaseField]:

    splits = PhaseField.Get_splits()
    regularizations = PhaseField.Get_regularisations()
    phaseFieldModels: list[PhaseField] = []

    splits_Isot = [
        PhaseField.SplitType.Amor,
        PhaseField.SplitType.Miehe,
        PhaseField.SplitType.Stress,
    ]

    for c in setup_elastic_materials:
        for s in splits:
            for r in regularizations:

                if (
                    isinstance(c, Elas_IsotTrans) or isinstance(c, Elas_Anisot)
                ) and s in splits_Isot:
                    continue

                pfm = PhaseField(c, s, r, 1, 1)
                phaseFieldModels.append(pfm)

    return phaseFieldModels


class TestPhaseField:

    def __cal_eps(self, dim) -> np.ndarray:

        mat = Elas_Isot(dim)

        L = 200
        H = 100
        domain = Geoms.Domain(Geoms.Point(), Geoms.Point(L, H))
        circle = Geoms.Circle(Geoms.Point(L / 2, H / 2), H / 3)

        if dim == 2:
            mesh = Mesher().Mesh_2D(domain, [circle], "TRI3")
        else:
            mesh = Mesher().Mesh_Extrude(domain, [circle], [0, 0, H / 3], [4], "PRISM6")

        simu = Simulations.ElasticSimu(mesh, mat, useIterativeSolvers=False)
        simu.add_dirichlet(
            mesh.Nodes_Conditions(lambda x, y, z: x == 0),
            [0] * simu.Get_dof_n(),
            simu.Get_unknowns(),
        )
        simu.add_dirichlet(
            mesh.Nodes_Conditions(lambda x, y, z: x == L),
            [L * 1e-4, -L * 1e-4],
            ["x", "y"],
        )
        u = simu.Solve()

        Epsilon_e_pg = simu._Calc_Epsilon_e_pg(u, "mass")

        return Epsilon_e_pg

    def test_split_phaseField(self, setup_pfm_materials):

        phaseFieldModels: list[PhaseField] = setup_pfm_materials

        print()

        # computes 2D strain field
        Epsilon2D_e_pg = self.__cal_eps(2)
        # comutes 3D strain field
        Epsilon3D_e_pg = self.__cal_eps(3)

        for pfm in phaseFieldModels:

            mat: _Elas = pfm.material

            print(
                f"{type(mat).__name__} {mat.simplification} {pfm.split} {pfm.regularization}"
            )

            if mat.dim == 2:
                Epsilon_e_pg = FeArray(Epsilon2D_e_pg)
            elif mat.dim == 3:
                Epsilon_e_pg = FeArray(Epsilon3D_e_pg)

            C_e_pg = Reshape_variable(mat.C, *Epsilon_e_pg.shape[:2])
            cP_e_pg, cM_e_pg = pfm.Calc_C(Epsilon_e_pg.copy(), verif=True)

            # Rounding errors in the construction of 3D eigen projectors see [Remark M] in EasyFEA/materials/_phaseField.py
            tol = 1e-12 if mat.dim == 2 else 1e-10

            # Check that cP + cM = c
            cpm = cP_e_pg + cM_e_pg
            decomp_C = C_e_pg - cpm
            test_C = Norm(decomp_C, axis=(-2, -1)) / Norm(mat.C, axis=(-2, -1))
            assert np.max(test_C) <= tol, f"test_C = {np.max(test_C):.3e}"

            # Check that SigP + SigM = Sig
            Sig_e_pg = C_e_pg @ Epsilon_e_pg
            SigP = cP_e_pg @ Epsilon_e_pg
            SigM = cM_e_pg @ Epsilon_e_pg
            decomp_Sig = Sig_e_pg - (SigP + SigM)
            test_Sig = Norm(decomp_Sig, axis=-1) / Norm(Sig_e_pg, axis=-1)
            if np.min(Norm(Sig_e_pg, axis=-1)) > 0:
                assert np.max(test_Sig) < tol, f"test_Sig = {np.max(test_Sig):.3e}"

            # Check that Eps:C:Eps = Eps:(cP+cM):Eps
            psi = 1 / 2 * (Sig_e_pg @ Epsilon_e_pg).sum((0, 1))
            psi_P = 1 / 2 * (SigP @ Epsilon_e_pg).sum((0, 1))
            psi_M = 1 / 2 * (SigM @ Epsilon_e_pg).sum((0, 1))
            test_psi = np.abs(psi - (psi_P + psi_M)) / psi
            if psi > 0:
                assert test_psi < tol, f"test_psi = {test_psi:.3e}"
