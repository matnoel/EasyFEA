# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from enum import Enum

import numpy as np

from EasyFEA import Display, Folder, PyVista, MatrixType, Models, Simulations, AlgoType
from EasyFEA.FEM import Operators
from EasyFEA.Utilities import _params

from utils import RESULTS_DIR, DATA_DIR, Get_config, Get_tau_and_pressure


class Config(str, Enum):
    D = "D"
    A = "A"
    B = "B"


class CardiacElastoDynamics(Simulations.HyperElastic):

    pressure = _params.PositiveScalarParameter()

    def __init__(
        self, mesh, model, folder="", tolConv=0.00001, maxIter=20, verbosity=False
    ):
        super().__init__(mesh, model, folder, tolConv, maxIter, verbosity)
        self.pressure = 0.0

    def Construct_local_matrix_system(self, problemType):
        results = super().Construct_local_matrix_system(problemType)

        # current Newton-Raphson iterate (updated via u += delta_u)
        displacement = self._Solver_Get_Newton_Raphson_current_solution()
        if self.algo in AlgoType.Get_Hyperbolic_Types():
            displacement, _, _ = self._Solver_Evaluate_u_v_a_for_time_scheme(
                problemType, displacement
            )

        for groupElem in self.mesh.Get_list_groupElem(self.dim - 1):

            # Newton: A(u) Δu = -R(u) = -(F(u) - b) = -F(u) + b
            # Slot F receives the load (+ contribution to b); for a stiffness
            # contribution α·M added to slot K, the matching residual
            # contribution −α·M·u_t must also be put in slot F: in the
            # Newton+hyperbolic path, _Solver_Apply_Neumann does NOT compute
            # `b -= K @ u_t` automatically (only C@v_t and M@a_t are auto).

            # Following pressure (operator returns slot-ready values)
            tangent_e, residual_e = Operators.NonLinear.FollowingPressure(
                displacement,
                groupElem,
                self.pressure,
                groupElem.Get_Elements_Tag("3"),
                MatrixType.mass,
            )

            # top — isotropic surface mass penalty (Robin α·u + β·u̇ = 0)
            M_e = Operators.Bilinear.UV(groupElem, dof_n=3)
            top_e = groupElem.Get_Elements_Tag("4")

            Ktop_e = np.zeros_like(M_e)
            alpha_top = 1e5
            Ktop_e[top_e] = alpha_top * M_e[top_e]

            Ctop_e = np.zeros_like(M_e)
            beta_top = 5e3
            Ctop_e[top_e] = beta_top * M_e[top_e]

            # epi — normal-direction mass penalty (Robin α·(u·n̂) + β·(u̇·n̂) = 0)
            Ms_e = Operators.Bilinear.MassAlongNormal(groupElem)

            epi_e = groupElem.Get_Elements_Tag("2")

            Kepi_e = np.zeros_like(Ms_e)
            alpha_epi = 1e8
            Kepi_e[epi_e] = alpha_epi * Ms_e[epi_e]

            Cepi_e = np.zeros_like(Ms_e)
            beta_epi = 5e3
            Cepi_e[epi_e] = beta_epi * Ms_e[epi_e]

            # Penalty residual contribution: −K_penalty · u_t at current iterate
            K_penalty_e = Ktop_e + Kepi_e
            assembly_e = groupElem.Get_assembly_e(self.dim)
            u_e = displacement[assembly_e]  # (Ne_surf, nPe·3)
            f_penalty_e = np.einsum("eij,ej->ei", K_penalty_e, u_e)

            results[groupElem] = (
                tangent_e + K_penalty_e,
                Ctop_e + Cepi_e,
                None,
                residual_e - f_penalty_e,
            )

        return results


if __name__ == "__main__":

    Display.Clear()

    # ----------------------------------------------
    # Config
    # ----------------------------------------------

    ellipsoid = "ellipsoid_0.01"

    config = Config.A

    results_dir = Folder.Join(RESULTS_DIR, ellipsoid + f"_{config.name}")

    doSimu = True
    plotGraph = False
    makeMovie = True

    # ----------------------------------------------

    if doSimu:

        # ----------------------------------------------
        # Mesh
        # ----------------------------------------------

        mesh, fibers_e_pg, sheets_e_pg = Get_config(
            Folder.Join(DATA_DIR, ellipsoid),
            MatrixType.rigi,
            plotMesh=False,
            plotTags=False,
            plotFibers=False,
        )

        # ----------------------------------------------
        #
        # ----------------------------------------------

        t_values, tau_values, p_values = Get_tau_and_pressure(Tmax=1.0, Nt=100)
        if config is Config.B:
            tau_values *= 0
        if config is Config.A:
            p_values *= 0

        if plotGraph:
            ax = Display.Init_Axes()
            ax.grid()
            ax.set_xlabel(r"$t$ [s]")
            ax.set_ylabel(r"$\tau(t)$ [Pa]")
            ax.plot(t_values, tau_values)
            name = "active_pressure"
            Display.Save_fig(results_dir, name)

            ax = Display.Init_Axes()
            ax.grid()
            ax.set_xlabel(r"$t$ [s]")
            ax.set_ylabel(r"$p(t)$ [Pa]")
            ax.plot(t_values, p_values)
            name = "pressure"
            Display.Save_fig(results_dir, name)

        # ----------------------------------------------
        # Material
        # ----------------------------------------------

        # solid
        a = 59.0
        a_f = 18472.0
        a_fs = 216.0
        a_s = 2481.0
        b = 8.023
        b_f = 16.026
        b_fs = 11.436
        b_s = 11.12

        material = Models.HyperElastic.HolzapfelOgden(
            dim=3,
            C0=a / 2 / b,
            C1=b,
            C2=a_f / 2 / b_f,
            C3=b_f,
            C4=a_s / 2 / b_s,
            C5=b_s,
            C6=a_fs / 2 / b_fs,
            C7=b_fs,
            K=1e6,
            Mu1=0.0,
            Mu2=0.0,
            T1=fibers_e_pg,
            T2=sheets_e_pg,
            ks=100,
        )
        material.eta = 100.0
        material.Set_active_stress_vec(material.T1)

        # ----------------------------------------------
        # Simulation
        # ----------------------------------------------

        simu = CardiacElastoDynamics(mesh, material)

        dt = t_values[1] - t_values[0]
        simu.Solver_Set_Hyperbolic_Algorithm(dt, algo=AlgoType.midpoint)
        simu.rho = 1000

        # evaluate displacement values
        list_p0_values = []
        list_p1_values = []
        p0 = (0.025, 0.03, 0)
        p1 = (0, 0.03, 0)
        evalCoords = np.array([p0, p1])
        evalElements = simu.mesh.groupElem._Get_nearby_elements(evalCoords)

        for t in t_values:
            simu.Bc_Init()
            simu.pressure = np.interp(t + dt / 2, t_values, p_values)
            material.active_stress = np.interp(t + dt / 2, t_values, tau_values)
            displacement = simu.Solve()
            simu.Save_Iter()
            # evaluate displacement values
            values = simu.mesh.Evaluate_dofsValues_at_coordinates(
                evalCoords, displacement, evalElements
            )
            list_p0_values.append(values[0])
            list_p1_values.append(values[1])

        simu.Save(results_dir)

        for list_p_values, name in zip((list_p0_values, list_p1_values), ("p0", "p1")):
            p_values = np.array(list_p_values)
            for c, u in enumerate(simu.Get_unknowns()):
                ax = Display.Init_Axes()
                ax.plot(t_values, p_values[:, c])
                ax.set_xlabel("Time [s]")
                ax.set_xlabel(f"Displacement {u}-component [m]")
                ax.set_title(f"Particle {name}")
                Display.Save_fig(results_dir, f"{name}_u{u}")

    else:
        simu = Simulations.Load_Simu(results_dir)

    if makeMovie:
        # values = [
        #     simu.results[i]["displacement"].reshape(-1, 3)[:, 0] for i in range(simu.Niter)
        # ]
        # clim = (np.min(values), np.max(values))
        PyVista.Movie_simu(
            simu, "ux", results_dir, "ux.mp4", deformFactor=1, plotMesh=True
        )

    Display.plt.show()
