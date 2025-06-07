# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import Display, plt, np
from EasyFEA.Geoms import Domain, Circle, Point, Line
from EasyFEA import Mesher, ElemType
from EasyFEA import Materials, Simulations


class TestBeam:

    def test_Beam(self):

        interfaceGmsh = Mesher()

        listProblem = ["Flexion", "Traction", "BiEnca"]
        listElemType = ["SEG2", "SEG3", "SEG4", "SEG5"]
        listBeamDim = [1, 2, 3]

        # Generating configs
        listConfig = [
            (problem, elemType, beamDim)
            for problem in listProblem
            for elemType in listElemType
            for beamDim in listBeamDim
        ]

        def PlotAndDelete():
            # plt.pause(1e-12)
            plt.close("all")

        for problem, elemType, beamDim in listConfig:

            if problem in ["Flexion", "BiEnca"] and beamDim == 1:
                # not available
                continue

            print(f"{problem} {elemType} {beamDim}")

            if problem in ["Flexion", "BiEnca"]:
                L = 120
                nL = 10
                h = 13
                b = 13
                E = 210000
                v = 0.3
                charge = 800

                ro = 1
                mass = L * h * b

            elif problem == "Traction":
                L = 10  # m
                nL = 10

                h = 0.1
                b = 0.1
                E = 200000e6
                ro = 7800
                v = 0.3
                g = 10
                q = ro * g * (h * b)
                charge = 5000

                mass = L * h * b * ro

            # Section
            section = interfaceGmsh.Mesh_2D(
                Domain(Point(x=-b / 2, y=-h / 2), Point(x=b / 2, y=h / 2))
            )

            # Mesh
            if problem in ["Traction"]:

                point1 = Point()
                point2 = Point(x=L)
                line = Line(point1, point2, L / nL)
                beam = Materials.Beam_Elas_Isot(beamDim, line, section, E, v)
                listePoutre = [beam]

            elif problem in ["Flexion", "BiEnca"]:

                point1 = Point()
                point2 = Point(x=L / 2)
                point3 = Point(x=L)

                line = Line(point1, point3, L / nL)
                beam = Materials.Beam_Elas_Isot(beamDim, line, section, E, v)
                listePoutre = [beam]

            Iz = beam.Iz

            assert (section.area - b * h) <= 1e-12
            assert (Iz - ((b * h**3) / 12)) <= 1e-12

            mesh = interfaceGmsh.Mesh_Beams(beams=listePoutre, elemType=elemType)

            # Modele poutre

            beamStruct = Materials.BeamStructure(listePoutre)

            # Simulation

            simu = Simulations.BeamSimu(mesh, beamStruct, verbosity=False)

            simu.rho = ro

            testMass = (simu.mass - mass) ** 2 / mass**2
            assert testMass <= 1e-12

            # Conditions

            if beamStruct.dim == 1:
                simu.add_dirichlet(mesh.Nodes_Point(point1), [0], ["x"])
                if problem == "BiEnca":
                    simu.add_dirichlet(mesh.Nodes_Point(point3), [0], ["x"])
            elif beamStruct.dim == 2:
                simu.add_dirichlet(
                    mesh.Nodes_Point(point1), [0, 0, 0], ["x", "y", "rz"]
                )
                if problem == "BiEnca":
                    simu.add_dirichlet(
                        mesh.Nodes_Point(point3), [0, 0, 0], ["x", "y", "rz"]
                    )
            elif beamStruct.dim == 3:
                simu.add_dirichlet(
                    mesh.Nodes_Point(point1),
                    [0, 0, 0, 0, 0, 0],
                    ["x", "y", "z", "rx", "ry", "rz"],
                )
                if problem == "BiEnca":
                    simu.add_dirichlet(
                        mesh.Nodes_Point(point3),
                        [0, 0, 0, 0, 0, 0],
                        ["x", "y", "z", "rx", "ry", "rz"],
                    )

            if problem == "Flexion":
                simu.add_neumann(mesh.Nodes_Point(point3), [-charge], ["y"])
                # simu.add_surfLoad(mesh.Nodes_Point(point2), [-charge/section.area],["y"])

            elif problem == "BiEnca":
                simu.add_neumann(mesh.Nodes_Point(point2), [-charge], ["y"])
            elif problem == "Traction":
                noeudsLine = mesh.Nodes_Line(line)
                simu.add_lineLoad(noeudsLine, [q], ["x"])
                simu.add_neumann(mesh.Nodes_Point(point2), [charge], ["x"])

            simu.Solve()

            Display.Plot_BoundaryConditions(simu)
            PlotAndDelete()
            Display.Plot_Result(simu, "ux", plotMesh=False)
            PlotAndDelete()
            if beamStruct.dim > 1:
                Display.Plot_Result(simu, "uy", plotMesh=False)
                PlotAndDelete()
                Display.Plot_Mesh(simu, deformFactor=10)
                PlotAndDelete()

            u = simu.Result("ux", nodeValues=True)
            if beamStruct.dim > 1:
                v = simu.Result("uy", nodeValues=True)
                rz = simu.Result("rz", nodeValues=True)

            listX = np.linspace(0, L, 100)
            erreurMaxAnalytique = 1e-2
            if problem == "Flexion":
                v_x = charge / (E * Iz) * (listX**3 / 6 - (L * listX**2) / 2)
                flecheanalytique = charge * L**3 / (3 * E * Iz)

                assert (
                    np.abs(flecheanalytique + v.min()) / flecheanalytique
                ) <= erreurMaxAnalytique

                ax = Display.Init_Axes()
                ax.plot(listX, v_x, label="Analytique", c="blue")
                ax.scatter(
                    mesh.coord[:, 0], v, label="EF", c="red", marker="x", zorder=2
                )
                ax.set_title(rf"$v(x)$")
                ax.legend()
                PlotAndDelete()

                rz_x = charge / E / Iz * (listX**2 / 2 - L * listX)
                rotalytique = -charge * L**2 / (2 * E * Iz)
                assert (
                    np.abs(rotalytique + rz.min()) / rotalytique
                ) <= erreurMaxAnalytique

                ax = Display.Init_Axes()
                ax.plot(listX, rz_x, label="Analytique", c="blue")
                ax.scatter(
                    mesh.coord[:, 0], rz, label="EF", c="red", marker="x", zorder=2
                )
                ax.set_title(rf"$r_z(x)$")
                ax.legend()
                PlotAndDelete()
            elif problem == "Traction":
                u_x = (charge * listX / (E * (section.area))) + (
                    ro * g * listX / 2 / E * (2 * L - listX)
                )

                assert (np.abs(u_x[-1] - u.max()) / u_x[-1]) <= erreurMaxAnalytique

                ax = Display.Init_Axes()
                ax.plot(listX, u_x, label="Analytique", c="blue")
                ax.scatter(
                    mesh.coord[:, 0], u, label="EF", c="red", marker="x", zorder=2
                )
                ax.set_title(rf"$u(x)$")
                ax.legend()
                PlotAndDelete()
            elif problem == "BiEnca":
                flecheanalytique = charge * L**3 / (192 * E * Iz)
                assert (
                    np.abs(flecheanalytique + v.min()) / flecheanalytique
                ) <= erreurMaxAnalytique

    def test_Update_Beam(self):
        """Function use to check that modifications on Beam material activate the update of the simulation"""

        def DoTest(simu: Simulations._Simu) -> None:
            assert simu.needUpdate == True  # should trigger the event
            simu.Need_Update(False)  # init

        sect1 = Mesher().Mesh_2D(Domain(Point(), Point(0.01, 0.01)))

        sect2 = sect1.copy()
        sect2.Rotate(30, sect2.center)

        sect3 = sect2.copy()
        sect3.Rotate(30, sect3.center)

        beam1 = Materials.Beam_Elas_Isot(
            2, Line(Point(), Point(5)), sect1, 210e9, v=0.1
        )
        beam2 = Materials.Beam_Elas_Isot(
            2, Line(Point(5), Point(10)), sect2, 210e9, v=0.1
        )

        beams = [beam1, beam2]

        structure = Materials.BeamStructure(beams)

        mesh = Mesher().Mesh_Beams(beams)

        simu = Simulations.BeamSimu(mesh, structure)
        simu.Get_K_C_M_F()
        assert (
            simu.needUpdate == False
        )  # check that need update is now set to false once Get_K_C_M_F() get called

        for beam in beams:
            beam.E *= 2
            DoTest(simu)
            beam.v = 0.4
            DoTest(simu)
            beam.section = sect3
            DoTest(simu)
