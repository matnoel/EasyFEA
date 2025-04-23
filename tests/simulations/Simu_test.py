# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import Display, plt, np
from EasyFEA.Geoms import Domain, Circle, Point, Line
from EasyFEA import Mesher, ElemType
from EasyFEA import Materials, Simulations

class TestBeamSimu:

    def test_Beam(self):
        
        interfaceGmsh = Mesher()

        listProblem = ["Flexion","Traction","BiEnca"]
        listElemType = ["SEG2","SEG3","SEG4","SEG5"]
        listBeamDim = [1,2,3]

        # Generating configs
        listConfig = [(problem, elemType, beamDim) for problem in listProblem for elemType in listElemType for beamDim in listBeamDim]

        def PlotAndDelete():
            plt.pause(1e-12)
            plt.close('all')            

        for problem, elemType, beamDim in listConfig:
            
            if problem in ["Flexion","BiEnca"] and beamDim == 1:
                # not available
                continue

            print(f"{problem} {elemType} {beamDim}")

            if problem in ["Flexion","BiEnca"]:
                L=120; nL=10
                h=13
                b=13
                E = 210000
                v = 0.3
                charge = 800

                ro = 1
                mass = L * h * b

            elif problem == "Traction":
                L=10 # m
                nL=10

                h=0.1
                b=0.1
                E = 200000e6
                ro = 7800
                v = 0.3
                g = 10
                q = ro * g * (h*b)
                charge = 5000

                mass = L * h * b * ro
            
            # Section
            section = interfaceGmsh.Mesh_2D(Domain(Point(x=-b/2, y=-h/2), Point(x=b/2, y=h/2)))

            # Mesh
            if problem in ["Traction"]:

                point1 = Point()
                point2 = Point(x=L)
                line = Line(point1, point2, L/nL)
                beam = Materials.Beam_Elas_Isot(beamDim, line, section, E, v)
                listePoutre = [beam]

            elif problem in ["Flexion","BiEnca"]:

                point1 = Point()
                point2 = Point(x=L/2)
                point3 = Point(x=L)
                
                line = Line(point1, point3, L/nL)
                beam = Materials.Beam_Elas_Isot(beamDim, line, section, E, v)
                listePoutre = [beam]

            Iz = beam.Iz

            assert (section.area - b*h) <= 1e-12
            assert (Iz - ((b*h**3)/12)) <= 1e-12

            mesh = interfaceGmsh.Mesh_Beams(beams=listePoutre, elemType=elemType)

            # Modele poutre

            beamStruct = Materials.BeamStructure(listePoutre)

            # Simulation

            simu = Simulations.BeamSimu(mesh, beamStruct, verbosity=False)

            simu.rho = ro

            testMass = (simu.mass - mass)**2/mass**2
            assert testMass <= 1e-12

            # Conditions

            if beamStruct.dim == 1:
                simu.add_dirichlet(mesh.Nodes_Point(point1),[0],["x"])
                if problem == "BiEnca":
                    simu.add_dirichlet(mesh.Nodes_Point(point3),[0],["x"])
            elif beamStruct.dim == 2:
                simu.add_dirichlet(mesh.Nodes_Point(point1),[0,0,0],["x","y","rz"])
                if problem == "BiEnca":
                    simu.add_dirichlet(mesh.Nodes_Point(point3),[0,0,0],["x","y","rz"])
            elif beamStruct.dim == 3:
                simu.add_dirichlet(mesh.Nodes_Point(point1),[0,0,0,0,0,0],["x","y","z","rx","ry","rz"])
                if problem == "BiEnca":
                    simu.add_dirichlet(mesh.Nodes_Point(point3),[0,0,0,0,0,0],["x","y","z","rx","ry","rz"])

            if problem == "Flexion":
                simu.add_neumann(mesh.Nodes_Point(point3), [-charge],["y"])
                # simu.add_surfLoad(mesh.Nodes_Point(point2), [-charge/section.area],["y"])
                
            elif problem == "BiEnca":
                simu.add_neumann(mesh.Nodes_Point(point2), [-charge],["y"])
            elif problem == "Traction":
                noeudsLine = mesh.Nodes_Line(line)
                simu.add_lineLoad(noeudsLine, [q],["x"])
                simu.add_neumann(mesh.Nodes_Point(point2), [charge],["x"])

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

            listX = np.linspace(0,L,100)
            erreurMaxAnalytique = 1e-2
            if problem == "Flexion":
                v_x = charge/(E*Iz) * (listX**3/6 - (L*listX**2)/2)
                flecheanalytique = charge*L**3/(3*E*Iz)

                assert (np.abs(flecheanalytique + v.min())/flecheanalytique) <= erreurMaxAnalytique

                ax = Display.Init_Axes()
                ax.plot(listX, v_x, label='Analytique', c='blue')
                ax.scatter(mesh.coord[:,0], v, label='EF', c='red', marker='x', zorder=2)
                ax.set_title(fr"$v(x)$")
                ax.legend()
                PlotAndDelete()

                rz_x = charge/E/Iz*(listX**2/2 - L*listX)
                rotalytique = -charge*L**2/(2*E*Iz)
                assert (np.abs(rotalytique + rz.min())/rotalytique) <= erreurMaxAnalytique

                ax = Display.Init_Axes()
                ax.plot(listX, rz_x, label='Analytique', c='blue')
                ax.scatter(mesh.coord[:,0], rz, label='EF', c='red', marker='x', zorder=2)
                ax.set_title(fr"$r_z(x)$")
                ax.legend()
                PlotAndDelete()
            elif problem == "Traction":
                u_x = (charge*listX/(E*(section.area))) + (ro*g*listX/2/E*(2*L-listX))

                assert (np.abs(u_x[-1] - u.max())/u_x[-1]) <= erreurMaxAnalytique

                ax = Display.Init_Axes()
                ax.plot(listX, u_x, label='Analytique', c='blue')
                ax.scatter(mesh.coord[:,0], u, label='EF', c='red', marker='x', zorder=2)
                ax.set_title(fr"$u(x)$")
                ax.legend()
                PlotAndDelete()
            elif problem == "BiEnca":
                flecheanalytique = charge * L**3 / (192*E*Iz)
                assert (np.abs(flecheanalytique + v.min())/flecheanalytique) <= erreurMaxAnalytique

    def test_Update_Beam(self):
        """Function use to check that modifications on Beam material activate the update of the simulation"""

        def DoTest(simu: Simulations._Simu)-> None:
            assert simu.needUpdate == True # should trigger the event
            simu.Need_Update(False) # init

        sect1 = Mesher().Mesh_2D(Domain(Point(), Point(.01,.01)))
        
        sect2 = sect1.copy()
        sect2.Rotate(30, sect2.center)

        sect3 = sect2.copy()
        sect3.Rotate(30, sect3.center)

        beam1 = Materials.Beam_Elas_Isot(2, Line(Point(), Point(5)), sect1, 210e9, v=.1)
        beam2 = Materials.Beam_Elas_Isot(2, Line(Point(5), Point(10)), sect2, 210e9, v=.1)

        beams = [beam1, beam2]

        structure = Materials.BeamStructure(beams)

        mesh = Mesher().Mesh_Beams(beams)

        simu = Simulations.BeamSimu(mesh, structure)
        simu.Get_K_C_M_F()
        assert simu.needUpdate == False # check that need update is now set to false once Get_K_C_M_F() get called

        for beam in beams:
            beam.E *= 2
            DoTest(simu)
            beam.v = .4
            DoTest(simu)
            beam.section = sect3
            DoTest(simu)

class TestElasticSimu:

    def test_Elastic(self):
        # For each type of mesh one simulates
        
        dim = 2

        # Load to apply
        P = -800 #N

        a = 1

        domain = Domain(Point(0, 0), Point(a, a), a/10)
        inclusions = [Circle(Point(a/2, a/2), a/3, a/10)]

        doMesh2D = lambda elemType: Mesher().Mesh_2D(domain, inclusions, elemType)
        doMesh3D = lambda elemType: Mesher().Mesh_Extrude(domain, inclusions, [0,0,-a], [3], elemType)

        listMesh = [doMesh2D(elemType) for elemType in ElemType.Get_2D()]
        [listMesh.append(doMesh3D(elemType)) for elemType in ElemType.Get_3D()]

        # For each mesh
        for mesh in listMesh:

            dim = mesh.dim

            comportement = Materials.Elas_Isot(dim, thickness=a)
            
            simu = Simulations.ElasticSimu(mesh, comportement, verbosity=False)

            noeuds_en_0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
            noeuds_en_L = mesh.Nodes_Conditions(lambda x,y,z: x == a)

            simu.add_dirichlet(noeuds_en_0, [0, 0], ["x","y"])            
            simu.add_surfLoad(noeuds_en_L, [P/a/a], ['y'])            
            
            simu.Solve()
            simu.Save_Iter()

            # static
            ax = Display.Plot_Result(simu, "ux", plotMesh=True, nodeValues=True)
            plt.pause(1e-12)
            plt.close(ax.figure)

            # dynamic      
            simu.Solver_Set_Hyperbolic_Algorithm(dt=0.1)
            simu.Solve()
            # don't plot because result is not relevant

    def test_Update_Elastic(self):
        """Function use to check that modifications on elastic material activate the update of the simulation"""

        def DoTest(simu: Simulations._Simu)-> None:
            assert simu.needUpdate == True # should trigger the event
            simu.Need_Update(False) # init

        mesh = Mesher().Mesh_2D(Domain(Point(), Point(1,1)))

        matIsot = Materials.Elas_Isot(2)
        # E, v, planeStress

        simu = Simulations.ElasticSimu(mesh, matIsot)
        simu.Get_K_C_M_F()
        assert simu.needUpdate == False # check that need update is now set to false once Get_K_C_M_F() get called
        matIsot.E *= 2
        DoTest(simu)
        matIsot.v = 0.2
        DoTest(simu)
        matIsot.planeStress = not matIsot.planeStress
        DoTest(simu)
        try:
            # must return an error
            matIsot.E = -10
        except AssertionError:
            assert simu.needUpdate == False
        try:
            # must return an error
            matIsot.v = 10
        except AssertionError:
            assert simu.needUpdate == False
        try:
            matIsot.planeStress = 10
        except AssertionError:
            assert simu.needUpdate == False


        matElasIsotTrans = Materials.Elas_IsotTrans(2, 10,10,10,0.1,0.1)
        # El, Et, Gl, vl, vt, planeStress
        simu = Simulations.ElasticSimu(mesh, matElasIsotTrans)
        simu.Get_K_C_M_F()
        assert simu.needUpdate == False
        matElasIsotTrans.El *= 2
        DoTest(simu)
        matElasIsotTrans.Et *= 2
        DoTest(simu)
        matElasIsotTrans.Gl *= 2
        DoTest(simu)
        matElasIsotTrans.vl = .2
        DoTest(simu)
        matElasIsotTrans.vt = .4
        DoTest(simu)
        matElasIsotTrans.planeStress = not matElasIsotTrans.planeStress
        DoTest(simu)

        matAnisot = Materials.Elas_Anisot(2, matElasIsotTrans.C, False, (0,1), (-1,0))
        # Set_C, 
        simu = Simulations.ElasticSimu(mesh, matAnisot)
        simu.Get_K_C_M_F()
        assert simu.needUpdate == False
        matAnisot.Set_C(matIsot.C, False)
        DoTest(simu)

class TestThermalSimu:

    def test_Thermal(self):

        a = 1

        domain = Domain(Point(0, 0), Point(a, a), a/10)
        inclusions = [Circle(Point(a/2, a/2), a/3, a/10)]

        doMesh2D = lambda elemType: Mesher().Mesh_2D(domain, inclusions, elemType)
        doMesh3D = lambda elemType: Mesher().Mesh_Extrude(domain, inclusions, [0,0,-a], [3], elemType)

        listMesh = [doMesh2D(elemType) for elemType in ElemType.Get_2D()]
        [listMesh.append(doMesh3D(elemType)) for elemType in ElemType.Get_3D()]

        self.thermalSimulation = []

        for mesh in listMesh:

            dim = mesh.dim

            thermalModel = Materials.Thermal(dim=dim, k=1, c=1, thickness=a)

            simu = Simulations.ThermalSimu(mesh , thermalModel, False)            

            noeuds0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
            noeudsL = mesh.Nodes_Conditions(lambda x,y,z: x == a)

            simu.add_dirichlet(noeuds0, [0], ["t"])
            simu.add_dirichlet(noeudsL, [40], ["t"])
            simu.Solve()
            simu.Save_Iter()

            ax = Display.Plot_Result(simu, "thermal", nodeValues=True, plotMesh=True)
            plt.pause(1e-12)
            plt.close(ax.figure)

    def test_Update_Thermal(self):
        """Function use to check that modifications on thermal material activate the update of the simulation"""

        def DoTest(simu: Simulations._Simu)-> None:
            assert simu.needUpdate == True # should trigger the event
            simu.Need_Update(False) # init

        mesh = Mesher().Mesh_2D(Domain(Point(), Point(1,1)))

        thermal = Materials.Thermal(2, 1, 1)
        # k, c

        simu = Simulations.ThermalSimu(mesh, thermal)
        simu.Get_K_C_M_F()
        assert simu.needUpdate == False # check that need update is now set to false once Get_K_C_M_F() get called
        thermal.k *= 2
        DoTest(simu)
        thermal.c *= 0.2
        DoTest(simu)

class TestSimu:

    def test_Update_Mesh(self):

        def DoTest(simu: Simulations._Simu)-> None:
            assert simu.needUpdate == True # should trigger the event
            simu.Need_Update(False) # init

        mesh = Mesher().Mesh_2D(Domain(Point(), Point(1,1)))

        thermal = Materials.Thermal(2, 1, 1)
        # k, c

        simu = Simulations.ThermalSimu(mesh, thermal)
        simu.Get_K_C_M_F()
        assert simu.needUpdate == False # check that need update is now set to false once Get_K_C_M_F() get called

        mesh.Rotate(45, mesh.center)
        DoTest(simu)

        mesh.Translate(dy=-10)
        DoTest(simu)

        mesh.Symmetry(mesh.center, (1,0))
        DoTest(simu)

        try:
            # must return an error
            mesh.Rotate(45, mesh.center, direction=(1,0))
        except AssertionError:
            assert simu.needUpdate == False

        try:
            # must return an error
            mesh.Translate(dz=20)
        except AssertionError:
            assert simu.needUpdate == False

        simu.mesh = mesh.copy()
        DoTest(simu)
    
class TestPhaseFieldSimu:
   
    def test_PhaseField(self):
        
        a = 1
        l0 = a/10
        meshSize = l0/2
        mesh = Mesher._Construct_2D_meshes(L=a, h=a, meshSize=meshSize)[5] # take the first mesh

        nodes_0 = mesh.Nodes_Conditions(lambda x,y,z: x==0)
        nodes_a = mesh.Nodes_Conditions(lambda x,y,z: x==a)

        material = Materials.Elas_Isot(2, E=210000, v=0.3, planeStress=True, thickness=1)

        splits = list(Materials.PhaseField.SplitType)
        regularizations = list(Materials.PhaseField.ReguType)

        for split in splits: 
            for regu in regularizations:

                pfm = Materials.PhaseField(material, split, regu, 2700, l0)

                print(f"{split} {regu}")

                simu = Simulations.PhaseFieldSimu(mesh, pfm)

                for ud in np.linspace(0, 5e-8*400, 3):

                    simu.Bc_Init()
                    simu.add_dirichlet(nodes_0, [0, 0], ['x', 'y'])
                    simu.add_dirichlet(nodes_a, [ud], ['x'])

                    simu.Solve()
                    simu.Save_Iter()

    def test_Update_PhaseField(self):
        """Function use to check that modifications on phase field material activate the update of the simulation"""

        def DoTest(simu: Simulations._Simu)-> None:
            assert simu.needUpdate == True # should trigger the event
            simu.Need_Update(False) # init

        mesh = Mesher().Mesh_2D(Domain(Point(), Point(1,1)))

        matIsot = Materials.Elas_Isot(2)
        # E, v, planeStress

        pfm = Materials.PhaseField(matIsot, 'He', 'AT1', 1, .01)
        # split, regu, split, Gc, l0, solver, A

        simu = Simulations.PhaseFieldSimu(mesh, pfm)

        simu.Get_K_C_M_F('elastic')
        assert simu.needUpdate == True
        simu.Get_K_C_M_F('damage')
        assert simu.needUpdate == False
        # matrices are updated once damage and displacement matrices are build

        matIsot.E *= 2
        DoTest(simu)
        matIsot.v = .1
        DoTest(simu)

        pfm.split = "Miehe"
        DoTest(simu)
        pfm.regularization = "AT2"
        DoTest(simu)
        pfm.Gc = 10
        DoTest(simu)
        pfm.l0 = 1
        DoTest(simu)
        pfm.solver = pfm.SolverType.BoundConstrain
        DoTest(simu)
        pfm.A = np.eye(2)*3
        DoTest(simu)