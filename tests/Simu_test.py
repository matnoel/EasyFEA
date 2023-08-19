import unittest
import Materials
from Geom import Domain, Circle, Point, Section, Line
import numpy as np
import Display as Display
from Interface_Gmsh import Interface_Gmsh, Mesh, GroupElem
import Simulations
from TicTac import Tic
import matplotlib.pyplot as plt

class Test_Simu(unittest.TestCase):
    
    def test_Beam(self):
        
        interfaceGmsh = Interface_Gmsh()

        listProblem = ["Flexion","Traction","BiEnca"]
        listElemType = ["SEG2","SEG3","SEG4"]
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
            
            # Section

            section = Section(interfaceGmsh.Mesh_2D(Domain(Point(x=-b/2, y=-h/2), Point(x=b/2, y=h/2))))

            self.assertTrue((section.area - b*h) <= 1e-12)
            self.assertTrue((section.Iz - ((b*h**3)/12)) <= 1e-12)

            # Mesh

            if problem in ["Traction"]:

                point1 = Point()
                point2 = Point(x=L)
                line = Line(point1, point2, L/nL)
                poutre = Materials.Beam_Elas_Isot(beamDim, line, section, E, v)
                listePoutre = [poutre]

            elif problem in ["Flexion","BiEnca"]:

                point1 = Point()
                point2 = Point(x=L/2)
                point3 = Point(x=L)
                
                line = Line(point1, point3, L/nL)
                poutre = Materials.Beam_Elas_Isot(beamDim, line, section, E, v)
                listePoutre = [poutre]

            mesh = interfaceGmsh.Mesh_Beams(beamList=listePoutre, elemType=elemType)

            # Modele poutre

            beamStruct = Materials.Beam_Structure(listePoutre)

            # Simulation

            simu = Simulations.Simu_Beam(mesh, beamStruct, verbosity=False)

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
                # simu.add_surfLoad(mesh.Nodes_Point(point2), [-charge/section.aire],["y"])
                
            elif problem == "BiEnca":
                simu.add_neumann(mesh.Nodes_Point(point2), [-charge],["y"])
            elif problem == "Traction":
                noeudsLine = mesh.Nodes_Line(line)
                simu.add_lineLoad(noeudsLine, [q],["x"])
                simu.add_neumann(mesh.Nodes_Point(point2), [charge],["x"])

            simu.Solve()

            Display.Plot_BoundaryConditions(simu)
            PlotAndDelete()
            Display.Plot_Result(simu, "ux", plotMesh=False, deformation=False)
            PlotAndDelete()
            if beamStruct.dim > 1:
                Display.Plot_Result(simu, "uy", plotMesh=False, deformation=False)
                PlotAndDelete()
                Display.Plot_Mesh(simu, deformation=True, facteurDef=10)
                PlotAndDelete()

        
            u = simu.Get_Result("ux", nodeValues=True)
            if beamStruct.dim > 1:
                v = simu.Get_Result("uy", nodeValues=True)
                rz = simu.Get_Result("rz", nodeValues=True)

            listX = np.linspace(0,L,100)
            erreurMaxAnalytique = 1e-2
            if problem == "Flexion":
                v_x = charge/(E*section.Iz) * (listX**3/6 - (L*listX**2)/2)
                flecheanalytique = charge*L**3/(3*E*section.Iz)

                self.assertTrue((np.abs(flecheanalytique + v.min())/flecheanalytique) <= erreurMaxAnalytique)

                fig, ax = plt.subplots()
                ax.plot(listX, v_x, label='Analytique', c='blue')
                ax.scatter(mesh.coordo[:,0], v, label='EF', c='red', marker='x', zorder=2)
                ax.set_title(fr"$v(x)$")
                ax.legend()
                PlotAndDelete()

                rz_x = charge/E/section.Iz*(listX**2/2 - L*listX)
                rotalytique = -charge*L**2/(2*E*section.Iz)
                self.assertTrue((np.abs(rotalytique + rz.min())/rotalytique) <= erreurMaxAnalytique)

                fig, ax = plt.subplots()
                ax.plot(listX, rz_x, label='Analytique', c='blue')
                ax.scatter(mesh.coordo[:,0], rz, label='EF', c='red', marker='x', zorder=2)
                ax.set_title(fr"$r_z(x)$")
                ax.legend()
                PlotAndDelete()
            elif problem == "Traction":
                u_x = (charge*listX/(E*(section.area))) + (ro*g*listX/2/E*(2*L-listX))

                self.assertTrue((np.abs(u_x[-1] - u.max())/u_x[-1]) <= erreurMaxAnalytique)

                fig, ax = plt.subplots()
                ax.plot(listX, u_x, label='Analytique', c='blue')
                ax.scatter(mesh.coordo[:,0], u, label='EF', c='red', marker='x', zorder=2)
                ax.set_title(fr"$u(x)$")
                ax.legend()
                PlotAndDelete()

    def test_Elasticity(self):
        # For each type of mesh one simulates
        
        dim = 2        

        # Load to apply
        P = -800 #N

        a = 1

        domain = Domain(Point(0, 0), Point(a, a), a/10)
        inclusions = [Circle(Point(a/2, a/2), a/3, a/10)]

        doMesh2D = lambda elemType: Interface_Gmsh().Mesh_2D(domain, inclusions, elemType)
        doMesh3D = lambda elemType: Interface_Gmsh().Mesh_3D(domain, inclusions, [0,0,-a], 3, elemType)

        listMesh = [doMesh2D(elemType) for elemType in GroupElem.get_Types2D()]
        [listMesh.append(doMesh3D(elemType)) for elemType in GroupElem.get_Types3D()]

        # For each mesh
        for mesh in listMesh:

            dim = mesh.dim

            comportement = Materials.Elas_Isot(dim, thickness=a)
            
            simu = Simulations.Simu_Displacement(mesh, comportement, verbosity=False)

            noeuds_en_0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
            noeuds_en_L = mesh.Nodes_Conditions(lambda x,y,z: x == a)

            simu.add_dirichlet(noeuds_en_0, [0, 0], ["x","y"])            
            simu.add_surfLoad(noeuds_en_L, [P/a/a], ['y'])            
            
            simu.Solve()

            # static
            fig, ax, cb = Display.Plot_Result(simu, "ux", plotMesh=True, nodeValues=True)
            plt.pause(1e-12)
            plt.close(fig)

            # dynamic      
            simu.Solver_Set_Newton_Raphson_Algorithm(dt=0.1)
            simu.Solve()
            # don't plot because result is not relevant

    def test_Thermal(self):

        a = 1

        domain = Domain(Point(0, 0), Point(a, a), a/10)
        inclusions = [Circle(Point(a/2, a/2), a/3, a/10)]

        doMesh2D = lambda elemType: Interface_Gmsh().Mesh_2D(domain, inclusions, elemType)
        doMesh3D = lambda elemType: Interface_Gmsh().Mesh_3D(domain, inclusions, [0,0,-a], 3, elemType)

        listMesh = [doMesh2D(elemType) for elemType in GroupElem.get_Types2D()]
        [listMesh.append(doMesh3D(elemType)) for elemType in GroupElem.get_Types3D()]

        self.thermalSimulation = []

        for mesh in listMesh:

            assert isinstance(mesh, Mesh)

            dim = mesh.dim

            thermalModel = Materials.Thermal_Model(dim=dim, k=1, c=1, thickness=a)

            simu = Simulations.Simu_Thermal(mesh , thermalModel, False)            

            noeuds0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
            noeudsL = mesh.Nodes_Conditions(lambda x,y,z: x == a)

            simu.add_dirichlet(noeuds0, [0], [""])
            simu.add_dirichlet(noeudsL, [40], [""])
            simu.Solve()
            simu.Save_Iter()

            fig, ax, cb = Display.Plot_Result(simu, "thermal", nodeValues=True, plotMesh=True)
            plt.pause(1e-12)
            plt.close(fig)

    # TODO test phase field

    def test_PhaseField(self):
        
        a = 1
        l0 = a/10
        meshSize = l0/2
        mesh = Interface_Gmsh.Construct_2D_meshes(L=a, h=a, taille=meshSize)[5] # take the first mesh

        nodes_0 = mesh.Nodes_Conditions(lambda x,y,z: x==0)
        nodes_a = mesh.Nodes_Conditions(lambda x,y,z: x==a)

        material = Materials.Elas_Isot(2, E=210000, v=0.3, planeStress=True, thickness=1)

        splits = list(Materials.PhaseField_Model.SplitType)
        regularizations = list(Materials.PhaseField_Model.RegularizationType)

        for split in splits: 
            for regu in regularizations:

                pfm = Materials.PhaseField_Model(material, split, regu, 2700, l0)

                print(f"{split} {regu}")

                simu = Simulations.Simu_PhaseField(mesh, pfm)

                for ud in np.linspace(0, 5e-8*400, 3):

                    simu.Bc_Init()
                    simu.add_dirichlet(nodes_0, [0, 0], ['x', 'y'])
                    simu.add_dirichlet(nodes_a, [ud], ['x'])

                    simu.Solve()
                    simu.Save_Iter()

if __name__ == '__main__':        
    try:
        Display.Clear()
        unittest.main(verbosity=2)    
    except:
        print("")   
