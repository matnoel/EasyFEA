import unittest
import Materials
from Geom import Domain, Circle, Point, Section, Line
import numpy as np
import Affichage as Affichage
from Mesh import Mesh
from Interface_Gmsh import Interface_Gmsh
import Simulations
from TicTac import Tic
import matplotlib.pyplot as plt

class Test_Simu(unittest.TestCase):
    
    def test_SimulationsPoutreUnitaire(self):
        
        interfaceGmsh = Interface_Gmsh()

        listProblem = ["Flexion","Traction","BiEnca"]
        listElemType = ["SEG2","SEG3","SEG4"]
        listBeamDim = [1,2,3]

        # Géneration des configs
        listConfig = [(problem, elemType, beamDim) for problem in listProblem for elemType in listElemType for beamDim in listBeamDim]

        def PlotAndDelete():
            plt.pause(1e-12)
            plt.close('all')
            

        for problem, elemType, beamDim in listConfig:
            
            if problem in ["Flexion","BiEnca"] and beamDim == 1:
                # Exclusion des configs impossible
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
            
            # SECTION

            section = Section(interfaceGmsh.Mesh_Domain_2D(Domain(Point(x=-b/2, y=-h/2), Point(x=b/2, y=h/2))))

            self.assertTrue((section.aire - b*h) <= 1e-12)
            self.assertTrue((section.Iz - ((b*h**3)/12)) <= 1e-12)

            # MAILLAGE

            if problem in ["Traction"]:

                point1 = Point()
                point2 = Point(x=L)
                line = Line(point1, point2, L/nL)
                poutre = Materials.Poutre_Elas_Isot(line, section, E, v)
                listePoutre = [poutre]

            elif problem in ["Flexion","BiEnca"]:

                point1 = Point()
                point2 = Point(x=L/2)
                point3 = Point(x=L)
                
                line = Line(point1, point3, L/nL)
                poutre = Materials.Poutre_Elas_Isot(line, section, E, v)
                listePoutre = [poutre]

            mesh = interfaceGmsh.Mesh_Lines_1D(listPoutres=listePoutre, elemType=elemType)

            # Modele poutre

            beamModel = Materials.Beam_Model(dim=beamDim, listePoutres=listePoutre)

            # Simulation

            simu = Simulations.Simu_Beam(mesh, beamModel, verbosity=False)

            # Conditions

            if beamModel.dim == 1:
                simu.add_dirichlet(mesh.Nodes_Point(point1),[0],["x"])
                if problem == "BiEnca":
                    simu.add_dirichlet(mesh.Nodes_Point(point3),[0],["x"])
            elif beamModel.dim == 2:
                simu.add_dirichlet(mesh.Nodes_Point(point1),[0,0,0],["x","y","rz"])
                if problem == "BiEnca":
                    simu.add_dirichlet(mesh.Nodes_Point(point3),[0,0,0],["x","y","rz"])
            elif beamModel.dim == 3:
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

            Affichage.Plot_BoundaryConditions(simu)
            PlotAndDelete()
            Affichage.Plot_Result(simu, "ux", plotMesh=False, deformation=False)
            PlotAndDelete()
            if beamModel.dim > 1:
                Affichage.Plot_Result(simu, "uy", plotMesh=False, deformation=False)
                PlotAndDelete()
                Affichage.Plot_Mesh(simu, deformation=True, facteurDef=10)
                PlotAndDelete()

        
            u = simu.Get_Resultat("ux", nodeValues=True)
            if beamModel.dim > 1:
                v = simu.Get_Resultat("uy", nodeValues=True)
                rz = simu.Get_Resultat("rz", nodeValues=True)

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
                u_x = (charge*listX/(E*(section.aire))) + (ro*g*listX/2/E*(2*L-listX))

                self.assertTrue((np.abs(u_x[-1] - u.max())/u_x[-1]) <= erreurMaxAnalytique)

                fig, ax = plt.subplots()
                ax.plot(listX, u_x, label='Analytique', c='blue')
                ax.scatter(mesh.coordo[:,0], u, label='EF', c='red', marker='x', zorder=2)
                ax.set_title(fr"$u(x)$")
                ax.legend()
                PlotAndDelete()

    def test_ResolutionDesSimulationsElastique(self):
        # Pour chaque type de maillage on simule
        
        dim = 2

        # Paramètres géométrie
        L = 120;  #mm
        h = 120;    
        b = 13

        # Charge a appliquer
        P = -800 #N

        # Paramètres maillage
        taille = L/2

        listMesh = Interface_Gmsh.Construction2D(L=L, h=h, taille=taille)
        listMesh.extend(Interface_Gmsh.Construction3D(L=L, h=h, b=b, taille=h/4))

        # Pour chaque type d'element 2D       
        for mesh in listMesh:           

            assert isinstance(mesh, Mesh)

            dim = mesh.dim

            comportement = Materials.Elas_Isot(dim, epaisseur=b)
            
            simu = Simulations.Simu_Displacement(mesh, comportement, verbosity=False)

            noeuds_en_0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
            noeuds_en_L = mesh.Nodes_Conditions(lambda x,y,z: x == L)

            simu.add_dirichlet(noeuds_en_0, [0, 0], ["x","y"], description="Encastrement")
            # simu.add_lineLoad(noeuds_en_L, [-P/h], ["y"])
            simu.add_dirichlet(noeuds_en_L, [lambda x,y,z: 1], ['x'])
            simu.add_surfLoad(noeuds_en_L, [P/h/b], ["y"])
            
            simu.Solve()

            fig, ax, cb = Affichage.Plot_Result(simu, "ux", plotMesh=True, nodeValues=True)
            plt.pause(1e-12)
            plt.close(fig)
            
            simu.Solver_Set_Newton_Raphson_Algorithm(dt=0.5)
            simu.Solve()
            fig, ax, cb = Affichage.Plot_Result(simu, "ax", plotMesh=True,nodeValues=True)
            plt.pause(1e-12)
            plt.close(fig)

    def test_SimulationsThermique(self):
        # Pour chaque type de maillage on simule

        a = 1

        listMesh = Interface_Gmsh.Construction2D(L=a, h=a, taille=a/10)

        listMesh.extend(Interface_Gmsh.Construction3D(L=a, h=a, b=a, taille=a/10, useImport3D=False))

        self.simulationsThermique = []

        for mesh in listMesh:

            assert isinstance(mesh, Mesh)

            dim = mesh.dim

            thermalModel = Materials.Thermal_Model(dim=dim, k=1, c=1, epaisseur=a)

            simu = Simulations.Simu_Thermal(mesh , thermalModel, False)            

            noeuds0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
            noeudsL = mesh.Nodes_Conditions(lambda x,y,z: x == a)

            simu.add_dirichlet(noeuds0, [0], [""])
            simu.add_dirichlet(noeudsL, [40], [""])
            simu.Solve()
            simu.Save_Iteration()

            fig, ax, cb = Affichage.Plot_Result(simu, "thermal", nodeValues=True, plotMesh=True)
            plt.pause(1e-12)
            plt.close(fig)

if __name__ == '__main__':        
    try:
        Affichage.Clear()
        unittest.main(verbosity=2)    
    except:
        print("")   
