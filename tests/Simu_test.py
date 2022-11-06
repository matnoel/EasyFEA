import unittest
from Materials import Poutre_Elas_Isot, Materiau, Elas_Isot, ThermalModel, BeamModel
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

            section = Section(interfaceGmsh.Mesh_Rectangle_2D(Domain(Point(x=-b/2, y=-h/2), Point(x=b/2, y=h/2))))

            self.assertTrue((section.aire - b*h) <= 1e-12)
            self.assertTrue((section.Iz - ((b*h**3)/12)) <= 1e-12)

            # MAILLAGE

            if problem in ["Traction"]:

                point1 = Point()
                point2 = Point(x=L)
                line = Line(point1, point2, L/nL)
                poutre = Poutre_Elas_Isot(line, section, E, v)
                listePoutre = [poutre]

            elif problem in ["Flexion","BiEnca"]:

                point1 = Point()
                point2 = Point(x=L/2)
                point3 = Point(x=L)
                
                line = Line(point1, point3, L/nL)
                poutre = Poutre_Elas_Isot(line, section, E, v)
                listePoutre = [poutre]

            mesh = interfaceGmsh.Mesh_From_Lines_1D(listPoutres=listePoutre, elemType=elemType)

            # Modele poutre

            beamModel = BeamModel(dim=beamDim, listePoutres=listePoutre)

            materiau = Materiau(beamModel, verbosity=False)

            # Simulation

            simu = Simulations.Simu_Beam(mesh, materiau, verbosity=False)

            # Conditions

            if beamModel.dim == 1:
                simu.add_dirichlet("beam", mesh.Nodes_Point(point1),[0],["x"])
                if problem == "BiEnca":
                    simu.add_dirichlet("beam", mesh.Nodes_Point(point3),[0],["x"])
            elif beamModel.dim == 2:
                simu.add_dirichlet("beam", mesh.Nodes_Point(point1),[0,0,0],["x","y","rz"])
                if problem == "BiEnca":
                    simu.add_dirichlet("beam", mesh.Nodes_Point(point3),[0,0,0],["x","y","rz"])
            elif beamModel.dim == 3:
                simu.add_dirichlet("beam", mesh.Nodes_Point(point1),[0,0,0,0,0,0],["x","y","z","rx","ry","rz"])
                if problem == "BiEnca":
                    simu.add_dirichlet("beam", mesh.Nodes_Point(point3),[0,0,0,0,0,0],["x","y","z","rx","ry","rz"])

            if problem == "Flexion":
                simu.add_pointLoad("beam", mesh.Nodes_Point(point3), [-charge],["y"])
                # simu.add_surfLoad("beam", mesh.Nodes_Point(point2), [-charge/section.aire],["y"])
                
            elif problem == "BiEnca":
                simu.add_pointLoad("beam", mesh.Nodes_Point(point2), [-charge],["y"])
            elif problem == "Traction":
                noeudsLine = mesh.Nodes_Line(line)
                simu.add_lineLoad("beam", noeudsLine, [q],["x"])
                simu.add_pointLoad("beam", mesh.Nodes_Point(point2), [charge],["x"])

            simu.Assemblage()

            simu.Solve()

            Affichage.Plot_BoundaryConditions(simu)
            PlotAndDelete()
            Affichage.Plot_Result(simu, "u", affichageMaillage=False, deformation=False)
            PlotAndDelete()
            if beamModel.dim > 1:
                Affichage.Plot_Result(simu, "v", affichageMaillage=False, deformation=False)
                PlotAndDelete()
                Affichage.Plot_Maillage(simu, deformation=True, facteurDef=10)
                PlotAndDelete()

        
            u = simu.Get_Resultat("u", valeursAuxNoeuds=True)
            if beamModel.dim > 1:
                v = simu.Get_Resultat("v", valeursAuxNoeuds=True)
                rz = simu.Get_Resultat("rz", valeursAuxNoeuds=True)

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

            comportement = Elas_Isot(dim, epaisseur=b)

            materiau = Materiau(comportement, verbosity=False)
            
            simu = Simulations.Simu_Displacement(mesh, materiau, verbosity=False)

            simu.Assemblage()

            noeuds_en_0 = mesh.Nodes_Conditions(conditionX=lambda x: x == 0)
            noeuds_en_L = mesh.Nodes_Conditions(conditionX=lambda x: x == L)

            simu.add_dirichlet("displacement", noeuds_en_0, [0, 0], ["x","y"], description="Encastrement")
            # simu.add_lineLoad("displacement",noeuds_en_L, [-P/h], ["y"])
            simu.add_dirichlet("displacement",noeuds_en_L, [lambda x,y,z: 1], ['x'])
            simu.add_surfLoad("displacement",noeuds_en_L, [P/h/b], ["y"])

            simu.Assemblage(steadyState=False)

            Ke_e = simu.ConstruitMatElem_Dep()
            self.__VerificationConstructionKe(simu, Ke_e)

            simu.Solve(steadyState=True)


            fig, ax, cb = Affichage.Plot_Result(simu, "dx", affichageMaillage=True, valeursAuxNoeuds=True)
            plt.pause(1e-12)
            plt.close(fig)
            
            simu.Set_Newton_Raphson(dt=0.5)
            simu.Solve(steadyState=False)
            fig, ax, cb = Affichage.Plot_Result(simu, "ax", affichageMaillage=True,valeursAuxNoeuds=True)
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

            thermalModel = ThermalModel(dim=dim, k=1, c=1, epaisseur=a)

            materiau = Materiau(thermalModel, verbosity=False)

            simu = Simulations.Simu_Thermal(mesh , materiau, False)

            noeuds0 = mesh.Nodes_Conditions(lambda x: x == 0)
            noeudsL = mesh.Nodes_Conditions(lambda x: x == a)

            simu.add_dirichlet("thermal", noeuds0, [0], [""])
            simu.add_dirichlet("thermal", noeudsL, [40], [""])
            simu.Assemblage(steadyState=True)
            simu.Solve(steadyState=True)
            simu.Save_Iteration()

            fig, ax, cb = Affichage.Plot_Result(simu, "thermal", valeursAuxNoeuds=True, affichageMaillage=True)
            plt.pause(1e-12)
            plt.close(fig)
    

    # ------------------------------------------- Vérifications ------------------------------------------- 

    def __VerificationConstructionKe(self, simu: Simulations.Simu, Ke_e, d=[]):
            """Ici on verifie quon obtient le meme resultat quavant verification vectorisation

            Parameters
            ----------
            Ke_e : nd.array par element
                Matrice calculé en vectorisant        
            d : ndarray
                Champ d'endommagement
            """

            tic = Tic()

            matriceType = "rigi"

            # Data
            mesh = simu.mesh
            nPg = mesh.Get_nPg(matriceType)
            listPg = list(range(nPg))
            Ne = mesh.Ne            
            materiau = simu.materiau
            C = materiau.comportement.get_C()

            listKe_e = []

            B_dep_e_pg = mesh.Get_B_dep_e_pg(matriceType)            

            jacobien_e_pg = mesh.Get_jacobien_e_pg(matriceType)
            poid_pg = mesh.Get_poid_pg(matriceType)
            for e in range(Ne):            
                # Pour chaque poing de gauss on construit Ke
                Ke = 0
                for pg in listPg:
                    jacobien = jacobien_e_pg[e,pg]
                    poid = poid_pg[pg]
                    B_pg = B_dep_e_pg[e,pg]

                    K = jacobien * poid * B_pg.T.dot(C).dot(B_pg)

                    if len(d)==0:   # probleme standart
                        
                        Ke += K
                    else:   # probleme endomagement
                        
                        de = np.array([d[mesh.connect[e]]])
                        
                        # Bourdin
                        g = (1-mesh.N_mass_pg[pg].dot(de))**2
                        # g = (1-de)**2
                        
                        Ke += g * K
                # # print(Ke-listeKe[e.id])
                if mesh.dim == 2:
                    listKe_e.append(Ke)
                else:
                    listKe_e.append(Ke)                

            tic.Tac("Matrices","Calcul des matrices elementaires (boucle)", False)
            
            # Verification
            Ke_comparaison = np.array(listKe_e)*simu.materiau.comportement.epaisseur
            test = Ke_e - Ke_comparaison

            test = np.testing.assert_array_almost_equal(Ke_e, Ke_comparaison, verbose=False)

            self.assertIsNone(test)
            

if __name__ == '__main__':        
    try:
        Affichage.Clear()
        unittest.main(verbosity=2)    
    except:
        print("")   
