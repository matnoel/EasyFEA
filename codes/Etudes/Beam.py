import matplotlib.pyplot as plt
import numpy as np
import Interface_Gmsh
from Geom import Domain, Line, Point, Section
import Affichage
import Materials
import Simu

Affichage.Clear()

interfaceGmsh = Interface_Gmsh.Interface_Gmsh(False, False, False)

problem = "Portique"

elemType = "SEG2"

beamDim = 2

if problem in ["Flexion","BiEnca","Portique"]:
    L=120; nL=1
    h=13
    b=13
    E = 210000
    v = 0.3
    charge = 800    

elif problem == "Traction":
    L=10 # m
    nL=1

    h=0.1
    b=0.1
    E = 200000e6
    ro = 7800
    v = 0.3
    g = 10
    q = ro * g * (h*b)
    charge = 5000

section = Section(interfaceGmsh.Mesh_Rectangle_2D(Domain(Point(x=-b/2, y=-h/2), Point(x=b/2, y=h/2))))

Affichage.Plot_Model(section.mesh)
# Affichage.Plot_Maillage(mesh)
plt.show()

if problem in ["Traction"]:

    point1 = Point()
    point2 = Point(x=L/2)
    point3 = Point(x=L)

    # # Poutre 1 partie
    # line1 = Line(point1, point3, L/nL)
    # poutre1 = Materials.Poutre_Elas_Isot(line1, section, E, v)
    # listePoutre = [poutre1]

    # Poutre 2 partie
    line1 = Line(point1, point2, L/nL)
    line2 = Line(point2, point3, L/nL)
    poutre1 = Materials.Poutre_Elas_Isot(line1, section, E, v)
    poutre2 = Materials.Poutre_Elas_Isot(line2, section, E, v)
    listePoutre = [poutre1, poutre2]
    
elif problem in ["Flexion","BiEnca"]:

    point1 = Point()
    point2 = Point(x=L/2)
    point3 = Point(x=L)

    # # Poutre en 1 partie
    # line = Line(point1, point3, L/nL)
    # poutre = Materials.Poutre_Elas_Isot(line, section, E, v)
    # listePoutre = [poutre]

    # Poutre en 2 partie
    line1 = Line(point1, point2, L/nL)
    line2 = Line(point2, point3, L/nL)
    line = Line(point1, point3)
    poutre1 = Materials.Poutre_Elas_Isot(line1, section, E, v)
    poutre2 = Materials.Poutre_Elas_Isot(line2, section, E, v)
    listePoutre = [poutre1, poutre2]

elif problem == "Portique":

    point1 = Point()
    point2 = Point(y=L)
    point3 = Point(y=L, x=L/2)

    # Poutre en 2 partie
    line1 = Line(point1, point2, L/nL)
    line2 = Line(point2, point3, L/nL)
    poutre1 = Materials.Poutre_Elas_Isot(line1, section, E, v)
    poutre2 = Materials.Poutre_Elas_Isot(line2, section, E, v)
    listePoutre = [poutre1, poutre2]

mesh = interfaceGmsh.Mesh_From_Lines_1D(listPoutres=listePoutre, elemType=elemType)

Affichage.Plot_Model(mesh)
# Affichage.Plot_Maillage(mesh)
plt.show()

beamModel = Materials.BeamModel(dim=beamDim, listePoutres=listePoutre)

materiau = Materials.Materiau(beamModel, verbosity=True)

simu = Simu.Simu(mesh, materiau, verbosity=True)

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


if beamModel.nbPoutres > 1:
    # verfie si il ya pas une poutre libre ?
    simu.add_liaison_Encastrement(mesh.Nodes_Point(point2))
    # simu.add_dirichlet("beam",mesh.Nodes_Point(point2), [0],['y'])
    # simu.add_liaison_Rotule(mesh.Nodes_Point(point2))
        




    
# TODO Rajouter les conditons entre les poutres 
# Faire en sorte de detecter qune poutre est libre ! b 



# simu.add_dirichlet("beam", mesh.Nodes_Point(point2),[-1],["y"])



if problem in ["Flexion"]:
    simu.add_pointLoad("beam", mesh.Nodes_Point(point3), [-charge],["y"])
    # simu.add_lineLoad("beam", mesh.Nodes_Line(line), [-800/L], ['y'])
    # simu.add_surfLoad("beam", mesh.Nodes_Point(point2), [-charge/section.aire],["y"])
elif problem == "Portique":
    simu.add_pointLoad("beam", mesh.Nodes_Point(point3), [-charge],["y"])
    
elif problem == "BiEnca":
    simu.add_pointLoad("beam", mesh.Nodes_Point(point2), [-charge],["y"])
elif problem == "Traction":
    noeudsLine = mesh.Nodes_Line(Line(point1, point3))
    simu.add_lineLoad("beam", noeudsLine, [q],["x"])
    simu.add_pointLoad("beam", mesh.Nodes_Point(point3), [charge],["x"])
    # simu.add_dirichlet("beam", mesh.Nodes_Point(point3), [1], ["x"])

Affichage.Plot_BoundaryConditions(simu)

Kbeam = simu.Assemblage_beam()

beamDisplacement = simu.Solve_beam()

stress = simu.Get_Resultat("Stress")

forces = stress/section.aire

affichage = lambda name, result: print(f"{name} = [{result.min():2.2}; {result.max():2.2}]") if isinstance(result, np.ndarray) else ""

Affichage.Plot_BoundaryConditions(simu)
Affichage.Plot_Result(simu, "u", affichageMaillage=False, deformation=False)
if beamModel.dim > 1:
    Affichage.Plot_Result(simu, "v", affichageMaillage=False, deformation=False)
    Affichage.Plot_Maillage(simu, deformation=True, facteurDef=10)

Affichage.NouvelleSection("Resultats")

print()
u = simu.Get_Resultat("u", valeursAuxNoeuds=True); affichage("u",u)
if beamModel.dim > 1:
    v = simu.Get_Resultat("v", valeursAuxNoeuds=True); affichage("v",v)
    rz = simu.Get_Resultat("rz", valeursAuxNoeuds=True); affichage("rz",rz)

    # fy = simu.Get_Resultat("fy", valeursAuxNoeuds=True)

listX = np.linspace(0,L,100)
if problem == "Flexion":
    v_x = charge/(E*section.Iz) * (listX**3/6 - (L*listX**2)/2)
    flecheanalytique = charge*L**3/(3*E*section.Iz)

    
    erreur  = np.abs(flecheanalytique + v.min())/flecheanalytique
    rapport  = np.abs(flecheanalytique / v.min())
    # print(f"\nerreur = {rapport:.2}")

    fig, ax = plt.subplots()
    ax.plot(listX, v_x, label='Analytique', c='blue')
    ax.scatter(mesh.coordo[:,0], v, label='EF', c='red', marker='x', zorder=2)
    ax.set_title(fr"$v(x)$")
    ax.legend()

    rz_x = charge/E/section.Iz*(listX**2/2 - L*listX)
    rotalytique = -charge*L**2/(2*E*section.Iz)
    # print(np.abs(rotalytique / rz.min()))

    fig, ax = plt.subplots()
    ax.plot(listX, rz_x, label='Analytique', c='blue')
    ax.scatter(mesh.coordo[:,0], rz, label='EF', c='red', marker='x', zorder=2)
    ax.set_title(fr"$r_z(x)$")
    ax.legend()
elif problem == "Traction":
    u_x = (charge*listX/(E*(section.aire))) + (ro*g*listX/2/E*(2*L-listX))
    rapport  = u_x[-1] / u.max()

    fig, ax = plt.subplots()
    ax.plot(listX, u_x, label='Analytique', c='blue')
    ax.scatter(mesh.coordo[:,0], u, label='EF', c='red', marker='x', zorder=2)
    ax.set_title(fr"$u(x)$")
    ax.legend()

# Affichage.Plot_ ElementsMaillage(section, showId=True)


plt.show()