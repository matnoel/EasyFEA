"""Performs a damage simulation on a wooden sample."""

# WARNING : This code is not validated

from EasyFEA import (Display, Folder, plt, np, pd,
                     Mesher, ElemType,
                     Materials, Simulations,
                     Paraview_Interface)
from EasyFEA.Geoms import Point, Points, Circle, Line

if __name__ == '__main__':

    Display.Clear()

    folder = Folder.New_File("TractionWood", results=True)

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    L = 105 # mm
    H = 70
    h = H/2
    r3 = 3
    a = 20
    c = 20
    d1 = 25.68
    d2 = 40.85

    # init crack
    useSmallCrack = True
    crackThickness = 0.5
    crackLength = 40

    alpha1 = 2.5 * np.pi/180
    alpha2 = 20 * np.pi/180
    alpha3 = 34 * np.pi/180

    betha = (np.pi - alpha3 - (np.pi/2-alpha1))/2
    d = r3/np.tan(betha)

    l0 = H/100
    clC = l0/2
    clD = l0*2

    makeParaview = True

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    p0 = Point(x=0, y=-crackThickness/2)
    p1 = Point(x=0, y=-h)
    p2 = Point(x=a, y=-h)
    p3 = Point(x=a+(d1+d)*np.sin(alpha1), y=-h+(d1+d)*np.cos(alpha1), r=r3)
    p4 = Point(x=L-c-d2*np.cos(alpha2), y=-h+d2*np.sin(alpha2))
    p5 = Point(x=L-c, y=-h)
    p6 = Point(x=L, y=-h)
    p7 = Point(x=L, y=h)
    p8 = Point(x=L-c, y=h)
    p9 = Point(x=L-c-d2*np.cos(alpha2), y=h-d2*np.sin(alpha2))
    p10 = Point(x=a+(d1+d)*np.sin(alpha1), y=h-(d1+d)*np.cos(alpha1), r=r3)
    p11 = Point(x=a, y=h)
    p12 = Point(x=0, y=h)
    p13 = Point(x=0, y=crackThickness/2)
    p14 = Point(x=crackLength, y=crackThickness/2, r=crackThickness/2)
    p15 = Point(x=crackLength, y=-crackThickness/2, r=crackThickness/2)

    if useSmallCrack:
        p0 = Point(isOpen=True)
        points = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]
        crack = Line(p0, Point(x=crackLength), clC, isOpen=True)
        cracks = [crack]
    else:
        points = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15]
        cracks = []
    points = Points(points, clD)

    diam = 5
    r = diam/2
    c1 = Circle(Point(a/2, -h+7.5), diam, clC, isHollow=True)
    c2 = Circle(Point(L-c/2, -h+7.5), diam, clC, isHollow=True)
    c3 = Circle(Point(L-c/2, h-7.5), diam, clC, isHollow=True)
    c4 = Circle(Point(a/2, h-7.5), diam, clC, isHollow=True)

    inclusions = [c1, c2, c3, c4]

    zone = 5
    # refineDomain = Domain(Point(crackLength-zone, -zone), Point(L, zone), meshSize=clC)
    refineDomain = Circle(Point(crackLength), 20, clC)
    mesh = Mesher().Mesh_2D(points, inclusions, ElemType.TRI3, cracks, refineGeoms=[refineDomain])

    Display.Plot_Tags(mesh)
    Display.Plot_Mesh(mesh)

    # ----------------------------------------------
    # Material
    # ----------------------------------------------
    # El=11580*1e6
    Gc = 0.07 # mJ/mm2
    El=12000 # MPa
    Et=500
    # Et=50
    Gl=450
    vl=0.02
    vt=0.44
    v=0
    material = Materials.Elas_IsotTrans(2, El=El, Et=Et, Gl=Gl, vl=vl, vt=vt,
                                        planeStress=True, thickness=12.5,
                                        axis_l=np.array([1,0,0]), axis_t=np.array([0,1,0]))

    a1 = np.array([0,1])
    M1 = np.einsum("i,j->ij", a1, a1)
    Betha = El/Et
    Betha = 50
    A = np.eye(2) + Betha * (np.eye(2) - M1)

    pfm = Materials.PhaseField(material, "AnisotStress", "AT1", Gc, l0, A=A)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    simu = Simulations.PhaseFieldSimu(mesh, pfm, verbosity=False)

    nodes_upper = mesh.Nodes_Circle(c4)
    nodes_upper = nodes_upper[np.where(mesh.coordGlob[nodes_upper,1]>=c4.center.y)] 

    nodes_lower = mesh.Nodes_Circle(c1)
    nodes_lower = nodes_lower[np.where(mesh.coordGlob[nodes_lower,1]<=c1.center.y)]

    noeudPoint = mesh.Nodes_Point(c1.center - [0, diam/2])

    # if len(cracks) > 0: Display.Plot_Nodes(mesh, mesh.Nodes_Line(cracks[0]), showId=True)

    if useSmallCrack:
        nodes_edges = mesh.Nodes_Tags(mesh.Get_list_groupElem(1)[0].nodeTags)
        nodes_crack = mesh.Nodes_Line(crack)
        nodes_edges = list(set(nodes_edges) - set(nodes_crack))
    else:
        nodes_edges = mesh.Nodes_Tags([f"L{i}" for i in range(15)])

    # Display.Plot_Nodes(mesh, noeudsBord)

    def Loading(force: float):
        simu.Bc_Init()

        SIG = force/(np.pi*r**2/2)
        
        simu.add_dirichlet(noeudPoint, [0], ["x"])
        simu.add_dirichlet(nodes_lower, [0], ["y"])
        simu.add_surfLoad(nodes_upper, [lambda x,y,z: SIG*(y-c4.center.y)/r * np.abs((y-c4.center.y)/r)], ["y"])

        # SIG *= 1/2
        # simu.add_surfLoad(noeudsHaut, [lambda x,y,z: SIG*(y-c4.center.y)/r * np.abs((y-c4.center.y)/r)], ["y"])
        # simu.add_surfLoad(noeudsBas, [lambda x,y,z: -SIG*(y-c4.center.y)/r * np.abs((y-c4.center.y)/r)], ["y"])

    Loading(0)

    Display.Plot_BoundaryConditions(simu)
    # plt.show()

    ax_Damage = Display.Plot_Result(simu, "damage")
    nf = 100
    forces = np.linspace(0, 35, nf)
    for iter, force in enumerate(forces):

        Loading(force)

        # simu.Solve(1e-1, maxIter=50, convOption=1)
        simu.Solve(1e-0, maxIter=50, convOption=2)

        simu.Save_Iter()

        depNum = np.max(simu.displacement[nodes_upper])

        # ecart = np.abs(depNum-dep)/dep
        # print(ecart)

        # Display.Plot_Result(simu, "Syy")
        # plt.show()

        pourcent = iter/nf

        simu.Results_Set_Iteration_Summary(iter, force, "N", pourcent, True)
        
        Display.Plot_Result(simu, "damage", ax=ax_Damage)
        plt.pause(1e-12)

        if np.max(simu.damage[nodes_edges]) >= 0.95:
            break
    if makeParaview:
        Paraview_Interface.Make_Paraview(simu, folder)

    plt.show()