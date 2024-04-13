"""The aim of this script is to study the impact of changing the problem size on the stress field around the hole."""
# TODO: Varying the size and position of the hole in the field could be of interest. 

from EasyFEA import (Display, Tic, plt, np,
                     Mesher, ElemType,
                     Materials, Simulations)
from EasyFEA.Geoms import Point, Domain, Line, Circle

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    E=12e9
    v=0.2
    planeStress = True

    # phase field
    comp = "Elas_Isot"
    split = "Miehe" # ["Bourdin","Amor","Miehe","Stress"]
    regu = "AT1" # "AT1", "AT2"
    gc = 1.4

    # geom
    unit = 1e-3
    L=15*unit
    H=30*unit
    ep=1*unit
    diam=6*unit
    r=diam/2
    l0 = 0.12 *unit

    # loading
    SIG = 10 #Pa

    # meshSize
    clD = l0*5 # domain
    clC = l0 # near crack

    list_SxxA = []
    list_SyyA = []
    list_SxyA = []
    list_SxxB = []
    list_SyyB = []
    list_SxyB = []

    param1 = H
    param2 = L
    param3 = diam
    
    # list_coef = np.linspace(1/2,3,10)
    list_coef = np.linspace(1/4,L/diam*0.8,10)

    for cc in list_coef:

        # H = param1 * cc
        # L = param2 * cc
        diam = param3 * cc

        if diam > L or diam > H: continue

        print(cc)

        point = Point()
        domain = Domain(point, Point(x=L, y=H), clD)
        circle = Circle(Point(x=L/2, y=H-H/2), diam, clC)
        refine = Circle(circle.center, diam*1.2, clC)

        mesher = Mesher(openGmsh=False, verbosity=False)
        mesh = mesher.Mesh_2D(domain, [circle], ElemType.TRI3, refineGeoms=[refine])

        # Display.Plot_Mesh(mesh)

        # Gets nodes
        B_lower = Line(point,Point(x=L))
        B_upper = Line(Point(y=H),Point(x=L, y=H))
        nodes0 = mesh.Nodes_Line(B_lower)
        nodesh = mesh.Nodes_Line(B_upper)
        node00 = mesh.Nodes_Point(Point())

        # Nodes in A and B
        pA = Point(x=L/2, y=H-H/2+diam/2)
        pB = Point(x=L/2+diam/2, y=H-H/2)
        nodeA = mesh.Nodes_Point(pA)
        nodeB = mesh.Nodes_Point(pB)

        # materials
        material = Materials.Elas_Isot(2, E=E, v=v, planeStress=True, thickness=ep)
        phasefield = Materials.PhaseField(material, split, regu, gc, l0)

        simu = Simulations.PhaseFieldSimu(mesh, phasefield, verbosity=False)

        simu.add_dirichlet(nodes0, [0], ["y"])
        simu.add_dirichlet(node00, [0], ["x"])
        simu.add_surfLoad(nodesh, [-SIG], ["y"])

        simu.Solve()

        # Stress in A
        list_SxxA.append(simu.Result("Sxx", True)[nodeA])
        list_SyyA.append(simu.Result("Syy", True)[nodeA])
        list_SxyA.append(simu.Result("Sxy", True)[nodeA])
        # Stress in B
        list_SxxB.append(simu.Result("Sxx", True)[nodeB])
        list_SyyB.append(simu.Result("Syy", True)[nodeB])
        list_SxyB.append(simu.Result("Sxy", True)[nodeB])

    ax = Display.init_Axes()
    ax = domain.Plot_Geoms([domain, circle], ax=ax, plotPoints=False, color='k')
    ax.plot(*pA.coord[:2], label='pA', ls='', marker='s', lw='10')
    ax.plot(*pB.coord[:2], label='pB', ls='', marker='s', lw='10')
    ax.legend()

    # ----------------------------------------------
    # Plot
    # ----------------------------------------------
    paramName=''
    if param1/H != 1: paramName += "H "
    if param2/L != 1: paramName += "L "
    if param3/diam != 1: paramName += "diam"

    Display.Plot_Mesh(mesh, title=f"mesh_{paramName}")
    Display.Plot_Result(simu, "Sxx", nodeValues=True, coef=1/SIG, title=r"$\sigma_{xx}/\sigma$", filename='Sxx')
    Display.Plot_Result(simu, "Syy", nodeValues=True, coef=1/SIG, title=r"$\sigma_{yy}/\sigma$", filename='Syy')
    Display.Plot_Result(simu, "Sxy", nodeValues=True, coef=1/SIG, title=r"$\sigma_{xy}/\sigma$", filename='Sxy')

    ax = Display.init_Axes()

    list_coef = [list_coef[i] for i in range(len(list_SxxA))]

    ax.plot(list_coef, np.array(list_SxxA)/SIG,label='SxxA/SIG')
    ax.plot(list_coef, np.array(list_SxyA)/SIG,label='SxyA/SIG')
    ax.plot(list_coef, np.array(list_SyyA)/SIG,label='SyyA/SIG')
    ax.plot(list_coef, np.array(list_SxxB)/SIG,label='SxxB/SIG')
    ax.plot(list_coef, np.array(list_SxyB)/SIG,label='SxyB/SIG')
    ax.plot(list_coef, np.array(list_SyyB)/SIG,label='SyyB/SIG')
    ax.grid()
    plt.legend()
    ax.set_title(paramName)
    ax.set_xlabel('coef')

    Tic.Resume()

    plt.show()