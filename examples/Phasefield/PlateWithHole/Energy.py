"""Represents the influence of the poisson's ratio to illustrate that cracks can appear in a compression zone."""

from EasyFEA import (Display, Tic, plt, np, pd,
                     Mesher, ElemType,
                     Materials, Simulations)
from EasyFEA.Geoms import Point, Domain, Line, Circle

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    plotAllResult = False

    # material
    E=12e9
    v=0.2
    planeStress = False

    # phase field
    comp = "Elas_Isot" # "Elas_Isot" "Elas_IsotTrans"
    split = "Miehe" # ["Bourdin","Amor","Miehe","Stress","AnisotMiehe","AnisotStress"]
    regu = "AT2"
    gc = 1.4

    name="_".join([comp, split, regu])

    # Geom
    L=15e-3
    h=30e-3
    ep=1e-3
    diam=6e-3
    r=diam/2
    l0 = 0.12e-3*2

    # loading
    SIG = 10 #Pa

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    clC = l0/2
    clD = l0

    point = Point()
    domain = Domain(point, Point(x=L, y=h), clD)
    circle = Circle(Point(x=L/2, y=h/2), diam, clC)

    mesher = Mesher(openGmsh=False)
    mesh = mesher.Mesh_2D(domain, [circle], ElemType.TRI3)

    # Nodes
    B_lower = Line(point,Point(x=L))
    B_upper = Line(Point(y=h),Point(x=L, y=h))
    nodesLower = mesh.Nodes_Line(B_lower)
    nodesUpper = mesh.Nodes_Line(B_upper)
    node00 = mesh.Nodes_Point(Point())
    nodesCircle = mesh.Nodes_Circle(circle)
    nodesCircle = nodesCircle[np.where(mesh.coord[nodesCircle,1]<= circle.center.y)]

    # Nodes in A and B
    pA = Point(x=L/2, y=h/2+r)
    pB = Point(x=L/2+r, y=h/2)
    nodeA = mesh.Nodes_Point(pA)
    nodeB = mesh.Nodes_Point(pB)

    if plotAllResult:
        ax = Display.Plot_Mesh(mesh)
        for ns in [nodesLower, nodesUpper, node00, nodeA, nodeB]:
            Display.Plot_Nodes(mesh, ax=ax, nodes=ns,c='red')

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    columns = ['v','A (an CP)','B (an CP)',
                'A (CP)','errA (CP)','B (CP)','errB (CP)',
                'A (DP)','errA (DP)','B (DP)','errB (DP)']

    df = pd.DataFrame(columns=columns)

    list_V = [0.2,0.3,0.4]

    Miehe_psiP_A = lambda v: 1**2*(v*(1-2*v)+1)/(2*(1+v))
    Miehe_psiP_B = lambda v: 3**2*v**2/(1+v)

    for v in list_V:
        result = {
            'v': v
        }
        for isCP in [False,True]:        
            material = Materials.Elas_Isot(2, E=E, v=v, planeStress=isCP, thickness=ep)
            phaseFieldModel = Materials.PhaseField(material, split, regu, gc, l0)

            simu = Simulations.PhaseFieldSimu(mesh, phaseFieldModel, verbosity=False)

            simu.add_dirichlet(nodesLower, [0], ["y"])
            simu.add_dirichlet(node00, [0], ["x"])
            simu.add_surfLoad(nodesUpper, [-SIG], ["y"])            

            simu.Solve()

            psipa = np.mean(simu.Result("psiP", True)[nodeA])*E/SIG**2
            psipb = np.mean(simu.Result("psiP", True)[nodeB])*E/SIG**2

            ext = "CP" if isCP else "DP"

            result[f'A ({ext})'] = psipa
            result[f'B ({ext})'] = psipb

            result[f'errA ({ext})'] = np.abs(psipa-Miehe_psiP_A(v))/Miehe_psiP_A(v)
            result[f'errB ({ext})'] = np.abs(psipb-Miehe_psiP_B(v))/Miehe_psiP_B(v)

            Display.Plot_Result(simu, "psiP", nodeValues=True, coef=E/SIG**2, title=fr"$\psi_{0}^+\ E / \sigma^2 \ \nu={v} \ {ext}$", filename=f"psiP {name} v={v}", colorbarIsClose=True)    

        result['A (an CP)'] = Miehe_psiP_A(v)
        result['B (an CP)'] = Miehe_psiP_B(v)
        
        new = pd.DataFrame(result, index=[0])

        df = pd.concat([df, new], ignore_index=True)

    ax = Display.init_Axes()
    ax = domain.Plot_Geoms([domain, circle], ax=ax, plotPoints=False, color='k')
    ax.plot(*pA.coord[:2], label='pA', ls='', marker='s', lw='10')
    ax.plot(*pB.coord[:2], label='pB', ls='', marker='s', lw='10')
    ax.legend()

    # ----------------------------------------------
    # Plot
    # ----------------------------------------------
    print(name+'\n')
    print(df)
    # Stress in A
    SxxA = simu.Result("Sxx", True)[nodeA][0]
    SyyA = simu.Result("Syy", True)[nodeA][0]
    SxyA = simu.Result("Sxy", True)[nodeA][0]

    Sig_A=np.array([[SxxA, SxyA, 0],[SxyA, SyyA, 0],[0,0,0]])
    print(f"\nEn A : Sig/SIG = \n{Sig_A/SIG}\n")

    SxxB = simu.Result("Sxx", True)[nodeB][0]
    SyyB = simu.Result("Syy", True)[nodeB][0]
    SxyB = simu.Result("Sxy", True)[nodeB][0]

    Sig_B=np.array([[SxxB, SxyB, 0],[SxyB, SyyB, 0],[0,0,0]])
    print(f"\nEn B : Sig/SIG = \n{Sig_B/SIG}\n")

    if plotAllResult:
        Display.Plot_Result(simu, "Sxx", nodeValues=True, coef=1/SIG, title=r"$\sigma_{xx} / \sigma$", filename='Sxx', colorbarIsClose=True)
        Display.Plot_Result(simu, "Syy", nodeValues=True, coef=1/SIG, title=r"$\sigma_{yy} / \sigma$", filename='Syy', colorbarIsClose=True)
        Display.Plot_Result(simu, "Sxy", nodeValues=True, coef=1/SIG, title=r"$\sigma_{xy} / \sigma$", filename='Sxy', colorbarIsClose=True)

    axp = Display.init_Axes()

    list_v = np.arange(0, 0.5,0.0005)

    # test = (vv*(1-2*vv)+1)/(2*(1+vv))

    # axp.plot(list_v, (list_v*(1-2*list_v)+1)/(2*(1+list_v)), label="psiP_A*E/Sig^2")
    axp.plot(list_v, Miehe_psiP_A(list_v), label='A')
    axp.plot(list_v, Miehe_psiP_B(list_v), label='B')
    axp.grid()
    axp.legend(fontsize=14)
    axp.set_xlabel("$\nu$",fontsize=14)
    axp.set_ylabel("$\psi_{0}^+\ E / \sigma^2$",fontsize=14)
    axp.set_title(r'Split sur $\varepsilon$ an',fontsize=14)

    list_Amor_psiP_A=[]
    list_Amor_psiP_B=[]

    list_Miehe_psiP_A=[]
    list_Miehe_psiP_B=[]

    list_Stress_psiP_A=[]
    list_Stress_psiP_B=[]

    # calc num part
    for v in list_v:
        
        Eps_A = (1+v)/E*Sig_A - v/E*np.trace(Sig_A)*np.eye(3); trEps_A = np.trace(Eps_A); trEpsP_A = (trEps_A+np.abs(trEps_A))/2
        Eps_B = (1+v)/E*Sig_B - v/E*np.trace(Sig_B)*np.eye(3); trEps_B = np.trace(Eps_B); trEpsP_B = (trEps_B+np.abs(trEps_B))/2

        # Eps_A = (1+v)/E*Sig_A - v/E*np.trace(Sig_A)*np.eye(3); trEps_A = np.trace(Eps_A); trEpsP_A = (trEps_A+np.abs(trEps_A))/2
        # Eps_B = (1+v)/E*Sig_B - v/E*np.trace(Sig_B)*np.eye(3); trEps_B = np.trace(Eps_B); trEpsP_B = (trEps_B+np.abs(trEps_B))/2

        l = v*E/((1+v)*(1-2*v))
        # l = v*E/(1-v**2)
        mu=E/(2*(1+v))

        # Split Amor
        bulk = l + 2/3*mu

        spherA = 1/3*trEps_A*np.eye(3)
        spherB = 1/3*trEps_B*np.eye(3)

        EpsD_A = Eps_A-spherA
        EpsD_B = Eps_B-spherB

        Amor_psi_A = 1/2*bulk*trEpsP_A**2 + mu * np.einsum('ij,ij', EpsD_A, EpsD_A)
        Amor_psi_B = 1/2*bulk*trEpsP_B**2 + mu * np.einsum('ij,ij', EpsD_B, EpsD_B)

        list_Amor_psiP_A.append(Amor_psi_A)
        list_Amor_psiP_B.append(Amor_psi_B)

        # Split Miehe
        Epsi_A = np.diag(np.linalg.eigvals(Eps_A)); Epsip_A = (Epsi_A+np.abs(Epsi_A))/2
        Epsi_B = np.diag(np.linalg.eigvals(Eps_B)); Epsip_B = (Epsi_B+np.abs(Epsi_B))/2

        Miehe_psiP_A = l/2*trEpsP_A**2 + mu*np.einsum('ij,ij',Epsip_A,Epsip_A)
        Miehe_psiP_B = l/2*trEpsP_B**2 + mu*np.einsum('ij,ij',Epsip_B,Epsip_B)

        list_Miehe_psiP_A.append(Miehe_psiP_A)
        list_Miehe_psiP_B.append(Miehe_psiP_B)

        # Split Stress
        Sigi_A = np.diag(np.linalg.eigvals(Sig_A)); Sigip_A = (Sigi_A+np.abs(Sigi_A))/2
        Sigi_B = np.diag(np.linalg.eigvals(Sig_B)); Sigip_B = (Sigi_B+np.abs(Sigi_B))/2
        
        trSig_A = np.trace(Sig_A); trSigP_A = (trSig_A+np.abs(trSig_A))/2
        trSig_B = np.trace(Sig_B); trSigP_B = (trSig_B+np.abs(trSig_B))/2

        Stress_psiP_A = ((1+v)/E*np.einsum('ij,ij',Sigip_A,Sigip_A) - v/E * trSigP_A**2)/2
        Miehe_psiP_B = ((1+v)/E*np.einsum('ij,ij',Sigip_B,Sigip_B) - v/E * trSigP_B**2)/2

        list_Stress_psiP_A.append(Stress_psiP_A)
        list_Stress_psiP_B.append(Miehe_psiP_B)

    ax1 = Display.init_Axes()

    ax1.plot(list_v, np.array(list_Miehe_psiP_A)*E/SIG**2, label='A')
    ax1.plot(list_v, np.array(list_Miehe_psiP_B)*E/SIG**2, label='B')
    ax1.grid()
    if split == "Miehe":
        ax1.scatter(list_V, np.array(df['A (CP)'].tolist()),label='num A')
        ax1.scatter(list_V, np.array(df['B (CP)'].tolist()),label='num B')
    ax1.legend(fontsize=14)
    ax1.set_xlabel(r"$\nu$",fontsize=14)
    ax1.set_ylabel("$\psi_{0}^+\ E / \sigma^2$",fontsize=14)
    ax1.set_title(r'Split sur $\varepsilon$ num',fontsize=14)

    ax2 = Display.init_Axes()

    stressA = lambda v: 1/E*(SxxA**2+SyyA**2-2*SxxA*SyyA*v)

    ax2.plot(list_v, np.array(list_Stress_psiP_A)*E/SIG**2, label='A')
    ax2.plot(list_v, np.array(list_Stress_psiP_B)*E/SIG**2, label='B')
    # ax2.plot(list_v, np.ones(list_v.shape), label='A')
    # ax2.plot(list_v, stressA(list_v)*E/SIG**2, label='AA')
    ax2.grid()
    if split == "Stress":    
        ax2.scatter(list_V, np.array(df['A (CP)'].tolist()),label='num A')
        ax2.scatter(list_V, np.array(df['B (CP)'].tolist()),label='num B')
    ax2.legend(fontsize=14)
    ax2.set_xlabel(r"$\nu$",fontsize=14)
    ax2.set_ylabel("$\psi_{0}^+\ E / \sigma^2$",fontsize=14)
    ax2.set_title('Split sur $\sigma$ num',fontsize=14)

    ax3 = Display.init_Axes()

    ax3.plot(list_v, np.array(list_Amor_psiP_A)*E/SIG**2, label='A')
    ax3.plot(list_v, np.array(list_Amor_psiP_B)*E/SIG**2, label='B')
    ax3.grid()
    if split == "Amor":    
        ax3.scatter(list_V, np.array(df['A (CP)'].tolist()),label='num A')
        ax3.scatter(list_V, np.array(df['B (CP)'].tolist()),label='num B')
    ax3.legend(fontsize=14)
    ax3.set_xlabel(r"$\nu$",fontsize=14)
    ax3.set_ylabel("$\psi_{0}^+\ E / \sigma^2$",fontsize=14)
    ax3.set_title('Split Amor num',fontsize=14)

    Tic.Resume()

    plt.show()