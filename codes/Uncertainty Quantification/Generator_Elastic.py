import Display
import Materials
from Display import plt, np
import Folder

from Materials import TensorProduct as prod
from Materials import Project_Kelvin as projKelvin

import pandas as pd
import mat73
from scipy import stats
from scipy.special import gamma, digamma
from scipy.optimize import minimize

Display.Clear()

folder = Folder.Get_Path(__file__)

# material_str = "Elas_Isot"
material_str = "Elas_IsotTrans"

# import 
dict_ET_GL = mat73.loadmat(Folder.Join(folder, "data_ET_GL.mat"))
dict_EL_NUL = mat73.loadmat(Folder.Join(folder, "data_EL_NUL.mat"))

# récupère les proritétés identifiées
folderPythonEF = Folder.Get_Path()

pathParams = Folder.Join(folderPythonEF,'codes','PhaseField','Identif', "params_article.xlsx")
dfParams = pd.read_excel(pathParams)

if material_str == "Elas_Isot":

    E_exp: np.ndarray = dict_ET_GL['mean_ET_data']*1e-3# [GPa]
    G_exp: np.ndarray = dict_ET_GL['mean_GL_data']*1e-3*13
    n = E_exp.size

    v_exp = E_exp/(2*G_exp)-1
    lambda_exp = E_exp*v_exp/((1+v_exp)*(1-2*v_exp)) 
    c1_exp = lambda_exp + 2/3*G_exp
    c2_exp = G_exp

    m_c1_d = np.mean(c1_exp)
    m_c2_d = np.mean(c2_exp)
    
    material = Materials.Elas_Isot(3, E_exp.mean(),v_exp.mean())
    C = material.C
    C1 = material.get_bulk()
    C2 = material.get_mu()
    Ivect = np.array([1,1,1,0,0,0])    
    Isym = np.eye(6)

    E1 = 1/3 * prod(Ivect, Ivect)
    E2 = Isym - E1

    test_C = np.linalg.norm((3*C1*E1  + 2*C2*E2) - C)/np.linalg.norm(C)
    assert test_C <= 1e-12    

    def J_ml(v):
        l = v[0] # < 1/5
        l1 = v[1] # > 0
        l2 = v[2] # > 0

        J1 = n*(1-l)*np.log(l1) - n*np.log(gamma(1-l)) - l*np.sum(np.log(c1_exp)) - l1*np.sum(c1_exp)
        J2 = n*(1-5*l)*np.log(l2) - n*np.log(gamma(1-5*l)) - 5*l*np.sum(np.log(c2_exp)) - l2*np.sum(c2_exp)

        return -(J1+J2)

    x0 = np.array([0,1/m_c1_d,1/m_c2_d])
    res = minimize(J_ml, x0, bounds=((-np.inf,1/5),(1e-12,np.inf),(1e-12,np.inf)))

    l,l1,l2 = tuple(res.x)

    a1 = 1-l
    b1 = 1/l1

    a2 = 1-5*l
    b2 = 1/l2

    c1_array = np.linspace(1e-12,8,1000)
    c2_array = np.linspace(1e-12,3,1000)
    c1_grid, c2_grid = np.meshgrid(c1_array,c1_array)

    gamma_pdf = lambda c,a,b: stats.gamma.pdf(c, a, scale=b)

    c1_pdf = gamma_pdf(c1_array, a1, b1)
    c2_pdf = gamma_pdf(c2_array, a2, b2)
    c_pdf = np.einsum('i,j->ij',c1_pdf,c2_pdf) # joint pdf

    # joint pdf
    fig = plt.figure()
    ax_join_pdf = fig.add_subplot(projection="3d")
    cc = ax_join_pdf.plot_surface(c1_grid, c2_grid, c_pdf, cmap='jet')
    fig.colorbar(cc, ticks=np.linspace(c_pdf.min(),c_pdf.max(),10))
    ax_join_pdf.set_xlabel('c1 [GPa]')
    ax_join_pdf.set_ylabel('c2 [GPa]')
    ax_join_pdf.set_title('joint pdf')

    # c1 pdf 
    ax_c1 = plt.subplots()[1]
    # ax_c1.plot(c1_array,gamma_pdf(c1_array, a1_exp, b1_exp), label="exp")
    ax_c1.plot(c1_array,c1_pdf, label="num")
    ax_c1.scatter(c1_exp, gamma_pdf(c1_exp, a1, b1), marker='+', c='red', label='samples')
    ax_c1.legend()
    ax_c1.set_xlabel("c1 [GPa]")
    ax_c1.set_title(r"$p_{\bf{C1}}(c1)$")

    # c2 pdf
    ax_c2 = plt.subplots()[1]
    # ax_c2.plot(c2_array,gamma_pdf(c2_array, a2_exp, b2_exp), label="exp")
    ax_c2.plot(c2_array,c2_pdf, label="num")
    ax_c2.scatter(c2_exp, gamma_pdf(c2_exp, a2, b2), marker='+', c='red', label='samples')
    ax_c2.legend()
    ax_c2.set_xlabel("c2 [GPa]")
    ax_c2.set_title(r"$p_{\bf{C2}}(c2)$")

elif material_str == "Elas_IsotTrans":    

    # Zhou
    ET_exp: np.ndarray = dict_ET_GL['mean_ET_data']*1e-3# [GPa]
    n = ET_exp.size
    GL_exp: np.ndarray = dict_ET_GL['mean_GL_data']*1e-3
    EL_exp: np.ndarray = dict_EL_NUL['mean_EL_data']*1e-3
    vL_exp: np.ndarray = dict_EL_NUL['mean_NUL_data']

    # Matthieu
    EL_exp: np.ndarray = dfParams["El"].values * 1e-3
    n = EL_exp.size
    ET_exp: np.ndarray = dfParams["Et"].values * 1e-3
    GL_exp: np.ndarray = dfParams["Gl"].values * 1e-3
    vL_exp = 0.01+0.1*np.random.rand(n)

    vT_exp = 0.1+0.2*np.random.rand(n); # artificial data for vT varying from 0.1 to 0.3
    GT_exp = ET_exp/(2*(1+vT_exp))
    kT_exp = (EL_exp*ET_exp)/(2*(1-vT_exp)*EL_exp-4*ET_exp*vL_exp**2)

    
    c1_d = EL_exp + 4*(vL_exp**2)*kT_exp
    c2_d = 2*kT_exp
    c3_d = 2*np.sqrt(2)*kT_exp*vL_exp
    c4_d = 2*GT_exp
    c5_d = 2*GL_exp

    
    material = Materials.Elas_IsotTrans(3,EL_exp.mean(),ET_exp.mean(),GL_exp.mean(),vL_exp.mean(),vT_exp.mean())
    kt = material.kt
    Gt = material.Gt
    c1 = material.El + 4*material.vl**2*kt
    c2 = 2*kt
    c3 = 2*np.sqrt(2)*kt*material.vl
    c4 = 2*Gt
    c5 = 2*material.Gl

    n = material.axis_l

    p = prod(n,n)
    q = np.eye(3) - p
    
    E1 = projKelvin(prod(p,p))
    E2 = projKelvin(1/2 * prod(q,q))
    E3 = projKelvin(1/np.sqrt(2) * prod(p,q))
    E4 = projKelvin(1/np.sqrt(2) * prod(q,p))
    E5 = projKelvin(prod(q,q,True) - 1/2*prod(q,q))
    I = projKelvin(prod(np.eye(3),np.eye(3),True))
    E6 = I - E1 - E2 - E5

    diff_C = material.C - (c1*E1 + c2*E2 + c3*(E3+E4) + c4*E5 + c5*E6)
    test_C = np.linalg.norm(diff_C)/np.linalg.norm(material.C)

    assert test_C <= 1e-12    

    # ----------------------------------------------
    # Metropolis hastings
    # ----------------------------------------------
    m_c1_d = np.mean(c1_d)
    m_c2_d = np.mean(c2_d)
    m_c3_d = np.mean(c3_d)
    m_c4_d = np.mean(c4_d)
    m_c5_d = np.mean(c5_d)
    m_c_d = np.array([m_c1_d, m_c2_d, m_c3_d, m_c4_d, m_c5_d])

    phiC_d = np.log((c1_d*c2_d - c3_d**2) * c4_d**2 * c5_d**2)
    vC_d = np.mean(phiC_d)

    c_0 = np.array([m_c1_d, m_c2_d, m_c3_d])
    cov = lambda l: 1/l * np.array([[-m_c1_d**2, -m_c3_d**2, -m_c1_d*m_c3_d],
                              [-m_c3_d**2, -m_c2_d**2, -m_c2_d*m_c3_d],
                              [-m_c1_d*m_c3_d, -m_c2_d*m_c3_d, -(m_c3_d**2 + m_c1_d*m_c2_d)/2]])

    # _d for data
    def p_C(c: np.ndarray, lamb: np.ndarray):
        """Function to samples in"""

        c1, c2, c3 = tuple(c[:3])
        l, l1, l2, l3 = tuple(lamb[:4])

        try:
            p = (c1*c2-c3**2)**-l * np.exp(-l1*c1 -l2*c2 -l3*c3)
            error = False
        except:
            error = True

        if error:
            pass

        return p

    def Metropolis_Hastings_C(c_0: np.ndarray, cov: np.ndarray, lamb: np.ndarray, burn_in: int, nSamples: int):
        # in this function we use a normal distribution as our guess function

        # pdf multivariate normal with mean t0 and covariance cov evaluated in t1    
        q = lambda c_t1, c_t0: stats.multivariate_normal.pdf(c_t1, c_t0, cov)

        c_t = c_0
        samples = []
        for i in range(burn_in+nSamples):
            c_tp1 = stats.multivariate_normal.rvs(c_t, cov)

            accept_prob = p_C(c_tp1, lamb)/p_C(c_t, lamb)
            # accept_prob = (p_C(c_tp1, lamb) * q(c_t, c_tp1))/(p_C(c_t, lamb) * q(c_tp1, c_t))
            # test = q(c_t, c_tp1)/q(c_tp1, c_t) # = 1 if the guess function is symmetric
            # assert test == 1
            # print(test)
            # then accept_prob = pZ(z_tp1)/pX(z_t)

            isAccepted = np.random.uniform(0, 1) < accept_prob
            
            if isAccepted:
                c_t = c_tp1

            if isAccepted and i >= burn_in:
                samples.append(c_t)

        assert len(samples) > 0

        rejectRatio: float = 1 - len(samples)/nSamples

        return np.array(samples), rejectRatio

    def J_ls(lamb:np.ndarray):

        l = lamb[0]
        l1 = lamb[1]
        l2 = lamb[2]
        l3 = lamb[3]
        l4 = lamb[4]
        l5 = lamb[5]

        test_lamb = 2*np.sqrt(l1*l2) - l3
        # print(test_lamb)
        assert test_lamb >= 1e-12

        a = 1-2*l
        b4 = 1/l4
        b5 = 1/l5

        # X ~ Gamma(a,b)

        # E[X] = a b for X ~ Gamma(a,b)
        # _s for samples
        m_c4_s: float = a*b4
        m_c5_s: float = a*b5

        # Phi(c) = ln((c1 c2 - c3^2) c4^2 c5^2)
        #        = ln(c1 c2 - c3^2) + 2*ln(c4) + 2*ln(c5)
        #        = PhiC123 + PhiC4 + PhiC5

        # E[ln(X)] = ψ(a) + ln(b)
        # ψ(k) is the digamma function
        mPhi_c4_s: float = 2 * (digamma(a) + np.log(b4))
        mPhi_c5_s: float = 2 * (digamma(a) + np.log(b5))

        # here we want to get the c1, c2, c3 samples
        # we use Metropolis-Hastings algo to make samples        
        
        samples, rejectRatio = Metropolis_Hastings_C(c_0, cov(l), lamb, 0, 10000)
        # _s for samples
        c1_s: np.ndarray = samples[:,0]; m_c1_s: float = np.mean(c1_s)
        c2_s: np.ndarray = samples[:,1]; m_c2_s: float = np.mean(c2_s)
        c3_s: np.ndarray = samples[:,2]; m_c3_s: float = np.mean(c3_s)

        m_c_s = np.array([m_c1_s, m_c2_s, m_c3_s, m_c4_s, m_c5_s])

        mPhi_c123_s: float = np.mean(np.log(c1_s*c2_s - c3_s**2))


        vC_s = mPhi_c123_s + mPhi_c4_s + mPhi_c5_s

        # J = 0.5 * np.linalg.norm(m_c_d - m_c_s)**2 + 0.5 * (vC_d-vC_s)**2
        
        sol_s = np.append(m_c_s,[vC_s])
        sol_d = np.append(m_c_d,[vC_d])
        J = np.linalg.norm(sol_d-sol_s)**2/np.linalg.norm(sol_d)**2

        print(f"{J:.3e}")

        return J

    l0 = -100
    cst = l0/(m_c1_d*m_c2_d - m_c3_d**2)
    lamd0 = np.array([l0,
                      -m_c2_d*cst,
                      -m_c1_d*cst,
                      2*m_c3_d*cst,
                      -2*l0/m_c4_d,
                      -2*l0/m_c5_d])
    
    bnds = ((-np.inf, 1/2),
            (1e-12, np.inf),
            (1e-12, np.inf),
            (1e-12, np.inf),
            (1e-12, np.inf),
            (1e-12, np.inf))
    
    cons = ({'type': 'ineq', 'fun': lambda l:  2*np.sqrt(l[1]*l[2])-l[3]-1e-12})
    
    methods = ["COBYLA", "SLSQP", "trust-constr"]
    # fonction que pour COBYLA

    res = minimize(J_ls, lamd0, bounds=bnds, constraints=cons, method=methods[0])
    lamb = res.x

    print()
    [print(f"l{i} = {l:.3f}") for i, l in enumerate(lamb)]

    # ax = 
    c1_array = np.linspace(0, c1_d.max()*1.2, 1000)
    c2_array = np.linspace(0, c2_d.max()*1.2, 1000)
    
    c1_grid, c2_grid = np.meshgrid(c1_array, c2_array)

    c3_grid = np.sqrt(c1_grid*c2_grid)

    # joint pdf
    fig = plt.figure()
    ax_join_pdf = fig.add_subplot(projection="3d")
    ax_join_pdf.set_xlabel('c1 [GPa]')
    ax_join_pdf.set_ylabel('c2 [GPa]')
    ax_join_pdf.set_zlabel('c3 [GPa]')
    ax_join_pdf.set_title('joint pdf')    
    ax_join_pdf.view_init(azim=45)
    # suport
    ax_join_pdf.plot_surface(c1_grid, c2_grid, c3_grid, color='gray', alpha=0.5)
    ax_join_pdf.plot_surface(c1_grid, c2_grid, -c3_grid, color='gray', alpha=0.5)
    # data
    ax_join_pdf.scatter(c1_d,c2_d,c3_d,zorder=3, c='red', marker='+')
    # samples
    samples, rejectRatio = Metropolis_Hastings_C(c_0, cov(lamb[0]), lamb, 100, 1000)
    ax_join_pdf.scatter(samples[:,0],samples[:,1],samples[:,2],zorder=3, c='blue', marker='.')

    plt.show()

else:
    raise Exception("Not implemented")




pass