import Display
import Materials
from Display import plt, np
import Folder

from Materials import Tensor_Product as tensProd
from Materials import Project_Kelvin as projKelvin

import pandas as pd
import pickle
import mat73
from scipy import stats
from scipy.special import gamma, digamma
from scipy.optimize import minimize

Display.Clear()

folder = Folder.Get_Path(__file__)

if __name__ == '__main__':

    # _d for data
    # _s for samples

    # material_str = "Elas_Isot"
    material_str = "Elas_IsotTrans"

    # import datas
    dict_ET_GL = mat73.loadmat(Folder.Join(folder, "data_ET_GL.mat"))
    dict_EL_NUL = mat73.loadmat(Folder.Join(folder, "data_EL_NUL.mat"))

    # récupère les proritétés identifiées
    import sys
    sys.path.append(Folder.Join(Folder.Get_Path(), 'codes'))
    from FCBA.Functions import dfParams

    if material_str == "Elas_Isot":

        # --------------------------------------------------------------------------------------------
        # Elas_Isot
        # --------------------------------------------------------------------------------------------

        E_exp: np.ndarray = dict_ET_GL['mean_ET_data']*1e-3# [GPa]
        G_exp: np.ndarray = dict_ET_GL['mean_GL_data']*1e-3*13
        v_exp = E_exp/(2*G_exp)-1

        n = E_exp.size
        
        mat = Materials.Elas_Isot(3, E_exp, v_exp)        
        c1_exp, c2_exp = mat.Walpole_Decomposition()[0]

        m_c1_d = np.mean(c1_exp)
        m_c2_d = np.mean(c2_exp)

        # --------------------------------------------------------------------------------------------
        # Maximum likehood
        # --------------------------------------------------------------------------------------------
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

        print(f'\nl1 = {l1:.4f}\t err_l1 = {np.abs((3.4910-l1)/l1):.2e}')
        print(f'l2 = {l2:.4f}\t err_l2 = {np.abs((45.7191-l2)/l2):.2e}')
        print(f'l = {l:.4f}\t err_l = {np.abs((-6.1280-l)/l):.2e}')

        print(f'\na1 = {a1:.4f}')
        print(f'b1 = {b1:.4f}')
        print(f'\na2 = {a2:.4f}')
        print(f'b2 = {b2:.4f}')

        # --------------------------------------------------------------------------------------------
        # Plot
        # --------------------------------------------------------------------------------------------
        mat = Materials.Elas_Isot(3, E_exp, v_exp)

        c1Max = mat.get_bulk().max()*1.1
        c2Max = mat.get_mu().max()*1.5

        c1_array = np.linspace(1e-12,c1Max,1000)
        c2_array = np.linspace(1e-12,c2Max,1000)
        c1_grid, c2_grid = np.meshgrid(c1_array,c1_array)

        pdfFunc = lambda c,a,b: stats.gamma.pdf(c, a, scale=b)

        c1_pdf = pdfFunc(c1_array, a1, b1)
        c2_pdf = pdfFunc(c2_array, a2, b2)
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
        ax_c1 = Display.init_Axes(2)
        # ax_c1.plot(c1_array,gamma_pdf(c1_array, a1_exp, b1_exp), label="exp")
        ax_c1.plot(c1_array,c1_pdf, label="num")
        ax_c1.scatter(c1_exp, pdfFunc(c1_exp, a1, b1), marker='+', c='red', label='samples',zorder=3)
        ax_c1.legend()
        ax_c1.set_xlabel("c1 [GPa]")
        ax_c1.set_title(r"$p_{\bf{C1}}(c1)$")

        # c2 pdf
        ax_c2 = Display.init_Axes(2)
        # ax_c2.plot(c2_array,gamma_pdf(c2_array, a2_exp, b2_exp), label="exp")
        ax_c2.plot(c2_array,c2_pdf, label="num")
        ax_c2.scatter(c2_exp, pdfFunc(c2_exp, a2, b2), marker='+', c='red', label='samples',zorder=3)
        ax_c2.legend()
        ax_c2.set_xlabel("c2 [GPa]")
        ax_c2.set_title(r"$p_{\bf{C2}}(c2)$")

    elif material_str == "Elas_IsotTrans":

        # --------------------------------------------------------------------------------------------
        # Elas_IsotTrans
        # --------------------------------------------------------------------------------------------

        useZhou = False

        if useZhou:
            # Zhou
            EL_exp: np.ndarray = dict_EL_NUL['mean_EL_data'] * 1e-3 # [GPa]
            n = EL_exp.size
            ET_exp: np.ndarray = dict_ET_GL['mean_ET_data'] * 1e-3
            GL_exp: np.ndarray = dict_ET_GL['mean_GL_data'] * 1e-3
            vL_exp: np.ndarray = dict_EL_NUL['mean_NUL_data']

        else:
            dfParams = dfParams[:-1]
            # Matthieu
            EL_exp: np.ndarray = dfParams["El"].values * 1e-3
            n = EL_exp.size
            ET_exp: np.ndarray = dfParams["Et"].values * 1e-3
            GL_exp: np.ndarray = dfParams["Gl"].values * 1e-3
            vL_exp = 0.01+0.1*np.random.rand(n) # artificial data for vL varying from 0.01 to 0.11

        vT_exp = 0.1+0.2*np.random.rand(n); # artificial data for vT varying from 0.1 to 0.3

        mat_exp = Materials.Elas_IsotTrans(3, EL_exp, ET_exp, GL_exp, vL_exp, vT_exp)
        c1_d, c2_d, c3_d, c4_d, c5_d = mat_exp.Walpole_Decomposition()[0]
        
        mat = Materials.Elas_IsotTrans(3,EL_exp.mean(),ET_exp.mean(),GL_exp.mean(),vL_exp.mean(),vT_exp.mean())
        c1, c2, c3, c4, c5 = mat.Walpole_Decomposition()[0]

        # --------------------------------------------------------------------------------------------
        # Metropolis hastings
        # --------------------------------------------------------------------------------------------
        m_c1_d = np.mean(c1_d)
        m_c2_d = np.mean(c2_d)
        m_c3_d = np.mean(c3_d)
        m_c4_d = np.mean(c4_d)
        m_c5_d = np.mean(c5_d)
        m_c_d = np.array([m_c1_d, m_c2_d, m_c3_d, m_c4_d, m_c5_d])

        phiC_d = np.log((c1_d*c2_d - c3_d**2) * c4_d**2 * c5_d**2)
        vC_d = np.mean(phiC_d)
        
        sol_d = np.append(m_c_d,[vC_d])

        c_0 = np.array([m_c1_d, m_c2_d, m_c3_d])
        cov = lambda l: 1/l * np.array([[-m_c1_d**2, -m_c3_d**2, -m_c1_d*m_c3_d],
                                [-m_c3_d**2, -m_c2_d**2, -m_c2_d*m_c3_d],
                                [-m_c1_d*m_c3_d, -m_c2_d*m_c3_d, -(m_c3_d**2 + m_c1_d*m_c2_d)/2]])

        
        def p_C(c: np.ndarray, lamb: np.ndarray):
            """Function to samples in"""

            c1, c2, c3 = tuple(c[:3])
            l, l1, l2, l3 = tuple(lamb[:4])

            p = (c1*c2-c3**2)**-l * np.exp(-l1*c1 -l2*c2 -l3*c3)
            # p = np.exp(-l * np.log(c1*c2-c3**2)) * np.exp(-l1*c1 -l2*c2 -l3*c3)

            return p

        def Metropolis_Hastings_C(c_0: np.ndarray, cov: np.ndarray, lamb: np.ndarray, burn_in: int, nSamples: int):
            # in this function we use a normal distribution as our guess function

            # pdf multivariate normal with mean t0 and covariance cov evaluated in t1    
            q = lambda c_t1, c_t0: stats.multivariate_normal.pdf(c_t1, c_t0, cov)

            c_t = c_0
            samples = []
            for i in range(burn_in+nSamples):
                # proposal / guess
                c_tp1 = stats.multivariate_normal.rvs(c_t, cov)

                # accept ratio
                # a = (p_C(c_tp1, lamb) * q(c_t, c_tp1))/(p_C(c_t, lamb) * q(c_tp1, c_t))

                a = p_C(c_tp1, lamb)/p_C(c_t, lamb)
                test = q(c_t, c_tp1)/q(c_tp1, c_t) # = 1 if the guess function is symmetric
                assert test == 1

                isAccepted = np.random.uniform(0, 1) < a
                
                if isAccepted:
                    c_t = c_tp1
                    if i >= burn_in:
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

            # test_lamb = 2*np.sqrt(l1*l2) - l3
            # # print(test_lamb)
            # assert test_lamb >= 1e-12

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
            # we use Metropolis-Hastings to make samples
            # cc = 1 if useZhou else .2
            cc = 1
            samples, rejectRatio = Metropolis_Hastings_C(c_0, cov(l)*cc, lamb, 0, 10000)            
            # _s for samples
            c1_s: np.ndarray = samples[:,0]; m_c1_s: float = np.mean(c1_s)
            c2_s: np.ndarray = samples[:,1]; m_c2_s: float = np.mean(c2_s)
            c3_s: np.ndarray = samples[:,2]; m_c3_s: float = np.mean(c3_s)

            m_c_s = np.array([m_c1_s, m_c2_s, m_c3_s, m_c4_s, m_c5_s])

            mPhi_c123_s: float = np.mean(np.log(c1_s*c2_s - c3_s**2))

            vC_s = mPhi_c123_s + mPhi_c4_s + mPhi_c5_s
            
            sol_s = np.append(m_c_s,[vC_s])
            J = np.linalg.norm(sol_d-sol_s)**2/np.linalg.norm(sol_d)**2
            # J = np.linalg.norm(sol_d-sol_s)**2

            print(f"{J:.3e}")

            return J

        l0 = -100
        # l0 = -300
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
                (-np.inf, np.inf),
                (1e-12, np.inf),
                (1e-12, np.inf))
        
        cons = ({'type': 'ineq', 'fun': lambda l:  2*np.sqrt(l[1]*l[2])-l[3]-1e-12})
        
        methods = ["COBYLA", "SLSQP", "trust-constr"]
        # only works with COBYLA
        res = minimize(J_ls, lamd0, bounds=bnds, constraints=cons, method=methods[0])
        print(res)
        lamb = res.x

        print()
        [print(f"l{i} = {l:.3f}") for i, l in enumerate(lamb)]

        if useZhou:
            print()
            [print(f"err_l{i} = {np.abs((l-lamb[i])/l):.2e}") for i, l in enumerate([-100.0306, 788.8486, 43.5508, -12.4068, 133.0846, 1878.4404])]

        print(f'\na4 = {1-2*lamb[0]:.4f}')
        print(f'b4 = {1/lamb[4]:.4f}')
        print(f'\na5 = {1-2*lamb[0]:.4f}')
        print(f'b5 = {1/lamb[5]:.4f}')

        # J = J_ls(lamb)
        
        # l0 = -100 -> l0 = -98.713, l1 = 7.019, l2 = 409.955, l3 = -0.217, l4 = 1254.545, l5 = 131.488
        # J = 5.207e-04

        # l0 = -50 -> l0 = -48.749, l1 = 3.488, l2 = 205.365, l3 = 0.376, l4 = 625.027, l5 = 65.141
        # J = 4.286e-04

        # l0 > 0 -> ne fonctionne pas

        # l0 = -200 -> l0 = -198.750, l1 = 13.952, l2 = 813.796, l3 = -2.798, l4 = 2522.943, l5 = 262.057
        # J = 7.087e-04

        # l0 = -500 -> 
        # J = 


        # file = Folder.Join(folder, 'lambda.pickle')
        # with open(file, 'wb') as f:
        #     pickle.dump(lamb, f)

        # --------------------------------------------------------------------------------------------
        # Plot
        # --------------------------------------------------------------------------------------------
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
        samples, rejectRatio = Metropolis_Hastings_C(c_0, cov(lamb[0]), lamb, 0, 10000)
        ax_join_pdf.scatter(samples[:,0],samples[:,1],samples[:,2],zorder=3, c='blue', marker='.')

        plt.show()

    else:
        raise Exception("Not implemented")




    pass