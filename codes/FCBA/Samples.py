import Display
import Folder
import Materials

import pickle
from Display import plt, np
from scipy import stats
import Functions

folder = Folder.Get_Path(__file__)

folder_save = Folder.New_File(Folder.Join('FCBA','Samples'), results=True)

if __name__ == '__main__':

    Display.Clear()

    N = int(1e6)
    saveFig = True

    samplesFile = Folder.Join(folder_save, 'samples.pickle')

    if Folder.Exists(samplesFile):
        user_input = input(f'{samplesFile} already exists, do you want to overwrite it?  (yes/no): ')
        input = user_input.lower()
        if input == 'yes':
            print('the file will be overwritten')
            makeSamples = True
        elif input == 'no':
            print('loading the file')
            makeSamples = False
        else:
            raise Exception('Type yes or no')
    else:
        makeSamples = True
        
    if makeSamples:

        # --------------------------------------------------------------------------------------------
        # Import
        # --------------------------------------------------------------------------------------------

        dfParams = Functions.dfParams.copy()
        n = dfParams.shape[0]
        EL_exp: np.ndarray = dfParams["El"].values * 1e-3 # GPa
        ET_exp: np.ndarray = dfParams["Et"].values * 1e-3
        GL_exp: np.ndarray = dfParams["Gl"].values * 1e-3
        vL_exp = 0.01+0.1*np.random.rand(n) # artificial data for vL varying from 0.01 to 0.11
        vT_exp = 0.1+0.2*np.random.rand(n); # artificial data for vT varying from 0.1 to 0.3

        mat_exp = Materials.Elas_IsotTrans(3, EL_exp, ET_exp, GL_exp, vL_exp, vT_exp)
        ci, Ei = mat_exp.Walpole_Decomposition()
        
        m_c_d = np.mean(ci, 1)
        m_c1_d, m_c2_d, m_c3_d = m_c_d[:3]
        
        # # ls
        # a_gc = 25.28238478947907
        # b_gc = 0.002563875860029356

        # ml
        a_gc = 27.213736823238317
        b_gc = 0.002381901273972565

        # lambdaFile = Folder.Join(folder, '_lambda.pickle')
        lambdaFile = Folder.Join(folder, '_lambda17.pickle') # with sample17 removed
        with open(lambdaFile, 'rb') as f:
            lamb = pickle.load(f)        

        l,l1,l2,l3,l4,l5 = lamb

        a4 = 1-2*l; a5 = a4
        b4 = 1/l4;  b5 = 1/l5

        # --------------------------------------------------------------------------------------------
        # Make samples
        # --------------------------------------------------------------------------------------------
        
        gc_s = stats.gamma.rvs(a_gc, scale=b_gc, size=N)
        c4_s = stats.gamma.rvs(a4, scale=b4, size=N)
        c5_s = stats.gamma.rvs(a5, scale=b5, size=N)

        # t = stats.gamma.interval(0.95, a_gc, scale=b_gc)

        coef = .2 # coef to multiply cov
        cov = 1/l * np.array([[-m_c1_d**2, -m_c3_d**2, -m_c1_d*m_c3_d],
                            [-m_c3_d**2, -m_c2_d**2, -m_c2_d*m_c3_d],
                            [-m_c1_d*m_c3_d, -m_c2_d*m_c3_d, -(m_c3_d**2 + m_c1_d*m_c2_d)/2]])
            
        def p_C(c: np.ndarray):
            """Function to samples in"""
            c1, c2, c3 = tuple(c[:3])
            p = (c1*c2-c3**2)**-l * np.exp(-l1*c1 -l2*c2 -l3*c3)
            return p

        def Metropolis_Hastings_C(c_0: np.ndarray, burn_in: int, nSamples: int):
            # in this function we use a normal distribution as our guess function

            # pdf multivariate normal with mean t0 and covariance cov evaluated in t1    
            q = lambda c_t1, c_t0: stats.multivariate_normal.pdf(c_t1, c_t0, cov)

            c_t = c_0
            samples = []
            
            i = -1

            while not len(samples) == nSamples:
            # while i < burn_in+nSamples:

                i += 1
            
                # proposal / guess
                c_tp1 = stats.multivariate_normal.rvs(c_t, cov*coef)

                # accept ratio
                a = p_C(c_tp1)/p_C(c_t)
                # a = (p_C(c_tp1) * q(c_t, c_tp1))/(p_C(c_t) * q(c_tp1, c_t))
                # test = q(c_t, c_tp1)/q(c_tp1, c_t) # = 1 if the guess function is symmetric
                # assert test == 1

                isAccepted = np.random.uniform(0, 1) < a
                
                if isAccepted:
                    c_t = c_tp1
                    if i >= burn_in:
                        samples.append(c_t)
                        print(f'{len(samples)/nSamples*100:3.0f} %', end='\r')

            assert len(samples) > 0

            rejectRatio: float = 1 - len(samples)/nSamples

            return np.array(samples), rejectRatio

        c1c2c3_s, rejectRatio = Metropolis_Hastings_C(m_c_d[:3], 0, N)
        

        print(f'\nrejectRatio {rejectRatio*100:3.1f} %') # [20 25] is good modify coef -> cov*coef

        # samples = (c1, c2, c3, c4, c5, gc) # MPa and mJ/mm2
        n = c1c2c3_s.shape[0]
        samples = np.zeros((n, 6), float)
        samples[:,:3] = c1c2c3_s
        samples[:,3] = c4_s[:n]
        samples[:,4] = c5_s[:n]
        samples[:,5] = gc_s[:n]
        
        C_s = np.mean(np.einsum('nc,cij->nij', samples[:,:-1], Ei), 0)

        C_mean = np.sum([c*e for c,e in zip(m_c_d, Ei)], 0)

        err = np.linalg.norm(m_c_d-np.mean(samples[:,:-1], 0))**2/np.linalg.norm(m_c_d)**2
        err2 = (m_c_d-np.mean(samples[:,:-1], 0))/np.linalg.norm(m_c_d)
        err3 = np.linalg.norm(C_mean-C_s)**2/np.linalg.norm(C_mean)**2
        assert err <= 1e-3, 'generated samples are not close enough to the data'

        with open(samplesFile, 'wb') as f:
            pickle.dump(samples, f)

    else:
        with open(samplesFile, 'rb') as f:
            samples: np.ndarray = pickle.load(f)

    # --------------------------------------------------------------------------------------------
    # Make samples
    # --------------------------------------------------------------------------------------------

    n = samples.shape[0]

    params = ['c1','c2','c3','c4','c5','gc']
    for i in range(6):

        ax = plt.subplots()[1]
        ax.plot(range(n), samples[:,i], ls='',marker='.', c='blue', label='samples')
        unit = ' [mJ/mm2]' if 'g' in params[i] else ' [GPa]'
        mean = np.mean(samples[:,i])
        ax.hlines(mean, 0, n, 'red', label=f'{mean:.3f}')
        ax.set_xlabel('N')
        ax.set_ylabel(params[i] + unit)
        ax.legend(loc='upper right')
        if saveFig:
            Display.Save_fig(folder_save, params[i], extension='png', dpi=400)

    plt.show()