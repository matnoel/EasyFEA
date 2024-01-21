import Display
import Folder
import Simulations
import Materials
import Functions
from Samples import folder_save as samplesFolder
from TicTac import Tic

import pickle
from Display import plt, np
import pandas as pd
import multiprocessing
from datetime import datetime

doSimulation = False
doPlot = True

useParallel = True
nProcs = 15 # None means every processors

folder = Folder.Get_Path(__file__)
folder_Sto = Folder.New_File(Folder.Join('FCBA','Sto'), results=True)

# simulations [start, start+N]
start = 0
N = 3000 # N simulations

# mesh
nL = 100
test = False
optimMesh = True

# phase field
split = "He" # he, Zhang, AnisotStress
regu = "AT1"
tolConv = 1e-2 # 1e-0, 1e-1, 1e-2
convOption = 2
# (0, bourdin)
# (1, crack energy)
# (2, crack + strain energy

Display.Clear()

# --------------------------------------------------------------------------------------------
# Mesh
# --------------------------------------------------------------------------------------------

L = 45
H = 90
D = 10
t = 20

l0 = L/nL

mesh = Functions.DoMesh(2,L,H,D,t,l0,test,optimMesh)
nodes_lower = mesh.Nodes_Conditions(lambda x,y,z: y==0)
nodes_upper = mesh.Nodes_Conditions(lambda x,y,z: y==H)
nodes_corner = mesh.Nodes_Conditions(lambda x,y,z: (x==0) & (y==0))
dofsY_upper = Simulations.BoundaryCondition.Get_dofs_nodes(2, 'displacement', nodes_upper, ['y'])
# Display.Plot_Mesh(mesh)

# --------------------------------------------------------------------------------------------
# Samples
# --------------------------------------------------------------------------------------------

with open(Folder.Join(samplesFolder, 'samples_article.pickle'), 'rb') as f:
    samples: np.ndarray = pickle.load(f)
    # [c1_s,c2_s,c3_s,c4_s,c5_s,gc_s] = (N,6)
    # GPa and mJ/mm2        
mean = np.mean(samples, 0)

mat_ex = Functions.Get_material(0,t,3)
E1,E2,E3,E4,E5 = mat_ex.Walpole_Decomposition()[1]

# do verif
dfParams = Functions.dfParams
EL_exp: np.ndarray = dfParams["El"].values * 1e-3
n = EL_exp.size
ET_exp: np.ndarray = dfParams["Et"].values * 1e-3
GL_exp: np.ndarray = dfParams["Gl"].values * 1e-3
vL_exp = 0.01+0.1*np.random.rand(n) # artificial data for vL varying from 0.01 to 0.11
vT_exp = 0.1+0.2*np.random.rand(n); # artificial data for vT varying from 0.1 to 0.3        
    
mat = Materials.Elas_IsotTrans(2,EL_exp.mean(),ET_exp.mean(),GL_exp.mean(),vL_exp.mean(),vT_exp.mean())
ci = mat.Walpole_Decomposition()[0]

errCi = np.linalg.norm(ci - mean[:-1])**2/np.linalg.norm(ci)**2
# print(f'errCi = {errCi:.3e}')

# --------------------------------------------------------------------------------------------
# Simu
# --------------------------------------------------------------------------------------------

# Loading increments
treshold = 0.2
inc0 = 5e-3
inc1 = 1e-3

def DoSimu(s: int, sample: np.ndarray) -> tuple[int, list, list, list, float]:

    tic = Tic()

    c1,c2,c3,c4,c5,gc = sample    

    C = c1*E1 + c2*E2 + c3*E3 + c4*E4 + c5*E5 # GPa
    C *= 1e3 # MPa

    material = Materials.Elas_Anisot(2, C, False, planeStress=True, thickness=t)

    pfm = Materials.PhaseField_Model(material, split, regu, gc, l0)    

    simu = Simulations.Simu_PhaseField(mesh, pfm, useNumba=not useParallel)

    list_du: list[float] = []
    list_f: list[float] = []
    list_d: list[float] = []

    du = -inc0
    while simu.damage.max() <= 1:

        du += inc0 if simu.damage.max() <= treshold else inc1

        simu.Bc_Init()
        simu.add_dirichlet(nodes_lower, [0], ['y'])
        simu.add_dirichlet(nodes_corner, [0], ['x'])
        simu.add_dirichlet(nodes_upper, [-du], ['y'])

        u, d, Kglob, convergence = simu.Solve(tolConv, 500, convOption)

        dmax = d.max()

        if convergence:
            f = - np.sum(Kglob[dofsY_upper, :] @ u) / 1000
            list_du.append(du)
            list_f.append(f)
            list_d.append(dmax)
        else:
            return (s, [], [], [], 0)
        
    time = tic.Tac()
    timeCoef, unite = Tic.Get_time_unity(time)
    print(f'{s}\t {s/N*100:3.2f}%\t {timeCoef:.2f} {unite}')    

    return (s, list_du, list_f, list_d, time)

# --------------------------------------------------------------------------------------------
# Simulations
# --------------------------------------------------------------------------------------------

if __name__ == '__main__':

    config = Functions.Config(start, N, test,
                              split, regu,
                              tolConv, convOption,
                              nL, optimMesh)
    
    label_u: str = "displacement [mm]"
    label_f: str = "force [kN]"
    label_d: str = "damage"

    folder_save = Folder.Join(folder_Sto, config.path)

    name = '_data_par.pickle' if useParallel else '_data.pickle'
    filePickle = Folder.Join(folder_save, name)

    if doSimulation:

        print(config.config_name)
        if useParallel:
            print(f"nProcs = {nProcs}")        
        print(f'working in\n{folder_save}')

        if not Folder.Exists(folder_save):
            Folder.os.makedirs(folder_save)

        results: list[dict] = []

        items = [(i, samples[i]) for i in range(start,start+N)]

        def addResult(res):
            i, list_du, list_f, list_d, time = tuple(res)
            result = {
                    "i": i,
                    label_u: np.asarray(list_du, dtype=float),
                    label_f: np.asarray(list_f, dtype=float),
                    label_d: np.asarray(list_d, dtype=float),
                    "sample": samples[i], 
                    'time': time
                }
            results.append(result)

        tic = Tic()

        if useParallel:
            with multiprocessing.Pool(nProcs) as pool:
                for res in pool.starmap(DoSimu, items):
                    addResult(res)
            # Display.Clear()

        else:
            for item in items:
                addResult(DoSimu(*item))

        timeSpend = tic.Tac()

        # Save simulation summary
        path_summary = Folder.Join(folder_save, "summary.txt")
        summary = f"Simulations completed on: {datetime.now()}"
        summary += f'\n\nWith config:\n{config.config_name}'
        time, unit = Tic.Get_time_unity(timeSpend)
        summary += f'\n\nElapsed time {time:.2f} {unit}'
        if useParallel:
            summary += f'\n\nnProcs {nProcs}'
        with open(path_summary, 'w', encoding='utf8') as file:
            file.write(summary)

        df = pd.DataFrame(results)
        df.set_index('i', inplace=True)
        df.sort_index()
        df.to_pickle(filePickle)

        print(f'saving:\n{filePickle}')

    else:
        if not Folder.Exists(filePickle):
            print(f"the file \n'{filePickle}'\n does not exists")
            exit()
        else:
            print(f'loading:\n{filePickle}')
            df = pd.read_pickle(filePickle)

    print(df)    

    # --------------------------------------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------------------------------------

    # # Make sure that we get the same results using useParallel or not
    # # see Test/0_20_He_AT1_tolConv1e-02_conv2_nL100_optimMesh
    # # Test/0_10_He_AT1_tolConv1e-02_conv2_nL100_optimMesh
    # # Test/0_5_He_AT1_tolConv1e-02_conv2_nL100_optimMesh
    # df1: pd.DataFrame = pd.read_pickle(Folder.Join(folder_save, '_data_par.pickle'))
    # df2: pd.DataFrame = pd.read_pickle(Folder.Join(folder_save, '_data.pickle'))

    # diff_u = df1[label_u].values - df2[label_u].values
    # diff_f = df1[label_f].values - df2[label_f].values
    # print(diff_u)
    # print(diff_f)
    # ax = plt.subplots()[1]
    # for i in range(N):
    #     ax.plot(df1[label_u][i], df1[label_f][i], c='blue')
    #     ax.plot(df2[label_u][i], df2[label_f][i], c='red')

    if doPlot:

        # get u, f and d for each simulations
        labels = [label_u, label_f, label_d]
        vals = [[df[label][i][[0, df[label][i].size//2 ,-1]] for label in labels] for i in range(N)]
        u, f, d = np.asarray(vals).transpose((1,0,2))
        # u = [u_min, u_mid, u_max]
        # f = [f_min, f_mid, f_max]
        # d = [d_min, d_mid, d_max]

        # y = a x + b
        a = (f[:,1] - f[:,0])/(u[:,1] - u[:,0])
        b = 0

        u_array = np.linspace(0, np.max(u), 1000)
        f_arrays = np.einsum('n,i->ni', a, u_array)
        
        from scipy import stats

        mu, sig = stats.norm.fit(f[:,-1])

        ff = np.linspace(f[:,-1].min(), f[:,-1].max(), 1000)

        axHs: list[plt.Axes] = plt.subplots(nrows=2, sharex=True)[1]
        axHs[0].plot(ff, stats.norm.cdf(ff, mu, sig))
        axHs[0].set_title('cdf'); axHs[0].grid()
        axHs[1].hist(f[:,-1], bins='auto', histtype='bar', density=True)
        axHs[1].plot(ff, stats.norm.pdf(ff,mu, sig), label=f"$\mu = {mu:.2f}, \sigma = {sig:.2f}$")
        axHs[1].legend(); axHs[1].set_xlabel('f [kN]'); axHs[1].set_title('pdf'); axHs[1].grid()
        # Display.Save_fig(folder_save, 'pdf de f')







        # get back the forces and displacements curve for each test
        list_forces = []
        list_displacements = []
        list_fcrit = []
        list_k_montage = []

        mesh = Functions.DoMesh(2,L,H,D,t,l0,True, False)
        nodes_lower = mesh.Nodes_Conditions(lambda x,y,z: y==0)
        nodes_upper = mesh.Nodes_Conditions(lambda x,y,z: y==H)
        nodes_corner = mesh.Nodes_Conditions(lambda x,y,z: (x==0) & (y==0))
        dofsY_upper = Simulations.BoundaryCondition.Get_dofs_nodes(2, 'displacement', nodes_upper, ['y'])

        for i in range(17):

            mat = Functions.Get_material(i, t, 2)

            forces, displacements, fcrit = Functions.Get_loads_informations(i, useRedim=True)

            simu = Simulations.Simu_Displacement(mesh, mat)

            simu.add_dirichlet(nodes_lower, [0], ['y'])
            simu.add_dirichlet(nodes_corner, [0], ['x'])
            # simu.add_surfLoad(nodes_upper, [-15000/(L*t)], ['y'])
            simu.add_dirichlet(nodes_upper, [-0.5], ['y'])
            simu.Solve()

            f = -np.sum(simu.Get_K_C_M_F()[0][dofsY_upper] @ simu.displacement)/1000
            u = - simu.displacement[dofsY_upper].mean()

            k_exp, __ = Functions.Calc_a_b(forces, displacements, 15)
            k_mat, __ = Functions.Calc_a_b([0, f], [0, u], f)
            k_montage = 1/(1/k_exp - 1/k_mat)

            list_k_montage.append(k_montage)
            list_displacements.append(displacements)
            list_forces.append(forces)
            list_fcrit.append(fcrit)

        

        ax = plt.subplots()[1]
        ax.set_xlabel(r"$\Delta u \ [mm]$")
        ax.set_ylabel(r"$f \ [kN]$")

        ax.grid()        

        # [ax.plot(u_array, f_arrays[i],c='gray', alpha=.3) for i in range(N)]
        # [ax.plot(df[label_u][i], df[label_f][i],c='gray', alpha=.3) for i in range(N)]
        # [ax.plot(df[label_u][i], df[label_f][i], alpha=.3) for i in range(N)]
        
        ax.plot(u[:,-1], f[:,-1], c='red', ls='', marker='.', label=f"{np.mean(f[:,2]):.2f} kN")

        a_lower, a_upper = np.quantile(a, (0.025, 0.975))

        ax.fill_between(u_array, a_lower*u_array, a_upper*u_array, zorder=8, alpha=.5)

        ax.plot(u_array, np.mean(a)*u_array, c='black', ls='--')


        k_montage = np.mean(list_k_montage)
        k_montage = np.mean(a)

        print(list_k_montage)
        print(np.std(list_k_montage)/k_montage)
        
        errors = [2, 8, 11, 13, 17]

        for i in range(17):

            # if i in errors: continue

            displacements = list_displacements[i]
            forces = list_forces[i]
            k_montage = list_k_montage[i]

            displacements = displacements - forces/k_montage

            idx = np.where(forces >= list_fcrit[i])[0][0]

            fcrit = forces[idx]
            displacement_crit = displacements[idx]

            idxMax = np.where(displacements >= 0.5)[0][0]

            ax.plot(displacements[:idxMax], forces[:idxMax], alpha=0.5)
            label = f"{np.mean(list_fcrit):.2f} kN" if i == 0 else ''
            ax.scatter(displacement_crit, fcrit, marker='+',c='black', zorder=10, label=label)
            # ax.text(displacement_crit, fcrit, '{}'.format(i), zorder=10)

        ax.legend()

        Display.Save_fig(folder_save, 'forcedep sto')

        plt.show()