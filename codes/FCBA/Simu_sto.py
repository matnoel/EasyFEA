import Display
import Folder
import Simulations
import Materials
import Functions
from Samples import folder_save as samplesFolder

import pickle
from Display import plt, np
import pandas as pd
import multiprocessing

folder = Folder.Get_Path(__file__)

folder_Sto = Folder.New_File(Folder.Join('FCBA','Sto'), results=True)

# Save in FCBA/Sto/start_N/config

doSimulation = False
useParallel = False
nProcs = 5

start = 0
N = 20 # N simulations
# t = list(range(start, start+N))

Display.Clear()

# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------

test = True    
optimMesh = True

L = 45
H = 90
D = 10
t = 20

nL = 100
l0 = L/nL

# mesh
mesh = Functions.DoMesh(L,H,D,l0,test,optimMesh)
nodes_lower = mesh.Nodes_Conditions(lambda x,y,z: y==0)
nodes_upper = mesh.Nodes_Conditions(lambda x,y,z: y==H)
nodes_corner = mesh.Nodes_Conditions(lambda x,y,z: (x==0) & (y==0))
dofsY_upper = Simulations.BoundaryCondition.Get_dofs_nodes(2, 'displacement', nodes_upper, ['y'])
# Display.Plot_Mesh(mesh)

# phase field
split = "He" # he, Zhang, AnisotStress
regu = "AT1"
tolConv = 1e-2 # 1e-0, 1e-1, 1e-2
convOption = 2
# (0, bourdin)
# (1, crack energy)
# (2, crack + strain energy

# Loading increments
treshold = 0.2
inc0 = 8e-3
inc1 = 2e-3

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
    
mat = Materials.Elas_IsotTrans(3,EL_exp.mean(),ET_exp.mean(),GL_exp.mean(),vL_exp.mean(),vT_exp.mean())
ci = mat.Walpole_Decomposition()[0]

errCi = np.linalg.norm(ci - mean[:-1])**2/np.linalg.norm(ci)**2
# print(f'errCi = {errCi:.3e}')

# --------------------------------------------------------------------------------------------
# Simu
# --------------------------------------------------------------------------------------------

def DoSimu(s: int, sample: np.ndarray, mesh: Simulations.Mesh) -> tuple[int, list, list, list]:

    c1,c2,c3,c4,c5,gc = sample    

    C = c1*E1 + c2*E2 + c3*E3 + c4*E4 + c5*E5

    material = Materials.Elas_Anisot(2, C*1e3, False)

    pfm = Materials.PhaseField_Model(material, split, regu, gc, l0)

    simu = Simulations.Simu_PhaseField(mesh, pfm)
    # if useParallel: simu.solver = 'cg'

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
            return (s, [], [], [])

    return (s, list_du, list_f, list_d)

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

        if not Folder.Exists(folder_save):
            Folder.os.makedirs(folder_save)

        results: list[dict] = []

        items = [(i, samples[i], mesh) for i in range(start,start+N)]

        def addResult(res):
            i, list_du, list_f, list_d = tuple(res)
            result = {
                    "i": i,
                    label_u: np.asarray(list_du, dtype=float),
                    label_f: np.asarray(list_f, dtype=float),
                    label_d: np.asarray(list_d, dtype=float),
                    "sample": samples[i]
                }
            results.append(result)
            print(f'{len(results)/len(items)*100:3.0f} %', end='\r')

        if useParallel:            
            with multiprocessing.Pool(nProcs) as pool:
                for res in pool.starmap(DoSimu, items):
                    addResult(res)
        else:
            for item in items:
                addResult(DoSimu(*item))

        df = pd.DataFrame(results)
        df.set_index('i', inplace=True)
        df.sort_index()
        df.to_pickle(filePickle)

    else:
        if not Folder.Exists(filePickle):
            print(f"the file \n'{filePickle}'\n does not exists")
            exit()
        df = pd.read_pickle(filePickle)


    # df1: pd.DataFrame = pd.read_pickle(Folder.Join(folder_save, '_data_par.pickle'))
    # df2: pd.DataFrame = pd.read_pickle(Folder.Join(folder_save, '_data.pickle'))

    # diff_u = (df1[label_u].values - df2[label_u].values).reshape(-1)
    # diff_f = (df1[label_f].values - df2[label_f].values).reshape(-1)
    # print(np.linalg.norm(diff_u))
    # print(np.linalg.norm(diff_f))
    # ax = plt.subplots()[1]
    # for i in range(N):
    #     ax.plot(df1[label_u][i], df1[label_f][i], c='blue')
    #     ax.plot(df2[label_u][i], df2[label_f][i], c='red')
    
    ax = plt.subplots()[1]
    for i in range(N):
        ax.plot(df[label_u][i], df[label_f][i])


    


    print(df)

    plt.show()