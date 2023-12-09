import Display
import Folder
import Simulations
import Materials
import Functions
from Samples import folder_save as samplesFolder

import pickle
from Display import plt, np

folder = Folder.Get_Path(__file__)

with open(Folder.Join(samplesFolder, 'samples_article.pickle'), 'rb') as f:
    samples: np.ndarray = pickle.load(f)
    # [c1_s,c2_s,c3_s,c4_s,c5_s,gc_s] = (N,6)

N = samples.shape[0]
mean = np.mean(samples, 0)

Display.Clear()

# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------

test = True
solve = True
optimMesh = True

L = 45
H = 90
D = 10
t = 20

nL = 100
l0 = L/nL

# phase field
split = "He" # he, Zhang, AnisotStress
regu = "AT1"
tolConv = 1e-2 # 1e-0, 1e-1, 1e-2
convOption = 2
# (0, bourdin)
# (1, crack energy)
# (2, crack + strain energy)

mesh = Functions.DoMesh(L,H,D,l0,test,optimMesh)
nodes_lower = mesh.Nodes_Conditions(lambda x,y,z: y==0)
nodes_upper = mesh.Nodes_Conditions(lambda x,y,z: y==H)
nodes_corner = mesh.Nodes_Conditions(lambda x,y,z: (x==0) & (y==0))

# Display.Plot_Mesh(mesh)

mat_ex = Functions.Get_material(0,t,3)

E1,E2,E3,E4,E5 = mat_ex.Walpole_Decomposition()[1]

def DoSimu(sample: np.ndarray):

    c1,c2,c3,c4,c5,gc = sample    

    C = c1*E1 + c2*E2 + c3*E3 + c4*E4 + c5*E5

    material = Materials.Elas_Anisot(2, C, False)

    pfm = Materials.PhaseField_Model(material, split, regu, gc, l0)


DoSimu(samples[0])
































plt.show()