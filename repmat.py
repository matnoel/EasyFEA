# %%
import os
import numpy as np
from numpy.core.fromnumeric import reshape
import numpy.matlib

os.system("cls")    #nettoie le terminal

# dN = [Ni,ksi ...
#       Ni,eta ...]
dN = np.array([[-1, 1, 0],[-1, 0, 1]])

dN = np.tile(dN, (3,1,1))

A = np.array([[1,2],[3,4]])

# B = numpy.matlib.repmat(A, (1, 1, 2))
B = np.tile(A, (100000,1,1))

# B = A[:,:, [np.newaxis]*2]

print(B[:,:,:])
# B[:,:,1]
# B[:,:,2]
print("")

dot = np.array([B[e,:,:].dot(B[e,:,:]) for e in range(B.shape[0])])
dot


# %%
