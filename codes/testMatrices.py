import numpy as np
import scikits.umfpack as umfpack
from petsc4py import PETSc

N=100
A = np.random.random((N,N))
b = np.random.random((N,1))

x = np.linalg.solve(A, b)
print(np.linalg.norm(A.dot(x)-b))

x = umfpack.spsolve(A, b)
print(np.linalg.norm(A.dot(x)-b))

# PETSc.