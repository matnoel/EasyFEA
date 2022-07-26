# import matplotlib
# # Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')

from petsc4py import PETSc
import numpy as np

n = 1000

nnz = 3 * np.ones(1000, dtype=np.int32)
nnz[0] = nnz[-1] = 2

A = PETSc.Mat()
A.createAIJ([n, n], nnz=nnz)

# First set the first row
A.setValue(0, 0, 2)
A.setValue(0, 1, -1)
# Now we fill the last row
A.setValue(999, 998, -1)
A.setValue(999, 999, 2)

# And now everything else
for index in range(1, n - 1):
    A.setValue(index, index - 1, -1)
    A.setValue(index, index, 2)
    A.setValue(index, index + 1, -1) 

A.assemble()

A.size

A.local_size

A.isSymmetric()

ksp = PETSc.KSP().create()

b = A.createVecLeft()
b.array[:] = 1

x = A.createVecRight()

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType('bcgs')
ksp.setConvergenceHistory()
ksp.getPC().setType('none')
ksp.solve(b, x)


from matplotlib import pyplot as plt

residuals = ksp.getConvergenceHistory()
plt.semilogy(residuals)

plt.show()
