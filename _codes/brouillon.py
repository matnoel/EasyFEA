
# %%

import numpy as np

A = np.array([[1,2],[3,4]])
print(A)


At = np.einsum('ij->ji', A)
print(At)

