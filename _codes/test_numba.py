import numba
import numpy as np
from cythonScripts import CalcCython


import TicTac
import Affichage
import matplotlib.pyplot as plt


Affichage.Clear()

A = np.array([[1,2],[3,4]])
B = A*2
C = A*3

ABC = np.dot(np.dot(A.T, B), C)

print(A.dot(B))

print(ABC)

res = np.zeros_like(A)

for l in np.arange(2):
    for j in np.arange(2):
        for k in np.arange(2):
            for i in np.arange(2):
                res[i,j] += A[k,i] * B[k,l] * C[l,j]

print(res-ABC)

@numba.njit()
def matmul(A, b):
    c = np.zeros_like(b)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            c[i] += A[i,j] * b[j]    
    return c

@numba.njit(parallel=True)
def matmulParal(A, b):
    c = np.zeros_like(b)
    for i in numba.prange(A.shape[0]):
        for j in numba.prange(A.shape[1]):        
            c[i] += A[i,j] * b[j]    
    return c

# @numba.guvectorize([(numba.float64[:,:], numba.float64[:], numba.float64[:])], '(i,j),(j)->(i)', nopython=True, target='parallel')
@numba.guvectorize([(numba.float64[:,:], numba.float64[:,:], numba.float64[:,:])], '(i,j),(j,k)->(i,k)',
target='cpu', nopython=True)
def ij_jk_to_ik(A, B, result):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(B.shape[1]):
                result[i,k] += A[i,j] * B[j,k]

result = ij_jk_to_ik(A, B, np.zeros((A.shape[0], A.shape[1])))

@numba.guvectorize([(numba.float64[:,:], numba.float64[:], numba.float64[:])], '(i,j),(j)->(i)',
nopython=True, target='parallel')
def ij_j_to_i(A, b, result):
    for i in np.arange(A.shape[0]):
        for j in np.arange(A.shape[1]):
            result[i] += A[i,j] * b[j]
                

# N = 20000
# # N = 30000
# # N = 40000
# list_N = np.arange(10000, N, 1000)

N = 100
# N = 30000
# N = 40000
list_N = np.arange(2, 10000, 100,)



list_npeinsum = []
list_npeinsum_True = []
list_npeinsum_optimal = []
list_npeinsum_greedy = []
list_matmulParal = []
list_ij_j_to_i = []
list_cythonij_j_to_i = []


for n in list_N:

    print(n)

    tic = TicTac.Tic()

    A = np.ones((n, n), dtype=float)
    b = np.ones((n), dtype=float)
    tic.Tac("Test", "A et b", False)

    # c = np.dot(A, b)
    # tic.Tac("Test", "np.dot", False)

    c = np.einsum('ij,j', A, b)
    list_npeinsum.append(tic.Tac("Test", "np.einsum", False))

    c = np.einsum('ij,j', A, b, optimize=True)
    list_npeinsum_True.append(tic.Tac("Test", "np.einsum optimize", False))

    c = np.einsum('ij,j', A, b, optimize='optimal')
    list_npeinsum_optimal.append(tic.Tac("Test", "np.einsum optimal", False))

    c = np.einsum('ij,j', A, b, optimize='greedy')
    list_npeinsum_greedy.append(tic.Tac("Test", "np.einsum greedy", False))

    # c = matmul(A, b)
    # tic.Tac("Test", "matmul compil", False)

    # c = matmul(A, b)
    # tic.Tac("Test", "matmul", False)

    c = matmulParal(A, b)
    tic.Tac("Test", "matmulParal compil", False)

    c = matmulParal(A, b)
    oldC = c.copy()
    list_matmulParal.append(tic.Tac("Test", "matmulParal", False))

    c = np.zeros_like(c)
    c = ij_j_to_i(A, b, c)
    list_ij_j_to_i.append(tic.Tac("Test", "matmulVect", False))

    c = CalcCython.ij_j_to_i(A, b) - c
    list_cythonij_j_to_i.append(tic.Tac("Test", "cython", False))

plt.plot(list_N, list_npeinsum, label='np.einsum')
plt.plot(list_N, list_npeinsum_True, label='np.einsum optimize')
plt.plot(list_N, list_npeinsum_optimal, label='einsum optimal')
plt.plot(list_N, list_npeinsum_greedy, label='np.einsum greedy')
plt.plot(list_N, list_matmulParal, label='matmulParal')
plt.plot(list_N, list_ij_j_to_i, label='guvetorize')
plt.plot(list_N, list_cythonij_j_to_i, label='cython')

plt.legend()

# plt.figure()
# rapport = np.array(list_npeinsum_optimal)*1/np.array(list_matmulParal)
# print(f"mean = {np.mean(rapport)}")
# plt.plot(list_N, rapport)
# plt.ylabel("einsum_optimal / matmulParal")




































plt.show()


# path = np.einsum_path('ij,j', A, b, optimize='optimal')
# print()
# print(path[0])
# print(path[1])



