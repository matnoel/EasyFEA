import numpy as np
cimport numpy as np

ctypedef np.float64_t dtype_t

cpdef np.ndarray[dtype_t, ndim=1] ij_j_to_i(np.ndarray[dtype_t, ndim=2] A, np.ndarray[dtype_t, ndim=1] b):
    cdef np.ndarray[dtype_t, ndim=1] result = np.zeros((A.shape[0]), dtype=A.dtype)
    cdef int i
    cdef int j
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            result[i] += A[i,j] * b[j]
    return result

