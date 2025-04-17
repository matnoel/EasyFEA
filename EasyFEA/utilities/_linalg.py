# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Linear algebra functions."""

import numpy as np
from ..fem import FeArray

def __CheckMat(mat: np.ndarray) -> None:
    assert isinstance(mat, np.ndarray) and mat.ndim >= 2 and mat.shape[-2] == mat.shape[-1], "must be a (..., dim, dim) array"
    dim = mat.shape[-1]
    assert dim > 0

def Transpose(mat: np.ndarray) -> np.ndarray:
    """Computes transpose(mat)"""
    assert isinstance(mat, np.ndarray) and mat.ndim >= 2
    res = np.einsum("...ij->...ji", mat, optimize="optimal")
    
    if isinstance(mat, FeArray):
        res = FeArray.asfearray(res)

    return res

def Trace(mat: np.ndarray) -> np.ndarray:
    """Computes trace(mat)"""
    __CheckMat(mat)
    # same as np.trace(A, axis1=-2, axis2=-1)
    res = np.einsum("...ii->...", mat, optimize="optimal")
    
    if isinstance(mat, FeArray):
        res = FeArray.asfearray(res)

    return res

def Det(mat: np.ndarray) -> np.ndarray:
    """Computes det(mat)"""
    __CheckMat(mat)

    dim = mat.shape[-1]

    if dim == 1:        
        det = mat[Ellipsis,0,0]

    elif dim == 2:
        a = mat[Ellipsis,0,0]
        b = mat[Ellipsis,0,1]
        c = mat[Ellipsis,1,0]
        d = mat[Ellipsis,1,1]
        
        det = (a*d)-(c*b)
    
    elif dim == 3:
        a11 = mat[Ellipsis,0,0]; a12 = mat[Ellipsis,0,1]; a13 = mat[Ellipsis,0,2]
        a21 = mat[Ellipsis,1,0]; a22 = mat[Ellipsis,1,1]; a23 = mat[Ellipsis,1,2]
        a31 = mat[Ellipsis,2,0]; a32 = mat[Ellipsis,2,1]; a33 = mat[Ellipsis,2,2]

        det = a11 * ((a22*a33)-(a32*a23)) - a12 * ((a21*a33)-(a31*a23)) + a13 * ((a21*a32)-(a31*a22))

    else:
        det = np.linalg.det(mat)
    
    if isinstance(mat, FeArray):
        det = FeArray.asfearray(det)

    return det

def Inv(mat: np.ndarray):
    """Computes inv(mat)"""
    __CheckMat(mat)

    dim = mat.shape[-1]

    if dim == 1:
        inv = 1/mat
        
    elif dim == 2:
        # mat = [alpha, beta          inv(mat) = 1/det * [b, -beta
        #        a    , b   ]                            -a,  alpha]
        
        inv = np.zeros_like(mat, dtype=float)

        det = Det(mat)

        alpha = mat[Ellipsis,0,0]
        beta = mat[Ellipsis,0,1]
        a = mat[Ellipsis,1,0]
        b = mat[Ellipsis,1,1]

        adj = np.zeros_like(mat)

        adj[Ellipsis,0,0] = b
        adj[Ellipsis,0,1] = -beta
        adj[Ellipsis,1,0] = -a
        adj[Ellipsis,1,1] = alpha

        inv = np.einsum('...,...ij->...ij',1/det, adj, optimize='optimal')
        
    elif dim == 3:
        # optimized such that invmat = 1/det * Adj(mat)
        # https://fr.wikihow.com/calculer-l'inverse-d'une-matrice-3x3

        det = Det(mat)

        matT = Transpose(mat)

        a00 = matT[Ellipsis,0,0]; a01 = matT[Ellipsis,0,1]; a02 = matT[Ellipsis,0,2]
        a10 = matT[Ellipsis,1,0]; a11 = matT[Ellipsis,1,1]; a12 = matT[Ellipsis,1,2]
        a20 = matT[Ellipsis,2,0]; a21 = matT[Ellipsis,2,1]; a22 = matT[Ellipsis,2,2]

        det00 = (a11*a22) - (a21*a12); det01 = (a10*a22) - (a20*a12); det02 = (a10*a21) - (a20*a11)
        det10 = (a01*a22) - (a21*a02); det11 = (a00*a22) - (a20*a02); det12 = (a00*a21) - (a20*a01)
        det20 = (a01*a12) - (a11*a02); det21 = (a00*a12) - (a10*a02); det22 = (a00*a11) - (a10*a01)

        adj = np.zeros_like(mat)

        # Don't forget the - or + !!!
        adj[Ellipsis,0,0] = det00; adj[Ellipsis,0,1] = -det01; adj[Ellipsis,0,2] = det02
        adj[Ellipsis,1,0] = -det10; adj[Ellipsis,1,1] = det11; adj[Ellipsis,1,2] = -det12
        adj[Ellipsis,2,0] = det20; adj[Ellipsis,2,1] = -det21; adj[Ellipsis,2,2] = det22

        inv = np.einsum('...,...ij->...ij',1/det, adj, optimize='optimal')

    else:

        inv = np.linalg.inv(mat)

    if isinstance(mat, FeArray):
        inv = FeArray.asfearray(inv)

    return inv

def TensorProd(A: np.ndarray, B: np.ndarray, symmetric=False, ndim:int=None) -> np.ndarray:
    """Computes tensor product.

    Parameters
    ----------
    A : np.ndarray
        array A
    B : np.ndarray
        array B 
    symmetric : bool, optional
        do symmetric product, by default False
    ndim : int, optional
        ndim=1 -> vect or ndim=2 -> matrix, by default None

    Returns
    -------
    np.ndarray
        the calculated tensor product
    """

    assert isinstance(A, np.ndarray)
    assert isinstance(B, np.ndarray)
        
    sizeA = A.size
    sizeB = B.size
    assert sizeA == sizeB, "A and B must have the same dimensions"

    if ndim is None:
        ndim = A.ndim

    assert ndim in [1,2], "A and B must be vectors (i) or matrices (ij)"

    if ndim == 1:
        # vectors
        # Ai Bj
        res = np.einsum('...i,...j->...ij', A, B)

    elif ndim == 2:
        # matrices
        if symmetric:
            # 1/2 * (Aik Bjl + Ail Bjk) = 1/2 (p1 + p2)
            p1 = np.einsum('...ik,...jl->...ijkl', A, B, optimize='optimal')
            p2 = np.einsum('...il,...jk->...ijkl', A, B, optimize='optimal')
            res = 1/2 * (p1 + p2)
        else:
            # Aij Bkl
            res = np.einsum('...ij,...kl->...ijkl', A, B, optimize='optimal')

    else:
        raise Exception("Not implemented")
    
    if isinstance(A, FeArray) or isinstance(B, FeArray):
        res = FeArray.asfearray(res)
    
    return res

def Norm(array: np.ndarray, **kwargs):
    """`np.linalg.norm()` wrapper.\n
    see https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html"""

    res = np.linalg.norm(array, **kwargs)

    if isinstance(array, FeArray):
        res = FeArray.asfearray(res)

    return res