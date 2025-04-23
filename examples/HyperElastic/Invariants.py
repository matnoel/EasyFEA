# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA import Display

try:
    import sympy    
except ModuleNotFoundError:
    raise Exception("sympy must be installed!")

def __Project_Mandel(A, orderA:int=4):

    assert orderA in [2, 4]
    
    # for xx, yy, zz, yz, xz, zy
    e = sympy.Array([
        [0, 5, 4],
        [5, 1, 3],
        [4, 3, 2]
    ])

    kron = lambda a, b: 1 if a == b else 0

    if orderA == 2:
        # Aij -> AI
        A_I = sympy.zeros(6,1)

        for i in range(3):
            for j in range(3):
                A_I[e[i,j]] = sympy.sqrt((2 - kron(i,j))) * A[i,j]

        res = A_I

    elif orderA == 4:
        # Aijkl -> AIJ

        A_IJ = sympy.zeros(6, 6)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        A_IJ[e[i,j],e[k,l]] = sympy.sqrt((2-kron(i,j))*(2-kron(k,l))) * A[i,j,k,l]

        res = A_IJ

    else:
        raise Exception("Not implemented.")

    return res

def __MyDiff(func, list_func: list, order:int=None):

    assert isinstance(list_func, list)

    diff = sympy.diff(func, *list_func).as_mutable()

    ndim = len(diff.shape)

    e = sympy.Array([
        [0, 5, 4],
        [5, 1, 3],
        [4, 3, 2]
    ])

    # The use of a symmetrical matrix C on sympy is problematic here.
    # The off-diagonal terms are incorrectly weighted and need to be corrected.
    # The following function is used to correct the coefficients.
    if ndim == 2:
        for i in range(diff.shape[0]):
            for j in range(diff.shape[1]):
                if e[i,j] > 2:
                    diff[i,j] /= 2
    elif ndim == 4:
        for i in range(diff.shape[0]):
            for j in range(diff.shape[1]):
                for k in range(diff.shape[2]):
                    for l in range(diff.shape[3]):
                        if e[i,j] > 2 and e[k,l] > 2:
                            diff[i,j,k,l] /= 4
                        elif e[i,j] > 2 or e[k,l] > 2:
                            diff[i,j,k,l] /= 2

    if order is not None:
        diff = __Project_Mandel(diff, order)

    diff = sympy.sympify(diff)

    return diff

def Compute(func, name: str, usepprint=True):

    Display.Section(name)

    myprint = sympy.pprint if usepprint else print

    func = sympy.sympify(sympy.expand(func))

    myprint(func)
    
    myprint(f"\nd{name}dC")
    myprint(__MyDiff(func, [C], 2))

    myprint(f"\nd2{name}dC")
    myprint(__MyDiff(func, [C, C], 4))

if __name__ == "__main__":

    Display.Clear()

    cxx, cyy, czz, cyz, cxz, cxy = sympy.symbols("cxx, cyy, czz, cyz, cxz, cxy")
    C = sympy.Matrix([
        [cxx, cxy, cxz],
        [cxy, cyy, cyz],
        [cxz, cyz, czz]
    ])

    sympy.pprint(C)
    print()
    sympy.pprint(__Project_Mandel(C, 2))
    
    # -------------------------------------
    # I1
    # -------------------------------------
    
    I1 = sympy.trace(C)

    Compute(I1, "I1")
    
    # -------------------------------------
    # I2
    # -------------------------------------
    
    I2 = 1/2 * (sympy.trace(C)**2 - sympy.trace(C*C))
    
    Compute(I2, "I2")

    # -------------------------------------
    # I3
    # -------------------------------------
    
    I3 = sympy.det(C)

    Compute(I3, "I3")

    # -------------------------------------
    # I4
    # -------------------------------------

    T = sympy.MatrixSymbol("T", 3, 1)

    I4 = T.transpose() * (C * T)

    Compute(I4, "I4")