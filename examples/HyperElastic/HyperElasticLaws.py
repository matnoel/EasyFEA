# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA import Display

try:
    import sympy    
except ModuleNotFoundError:
    raise Exception("sympy must be installed!")

def Compute(W, params: list, details=True):

    print(f"W = {W}\n")

    # dW
    dW = ""
    for i, param_i in enumerate(params):        
        dWdIi = sympy.diff(W, param_i)
        if dWdIi != 0:
            dW += " + "
            if details:
                print(f"dWdI{i+1} = {dWdIi}")
                dW += f"dWdI{i+1} * dI{i+1}dC"
            else:
                dW += f"({dWdIi}) * dI{i+1}dC"

    dW = f"dW = 2 * ({dW})\n"
    dW = dW.replace("+ -", "- ")
    dW = dW.replace("( + ", "(")
    print(dW)

    # d2W
    d2W1 = ""
    d2W2 = ""

    for i, param_i in enumerate(params):
        dWdIi = sympy.diff(W, param_i)
        if dWdIi != 0:
            d2W1 += " + "
            if details:
                print(f"dWdI{i+1} = {dWdIi}")
                d2W1 += f"dWdI{i+1} * d2I{i+1}dC"
            else:
                d2W1 += f"({dWdIi}) * d2I{i+1}dC"
                    
        for j, param_j in enumerate(params):
            d2WdIiIj = sympy.diff(dWdIi, param_j)
            if d2WdIiIj != 0:
                d2W2 += " + "
                if details:
                    print(f"d2WdI{i+1}I{j+1} = {d2WdIiIj}")
                    d2W2 += f"d2WdI{i+1}I{j+1} * dI{i+1}dC @ dI{j+1}dC.T"
                else:
                    d2W2 += f"({d2WdIiIj}) * dI{i+1}dC @ dI{j+1}dC.T"

    if d2W2 == "":
        d2W = f"d2W = 4 * ({d2W1})"
    else:
        d2W = f"d2W = 4 * ({d2W1}) + 4 * ({d2W2})"
    d2W = d2W.replace("+ -", "- ")
    d2W = d2W.replace("( + ", "(")
    print(d2W)

if __name__ == "__main__":

    Display.Clear()

    I1, I2, I3, I4 = sympy.symbols("I1, I2, I3, I4")

    J1 = I1 * I3**(sympy.Rational(-1,3))
    J2 = I2 * I3**(sympy.Rational(-2,3))
    J = I3**(sympy.Rational(1,2))

    # -------------------------------------
    # Neo-Hookean
    # -------------------------------------

    Display.Section("Neo-Hookean")

    K = sympy.symbols("K")
    
    W = K * (J1 - 3)

    Compute(W, [I1, I2, I3])

    # -------------------------------------
    # Mooney-Rivlin
    # -------------------------------------   

    Display.Section("Mooney-Rivlin")

    K1, K2 = sympy.symbols("K1, K2")
    
    W = K1 * (J1 - 3) + K2 * (J2 - 3)

    Compute(W, [I1, I2, I3])

    # -------------------------------------
    # Saint-Venant-Kirchhoff
    # -------------------------------------   

    Display.Section("Saint-Venant-Kirchhoff")

    lmbda, mu = sympy.symbols("lmbda, mu")
    
    # W = lmbda/8 * (I1**2 - 6*I1 + 9) + mu/4 * (I1**2 - 2*I1 - 2*I2 + 3)
    W = (lmbda/8 + mu/4) * I1**2 - mu*I2/2  - (3*lmbda/4 + mu/2) * I1 + 9*lmbda/8 + 3*mu/4

    Compute(W, [I1, I2, I3])