# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
ComputeHyperelasticLaws
=======================

Compute hyperelastic constitutive laws.
"""

from EasyFEA import Display

try:
    import sympy
except ModuleNotFoundError:
    raise Exception("sympy must be installed!")


def Compute(W, params: list, details=True):
    print(f"W = {W}\n")

    # dW
    dW = ""
    for param_i in params:
        p_i = str(param_i)
        dWdIi = sympy.diff(W, param_i)
        if dWdIi != 0:
            dW += " + "
            if details:
                print(f"dWd{p_i} = {dWdIi}")
                dW += f"dWd{p_i} * d{p_i}dC"
            else:
                dW += f"({dWdIi}) * d{p_i}dC"

    dW = f"dW = 2 * ({dW})\n"
    dW = dW.replace("+ -", "- ")
    dW = dW.replace("( + ", "(")
    print(dW)

    # d2W
    d2W1 = ""
    d2W2 = ""

    for param_i in params:
        p_i = str(param_i)
        dWdIi = sympy.diff(W, param_i)
        if dWdIi != 0:
            d2W1 += " + "
            if details:
                print(f"dWd{p_i} = {dWdIi}")
                d2W1 += f"dWd{p_i} * d2{p_i}dC"
            else:
                d2W1 += f"({dWdIi}) * d2{p_i}dC"

        for param_j in params:
            p_j = str(param_j)
            d2WdIiIj = sympy.diff(dWdIi, param_j)
            if d2WdIiIj != 0:
                d2W2 += " + "
                if details:
                    print(f"d2Wd{p_i}d{p_j} = {d2WdIiIj}")
                    d2W2 += f"d2Wd{p_i}d{p_j} * TensorProd(d{p_i}dC, d{p_j}dC)"
                else:
                    d2W2 += f"({d2WdIiIj}) * TensorProd(d{p_i}dC, d{p_j}dC)"

    if d2W2 == "":
        d2W = f"d2W = 4 * ({d2W1})"
    else:
        d2W = f"d2W = 4 * ({d2W1}) + 4 * ({d2W2})"
    d2W = d2W.replace("+ -", "- ")
    d2W = d2W.replace("( + ", "(")
    print(d2W)


if __name__ == "__main__":
    Display.Clear()

    I1, I2, I3, I4, I6, I8 = sympy.symbols("I1, I2, I3, I4, I6, I8")

    J1 = I1 * I3 ** (sympy.Rational(-1, 3))
    J2 = I2 * I3 ** (sympy.Rational(-2, 3))
    J = I3 ** (sympy.Rational(1, 2))

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

    W = K1 * (J1 - 3) + K2 * (J2 - 3) + K * (J - 1) ** 2

    Compute(W, [I1, I2, I3])

    # -------------------------------------
    # Saint-Venant-Kirchhoff
    # -------------------------------------

    Display.Section("Saint-Venant-Kirchhoff")

    lmbda, mu = sympy.symbols("lmbda, mu")

    # W = lmbda/8 * (I1**2 - 6*I1 + 9) + mu/4 * (I1**2 - 2*I1 - 2*I2 + 3)
    W = (
        (lmbda / 8 + mu / 4) * I1**2
        - mu * I2 / 2
        - (3 * lmbda / 4 + mu / 2) * I1
        + 9 * lmbda / 8
        + 3 * mu / 4
        + 1 / 2 * K * (I3 - 1) ** 2
    )

    Compute(W, [I1, I2, I3])

    # -------------------------------------
    # Holzapfel-Ogden
    # -------------------------------------

    Display.Section("Holzapfel-Ogden")

    C0, C1, C2, C3, C4, C5, C6, C7 = sympy.symbols("C0:8")

    ks = sympy.symbols("ks")
    bulk, mu1, mu2 = sympy.symbols("bulk, mu1, mu2")

    chi = lambda Ii: 1 / (1 + sympy.exp(-ks * (Ii - 1)))

    W = (
        C0 * (sympy.exp(C1 * (J1 - 3)) - 1)
        + C2 * chi(I4) * (sympy.exp(C3 * (I4 - 1) ** 2) - 1)
        + C4 * chi(I6) * (sympy.exp(C5 * (I6 - 1) ** 2) - 1)
        + C6 * (sympy.exp(C7 * I8**2) - 1)
        + bulk / 4 * (J**2 - 1 - 2 * sympy.ln(J))
        + mu1 * (J1 - 3)
        + mu2 * (J2 - 3)
    )

    Compute(W, [I1, I2, I3, I4, I6, I8])
