# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
ShapeFunctions
==============

Create Lagrange finite element shape functions.
"""
# sphinx_gallery_thumbnail_number = 6

import numpy as np
import matplotlib.pyplot as plt
from EasyFEA import Display

try:
    import sympy
except ModuleNotFoundError:
    raise Exception("sympy must be installed!")

Display.Clear()

# SEGMENTS
# TRIANGLES
# QUADRANGLES
# TETRAHEDRON
# HEXAHEDRON
# PRISM

# ----------------------------------------------
# Options
# ----------------------------------------------

plot_dN = True
plot_ddN = True
plot_dddN = True
plot_ddddN = True

# coords = sympy.symbols("x, y, z")
coords = sympy.symbols("r, s, t")

# ----------------------------------------------
# Public functions
# ----------------------------------------------


def Compute_and_Print(
    polynom, *args, useSimplify=True, useFactor=True, useEulerBernoulli=False
):
    """Compute and print shape functions and their derivatives for a given polynom.

    Parameters
    ----------
    polynom : function
        Polynomial basis function taking 1, 2, or 3 arguments.
    args : tuple
        Coordinates (1, 2, or 3 lists of x, y, z).
    useSimplify : bool, optional
        Simplify the shape functions. Default is True.
    useFactor : bool, optional
        Factor the shape functions. Default is True.
    useEulerBernoulli : bool, optional
        useEulerBernoulli shape functions. Default is False.
    """

    local_coords, dim = __Get_local_coords_and_dim(*args)

    shape_functions = __Get_shape_functions(
        polynom, local_coords, dim, useSimplify, useFactor, useEulerBernoulli
    )

    __Print_functions(shape_functions, dim, "N")

    # derivative_shape_functions
    if plot_dN:
        dN_functions = __Get_derivative_functions(shape_functions, dim, 1)
        __Print_functions(dN_functions, dim, "dN")

    if plot_ddN:
        ddN_functions = __Get_derivative_functions(shape_functions, dim, 2)
        __Print_functions(ddN_functions, dim, "ddN")

    if plot_dddN:
        dddN_functions = __Get_derivative_functions(shape_functions, dim, 3)
        __Print_functions(dddN_functions, dim, "dddN")

    if plot_ddddN:
        ddddN_functions = __Get_derivative_functions(shape_functions, dim, 4)
        __Print_functions(ddddN_functions, dim, "ddddN")


def Plot_Nodes(title: str, *args):
    local_coords, dim = __Get_local_coords_and_dim(*args)
    nPe = local_coords.shape[0]

    list_x, list_y, list_z = local_coords.T

    if dim == 3:
        ax = Display.Init_Axes(3, elev=16, azim=37)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(title)
        ax.axis("equal")

        ax.scatter(list_x, list_y, list_z)
        [ax.text(list_x[i], list_y[i], list_z[i], i) for i in range(nPe)]
    else:
        ax = Display.Init_Axes(2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.axis("equal")
        ax.grid(zorder=-10)

        ax.scatter(list_x, list_y)
        [ax.text(list_x[i], list_y[i], i + 1) for i in range(nPe)]


# ----------------------------------------------
# Private functions do not touch
# ----------------------------------------------


def __Get_local_coords_and_dim(*args):
    # Get dim
    dim = len(args)
    assert dim in [1, 2, 3], "The number of lists in args must be 1, 2, or 3."

    # Get coordinates
    list_x = args[0]
    nPe = len(list_x)

    list_y = args[1] if dim > 1 else [0] * nPe
    assert (
        len(list_y) == nPe
    ), "The length of list_y must be equal to the length of list_x."

    list_z = args[2] if dim > 2 else [0] * nPe
    assert (
        len(list_z) == nPe
    ), "The length of list_z must be equal to the length of list_x."

    local_coords = np.array([list_x, list_y, list_z]).T

    return local_coords, dim


def __Get_shape_functions(
    polynom,
    local_coords: np.ndarray,
    dim: int,
    useSimplify=True,
    useFactor=True,
    useEulerBernoulli=False,
) -> list:
    nPe = local_coords.shape[0]

    nF = __Get_functions_per_node(polynom, local_coords, dim)

    if useEulerBernoulli:
        assert nF == 2, "euler bernoulli shape functions use 2 functions per node!"

    # construct matrix a
    matrix_A = __Get_matrix_A(polynom, local_coords, dim)

    # Get symbols and coords symbols
    symbols = sympy.symbols(f"x0:{nPe * nF}")

    functions = []

    for i in range(nPe * nF):
        # construct vector b
        vector_b = np.zeros(nPe * nF)
        if useEulerBernoulli:
            vector_b[i] = 1 if i % 2 == 0 else 1 / 2
            coefs = polynom(*coords[:dim])[0]
        else:
            vector_b[i] = 1
            coefs = polynom(*coords[:dim])

        # solve x from A x = b
        vector_x = np.linalg.solve(matrix_A, vector_b)
        # check that A x = b
        assert np.linalg.norm(matrix_A @ vector_x - vector_b) <= 1e-12

        # construct shape function
        function = sum(coef * term for coef, term in zip(symbols, coefs))
        # apply values
        function = function.subs({key: value for key, value in zip(symbols, vector_x)})

        # apply function display properties
        function = __chop(function)
        if useSimplify:
            function = function.nsimplify()
        if useFactor:
            function = function.factor()

        functions.append(function)

    return functions


def __Get_matrix_A(polynom, local_coords: np.ndarray, dim: int):
    nPe = local_coords.shape[0]

    nF = __Get_functions_per_node(polynom, local_coords, dim)
    indexes = np.arange(nPe * nF)
    if nF > 1:
        indexes = indexes.reshape(-1, 2)

    matrix_A = np.zeros((nPe * nF, nPe * nF))

    for n in range(nPe):
        matrix_A[indexes[n], :] = polynom(*local_coords[n, :dim])

    return matrix_A


def __Get_functions_per_node(polynom, local_coords: np.ndarray, dim: int) -> int:
    eval = np.array(polynom(*local_coords[0, :dim]))
    if len(eval.shape) == 2:
        return eval.shape[0]
    elif len(eval.shape) == 1:
        return 1
    else:
        raise Exception("polynom must be a function or a list of functions.")


def __Get_derivative_functions(functions, dim, order):
    assert isinstance(functions, list), "functions must be a list"
    assert dim in [1, 2, 3]
    assert order >= 1 and isinstance(order, int), "order must be >= 1"

    derivative_functions = []

    for function in functions:
        functions_per_dim = []

        # loop on dimensions
        for coord in coords[:dim]:
            func = function  # copy of `function`
            # loop to derive the func function
            for _ in range(order):
                func = func.diff(coord)
            functions_per_dim.append(func)

        derivative_functions.append(functions_per_dim)

    return derivative_functions


def __Print_functions(functions: list, dim: int, name="", printArray=False):
    # lamba string (e.g. lamda r, s)
    lambda_str = f"lambda {', '.join(str(coord) for coord in coords[:dim])}"

    # print each functions
    for i, function in enumerate(functions):
        if isinstance(function, list):
            print(
                f"{name}{i + 1} = [{', '.join(f'{lambda_str} : {func}' for func in function)}]"
            )
        else:
            print(f"{name}{i + 1} = {lambda_str} : {function}")

    print()
    if printArray:
        end = ".reshape(-1, 1)" if len(np.shape(functions)) == 1 else ""
        nF = len(functions)
        print(
            f"{name} = np.array([{', '.join(f'{name}{i + 1}' for i in range(nF))}]){end}\n"
        )


def __chop(expr):
    """Recursive function that replaces small values in the sympy expression with zero."""
    if isinstance(expr, sympy.Float) and abs(expr) < 1e-12:
        return sympy.S.Zero
    elif isinstance(expr, sympy.Add):
        return sympy.Add(*[__chop(arg) for arg in expr.args])
    elif isinstance(expr, sympy.Mul):
        return sympy.Mul(*[__chop(arg) for arg in expr.args])
    elif isinstance(expr, sympy.Pow):
        return sympy.Pow(__chop(expr.base), expr.exp)
    else:
        return expr


# ----------------------------------------------
# SEGMENTS
# ----------------------------------------------


def Do_Segments():
    # ----------------------------------------------
    # SEG 2
    # ----------------------------------------------
    #       v
    #       ^
    #       |
    #       |
    #  0----+----1 --> u
    # ----------------------------------------------

    name = "SEG2"
    Display.Section(name)

    list_x = [-1, 1]
    Plot_Nodes(name, list_x)

    polynom = lambda x: [x, 1]
    Compute_and_Print(polynom, list_x)

    Display.Section(name + " EulerBernoulli")
    polynom = lambda x: [[1, x, x**2, x**3], [0, 1, 2 * x, 3 * x**2]]
    Compute_and_Print(polynom, list_x, useEulerBernoulli=True)

    # ----------------------------------------------
    # SEG 3
    # ----------------------------------------------
    #       v
    #       ^
    #       |
    #       |
    #  0----2----1 --> u
    # ----------------------------------------------

    name = "SEG3"
    Display.Section(name)

    list_x = [-1, 1, 0]
    Plot_Nodes(name, list_x)

    polynom = lambda x: [x**2, x, 1]
    Compute_and_Print(polynom, list_x)

    Display.Section(name + " EulerBernoulli")
    polynom = lambda x: [
        [1, x, x**2, x**3, x**4, x**5],
        [0, 1, 2 * x, 3 * x**2, 4 * x**3, 5 * x**4],
    ]
    Compute_and_Print(polynom, list_x, useEulerBernoulli=True)

    # ----------------------------------------------
    # SEG 4
    # ----------------------------------------------
    #        v
    #        ^
    #        |
    #        |
    #  0---2-+-3---1 --> u
    # ----------------------------------------------

    name = "SEG4"
    Display.Section(name)

    list_x = [-1, 1, -1 / 3, 1 / 3]
    Plot_Nodes(name, list_x)

    polynom = lambda x: [x**3, x**2, x, 1]
    Compute_and_Print(polynom, list_x, useFactor=False)

    Display.Section(name + " EulerBernoulli")
    polynom = lambda x: [
        [1, x, x**2, x**3, x**4, x**5, x**6, x**7],
        [0, 1, 2 * x, 3 * x**2, 4 * x**3, 5 * x**4, 6 * x**5, 7 * x**6],
    ]
    Compute_and_Print(polynom, list_x, useEulerBernoulli=True, useFactor=False)

    # ----------------------------------------------
    # SEG 5
    # ----------------------------------------------
    #        v
    #        ^
    #        |
    #        |
    #  0--2--3--4--1 --> u
    # ----------------------------------------------

    name = "SEG5"
    Display.Section(name)

    list_x = [-1, 1, -1 / 2, 0, 1 / 2]
    Plot_Nodes(name, list_x)

    polynom = lambda x: [x**4, x**3, x**2, x, 1]

    Compute_and_Print(polynom, list_x, useFactor=False)

    Display.Section(name + " EulerBernoulli")
    polynom = lambda x: [
        [1, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9],
        [
            0,
            1,
            2 * x,
            3 * x**2,
            4 * x**3,
            5 * x**4,
            6 * x**5,
            7 * x**6,
            8 * x**7,
            9 * x**8,
        ],
    ]
    Compute_and_Print(polynom, list_x, useEulerBernoulli=True, useFactor=False)


# ----------------------------------------------
# TRIANGLES
# ----------------------------------------------


def Do_Triangles():
    # ----------------------------------------------
    # TRI3
    # ----------------------------------------------
    # v
    # ^
    # |
    # 2
    # |`\
    # |  `\
    # |    `\
    # |      `\
    # |        `\
    # 0----------1 --> u
    # ----------------------------------------------

    name = "TRI3"
    Display.Section(name)

    list_x = [0, 1, 0]
    list_y = [0, 0, 1]
    Plot_Nodes(name, list_x, list_y)

    polynom = lambda x, y: [x, y, 1]

    Compute_and_Print(polynom, list_x, list_y)

    # ----------------------------------------------
    # TRI6
    # ----------------------------------------------
    # v
    # ^
    # |
    # 2
    # |`\
    # |  `\
    # 5    `4
    # |      `\
    # |        `\
    # 0----3-----1 --> u
    # ----------------------------------------------

    name = "TRI6"
    Display.Section(name)

    list_x = [0, 1, 0, 0.5, 0.5, 0]
    list_y = [0, 0, 1, 0, 0.5, 0.5]
    Plot_Nodes(name, list_x, list_y)

    polynom = lambda x, y: [x**2, y**2, x * y, x, y, 1]

    Compute_and_Print(polynom, list_x, list_y)

    # ----------------------------------------------
    # TRI10
    # ----------------------------------------------
    # v
    # ^
    # |
    # 2
    # | \
    # 7   6
    # |     \
    # 8  (9)  5
    # |         \
    # 0---3---4---1 --> u
    # ----------------------------------------------

    name = "TRI10"
    Display.Section(name)

    # fmt: off
    list_x = [
        0,1,0,
        1/3,2/3,
        2/3, 1/3,
        0,0,
        1/3
    ]
    list_y = [
        0,0,1,
        0,0,
        1/3, 2/3,
        2/3,1/3,
        1/3
    ]
    # fmt: off
    Plot_Nodes(name, list_x, list_y)

    polynom = lambda x, y: [x**3, y**3, x**2 * y, x * y**2, x**2, y**2, x * y, x, y, 1]

    Compute_and_Print(polynom, list_x, list_y, useFactor=False)

    # ----------------------------------------------
    # TRI15
    # ----------------------------------------------
    #
    # 2
    # | \
    # 9   8
    # |     \
    # 10 (14)  7
    # |         \
    # 11 (12) (13) 6
    # |             \
    # 0---3---4---5---1
    # ----------------------------------------------

    name = "TRI15"
    Display.Section(name)

    # fmt: off
    list_x = [
        0,1,0,
        1/4,1/2,3/4,
        3/4,1/2,1/4,
        0,0,0,
        1/4,1/2,1/4
    ]
    list_y = [
        0,0,1,
        0,0,0,
        1/4,1/2,3/4,
        3/4,1/2,1/4,
        1/4,1/4,1/2
    ]
    # fmt: on
    Plot_Nodes(name, list_x, list_y)

    polynom = lambda x, y: [
        x**4,
        x**3 * y,
        x**2 * y**2,
        x * y**3,
        y**4,
        x**3,
        x**2 * y,
        x * y**2,
        y**3,
        x**2,
        x * y,
        y**2,
        x,
        y,
        1,
    ]

    Compute_and_Print(polynom, list_x, list_y, useFactor=False)


# ----------------------------------------------
# QUADRANGLES
# ----------------------------------------------


def Do_Quadrangles():
    # ----------------------------------------------
    # QUAD4
    # ----------------------------------------------
    #       v
    #       ^
    #       |
    # 3-----------2
    # |     |     |
    # |     |     |
    # |     +---- | --> u
    # |           |
    # |           |
    # 0-----------1
    # ----------------------------------------------

    name = "QUAD4"
    Display.Section(name)

    list_x = [-1, 1, 1, -1]
    list_y = [-1, -1, 1, 1]
    Plot_Nodes(name, list_x, list_y)

    polynom = lambda x, y: [x * y, x, y, 1]

    Compute_and_Print(polynom, list_x, list_y)

    # ----------------------------------------------
    # QUAD8
    # ----------------------------------------------
    #       v
    #       ^
    #       |
    # 3-----6-----2
    # |     |     |
    # |     |     |
    # 7     +---- 5 --> u
    # |           |
    # |           |
    # 0-----4-----1
    # ----------------------------------------------

    name = "QUAD8"
    Display.Section(name)

    list_x = [-1, 1, 1, -1, 0, 1, 0, -1]
    list_y = [-1, -1, 1, 1, -1, 0, 1, 0]
    Plot_Nodes(name, list_x, list_y)

    polynom = lambda x, y: [x**2 * y, y**2 * x, x**2, y**2, x * y, x, y, 1]

    Compute_and_Print(polynom, list_x, list_y, useFactor=True)

    # ----------------------------------------------
    # QUAD9
    # ----------------------------------------------
    #       v
    #       ^
    #       |
    # 3-----6-----2
    # |     |     |
    # |     |     |
    # 7     8---- 5 --> u
    # |           |
    # |           |
    # 0-----4-----1
    # ----------------------------------------------

    name = "QUAD9"
    Display.Section(name)

    list_x = [-1, 1, 1, -1, 0, 1, 0, -1, 0]
    list_y = [-1, -1, 1, 1, -1, 0, 1, 0, 0]
    Plot_Nodes(name, list_x, list_y)

    polynom = lambda x, y: [x**2 * y**2, x**2 * y, y**2 * x, x**2, y**2, x * y, x, y, 1]

    Compute_and_Print(polynom, list_x, list_y, useFactor=True)


# ----------------------------------------------
# TETRAHEDRON
# ----------------------------------------------


def Do_Tetrahedron():
    # ----------------------------------------------
    # TETRA4
    # ----------------------------------------------
    #                    v
    #                  .
    #                ,/
    #               /
    #            2
    #          ,/|`\
    #        ,/  |  `\
    #      ,/    '.   `\
    #    ,/       |     `\
    #  ,/         |       `\
    # 0-----------'.--------1 --> u
    #  `\.         |      ,/
    #     `\.      |    ,/
    #        `\.   '. ,/
    #           `\. |/
    #              `3
    #                 `\.
    #                    ` w
    # ----------------------------------------------

    name = "TETRA4"
    Display.Section(name)

    list_x = [0, 1, 0, 0]
    list_y = [0, 0, 1, 0]
    list_z = [0, 0, 0, 1]
    Plot_Nodes(name, list_x, list_y, list_z)

    polynom = lambda x, y, z: [x, y, z, 1]

    Compute_and_Print(polynom, list_x, list_y, list_z)

    # ----------------------------------------------
    # TETRA10
    # ----------------------------------------------
    #                    v
    #                  .
    #                ,/
    #               /
    #            2
    #          ,/|`\
    #        ,/  |  `\
    #      ,6    '.   `5
    #    ,/       8     `\
    #  ,/         |       `\
    # 0--------4--'.--------1 --> u
    #  `\.         |      ,/
    #     `\.      |    ,9
    #        `7.   '. ,/
    #           `\. |/
    #              `3
    #                 `\.
    #                    ` w
    # ----------------------------------------------

    name = "TETRA10"
    Display.Section(name)

    list_x = [0, 1, 0, 0, 0.5, 0.5, 0, 0, 0, 0.5]
    list_y = [0, 0, 1, 0, 0, 0.5, 0.5, 0, 0.5, 0]
    list_z = [0, 0, 0, 1, 0, 0, 0, 0.5, 0.5, 0.5]
    Plot_Nodes(name, list_x, list_y, list_z)

    polynom = lambda x, y, z: [x**2, y**2, z**2, x * y, x * z, y * z, x, y, z, 1]

    Compute_and_Print(polynom, list_x, list_y, list_z)


# ----------------------------------------------
# HEXAHEDRON
# ----------------------------------------------


def Do_Hexahedron():
    # ----------------------------------------------
    # HEXA8
    # ----------------------------------------------
    #        v
    # 3----------2
    # |\     ^   |\
    # | \    |   | \
    # |  \   |   |  \
    # |   7------+---6
    # |   |  +-- |-- | -> u
    # 0---+---\--1   |
    #  \  |    \  \  |
    #   \ |     \  \ |
    #    \|      w  \|
    #     4----------5
    # ----------------------------------------------

    name = "HEXA8"
    Display.Section(name)

    list_x = [-1, 1, 1, -1, -1, 1, 1, -1]
    list_y = [-1, -1, 1, 1, -1, -1, 1, 1]
    list_z = [-1, -1, -1, -1, 1, 1, 1, 1]
    Plot_Nodes(name, list_x, list_y, list_z)

    polynom = lambda x, y, z: [x * y * z, x * y, x * z, y * z, x, y, z, 1]

    Compute_and_Print(polynom, list_x, list_y, list_z)

    # ----------------------------------------------
    # HEXA20
    # ----------------------------------------------
    #        v
    # 3----13----2
    # |\     ^   |\
    # | 15   |   | 14
    # 9  \   |   11 \
    # |   7----19+---6
    # |   |  +-- |-- | -> u
    # 0---+-8-\--1   |
    #  \  17   \  \  18
    #  10 |     \  12|
    #    \|      w  \|
    #     4----16----5
    # ----------------------------------------------

    name = "HEXA20"
    Display.Section(name)

    # fmt: off
    list_x = [-1,1,1,-1,
            -1,1,1,-1,
            0,-1,-1,1,
            1,0,1,-1,
            0,-1,1,0]
    list_y = [-1,-1,1,1,
            -1,-1,1,1,
            -1,0,-1,0,
            -1,1,1,1,
            -1,0,0,1]
    list_z = [-1,-1,-1,-1,
            1,1,1,1,
            -1,-1,0,-1,
            0,-1,0,0,
            1,1,1,1]
    # fmt: on
    Plot_Nodes(name, list_x, list_y, list_z)

    polynom = lambda x, y, z: [
        x**2 * y * z,
        y**2 * x * z,
        z**2 * x * y,
        x**2 * y,
        y**2 * x,
        z**2 * x,
        x**2 * z,
        y**2 * z,
        z**2 * y,
        x**2,
        y**2,
        z**2,
        x * y,
        x * z,
        y * z,
        x,
        y,
        z,
        x * y * z,
        1,
    ]

    Compute_and_Print(polynom, list_x, list_y, list_z, useSimplify=True, useFactor=True)

    # ----------------------------------------------
    # HEXA27
    # ----------------------------------------------
    #
    # 3----13----2
    # |\         |\
    # |15    24  | 14
    # 9  \ 20    11 \
    # |   7----19+---6
    # |22 |  26  | 23|
    # 0---+-8----1   |
    #  \ 17    25 \  18
    #  10 |  21    12|
    #    \|         \|
    #     4----16----5
    # ----------------------------------------------

    name = "HEXA27"
    Display.Section(name)

    # fmt: off
    list_x = [
        -1,1,1,-1,
        -1,1,1,-1,
        0,-1,-1,1,
        1,0,1,-1,
        0,-1,1,0,
        0,0,-1,1,
        0,0,0
    ]
    list_y = [
        -1,-1,1,1,
        -1,-1,1,1,
        -1,0,-1,0,
        -1,1,1,1,
        -1,0,0,1,
        0,-1,0,0,
        1,0,0
    ]
    list_z = [
        -1,-1,-1,-1,
        1,1,1,1,
        -1,-1,0,-1,
        0,-1,0,0,
        1,1,1,1,
        -1,0,0,0,
        0,1,0
    ]
    # fmt: on
    Plot_Nodes(name, list_x, list_y, list_z)

    polynom = lambda x, y, z: [
        x**2 * z**2 * y,
        x**2 * y**2 * z,
        y**2 * z**2 * x,
        x**2 * z**2 * y**2,
        x**2 * y**2,
        x**2 * z**2,
        y**2 * z**2,
        x**2 * y * z,
        y**2 * x * z,
        z**2 * x * y,
        x**2 * y,
        y**2 * x,
        z**2 * x,
        x**2 * z,
        y**2 * z,
        z**2 * y,
        x**2,
        y**2,
        z**2,
        x * y,
        x * z,
        y * z,
        x,
        y,
        z,
        x * y * z,
        1,
    ]

    Compute_and_Print(polynom, list_x, list_y, list_z, useSimplify=True, useFactor=True)


# ----------------------------------------------
# PRISM
# ----------------------------------------------


def Do_Prism():
    # ----------------------------------------------
    # PRISM6
    # ----------------------------------------------
    #            w
    #            ^
    #            |
    #            3
    #          ,/|`\
    #        ,/  |  `\
    #      ,/    |    `\
    #     4------+------5
    #     |      |      |
    #     |    ,/|`\    |
    #     |  ,/  |  `\  |
    #     |,/    |    `\|
    #    ,|      |      |\
    #  ,/ |      0      | `\
    # u   |    ,/ `\    |    v
    #     |  ,/     `\  |
    #     |,/         `\|
    #     1-------------2
    # ----------------------------------------------

    name = "PRISM6"
    Display.Section(name)

    list_x = [0, 1, 0, 0, 1, 0]
    list_y = [0, 0, 1, 0, 0, 1]
    list_z = [-1, -1, -1, 1, 1, 1]
    Plot_Nodes(name, list_x, list_y, list_z)

    polynom = lambda x, y, z: [x * z, y * z, x, y, z, 1]

    Compute_and_Print(polynom, list_x, list_y, list_z)

    # ----------------------------------------------
    # PRISM15
    # ----------------------------------------------
    #            w
    #            ^
    #            |
    #            3
    #          ,/|`\
    #        12  |  13
    #      ,/    |    `\
    #     4------14-----5
    #     |      8      |
    #     |    ,/|`\    |
    #     |  ,/  |  `\  |
    #     |,/    |    `\|
    #    ,10     |      11
    #  ,/ |      0      | \
    # u   |    ,/ `\    |   v
    #     |  ,6     `7  |
    #     |,/         `\|
    #     1------9------2
    # ----------------------------------------------

    name = "PRISM15"
    Display.Section(name)

    list_x = [0, 1, 0, 0, 1, 0, 0.5, 0, 0, 0.5, 1, 0, 0.5, 0, 0.5]
    list_y = [0, 0, 1, 0, 0, 1, 0, 0.5, 0, 0.5, 0, 1, 0, 0.5, 0.5]
    list_z = [-1, -1, -1, 1, 1, 1, -1, -1, 0, -1, 0, 0, 1, 1, 1]
    Plot_Nodes(name, list_x, list_y, list_z)

    polynom = lambda x, y, z: [
        x**2 * z,
        y**2 * z,
        z**2 * x,
        z**2 * y,
        x * y * z,
        x**2,
        y**2,
        z**2,
        x * y,
        x * z,
        y * z,
        x,
        y,
        z,
        1,
    ]

    Compute_and_Print(polynom, list_x, list_y, list_z)

    # ----------------------------------------------
    # PRISM18
    # ----------------------------------------------
    #            w
    #            ^
    #            |
    #            3
    #          ,/|`\
    #        12  |  13
    #      ,/    |    `\
    #     4------14-----5
    #     |      8      |
    #     |    ,/|`\    |
    #     |  15  |  16  |
    #     |,/    |    `\|
    #    ,10-----17-----11
    #  ,/ |      0      | \
    # u   |    ,/ `\    |   v
    #     |  ,6     `7  |
    #     |,/         `\|
    #     1------9------2
    # ----------------------------------------------

    name = "PRISM18"
    Display.Section(name)

    list_x = [0, 1, 0, 0, 1, 0, 0.5, 0, 0, 0.5, 1, 0, 0.5, 0, 0.5, 0.5, 0, 0.5]
    list_y = [0, 0, 1, 0, 0, 1, 0, 0.5, 0, 0.5, 0, 1, 0, 0.5, 0.5, 0, 0.5, 0.5]
    list_z = [-1, -1, -1, 1, 1, 1, -1, -1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0]
    Plot_Nodes(name, list_x, list_y, list_z)

    polynom = lambda x, y, z: [
        x**2,
        y**2,
        x * y,
        x,
        y,
        1,
        x**2 * z,
        y**2 * z,
        x * y * z,
        x * z,
        y * z,
        z,
        x**2 * z**2,
        y**2 * z**2,
        x * y * z**2,
        x * z**2,
        y * z**2,
        z**2,
    ]

    Compute_and_Print(polynom, list_x, list_y, list_z)


# ----------------------------------------------
# MAIN
# ----------------------------------------------

if __name__ == "__main__":
    Do_Segments()

    Do_Triangles()

    Do_Quadrangles()

    Do_Tetrahedron()

    Do_Hexahedron()

    Do_Prism()

    plt.show()
