# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA.fem import FeArray
import numpy as np


def _check_arrays(array1, array2):
    if isinstance(array1, FeArray):
        array1 = np.asarray(array1)
    if isinstance(array2, FeArray):
        array2 = np.asarray(array2)
    norm_diff = np.linalg.norm(array1 - array2)
    assert norm_diff < 1e-12


def _check_ValueError(func):
    valueErrorDetected = False
    try:
        func()
    except ValueError:
        valueErrorDetected = True
    assert valueErrorDetected


def _check_operation(
    array1, op, array2, reshape1=(), reshape2=(), willTriggerError=False
):

    if reshape1 == ():
        reshape1 = np.shape(array1)
    reshaped1 = np.asarray(array1).reshape(reshape1)

    if reshape2 == ():
        reshape2 = np.shape(array2)
    reshaped2 = np.asarray(array2).reshape(reshape2)

    try:
        if op == "+":
            computed = array1 + array2
            excepted = reshaped1 + reshaped2
        elif op == "-":
            computed = array1 - array2
            excepted = reshaped1 - reshaped2
        elif op == "*":
            computed = array1 * array2
            excepted = reshaped1 * reshaped2
        elif op == "/":
            computed = array1 / array2
            excepted = reshaped1 / reshaped2
        else:
            raise Exception("unknown operator")
        _check_arrays(computed, excepted)
    except ValueError:
        if willTriggerError:
            pass
        else:
            # should not be here
            raise ValueError("ValueError detected")


def FeArrays():

    Ne, nPg = 1000, 4
    scalar_e_pg = FeArray.asfearray(np.random.randint(1, 10, (Ne, nPg)) * 0.1)
    vector_e_pg = FeArray.asfearray(np.random.randint(1, 10, (Ne, nPg, 3)) * 0.1)
    matrix_e_pg = FeArray.asfearray(np.random.randint(1, 10, (Ne, nPg, 3, 3)) * 0.1)
    tensor_e_pg = FeArray.asfearray(
        np.random.randint(1, 10, (Ne, nPg, 3, 3, 3, 3)) * 0.1
    )

    return [scalar_e_pg, vector_e_pg, matrix_e_pg, tensor_e_pg]


def do_operation(op: str):

    scalar_e_pg, vector_e_pg, matrix_e_pg, tensor_e_pg = FeArrays()

    Ne, nPg = scalar_e_pg.shape

    scalar = np.random.randint(1, 10)
    vector = np.random.randint(1, 10, (3)) * 0.1
    matrix = np.random.randint(1, 10, (3, 3)) * 0.1
    tensor = np.random.randint(1, 10, (3, 3, 3, 3)) * 0.1

    # scalar + ...
    _check_operation(scalar_e_pg, op, scalar)
    _check_operation(scalar_e_pg, op, vector, (Ne, nPg, 1))
    _check_operation(scalar_e_pg, op, matrix, (Ne, nPg, 1, 1))
    _check_operation(scalar_e_pg, op, tensor, (Ne, nPg, 1, 1, 1, 1))

    # ... + scalar
    _check_operation(scalar, op, scalar_e_pg)
    _check_operation(scalar, op, vector_e_pg)
    _check_operation(scalar, op, matrix_e_pg)
    _check_operation(scalar, op, tensor_e_pg)

    # vector + ...
    _check_operation(vector_e_pg, op, scalar)
    _check_operation(vector_e_pg, op, vector)
    _check_operation(vector_e_pg, op, matrix, willTriggerError=True)
    _check_operation(vector_e_pg, op, tensor, willTriggerError=True)

    # ... + vector
    _check_operation(scalar, op, vector_e_pg)
    _check_operation(vector, op, vector_e_pg)
    _check_operation(matrix, op, vector_e_pg, willTriggerError=True)
    _check_operation(tensor, op, vector_e_pg, willTriggerError=True)

    # matrix + ...
    _check_operation(matrix_e_pg, op, scalar)
    _check_operation(matrix_e_pg, op, vector, willTriggerError=True)
    _check_operation(matrix_e_pg, op, matrix)
    _check_operation(matrix_e_pg, op, tensor, willTriggerError=True)

    # ... + matrix
    _check_operation(scalar, op, matrix_e_pg)
    _check_operation(vector, op, matrix_e_pg, willTriggerError=True)
    _check_operation(matrix, op, matrix_e_pg)
    _check_operation(tensor, op, matrix_e_pg, willTriggerError=True)

    # tensor + ...
    _check_operation(tensor_e_pg, op, scalar)
    _check_operation(tensor_e_pg, op, vector, willTriggerError=True)
    _check_operation(tensor_e_pg, op, matrix, willTriggerError=True)
    _check_operation(tensor_e_pg, op, tensor)

    # ... + matrix
    _check_operation(scalar, op, tensor_e_pg)
    _check_operation(vector, op, tensor_e_pg, willTriggerError=True)
    _check_operation(matrix, op, tensor_e_pg, willTriggerError=True)
    _check_operation(tensor, op, tensor_e_pg)


class TestFeArray:

    def test_new_array(self):

        try:
            FeArray([0, 1])
        except ValueError:
            pass

        scalar_e_pg, vector_e_pg, matrix_e_pg, tensor_e_pg = FeArrays()

        assert scalar_e_pg._idx == ""
        assert vector_e_pg._idx == "i"
        assert matrix_e_pg._idx == "ij"
        assert tensor_e_pg._idx == "ijkl"

        assert scalar_e_pg._type == "scalar"
        assert vector_e_pg._type == "vector"
        assert matrix_e_pg._type == "matrix"
        assert tensor_e_pg._type == "tensor"

        _check_ValueError(lambda: FeArray(0, False))
        _check_ValueError(lambda: FeArray([0], False))

    def test_add_array(self):

        do_operation("+")

    def test_sub_array(self):

        do_operation("-")

    def test_mul_array(self):

        do_operation("*")

    def test_trudiv_array(self):

        do_operation("/")

    def test_T(self):

        scalar_e_pg, vector_e_pg, matrix_e_pg, tensor_e_pg = FeArrays()

        _check_arrays(scalar_e_pg.T, scalar_e_pg)
        _check_arrays(vector_e_pg.T, vector_e_pg)
        _check_arrays(matrix_e_pg.T, matrix_e_pg.transpose((0, 1, 3, 2)))
        _check_arrays(tensor_e_pg.T, tensor_e_pg.transpose((0, 1, 5, 4, 3, 2)))

    def test_dot(self):

        _, vector_e_pg, matrix_e_pg, tensor_e_pg = FeArrays()

        # Avoid testing scalars, as this holds no significance.

        # i i
        _check_arrays(
            vector_e_pg.dot(vector_e_pg),
            np.einsum("...i,...i->...", vector_e_pg, vector_e_pg),
        )
        # i ij
        _check_arrays(
            vector_e_pg.dot(matrix_e_pg),
            np.einsum("...i,...ij->...j", vector_e_pg, matrix_e_pg),
        )
        # i ijkl
        _check_arrays(
            vector_e_pg.dot(tensor_e_pg),
            np.einsum("...i,...ijkl->...jkl", vector_e_pg, tensor_e_pg),
        )

        # ij j
        _check_arrays(
            matrix_e_pg.T.dot(vector_e_pg),
            np.einsum("...ji,...j->...i", matrix_e_pg, vector_e_pg),
        )
        # ij jk
        _check_arrays(
            matrix_e_pg.T.dot(matrix_e_pg),
            np.einsum("...ji,...jk->...ik", matrix_e_pg, matrix_e_pg),
        )
        # ij jklm
        _check_arrays(
            matrix_e_pg.T.dot(tensor_e_pg),
            np.einsum("...ji,...jklm->...iklm", matrix_e_pg, tensor_e_pg),
        )

        # ijkl l
        _check_arrays(
            tensor_e_pg.dot(vector_e_pg),
            np.einsum("...ijkl,...l->...ijk", tensor_e_pg, vector_e_pg),
        )
        # ijkl lm
        _check_arrays(
            tensor_e_pg.dot(matrix_e_pg),
            np.einsum("...ijkl,...lm->...ijkm", tensor_e_pg, matrix_e_pg),
        )
        # ijkl lmno
        _check_arrays(
            tensor_e_pg.dot(tensor_e_pg),
            np.einsum("...ijkl,...lmno->...ijkmno", tensor_e_pg, tensor_e_pg),
        )

    def test_matmul(self):

        _, vector_e_pg, matrix_e_pg, tensor_e_pg = FeArrays()

        # Avoid testing scalars, as this holds no significance.

        # i i
        _check_arrays(
            vector_e_pg @ vector_e_pg,
            np.einsum("...i,...i->...", vector_e_pg, vector_e_pg),
        )
        # i ij
        _check_arrays(
            vector_e_pg @ matrix_e_pg,
            np.einsum("...i,...ij->...j", vector_e_pg, matrix_e_pg),
        )
        # i ijkl
        _check_arrays(
            vector_e_pg @ tensor_e_pg,
            np.einsum("...i,...ijkl->...jkl", vector_e_pg, tensor_e_pg),
        )

        # ij j
        _check_arrays(
            matrix_e_pg.T @ vector_e_pg,
            np.einsum("...ji,...j->...i", matrix_e_pg, vector_e_pg),
        )
        # ij jk
        _check_arrays(
            matrix_e_pg.T @ matrix_e_pg,
            np.einsum("...ji,...jk->...ik", matrix_e_pg, matrix_e_pg),
        )
        # ij jklm
        _check_arrays(
            matrix_e_pg.T @ tensor_e_pg,
            np.einsum("...ji,...jklm->...iklm", matrix_e_pg, tensor_e_pg),
        )

        # ijkl l
        _check_arrays(
            tensor_e_pg @ vector_e_pg,
            np.einsum("...ijkl,...l->...ijk", tensor_e_pg, vector_e_pg),
        )
        # ijkl lm
        _check_arrays(
            tensor_e_pg @ matrix_e_pg,
            np.einsum("...ijkl,...lm->...ijkm", tensor_e_pg, matrix_e_pg),
        )
        # ijkl lmno
        _check_arrays(
            tensor_e_pg @ tensor_e_pg,
            np.einsum("...ijkl,...lmno->...ijkmno", tensor_e_pg, tensor_e_pg),
        )

    def test_ddot(self):

        _, vector_e_pg, matrix_e_pg, tensor_e_pg = FeArrays()

        # Avoid testing scalars, as this holds no significance.

        # i i
        _check_ValueError(lambda: vector_e_pg.ddot(vector_e_pg))  # wrong dimensions
        # i ij
        _check_ValueError(lambda: vector_e_pg.ddot(matrix_e_pg))  # wrong dimensions
        # i ijkl
        _check_ValueError(lambda: vector_e_pg.ddot(tensor_e_pg))  # wrong dimensions

        # ij i
        _check_ValueError(lambda: matrix_e_pg.ddot(vector_e_pg))
        # ij ij
        _check_arrays(
            matrix_e_pg.ddot(matrix_e_pg),
            np.einsum("...ij,...ij->...", matrix_e_pg, matrix_e_pg),
        )
        # ij ijkl
        _check_arrays(
            matrix_e_pg.ddot(tensor_e_pg),
            np.einsum("...ij,...ijkl->...kl", matrix_e_pg, tensor_e_pg),
        )

        # ijkl l
        _check_ValueError(lambda: tensor_e_pg.ddot(vector_e_pg))  # wrong dimensions
        # ijkl kl
        _check_arrays(
            tensor_e_pg.ddot(matrix_e_pg),
            np.einsum("...ijkl,...kl->...ij", tensor_e_pg, matrix_e_pg),
        )
        # ijkl lmno
        _check_arrays(
            tensor_e_pg.ddot(tensor_e_pg),
            np.einsum("...ijkl,...klmn->...ijmn", tensor_e_pg, tensor_e_pg),
        )
