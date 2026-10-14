# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""A smooth synthetic image and its rigid translation by (UX, UY) pixels: the exact displacement is known, recovered to the spline interpolation and the finite-difference image gradient, ~3e-3."""

import numpy as np
import pytest

from EasyFEA import ElemType, Mesh, Simulations
from EasyFEA.Geoms import Domain

from _numerics import rel_error

N = 64
UX, UY = 0.4, -0.3
MESH_SIZE = 8.0
TOL = 5e-3


def _pattern(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (
        100
        + 40 * np.sin(2 * np.pi * x / 13) * np.cos(2 * np.pi * y / 11)
        + 30 * np.cos(2 * np.pi * (x + y) / 17)
    )


def _images() -> tuple[np.ndarray, np.ndarray]:
    """Reference f and deformed g(x) = f(x - u), so that f(x) = g(x + u)."""
    y, x = np.mgrid[0:N, 0:N].astype(float)
    return _pattern(x, y), _pattern(x - UX, y - UY)


def _mesh() -> Mesh:
    domain = Domain((8, 8), (56, 56), MESH_SIZE)
    return domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)


def _dic(lr: float = 0.0) -> Simulations.DIC:
    return Simulations.DIC(_mesh(), _images()[0], lr)


def _uniform(mesh: Mesh, ux: float, uy: float) -> np.ndarray:
    return np.tile([ux, uy], mesh.Nn)


def test_DIC_rejects_a_3d_mesh():
    domain = Domain((8, 8), (56, 56), MESH_SIZE)
    mesh = domain.Mesh_Extrude([], [0, 0, 8], [1], ElemType.HEXA8)

    with pytest.raises(AssertionError, match="2D"):
        Simulations.DIC(mesh, _images()[0])


def test_DIC_rejects_a_mesh_change():
    dic = _dic()

    with pytest.raises(Exception, match="modifications to the mesh"):
        dic.mesh.Translate(dx=1)


def test_DIC_mesh_is_the_given_mesh():
    mesh = _mesh()

    assert Simulations.DIC(mesh, _images()[0]).mesh is mesh


def test_DIC_ldic_is_eight_mean_element_sizes():
    assert _dic().ldic == 8 * MESH_SIZE


def test_DIC__Get_ldic_scales_the_mean_element_size():
    dic = _dic()

    assert dic._Get_ldic(2.0) == 2 * MESH_SIZE
    with pytest.raises(AssertionError):
        dic._Get_ldic(0.0)


def test_DIC_Get_scaled_mesh_scales_a_free_copy():
    dic = _dic()

    scaled = dic.Get_scaled_mesh(0.5)
    scaled.Translate(dx=1)  # free of the DIC, so no error

    assert rel_error(scaled.coord[:, :2] - [1, 0], 0.5 * dic.mesh.coord[:, :2]) < 1e-15
    assert dic.mesh.coord[:, 0].min() == 8


def test_DIC__imgRef_is_a_copy_of_the_reference_image():
    dic = _dic()

    imgRef = dic._imgRef
    imgRef[:] = 0

    assert np.array_equal(dic._imgRef, _images()[0])


def test_DIC__imgRef_rejects_what_is_not_an_array():
    dic = _dic()

    with pytest.raises(AssertionError, match="numpy"):
        dic._imgRef = [[0.0]]


def test_DIC_shape_is_the_reference_image_shape():
    assert _dic().shape == (N, N)


def test_DIC__lr_sets_alpha_and_the_regularized_operator():
    dic = _dic()
    M_dic = dic._M_dic.toarray()

    dic._lr = 16.0

    assert dic.alpha == (16.0 / dic.ldic) ** 2
    w = dic._Get_w()
    expected = M_dic / (w @ M_dic @ w) + dic.alpha * dic._R / (w @ dic._R @ w)
    assert rel_error(dic._M_reg.toarray(), expected) < 1e-12
    with pytest.raises(AssertionError, match="lr"):
        dic._lr = -1.0


def test_DIC_alpha_is_zero_without_regularization():
    assert _dic().alpha == 0.0


def test_DIC__Get_coords_lists_the_pixels_row_by_row():
    img = np.zeros((2, 3))

    coords = Simulations.DIC._Get_coords(img)

    assert coords.tolist() == [
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [2, 1, 0],
    ]


def test_DIC_roi_flags_the_pixels_inside_the_mesh():
    """An element holds its pixels on [min, max) along each axis."""
    dic = _dic()
    coords = Simulations.DIC._Get_coords(_images()[0])
    inside = (coords[:, :2] >= 8).all(1) & (coords[:, :2] < 56).all(1)

    assert np.array_equal(dic.roi, inside)


def test_DIC_ROI_is_the_roi_as_an_image():
    dic = _dic()

    assert np.array_equal(dic.ROI, dic.roi.reshape(N, N))
    assert dic.ROI[8:56, 8:56].all() and dic.ROI.sum() == 48**2


def test_DIC__Get_w_is_a_plane_wave_on_both_components():
    dic = _dic()
    x, y = dic.mesh.coord[:, 0], dic.mesh.coord[:, 1]

    w = dic._Get_w()

    expected = np.cos(2 * np.pi * x / dic.ldic) * np.sin(2 * np.pi * y / dic.ldic)
    assert np.array_equal(w, expected.repeat(2))


def test_DIC__Compute_L_M_projects_the_image_gradient():
    """For a uniform nodal field, L·u is the image gradient along u summed over the roi."""
    dic = _dic()
    f = _images()[0]

    dic._Compute_L_M(f)

    gradY, gradX = np.gradient(f)
    u = _uniform(dic.mesh, 1.0, 0.0)
    assert rel_error(u @ dic._L, np.where(dic.roi, gradX.ravel(), 0)) < 1e-12
    M = dic._M_dic.toarray()
    assert rel_error(M, M.T) < 1e-15


def test_DIC_Calc_pixelDisplacement_of_a_uniform_field_is_uniform():
    dic = _dic()

    ux_p, uy_p = dic.Calc_pixelDisplacement(_uniform(dic.mesh, UX, UY))

    assert rel_error(ux_p[dic.roi], UX) < 1e-12
    assert rel_error(uy_p[dic.roi], UY) < 1e-12
    assert np.all(ux_p[~dic.roi] == 0)


def test_DIC_Calc_pixelDisplacement_rejects_a_wrong_size():
    dic = _dic()

    with pytest.raises(AssertionError, match="wrong size"):
        dic.Calc_pixelDisplacement(np.zeros(3))


def test_DIC_Calc_r_dic_vanishes_at_the_translation():
    """Only the spline interpolation of g is left."""
    dic = _dic()
    f, g = _images()

    r = dic.Calc_r_dic(_uniform(dic.mesh, UX, UY), g)

    assert r.shape == (N, N)
    assert np.abs(r[dic.ROI]).max() < 1e-2 * np.abs(f - g).max()
    assert rel_error(dic.Calc_r_dic(0 * _uniform(dic.mesh, 1, 1), g), f - g) < 1e-12


def test_DIC_Solve_recovers_the_rigid_translation(subtests):
    """A rigid field has no Laplacian, so the regularization leaves it alone."""
    g = _images()[1]
    for lr in (0.0, 16.0):
        with subtests.test(lr=lr):
            dic = _dic(lr)

            u = dic.Solve(g, np.zeros(dic.mesh.Nn * 2), verbosity=False)

            assert rel_error(u, _uniform(dic.mesh, UX, UY)) < TOL


def test_DIC_Solve_from_zero_without_opencv():
    dic = _dic()
    g = _images()[1]
    if Simulations._dic.CAN_USE_CV2:
        pytest.skip("starts from the optical flow")

    u = dic.Solve(g, verbosity=False)

    assert rel_error(u, _uniform(dic.mesh, UX, UY)) < TOL


def test_DIC_Solve_rejects_identical_images():
    dic = _dic()

    with pytest.raises(AssertionError, match="identical"):
        dic.Solve(_images()[0], verbosity=False)


def test_DIC_Solve_rejects_a_wrong_u0_or_image():
    dic = _dic()
    g = _images()[1]

    with pytest.raises(AssertionError, match="u0"):
        dic.Solve(g, np.zeros(3), verbosity=False)
    with pytest.raises(AssertionError, match="shape"):
        dic.Solve(g[1:], verbosity=False)
    with pytest.raises(AssertionError, match="shape"):
        dic.Solve(g, imgRef=g[1:], verbosity=False)


def test_DIC_Solve_raises_when_it_does_not_converge():
    dic = _dic()

    with pytest.raises(Exception, match="did not converge"):
        dic.Solve(_images()[1], np.zeros(dic.mesh.Nn * 2), iterMax=2, verbosity=False)


def test_DIC__Get_u_from_images_projects_the_optical_flow():
    pytest.importorskip("cv2")
    dic = _dic()
    f, g = _images()

    u0 = dic._Get_u_from_images(f, g)

    assert rel_error(u0, _uniform(dic.mesh, UX, UY)) < 0.5


def test_DIC_Save_keeps_the_analysis_usable(tmp_path):
    dic = _dic()
    g = _images()[1]

    dic.Save(str(tmp_path))

    u = dic.Solve(g, np.zeros(dic.mesh.Nn * 2), verbosity=False)
    assert rel_error(u, _uniform(dic.mesh, UX, UY)) < TOL


def test_Load_DIC_restores_a_usable_analysis(tmp_path):
    dic = _dic(16.0)
    g = _images()[1]
    dic.Save(str(tmp_path), "run")

    loaded = Simulations.Load_DIC(str(tmp_path), "run")

    assert loaded._lr == 16.0 and np.array_equal(loaded.roi, dic.roi)
    u = loaded.Solve(g, np.zeros(dic.mesh.Nn * 2), verbosity=False)
    assert rel_error(u, _uniform(dic.mesh, UX, UY)) < TOL


def test_Load_DIC_rejects_a_missing_file(tmp_path):
    with pytest.raises(Exception, match="does not exist"):
        Simulations.Load_DIC(str(tmp_path))


def _disk(xc: float, yc: float, radius: float) -> np.ndarray:
    y, x = np.mgrid[0:N, 0:N].astype(float)
    return np.where((x - xc) ** 2 + (y - yc) ** 2 <= radius**2, 0.0, 255.0)


def test_DIC_Get_Circle_finds_a_dark_disk():
    XC, YC, radius = Simulations.DIC.Get_Circle(_disk(30, 20, 8), threshold=100)

    assert (XC, YC) == (30, 20)
    assert radius == 8


def test_DIC_Get_Circle_looks_only_within_the_boundary():
    img = np.minimum(_disk(15, 15, 5), _disk(45, 45, 6))

    XC, YC, radius = Simulations.DIC.Get_Circle(
        img, threshold=100, boundary=((30, 63), (30, 63)), radiusCoef=2.0
    )

    assert (XC, YC, radius) == (45, 45, 12)
