# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""DIC analysis module"""

import numpy as np
from scipy import interpolate, sparse
from scipy.sparse.linalg import splu
import pickle
import cv2  # need opencv-python library
from typing import Optional

# utilities
from ..Utilities import Tic, Folder, Display
from ..Utilities._observers import Observable, _IObserver
from ..Utilities import _types

# fem
from ..FEM import Mesh, BoundaryCondition, FeArray, MatrixType


class DIC(_IObserver):
    def __init__(
        self,
        mesh: Mesh,
        idxImgRef: int,
        imgRef: _types.FloatArray,
        lr: float = 0.0,
        forces: Optional[_types.FloatArray] = None,
        displacements: Optional[_types.FloatArray] = None,
        verbosity=False,
    ):
        """Creates a DIC analysis.

        Parameters
        ----------
        mesh : Mesh
            pixel-based mesh used for dic.
        idxImgRef : int
            reference image index in _forces (or in the folder).
        imgRef : _types.FloatArray
            reference image (f)
        lr : float | int, optional
            regularization length, by default 0.0
        forces : _types.FloatArray, optional
            forces measured during the tests, by default None
        displacements : _types.FloatArray, optional
            displacements measured during the tests, by default None
        verbosity : bool, optional
            can write in the terminal, by default False

        Returns
        -------
        DIC
            DIC object
        """

        # mesh
        assert mesh.dim == 2, "Must be a 2D mesh."
        mesh._Add_observer(self)
        self.__mesh = mesh

        # images
        self.__idxImgRef: int = idxImgRef
        assert isinstance(imgRef, np.ndarray), "Must be a numpy array."
        self.__imgRef = imgRef

        # data
        self._forces = forces
        """forces measured during the tests."""
        self._displacements = displacements
        """displacements measured during the tests."""

        # results
        self.__list_idx: list[int] = []
        self.__list_u: list[_types.FloatArray] = []
        self.__list_img: list[_types.FloatArray] = []

        self._verbosity: bool = verbosity

        # initialize ROI and shape functions and shape function derivatives
        self.__init__roi()
        self.__init__Phi_opLap()

        # regul
        self._lr = lr
        # Updating self._lr will automatically update the matrices
        # That's why we can comment on the following line
        # self.Compute_L_M(imgRef)

    # mesh properties

    @property
    def mesh(self) -> Mesh:
        """pixel-based mesh used for dic."""
        return self.__mesh

    @property
    def ldic(self) -> float:
        """8 * mean(meshSize) (see Get_ldic())"""
        return self.Get_ldic()

    def Get_scaled_mesh(self, scale: float = 1.0) -> Mesh:
        assert scale != 0.0
        meshC = self.__mesh.copy()
        [meshC._Remove_observer(observer) for observer in meshC.observers.copy()]
        meshC.coord = meshC.coord * scale
        return meshC

    # image properties

    @property
    def idxImgRef(self) -> int:
        """reference image index in _forces (or in the folder)."""
        return self.__idxImgRef

    @property
    def imgRef(self) -> _types.FloatArray:
        """reference image (f)"""
        return self.__imgRef.copy()

    @property
    def shape(self) -> tuple[int, int]:
        """reference image shape"""
        return self.__imgRef.shape  # type: ignore [return-value]

    # regularization properties

    @property
    def _lr(self) -> float:
        """regularization length"""
        return self.__lr

    @_lr.setter
    def _lr(self, value: float) -> None:
        """
        WARNING
        -------
        Changing this parameter will automatically update the matrices with the _Compute_L_M() function!
        """
        assert value >= 0.0, "lr must be >= 0.0"
        self.__lr = value
        self._Compute_L_M(self.__imgRef)

    @property
    def alpha(self) -> float:
        return (self._lr / self.ldic) ** 2

    # solution properties

    @property
    def list_idx(self) -> list[int]:
        """copy of the list containing indexes for which the displacement field has been calculated."""
        return self.__list_idx.copy()

    @property
    def list_u(self) -> list[_types.FloatArray]:
        """copy of the list containing the calculated displacement fields."""
        return self.__list_u.copy()

    @property
    def list_img(self) -> list[_types.FloatArray]:
        """copy of the list containing images for which the displacement field has been calculated."""
        return self.__list_img.copy()

    def _Update(self, observable: Observable, event: str) -> None:
        if isinstance(observable, Mesh):
            raise Exception(
                "The current implementation does not allow you to make any modifications to the mesh."
            )
        else:
            Display.MyPrintError("Notification not yet implemented")

    def __init__roi(self) -> None:
        """Initializes the Region of Interest (ROI)"""

        tic = Tic()

        imgRef = self.__imgRef
        mesh = self.__mesh

        # get pixels' coordinates
        coordPx = (
            np.arange(imgRef.shape[1])
            .reshape((1, -1))
            .repeat(imgRef.shape[0], 0)
            .ravel()
        )
        coordPy = (
            np.arange(imgRef.shape[0]).reshape((-1, 1)).repeat(imgRef.shape[1]).ravel()
        )
        coordPixel = np.zeros((coordPx.shape[0], 3), dtype=int)
        coordPixel[:, 0] = coordPx
        coordPixel[:, 1] = coordPy

        # get pixels used in elements with their coordinates
        pixels, _, connectPixel, coordPixelInElem = mesh.groupElem.Get_Mapping(
            coordPixel, needCoordinates=True
        )
        # mean_pixels = np.mean([connectPixel[e].size for e in range(mesh.Ne)])

        self.__connectPixel = connectPixel
        """connectivity matrix linking the pixels used for each element."""
        self.__coordPixelInElem: _types.FloatArray = coordPixelInElem  # type: ignore
        """pixel coordinates in the reference element (xi, eta)."""

        # create roi (as a vector)
        roi = np.zeros(coordPx.shape[0], dtype=int)
        roi[pixels] = 1
        self.__roi = np.asarray(roi == 1, dtype=bool)

        tic.Tac("DIC", "ROI", self._verbosity)

    @property
    def roi(self) -> _types.IntArray:
        """roi as a vector."""
        return self.__roi.copy()

    @property
    def ROI(self) -> _types.IntArray:
        """roi as a matrix."""
        return self.roi.reshape(self.shape)

    def __init__Phi_opLap(self) -> None:
        """Initializes shape functions and the Laplacian operator."""

        mesh = self.__mesh
        dim = 2
        Ndof = self.__mesh.Nn * dim

        connectPixel = self.__connectPixel
        coordInElem = self.__coordPixelInElem
        Ntild = mesh.groupElem._N()

        # ----------------------------------------------
        # Build the shape function matrix for pixels (N)
        # ----------------------------------------------
        lines_x: list[int] = []
        lines_y: list[int] = []
        columns_Phi: list[int] = []
        values_phi: list[float] = []

        # Evaluate shape functions for each pixels' coordinates
        x_p, y_p = coordInElem[:, 0], coordInElem[:, 1]
        phi_n_pixels = np.array(
            [np.reshape([Ntild[n, 0](x_p, y_p)], -1) for n in range(mesh.nPe)]
        )

        tic = Tic()

        # Possible without the loop?
        # No, it is not possible without the loop because connectPixel doesn't have the same number of columns in each row.
        # In addition, if you remove it, you'll have to make several list comprehension.
        for e in range(mesh.Ne):
            # Get the nodes and pixels used by the element
            nodes = mesh.connect[e]
            pixels = connectPixel[e]
            # Retrieve evaluated functions
            phi = phi_n_pixels[:, pixels]

            # line construction
            linesX = (
                BoundaryCondition.Get_dofs_nodes(["x", "y"], nodes, ["x"])
                .reshape(-1, 1)
                .repeat(pixels.size)
            )
            # linesY = BoundaryCondition.Get_dofs_nodes(["x","y"], nodes, ["y"]).reshape(-1,1).repeat(pixels.size)
            # same as
            linesY = linesX + 1
            # get columns in which for placing values
            columns = pixels.reshape(1, -1).repeat(mesh.nPe, 0).ravel()

            lines_x.extend(linesX.tolist())
            lines_y.extend(linesY.tolist())
            columns_Phi.extend(columns)
            values_phi.extend(phi.ravel().tolist())

        self._N_x = sparse.csr_matrix(
            (values_phi, (lines_x, columns_Phi)), (Ndof, coordInElem.shape[0])
        )
        """Shape function matrix Nx (Ndof, nPixels)"""
        self._N_y = sparse.csr_matrix(
            (values_phi, (lines_y, columns_Phi)), (Ndof, coordInElem.shape[0])
        )
        """Shape function matrix Ny (Ndof, nPixels)"""

        Op: sparse.csr_matrix = self._N_x @ self._N_x.T + self._N_y @ self._N_y.T

        self.__Op_LU = splu(Op.tocsc())

        tic.Tac("DIC", "N_x and N_y", self._verbosity)

        # ----------------------------------------------
        # Build the Laplacian operator (R)
        # ----------------------------------------------
        matrixType = MatrixType.mass
        wJ_e_pg = mesh.Get_weightedJacobian_e_pg(matrixType)  # (Ne, nPg)
        dN_e_pg = mesh.Get_dN_e_pg(matrixType)  # (Ne, nPg, dim, nPe)

        dNdx = dN_e_pg[:, :, 0]
        dNdy = dN_e_pg[:, :, 1]

        ind_x = np.arange(0, mesh.nPe * dim, dim)
        ind_y = ind_x + 1

        dN_x = FeArray.zeros(*dN_e_pg.shape[:2], 2, 2 * mesh.nPe)
        dN_y = np.zeros_like(dN_x)

        dN_x[:, :, 0, ind_x] = dNdx
        dN_x[:, :, 1, ind_y] = dNdx
        Bx_e = (wJ_e_pg * dN_x.T @ dN_x).sum(1)

        dN_y[:, :, 0, ind_x] = dNdy
        dN_y[:, :, 1, ind_y] = dNdy
        By_e = (wJ_e_pg * dN_y.T @ dN_y).sum(1)

        B_e = Bx_e + By_e

        rows = mesh.Get_rows_e(dim).ravel()
        columns = mesh.Get_columns_e(dim).ravel()

        self._R = sparse.csr_matrix((B_e.ravel(), (rows, columns)), (Ndof, Ndof))
        """Laplacian operator"""

        tic.Tac("DIC", "Laplacian operator", self._verbosity)

    def Get_ldic(self, coef: float = 8.0) -> float:
        """Get the characteristic length of the mesh, i.e. coef * the average length of the edges of the elements."""

        assert coef > 0
        # Calculation of average element size
        l_dic = coef * self.__mesh.Get_meshSize(False).mean()

        return l_dic

    def Get_w(self) -> _types.FloatArray:
        """Returns the 2D periodic vector field."""

        ldic = self.ldic

        x_n = self.__mesh.coord[:, 0]
        y_n = self.__mesh.coord[:, 1]

        w = np.cos(2 * np.pi * x_n / ldic) * np.sin(2 * np.pi * y_n / ldic)

        w = w.repeat(2)

        return w

    def _Compute_L_M(self, img: _types.FloatArray) -> None:
        """Computes DIC matrices."""

        tic = Tic()

        # get image gradient
        grid_Gradfy, grid_Gradfx = np.gradient(img)
        gradY = grid_Gradfy.ravel()
        gradX = grid_Gradfx.ravel()

        roi = self.roi

        self._L = self._N_x @ sparse.diags(gradX) + self._N_y @ sparse.diags(gradY)

        self._M_dic: sparse.csr_matrix = self._L[:, roi] @ self._L[:, roi].T

        # plane wave
        w = self.Get_w()
        # coefs
        w_M = w.T @ self._M_dic @ w
        w_R = w.T @ self._R @ w
        self.__w_M = w_M
        self.__w_R = w_R

        self._M_reg: sparse.csr_matrix = (
            1 / w_M * self._M_dic + self.alpha / w_R * self._R
        )

        # self._M_reg_LU = splu(self._M.tocsc(), permc_spec="MMD_AT_PLUS_A")
        self._M_reg_LU = splu(self._M_reg.tocsc())

        tic.Tac("DIC", "Construct L and M", self._verbosity)

    def _Get_u_from_images(
        self, img1: _types.FloatArray, img2: _types.FloatArray
    ) -> _types.FloatArray:
        """Use open cv to calculate displacements between images."""

        # get the optical flow
        DIS = cv2.DISOpticalFlow_create()  # type: ignore
        IMG1_uint8 = np.uint8(img1 * 2 ** (8 - round(np.log2(img1.max()))))
        IMG2_uint8 = np.uint8(img2 * 2 ** (8 - round(np.log2(img1.max()))))
        # optical flow
        Flow = DIS.calc(IMG1_uint8, IMG2_uint8, None)

        # Project these displacements onto the pixels
        vx = Flow[:, :, 0]
        vy = Flow[:, :, 1]
        b = self._N_x @ vx.ravel() + self._N_y @ vy.ravel()

        u0 = self.__Op_LU.solve(b)  # type: ignore

        return u0

    def __Test_img(self, img: _types.FloatArray) -> None:
        """Checks whether the image is in the right shape."""

        assert img.shape == self.shape, f"Wrong shape, must be {self.shape}"

    def __Get_imgRef(self, imgRef) -> _types.FloatArray:
        """Returns the reference image or checks whether the image entered is the correct shape."""

        if imgRef is None:
            imgRef = self.__imgRef
        else:
            assert isinstance(imgRef, np.ndarray), "Must be an numpy array."
            assert imgRef.size == self.roi.size, f"Wrong shape, must be {self.shape}"

        return imgRef

    def Solve(
        self,
        img: _types.FloatArray,
        u0: Optional[_types.FloatArray] = None,
        iterMax: int = 1000,
        tolConv: float = 1e-6,
        imgRef: Optional[_types.FloatArray] = None,
        verbosity=True,
    ) -> _types.FloatArray:
        """Computes the displacement field between the two images.

        Parameters
        ----------
        img : _types.FloatArray
            deformed image (g)
        u0 : _types.FloatArray, optional
            initial displacement field, by default None\n
            If u0 is None, the field is initialized with _Get_u_from_images(imgRef, img) function
        iterMax : int, optional
            maximum number of iterations, by default 1000
        tolConv : float, optional
            convergence tolerance (converged once ||b|| <= tolConv), by default 1e-6
        imgRef : _types.FloatArray, optional
            reference image (f), by default None
        verbosity : bool, optional
            display iterations, by default True

        Returns
        -------
        _types.FloatArray
            computed displacement field (Ndof)
        """

        self.__Test_img(img)
        imgRef = self.__Get_imgRef(imgRef)

        # initial displacement vector
        if u0 is None:
            u0 = self._Get_u_from_images(imgRef, img)
        else:
            assert (
                u0.size == self.__mesh.Nn * 2
            ), "u0 must be a vector of dimension (2*Nn, 1)"
        u = u0.copy()

        # get pixels' coordinates
        gridX, gridY = np.meshgrid(
            np.arange(imgRef.shape[1]), np.arange(imgRef.shape[0])
        )
        coordX, coordY = gridX.ravel(), gridY.ravel()

        img_fct = interpolate.RectBivariateSpline(
            np.arange(img.shape[0]), np.arange(img.shape[1]), img
        )
        roi = self.roi
        f = imgRef.ravel()[roi]  # reference image as a vector within the roi

        # Assume both images have identical gradients
        R_reg = self.alpha * self._R / self.__w_R
        Lcoef = self._L[:, roi] / self.__w_M

        for iter in range(iterMax):
            ux_p, uy_p = self.Calc_pixelDisplacement(u)

            g = img_fct.ev((coordY + uy_p)[roi], (coordX + ux_p)[roi])
            r = f - g

            b = Lcoef @ r - R_reg @ u
            du = self._M_reg_LU.solve(b)  # type: ignore
            u += du

            norm_b = np.linalg.norm(b)

            if verbosity:
                print(f"Iter {iter + 1:2d} ||b|| {norm_b:.1e}     ", end="\r")

            if norm_b < tolConv:
                break

        if iter + 1 == iterMax:  # type: ignore
            raise Exception("Image correlation analysis did not converge.")

        return u

    def Calc_r_dic(
        self,
        u: _types.FloatArray,
        img: _types.FloatArray,
        imgRef: Optional[_types.FloatArray] = None,
    ) -> _types.FloatArray:
        """Computes the dic residual between img and imgRef (as a Np x Np matrix).\n
        r_dic = f(x) - g(x + u(x))

        Parameters
        ----------
        u : _types.FloatArray
            finite element displacement field (Ndof)
        img : _types.FloatArray
            deformed image
        imgRef : _types.FloatArray, optional
            reference image, by default None

        Returns
        -------
        _types.FloatArray
            the dic residual between images (as a Np x Np matrix)
        """

        self.__Test_img(img)

        imgRef = self.__Get_imgRef(imgRef)

        # Get pixel coordinates
        gridX, gridY = np.meshgrid(
            np.arange(imgRef.shape[1]), np.arange(imgRef.shape[0])
        )
        coordX, coordY = gridX.ravel(), gridY.ravel()

        img_fct = interpolate.RectBivariateSpline(
            np.arange(img.shape[0]), np.arange(img.shape[1]), img
        )

        f = imgRef.ravel()  # reference image as a vector

        ux_p, uy_p = self.Calc_pixelDisplacement(u)

        g = img_fct.ev((coordY + uy_p), (coordX + ux_p))
        r = f - g

        r_dic = np.reshape(r, self.shape)  # as a matrix

        return r_dic

    def Calc_pixelDisplacement(
        self, u: _types.FloatArray
    ) -> tuple[_types.FloatArray, _types.FloatArray]:
        """Computes pixel displacements based on the finite element displacement field."""

        assert (
            u.size == self.__mesh.Nn * 2
        ), f"The displacement vector field has the wrong size. It must be of size {self.__mesh.Nn * 2}"

        ux_p = u @ self._N_x
        uy_p = u @ self._N_y

        return ux_p, uy_p

    def Add_Result(
        self, idx: int, u_exp: _types.FloatArray, img: _types.FloatArray
    ) -> None:
        """Adds the computed dic results.

        Parameters
        ----------
        idx : int
            image index
        u_exp : _types.FloatArray
            finite element displacement field (Ndof)
        img : _types.FloatArray
            image used
        """
        Ndof = self.__mesh.Nn * 2

        if idx not in self.list_idx:
            self.__Test_img(img)
            if u_exp.size != Ndof:
                print(
                    f"The displacement vector field has the wrong size. It must be of size  {Ndof}"
                )
                return

            self.__list_idx.append(idx)
            self.__list_u.append(u_exp)
            self.__list_img.append(img)

    def Save(self, folder: str, filename: str = "dic") -> None:
        """Saves the dic analysis in folder as 'filename.pickle'."""
        path_dic = Folder.Join(folder, f"{filename}.pickle", mkdir=True)
        with open(path_dic, "wb") as file:
            # don't remove
            self.__Op_LU = None
            self._M_reg_LU = None
            pickle.dump(self, file)


# ----------------------------------------------
# DIC Functions
# ----------------------------------------------


def Load_DIC(folder: str, filename: str = "dic") -> DIC:
    """Loads the dic analysis from the specified folder.

    Parameters
    ----------
    folder : str
        The name of the folder where the dic analysis is saved.
    filename : str, optional
        The dic's file name, by default "dic".

    Returns
    -------
    DIC
        The loaded dic analysis."""

    path_dic = Folder.Join(folder, f"{filename}.pickle")
    if not Folder.Exists(path_dic):
        raise Exception(f"The dic analysis does not exist in {path_dic}")

    with open(path_dic, "rb") as file:
        dic: DIC = pickle.load(file)

    return dic


def Get_Circle(
    img: _types.FloatArray, threshold: float, boundary=None, radiusCoef=1.0
) -> tuple[float, float, float]:
    """Returns the circle properties in the image.

    Parameters
    ----------
    img : _types.FloatArray
        image used
    threshold : float
        threshold for pixel color
    boundary: tuple[tuple[float, float], tuple[float, float]], optional
        ((xMin, xMax),(yMin, yMax)), by default None
    radiusCoef : float, optional
        multiplier coef for radius, by default 1.0

    Returns
    -------
    XC, YC, radius
        circle coordinates and radius
    """

    yColor, xColor = np.where(img <= threshold)

    if boundary is None:
        xMin, xMax = 0, img.shape[1]
        yMin, yMax = 0, img.shape[0]
    else:
        assert isinstance(boundary[0], tuple), "Must be a tuple list."
        assert isinstance(boundary[1], tuple), "Must be a tuple list."

        xMin, xMax = boundary[0]
        yMin, yMax = boundary[1]

    filtre = np.where(
        (xColor >= xMin) & (xColor <= xMax) & (yColor >= yMin) & (yColor <= yMax)
    )[0]

    coordoTresh = np.zeros((filtre.size, 2), dtype=float)
    coordoTresh[:, 0] = xColor[filtre]
    coordoTresh[:, 1] = yColor[filtre]

    XC = np.mean(coordoTresh[:, 0]).astype(float)
    YC = np.mean(coordoTresh[:, 1]).astype(float)

    rays = np.linalg.norm(coordoTresh - [XC, YC], axis=1)
    radius: float = np.max(rays) * radiusCoef

    # rays = [np.max(coordoSeuil[:,0]) - XC]
    # rays.append(XC - np.min(coordoSeuil[:,0]))
    # rays.append(YC - np.min(coordoSeuil[:,1]))
    # rays.append(np.max(coordoSeuil[:,1]) - YC)
    # radius = np.max(rayons) * radiusCoef

    return XC, YC, radius
