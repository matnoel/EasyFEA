# Copyright (C) 2021-2024 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""DIC analysis module"""

import numpy as np
from scipy import interpolate, sparse
from scipy.sparse.linalg import splu
import pickle
import cv2 # need opencv-python library

# utilities
from ..utilities import Tic, Folder, Display
from ..utilities._observers import Observable, _IObserver
# fem
from ..fem import Mesh, BoundaryCondition

class DIC(_IObserver):

    def __init__(self, mesh: Mesh, idxImgRef: int, imgRef: np.ndarray, coef: float=1.0,
                 forces: np.ndarray=None, displacements: np.ndarray=None,
                 lr=0.0, verbosity=False):
        """Creates a DIC analysis.

        Parameters
        ----------
        mesh : Mesh
            ROI mesh
        idxImgRef : int
            index of reference image in forces
        imgRef : np.ndarray
            reference image
        coef : float, optional
            scaling coef (image scale [mm/px]) or pixel size, by default 1.0
        forces : np.ndarray, optional
            force vectors, by default None
        displacements : np.ndarray, optional
            displacement vectors, by default None
        lr : float, optional
            regularization length, by default 0.0
        verbosity : bool, optional
            analysis can write to console, by default False

        Returns
        -------
        AnalyseDiC
            Object for image correlation
        """

        # mesh
        assert mesh.dim == 2, "Must be a 2D mesh."
        mesh._Add_observer(self)
        
        self._mesh = mesh #[rm]
        # self.__mesh = mesh #[uncom]

        # self.__mesh: Mesh = mesh # TODO Remove comments identified by #[uncom] after the end of the thesis.
        # also remove the uncommented lines #[rm]
        # For now, comments are not removed because I need to exploit saved correlation results that are not compatible with the commented code!
        assert coef >= 0
        self._coef = coef

        meshC = mesh.copy() #[rm]
        meshC._ResetMatrix() #[rm]
        meshC.coordGlob = meshC.coordGlob * self._coef #[rm]
        self._meshCoef = meshC #[rm]

        # solutions

        self._forces = forces
        """forces measured during the tests."""

        self._displacements = displacements
        """displacements measured during the tests."""

        self._idxImgRef = idxImgRef #[rm]
        # self.__idxImgRef: int = idxImgRef #[uncom]
        """Reference image index in _loads."""

        self._imgRef = imgRef #[rm]
        # self.__imgRef: np.ndarray = imgRef #[uncom]
        """Image used as reference."""        

        self._list_idx_exp: list[int] = [] #[rm]
        # self.__list_idx_exp: list[int] = [] #[uncom]
        """List containing indexes for which the displacement field has been calculated."""

        self._list_u_exp: list[np.ndarray] = [] #[rm]
        # self.__list_u_exp: list[np.ndarray] = [] #[uncom]
        """List containing calculated displacement fields."""

        self._list_img_exp: list[np.ndarray] = [] #[rm]
        # self.__list_img_exp: list[np.ndarray] = [] #[uncom]
        """List containing images for which the displacement field has been calculated."""
        
        self._verbosity: bool = verbosity

        # initialize ROI and shape functions and shape function derivatives        
        self.__init__roi()
        self.__init__Phi_opLap()
        
        # regul
        self.__lr = 0.0
        self._lr = lr
        # Updating self._lr will automatically update the matrices
        # That's why we can comment on the following line
        # self.Compute_L_M(imgRef)

    # mesh properties

    # @property #[uncom]
    # def _mesh(self) -> Mesh:
    #     """pixel-based mesh used for image correlation."""
    #     return self.__mesh
    
    @property
    def dim(self) -> int:
        """mesh dim."""
        return self._mesh.dim #[rm]
        # return self.__mesh.dim #[uncom]
    
    @property
    def Nn(self) -> int:
        return self._mesh.Nn #[rm]
        # return self.__mesh.Nn #[uncom]
    
    @property
    def Ndof(self) -> int:
        return self.Nn * self.dim
    
    @property
    def ldic(self) -> float:
        """8 * mean(meshSize)"""
        return self._Get_ldic()
    
    # @property #[uncom]
    # def _coef(self) -> float:
    #     """scaling coef (image scale [mm/px]) or pixel size."""
    #     return self.__coef    
    
    # @_coef.setter #[uncom]
    # def _coef(self, value: float) -> None:
    #     assert value != 0.0
    #     self.__coef = value
 
    # @property #[uncom]
    # def _meshCoef(self) -> Mesh:
    #     """scaled mesh."""
    #     meshC = self.__mesh.copy()
    #     meshC._ResetMatrix()
    #     meshC.coordGlob = meshC.coordGlob * self._coef
    #     return meshC
    
    # image properties

    # @property #[uncom]
    # def _idxImgRef(self) -> int:
    #     """Reference image index in _loads (or in the folder)."""
    #     return self.__idxImgRef

    # @property
    # def _imgRef(self) -> np.ndarray:
    #     """Image used as reference."""
    #     return self.__imgRef.copy()
    
    @property
    def _shapeImages(self) -> tuple[int, int]:        
        """Image dimension required"""
        return self._imgRef.shape #[rm]
        # return self.__imgRef.shape #[uncom]

    # regularisation properties

    @property
    def _lr(self) -> float:
        """regulation length."""
        return self.__lr

    @_lr.setter
    def _lr(self, value: float) -> None:
        """# Warning!\n
        Changing this parameter will automatically update the matrices with Compute_L_M function!"""
        assert value >= 0.0, "lr must be >= 0.0"
        self.__lr = value
        self._Compute_L_M(self._imgRef)

    @property
    def _alpha(self) -> float:
        if self._lr == 0.0:
            return 0
        else:
            return (self.ldic/self._lr)**2

    # solution properties
    
    # @property #[uncom]
    # def _list_idx_exp(self) -> list[int]:
    #     """Copy of the list containing indexes for which the displacement field has been calculated."""
    #     return self.__list_idx_exp.copy()

    # @property #[uncom]
    # def _list_u_exp(self) -> list[np.ndarray]:
    #     """Copy of the list containing the calculated displacement fields."""
    #     return self.__list_u_exp.copy()
    
    # @property #[uncom]
    # def _list_img_exp(self) -> list[np.ndarray]:
    #     """Copy of the list containing images for which the displacement field has been calculated."""
    #     return self.__list_img_exp.copy()

    def _Update(self, observable: Observable, event: str) -> None:
        if isinstance(observable, Mesh):
            raise Exception("The current implementation does not allow you to make any modifications to the mesh.")
        else:
            Display.MyPrintError("Notification not yet implemented")

    def __init__roi(self) -> None:
        """ROI initialization."""

        tic = Tic()

        imgRef = self._imgRef
        mesh = self._mesh

        # recovery of pixel coordinates
        coordPx = np.arange(imgRef.shape[1]).reshape((1,-1)).repeat(imgRef.shape[0], 0).ravel()
        coordPy = np.arange(imgRef.shape[0]).reshape((-1,1)).repeat(imgRef.shape[1]).ravel()
        coordPixel = np.zeros((coordPx.shape[0], 3), dtype=int);  coordPixel[:,0] = coordPx;  coordPixel[:,1] = coordPy

        # recovery of pixels used in elements with their coordinates
        pixels, __, connectPixel, coordPixelInElem = mesh.groupElem.Get_Mapping(coordPixel)
        # mean_pixels = np.mean([connectPixel[e].size for e in range(mesh.Ne)])

        self.__connectPixel: np.ndarray = connectPixel
        """connectivity matrix which links the pixels used for each element."""
        self.__coordPixelInElem: np.ndarray = coordPixelInElem
        """pixel coordinates in the reference element."""
        
        # ROI creation
        roi: np.ndarray = np.zeros(coordPx.shape[0])
        roi[pixels] = 1
        self.__roi = np.asarray(roi == 1, dtype=bool)

        tic.Tac("DIC", "ROI", self._verbosity)
    
    @property
    def _roi(self) -> np.ndarray[bool]:
        """roi as a vector."""
        return self.__roi.copy()

    @property
    def _ROI(self) -> np.ndarray[bool]:
        """roi as a matrix."""
        return self._roi.reshape(self._shapeImages)

    def __init__Phi_opLap(self) -> None:
        """Initializing shape functions and the Laplacian operator."""
        
        mesh = self._mesh 
        dim = self.dim       
        Ndof = self.Ndof

        connectPixel = self.__connectPixel
        coordInElem = self.__coordPixelInElem        
        
        # Initializing shape functions and the Laplacian operator
        matrixType = "mass"
        Ntild = mesh.groupElem._Ntild()
        jacobian_e_pg = mesh.Get_jacobian_e_pg(matrixType) # (e, p)
        weight_pg = mesh.Get_weight_pg(matrixType) # (p)        
        dN_e_pg = mesh.groupElem.Get_dN_e_pg(matrixType) # (e, p, dim, nPe)

        # ----------------------------------------------
        # Construction of shape function matrix for pixels (N)
        # ----------------------------------------------
        lines_x = []
        lines_y = []
        columns_Phi = []
        values_phi = []

        # Evaluating shape functions for the pixels used
        x_p, y_p = coordInElem[:,0], coordInElem[:,1]
        phi_n_pixels = np.array([np.reshape([Ntild[n,0](x_p, y_p)], -1) for n in range(mesh.nPe)])
         
        tic = Tic()
        
        # Possible without the loop?
        # No, it is not possible without the loop because connectPixel doesn't have the same number of columns in each row.
        # In addition, if you remove it, you'll have to make several list comprehension.
        for e in range(mesh.Ne):

            # Retrieve element nodes and pixels
            nodes = mesh.connect[e]            
            pixels: np.ndarray = connectPixel[e]
            # Retrieves evaluated functions
            phi = phi_n_pixels[:,pixels]

            # line construction
            linesX = BoundaryCondition.Get_dofs_nodes(["x","y"], nodes, ["x"]).reshape(-1,1).repeat(pixels.size)
            # linesY = BoundaryCondition.Get_dofs_nodes(["x","y"], nodes, ["y"]).reshape(-1,1).repeat(pixels.size) 
            # same as
            linesY = linesX + 1            
            # construction of columns in which to place values
            colonnes = pixels.reshape(1,-1).repeat(mesh.nPe, 0).ravel()            

            lines_x.extend(linesX)
            lines_y.extend(linesY)
            columns_Phi.extend(colonnes)
            values_phi.extend(np.reshape(phi, -1))    

        self._N_x = sparse.csr_matrix((values_phi, (lines_x, columns_Phi)), (Ndof, coordInElem.shape[0]))
        """Shape function matrix x (nDof, nPixels)"""
        self._N_y = sparse.csr_matrix((values_phi, (lines_y, columns_Phi)), (Ndof, coordInElem.shape[0]))
        """Shape function matrix y (nDof, nPixels)"""

        Op: sparse.csr_matrix = self._N_x @ self._N_x.T + self._N_y @ self._N_y.T

        self.__Op_LU = splu(Op.tocsc())
        
        tic.Tac("DIC", "Phi_x and Phi_y", self._verbosity)

        # ----------------------------------------------
        # Construction of the Laplacian operator (R)
        # ----------------------------------------------
        dNdx = dN_e_pg[:,:,0]
        dNdy = dN_e_pg[:,:,1]

        ind_x = np.arange(0, mesh.nPe*dim, dim)
        ind_y = ind_x + 1

        dN_x = np.zeros((mesh.Ne, weight_pg.size, 2, 2*mesh.nPe))
        dN_y = np.zeros_like(dN_x)

        dN_x[:,:,0,ind_x] = dNdx
        dN_x[:,:,1,ind_y] = dNdx
        Bx_e = np.einsum('ep,p,epdi,epdj->eij', jacobian_e_pg, weight_pg, dN_x, dN_x, optimize='optimal')

        dN_y[:,:,0,ind_x] = dNdy
        dN_y[:,:,1,ind_y] = dNdy
        By_e = np.einsum('ep,p,epdi,epdj->eij', jacobian_e_pg, weight_pg, dN_y, dN_y, optimize='optimal')

        B_e = Bx_e + By_e

        lines = mesh.linesVector_e.ravel()
        columns = mesh.columnsVector_e.ravel()

        self._R = sparse.csr_matrix((B_e.ravel(), (lines, columns)), (Ndof, Ndof))
        """Laplacian operator"""

        tic.Tac("DIC", "Laplacian operator", self._verbosity)    

    def _Get_ldic(self, coef=8) -> float:
        """Calculation ldic the characteristic length of the mesh, i.e. 8 x the average length of the edges of the elements."""

        assert coef > 0
        # Calculation of average element size
        ldic = 8 * self._mesh.Get_meshSize(False).mean()

        return ldic

    def _Get_w(self) -> np.ndarray:
        """Returns characteristic sinusoidal displacement corresponding to element size."""

        ldic = self.ldic

        x_n = self._mesh.coord[:,0]
        y_n = self._mesh.coord[:,1]
        
        w = np.cos(2*np.pi*x_n/ldic) * np.sin(2*np.pi*y_n/ldic)

        w = w.repeat(2)

        return w

    def _Compute_L_M(self, img: np.ndarray) -> None:
        """Updating matrix to produce for DIC with TIKONOV."""

        tic = Tic()
        
        # Recover image gradient
        grid_Gradfy, grid_Gradfx = np.gradient(img)
        gradY = grid_Gradfy.ravel()
        gradX = grid_Gradfx.ravel()        
        
        roi = self._roi

        self.L = self._N_x @ sparse.diags(gradX) + self._N_y @ sparse.diags(gradY)

        self.M_dic: sparse.csr_matrix = self.L[:,roi] @ self.L[:,roi].T

        # plane wave
        w = self._Get_w()
        # coefs
        w_M = w.T @ self.M_dic @ w
        w_R = w.T @ self._R @ w        
        self.__w_M = w_M
        self.__w_R = w_R

        self._M_reg: sparse.csr_matrix = 1/w_M * self.M_dic + self._alpha/w_R * self._R  
        
        # self._M_reg_LU = splu(self._M.tocsc(), permc_spec="MMD_AT_PLUS_A")
        self._M_reg_LU = splu(self._M_reg.tocsc())

        tic.Tac("DIC", "Construct L and M", self._verbosity)

    def _Get_u_from_images(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Use open cv to calculate displacements between images."""
        
        DIS = cv2.DISOpticalFlow_create()
        IMG1_uint8 = np.uint8(img1*2**(8-round(np.log2(img1.max()))))
        IMG2_uint8 = np.uint8(img2*2**(8-round(np.log2(img1.max()))))
        Flow = DIS.calc(IMG1_uint8,IMG2_uint8,None)

        # Project these displacements onto the pixels
        ux_p = Flow[:,:,0]
        uy_p = Flow[:,:,1]

        b =  self._N_x @ ux_p.ravel() + self._N_y @ uy_p.ravel()

        u0 = self.__Op_LU.solve(b)

        return u0

    def __Test_img(self, img: np.ndarray) -> None:
        """Function to test whether the image is the right size."""

        assert img.shape == self._shapeImages, f"The image entered is the wrong size. Must be {self._shapeImages}"

    def __Get_imgRef(self, imgRef) -> np.ndarray:
        """Function that returns the reference image or checks whether the image entered is the correct size."""

        if imgRef is None:
            imgRef = self._imgRef
        else:
            assert isinstance(imgRef, np.ndarray), "The reference image must be an numpy array."
            assert imgRef.size == self._roi.size, f"The reference image entered is the wrong size. Must be {self._shapeImages}"

        return imgRef

    def Solve(self, img: np.ndarray, u0: np.ndarray=None, iterMax=1000, tolConv=1e-6, imgRef=None, verbosity=True) -> np.ndarray:
        """Displacement field between the img and the imgRef.

        Parameters
        ----------
        img : np.ndarray
            image used for calculation
        u0 : np.ndarray, optional
            initial displacement field, by default None\n
            If u0 == None, the field is initialized with _Get_u_from_images(imgRef, img)
        iterMax : int, optional
            maximum number of iterations, by default 1000
        tolConv : float, optional
            convergence tolerance (converged once ||b|| <= tolConv), by default 1e-6
        imgRef : np.ndarray, optional
            reference image to use, by default None
        verbosity : bool, optional
            display iterations, by default True

        Returns
        -------
        u
            displacement field
        """

        self.__Test_img(img)
        imgRef = self.__Get_imgRef(imgRef)

        # initial displacement vector
        if u0 is None:
            u0 = self._Get_u_from_images(imgRef, img)
        else:
            assert u0.size == self._mesh.Nn * 2, "u0 must be a vector of dimension (Nn*2, 1)"
        u = u0.copy()

        # Recovery of image pixel coordinates
        gridX, gridY = np.meshgrid(np.arange(imgRef.shape[1]),np.arange(imgRef.shape[0]))
        coordX, coordY = gridX.ravel(), gridY.ravel()

        img_fct = interpolate.RectBivariateSpline(np.arange(img.shape[0]),np.arange(img.shape[1]),img)
        roi = self._roi
        f = imgRef.ravel()[roi] # reference image as a vector and retrieving pixels in the roi
        
        # Here the small displacement hypothesis is used
        # The gradient of the two images is assumed to be identical
        # For large displacements, the matrices would have to be recalculated using Compute_L_M
        R_reg = self._alpha * self._R / self.__w_R # operator laplacian regularized
        Lcoef = self.L[:,roi] / self.__w_M

        for iter in range(iterMax):

            ux_p, uy_p = self._Calc_pixelDisplacement(u)

            g = img_fct.ev((coordY + uy_p)[roi], (coordX + ux_p)[roi])
            r = f - g

            b = Lcoef @ r - R_reg @ u
            du = self._M_reg_LU.solve(b)
            u += du
            
            norm_b = np.linalg.norm(b)

            if verbosity:
                print(f"Iter {iter+1:2d} ||b|| {norm_b:.3}     ", end='\r')             
            
            # if iter == 0:
            #     b0 = norm_b.copy()
            # if norm_b < b0*tolConv:            
            if norm_b < tolConv:
                break

        if iter+1 == iterMax:
            raise Exception("Image correlation analysis did not converge.")

        return u

    def Residual(self, u: np.ndarray, img: np.ndarray, imgRef=None) -> np.ndarray:
        """Residual calculation between images (f-g).

        Parameters
        ----------
        u : np.ndarray
            displacement field
        img : np.ndarray
            image used for calculation
        imgRef : np.ndarray, optional
            reference image to use, by default None

        Returns
        -------
        np.ndarray
            residual between images
        """
        
        self.__Test_img(img)

        imgRef = self.__Get_imgRef(imgRef)

        # Recover image pixel coordinates
        gridX, gridY = np.meshgrid(np.arange(imgRef.shape[1]),np.arange(imgRef.shape[0]))
        coordX, coordY = gridX.ravel(), gridY.ravel()

        img_fct = interpolate.RectBivariateSpline(np.arange(img.shape[0]),np.arange(img.shape[1]),img)

        f = imgRef.ravel() # reference image as a vector and retrieving pixels in the roi

        ux_p, uy_p = self._Calc_pixelDisplacement(u)

        g = img_fct.ev((coordY + uy_p), (coordX + ux_p))
        r = f - g

        r_dic = np.reshape(r, self._shapeImages)

        return r_dic

    def _Calc_pixelDisplacement(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculates pixel displacement from mesh node displacement using shape functions."""        

        ux_p = u @ self._N_x
        uy_p = u @ self._N_y
        
        return ux_p, uy_p

    def Add_Result(self, idx: int, u_exp: np.ndarray, img: np.ndarray) -> None:
        """Adds the calculated displacement field.

        Parameters
        ----------
        idx : int
            image index
        u_exp : np.ndarray
            displacement field
        img : np.ndarray
            image used
        """
        if idx not in self._list_idx_exp:
            
            self.__Test_img(img)
            if u_exp.size != self.Ndof:
                print(f"The displacement vector field is the wrong dimension. Must be of dimension {self.Ndof}")
                return

            self._list_idx_exp.append(idx)
            self._list_u_exp.append(u_exp)
            self._list_img_exp.append(img)

    def Save(self, folder: str, filename: str="dic") -> None:
        """Saves the dic analysis as 'filename.pickle'."""
        path_dic = Folder.New_File(f"{filename}.pickle", folder)
        with open(path_dic, 'wb') as file:
            self.__Op_LU = None
            self._M_reg_LU = None
            pickle.dump(self, file)

# ----------------------------------------------
# DIC Functions
# ----------------------------------------------

def Load_DIC(folder: str, filename: str="dic") -> DIC:
    """Load the dic analysis from the specified folder.

    Parameters
    ----------
    folder : str
        The name of the folder where the simulation is saved.
    filename : str, optional
        The simualtion name, by default "dic".

    Returns
    -------
    DIC
        The loaded dic analysis."""
    
    path_dic = Folder.Join(folder, f"{filename}.pickle")
    if not Folder.Exists(path_dic):
        raise Exception(f"The dic analysis does not exist in {path_dic}")

    with open(path_dic, 'rb') as file:
        dic: DIC = pickle.load(file)

    return dic

def Get_Circle(img:np.ndarray, threshold: float, boundary=None, radiusCoef=1.0) -> tuple[float, float, float]:
    """Recovers the circle in the image.

    Parameters
    ----------
    img : np.ndarray
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

    filtre = np.where((xColor>=xMin) & (xColor<=xMax) & (yColor>=yMin) & (yColor<=yMax))[0]

    coordoTresh = np.zeros((filtre.size, 2))
    coordoTresh[:,0] = xColor[filtre]
    coordoTresh[:,1] = yColor[filtre]

    XC: float = np.mean(coordoTresh[:,0])
    YC: float = np.mean(coordoTresh[:,1])

    rays = np.linalg.norm(coordoTresh - [XC,YC],axis=1)
    radius: float = np.max(rays) * radiusCoef

    # rays = [np.max(coordoSeuil[:,0]) - XC]
    # rays.append(XC - np.min(coordoSeuil[:,0]))
    # rays.append(YC - np.min(coordoSeuil[:,1]))
    # rays.append(np.max(coordoSeuil[:,1]) - YC)    
    # radius = np.max(rayons) * radiusCoef

    return XC, YC, radius