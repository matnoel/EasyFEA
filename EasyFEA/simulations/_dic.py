# Copyright (C) 2021-2024 Université Gustave Eiffel. All rights reserved.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""DIC analysis module"""

import numpy as np
from scipy import interpolate, sparse
from scipy.sparse.linalg import splu
import pickle
import cv2 # need opencv-python library

# utilities
from ..utilities import Tic, Folder
# fem
from ..fem import Mesh, BoundaryCondition

class DIC:

    def __init__(self, mesh: Mesh, idxImgRef: int, imgRef: np.ndarray,
                 forces: np.ndarray=None, displacements: np.ndarray=None, lr=0.0, verbosity=False):
        """Creates a DIC analysis.

        Parameters
        ----------
        mesh : Mesh
            ROI mesh
        idxImgRef : int
            index of reference image in forces
        imgRef : np.ndarray
            reference image
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

        self._forces = forces
        """forces measured during the tests."""

        self._displacements = displacements
        """displacements measured during the tests."""

        self._mesh: Mesh = mesh
        """pixel-based mesh used for image correlation."""
        assert mesh.dim == 2, "Must be a 2D mesh."

        self._meshCoef = None
        """scaled mesh."""
        self._coef = 1.0
        """scaling coef (image scale [mm/px])."""

        self.__Nn: int = mesh.Nn
        self.__dim: int = mesh.dim
        self.__nDof: int = self.__Nn * self.__dim
        self.__ldic: float = self.__Get_ldic()

        self._idxImgRef: int = idxImgRef
        """Reference image index in _loads."""

        self._imgRef: np.ndarray = imgRef        
        """Image used as reference."""

        self.__shapeImages: tuple[int, int] = imgRef.shape
        """Shape of images to be used for analysis."""

        self._list_u_exp: list[np.ndarray] = []
        """List containing calculated displacement fields."""

        self._list_idx_exp: list[int] = []
        """List containing indexes for which the displacement field has been calculated."""

        self._list_img_exp: list[np.ndarray] = []
        """List containing images for which the displacement field has been calculated."""

        self.__lr: float = lr
        """regulation length."""

        self._verbosity: bool = verbosity

        # initialize ROI and shape functions and shape function derivatives        
        self.__init__roi()       

        self.__init__Phi_opLap()        

        self.Compute_L_M(imgRef)

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

        self.__connectPixel: np.ndarray = connectPixel
        """connectivity matrix which links the pixels used for each element."""
        self.__coordPixelInElem: np.ndarray = coordPixelInElem
        """pixel coordinates in the reference element."""
        
        # ROI creation
        self._roi: np.ndarray = np.zeros(coordPx.shape[0])
        self._roi[pixels] = 1
        self._roi = np.asarray(self._roi == 1, dtype=bool)
        """vector filter for accessing the pixels contained in the mesh."""
        self._ROI: np.ndarray = self._roi.reshape(self.__shapeImages)
        """matrix filter for accessing the pixels contained in the mesh."""

        tic.Tac("DIC", "ROI", self._verbosity)
    
    def __init__Phi_opLap(self) -> None:
        """Initializing shape functions and the Laplacian operator."""
        
        mesh = self._mesh 
        dim = self.__dim       
        nDof = self.__nDof

        connectPixel = self.__connectPixel
        coordInElem = self.__coordPixelInElem        
        
        # Initializing shape functions and the Laplacian operator
        matrixType="mass"
        Ntild = mesh.groupElem._Ntild()
        dN_pg = mesh.groupElem.Get_dN_pg(matrixType)
        invF_e_pg = mesh.groupElem.Get_invF_e_pg(matrixType)
        jacobien_e_pg = mesh.Get_jacobian_e_pg(matrixType)
        poid_pg = mesh.Get_weight_pg(matrixType)        

        # ----------------------------------------------
        # Construction of shape function matrix for pixels
        # ----------------------------------------------
        lines_x = []
        lines_y = []
        columns_Phi = []
        values_phi = []

        # Evaluation of shape functions for the pixels used        
        phi_n_pixels = np.array([np.reshape([Ntild[n,0](coordInElem[:,0], coordInElem[:,1])], -1) for n in range(mesh.nPe)])
         
        tic = Tic()

        # TODO possible without the loop?
        for e in range(mesh.Ne):

            # Retrieve element nodes and pixels
            nodes = mesh.connect[e]            
            pixels: np.ndarray = connectPixel[e]
            # Retrieves evaluated functions
            phi = phi_n_pixels[:,pixels]

            # line construction
            linesX = BoundaryCondition.Get_dofs_nodes(["x","y"], nodes, ["x"]).reshape(-1,1).repeat(pixels.size)
            linesY = BoundaryCondition.Get_dofs_nodes(["x","y"], nodes, ["y"]).reshape(-1,1).repeat(pixels.size)
            # construction of columns in which to place values
            colonnes = pixels.reshape(1,-1).repeat(mesh.nPe, 0).ravel()            

            lines_x.extend(linesX)
            lines_y.extend(linesY)
            columns_Phi.extend(colonnes)
            values_phi.extend(np.reshape(phi, -1))        

        self._Phi_x = sparse.csr_matrix((values_phi, (lines_x, columns_Phi)), (nDof, coordInElem.shape[0]))
        """Shape function matrix x (nDof, nPixels)"""
        self._Phi_y = sparse.csr_matrix((values_phi, (lines_y, columns_Phi)), (nDof, coordInElem.shape[0]))
        """Shape function matrix y (nDof, nPixels)"""

        Op = self._Phi_x @ self._Phi_x.T + self._Phi_y @ self._Phi_y.T
        self.__Op_LU = splu(Op.tocsc())
        
        tic.Tac("DIC", "Phi_x and Phi_y", self._verbosity)

        # ----------------------------------------------
        # Construction of the Laplacian operator
        # ----------------------------------------------
        
        dN_e_pg = np.array(np.einsum('epki,pkj->epij', invF_e_pg, dN_pg, optimize='optimal'))

        dNxdx = dN_e_pg[:,:,0]
        dNydy = dN_e_pg[:,:,1]

        ind_x = np.arange(0, mesh.nPe*dim, dim)
        ind_y = ind_x + 1        

        dN_vector = np.zeros((dN_e_pg.shape[0], dN_e_pg.shape[1], 3, mesh.nPe*dim))            
        dN_vector[:,:,0,ind_x] = dNxdx
        dN_vector[:,:,1,ind_y] = dNydy            
        dN_vector[:,:,2,ind_x] = dNydy; dN_vector[:,:,2,ind_y] = dNxdx

        B_e = np.einsum('ep,p,epji,epjk->eik', jacobien_e_pg, poid_pg, dN_vector, dN_vector, optimize='optimal')
        
        # Retrieve rows and columns or apply 0s
        lignes0 = np.arange(mesh.nPe*dim).repeat(mesh.nPe)
        ddlsX = np.arange(0, mesh.nPe*dim, dim)
        colonnes0 = np.concatenate((ddlsX+1, ddlsX)).reshape(1,-1).repeat(mesh.nPe, axis=0).ravel()

        B_e[:,lignes0, colonnes0] = 0

        linesB = mesh.linesVector_e
        columnsB = mesh.columnsVector_e        

        self._opLap = sparse.csr_matrix((B_e.ravel(), (linesB.ravel(), columnsB.ravel())), (nDof, nDof))  
        """Laplacian operator"""      

        tic.Tac("DIC", "Laplacian operator", self._verbosity)

    def __Get_ldic(self) -> float:
        """Calculation ldic the characteristic length of the mesh, i.e. 8 x the average length of the edges of the elements."""

        indexReord = np.append(np.arange(1, self._mesh.nPe), 0)
        coord = self._mesh.coord
        connect = self._mesh.connect        

        # Calculation of average element size
        bords_e_b_c = coord[connect[:,indexReord]] - coord[connect] # edge vectors
        h_e_b = np.sqrt(np.sum(bords_e_b_c**2, 2)) # edge lengths
        ldic = 8 * np.mean(h_e_b)
        
        return ldic

    def __Get_v(self) -> np.ndarray:
        """Returns characteristic sinusoidal displacement corresponding to element size."""

        ldic = self.__ldic

        coordX = self._mesh.coord[:,0]
        coordY = self._mesh.coord[:,1]
        
        v = np.cos(2*np.pi*coordX/ldic) * np.sin(2*np.pi*coordY/ldic)

        v = v.repeat(2)

        return v

    def Compute_L_M(self, img: np.ndarray, lr=None) -> None:
        """Updating matrix to produce for DIC with TIKONOV."""

        tic = Tic()

        if lr is None:
            lr = self.__lr
        else:
            assert lr >= 0.0, "lr must be >= 0"
            self.__lr = lr
        
        # Recover image gradient
        grid_Gradfy, grid_Gradfx = np.gradient(img)
        gradY = grid_Gradfy.ravel()
        gradX = grid_Gradfx.ravel()        
        
        roi = self._roi

        self.L = self._Phi_x @ sparse.diags(gradX) + self._Phi_y @ sparse.diags(gradY)

        self.M_Dic = self.L[:,roi] @ self.L[:,roi].T

        v = self.__Get_v()
        # plane wave

        coef_M_Dic = v.T @ self.M_Dic @ v
        coef_Op = v.T @ self._opLap @ v
        
        self.__coef_M_Dic = coef_M_Dic
        self.__coef_opLap = coef_Op
        
        if lr == 0.0:
            self.__alpha = 0
        else:
            self.__alpha = (self.__ldic/lr)**2

        self._M = self.M_Dic / coef_M_Dic + self.__alpha * self._opLap / coef_Op 
        
        # self._M_LU = splu(self._M.tocsc(), permc_spec="MMD_AT_PLUS_A")
        self._M_LU = splu(self._M.tocsc())

        tic.Tac("DIC", "Construct L and M", self._verbosity)

    def __Get_u_from_images(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Use open cv to calculate displacements between images."""
        
        DIS = cv2.DISOpticalFlow_create()        
        IMG1_uint8 = np.uint8(img1*2**(8-round(np.log2(img1.max()))))
        IMG2_uint8 = np.uint8(img2*2**(8-round(np.log2(img1.max()))))
        Flow = DIS.calc(IMG1_uint8,IMG2_uint8,None)

        # Project these displacements onto the mesh nodes
        mapx = Flow[:,:,0]
        mapy = Flow[:,:,1]

        Phix = self._Phi_x
        Phiy = self._Phi_y

        Op_LU = self.__Op_LU

        b = Phix @ mapx.ravel() + Phiy @ mapy.ravel()

        DofValues = Op_LU.solve(b)

        return DofValues

    def __Test_img(self, img: np.ndarray) -> None:
        """Function to test whether the image is the right size."""

        assert img.shape == self.__shapeImages, f"The image entered is the wrong size. Must be {self.__shapeImages}"

    def __Get_imgRef(self, imgRef) -> np.ndarray:
        """Function that returns the reference image or checks whether the image entered is the correct size."""

        if imgRef is None:
            imgRef = self._imgRef
        else:
            assert isinstance(imgRef, np.ndarray), "The reference image must be an numpy array."
            assert imgRef.size == self._roi.size, f"The reference image entered is the wrong size. Must be {self.__shapeImages}"

        return imgRef

    def Solve(self, img: np.ndarray, iterMax=1000, tolConv=1e-6, imgRef=None, verbosity=True) -> np.ndarray:
        """Displacement field between the img and the imgRef.

        Parameters
        ----------
        img : np.ndarray
            image used for calculation
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
        # initalization of displacement vector
        u = self.__Get_u_from_images(imgRef, img)

        # Recovery of image pixel coordinates
        gridX, gridY = np.meshgrid(np.arange(imgRef.shape[1]),np.arange(imgRef.shape[0]))
        coordX, coordY = gridX.ravel(), gridY.ravel()

        img_fct = interpolate.RectBivariateSpline(np.arange(img.shape[0]),np.arange(img.shape[1]),img)
        roi = self._roi
        f = imgRef.ravel()[roi] # reference image as a vector and retrieving pixels in the roi
        
        # Here the small displacement hypothesis is used
        # The gradient of the two images is assumed to be identical
        # For large displacements, the matrices would have to be recalculated using Compute_L_M
        opLapReg = self.__alpha * self._opLap / self.__coef_opLap # operator laplacian regularized
        Lcoef = self.L[:,roi] / self.__coef_M_Dic

        for iter in range(iterMax):

            ux_p, uy_p = self.__Calc_pixelDisplacement(u)

            g = img_fct.ev((coordY + uy_p)[roi], (coordX + ux_p)[roi])
            r = f - g

            b = Lcoef @ r - opLapReg @ u
            du = self._M_LU.solve(b)
            u += du
            
            norm_b = np.linalg.norm(b)

            if verbosity:
                print(f"Iter {iter+1:2d} ||b|| {norm_b:.3}     ", end='\r')             
            
            if norm_b <= tolConv:
                break

        if iter+1 > iterMax:
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

        ux_p, uy_p = self.__Calc_pixelDisplacement(u)

        g = img_fct.ev((coordY + uy_p), (coordX + ux_p))
        r = f - g

        r_dic = np.reshape(r, self.__shapeImages)

        return r_dic

    def Set_meshCoef_coef(self, mesh: Mesh, imgScale: float) -> None:
        """Set mesh size and scaling factor

        Parameters
        ----------
        mesh : Mesh
            mesh
        imgScale : float
            scaling coefficient [mm/px]
        """
        assert isinstance(mesh, Mesh) and mesh.dim == 2, "Must be a 2D mesh."
        self._meshCoef = mesh
        self._coef = imgScale


    def __Calc_pixelDisplacement(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculates pixel displacement from mesh node displacement using shape functions."""        

        ux_p = u @ self._Phi_x
        uy_p = u @ self._Phi_y
        
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
            if u_exp.size != self.__nDof:
                print(f"The displacement vector field is the wrong dimension. Must be of dimension {self.__nDof}")
                return

            self._list_idx_exp.append(idx)
            self._list_u_exp.append(u_exp)
            self._list_img_exp.append(img)

    def Save(self, folder: str, filename: str="dic") -> None:
        """Saves the dic analysis as 'filename.pickle'."""
        path_dic = Folder.New_File(f"{filename}.pickle", folder)
        with open(path_dic, 'wb') as file:
            self.__Op_LU = None
            self._M_LU = None
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