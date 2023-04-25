import numpy as np
from scipy import interpolate, sparse
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
import os
import pickle
import cv2

from BoundaryCondition import BoundaryCondition
from Mesh import Mesh
import TicTac

class AnalyseDiC:

    def __init__(self, mesh: Mesh, idxImgRef: int, imgRef: np.ndarray, forces=None, deplacements=None, lr=0.0, verbosity=False):
        """Analyse de corrélation d'images

        Parameters
        ----------
        mesh : Mesh
            maillage de la ROI
        idxImgRef : int
            index de l'image de référence dans forces
        imgRef : np.ndarray
            image de référence
        forces : np.ndarray, optional
            vecteurs de force, by default None
        deplacements : np.ndarray, optional
            vecteurs de déplacement, by default None
        lr : float, optional
            longueur de régularisation, by default 0.0
        verbosity : bool, optional
            l'analyse peut ecrire dans la console, by default False

        Returns
        -------
        AnalyseDiC
            Objet pour réaliser la corrélation d'images
        """

        self._forces = forces
        """valeur des efforts bruts relevés pendant les essais en kN"""

        self._deplacements = deplacements
        """valeur des déplacements bruts relevés pendant les essais en mm"""

        self._mesh = mesh
        """maillage dans la base en pixel utilisé pour réaliser la corrélation d'images"""
        self._meshCoef = None
        """maillage mis à l'échelle"""
        self._coef = 1.0
        """coef de mise à l'echelle"""

        self.__Nn = mesh.Nn
        self.__dim = mesh.dim
        self.__nDof = self.__Nn * self.__dim
        self.__ldic = self.__Get_ldic()

        self._idxImgRef = idxImgRef
        """Indexe de l'image de référence"""

        self._imgRef = imgRef        
        """Image utilisée comme image de référence"""

        self.__shapeImages = imgRef.shape
        """Shape des images à utiliser pour les analyses"""

        self._list_u_exp = []
        """liste qui contient les champs de déplacement calculés"""

        self._list_idx_exp = []
        """liste qui contient indexes pour lesquels on a calculé le champ de déplacement"""

        self._list_img_exp = []
        """liste qui contient image pour lesquels on a calculé le champ de déplacement"""

        self.__lr = lr
        """longeur de régularisation"""

        self._verbosity = verbosity

        # initialise la ROI et les fonctions de formes et dérivés des fonction de formes
        
        self.__init__roi()       

        self.__init__Phi_opLap()        

        self.Compute_L_M(imgRef)

    def __init__roi(self):
        """Initialisation de la ROI"""

        tic = TicTac.Tic()

        imgRef = self._imgRef
        mesh = self._mesh

        # récupération de la coordonnées des pixels
        coordPx = np.arange(imgRef.shape[1]).reshape((1,-1)).repeat(imgRef.shape[0], 0).reshape(-1)
        coordPy = np.arange(imgRef.shape[0]).reshape((-1,1)).repeat(imgRef.shape[1]).reshape(-1)
        coordPixel = np.zeros((coordPx.shape[0], 3), dtype=int);  coordPixel[:,0] = coordPx;  coordPixel[:,1] = coordPy

        # récupération des pixels utilisés dans les elements avec leurs coordonnées
        pixels, connectPixel, coordPixelInElem = mesh.groupElem.Get_Nodes_Connect_CoordoInElemRef(coordPixel)

        self.__connectPixel = connectPixel
        """matrice de connectivité qui relié pour chaque element les pixels utilisés"""
        self.__coordPixelInElem = coordPixelInElem
        """coordonnées des pixels dans l'element de référence"""        
        
        # Création de la ROI
        self._roi = np.zeros(coordPx.shape[0])
        self._roi[pixels] = 1
        self._roi = np.asarray(self._roi == 1, dtype=bool)
        """filre permettant d'acceder aux pixels contenues dans le maillage"""

        # ax = plt.subplots()[1]
        # ax.imshow(imgRef)
        # ax.scatter(coordPx[self._roi], coordPy[self._roi], c='white')
        # Affichage.Plot_Mesh(mesh, ax=ax, alpha=0)
        # # # [ax.text(mesh.coordo[n,0], mesh.coordo[n,1], n, c='red')for n in mesh.nodes]

        tic.Tac("DIC", "ROI", self._verbosity)
    
    def __init__Phi_opLap(self):
        """Initialisation des fonctions de forme et de l'opérateur laplacien"""
        
        mesh = self._mesh 
        dim = self.__dim       
        nDof = self.__nDof

        connectPixel = self.__connectPixel
        coordInElem = self.__coordPixelInElem        
        
        # Données du maillage
        matriceType="masse"
        Ntild = mesh.groupElem.Ntild()
        dN_pg = mesh.groupElem.Get_dN_pg(matriceType)
        invF_e_pg = mesh.groupElem.Get_invF_e_pg(matriceType) #; print(invF_e_pg[0,0]); print()
        jacobien_e_pg = mesh.Get_jacobien_e_pg(matriceType)
        poid_pg = mesh.Get_poid_pg(matriceType)        

        # ----------------------------------------------
        # Construction de la matrice de fonction de formes pour les pixels
        # ----------------------------------------------
        lignes_x = []
        lignes_y = []
        colonnes_Phi = []
        values_phi = []

        # Evaluation des fonctions de formes pour les pixels utilisés
        arrayCoordInElem = coordInElem
        phi_n_pixels = np.array([np.reshape([Ntild[n,0](arrayCoordInElem[:,0], arrayCoordInElem[:,1])], -1) for n in range(mesh.nPe)])
         
        tic = TicTac.Tic()

        # TODO possible sans la boucle ?

        for e in range(mesh.Ne):

            # Récupération des noeuds et des pixels de l'element            
            nodes = mesh.connect[e]
            # pixels = sparse.find(connectPixel[e])[1]
            pixels = connectPixel[e]
            # Récupère les fonctions évaluées            
            phi = phi_n_pixels[:,pixels]

            # construction des lignes 
            lignesX = BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodes, ["x"]).reshape(-1,1).repeat(pixels.size)
            lignesY = BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodes, ["y"]).reshape(-1,1).repeat(pixels.size)            

            # construction des colonnes où placer les valeurs
            colonnes = pixels.reshape(1,-1).repeat(mesh.nPe, 0).reshape(-1)            

            lignes_x.extend(lignesX)
            lignes_y.extend(lignesY)
            colonnes_Phi.extend(colonnes)
            values_phi.extend(np.reshape(phi, -1))

        self._Phi_x = sparse.csr_matrix((values_phi, (lignes_x, colonnes_Phi)), (nDof, coordInElem.shape[0]))
        """Matrice des fonctions de formes x (nDof, nPixels)"""
        self._Phi_y = sparse.csr_matrix((values_phi, (lignes_y, colonnes_Phi)), (nDof, coordInElem.shape[0]))
        """Matrice des fonctions de formes y (nDof, nPixels)"""

        Op = self._Phi_x @ self._Phi_x.T + self._Phi_y @ self._Phi_y.T
        self.__Op_LU = splu(Op.tocsc())
        
        tic.Tac("DIC", "Phi_x et Phi_y", self._verbosity)

        # ----------------------------------------------
        # Construction de l'opérateur laplacien
        # ----------------------------------------------

        # dN_e_pg = mesh.Get_dN_sclaire_e_pg(matriceType)
        # dN_e_pg = np.array(np.einsum('epik,pkj->epij', invF_e_pg, dN_pg, optimize='optimal'))
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
        
        # Récupération des lignes et des colonnes ou appliquer les 0
        lignes0 = np.arange(mesh.nPe*dim).repeat(mesh.nPe)
        ddlsX = np.arange(0, mesh.nPe*dim, dim)
        colonnes0 = np.concatenate((ddlsX+1, ddlsX)).reshape(1,-1).repeat(mesh.nPe, axis=0).reshape(-1)

        B_e[:,lignes0, colonnes0] = 0

        lignesB = mesh.lignesVector_e
        colonnesB = mesh.colonnesVector_e        

        self._opLap = sparse.csr_matrix((B_e.reshape(-1), (lignesB.reshape(-1), colonnesB.reshape(-1))), (nDof, nDof))  
        """opérateur laplacien"""      

        tic.Tac("DIC", "Operateur laplacien", self._verbosity)

    def __Get_ldic(self):
        """Calcul ldic la longeur caractéristique du maillage soit 8 x la moyenne de la longeur des bords des elements"""

        indexReord = np.append(np.arange(1, self._mesh.nPe), 0)
        coord = self._mesh.coordo
        connect = self._mesh.connect        

        # Calcul de la taille des elements moyen         
        bords_e_b_c = coord[connect[:,indexReord]] - coord[connect] # vecteurs des bords
        h_e_b = np.sqrt(np.sum(bords_e_b_c**2, 2)) # longeur des bords
        ldic = 8 * np.mean(h_e_b)
        
        return ldic

    def __Get_v(self):
        """Renvoie un déplacement sinusoïdal caractéristique qui correspond à la taille d'element"""

        ldic = self.__ldic

        coordX = self._mesh.coordo[:,0]
        coordY = self._mesh.coordo[:,1]
        
        v = np.cos(2*np.pi*coordX/ldic) * np.sin(2*np.pi*coordY/ldic)

        v = v.repeat(2)

        return v

    def Compute_L_M(self, img: np.ndarray, lr=None):
        """Mise à jour des matrices pour réaliser la DIC"""

        tic = TicTac.Tic()

        if lr == None:
            lr = self.__lr
        else:
            assert lr >= 0.0, "lr doit être >= 0"
            self.__lr = lr
        
        # Récupèré le gradient de l'image
        grid_Gradfy, grid_Gradfx = np.gradient(img)
        gradY = grid_Gradfy.reshape(-1)
        gradX = grid_Gradfx.reshape(-1)        
        
        roi = self._roi

        self.L = self._Phi_x @ sparse.diags(gradX) + self._Phi_y @ sparse.diags(gradY)

        self.M_Dic = self.L[:,roi] @ self.L[:,roi].T

        v = self.__Get_v()
        # onde plane

        coef_M_Dic = v.T @ self.M_Dic @ v
        coef_Op = v.T @ self._opLap @ v
        
        self.__coef_M_Dic = coef_M_Dic
        self.__coef_opLap = coef_Op
        
        if lr == 0.0:
            self.__alpha = 0
        else:
            self.__alpha = (self.__ldic/lr)**2

        self._M = self.M_Dic / coef_M_Dic + self.__alpha * self._opLap / coef_Op        

        # TIKONOV

        # self._M_LU = splu(self._M.tocsc(), permc_spec="MMD_AT_PLUS_A")
        self._M_LU = splu(self._M.tocsc())

        tic.Tac("DIC", "Construit L et M", self._verbosity)

    def __Get_u_from_images(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Utilise open cv pour calculer les déplacements entre images."""        
        
        DIS = cv2.DISOpticalFlow_create()        
        IMG1_uint8 = np.uint8(img1*2**(8-round(np.log2(img1.max()))))
        IMG2_uint8 = np.uint8(img2*2**(8-round(np.log2(img1.max())))) # TODO Ici normal que img1 ?
        Flow = DIS.calc(IMG1_uint8,IMG2_uint8,None)

        # Projete ces déplacements sur les noeuds du maillage

        mapx = Flow[:,:,0]
        mapy = Flow[:,:,1]

        Phix = self._Phi_x
        Phiy = self._Phi_y

        Op_LU = self.__Op_LU

        b = Phix @ mapx.ravel() + Phiy @ mapy.ravel()

        DofValues = Op_LU.solve(b)

        return DofValues

    def __Test_img(self, img: np.ndarray):
        """Fonction qui test si l'image est de la bonne dimension"""

        assert img.shape == self.__shapeImages, f"L'image renseignée n'est pas de la bonne dimension. Doit être de dimension {self.__shapeImages}"

    def __Get_imgRef(self, imgRef) -> np.ndarray:
        """Fonction qui renvoie l'image de référence ou verifie si l'image renseigné est bien de la bonne taille"""

        if imgRef == None:
            imgRef = self._imgRef
        else:
            assert isinstance(imgRef, np.ndarray), "L'image de référence doit être une array numpy"
            assert imgRef.size == self._roi.size, f"L'image de référence renseignée n'est pas de la bonne dimension. Doit être de dimension {self.__shapeImages}"

        return imgRef

    def Solve(self, img: np.ndarray, iterMax=1000, tolConv=1e-6, imgRef=None, verbosity=True):
        """Résolution du champ de déplacement.

        Parameters
        ----------
        img : np.ndarray
            image utilisée pour le calcul
        iterMax : int, optional
            nombre d'itération maximum, by default 1000
        tolConv : float, optional
            tolérance de convergence, by default 1e-6
        imgRef : np.ndarray, optional
            image de référence à utiliser, by default None
        verbosity : bool, optional
            affiche les itérations, by default True

        Returns
        -------
        u, iter
            champ de déplacement et nombre d'itérations pou convergence
        """

        self.__Test_img(img)
        imgRef = self.__Get_imgRef(imgRef)
        # initalisation du vecteur de déplacement
        u = self.__Get_u_from_images(imgRef, img)

        # Récupération de le coordonnées des pixels de l'image
        gridX, gridY = np.meshgrid(np.arange(imgRef.shape[1]),np.arange(imgRef.shape[0]))
        coordX, coordY = gridX.reshape(-1), gridY.reshape(-1)

        img_fct = interpolate.RectBivariateSpline(np.arange(img.shape[0]),np.arange(img.shape[1]),img)
        roi = self._roi
        f = imgRef.reshape(-1)[roi] # image de référence mi sous forme d'un vecteur et en récupérant les pixels dans la Roi
        
        # Ici l'hypothèse des petits déplacements est utilisée
        # On suppose que le gradiant des deux images et identiques
        # Pour des grands déplacements, il faudrait recalculer les matrices en utilisant Compute_L_M
        opLapReg = self.__alpha * self._opLap / self.__coef_opLap # opérateur laplacian régularisé
        Lcoef = self.L[:,roi] / self.__coef_M_Dic

        for iter in range(iterMax):

            ux_p, uy_p = self.__Calc_pixelDisplacement(u)

            g = img_fct.ev((coordY + uy_p)[roi], (coordX + ux_p)[roi])
            r = f - g

            b = Lcoef @ r - opLapReg @ u
            du = self._M_LU.solve(b)
            u += du
            
            if verbosity:
                print(f"Iter {iter+1:2d} ||b|| {np.linalg.norm(b):.3}     ", end='\r')
            if iter == 0:
                b0 = np.linalg.norm(b)
            elif np.linalg.norm(b) < b0 * tolConv:
                break

        return u, iter

    def Residu(self, u: np.ndarray, img: np.ndarray, imgRef=None) -> np.ndarray:
        """Calcul du résidu entre les images.

        Parameters
        ----------
        u : np.ndarray
            champ de déplacement
        img : np.ndarray
            image utilisée pour le calcul
        imgRef : np.ndarray, optional
            image de référence à utiliser, by default None

        Returns
        -------
        np.ndarray
            résidu entre les images
        """
        
        self.__Test_img(img)

        imgRef = self.__Get_imgRef(imgRef)

        # Récupération de le coordonnées des pixels de l'image
        gridX, gridY = np.meshgrid(np.arange(imgRef.shape[1]),np.arange(imgRef.shape[0]))
        coordX, coordY = gridX.reshape(-1), gridY.reshape(-1)

        img_fct = interpolate.RectBivariateSpline(np.arange(img.shape[0]),np.arange(img.shape[1]),img)

        f = imgRef.reshape(-1) # image de référence mi sous forme d'un vecteur et en récupérant les pixels dans la Roi

        ux_p, uy_p = self.__Calc_pixelDisplacement(u)

        g = img_fct.ev((coordY + uy_p), (coordX + ux_p))
        r = f - g

        r_dic = np.reshape(r, self.__shapeImages)

        return r_dic

    def Set_meshCoef_coef(self, mesh: Mesh, coef: float):
        """Renseigne le maillage et le coef de mise à l'échelle

        Parameters
        ----------
        mesh : Mesh
            maillage
        coef : float
            coefficient de mise à l'échelle
        """
        assert isinstance(mesh, Mesh), "Doit être un maillage"
        self._meshCoef = mesh
        self._coef = coef


    def __Calc_pixelDisplacement(self, u: np.ndarray):
        """Calcul le déplacement des pixels depuis le déplacement des noeuds du maillage en utilisant les fonctions de formes"""        

        ux_p = u @ self._Phi_x
        uy_p = u @ self._Phi_y
        
        return ux_p, uy_p

    def Add_Result(self, idx: int, u_exp: np.ndarray, img: np.ndarray):
        """Ajoute le champ de déplacement calculé.

        Parameters
        ----------
        idx : int
            indexe de l'image
        u_exp : np.ndarray
            champ de déplacement
        img : np.ndarray
            image utilisée
        """
        if idx not in self._list_idx_exp:            
            # verifications
            self.__Test_img(img)
            u_exp.size == self.__nDof, f"Le champ vectorielle de déplacement n'est pas de la bonne dimension. Doit être de dimension {self.__nDof}"

            self._list_idx_exp.append(idx)
            self._list_u_exp.append(u_exp)
            self._list_img_exp.append(img)

    def Save(self, pathname: str):
        with open(pathname, 'wb') as file:
            self.__Op_LU = None
            self._M_LU = None
            pickle.dump(self, file)


def Load_Analyse(path: str) -> AnalyseDiC:
    """Procédure de chargement"""

    if not os.path.exists(path):
        raise Exception(f"L'analyse n'existe pas dans {path}")

    with open(path, 'rb') as file:
        analyseDic = pickle.load(file)

    assert isinstance(analyseDic, AnalyseDiC)

    return analyseDic

def CalculEnergie(deplacements: np.ndarray, forces: np.ndarray, ax=None) -> float:
    """Fonction qui calul l'energie sous la courbe"""

    if isinstance(ax, plt.Axes):
        ax.plot(deplacements, forces)
        canPlot = True
    else:
        canPlot = False

    energie = 0

    listIndexes = np.arange(deplacements.shape[0]-1)

    for idx0 in listIndexes:

        idx1 = idx0+1

        idxs = [idx0, idx1]

        ff = forces[idxs]

        largeur = deplacements[idx1]-deplacements[idx0]
        hauteurRectangle = np.min(ff)

        hauteurTriangle = np.max(ff)-np.min(ff)

        energie += largeur * (hauteurRectangle + hauteurTriangle/2)

        if canPlot:
            ax.fill_between(deplacements[idxs], forces[idxs], color='red')

            # if idx0 > 0:
            #     sc.remove()
            # sc = ax.scatter(deplacements[idxs[1]], forces[idxs[1]], c='black')
            # plt.pause(1e-12)

    return energie

def Get_Circle(img:np.ndarray, seuil: float, boundary=None, coefRayon=1.0):
    """Recupère le cercle dans l'image.

    Parameters
    ----------
    img : np.ndarray
        image utilisée
    seuil : float
        seuil pour la couleur des pixels
    boundary : tuple[tuple[float, float], tuple[float, float]], optional
        ((xMin, xMax),(yMin, yMax)), by default None
    coefRayon : float, optional
        coef multiplicateur pour le rayon, by default 1.0

    Returns
    -------
    XC, YC, rayon
        coordonnées et rayon du cercle
    """

    yColor, xColor = np.where(img <= seuil)

    if boundary == None:
        xMin, xMax = 0, img.shape[1]
        yMin, yMax = 0, img.shape[0]
    else:
        assert isinstance(boundary[0], tuple), "Doit etre une liste de tuple"
        assert isinstance(boundary[1], tuple), "Doit etre une liste de tuple"

        xMin, xMax = boundary[0]
        yMin, yMax = boundary[1]        

    filtre = np.where((xColor>=xMin) & (xColor<=xMax) & (yColor>=yMin) & (yColor<=yMax))[0]

    coordoSeuil = np.zeros((filtre.size, 2))
    coordoSeuil[:,0] = xColor[filtre]
    coordoSeuil[:,1] = yColor[filtre]

    XC = np.mean(coordoSeuil[:,0])
    YC = np.mean(coordoSeuil[:,1])
    rayons = [np.max(coordoSeuil[:,0]) - XC]
    rayons.append(XC - np.min(coordoSeuil[:,0]))
    rayons.append(YC - np.min(coordoSeuil[:,1]))
    rayons.append(np.max(coordoSeuil[:,1]) - YC)
    
    rayon = np.max(rayons) * coefRayon

    return XC, YC, rayon