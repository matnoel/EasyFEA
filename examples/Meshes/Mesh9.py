"""Simplified turbine mesh with data extraction in matlab."""

from EasyFEA import (Display, Folder, np,
                     Mesher, ElemType, 
                     PyVista_Interface as pvi)
from EasyFEA.Geoms import Point, Points, Contour, CircleArc, Line

import scipy.io

if __name__ == '__main__':

    folder = Folder.New_File(Folder.Join('Meshes', 'Blade'), results=True)

    N=20 # elements in the blade lenght l
    addCylinder = True
    repeat = False
    angleRev = 2*np.pi/20 # rad
    saveToMatlab = False

    # elemType = ElemType.TETRA4
    elemType = ElemType.PRISM6
    # elemType = ElemType.HEXA8

    Display.Clear()

    mesher = Mesher(False, True, True)
    factory = mesher._factory

    # ----------------------------------------------
    # Geom
    # ----------------------------------------------
    
    l = 2
    t = 0.3
    H1 = 5; c1 = 1
    H2 = 5; c2 = 2
    H3 = 5; c3 = 2
    R = 5
    e = 1
    
    rot = 15;  coefRot = rot/(H1+H2+H3) # deg    
    rot0 = 0
    rot1 = H1 * coefRot
    rot2 = (H1+H2) * coefRot - rot1
    rot3 = (H1+H2+H3) * coefRot - rot2
    assert t<R

    mS = l/N

    vols: list[tuple[Contour, Contour]] = []

    def bladeSection(l:float, t:float, center:np.ndarray, angle=0.0, R:float=0.0):
        """Create a blade section

        Parameters
        ----------
        l : float
            blade length
        t : float
            blade thickness
        center : np.ndarray
            center of mass for the blade section
        angle : float, optional
            rotation in deg along z axis (0,0,1), by default 0.0
        R : float, optional
            radius used to project on the cylinder, by default 0.0

        Returns
        -------
        Contour
            created contour
        """

        # crete points coordinates
        n = 11
        y = np.linspace(0,l,n)
        y = np.append(y, y[::-1])

        mat = np.array([[0, 0, 1],[(l/2)**2, l/2, 1],[l**2, l, 1]])
        a,b,c = tuple(np.linalg.inv(mat) @ [0,-t/2,0])
        x: np.ndarray = a*y**2 + b*y + c

        x[y.size//2:] *= -1
        
        z = np.zeros_like(y)

        coord = np.array([x,y,z]).T
        
        # first center the surface on (0,0,0) to do the rotation
        coord = coord - np.array([0,l/2,0])

        # rotate along z axis
        rot = angle * np.pi/180
        rotMat = np.array([[np.cos(rot), -np.sin(rot), 0],
                           [np.sin(rot), np.cos(rot), 0],
                           [0, 0, 1]])        
        coord = coord @ rotMat

        # project on R radius the z coordinates
        if R != 0:
            coord[:,2] = -(R - np.sqrt(R**2 - coord[:,0]**2))
        
        # center the surface
        coord += center

        # create points
        points = [Point(*co) for co in coord]
        
        # split points in 4 groups to construct 4 points list
        split1, split2 = np.reshape(points, (2,-1))

        idxM = len(split1)//2
        spline1 = Points(split1[:idxM+1], mS)
        spline2 = Points(split1[idxM:], mS)
        spline3 = Points(split2[:idxM+1], mS)
        spline4 = Points(split2[idxM:], mS)
        
        # construct contour
        contour = Contour([spline1, spline2, spline3, spline4])

        # contour.Plot()

        return contour, np.asarray(coord)
    
    def Cylinder(R, angleRev: float, coord: np.ndarray, y1: float) -> tuple[Contour, Contour]:

        s = np.sin(angleRev/2)
        c = np.cos(angleRev/2)

        P1 = Point(-s*R, coord[0,1], c*R)
        P2 = Point(*coord[0])
        P3 = Point(s*R, P2.y, c*R)        
        P4 = Point(s*R, y1, c*R)
        P5 = Point(P2.x,y1,P2.z)
        P6 = Point(-s*R, y1, c*R)
        
        C1 = Point(y=P2.y)
        C2 = Point(y=y1)

        L1_l = CircleArc(P1, P2, C1, meshSize=mS)
        L2_l = Line(P2,P5, mS)
        L3_l = CircleArc(P5, P6, C2, meshSize=mS)
        L4_l = Line(P6,P1, mS)

        contour_l = Contour([L1_l,L2_l,L3_l,L4_l])

        L1_r = CircleArc(P2, P3, C1, meshSize=mS)
        L2_r = Line(P3,P4, mS)
        L3_r = CircleArc(P4, P5, C2, meshSize=mS)
        L4_r = Line(P5,P2, mS)

        contour_r = Contour([L1_r,L2_r,L3_r,L4_r])

        return contour_l, contour_r
    
    def CylinderBlade(R, angleRev: float, coord: np.ndarray) -> Contour:

        s = np.sin(angleRev/2)
        c = np.cos(angleRev/2)

        y0 = coord[0,1]
        y1 = coord[-1,1]
        
        P1 = Point(-s*R, y0, c*R)
        P2 = Point(*coord[0])
        P3 = Point(*coord[-1])
        P4 = Point(-s*R, y1, c*R)

        C1 = Point(0,y0,0)
        C2 = Point(0,y1,0)

        L1 = CircleArc(P1,P2,C1, meshSize=mS)
        L2 = Points([Point(*co) for co in coord], mS)
        L3 = CircleArc(P3,P4,C2, meshSize=mS)
        L4 = Line(P4,P1, mS)
        
        contour = Contour([L1,L2,L3,L4])

        return contour

    def LinkEveryone(vols: list[tuple[Contour, Contour, list[int]]], rot=0.0) -> list[tuple]:
        """contour1, contour2, nLayers, numElems
        return created entities"""
        allEntities = []
        for vol in vols:
            contour1, contour2, nLayers, numElems = vol
            contour1 = contour1.Copy(); contour1.Rotate(rot, direction=(0,1,0))
            contour2 = contour2.Copy(); contour2.Rotate(rot, direction=(0,1,0))
            ents = mesher._Link_Contours(contour1, contour2, elemType, nLayers, numElems)

            allEntities.extend(ents)

        return allEntities

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    bladeInf, coordInf = bladeSection(l, t, (0,0,R), rot0, R) # blade in z = R
    bladeSup, coordSup = bladeSection(l, t, (0,0,R+e), rot0, R+e) # blade in z = R+e

    coordInfL, coordInfR = np.reshape(coordInf, (2,-1,3))
    coordSupL, coordSupR = np.reshape(coordSup, (2,-1,3))

    blade1, __ = bladeSection(l*c1,t,(0,-(c1*l-l)/2,R+e+H1), rot1)
    blade2, __ = bladeSection(l*c2,t,(0,-(c2*l-l)/2,R+e+H1+H2), rot2)
    blade3, __ = bladeSection(l*c3,t,(0,-(c3*l-l)/2,R+e+H1+H2+H3), rot3)

    elems = [N/2] * 4 # number of elements for each lines for blades
    
    # contour1, contour2, nLayers, numElems
    vols.append((bladeInf, bladeSup, e/mS, elems))
    vols.append((bladeSup, blade1, H1/mS, elems))
    vols.append((blade1, blade2, H2/mS, elems))
    vols.append((blade2, blade3, H3/mS, elems))
    
    if addCylinder:

        elems = [N, l/2/mS] * 2        
        
        # cylinder y=-l
        cylinInf1_l, cylinInf1_r = Cylinder(R, angleRev, coordInfL, -l)
        cylinSup1_l, cylinSup1_r = Cylinder(R+e, angleRev, coordSupL, -l)

        vols.append((cylinInf1_l, cylinSup1_l, e/mS, elems))
        vols.append((cylinInf1_r, cylinSup1_r, e/mS, elems))

        # cylinder y=l
        cylinInf2_l, cylinInf2_r = Cylinder(R, angleRev, coordInfR, l)
        cylinSup2_l, cylinSup2_r = Cylinder(R+e, angleRev, coordSupR, l)        
        vols.append((cylinInf2_l, cylinSup2_l, e/mS, elems))
        vols.append((cylinInf2_r, cylinSup2_r, e/mS, elems))
        
        elems = [N]*4
        # cylinder blade left
        coordInfL = coordInf[:coordInf.shape[0]//2]
        coordSupfL = coordSup[:coordSup.shape[0]//2]        
        contourInf = CylinderBlade(R, angleRev, coordInfL)
        contourSup = CylinderBlade(R+e, angleRev, coordSupfL)
        
        # cylinder blade right
        coordInfR = coordInf[coordInf.shape[0]//2:]
        coordSupfR = coordSup[coordSup.shape[0]//2:]        
        contourInf_c = CylinderBlade(R, -angleRev, coordInfR[::-1])
        contourSup_c = CylinderBlade(R+e, -angleRev, coordSupfR[::-1])
        
        vols.append((contourInf, contourSup, e/mS, elems))
        vols.append((contourInf_c, contourSup_c, e/mS, elems))
    
    # axGeom = Display.init_Axes(3)
    # axGeom.axis('off')
    # for v, vol in enumerate(vols):        
    #     color = 'blue'        
    #     vol[0].Plot(axGeom, color=color)
    #     vol[1].Plot(axGeom, color=color)        
    #     # ax.legend()

    firstPart = LinkEveryone(vols)

    mesher._Set_algorithm(elemType)

    # mesher._synchronize()
    # parts = factory.getEntities()

    if repeat:

        na = int(np.pi*2/angleRev)

        for i in range(na-1):
            print(f"{(i+1)/(na-1)*100:2.0f} %")
            LinkEveryone(vols, angleRev*(i+1)*180/np.pi)

            # # dont work
            # partsC = factory.copy(parts)
            # factory.rotate(partsC, 0,0,0,0,0,1, angleRev*(i+1))
    
    mesher._Meshing(3, elemType, folder=folder, filename='blade')

    mesh = mesher._Construct_Mesh()

    nodesCircle = mesh.Nodes_Conditions(lambda x,y,z: np.sqrt(x**2+z**2)<=R+1e-2)
    nodesUpper = mesh.Nodes_Conditions(lambda x,y,z: z>=mesh.coordGlob[:,2].max()-1e-2)

    nodesBlades = mesh.Nodes_Conditions(lambda x,y,z: np.sqrt(x**2+z**2)>=R+e+1e-1)
    nodesCyl = mesh.Nodes_Conditions(lambda x,y,z: np.sqrt(x**2+z**2)<=R+e)

    mesh.groupElem.Set_Nodes_Tag(nodesCircle, 'blades')
    mesh.groupElem.Set_Nodes_Tag(nodesCyl, 'cylindre')

    if saveToMatlab:
        matFile = Folder.Join(folder, 'blade.mat')
        msh = {
            'connect': np.asarray(mesh.connect+1, dtype=float),
            'coordo': mesh.coordGlob,
            'bladesNodes': np.asarray(nodesBlades + 1, dtype=float),
            'bladesElems': np.asarray(mesh.Elements_Nodes(nodesBlades) + 1, dtype=float),
            'cylindreNodes': np.asarray(nodesCyl + 1, dtype=float),
            'cylindreElems': np.asarray(mesh.Elements_Nodes(nodesCyl) + 1, dtype=float),
        }
        scipy.io.savemat(matFile, msh)

    # ----------------------------------------------
    # Plot
    # ----------------------------------------------
        
    pvi.Plot_Mesh(mesh).show()

    qual = mesh.Get_Quality('aspect')
    pvi.Plot(mesh, qual, nodeValues=False, show_edges=True, cmap='viridis', clim=(0,1), n_colors=11).show()

    Display.plt.show()