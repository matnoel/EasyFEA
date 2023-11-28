from Interface_Gmsh import gmsh, Interface_Gmsh, ElemType
from Geom import Point, PointsList, Contour, CircleArc, Line
import Display
import Folder
import Simulations
import Materials
import PostProcessing

import numpy as np
import scipy.io

folder = Folder.Get_Path(__file__)

addCylinder = True
repeat = False
N=4

# elemType = ElemType.TETRA4
elemType = ElemType.PRISM6
# elemType = ElemType.HEXA8

if __name__ == '__main__':

    Display.Clear()

    interface = Interface_Gmsh(False, True)
    factory = gmsh.model.occ

    # --------------------------------------------------------------------------------------------
    # Geom
    # --------------------------------------------------------------------------------------------
    
    l = 2
    t = 0.3
    H1 = 5
    H2 = 5
    H3 = 5    
    R = 5
    e = 1
    
    angleRev = 2*np.pi/10 # rad    
    rot = 30;  coefRot = rot/(H1+H2+H3) # deg
    rot0 = 0 
    rot1 = H1 * coefRot - rot0
    rot2 = (H1+H2) * coefRot - rot1
    rot3 = (H1+H2+H3) * coefRot - rot2
    assert t<R

    mS = l/N

    vols: list[tuple[Contour, Contour]] = []

    def bladeSection(l:float, t:float, center:np.ndarray, angle=0.0, R:float=0.0, coef=2):
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
        spline1 = PointsList(split1[:idxM+1], mS)
        spline2 = PointsList(split1[idxM:], mS)
        spline3 = PointsList(split2[:idxM+1], mS)
        spline4 = PointsList(split2[idxM:], mS)
        
        # construct contour
        contour = Contour([spline1, spline2, spline3, spline4])

        return contour, np.asarray(coord)
    
    def Cylinder(R, angleRev: float) -> Contour:

        s = np.sin(angleRev/2)
        c = np.cos(angleRev/2)

        P1 = Point(-s*R, -l, c*R)
        P2 = Point(s*R, -l, c*R)
        P3 = Point(s*R, -l/2, c*R)
        P4 = Point(-s*R, -l/2, c*R)
        C1 = Point(y=-l)
        C2 = Point(y=-l/2)

        L1 = CircleArc(P1, C1, P2, mS)
        L2 = Line(P2,P3, mS)
        L3 = CircleArc(P3, C2, P4, mS)
        L4 = Line(P4,P1, mS)

        contour = Contour([L1,L2,L3,L4])

        return contour
    
    def CylinderBlade(P1: Point, L2: PointsList) -> Contour:
        P2 = L2.pt1
        P3 = L2.pt2
        P4 = P1.copy(); P4.translate(dy=l)

        C1 = Point(0,P2.y,0)
        C2 = Point(0,P3.y,0)

        L1 = CircleArc(P1,C1,P2, mS)        
        L3 = CircleArc(P3,C2,P4, mS)
        L4 = Line(P4,P1, mS)

        # contour = Contour([L1,L2,L3,L4])
        contour = Contour([L1,L2,L3,L4])

        return contour

    def LinkEveryone(vols: list[tuple[Contour, Contour, list[int]]], rot=0.0) -> list[tuple]:
        allEntities = []
        for vol in vols:
            contour1, contour2, nLayers, numElems = vol
            contour1 = contour1.copy(); contour1.rotate(rot, direction=(0,1,0))
            contour2 = contour2.copy(); contour2.rotate(rot, direction=(0,1,0))
            ents = interface._Link_Contours(contour1, contour2, elemType, nLayers, numElems)

            allEntities.extend(ents)

        return allEntities

    # --------------------------------------------------------------------------------------------
    # Mesh
    # --------------------------------------------------------------------------------------------

    bladeInf, coordInf = bladeSection(l, t, (0,0,R), rot0, R) # blade in contact with the cylinder in z = R
    bladeSup, coordSup = bladeSection(l, t, (0,0,R+e), rot0, R) # blade in contact with the cylinder in z = R+e


    blade1, __ = bladeSection(l,t,(0,0,R+e+H1), rot1)
    blade2, __ = bladeSection(l*2,t,(0,-l/2,R+e+H1+H2), rot2)
    blade3, __ = bladeSection(l*2,t,(0,-l/2,R+e+H1+H2+H3), rot3)

    elems = [N/2] * 4 # number of elements for each lines for blades
    
    vols.append((bladeInf, bladeSup, e/mS, elems))
    vols.append((bladeSup, blade1, H1/mS, elems))
    vols.append((blade1, blade2, H2/mS, elems))
    vols.append((blade2, blade3, H3/mS, elems))
    
    if addCylinder:

        elems = [N*2, l/2/mS] * 2
        
        # cylinder y=-l
        contourInf = Cylinder(R, angleRev)
        contourSup = Cylinder(R+e, angleRev)

        # cylinder y=l
        contourInf_c = contourInf.copy(); contourInf_c.translate(dy=l+l/2)
        contourSup_c = contourSup.copy(); contourSup_c.translate(dy=l+l/2)
                
        vols.append((contourInf, contourSup, e/mS, elems))
        vols.append((contourInf_c, contourSup_c, e/mS, elems))
        
        elems = [N]*4
        # cylinder blade left
        coordInfL = coordInf[:coordInf.shape[0]//2]
        splineInfL = PointsList([Point(*co) for co in coordInfL], mS)
        splineSupL = splineInfL.copy(); splineSupL.translate(dz=e)        
        contourInf = CylinderBlade(contourInf.points[-1], splineInfL)
        contourSup = CylinderBlade(contourSup.points[-1], splineSupL)        
        # cylinder blade right
        contourSup_c = contourSup.copy(); contourSup_c.rotate(np.pi)
        contourInf_c = contourInf.copy(); contourInf_c.rotate(np.pi)
        
        vols.append((contourInf, contourSup, e/mS, elems))
        vols.append((contourInf_c, contourSup_c, e/mS, elems))

    




    for v, vol in enumerate(vols):

        color = 'blue'

        if v==0:
            ax = vol[0].Plot(color=color)
        else:
            vol[0].Plot(ax, color=color)
        vol[1].Plot(ax, color=color)

    firstPart = LinkEveryone(vols)

    factory.synchronize()


    interface._Set_algorithm(elemType)

    parts = factory.getEntities()

    if repeat:

        na = int(np.pi*2//angleRev)

        for i in range(na-1):
            print(f"{(i+1)/(na-1)*100:2.0f} %")
            LinkEveryone(vols, angleRev*(i+1))

    factory.synchronize()

    # if mS > 0:
    #     gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mS)

    interface._Set_PhysicalGroups(setPoints=False)
    
    interface._Meshing(3, elemType, isOrganised=False)

    # gmsh.write(Folder.Join([folder,"blade.msh"]))

    mesh = interface._Construct_Mesh()

    Display.Plot_Mesh(mesh)

    print(mesh)

    nodesCircle = mesh.Nodes_Conditions(lambda x,y,z: np.sqrt(x**2+z**2)<=R+1e-2)
    nodesUpper = mesh.Nodes_Conditions(lambda x,y,z: z>=mesh.coordoGlob[:,2].max()-1e-2)

    nodesBlades = mesh.Nodes_Conditions(lambda x,y,z: np.sqrt(x**2+z**2)>=R+e+1e-1)
    nodesCyl = mesh.Nodes_Conditions(lambda x,y,z: np.sqrt(x**2+z**2)<=R+e)

    mesh.groupElem.Set_Nodes_Tag(nodesCircle, 'blades')
    mesh.groupElem.Set_Nodes_Tag(nodesCyl, 'cylindre')

    # --------------------------------------------------------------------------------------------
    # Simu
    # --------------------------------------------------------------------------------------------

    # material = Materials.Elas_Isot(mesh.dim)

    # simu = Simulations.Simu_Displacement(mesh, material)

    # uz = 1e-1
    # simu.add_dirichlet(nodesCircle, [0]*mesh.dim, simu.Get_directions())
    # simu.add_dirichlet(nodesUpper, [uz], ["z"])    
    # simu.Solve()
    # simu.Save_Iter()

    # --------------------------------------------------------------------------------------------
    # Results
    # --------------------------------------------------------------------------------------------

    # deformFactor = uz / uz

    # Display.Plot_Result(simu, 'uz', deformFactor)
    
    # Display.Plot_Result(simu, 'Svm', deformFactor)

    # Display.Plot_Mesh(simu, deformFactor)

    # Display.Plot_BoundaryConditions(simu)
        
    Display.Plot_Model(mesh, alpha=0.1, showId=False)

    # ax = Display.Plot_Elements(mesh, nodesCyl, c='green')
    # Display.Plot_Elements(mesh, nodesBlades, c='grey', ax=ax)
    # ax.legend()

    # PostProcessing.Make_Paraview(folder, simu)

    # print(simu)

    matFile = Folder.Join([folder, 'mesh.mat'])
    msh = {
        'connect': np.asarray(mesh.connect+1, dtype=float),
        'coordo': mesh.coordoGlob,
        'bladesNodes': np.asarray(nodesBlades + 1, dtype=float),
        'bladesElems': np.asarray(mesh.Elements_Nodes(nodesBlades) + 1, dtype=float),
        'cylindreNodes': np.asarray(nodesCyl + 1, dtype=float),
        'cylindreElems': np.asarray(mesh.Elements_Nodes(nodesCyl) + 1, dtype=float),
    }
    scipy.io.savemat(matFile, msh)


    Display.plt.show()