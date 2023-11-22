from Interface_Gmsh import gmsh, Interface_Gmsh, ElemType, Point, PointsList
import Display
import Folder
import Simulations
import Materials
import PostProcessing

import numpy as np
import scipy.io

folder = Folder.Get_Path(__file__)

addCylinder = True
repeat = True
isOrganised = False

if __name__ == '__main__':

    Display.Clear()

    # --------------------------------------------------------------------------------------------
    # Geom
    # --------------------------------------------------------------------------------------------
    
    l = 2
    t = 0.3
    H1 = 5
    H2 = 5
    H3 = 5
    h = 1e-2
    R = 5
    e = 1
    
    angleCylinder = 2*np.pi/20 # rad    
    rot = 30; coefRot = rot/(H1+H2+H3)
    rot1 = 90 # deg
    rot2 = rot1 + H1*coefRot # deg
    rot3 = rot2 + H2*coefRot # deg
    rot4 = rot3 + H3*coefRot # deg
    assert t<R

    meshSize = l/10*4    

    def getXY(l,t, plot=False):

        x = np.linspace(0,l,11)
        x = np.append(x, x[::-1])

        mat = np.array([[0, 0, 1],[(l/2)**2, l/2, 1],[l**2, l, 1]])
        a,b,c = tuple(np.linalg.inv(mat) @ [0,t/2,0])
        y: np.ndarray = a*x**2 + b*x + c

        y[x.size//2:] *= -1

        if plot:
            axBlade = Display.plt.subplots()[1]
            axBlade.plot(x, y)
            axBlade.plot(x, y, ls='', marker='.',c='black')
            axBlade.axis('equal')

        return x,y
    
    x, y = getXY(l, t)

    # --------------------------------------------------------------------------------------------
    # Mesh
    # --------------------------------------------------------------------------------------------
    interface = Interface_Gmsh(True, True)
    # dim, elemType = 2, ElemType.TRI3
    dim, elemType = 3, ElemType.TETRA10

    factory = gmsh.model.occ

    

    def addBladeSection(x: np.ndarray, y: np.ndarray, center: np.ndarray,
                        rot: float, ax=0, ay=0, az=1) -> list[tuple]:
        """Builds a blade section\n
        
        return
        entities = [
            (0,p0),\n
            (0,p1),\n
            (1,spline1),\n
            (1,spline2),\n
            (2,surf)\n
            ]"""

        # x and y must be cut in half

        points: list[int] = []        
        coords: list[tuple] = []

        for i in range(x.size):

            coord = (x[i], y[i], 0.0)

            if coord not in coords:                
                point = factory.addPoint(*coord)                

            else:
                p = [p for p in range(len(points)) if coords[p] == coord][0]
                point = points[p]

            coords.append(coord)
            points.append(point)
                
        points1, points2 = np.reshape(points, (2,-1))

        p0, p1 = points1[[0,-1]]

        spline1 = factory.addSpline(points1)
        spline2 = factory.addSpline(points2)        

        rmPoints = list(set(points) - set([p0, p1]))
        factory.remove([(0,point) for point in rmPoints])

        loop = factory.addCurveLoop([spline1, spline2])

        surf = factory.addSurfaceFilling(loop)

        factory.remove([(1, loop)])

        entities = [
            (0,p0),
            (0,p1),
            (1,spline1),
            (1,spline2),
            (2,surf)
            ]
        
        centerCoord = factory.getCenterOfMass(2, surf)

        dec = np.array(center) - np.array(centerCoord)

        xc,yc,zc = centerCoord + dec

        factory.rotate(entities, *centerCoord, ax,ay,az, rot)
        factory.translate(entities, *dec)

        if isOrganised:
            factory.synchronize()        
            coords1 = np.array([x[:x.size//2],y[:x.size//2]]).T
            dist = np.linalg.norm(coords1[1:]-coords1[:-1], axis=1)
            dist = np.sum(dist)        
            N = int(dist/meshSize)

            [gmsh.model.mesh.setTransfiniteCurve(line, 10) for line in [spline1,spline2]]

        return surf, entities
    
    def addBlade(entities1: list[tuple], entities2: list[tuple]):
        """Builds a blade using 2 blade sections"""

        # points, lines and surface for the first entities
        p1 = entities1[0][1]
        p2 = entities1[1][1]
        l1 = entities1[2][1]
        l2 = entities1[3][1]
        surf1 = entities1[4][1]
        # points, lines and surface for the second entities
        p3 = entities2[0][1]
        p4 = entities2[1][1]
        l3 = entities2[2][1]
        l4 = entities2[3][1]
        surf2 = entities2[4][1]
        # create lines between surfaces corners
        l5 = factory.addLine(p1,p3)
        l6 = factory.addLine(p2,p4)
        # create loops to create surface 3 & 4
        loop3 = factory.addCurveLoop([l1,l6,l3,l5])
        loop4 = factory.addCurveLoop([l2,l6,l4,l5])
        # creates surfaces 4 & 5
        surf3 = factory.addSurfaceFilling(loop3)
        surf4 = factory.addSurfaceFilling(loop4)
        # create the volume
        vol = factory.addSurfaceLoop([surf1, surf2, surf3, surf4])        
        factory.addVolume([vol])
        
        # return entities
        entities = [(0, point) for point in [p1,p2,p3,p4]]
        entities.extend([(1, line) for line in [l1,l2,l3,l4]])
        entities.extend([(2, surf) for surf in [surf1,surf2,surf3,surf4]])
        entities.append((3, vol))

        if isOrganised:
            factory.synchronize()

            coord1 = gmsh.model.getValue(0,p1,[])
            coord3 = gmsh.model.getValue(0,p3,[])
            dist = np.linalg.norm(coord1-coord3)
            # dddd = gmsh.model.getValue(1,l1,[])
            N = int(np.floor(dist/meshSize))

            [gmsh.model.mesh.setTransfiniteCurve(line, N) for line in [l5,l6]]
            gmsh.model.mesh.setTransfiniteSurface(surf3, cornerTags=[])
            gmsh.model.mesh.setTransfiniteSurface(surf4, cornerTags=[])        

        return entities
    
    # creates the first blade surface near the cylinder
    surf1, entities1 = addBladeSection(x,y,(0,0,R+e+h), rot1*np.pi/180)
    # creates the 2nd blade surface
    surf2, entities2 = addBladeSection(x,y,(0,0,R+e+h+H1), rot2*np.pi/180)
    # creates the 2nd blade shape
    l2 = l*1.5
    decY = (l2-l)/2
    xn, yn = getXY(l2,t)
    # creates the 3nd blade surface
    surf3, entities3 = addBladeSection(xn,yn,(0,-decY,R+e+h+H1+H2), rot3*np.pi/180)
    # creates the 4nd blade surface
    surf4, entities4 = addBladeSection(xn,yn,(0,-decY,R+e+h+H1+H2+H3), rot4*np.pi/180)
    
    # create 3 blades
    blade1 = addBlade(entities1, entities2)    
    blade2 = addBlade(entities2, entities3)
    blade3 = addBlade(entities3, entities4)
    # Extrusion of the blade that will be in contact with the cylinder
    extrude = factory.extrude([(2, surf1)], 0,0,-h-2*e, [1])

    vol_blade = factory.getEntities(3)

    factory.synchronize()

    if addCylinder:

        # contsruction the contour
        P1 = Point(0,-l,R)
        P2 = Point(0,l,R)
        P3 = Point(0,l,R+e)
        P4 = Point(0,-l,R+e)
        contour = PointsList([P1,P2,P3,P4], meshSize)
        # create the surface associated to the contour
        surf_cyl = interface._Surfaces(contour)[0]
        # revole this surface 
        rev1 = factory.revolve([(2, surf_cyl)], 0,0,0,0,-1,0,angleCylinder/2, [])
        rev2 = factory.revolve([(2, surf_cyl)], 0,0,0,0,-1,0,-angleCylinder/2, [])
        # created volumes
        vol_rev = [rev1[1], rev2[1]]

        # fragment the cyliender entities in the blade volume
        factory.fragment(vol_blade, vol_rev)
        # get the volume entities
        ents = factory.getEntities(3)
        # creation of a cylinder to be used to remove the part of the blade protruding from the cylinder
        cylin = factory.addCylinder(0,-l,0,0,l*2,0,R)
        factory.cut(ents, [(3, cylin)])

    factory.synchronize()


    interface._Set_algorithm(elemType)

    # [gmsh.model.mesh.setRecombine(2, ent[1]) for ent in gmsh.model.getEntities(2)]


    factory.synchronize()


    parts = factory.getEntities()

    if repeat:

        N = int(np.pi*2//angleCylinder)

        for n in range(N-1):

            copy = factory.copy(parts)

            factory.rotate(copy, 0,0,0,0,1,0, angleCylinder*(n+1))            

    factory.synchronize()    
    

    if meshSize > 0:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), meshSize)

    interface._Set_PhysicalGroups(setPoints=False)
    
    interface._Meshing(3, elemType, isOrganised=False)

    # gmsh.write(Folder.Join([folder,"blade.msh"]))

    mesh = interface._Construct_Mesh()

    nodesCircle = mesh.Nodes_Conditions(lambda x,y,z: np.sqrt(x**2+z**2)<=R+1e-6)
    nodesUpper = mesh.Nodes_Conditions(lambda x,y,z: z==mesh.coordoGlob[:,2].max())

    nodesBlades = mesh.Nodes_Conditions(lambda x,y,z: np.sqrt(x**2+z**2)>=R+e+h)
    nodesCyl = mesh.Nodes_Conditions(lambda x,y,z: np.sqrt(x**2+z**2)<R+e+h)


    Display.Plot_Nodes(mesh, nodesBlades)

    elements = mesh.Elements_Nodes(nodesBlades)
    Display.Plot_Elements(mesh, nodesBlades, 2)

    mesh.groupElem.Set_Nodes_Tag(nodesCircle, 'blades')
    mesh.groupElem.Set_Nodes_Tag(nodesCyl, 'cylindre')

    # --------------------------------------------------------------------------------------------
    # Simu
    # --------------------------------------------------------------------------------------------

    material = Materials.Elas_Isot(mesh.dim)

    simu = Simulations.Simu_Displacement(mesh, material)

    uz = 1e-1
    simu.add_dirichlet(nodesCircle, [0]*mesh.dim, simu.Get_directions())
    simu.add_dirichlet(nodesUpper, [uz], ["z"])    
    simu.Solve()
    simu.Save_Iter()

    # --------------------------------------------------------------------------------------------
    # Results
    # --------------------------------------------------------------------------------------------

    deformFactor = uz / uz

    Display.Plot_Result(simu, 'uz', deformFactor)
    
    Display.Plot_Result(simu, 'Svm', deformFactor)

    Display.Plot_Mesh(simu, deformFactor)

    Display.Plot_BoundaryConditions(simu)
        
    Display.Plot_Model(mesh, alpha=0.1, showId=False)

    ax = Display.Plot_Elements(mesh, nodesCyl, c='green')
    Display.Plot_Elements(mesh, nodesBlades, c='grey', ax=ax)
    ax.legend()

    # PostProcessing.Make_Paraview(folder, simu)

    print(simu)

    # matFile = Folder.Join([folder, 'mesh.mat'])
    # msh = {
    #     'connect': np.asarray(mesh.connect+1, dtype=float),
    #     'coordo': mesh.coordoGlob,
    #     'bladesNodes': np.asarray(nodesBlades + 1, dtype=float),
    #     'bladesElems': np.asarray(mesh.Elements_Nodes(nodesBlades) + 1, dtype=float),
    #     'cylindreNodes': np.asarray(nodesCyl + 1, dtype=float),
    #     'cylindreElems': np.asarray(mesh.Elements_Nodes(nodesCyl) + 1, dtype=float),
    # }
    # scipy.io.savemat(matFile, msh)


    Display.plt.show()