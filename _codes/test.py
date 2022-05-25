import matplotlib.pyplot as plt
import matplotlib.collections
import numpy as np
import scipy.sparse as sp

import gmsh
import sys

import Affichage

Affichage.Clear()

isOpen = True

gmsh.initialize(sys.argv)

surf = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
pt1 = gmsh.model.occ.addPoint(0, 0.5, 0)
pt2 = gmsh.model.occ.addPoint(0.5, 0.5, 0)
line = gmsh.model.occ.addLine(pt1, pt2)

o, m = gmsh.model.occ.fragment([(0, pt1), (1, line)], [(2, surf)])
gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(2, [surf], 100)
gmsh.model.addPhysicalGroup(1, [line], 101)
gmsh.model.addPhysicalGroup(0, [pt1], 102)

gmsh.option.setNumber('Mesh.MeshSizeMin', 2)

gmsh.model.mesh.generate(2)

if isOpen:
    gmsh.plugin.setNumber("Crack", "PhysicalGroup", 101)
    gmsh.plugin.setNumber("Crack", "OpenBoundaryPhysicalGroup", 102)
    # gmsh.plugin.setNumber("Crack", "DebugView", 1)
    gmsh.plugin.run("Crack")

# print(gmsh.model.mesh.getNodes()) # does indeed contain 1 new node

# gmsh.fltk.run()

# Here I want to recover coordo and connect

elementTypes = gmsh.model.mesh.getElementTypes()
nodes, coord, parametricCoord = gmsh.model.mesh.getNodes()

nodes = np.array(nodes-1)
Nn = nodes.shape[0]

# Organise les noeuds du plus petits au plus grand
sortedIndices = np.argsort(nodes)
sortedNodes = nodes[sortedIndices]

decalage = sortedNodes - np.arange(Nn)

noeudsAChanger = np.where(decalage>0)[0]

changes = np.zeros((noeudsAChanger.shape[0],2), dtype=int)

Nodes = np.array(sortedNodes - decalage, dtype=int)

changes[:,0] = sortedNodes[noeudsAChanger]
changes[:,1] = noeudsAChanger

coord = coord.reshape(-1,3)

coordo = coord[sortedIndices]

# shapeCoordo = int(nodes.max()+1)

# lignes = np.repeat(nodes, 3)
# colonnes = np.repeat(np.array([[0,1,2]]), Nn, axis=0).reshape(-1)
# coordo = sp.csr_matrix((coord,(lignes, colonnes)), shape=(shapeCoordo,3))

fig, ax = plt.subplots()

for t, elemType in enumerate(elementTypes):

    elementTags, nodeTags = gmsh.model.mesh.getElementsByType(elemType)

    Ne = elementTags.shape[0]

    nodeTags = np.array(nodeTags-1, dtype=int)

    connect = nodeTags.reshape(Ne,-1)

    for indice in range(changes.shape[0]):
        old = changes[indice,0]
        new = changes[indice, 1]
        l, c = np.where(connect==old)
        connect[l, c] = new

    nodes = np.unique(nodeTags)

    Connect = nodes[connect]

    Nmax = nodes.max()
    assert Nmax <= (coordo.shape[0]-1), f"Nodes {Nmax} doesn't exist in coordo"
    
    match elemType:
        case 1: #SEG2
            connectFaces = connect            
            color='black'
            lw=2
            label='SEG2'
        case 2: #TRI3
            connectFaces = connect[:, [0,1,2,0]]
            color='red'
            lw=0.5
            label='TRI3'        
        case _:
            continue
    coordFaces = coordo[connectFaces,:2]

    coord=coordo[nodes]
    
    ax.scatter(coord[:,0], coord[:,1], label=label)

    pc = matplotlib.collections.LineCollection(coordFaces, edgecolor=color, lw=lw, label=label)
    ax.add_collection(pc)

    plt.legend()

    plt.pause(2)

gmsh.finalize()

plt.show()

