# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
MeshOptim0
==========

Optimization of a happy mesh with quality criteria.
"""

# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np

from EasyFEA import Terminal, Matplotlib, ElemType, Mesh, PyVista
from EasyFEA.Geoms import Circle, CircleArc, Contour
from EasyFEA.FEM import Mesh_Optim

if __name__ == "__main__":
    Terminal.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # geom
    D = 1
    r = D * 1 / 4
    e = 0.1
    b = D * 0.1

    # criteria
    criteria = "aspect"
    quality = 0.8  # lower bound of the target quality
    ratio = 0.6  # the ratio of mesh elements that must satisfy the target
    iterMax = 20  # Maximum number of iterations
    coef = 1 / 2  # Scaling coefficient for the optimization process

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    elemType = ElemType.TRI3
    mS = e / 3 * 10

    # face
    circle = Circle((0, 0), D, mS)

    # eyes
    theta = 45 * np.pi / 180
    sin = np.sin(theta)
    cos = np.cos(theta)
    eye1 = Circle((-r * sin, r * cos), e, mS)
    eye2 = Circle((r * sin, r * cos), e, mS)

    # happy smile
    s = (D * 0.6) / 2
    s1 = CircleArc((-s, 0), (s, 0), (0, 0), coef=-1, meshSize=mS)
    s2 = CircleArc((s, 0), (s - e, 0), (s - e / 2, 0), coef=-1, meshSize=mS)
    s3 = CircleArc((s - e, 0), (-s + e, 0), (0, 0), meshSize=mS)
    s4 = CircleArc((-s + e, 0), (-s, 0), (-s + e / 2, 0), coef=-1, meshSize=mS)
    happy = Contour([s1, s2, s3, s4])

    inclusions = [eye1, eye2, happy]

    def DoMesh(refineGeom=None) -> Mesh:
        """Function used for mesh generation"""
        return circle.Mesh_2D(inclusions, elemType, [], [refineGeom])

    # Construct the initial mesh
    mesh = DoMesh()
    PyVista.Plot_Mesh(mesh).show()

    def Quality(mesh: Mesh) -> tuple[bool, np.ndarray]:
        """Converged once more than `ratio` of the elements reach `quality`; the error is the relative gap to it."""
        qual_e = mesh.Get_Quality(criteria, False)
        print(f"ratio = {np.mean(qual_e >= quality) * 100:.3f} %")
        return np.mean(qual_e >= quality) > ratio, np.abs(qual_e - quality) / quality

    mesh = Mesh_Optim(DoMesh, Quality, iterMax, coef)

    # ----------------------------------------------
    # Plot
    # ----------------------------------------------

    qual_e = mesh.Get_Quality(criteria, False)
    meshRatio = np.mean(qual_e >= quality)

    PyVista.Plot_Mesh(mesh).show()
    PyVista.Plot(
        mesh,
        qual_e,
        nodeValues=False,
        plotMesh=True,
        clim=(0, quality),
        cmap="viridis",
    ).show()

    axHist = Matplotlib.Init_Axes()

    axHist.hist(qual_e, 11, (0, 1))
    axHist.set_xlabel("quality")
    axHist.set_ylabel("elements")
    axHist.set_title(f"ratio = {meshRatio * 100:.3f} %")
    axHist.vlines([quality], [0], [meshRatio * mesh.Ne], color="red")

    plt.show()
