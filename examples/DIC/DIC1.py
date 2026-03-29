# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.


"""
DIC1
====

Performs digital image correlation analyses on images obtained experimentally in this scientific article: https://univ-eiffel.hal.science/hal-05115523.

The images were downloaded from `Recherche Data Gouv <https://doi.org/10.57745/NGOKFP>`_ and are distributed under the `Etalab Open License 2.0 <https://spdx.org/licenses/etalab-2.0.html>`_. For further details, see the `DATA_LICENSE.md <https://gitlab.univ-eiffel.fr/collaboration-msme-fcba/spruce-params/-/blob/V5.1/DATA_LICENSE.md?ref_type=tags>`_ file.

Further implementation details are available in my `PhD thesis <https://hal.univ-lorraine.fr/MSME_MECA/tel-04866760v1>`_ (Section 2, Chapter 2, written in French).
"""
# sphinx_gallery_thumbnail_number = -2

import matplotlib.pyplot as plt
from PIL import Image  # matplotlib dependency
import numpy as np

from EasyFEA import Display, Folder, Models
from EasyFEA.Simulations import Elastic, DIC
from EasyFEA.Geoms import Circle, Domain


def Plot_Result(simu: Elastic, result: str, img=None, title="", plotMesh=True):
    ax = Display.Init_Axes()
    if img is not None:
        ax.imshow(img, cmap="gray")
    Display.Plot_Result(simu, result, title=title, plotMesh=plotMesh, ax=ax)


if __name__ == "__main__":

    # ----------------------------------------------
    # Config
    # ----------------------------------------------

    useRegularization = True

    imagesDir = Folder.Join(Folder.Dir(), "_images1")

    # ----------------------------------------------
    # Load images
    # ----------------------------------------------

    images = [
        Folder.Join(imagesDir, image)
        for image in sorted(Folder.os.listdir(imagesDir))
        if image.endswith(".png")
    ]

    imgRef = np.asarray(Image.open(images[0]), dtype=int)

    ax = Display.Init_Axes()
    ax.imshow(imgRef, cmap="gray")

    # ----------------------------------------------
    # Create Mesh
    # ----------------------------------------------

    imgScale = 0.11479591836734694  # mm/px

    x0, y0 = 35, 25
    x1, y1 = 423, 814
    meshSize = (x1 - x0) / 20
    contour = Domain((x0, y0), (x1, y1), meshSize)
    contour.Plot(ax)

    xC, yC, radius = DIC.Get_Circle(imgRef, 30.0, [(150, 350), (350, 500)])
    circle = Circle((xC, yC), 2 * radius, meshSize)
    circle.Plot(ax)

    mesh = contour.Mesh_2D([circle])

    Display.Plot_Mesh(mesh, alpha=0, edgecolor="black", ax=ax)
    ax.legend()

    # ----------------------------------------------
    # Conduct DIC analyses
    # ----------------------------------------------

    lr = int(np.sqrt(0.25) * 8 * meshSize) if useRegularization else 0
    dic = DIC(mesh, imgRef, lr, verbosity=True)

    # Generate an elastic simulation to streamline the visualization of results.
    simu = Elastic(mesh, Models.Elastic.Isotropic(2, E=1, v=0.3))

    for image in images[1:]:

        imgStr = image.removeprefix(imagesDir)
        print(f"\n {imgStr}")
        img = np.asarray(Image.open(image), dtype=int)

        u = dic.Solve(img)

        simu._Set_solutions(simu.problemType, u * imgScale)
        simu.Save_Iter()

        if image == images[-1]:
            Plot_Result(simu, "ux", img, f"{imgStr} ux")
            Plot_Result(simu, "uy", img, f"{imgStr} uy")
            Plot_Result(simu, "Exx", img, f"{imgStr} Exx")

    plt.show()
