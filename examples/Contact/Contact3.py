# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Contact3
========

Reproduces the "From indenter's shape to pressure" insert of Yastrebov, *Contact Mechanics and Elements of Tribology* (§1.5). A rigid indenter is pressed into an elastic half-space (2D, **plane strain**) and the finite-element contact pressure is compared to the closed-form solutions, with ``E* = E/(1-ν²)``:

- **parabola** ``z = x²/2R`` (Hertz line contact): ``p(x) = p₀√(1-x²/a²)``,
  ``p₀ = a·E*/(2R)`` — a semi-ellipse.
- **wedge** ``z = |x|·tanθ``: ``p(x) = (E*·tanθ/π)·arccosh(a/|x|)`` — a log peak.

The contact half-width ``a`` is read from the FE solution; the analytical curve is plotted with that same ``a`` (so the comparison is shape + magnitude).

The contact pressure is obtained directly from the penalty contact as ``p = εₙ⟨-g⟩`` on the surface (``RigidContact`` from ``_utils.py``). Each indenter spans the whole surface and rises away from the contact, so points outside the contact patch simply have a positive gap (no spurious contact).

ACCURACY
--------
``p = εₙ⟨-g⟩`` is the contact traction at convergence. Two resolutions govern the match: the body mesh AND how finely the indenter is sampled across the contact patch. A curved indenter is gauged by nearest-sample projection, so it must be well resolved over the (small) contact — hence the body mesh is refined near x=0 and the parabola
samples are clustered there. With both, each case matches the analytical pressure to ~1% over the inner patch (the wedge is piecewise-linear, so essentially exact). The penalty still amplifies the discrete gap (an excessive penalty re-roughens the profile), and the flat-punch edge singularity is not reproducible with a penalty method.
"""

import numpy as np
import matplotlib.pyplot as plt

from EasyFEA import Terminal, Folder, Models, ElemType, Mesh, PyVista
from EasyFEA.Geoms import Domain, Points, Circle
from EasyFEA.FEM import MatrixType

from _utils import RigidContact


# ----------------------------------------------
# Rigid indenter profiles z(x), spanning [0, W] and rising away from x=0
# ----------------------------------------------
def parabola_indenter() -> Mesh:
    # cluster samples near x=0: a curved indenter is gauged by nearest-sample
    # projection, so it must be resolved finely over the small contact (a << W),
    # otherwise that sampling — not the body mesh — dominates the pressure error
    xs = W * np.linspace(0, 1, 100) ** 2
    geom = Points(
        [
            *[(x, x**2 / (2 * Rc)) for x in xs],  # lower bound
            *[(W, W**2 / (2 * Rc) + 2), (0, 2)],
        ]
    )
    mesh = geom.Mesh_2D([], ElemType.TRI3)
    nodes = mesh.Nodes_Conditions(lambda x, y, z: y <= x**2 / (2 * Rc) + 1e-9)
    mesh.Set_Tag(nodes, "contact")
    return mesh


def wedge_indenter() -> Mesh:
    geom = Points(
        [
            (0, 0),
            (W, W * np.tan(theta)),
            (W, W * np.tan(theta) + 2),
            (0, 2),
        ]
    )
    mesh = geom.Mesh_2D([], ElemType.TRI3)
    nodes = mesh.Nodes_Conditions(lambda x, y, z: y <= x * np.tan(theta) + 1e-9)
    mesh.Set_Tag(nodes, "contact")
    return mesh


def parabola_analytic(x, a):
    return a * Es / (2 * Rc) * np.sqrt(np.clip(1 - (x / a) ** 2, 0, None))


def wedge_analytic(x, a):
    return Es * np.tan(theta) / np.pi * np.arccosh(np.clip(a / np.abs(x), 1.0, None))


indenter_cases = {
    "parabola": (parabola_indenter, parabola_analytic),
    "wedge": (wedge_indenter, wedge_analytic),
}


def contact_pressure(simu: RigidContact, indenter: Mesh):
    """Penalty contact pressure ``εₙ⟨-g⟩`` and x-coordinate on the surface Gauss points."""

    matrixType = MatrixType.mass

    list_p = []
    list_x = []

    for group1d in simu.mesh.Get_list_groupElem(1):
        N_pg = group1d.Get_N_pg(matrixType)[:, 0, :]
        # x = X + u
        elements = group1d.Get_Elements_Tag("top")
        X_e_pg = group1d.Get_GaussCoordinates_e_pg(matrixType)[elements]
        u_e_pg = simu.displacement.reshape(simu.mesh.Nn, 2)
        x_e_pg = X_e_pg.copy()
        x_e_pg[..., :2] += np.einsum(
            "pn,enc->epc", N_pg, u_e_pg[group1d.connect[elements]]
        )
        list_x.extend(x_e_pg[..., 0].ravel())
        # get pressure
        for groupIndent in indenter.Get_list_groupElem(1):
            gap_e_pg, _ = groupIndent._Get_gap_and_normal(
                x_e_pg=x_e_pg,
                elements=groupIndent.Get_Elements_Tag("contact"),
                coord=indenter.center,
                matrixType=matrixType,
            )
            p_e_pg = simu.penalty * np.maximum(-gap_e_pg, 0.0)
            list_p.extend(p_e_pg.ravel())

    return np.asarray(list_p), np.asarray(list_x)


if __name__ == "__main__":
    Terminal.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    folder = Folder.Results_Dir()
    result = "Svm"

    E, v = 210000.0, 0.3
    Es = E / (1 - v**2)

    W, D = 6.0, 6.0  # elastic half-space (symmetry about x=0): wide & deep vs contact
    meshSize = W / 10
    N = 12  # indentation steps
    penalty = 100 * Es

    Rc = 6.0  # parabola radius of curvature
    theta = np.deg2rad(6)  # wedge half-angle
    indeter_delta = {
        "parabola": 0.05,
        "wedge": 0.12,
    }  # indentation depth per case

    # ----------------------------------------------
    # Elastic half-space (surface at y=0)
    # ----------------------------------------------
    body = Domain((0, -D), (W, 0), meshSize)

    refineGeoms = [
        Circle((0, 0), W * coef * 2, W / N)
        for coef, N in zip(
            [0.2, 0.4],
            [200, 100],
        )
    ]

    mesh = body.Mesh_2D([], ElemType.TRI3, refineGeoms=refineGeoms)
    nodes_bottom = mesh.Nodes_Conditions(lambda x, y, z: y == -D)
    nodes_sym = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodes_top = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
    mesh.Set_Tag(nodes_top, "top")

    PyVista.Plot_Mesh(mesh).show()

    material = Models.Elastic.Isotropic(2, E=E, v=v, planeStress=False)

    # ----------------------------------------------
    # Solve each case and compare FE pressure to theory
    # ----------------------------------------------

    fig, axes = plt.subplots(1, len(indenter_cases), figsize=(11, 4))
    for ax, (name, (build, analytic)) in zip(axes, indenter_cases.items()):
        simu = RigidContact(mesh, material, penalty)
        indenter = build()

        list_indeter = [indenter]
        delta = indeter_delta[name]

        print(f"\n[{name}] pressing the rigid indenter (Newton per step):")
        for i in range(N):
            # update indeter
            indenter = list_indeter[0]
            indenter = indenter.copy()
            indenter.Translate(dy=-(i + 1) / N * delta)
            list_indeter.append(indenter)
            simu._contactMesh = indenter
            # solve contact
            simu.Bc_Init()
            simu.add_dirichlet(nodes_bottom, [0, 0], simu.Get_unknowns())
            simu.add_dirichlet(nodes_sym, [0], ["x"])
            simu.Solve()
            simu.Save_Iter()

        pg, xg = contact_pressure(simu, indenter)
        order = np.argsort(xg)
        xg, pg = xg[order], pg[order]
        a = xg[pg > 1e-6 * pg.max()].max()  # FE contact half-width
        # relative error over the inner patch (skip the singular center & edge),
        # normalised by the FE peak pressure
        band = (xg > 0.1 * a) & (xg < 0.85 * a)
        rel = np.abs(pg[band] - analytic(xg[band], a)) / pg.max()
        print(
            f"\n[{name}] a={a:.3f}  error vs analytical over 0.1a-0.85a:  "
            f"mean {rel.mean() * 100:.0f}%   max {rel.max() * 100:.0f}%"
        )

        # ----------------------------------------------
        # Results
        # ----------------------------------------------

        ax.plot(xg, pg, "o", ms=3, label="FE  εₙ⟨-g⟩")
        xa = np.linspace(1e-3, a, 200)
        ax.plot(xa, analytic(xa, a), "k-", label="analytical")
        ax.set_title(f"{name}  (a = {a:.3f})")
        ax.set_xlabel("x")
        ax.set_ylabel("contact pressure")
        ax.set_xlim(0, 1.6 * a)
        ax.legend()
        ax.grid(True)

        def Plot_Iter(plotter: PyVista.pv.Plotter, n):
            simu.Set_Iter(n)
            PyVista.Plot(
                simu, result, 1, color="k", nColors=10, show_grid=True, plotter=plotter
            )
            PyVista.Plot(list_indeter[n], color="gray", plotter=plotter, linewidth=0.1)
            PyVista._setCameraPosition(plotter, 2)

        PyVista.Movie_func(Plot_Iter, N, folder=folder, filename=f"{name}.gif")

    fig.tight_layout()

    plotter = PyVista._Plotter()
    Plot_Iter(plotter, -1)
    plotter.show()

    plt.show()
