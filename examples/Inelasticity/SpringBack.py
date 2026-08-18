# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""
SpringBack
==========

Kinematic hardening on a mesh, where it changes the answer.

Bend a strip past yield, release it, and ask how much curvature it keeps. On release the outer
fibres go into reverse: with **kinematic** hardening the surface has moved, so reverse yielding
starts early and the strip recovers further. Isotropic hardening therefore under-predicts
springback, which is why sheet forming is simulated with kinematic or Chaboche models.

Springback is taken as the curvature at which the section carries no moment, found by ramping
the curvature back down and interpolating :math:`M(\kappa)` to zero. Releasing the end
constraint in one step instead would ask the Newton to jump from a fully constrained end to a
free one in a single increment.
"""

# sphinx_gallery_thumbnail_number = 1

from enum import Enum

import numpy as np

from EasyFEA import Matplotlib, ElemType, Models, Simulations
from EasyFEA.Geoms import Domain
from EasyFEA.Models.Elastic._laws import Isotropic

# ----------------------------------------------
# Configuration
# ----------------------------------------------
L, h, w = 80.0, 20.0, 5.0  # mm
E, v = 210000.0, 0.3  # MPa
sigma_y = 250.0  # MPa

kappa_e = 2 * sigma_y / (E * h)  # curvature at first yield
kappa_max = 4 * kappa_e  # bend well past yield
M_e = sigma_y * w * h**2 / 6

elastic = Isotropic(3, E=E, v=v)
KH = Models.KinematicHardening


class Laws(str, Enum):
    Isotropic = "isotropic"
    ArmstrongFrederick = "kinematic (ArmstrongFrederick)"
    Chaboche = "kinematic (Chaboche, 3)"

    def __str__(self):
        return self.name


# tuned to the same uniaxial hardening modulus, so the loading curves nearly coincide
laws = {
    Laws.Isotropic: dict(hardening=Models.IsotropicHardening.Linear(20000.0)),
    Laws.ArmstrongFrederick: dict(kinematic=KH.ArmstrongFrederick(60000.0, 500.0)),
    Laws.Chaboche: dict(
        kinematic=KH.Chaboche((60000.0, 500.0), (20000.0, 100.0), (2000.0, 0.0))
    ),
}

# ----------------------------------------------
# Mesh
# ----------------------------------------------
# the imposed u_x = -kappa L y makes the strain exactly linear in y, which QUAD8 reproduces on
# any mesh, so a coarse one costs nothing: h/4 through h/12 give the same springback for all
# three laws. h/3 does not -- Chaboche shifts by 3 points, since a back-stress with several
# time constants needs the plastic front through the thickness resolved a little better.
domain = Domain((0, -h / 2), (L, h / 2), h / 4)
mesh = domain.Mesh_2D([], ElemType.QUAD8, isOrganised=True)

nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)
origin = mesh.Nodes_Conditions(lambda x, y, z: (x == 0) & (np.abs(y) < 1e-9))
nodesMid = mesh.Nodes_Conditions(lambda x, y, z: np.isclose(x, L / 2))

order = np.argsort(mesh.coord[nodesMid, 1])
yMid = mesh.coord[nodesMid, 1][order]

# bend, then unload past zero moment into reverse
nLoad, nUnload = 20, 48
path = np.concatenate(
    [
        np.linspace(kappa_max / nLoad, kappa_max, nLoad),
        np.linspace(kappa_max, -0.5 * kappa_max, nUnload)[1:],
    ]
)


def Bend(material):
    """Walks the curvature path, returning the simulation and (kappa, moment) along it."""
    simu = Simulations.Behaviour(mesh, material)
    kappa, moment = [], []
    for k in path:
        simu.Bc_Init()
        simu.add_dirichlet(nodesX0, [0], ["x"])
        simu.add_dirichlet(origin, [0], ["y"])
        simu.add_dirichlet(nodesXL, [lambda x, y, z: -k * L * y], ["x"])
        simu.Solve()
        simu.Save_Iter()

        sig_xx = simu.Result("Sxx")[nodesMid][order]
        kappa.append(k)
        moment.append(np.trapezoid(sig_xx * yMid, yMid) * w)
    return simu, np.array(kappa), np.array(moment)


# ----------------------------------------------
# Solve
# ----------------------------------------------
print(f"bent to kappa/kappa_e = {kappa_max / kappa_e:.1f}, then released\n")
results, springback, simus, released = {}, {}, {}, {}
for label, kwargs in laws.items():
    material = Models.Behaviour(
        2,
        elastic,
        yieldSurface=Models.Yield.VonMises(sigma_y),
        thickness=w,
        planeStress=True,
        **kwargs,
    )
    simu, kappa, moment = Bend(material)
    results[label] = (kappa, moment)
    simus[label] = simu

    # the released state is where the section carries no moment
    unload = slice(nLoad - 1, None)
    k_un, M_un = kappa[unload], moment[unload]
    cross = int(np.where(np.diff(np.sign(M_un)))[0][0])
    kappa_res = np.interp(
        0.0, [M_un[cross + 1], M_un[cross]], [k_un[cross + 1], k_un[cross]]
    )
    springback[label] = 100 * (kappa_max - kappa_res) / kappa_max
    released[label] = nLoad - 1 + cross  # the saved iteration nearest to M = 0
    print(
        f"  {label!s:20s} residual curvature {kappa_res / kappa_e:5.3f} kappa_e"
        f"   springback {springback[label]:5.2f} %"
    )

# the point of the example: reverse yielding starts earlier with a back-stress, so the
# unloading is softer and the strip recovers further
assert (
    springback[Laws.Isotropic] < springback[Laws.ArmstrongFrederick]
), "isotropic hardening did not under-predict springback"
assert (
    springback[Laws.ArmstrongFrederick] < springback[Laws.Chaboche]
), "the linear Chaboche component did not soften the unloading further"

# ----------------------------------------------
# Results
# ----------------------------------------------
ax = Matplotlib.Init_Axes()
for label, (kappa, moment) in results.items():
    ax.plot(kappa / kappa_e, moment / M_e, lw=1.2, label=label)
ax.axhline(0.0, ls=":", c="k", lw=0.8)
ax.set_xlabel(r"$\kappa/\kappa_e$")
ax.set_ylabel("$M/M_e$")
ax.set_title("Bend and release: the unloading branch decides the springback")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# zoom on where the unloading branches cross zero moment
ax = Matplotlib.Init_Axes()
for label, (kappa, moment) in results.items():
    unload = slice(nLoad - 1, None)
    ax.plot(kappa[unload] / kappa_e, moment[unload] / M_e, "o-", ms=2.5, label=label)
ax.axhline(0.0, ls=":", c="k", lw=0.8)
ax.set_xlim(1.5, 3.2)
ax.set_ylim(-0.4, 0.6)
ax.set_xlabel(r"$\kappa/\kappa_e$")
ax.set_ylabel("$M/M_e$")
ax.set_title("Released state: $M = 0$, reached at a different curvature each time")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# the strip itself, bent and then released. Springback is not elastic recovery of a uniform
# state: the outer fibres yielded and the core did not, so once the moment is back to zero the
# section is left holding self-equilibrating residual stresses.
simu = simus[Laws.Chaboche]

simu.Set_Iter(nLoad - 1)
Matplotlib.Plot(
    simu,
    "p",
    plotMesh=True,
    deformFactor=1,
    ncolors=11,
    title=f"{Laws.Chaboche}: plastic strain at peak curvature",
)

simu.Set_Iter(released[Laws.Chaboche])
Matplotlib.Plot(
    simu,
    "Sxx",
    plotMesh=True,
    deformFactor=1,
    ncolors=11,
    title=f"{Laws.Chaboche}: residual $\\sigma_{{xx}}$ once released [MPa]",
)

print(
    "\nIsotropic hardening under-predicts springback; that is why forming uses Chaboche."
)

Matplotlib.plt.show()
