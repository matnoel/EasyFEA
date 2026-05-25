# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np

# utilities
from ..Utilities import _params

# fem
from ..FEM._mesh import Mesh


class Solid:

    rho = _params.PositiveScalarParameter()

    def __init__(self, mesh: Mesh, rho: float = 1.0):
        assert mesh.dim == 3, "Must be a 3d mesh."
        self.__mesh = mesh
        self.rho = rho

    @property
    def mesh(self) -> Mesh:
        return self.__mesh

    @property
    def mass(self) -> float:
        return self.mesh.volume * self.rho

    def Get_M(self) -> np.ndarray:
        """
        Construct 6x6 rigid body mass matrix.

        Returns:
            M: 6x6 mass matrix [m*I₃ 0; 0 J_cm]
        """
        mesh = self.mesh
        rho = self.rho
        m = self.mass

        # Get center of mass from mesh.center (valid for uniform density)
        x0, y0, z0 = mesh.center

        # ===== STEP 1: Compute inertia about ORIGIN =====

        # Moments (diagonal terms)
        Ixx_O = mesh.groupElem.Integrate_e(lambda x, y, z: rho * (y**2 + z**2))
        Iyy_O = mesh.groupElem.Integrate_e(lambda x, y, z: rho * (x**2 + z**2))
        Izz_O = mesh.groupElem.Integrate_e(lambda x, y, z: rho * (x**2 + y**2))

        # Products (off-diagonal) - CORRECTED FORMULAS
        Ixy_O = mesh.groupElem.Integrate_e(lambda x, y, z: rho * x * y)
        Ixz_O = mesh.groupElem.Integrate_e(lambda x, y, z: rho * x * z)
        Iyz_O = mesh.groupElem.Integrate_e(lambda x, y, z: rho * y * z)

        # ===== STEP 2: Transform to CENTER OF MASS (parallel axis theorem) =====

        # Diagonal terms (subtract)
        Ixx_cm = Ixx_O - m * (y0**2 + z0**2)
        Iyy_cm = Iyy_O - m * (x0**2 + z0**2)
        Izz_cm = Izz_O - m * (x0**2 + y0**2)

        # Off-diagonal terms (add - note the sign!)
        Ixy_cm = Ixy_O + m * x0 * y0
        Ixz_cm = Ixz_O + m * x0 * z0
        Iyz_cm = Iyz_O + m * y0 * z0

        # ===== STEP 3: Build 3×3 inertia tensor =====
        # Note: NEGATIVE signs on all off-diagonal terms!

        J_cm = np.array(
            [
                [Ixx_cm, -Ixy_cm, -Ixz_cm],
                [-Ixy_cm, Iyy_cm, -Iyz_cm],
                [-Ixz_cm, -Iyz_cm, Izz_cm],
            ]
        )

        # ===== STEP 4: Assemble 6×6 mass matrix =====

        M = np.zeros((6, 6))

        # Top-left 3×3: translational inertia
        M[0, 0] = m
        M[1, 1] = m
        M[2, 2] = m

        # Bottom-right 3×3: rotational inertia
        M[3:6, 3:6] = J_cm

        return M
