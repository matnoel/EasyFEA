# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Colours for exported values (matplotlib at call time)."""

import numpy as np

from ..Utilities import Folder
from ..Utilities._mpi import rank0_only
from ..Utilities._requires import Create_requires_decorator

requires_matplotlib = Create_requires_decorator("matplotlib")


@requires_matplotlib
def _Get_colors_for_values(
    values: np.ndarray,
    vMin: float = None,
    vMax: float = None,
    cmap: str = "jet",
) -> np.ndarray:
    """
    Generates RGB colors for scalar values using a matplotlib colormap.

    Parameters
    ----------
    values : np.ndarray
        1D array of scalar values to be mapped to colors
    vMin : float, optional
        Minimum value for normalization. If None, uses the minimum of values.
    vMax : float, optional
        Maximum value for normalization. If None, uses the maximum of values.
    cmap: str, optional
        the color map used near the figure, by default "jet" \n
        ["jet", "seismic", "binary", "viridis"] -> https://matplotlib.org/stable/tutorials/colors/colormaps.html

    Returns
    -------
    np.ndarray
        Array of RGB colors with shape (N, 3) and values in range [0, 1]

    Notes
    -----
    The function normalizes input values to [0, 1] and maps them through the specified
    colormap. The alpha channel is discarded, returning only RGB components.
    """

    import matplotlib.pyplot as plt

    assert isinstance(values, np.ndarray), "values must be a numpy array"
    assert values.ndim == 1, "values must be a 1D array"

    # Determine normalization bounds
    vMin = values.min() if vMin is None else vMin
    vMax = values.max() if vMax is None else vMax

    # Normalize values to [0, 1] range
    if vMax > vMin:
        normalizedValues = (values - vMin) / (vMax - vMin)
    else:
        normalizedValues = np.zeros_like(values)

    # Apply colormap and extract RGB components
    colormap = plt.get_cmap(cmap)
    colors = colormap(normalizedValues)[:, :3]  # Discard alpha channel

    return colors


@rank0_only
@requires_matplotlib
def _Save_colorbar(
    vMin: float,
    vMax: float,
    folder: str,
    filename="colorbar",
    cmap="jet",
    orientation="vertical",
    label="",
):
    """
    Generates and save colorbar.

    Parameters
    ----------
    vMin : float, optional
        Minimum value for normalization. If None, uses the minimum of values.
    vMax : float, optional
        Maximum value for normalization. If None, uses the maximum of values.
    folder : str, optional
        save folder, by default "".
    filename : str, optional
        filename, by default "colorbar"
    cmap: str, optional
        the color map used near the figure, by default "jet" \n
        ["jet", "seismic", "binary", "viridis"] -> https://matplotlib.org/stable/tutorials/colors/colormaps.html
    orientation : str, optional
        orientation, by default "vertical"
    label : str, optional
        label, by default ""
    """

    import matplotlib.pyplot as plt
    from matplotlib import colors, colorbar

    fig = plt.figure(figsize=(1.5, 6) if orientation == "vertical" else (6, 1.5))
    ax = fig.add_axes(
        (
            0.05,
            0.05,
            0.15 if orientation == "vertical" else 0.9,
            0.9 if orientation == "vertical" else 0.15,
        )
    )

    norm = colors.Normalize(vmin=vMin, vmax=vMax)

    cb = colorbar.ColorbarBase(
        ax, cmap=plt.get_cmap(cmap), norm=norm, orientation=orientation
    )

    # set explicit ticks
    nTicks = 5  # Number of tick marks
    tick_values = np.linspace(vMin, vMax, nTicks)
    cb.set_ticks(tick_values.tolist())

    # set label
    if label != "":
        cb.set_label(label, fontsize=12)

    path = Folder.Join(folder, filename + ".png", mkdir=True)
    plt.savefig(
        path,
        dpi=150,
        # bbox_inches="tight",
        transparent=True,
        pad_inches=0.1,
    )
    plt.close()
