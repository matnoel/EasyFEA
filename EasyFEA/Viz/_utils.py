# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Helpers shared by the viewers."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..Geoms._geom import _Geom


def _Flatten_geoms(geoms: tuple) -> list[_Geom]:
    """Unpacks the lists among `geoms`."""
    return [g for geom in geoms for g in (geom if isinstance(geom, list) else [geom])]
