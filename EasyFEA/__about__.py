# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

try:
    # Python 3.8+
    from importlib import metadata

    __version__ = metadata.version("EasyFEA")
except ImportError:
    __version__ = "unknown"
