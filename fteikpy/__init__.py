# -*- coding: utf-8 -*-

"""
FTeikPy is a Python module that computes accurate first arrival traveltimes in
2-D and 3-D heterogeneous isotropic velocity model.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

from .ttgrid import TTGrid
from .eikonal import Eikonal
from .ray import Ray
from .layered_model import lay2vel, lay2tt
from .bspline_model import bspline1, bspline2

__version__ = "1.4.3"
__all__ = [
    "TTGrid",
    "Eikonal",
    "Ray",
    "lay2vel",
    "lay2tt",
    "bspline1",
    "bspline2",
    ]
