# -*- coding: utf-8 -*-

"""
FTeikPy is a Python module that computes accurate first arrival traveltimes in
2-D and 3-D heterogeneous isotropic velocity model.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

from .ttgrid import TTGrid
from .eikonal import Eikonal
from .layered_model import lay2vel, lay2tt

__version__ = "1.3.1"
__all__ = [ "TTGrid", "Eikonal", "lay2vel", "lay2tt" ]
