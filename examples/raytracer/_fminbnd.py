# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np

__all__ = [ "fminbnd" ]


def fminbnd(f, lower, upper, eps = 1e-4, args = (), kwargs = {}):
    # Define func
    func = lambda x: f(x, *args, **kwargs)
    
    # Golden ratio
    gr = 0.61803398874989479
    
    # Golden section search
    x1 = np.array([ lower ])
    x2 = np.array([ upper ])
    x3 = x2 - gr * (x2 - x1)
    x4 = x1 + gr * (x2 - x1)
    while np.abs(x3 - x4) > eps:
        if func(x3) < func(x4):
            x2 = x4
        else:
            x1 = x3
        x3 = x2 - gr * (x2 - x1)
        x4 = x1 + gr * (x2 - x1)
    xmin = 0.5 * (x1 + x2)
    return xmin, func(xmin)