# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
from ._bspline import bspline as bspl

__all__ = [ "bspline1", "bspline2" ]


def bspline1(x, y, xq, order = 4, check = True):
    """
    1-D B-Spline approximation.
    
    Parameters
    ----------
    x : ndarray or list
        1-D array of real values.
    y : ndarray or list
        1-D array of real values. The length of y along the interpolation axis
        must be equal to the length of x.
    xq : ndarray of list
        1-D array of real values to query.
    order : int, default 4
        Order of spline. Order should be less than the number of control
        points.
    check : bool
        Check inputs consistency. Disable checking if you need to call
        'bspline1' a lot of times for better performance.
    
    Returns
    -------
    yq : ndarray of list
        1-D array of real values at query points.
    """
    # Check inputs
    if check:
        if not isinstance(x, (np.ndarray, list)) and not np.asarray(x).ndim != 1:
            raise ValueError("x must be a 1-D ndarray")
        if not isinstance(y, (np.ndarray, list)) and not np.asarray(y).ndim != 1:
            raise ValueError("y must be a 1-D ndarray")
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if not isinstance(xq, (np.ndarray, list)) and not np.asarray(xq).ndim != 1:
            raise ValueError("xq must be a 1-D ndarray")
        if xq[0] < x[0] or xq[-1] > x[-1]:
            raise ValueError("elements in xq should be within x.min and x.max")
        if not isinstance(order, int) or not 2 <= order <= len(x):
            raise ValueError("order should be an integer in [ %d, %d ], got %d" % (2, len(x), order))
    
    # 1-D B-Spline approximation
    return bspl.bspline1(x, y, xq, order = order)


def bspline2(x, y, z, xq, yq, order = 4, n_threads = 1, check = True):
    """
    2-D B-Spline approximation.
    
    Parameters
    ----------
    x, y : ndarray
        1-D array of real values.
    z : ndarray
        2-D array of real values. The shape of z along the interpolation
        axis must be consistent with the lengths of x and y.
    xq, yq : ndarray
        1-D or 2-D array of real values to query.
    order : int, default 4
        Order of spline. Order should be less than the number of control
        points in both dimensions.
    n_threads : int, default 1
        Number of threads to pass to OpenMP.
    check : bool
        Check inputs consistency. Disable checking if you need to call
        'bspline2' a lot of times for better performance.
    
    Returns
    -------
    zq : ndarray
        2-D array of real values at query points.
    """
    # Check inputs
    if check:
        if not isinstance(x, (np.ndarray, list)) and not np.asarray(x).ndim != 1:
            raise ValueError("x must be a 1-D ndarray")
        if not isinstance(y, (np.ndarray, list)) and not np.asarray(y).ndim != 1:
            raise ValueError("y must be a 1-D ndarray")
        if not isinstance(z, np.ndarray) and z.ndim != 2:
            raise ValueError("z must be a 2-D ndarray")
        if z.shape != ( len(y), len(x) ):
            raise ValueError("z must be of shape (%d, %d), got %s" \
                             % (len(y), len(x), z.shape))
        if not isinstance(xq, (np.ndarray, list)) and not np.asarray(xq).ndim not in [ 1, 2 ]:
            raise ValueError("xq must be a 1-D or 2-D ndarray")
        if not isinstance(yq, (np.ndarray, list)) and not np.asarray(yq).ndim not in [ 1, 2 ]:
            raise ValueError("yq must be a 1-D or 2-D ndarray")
        if np.asarray(xq).ndim != np.asarray(yq).ndim:
            raise ValueError("inconsistent ndim for xq and yq")
        if np.asarray(xq).ndim == 1:
            if xq[0] < x[0] or xq[-1] > x[-1]:
                raise ValueError("elements in xq should be within x.min and x.max")
            if yq[0] < y[0] or yq[-1] > y[-1]:
                raise ValueError("elements in yq should be within y.min and y.max")
        elif np.asarray(xq).ndim == 2:
            if xq.shape != yq.shape:
                raise ValueError("xq and yq must have the same shape")
        if not 2 <= order <= min(len(x), len(y)):
            raise ValueError("order should be in [ %d, %d ], got %d" % (2, min(len(x), len(y)), order))
        if not isinstance(order, int) or not isinstance(n_threads, int) or n_threads < 1:
            raise ValueError("n_threads must be atleast 1, got %s" % n_threads)
            
    # 2-D B-Spline approximation
    if np.asarray(xq).ndim == 1:
        XQ, YQ = np.meshgrid(xq, yq)
        return bspl.bspline2(x, y, z, XQ, YQ, order = order, n_threads = n_threads)
    elif np.asarray(xq).ndim == 2:
        return bspl.bspline2(x, y, z, xq, yq, order = order, n_threads = n_threads)