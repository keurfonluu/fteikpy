# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

__all__ = [ "Ray" ]


class Ray:
    """
    Ray object.
    
    Parameters
    ----------
    z : ndarray or list
        Z coordinates of the ray.
    x : ndarray or list
        X coordinates of the ray.
    y : ndarray, list or None, default None
        Y coordinates of the ray.
    """
    
    def __init__(self, z, x, y = None):
        if not isinstance(z, (np.ndarray, list)) and np.asarray(z).ndim != 1:
            raise ValueError("z must be a 1-D ndarray or list")
        if not isinstance(x, (np.ndarray, list)) and np.asarray(x).ndim != 1:
            raise ValueError("x must be a 1-D ndarray or list")
        if len(z) != len(x):
            raise ValueError("z and x must have the same length")
        else:
            self._z = z
            self._x = x
        if y is not None:
            if not isinstance(y, (np.ndarray, list)) and np.asarray(y).ndim != 1:
                raise ValueError("y must be a 1-D ndarray or list")
            if len(y) != len(z):
                raise ValueError("z, x and y must have the same length")
            else:
                self._y = y
        else:
            self._y = y
                
    def plot(self, axes = None, figsize = (10, 8), plt_kws = {}):
        """
        Plot the ray from receiver to source.
        
        Parameters
        ----------
        axes : matplotlib axes or None, default None
            Axes used for plot.
        figsize : tuple, default (8, 8)
            Figure width and height if axes is None.
        plt_kws : dict
            Keyworded arguments passed to plot.
        
        Returns
        -------
        ax1 : matplotlib axes
            Axes used for plot.
        """
        if axes is not None and not isinstance(axes, Axes):
            raise ValueError("axes must be Axes")
        if not isinstance(figsize, (list, tuple)) or len(figsize) != 2:
            raise ValueError("figsize must be a tuple with 2 elements")
        if not isinstance(plt_kws, dict):
            raise ValueError("plt_kws must be a dictionary")
        
        if self._y is None:
            if axes is None:
                fig = plt.figure(figsize = figsize, facecolor = "white")
                ax1 = fig.add_subplot(1, 1, 1)
            else:
                ax1 = axes
            ax1.plot(self._x, self._z, **plt_kws)
            return ax1
        else:
            raise ValueError("plot unavailable in 3-D")
                
    @property
    def z(self):
        """
        ndarray or list
        Z coordinates of the ray.
        """
        return self._z
    
    @z.setter
    def z(self, value):
        self._z = value
        
    @property
    def x(self):
        """
        ndarray or list
        X coordinates of the ray.
        """
        return self._x
    
    @x.setter
    def x(self, value):
        self._x = value
        
    @property
    def y(self):
        """
        ndarray or list
        Y coordinates of the ray.
        """
        return self._y
    
    @y.setter
    def y(self, value):
        self._y = value
        
    @property
    def npts(self):
        """
        int
        Number of points that compose the ray.
        """
        return len(self._z)