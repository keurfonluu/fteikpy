# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

__all__ = [ "Ray", "ray_coverage" ]


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
    
    
def ray_coverage(rays, grid_shape, grid_size, normed = False,
                 zmin = 0., xmin = 0., ymin = 0.):
    """
    Estimate ray coverage on a grid.
    
    Parameters
    ----------
    rays : list of Ray objects
        Rays to estimate coverage.
    grid_shape : tuple (nz, nx[, ny])
        Gris shape.
    grid_size : tuple (dz, dx[, dy])
        Grid size in meters.
    normed : bool, default False
        If False, returns the number of samples in each bin. If True, returns
        the bin density in percentage.
    zmin : int or float, default 0.
        Z axis first coordinate.
    xmin : int or float, default 0.
        X axis first coordinate.
    ymin : int or float, default 0.
        Y axis first coordinate. Only used if 3-D velocity model.
        
    Returns
    -------
    cover : ndarray of shape (nz, nx[, ny])
        Histogram of ray coverage.
    axes : list of lists (az, ax[, ay])
        Output 'cover' array axis.
    """
    # Check inputs
    if not isinstance(rays, (list, tuple)) or not np.all([isinstance(ray, Ray) for ray in rays]):
        raise ValueError("rays must be a list of Ray objects")
    if not np.all([isinstance(n, int) for n in grid_shape]) or len(grid_shape) not in [ 2, 3 ]:
        raise ValueError("grid_shape must be a tuple of integers of size 2 or 3")
    else:
        n_dim = len(grid_shape)
    if len(grid_size) != n_dim:
        raise ValueError("grid_size should be of length %d, got %d" % (n_dim, len(grid_size)))
    if np.any(np.array(grid_size) <= 0.):
        raise ValueError("elements in grid_size must be positive")
    if not isinstance(normed, bool):
        raise ValueError("normed must be True or False")
    if not isinstance(zmin, (int, float)):
        raise ValueError("zmin must be an integer or float")
    if not isinstance(xmin, (int, float)):
        raise ValueError("xmin must be an integer or float")
    if not isinstance(ymin, (int, float)):
        raise ValueError("ymin must be an integer or float")
    
    # Estimate ray coverage
    if n_dim == 2:
        nz, nx = grid_shape
        dz, dx = grid_size
        zmax = zmin + dz * nz
        xmax = xmin + dx * nx
        zray = np.concatenate([ ray.z for ray in rays ])
        xray = np.concatenate([ ray.x for ray in rays ])
        sample = np.vstack((zray, xray)).transpose()
        bounds = [ [ zmin, zmax ], [ xmin, xmax ] ]
    elif n_dim == 3:
        nz, nx, ny = grid_shape
        dz, dx, dy = grid_size
        zmax = zmin + dz * nz
        xmax = xmin + dx * nx
        ymax = ymin + dy * ny
        zray = np.concatenate([ ray.z for ray in rays ])
        xray = np.concatenate([ ray.x for ray in rays ])
        yray = np.concatenate([ ray.y for ray in rays ])
        sample = np.vstack((zray, xray, yray)).transpose()
        bounds = [ [ zmin, zmax ], [ xmin, xmax ], [ ymin, ymax ] ]
    cover, edges = np.histogramdd(sample, grid_shape, bounds)
    if normed:
        cover /= np.sum(cover)
    axes = [ e[:-1] for e in edges ]
    return cover, axes