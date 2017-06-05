# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from ._interpolate import interp2, interp3

__all__ = [ "TTGrid" ]


class TTGrid:
    """
    Traveltime grid.
    
    Parameters
    ----------
    grid: ndarray of shape (nz, nx[, ny])
        Traveltime grid.
    source: ndarray
        Source coordinates (Z, X[, Y]).
    zaxis: ndarray
        Z coordinates of the grid.
    xaxis: ndarray
        X coordinates of the grid.
    yaxis: ndarray
        Y coordinates of the grid.
    """
    
    def __init__(self, grid = None, source = None, zaxis = None, xaxis = None,
                 yaxis = None):
        if grid is not None and not isinstance(grid, np.ndarray) \
            and not grid.ndim in [ 2, 3 ]:
            raise ValueError("grid must be a 2-D or 3-D ndarray")
        if grid is not None and np.any(grid <= 0.):
            raise ValueError("grid must be positive")
        else:
            self._grid = grid
            if grid is not None:
                self._shape = grid.shape
                self._n_dim = grid.ndim
        if grid is not None and len(source) != len(self._shape):
            raise ValueError("source should have %d coordinates, got %d" \
                             % (len(self._shape), len(source)))
        else:
            self._source = source
        self._zaxis = zaxis
        self._xaxis = xaxis
        self._yaxis = yaxis
        return
    
    def get(self, zq, xq, yq = None, check = True):
        """
        Get the traveltime value given grid point coordinate.
        
        Parameters
        ----------
        zq: scalar
            Z coordinate of the grid point.
        xq: scalar
            X coordinate of the grid point.
        yq: scalar or None, default None
            Y coordinate of the grid point. yq should be None if grid is
            a 2-D array.
        check: bool
            Check input zq, xq and yq to avoid crashes when interpolating
            outside the grid (as Fortran interpolation code will try to access
            inexistent values). Disable checking if you need to call 'get'
            method a lot of times for better performance.
            
        Returns
        -------
        tq: scalar or ndarray
            Traveltime value(s).
            
        Notes
        -----
        The method uses velocity interpolation to estimate more accurate
        traveltimes (should be exact in a homogenous velocity model).
        """
        if check:
            if not isinstance(zq, (int, float)):
                raise ValueError("zq must be a scalar")
            if not isinstance(xq, (int, float)):
                raise ValueError("xq must be a scalar")
            if not 0. <= zq <= self._zaxis[-1]:
                raise ValueError("zq out of bounds")
            if not 0. <= xq <= self._xaxis[-1]:
                raise ValueError("xq out of bounds")
            if yq is not None:
                if not isinstance(yq, (int, float)):
                    raise ValueError("yq must be a scalar")
                if not 0. <= yq <= self._yaxis[-1]:
                    raise ValueError("yq out of bounds")
            
        if self._n_dim == 2:
            tq = interp2(self._source, self._zaxis, self._xaxis, self._grid, zq, xq)
        elif self._n_dim == 3:
            tq = interp3(self._source, self._zaxis, self._xaxis, self._yaxis, self._grid,
                         zq, xq, yq)
        return tq
    
    def plot(self, axes = None, n_levels = 20, figsize = (10, 8), kwargs = {}):
        """
        Plot the traveltime grid.
        
        Parameters
        ----------
        axes: matplotlib axes or None, default None
            Axes used for plot.
        n_levels: int, default 20
            Number of levels for contour.
        figsize: tuple, default (8, 8)
            Figure width and height if axes is None.
        
        Returns
        -------
        ax1: matplotlib axes
            Axes used for plot.
        """
        if self._n_dim == 2:
            if axes is None:
                fig = plt.figure(figsize = figsize, facecolor = "white")
                ax1 = fig.add_subplot(1, 1, 1)
            else:
                ax1 = axes
            ax1.contour(self._xaxis, self._zaxis, self._grid, n_levels, **kwargs)
            return ax1
        else:
            raise ValueError("plot unavailable for 3-D grid")

    @property
    def grid(self):
        """
        ndarray of shape (nz, nx[, ny])
        Traveltime grid.
        """
        return self._grid
    
    @grid.setter
    def grid(self, value):
        self._grid = value
        
    @property
    def shape(self):
        """
        tuple (nz, nx[, ny])
        Traveltime grid's shape.
        """
        return self._shape
    
    @shape.setter
    def shape(self, value):
        self._shape = value
        
    @property
    def n_dim(self):
        """
        int (2 or 3)
        Traveltime grid's dimension.
        """
        return self._n_dim
    
    @n_dim.setter
    def n_dim(self, value):
        self._n_dim = value
        
    @property
    def source(self):
        """
        tuple of size n_dim
        Source coordinates (Z, X[, Y]).
        """
        return self._source
    
    @source.setter
    def source(self, value):
        self._source = value
    
    @property
    def zaxis(self):
        """
        ndarray of size nz
        Z coordinates of the grid.
        """
        return self._zaxis
    
    @zaxis.setter
    def zaxis(self, value):
        self._zaxis = value
    
    @property
    def xaxis(self):
        """
        ndarray of size nx
        X coordinates of the grid.
        """
        return self._xaxis
    
    @xaxis.setter
    def xaxis(self, value):
        self._xaxis = value
    
    @property
    def yaxis(self):
        """
        ndarray of size ny
        Y coordinates of the grid.
        """
        return self._yaxis
    
    @yaxis.setter
    def yaxis(self, value):
        self._yaxis = value