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
        
    def get(self, zq, xq, yq = None):
        """
        Get the traveltime value given grid point coordinates.
        
        Parameters
        ----------
        zq: scalar or ndarray
            Z coordinate(s) of the grid point(s).
        xq: scalar or ndarray
            X coordinate(s) of the grid point(s).
        yq: scalar or ndarray or None, default None
            Y coordinate(s) of the grid point(s). yq should be None if grid is
            a 2-D array.
            
        Returns
        -------
        tq: scalar or ndarray
            Traveltime value(s).
        """
        # Check inputs
        if not isinstance(zq, (int, float, np.ndarray)):
            raise ValueError("zq must be a scalar or a ndarray")
        if not isinstance(xq, (int, float, np.ndarray)):
            raise ValueError("xq must be a scalar or a ndarray")
        if yq is not None and not isinstance(yq, (int, float, np.ndarray)):
            raise ValueError("yq must be a scalar or a ndarray")
            
        # Interpolate
        if isinstance(zq, np.ndarray):
            nq = len(zq)
        else:
            nq = 1
        if yq is None:
            if nq > 1:
                tq = np.zeros(nq)
                for i in range(nq):
                    tq[i] = interp2(self.zaxis, self.xaxis, self.grid, zq[i], xq[i])
            else:
                tq =  interp2(self.zaxis, self.xaxis, self.grid, zq, xq)
        else:
            if nq > 1:
                tq = np.zeros(nq)
                for i in range(nq):
                    tq[i] = interp3(self.zaxis, self.xaxis, self.yaxis, self.grid,
                                    zq[i], xq[i], yq[i])
            else:
                tq =  interp3(self.zaxis, self.xaxis, self.yaxis, self.grid, zq, xq, yq)
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