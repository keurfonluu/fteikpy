# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from ._fteik2d import fteik2d
from ._fteik3d import fteik3d
from ._interpolate import interp2, interp3

__all__ = [ "TTGrid", "Eikonal" ]


class TTGrid:
    
    def __init__(self, grid = None, zaxis = None, xaxis = None, yaxis = None):
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
        self._zaxis = zaxis
        self._xaxis = xaxis
        self._yaxis = yaxis
        return
        
    def get(self, zq, xq, yq = None):
        # Check inputs
        if zq.size != xq.size:
            raise ValueError("zq should have the same size as xq")
        if yq is not None and yq.size != xq.size:
            raise ValueError("yq should have the same size as xq")
            
        # Interpolate
        nq = zq.size
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
        return self._grid
    
    @grid.setter
    def grid(self, value):
        self._grid = value
        
    @property
    def shape(self):
        return self._shape
    
    @shape.setter
    def shape(self, value):
        self._shape = value
        
    @property
    def n_dim(self):
        return self._n_dim
    
    @n_dim.setter
    def n_dim(self, value):
        self._n_dim = value
    
    @property
    def zaxis(self):
        return self._zaxis
    
    @zaxis.setter
    def zaxis(self, value):
        self._zaxis = value
    
    @property
    def xaxis(self):
        return self._xaxis
    
    @xaxis.setter
    def xaxis(self, value):
        self._xaxis = value
    
    @property
    def yaxis(self):
        return self._yaxis
    
    @yaxis.setter
    def yaxis(self, value):
        self._yaxis = value


class Eikonal:
    
    def __init__(self, velocity_model, grid_size, n_sweep = 1):
        if not isinstance(velocity_model, np.ndarray) \
            and not velocity_model.ndim in [ 2, 3 ]:
            raise ValueError("velocity_model must be a 2-D or 3-D ndarray")
        if np.any(velocity_model <= 0.):
            raise ValueError("velocity_model must be positive")
        else:
            self._velocity_model = np.array(velocity_model)
            self._grid_shape = velocity_model.shape
        if len(grid_size) != len(self._grid_shape):
            raise ValueError("grid_size should be of length %d, got %d" \
                             % (len(self._grid_shape), len(grid_size)))
        if np.any(np.array(grid_size) <= 0.):
            raise ValueError("elements in grid_size must be positive")
        else:
            self._grid_size = grid_size
        if not isinstance(n_sweep, int) or n_sweep <= 0:
            raise ValueError("n_sweep must be a positive integer, got %s" % n_sweep)
        else:
            self._n_sweep = n_sweep
            
    def solve(self, source):
        # Check inputs
        if len(source) != len(self._grid_shape):
            raise ValueError("source should have %d elements, got %d" \
                             % (len(self._grid_shape), len(source)))
        
        # Call Eikonal solver
        tt = TTGrid()
        if len(source) == 2:
            xsrc, zsrc = source
            dz, dx = self._grid_size
            nz, nx = self._grid_shape
            tt.grid = fteik2d(self._velocity_model, zsrc, xsrc, dz, dx,
                              self._n_sweep, 5.)
            tt.zaxis = dz * np.arange(nz)
            tt.xaxis = dx * np.arange(nx)
        elif len(source) == 3:
            xsrc, ysrc, zsrc = source
            dz, dx, dy = self._grid_size
            nz, nx, ny = self._grid_shape
            tt.grid = fteik3d(self._velocity_model, zsrc, xsrc, ysrc, dz, dx, dy,
                              self._n_sweep, 5.)
            tt.zaxis = dz * np.arange(nz)
            tt.xaxis = dx * np.arange(nx)
            tt.yaxis = dy * np.arange(ny)
        tt.shape = tt.grid.shape
        tt.n_dim = tt.grid.ndim
        return tt