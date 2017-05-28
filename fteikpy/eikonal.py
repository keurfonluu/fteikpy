# -*- coding: utf-8 -*-

"""
Eikonal interfaces eikonal solver routines written in Fortran with Python.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
from ._fteik2d import fteik2d
from ._fteik3d import fteik3d
from .ttgrid import TTGrid

__all__ = [ "Eikonal" ]


class Eikonal:
    """
    Compute first arrival traveltime solving the eikonal equation using a
    finite difference scheme in 2-D and 3-D in isotropic velocity medium.
    
    Parameters
    ----------
    velocity_model: ndarray of shape (nz, nx[, ny])
        Velocity model grid in m/s.
    Grid size: tuple (dz, dx[, dy])
        Grid size in meters.
    n_sweep: int, default 1
        Number of sweeps.
    """
    
    def __init__(self, velocity_model, grid_size, n_sweep = 1):
        if not isinstance(velocity_model, np.ndarray) \
            and not velocity_model.ndim in [ 2, 3 ]:
            raise ValueError("velocity_model must be a 2-D or 3-D ndarray")
        if np.any(velocity_model <= 0.):
            raise ValueError("velocity_model must be positive")
        else:
            self._velocity_model = np.array(velocity_model)
        if np.any(np.array(velocity_model.shape) < 4):
            raise ValueError("grid size should be at least 4")
        else:
            self._grid_shape = velocity_model.shape
        if len(grid_size) != len(self._grid_shape):
            raise ValueError("grid_size should be of length %d, got %d" \
                             % (len(self._grid_shape), len(grid_size)))
        if np.any(np.array(grid_size) <= 0.):
            raise ValueError("grid_size must be positive")
        else:
            self._grid_size = grid_size
        if not isinstance(n_sweep, int) or n_sweep <= 0:
            raise ValueError("n_sweep must be a positive integer, got %s" % n_sweep)
        else:
            self._n_sweep = n_sweep
            
    def solve(self, source):
        """
        Compute the traveltime grid associated to a source point.
        
        Parameters
        ----------
        source: ndarray
            Source coordinates (Z, X[, Y]).
            
        Returns
        -------
        tt: TTGrid
            Traveltime grid.
        """
        # Check inputs
        if len(source) != len(self._grid_shape):
            raise ValueError("source should have %d coordinates, got %d" \
                             % (len(self._grid_shape), len(source)))
        
        # Call Eikonal solver
        tt = TTGrid(source = source)
        if len(source) == 2:
            zsrc, xsrc = source
            dz, dx = self._grid_size
            nz, nx = self._grid_shape
            self._check_2d(zsrc, xsrc, dz, dx, nz, nx)
            tt.grid = fteik2d(self._velocity_model, zsrc, xsrc, dz, dx,
                              self._n_sweep, 5.)
            tt.zaxis = dz * np.arange(nz)
            tt.xaxis = dx * np.arange(nx)
        elif len(source) == 3:
            zsrc, xsrc, ysrc = source
            dz, dx, dy = self._grid_size
            nz, nx, ny = self._grid_shape
            self._check_3d(zsrc, xsrc, ysrc, dz, dx, dy, nz, nx, ny)
            tt.grid = fteik3d(self._velocity_model, zsrc, xsrc, ysrc, dz, dx, dy,
                              self._n_sweep, 5.)
            tt.zaxis = dz * np.arange(nz)
            tt.xaxis = dx * np.arange(nx)
            tt.yaxis = dy * np.arange(ny)
        tt.shape = tt.grid.shape
        tt.n_dim = tt.grid.ndim
        return tt
    
    def _check_2d(self, zsrc, xsrc, dz, dx, nz, nx):
        zmax, xmax = (nz-1)*dz, (nx-1)*dx
        if np.logical_or(np.any(zsrc < 0.), np.any(zsrc > zmax)):
            raise ValueError("zsrc should be in [ 0, %.2f ]" % zmax)
        if np.logical_or(np.any(xsrc < 0.), np.any(xsrc > xmax)):
            raise ValueError("xsrc should be in [ 0, %.2f ]" % xmax)
            
    def _check_3d(self, zsrc, xsrc, ysrc, dz, dx, dy, nz, nx, ny):
        self._check_2d(zsrc, xsrc, dz, dx, nz, nx)
        ymax = (ny-1)*dy
        if np.logical_or(np.any(ysrc < 0.), np.any(ysrc > ymax)):
            raise ValueError("ysrc should be in [ 0, %.2f ]" % ymax)
            
    @property
    def velocity_model(self):
        """
        ndarray of shape (nz, nx[, ny])
        Velocity model grid in m/s.
        """
        return self._velocity_model
    
    @velocity_model.setter
    def velocity_model(self, value):
        self._velocity_model = value
        
    @property
    def grid_shape(self):
        """
        tuple (nz, nx[, ny])
        Velocity grid's shape.
        """
        return self._grid_shape
    
    @grid_shape.setter
    def grid_shape(self, value):
        self._grid_shape = value
        
    @property
    def grid_size(self):
        """
        tuple (dz, dx[, dy])
        Grid size in meters.
        """
        return self._grid_size
    
    @grid_size.setter
    def grid_size(self, value):
        self._grid_size = value
        
    @property
    def n_sweep(self):
        """
        int
        Number of sweeps.
        """
        return self._n_sweep
    
    @n_sweep.setter
    def n_sweep(self, value):
        self._n_sweep = value