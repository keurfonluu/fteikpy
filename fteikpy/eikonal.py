# -*- coding: utf-8 -*-

"""
Eikonal interfaces eikonal solver routines written in Fortran with Python.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
from ._fteik2d import fteik2d
from ._fteik3d import fteik3d
from scipy.interpolate import RegularGridInterpolator
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
    grid_size: tuple (dz, dx[, dy])
        Grid size in meters.
    n_sweep: int, default 1
        Number of sweeps.
    zmin: int or float, default 0.
        Z axis first coordinate.
    xmin: int or float, default 0.
        X axis first coordinate.
    ymin: int or float, default 0.
        Y axis first coordinate. Only used if velocity model's shape is 3.
    """
    
    def __init__(self, velocity_model, grid_size, n_sweep = 1,
                 zmin = 0., xmin = 0., ymin = 0.):
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
        if not isinstance(zmin, (int, float)):
            raise ValueError("zmin must be an integer or float")
        else:
            self._zmin = zmin
            self._zaxis = zmin + grid_size[0] * np.arange(self._grid_shape[0])
        if not isinstance(xmin, (int, float)):
            raise ValueError("xmin must be an integer or float")
        else:
            self._xmin = xmin
            self._xaxis = xmin + grid_size[1] * np.arange(self._grid_shape[1])
        if velocity_model.ndim == 3:
            if not isinstance(ymin, (int, float)):
                raise ValueError("ymin must be an integer or float")
            else:
                self._ymin = ymin
                self._yaxis = ymin + grid_size[2] * np.arange(self._grid_shape[2])
            
    def rescale(self, new_shape):
        """
        Upscale or downscale velocity model.
        
        Parameters
        ----------
        new_shape: tup
            New shape.
        """
        if len(new_shape) != len(self._grid_shape) \
            or not np.all([ isinstance(n, int) for n in new_shape ]):
            raise ValueError("new_shape must be a tuple with %d integers" % len(self._grid_shape))
        if np.any([ n < 4 for n in new_shape ]):
            raise ValueError("elements in new_shape must be at least 4")
            
        if new_shape == self._grid_shape:
            pass
        elif len(new_shape) == 2:
            fn = RegularGridInterpolator((self._zaxis, self._xaxis), self._velocity_model)
            zaxis = np.linspace(self._zmin, self._zaxis[-1], new_shape[0])
            xaxis = np.linspace(self._xmin, self._xaxis[-1], new_shape[1])
            Z, X = np.meshgrid(zaxis, xaxis, indexing = "ij")
            cz, cx = [ new / old for new, old in zip(new_shape, self._grid_shape) ]
            self._velocity_model = fn([ [ z, x ] for z, x in zip(Z.ravel(), X.ravel()) ]).reshape(new_shape)
            self._grid_shape = new_shape
            self._grid_size = (self._grid_size[0] / cz, self._grid_size[1] / cx)
            self._zaxis = self._zmin + self._grid_size[0] * np.arange(self._grid_shape[0])
            self._xaxis = self._xmin + self._grid_size[1] * np.arange(self._grid_shape[1])
        elif len(new_shape) == 3:
            fn = RegularGridInterpolator((self._zaxis, self._xaxis, self._yaxis), self._velocity_model)
            zaxis = np.linspace(self._zmin, self._zaxis[-1], new_shape[0])
            xaxis = np.linspace(self._xmin, self._xaxis[-1], new_shape[1])
            yaxis = np.linspace(self._ymin, self._yaxis[-1], new_shape[2])
            Z, X, Y = np.meshgrid(zaxis, xaxis, yaxis, indexing = "ij")
            cz, cx, cy = [ new / (old+1) for new, old in zip(new_shape, self._grid_shape) ]
            self._velocity_model = fn([ [ z, x, y ] for z, x, y in zip(Z.ravel(), X.ravel(), Y.ravel()) ]).reshape(new_shape)
            self._grid_shape = new_shape
            self._grid_size = (self._grid_size[0] / cz, self._grid_size[1] / cx, self._grid_size[2] / cy)
            self._zaxis = self._zmin + self._grid_size[0] * np.arange(self._grid_shape[0])
            self._xaxis = self._xmin + self._grid_size[1] * np.arange(self._grid_shape[1])
            self._yaxis = self._ymin + self._grid_size[2] * np.arange(self._grid_shape[2])
        return self
            
    def solve(self, source, dtype = "float32"):
        """
        Compute the traveltime grid associated to a source point.
        
        Parameters
        ----------
        source: ndarray
            Source coordinates (Z, X[, Y]).
        dtype: {'float32', 'float64'}, default 'float32'
            Traveltime grid data type.
            
        Returns
        -------
        tt: TTGrid
            Traveltime grid.
        """
        # Check inputs
        if len(source) != len(self._grid_shape):
            raise ValueError("source should have %d coordinates, got %d" \
                             % (len(self._grid_shape), len(source)))
        if dtype not in [ "float32", "float64" ]:
            raise ValueError("dtype must be 'float32' or 'float64'")
        
        # Call Eikonal solver
        if len(source) == 2:
            zsrc, xsrc = self._shift(source)
            dz, dx = self._grid_size
            nz, nx = self._grid_shape
            self._check_2d(zsrc, xsrc, dz, dx, nz, nx)
            grid = fteik2d(self._velocity_model, zsrc, xsrc, dz, dx, self._n_sweep, 5.)
            tt = TTGrid(grid = np.array(grid, dtype = dtype),
                        source = source,
                        grid_size = self._grid_size,
                        zmin = self._zmin,
                        xmin = self._xmin)
        elif len(source) == 3:
            zsrc, xsrc, ysrc = self._shift(source)
            dz, dx, dy = self._grid_size
            nz, nx, ny = self._grid_shape
            self._check_3d(zsrc, xsrc, ysrc, dz, dx, dy, nz, nx, ny)
            grid = fteik3d(self._velocity_model, zsrc, xsrc, ysrc, dz, dx, dy, self._n_sweep, 5.)
            tt = TTGrid(grid = np.array(grid, dtype = dtype),
                        source = source,
                        grid_size = self._grid_size,
                        zmin = self._zmin,
                        xmin = self._xmin,
                        ymin = self._ymin)
        return tt
    
    def _shift(self, source):
        if len(source) == 2:
            return np.array(source) - np.array([ self._zmin, self._xmin ])
        elif len(source) == 3:
            return np.array(source) - np.array([ self._zmin, self._xmin, self._ymin ])
    
    def _check_2d(self, zsrc, xsrc, dz, dx, nz, nx):
        zmax, xmax = (nz-1)*dz, (nx-1)*dx
        if np.logical_or(np.any(zsrc < 0.), np.any(zsrc > zmax)):
            raise ValueError("zsrc out of bounds")
        if np.logical_or(np.any(xsrc < 0.), np.any(xsrc > xmax)):
            raise ValueError("xsrc out of bounds")
            
    def _check_3d(self, zsrc, xsrc, ysrc, dz, dx, dy, nz, nx, ny):
        self._check_2d(zsrc, xsrc, dz, dx, nz, nx)
        ymax = (ny-1)*dy
        if np.logical_or(np.any(ysrc < 0.), np.any(ysrc > ymax)):
            raise ValueError("ysrc out of bounds")
            
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