# -*- coding: utf-8 -*-

"""
Eikonal interfaces eikonal solver routines written in Fortran with Python.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
from ._fteik2d import eikonal2d
from ._fteik3d import eikonal3d
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from .ttgrid import TTGrid

__all__ = [ "Eikonal" ]


class Eikonal:
    """
    Compute first arrival traveltime solving the eikonal equation using a
    finite difference scheme in 2-D and 3-D in isotropic velocity medium.
    
    Parameters
    ----------
    velocity_model : ndarray of shape (nz, nx[, ny])
        Velocity model grid in m/s.
    grid_size : tuple (dz, dx[, dy])
        Grid size in meters.
    n_sweep : int, default 1
        Number of sweeps.
    zmin : int or float, default 0.
        Z axis first coordinate.
    xmin : int or float, default 0.
        X axis first coordinate.
    ymin : int or float, default 0.
        Y axis first coordinate. Only used if 3-D velocity model.
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
            self._n_dim = velocity_model.ndim
        if np.any(np.array(velocity_model.shape) < 4):
            raise ValueError("grid shape must be at least 4")
        else:
            self._grid_shape = velocity_model.shape
        if not isinstance(grid_size, (list, tuple, np.ndarray)):
            raise ValueError("grid_size must be a list, tuple or ndarray")
        if len(grid_size) != self._n_dim:
            raise ValueError("grid_size should be of length %d, got %d" \
                             % (self._n_dim, len(grid_size)))
        if np.any(np.array(grid_size) <= 0.):
            raise ValueError("elements in grid_size must be positive")
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
        if self._n_dim == 3:
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
        new_shape : tup
            New shape.
        """
        if len(new_shape) != self._n_dim \
            or not np.all([ isinstance(n, int) for n in new_shape ]):
            raise ValueError("new_shape must be a tuple with %d integers" % self._n_dim)
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
    
    def smooth(self, sigma, mode = "reflect", cval = 0., truncate = 4.):
        """
        Smooth velocity model. This method uses SciPy's gaussian_filter
        function.
        
        Parameters
        ----------
        sigma : int, float or tuple
            Standard deviation for Gaussian kernel. The standard deviations of
            the Gaussian filter are given for each axis as a sequence, or as a
            single number, in which case it is equal for all axes.
        """
        self._velocity_model = gaussian_filter(self._velocity_model, sigma)
        return self
            
    def solve(self, sources, dtype = "float32", n_threads = 1):
        """
        Compute the traveltime grid associated to a source point.
        
        Parameters
        ----------
        sources : list or ndarray
            Source coordinates (Z, X[, Y]).
        dtype : {'float32', 'float64'}, default 'float32'
            Traveltime grid data type.
        n_threads : int, default 1
            Number of threads to pass to OpenMP.
            
        Returns
        -------
        tt : TTGrid
            Traveltime grid.
        """
        # Check inputs
        if not isinstance(sources, (list, tuple, np.ndarray)):
            raise ValueError("sources must be a list, tuple or ndarray")
        if isinstance(sources, np.ndarray) and sources.ndim not in [ 1, 2 ]:
            raise ValueError("sources must be 1-D or 2-D ndarray")
        if isinstance(sources, (list, tuple)) and len(sources) != self._n_dim:
            raise ValueError("sources should have %d coordinates, got %d" \
                             % (self._n_dim, len(sources)))
        elif isinstance(sources, np.ndarray) and sources.ndim == 1 and len(sources) != self._n_dim:
            raise ValueError("sources should have %d coordinates, got %d" \
                             % (self._n_dim, len(sources)))
        elif isinstance(sources, np.ndarray) and sources.ndim == 2 and sources.shape[1] != self._n_dim:
            raise ValueError("sources should have %d coordinates, got %d" \
                             % (self._n_dim, sources.shape[1]))
        if dtype not in [ "float32", "float64" ]:
            raise ValueError("dtype must be 'float32' or 'float64'")
        
        # Define src array
        if isinstance(sources, (list, tuple)) or sources.ndim == 1:
            src = np.array(sources)[None,:]
        else:
            src = np.array(sources)
        src_shift = np.array([ self._shift(s) for s in src ])
        nsrc = src.shape[0]
        
        # Call Eikonal solver
        if self._n_dim == 2:
            dz, dx = self._grid_size
            nz, nx = self._grid_shape
            for i in range(nsrc):
                self._check_2d(src_shift[i,0], src_shift[i,1], dz, dx, nz, nx)
            grid = eikonal2d.solve(self._velocity_model, src_shift[:,0], src_shift[:,1],
                                   dz, dx, self._n_sweep, n_threads)
            tt = [ TTGrid(grid = np.array(grid[:,:,i], dtype = dtype),
                          source = src[i,:],
                          grid_size = self._grid_size,
                          zmin = self._zmin,
                          xmin = self._xmin) for i in range(nsrc) ]
        elif len(self._grid_shape) == 3:
            dz, dx, dy = self._grid_size
            nz, nx, ny = self._grid_shape
            for i in range(nsrc):
                self._check_3d(src[i,0], src[i,1], src[i,2], dz, dx, dy, nz, nx, ny)
            grid = eikonal3d.solve(self._velocity_model, src_shift[:,0], src_shift[:,1], src_shift[:,2],
                                   dz, dx, dy, self._n_sweep, n_threads)
            tt = [ TTGrid(grid = np.array(grid[:,:,:,i], dtype = dtype),
                          source = src[i,:],
                          grid_size = self._grid_size,
                          zmin = self._zmin,
                          xmin = self._xmin,
                          ymin = self._ymin) for i in range(nsrc) ]
        if isinstance(sources, (list, tuple)) or sources.ndim == 1:
            return tt[0]
        else:
            return tt
    
    def _shift(self, sources):
        if len(sources) == 2:
            return np.array(sources) - np.array([ self._zmin, self._xmin ])
        elif len(sources) == 3:
            return np.array(sources) - np.array([ self._zmin, self._xmin, self._ymin ])
    
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
    def n_dim(self):
        """
        int
        Number of dimensions (2 or 3).
        """
        return self._n_dim
    
    @n_dim.setter
    def n_dim(self, value):
        self._n_dim = value
        
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