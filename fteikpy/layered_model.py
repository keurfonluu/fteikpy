# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
from ._fteik2d import fteik2d
from ._lay2vel import lay2vel as l2vf

__all__ = [ "lay2vel", "lay2tt" ]


def lay2vel(lay, dz, grid_shape):
    """
    Convert a layered model to a continuous velocity model.
    
    Parameters
    ----------
    lay : ndarray
        Layer velocities (first column) and interface depth (second column).
    dz : float
        Grid size in Z coordinate in meters.
    grid_shape : tuple (nz, nx[, ny])
        Gris shape.
    
    Returns
    -------
    vel : ndarray
        Velocity model grid in m/s.
    """
    # Check inputs
    if not isinstance(lay, np.ndarray) and not lay.ndim in [ 1, 2 ]:
        raise ValueError("grid must be a 1-D or 2-D ndarray")
    if (lay.ndim == 1 and lay[0] < 0.) or (lay.ndim == 2 and lay[:,0].min() < 0.):
        raise ValueError("velocities must be positive")
    if not isinstance(dz, (float, int)) or dz < 0.:
        raise ValueError("dz must be positive")
    zmax = dz*grid_shape[0]
    if (lay.ndim == 1 and lay[1] > zmax) or (lay.ndim == 2 and lay[:,1].max() > zmax):
        raise ValueError("last layer depth must be %.2f" % zmax)
    if not np.all([isinstance(n, int) for n in grid_shape]) or len(grid_shape) not in [ 1, 2, 3 ]:
        raise ValueError("grid_shape must be a tuple of integers of size 1, 2 or 3")
    
    # Create a continuous velocity model
    if len(grid_shape) == 1:
        return l2vf.lay2vel1(lay, dz, *grid_shape)
    elif len(grid_shape) == 2:
        return l2vf.lay2vel2(lay, dz, *grid_shape)
    elif len(grid_shape) == 3:
        return l2vf.lay2vel3(lay, dz, *grid_shape)


def lay2tt(velocity_model, grid_size, sources, receivers, n_sweep = 1, n_threads = 1):
    """
    Given a layered velocity model, compute the first arrivel traveltime for
    each source and each receiver. Only useful if working in 3-D as a 2-D
    eikonal solver is used for traveltime computation.
    
    Parameters
    ----------
    velocity_model : ndarray of shape (nz, nx)
        Velocity model grid in m/s.
    grid_size : tuple (dz, dx)
        Grid size in meters.
    sources : ndarray
        Sources coordinates (Z, X[, Y]).
    receivers : ndarray
        Receivers coordinates (Z, X[, Y]).
    n_sweep : int, default 1
        Number of sweeps.
    n_threads : int, default 1
        Number of threads to pass to OpenMP.
        
    Returns
    -------
    tcalc : ndarray of shape (nrcv, nsrc)
        Traveltimes for each source and each receiver.
    """
    # Check inputs
    if not isinstance(velocity_model, np.ndarray) or velocity_model.ndim != 2:
        raise ValueError("velocity_model must be a 2-D ndarray")
    if np.any(velocity_model <= 0.):
        raise ValueError("velocity_model must be positive")
    if not isinstance(grid_size, (list, tuple, np.ndarray)):
        raise ValueError("grid_size must be a list, tuple or ndarray")
    if len(grid_size) != 2:
        raise ValueError("grid_size should be of length 2, got %d" % len(grid_size))
    if np.any(np.array(grid_size) <= 0.):
        raise ValueError("elements in grid_size must be positive")
    if not isinstance(sources, np.ndarray) or sources.shape[1] != 3:
        raise ValueError("sources must be ndarray with 3 columns")
    if not isinstance(receivers, np.ndarray) or receivers.shape[1] != 3:
        raise ValueError("receivers must be ndarray with 3 columns")
    if not isinstance(n_sweep, int) or n_sweep <= 0:
        raise ValueError("n_sweep must be a positive integer, got %s" % n_sweep)
    if not isinstance(n_threads, int) or n_threads < 1:
        raise ValueError("n_threads must be atleast 1, got %s" % n_threads)
    
    dz, dx = grid_size
    tcalc = fteik2d.lay2tt(1./velocity_model, dz, dx, sources[:,0], sources[:,1], sources[:,2],
                           receivers[:,0], receivers[:,1], receivers[:,2], n_sweep, n_threads = n_threads)
    return tcalc