# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np

__all__ = [ "lay2vel", "lay2tt" ]


def lay2vel(lay, dz, grid_shape, smooth = False):
    """
    Convert a layered model to a continuous velocity model.
    
    Parameters
    ----------
    lay: ndarray
        Layer velocities (first column) and interface depth (second column).
    dz: float
        Grid size in Z coordinate in meters.
    grid_shape: tuple (nz, nx[, ny])
        Gris shape.
    smooth: bool, default False
        If True, smooth the velocity model.
    
    Returns
    -------
    vel: ndarray
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
    if not isinstance(smooth, bool):
        raise ValueError("smooth must either be True or False")
    
    # Create velocity model
    if len(lay.shape) == 1:
        vel = np.full(grid_shape, lay[0])
    else:
        nlayer = lay.shape[0]
        vel1d = np.zeros(grid_shape[0])
        ztop = 0
        zbot = int(np.floor(lay[0,1]/dz)) - 1
        if smooth:
            vel1d[ztop:zbot+1] = np.linspace(lay[0,0], lay[1,0], zbot-ztop+1)
            for i in range(1, nlayer-1):
                ztop = zbot + 1
                zbot = int(np.floor(lay[i,1]/dz)) - 1
                vel1d[ztop:zbot+1] = np.linspace(lay[i,0], lay[i+1,0], zbot-ztop+1)
        else:
            vel1d[ztop:zbot+1].fill(lay[0,0])
            for i in range(1, nlayer-1):
                ztop = zbot + 1
                zbot = int(np.floor(lay[i,1]/dz)) - 1
                vel1d[ztop:zbot+1].fill(lay[i,0])
        vel1d[zbot+1:].fill(lay[-1,0])
        
        vel = vel1d
        if len(grid_shape) > 1:
            vel = np.tile(vel[:,None], grid_shape[1])
        if len(grid_shape) > 2:
            vel = np.tile(vel[:,:,None], grid_shape[2])
    return vel


def lay2tt(eikonal, sources, receivers):
    """
    Given a layered velocity model, compute the first arrivel traveltime for
    each source and each receiver. Only useful if working in 3-D as a 2-D
    eikonal solver is used for traveltime computation.
    
    Parameters
    ----------
    eikonal: Eikonal object
        2-D eikonal solver with a layered velocity model.
    sources: ndarray
        Sources positions.
    receivers: ndarray
        Receivers positions.
        
    Returns
    -------
    tcalc: ndarray of shape (nrcv, nsrc)
        Traveltimes for each source and each receiver.
    """
    # Check inputs
    if not hasattr(eikonal, "_velocity_model"):
        raise ValueError("eikonal must have a defined velocity model")
    if not isinstance(sources, np.ndarray) or sources.shape[1] != 3:
        raise ValueError("sources must be ndarray with 3 columns")
    if not isinstance(receivers, np.ndarray) or receivers.shape[1] != 3:
        raise ValueError("receivers must be ndarray with 3 columns")
    
    # Parameters
    nrcv = receivers.shape[0]
    nsrc = sources.shape[0]
    
    # Switch sources and receivers to minimize calls to eikonals
    n1 = min(nrcv, nsrc)
    n2 = max(nrcv, nsrc)
    tcalc = np.zeros((n2, n1))
    if n1 == nsrc:
        rcv, src = np.array(receivers), np.array(sources)
    else:
        rcv, src = np.array(sources), np.array(receivers)
    
    # Compute traveltimes using eikonal solver
    dhorz = np.zeros(n2)        
    for i in range(n1):
        for j in range(n2):
            dhorz[j] = np.linalg.norm(src[i,1:] - rcv[j,1:])
        tt = eikonal.solve((src[i,0], 0.))
        tcalc[:,i] = [ tt.get(zrcv, xrcv) for zrcv, xrcv in zip(rcv[:,0], dhorz) ]
            
    # Transpose to reshape to [ nrcv, nsrc ]
    if n1 == nrcv:
        return tcalc.transpose()
    else:
        return tcalc