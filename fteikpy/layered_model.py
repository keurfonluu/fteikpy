# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np

__all__ = [ "lay2vel", "lay2tt" ]


def lay2vel(lay, dz, nz, nx, ny = None, grad = False):
    if ny is None:
        vel = np.zeros((nz, nx))
    else:
        vel = np.zeros((nz, nx, ny))
    if len(lay.shape) == 1:
        vel.fill(lay[0])
    else:
        nlayer = lay.shape[0]
        vel1d = np.zeros(nz)
        ztop = 0
        zbot = int(np.floor(lay[0,1]/dz)) - 1
        if grad:
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
        
        if ny is None:
            for i in range(nx):
                vel[:,i] = vel1d
        else:
            for i in range(nx):
                for j in range(ny):
                    vel[:,i,j] = vel1d
    return vel

def lay2tt(eikonal, src, rcv):
    # Check inputs
    assert src.shape[1] == 3
    assert rcv.shape[1] == 3
    
    # Parameters
    nrcv = rcv.shape[0]
    nsrc = src.shape[0]
    
    # Switch sources and receivers to minimize calls to eikonals
    n1 = min(nrcv, nsrc)
    n2 = max(nrcv, nsrc)
    tcalc = np.zeros((n2, n1))
    if n1 == nsrc:
        receivers, sources = rcv.copy(), src.copy()
    else:
        receivers, sources = src.copy(), rcv.copy()
    
    # Compute traveltimes using Eikonal solver
    dhorz = np.zeros(n2)        
    for i in range(n1):
        for j in range(n2):
            dhorz[j] = np.linalg.norm(sources[i,:2] - receivers[j,:2])
        tt = eikonal.solve((0., sources[i,2]))
        tcalc[:,i] = tt.get(receivers[:,2], dhorz)
            
    # Transpose to reshape to [ nrcv, nsrc ]
    if n1 == nrcv:
        return np.transpose(tcalc)
    else:
        return tcalc