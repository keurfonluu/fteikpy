# -*- coding: utf-8 -*-

"""
This example show that velocity interpolation estimate more accurately
traveltimes. In this example, the eikonal equation is solved on a 6-by-6 grid.
Traveltimes obtained with velocity interpolation are compared to time
interpolation and the analytical solution.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
try:
    from fteikpy import Eikonal
except ImportError:
    import sys
    sys.path.append("../")
    from fteikpy import Eikonal


def traveltime(vel, src, rcv):
    return np.linalg.norm(np.array(src) - np.array(rcv)) / vel


if __name__ == "__main__":
    # Parameters
    nz, nx = 100, 100
    dz, dx = 1., 1.
    source = ( 50., 50. )
    velocity = 1500.
    
    # Analytical solution
    az = dz * np.arange(nz)
    ax = dx * np.arange(nx)
    Z, X = np.meshgrid(az, ax, indexing = "ij")
    tt_true = np.array([ traveltime(velocity, source, (z, x))
                            for z, x in zip(Z.ravel(), X.ravel()) ]).reshape((nz, nx))
    
    # Eikonal solver
    eik = Eikonal(np.full((6, 6), velocity), (20., 20.))
    ttgrid = eik.solve(source)
    
    # Time interpolation
    fn = RegularGridInterpolator((20. * np.arange(6), 20. * np.arange(6)), ttgrid.grid)
    tt_time = np.array([ fn([ z, x ]) for z, x in zip(Z.ravel(), X.ravel()) ]).reshape((nz, nx))
    
    # Velocity interpolation (using ttgrid's get method)
    tt_vel = np.array([ ttgrid.get(z, x) for z, x in zip(Z.ravel(), X.ravel()) ]).reshape((nz, nx))
    
    # Plot traveltime grids
    fig = plt.figure(figsize = (12, 4), facecolor = "white")
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    
    ax1.imshow(tt_true); ax1.set_title("Analytical solution")
    ax2.imshow(tt_time); ax2.set_title("Time interpolation")
    ax3.imshow(tt_vel); ax3.set_title("Velocity interpolation")