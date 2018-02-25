# -*- coding: utf-8 -*-

"""
This example benchmarks the performances of a ray tracer with the 2D and 3D
Eikonal solvers on a stratified medium.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import time
from raytracer.raytracer import Ray3D
try:
    from fteikpy import Eikonal, lay2vel, lay2tt
except ImportError:
    import sys
    sys.path.append("../")
    from fteikpy import Eikonal, lay2vel, lay2tt


if __name__ == "__main__":
    # Parameters
    sources = np.loadtxt("shots.txt")
    receivers = np.loadtxt("stations.txt")
    dz, dx, dy = 2.5, 2.5, 2.5
    nz, nx, ny = 400, 280, 4
    n_threads = 8
    
    # Make a layered velocity model
    lay = 1500. + 250. * np.arange(10)
    zint = 100. + 100. * np.arange(10)
    vel2d = lay2vel(np.hstack((lay[:,None], zint[:,None])), dz, (nz, nx))
    vel3d = np.tile(vel2d[:,:,None], ny)
    
    # Ray tracer
    start_time = time.time()
    ray = Ray3D()
    tcalc_ray = ray.lay2tt(sources[:,[1,2,0]], receivers[:,[1,2,0]], lay, zint)
    print("Ray tracer: %.3f seconds" % (time.time() - start_time))
    
    # Eikonal 2D
    start_time = time.time()
    tcalc_eik2d = lay2tt(vel2d, (dz, dx), sources, receivers, n_sweep = 2, n_threads = n_threads)
    print("\nEikonal 2D: %.3f seconds" % (time.time() - start_time))
    print("Mean residual (2D): ", (tcalc_eik2d - tcalc_ray).mean())
    
    # Eikonal 3D
    start_time = time.time()
    eik3d = Eikonal(vel3d, (dz, dx, dy), n_sweep = 2)
    tt = eik3d.solve(sources, n_threads = n_threads)
    tcalc_eik3d = np.array([ [ grid.get(z, x, y, check = False) for z, x, y in receivers ]
                                for grid in tt ]).transpose()
    print("\nEikonal 3D: %.3f seconds" % (time.time() - start_time))
    print("Mean residual (3D): ", (tcalc_eik3d - tcalc_ray).mean())