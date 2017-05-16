*******
FTeikPy
*******

FTeikPy is a Python module that computes accurate first arrival traveltimes in
2-D and 3-D heterogeneous isotropic velocity model, with the possibility to use
a different grid spacing in Z, X and Y directions. The algorithm handles
properly the curvature of wavefronts close to the source. The source can be
placed without any problem between grid points.

:Version: 1.0.0
:Author: Mark Noble
:Maintainer: Keurfon Luu
:Web site: https://github.com/keurfonluu/fteikpy
:Copyright: This document has been placed in the public domain.
:License: FTeikPy is released under the MIT License.

**NOTE**: the 2-D and 3-D Eikonal solvers included in FTeikPy are written in
Fortran. The original source codes can be found `here <https://github.com/Mark-Noble/FTEIK2D>`__.
Detailed implementation of local operators and global propagation scheme
implemented in this module are inspired from [1]. If you find this algorithm
and/or module useful, citing this paper would be appreciated.


Installation
============

Download and extract the package, then run:

.. code-block:: bash

    python setup.py install
    
    
Troubleshooting on Windows
==========================

A Fortran compiler is required to install this module. While it is
straighforward on Unix systems, it can be quite a pain on Windows. We recommend
installing `Anaconda <https://www.continuum.io/downloads>`__ that contains all
the required packages to install FTeikPy on Windows systems.

If you got the error:

    Error: ValueError: Unknown MS Compiler version 1900
    
You may need to manually patch the file `cygwinccompiler.py` located in:

    <Anaconda directory>/Lib/distutils/
    
by replacing:

    self.dll_libraries = get_msvcr()
    
in lines 157 and 318 by the word

    pass


References
==========
.. [1] M. Noble, A. Gesret and N. Belayouni, *Accurate 3-D finite difference
       computation of traveltimes in strongly heterogeneous media*, Geophysical
       Journal International, 2014, 199(3): 1572-1585