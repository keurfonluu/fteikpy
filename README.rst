fteikpy
=======

|License| |Stars| |Pyversions| |Version| |Downloads| |Code style: black| |Codacy Badge| |Codecov| |Build| |Travis| |DOI|

**fteikpy** is a Python library that computes accurate first arrival traveltimes in 2D and 3D heterogeneous isotropic velocity models. The algorithm handles properly the curvature of wavefronts close to the source which can be placed without any problem between grid points.

The code is based on `FTeik <https://github.com/Mark-Noble/FTeik-Eikonal-Solver>`__ implemented in Python and compiled `just-in-time <https://en.wikipedia.org/wiki/Just-in-time_compilation>`__ with `numba <https://numba.pydata.org/>`__.

.. figure:: https://raw.githubusercontent.com/keurfonluu/fteikpy/master/.github/sample.gif
   :alt: sample-marmousi
   :width: 100%
   :align: center

   Computation of traveltimes and ray-tracing on smoothed Marmousi velocity model.

Features
--------

Forward modeling:

-  Compute traveltimes in 2D and 3D Cartesian grids with the possibility to use a different grid spacing in Z, X and Y directions,
-  Compute traveltime gradients at runtime or a posteriori,
-  A posteriori 2D and 3D ray-tracing.

Parallel:

-  Traveltime grids are seemlessly computed in parallel for different sources,
-  Raypaths from a given source to different locations are also evaluated in parallel.

Installation
------------

The recommended way to install **fteikpy** and all its dependencies is through the Python Package Index:

.. code:: bash

   pip install fteikpy --user

Otherwise, clone and extract the package, then run from the package location:

.. code:: bash

   pip install . --user

To test the integrity of the installed package, check out this repository and run:

.. code:: bash

   pytest

Documentation
-------------

Refer to the online `documentation <https://keurfonluu.github.io/fteikpy/>`__ for detailed description of the API and examples.

Alternatively, the documentation can be built using `Sphinx <https://www.sphinx-doc.org/en/master/>`__:

.. code:: bash

   pip install -r doc/requirements.txt
   sphinx-build -b html doc/source doc/build

Usage
-----

The following example computes the traveltime grid in a 3D homogeneous velocity model:

.. code-block:: python

   import numpy as np
   from fteikpy import Eikonal3D

   # Velocity model
   velocity_model = np.ones((8, 8, 8))
   dz, dx, dy = 1.0, 1.0, 1.0

   # Solve Eikonal at source
   eik = Eikonal3D(velocity_model, gridsize=(dz, dx, dy))
   tt = eik.solve((0.0, 0.0, 0.0))

   # Get traveltime at specific grid point
   t1 = tt[0, 1, 2]

   # Or get traveltime at any point in the grid
   t2 = tt(np.random.rand(3) * 7.0)

Contributing
------------

Please refer to the `Contributing
Guidelines <https://github.com/keurfonluu/fteikpy/blob/master/CONTRIBUTING.rst>`__ to see how you can help. This project is released with a `Code of Conduct <https://github.com/keurfonluu/fteikpy/blob/master/CODE_OF_CONDUCT.rst>`__ which you agree to abide by when contributing.

.. |License| image:: https://img.shields.io/github/license/keurfonluu/fteikpy
   :target: https://github.com/keurfonluu/fteikpy/blob/master/LICENSE

.. |Stars| image:: https://img.shields.io/github/stars/keurfonluu/fteikpy?logo=github
   :target: https://github.com/keurfonluu/fteikpy

.. |Pyversions| image:: https://img.shields.io/pypi/pyversions/fteikpy.svg?style=flat
   :target: https://pypi.org/pypi/fteikpy/

.. |Version| image:: https://img.shields.io/pypi/v/fteikpy.svg?style=flat
   :target: https://pypi.org/project/fteikpy

.. |Downloads| image:: https://pepy.tech/badge/fteikpy
   :target: https://pepy.tech/project/fteikpy

.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat
   :target: https://github.com/psf/black

.. |Codacy Badge| image:: https://img.shields.io/codacy/grade/bec3d2ad6b8c45cf9bb0da110fe04838.svg?style=flat
   :target: https://www.codacy.com/gh/keurfonluu/fteikpy/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=keurfonluu/fteikpy&amp;utm_campaign=Badge_Grade

.. |Codecov| image:: https://img.shields.io/codecov/c/github/keurfonluu/fteikpy.svg?style=flat
   :target: https://codecov.io/gh/keurfonluu/fteikpy

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4269352.svg?style=flat
   :target: https://doi.org/10.5281/zenodo.4269352

.. |Build| image:: https://img.shields.io/github/workflow/status/keurfonluu/fteikpy/Python%20package
   :target: https://github.com/keurfonluu/fteikpy

.. |Travis| image:: https://img.shields.io/travis/com/keurfonluu/fteikpy/master?label=docs
   :target: https://keurfonluu.github.io/fteikpy/
