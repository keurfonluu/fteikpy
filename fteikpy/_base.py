from abc import ABC

import numpy

from ._interp import interp2d, interp3d


class BaseGrid(ABC):
    def __init__(self, grid, gridsize, origin, **kwargs):
        super().__init__(**kwargs)
        self._grid = numpy.asarray(grid, dtype=numpy.float64)
        self._gridsize = tuple(float(x) for x in gridsize)
        self._origin = numpy.asarray(origin, dtype=numpy.float64)

    def __getitem__(self, islice):
        return self._grid[islice]

    @property
    def grid(self):
        return self._grid

    @property
    def gridsize(self):
        return self._gridsize

    @property
    def origin(self):
        return self._origin

    @property
    def shape(self):
        return self._grid.shape

    @property
    def size(self):
        return self._grid.size

    @property
    def ndim(self):
        return self._grid.ndim


class BaseGrid2D(BaseGrid):
    def __call__(self, points):
        return interp2d(
            self.zaxis,
            self.xaxis,
            self._grid,
            numpy.asarray(points, dtype=numpy.float64),
        )

    @property
    def zaxis(self):
        return self._origin[0] + self._gridsize[0] * numpy.arange(self.shape[0])

    @property
    def xaxis(self):
        return self._origin[1] + self._gridsize[1] * numpy.arange(self.shape[1])


class BaseGrid3D(BaseGrid2D):
    def __call__(self, points):
        return interp3d(
            self.zaxis,
            self.xaxis,
            self.yaxis,
            self._grid,
            numpy.asarray(points, dtype=numpy.float64),
        )

    @property
    def yaxis(self):
        return self._origin[2] + self._gridsize[2] * numpy.arange(self.shape[2])


class BaseTraveltime(ABC):
    def __init__(self, source, grad, vzero, **kwargs):
        super().__init__(**kwargs)
        self._source = source
        self._grad = grad
        self._vzero = vzero

    @property
    def source(self):
        return self._source

    @property
    def grad(self):
        return self._grad
