from abc import ABC, abstractmethod

import numpy


class BaseEikonalSolver(ABC):
    def __init__(self, velocity_model, gridsize, origin):
        self._velocity_model = numpy.array(velocity_model, dtype=numpy.float64)
        self._gridsize = tuple(float(x) for x in gridsize)
        self._origin = numpy.array(origin, dtype=numpy.float64)

    @abstractmethod
    def solve(self, source, max_sweep=2):
        pass

    @property
    def velocity_model(self):
        return self._velocity_model

    @property
    def gridsize(self):
        return self._gridsize

    @property
    def origin(self):
        return self._origin


class BaseTraveltimeGrid(ABC):
    def __init__(self, grid, gridsize, origin, source, vzero):
        self._grid = grid
        self._gridsize = gridsize
        self._origin = origin
        self._source = source
        self._vzero = vzero

    @abstractmethod
    def __call__(self, points):
        pass

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
    def source(self):
        return self._source

    @property
    def shape(self):
        return self._grid.shape

    @property
    def size(self):
        return self._grid.size

    @property
    def ndim(self):
        return self._grid.ndim

    @property
    def zaxis(self):
        return self._origin[0] + self._gridsize[0] * numpy.arange(self.shape[0])

    @property
    def xaxis(self):
        return self._origin[1] + self._gridsize[1] * numpy.arange(self.shape[1])
