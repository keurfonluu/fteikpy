from abc import ABC

import numpy

from ._interp import interp2d, interp3d


class BaseGrid(ABC):
    def __init__(self, grid, gridsize, origin, **kwargs):
        """Base grid class"""
        super().__init__(**kwargs)
        self._grid = numpy.asarray(grid, dtype=numpy.float64)
        self._gridsize = tuple(float(x) for x in gridsize)
        self._origin = numpy.asarray(origin, dtype=numpy.float64)

    def __getitem__(self, islice):
        """Slice grid."""
        return self._grid[islice]

    @property
    def grid(self):
        """Return grid."""
        return self._grid

    @property
    def gridsize(self):
        """Return grid size."""
        return self._gridsize

    @property
    def origin(self):
        """Return grid origin coordinates."""
        return self._origin

    @property
    def shape(self):
        """Return grid shape."""
        return self._grid.shape

    @property
    def size(self):
        """Return grid size."""
        return self._grid.size

    @property
    def ndim(self):
        """Return grid number of dimensions."""
        return self._grid.ndim


class BaseGrid2D(BaseGrid):
    def __call__(self, points, fill_value=numpy.nan):
        """
        Bilinear interpolation.

        Parameters
        ----------
        points : array_like
            Query point coordinates or list of point coordinates.
        fill_value : scalar, optional, default nan
            Returned value for out-of-bound query points.

        Returns
        -------
        scalar or :class:`numpy.ndarray`
            Interpolated value(s).
        
        """
        return interp2d(
            self.zaxis,
            self.xaxis,
            self._grid,
            numpy.asarray(points, dtype=numpy.float64),
            fill_value,
        )

    @property
    def zaxis(self):
        """Return grid Z axis."""
        return self._origin[0] + self._gridsize[0] * numpy.arange(self.shape[0])

    @property
    def xaxis(self):
        """Return grid X axis."""
        return self._origin[1] + self._gridsize[1] * numpy.arange(self.shape[1])


class BaseGrid3D(BaseGrid):
    def __call__(self, points, fill_value=numpy.nan):
        """
        Trilinear interpolaton.
        
        Parameters
        ----------
        points : array_like
            Query point coordinates or list of point coordinates.
        fill_value : scalar, optional, default nan
            Returned value for out-of-bound query points.

        Returns
        -------
        scalar or :class:`numpy.ndarray`
            Interpolated value(s).
        
        """
        return interp3d(
            self.zaxis,
            self.xaxis,
            self.yaxis,
            self._grid,
            numpy.asarray(points, dtype=numpy.float64),
            fill_value,
        )

    @property
    def zaxis(self):
        """Return grid Z axis."""
        return self._origin[0] + self._gridsize[0] * numpy.arange(self.shape[0])

    @property
    def xaxis(self):
        """Return grid X axis."""
        return self._origin[1] + self._gridsize[1] * numpy.arange(self.shape[1])

    @property
    def yaxis(self):
        """Return grid Y axis."""
        return self._origin[2] + self._gridsize[2] * numpy.arange(self.shape[2])


class BaseTraveltime(ABC):
    def __init__(self, source, gradient, vzero, **kwargs):
        """Traveltime base class."""
        super().__init__(**kwargs)
        self._source = source
        self._gradient = gradient
        self._vzero = vzero

    @property
    def source(self):
        """Return source coordinates."""
        return self._source
