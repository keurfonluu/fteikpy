from abc import ABC

import numpy
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from ._interp import interp2d, interp3d


class BaseGrid(ABC):
    def __init__(self, grid, gridsize, origin, **kwargs):
        """Base grid class."""
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
    _ndim = 2

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

    def resample(self, new_shape, method="linear"):
        """
        Resample grid.

        Parameters
        ----------
        new_shape : array_like
            New grid shape (nz, nx).
        method : str ('linear' or 'nearest'), optional, default 'linear'
            Interpolation method.

        """
        zaxis = self.zaxis
        xaxis = self.xaxis
        Z, X = numpy.meshgrid(
            numpy.linspace(zaxis[0], zaxis[-1], new_shape[0]),
            numpy.linspace(xaxis[0], xaxis[-1], new_shape[1]),
            indexing="ij",
        )

        fn = RegularGridInterpolator(
            points=(zaxis, xaxis), values=self._grid, method=method, bounds_error=False,
        )
        self._grid = fn([[z, x] for z, x in zip(Z.ravel(), X.ravel())]).reshape(
            new_shape
        )

        self._gridsize = tuple(
            a * b / c for a, b, c in zip(self.gridsize, self.shape, new_shape)
        )

    def smooth(self, sigma):
        """
        Smooth grid.

        Parameters
        ----------
        sigma : scalar or array_like
            Standard deviation in meters for Gaussian kernel.

        """
        sigma = numpy.full(2, sigma) if numpy.ndim(sigma) == 0 else numpy.asarray(sigma)
        self._grid = gaussian_filter(self._grid, sigma / self._gridsize)

    @property
    def zaxis(self):
        """Return grid Z axis."""
        return self._origin[0] + self._gridsize[0] * numpy.arange(self.shape[0])

    @property
    def xaxis(self):
        """Return grid X axis."""
        return self._origin[1] + self._gridsize[1] * numpy.arange(self.shape[1])


class BaseGrid3D(BaseGrid):
    _ndim = 3

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

    def resample(self, new_shape, method="linear"):
        """
        Resample grid.

        Parameters
        ----------
        new_shape : array_like
            New grid shape (nz, nx, ny).
        method : str ('linear' or 'nearest'), optional, default 'linear'
            Interpolation method.

        """
        zaxis = self.zaxis
        xaxis = self.xaxis
        yaxis = self.yaxis
        Z, X, Y = numpy.meshgrid(
            numpy.linspace(zaxis[0], zaxis[-1], new_shape[0]),
            numpy.linspace(xaxis[0], xaxis[-1], new_shape[1]),
            numpy.linspace(yaxis[0], yaxis[-1], new_shape[2]),
            indexing="ij",
        )

        fn = RegularGridInterpolator(
            points=(zaxis, xaxis, yaxis),
            values=self._grid,
            method=method,
            bounds_error=False,
        )
        self._grid = fn(
            [[z, x, y] for z, x, y in zip(Z.ravel(), X.ravel(), Y.ravel())]
        ).reshape(new_shape)

        self._gridsize = tuple(
            a * b / c for a, b, c in zip(self.gridsize, self.shape, new_shape)
        )

    def smooth(self, sigma):
        """
        Smooth grid.

        Parameters
        ----------
        sigma : scalar or array_like
            Standard deviation in meters for Gaussian kernel.

        """
        sigma = numpy.full(3, sigma) if numpy.ndim(sigma) == 0 else numpy.asarray(sigma)
        self._grid = gaussian_filter(self._grid, sigma / self._gridsize)

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
