import numpy

from ._base import BaseGrid2D, BaseGrid3D, BaseTraveltime
from ._fteik import ray2d, ray3d
from ._interp import vinterp2d, vinterp3d


class Grid2D(BaseGrid2D):
    def __init__(self, *args, **kwargs):
        """
        2D grid class.

        Parameters
        ----------
        grid : array_like
            Grid array.
        gridsize : array_like
            Grid size (dz, dx).
        origin : array_like
            Grid origin coordinates.

        """
        super().__init__(*args, **kwargs)


class Grid3D(BaseGrid3D):
    def __init__(self, *args, **kwargs):
        """
        3D grid class.

        Parameters
        ----------
        grid : array_like
            Grid array.
        gridsize : array_like
            Grid size (dz, dx, dy).
        origin : array_like
            Grid origin coordinates.

        """
        super().__init__(*args, **kwargs)


class TraveltimeGrid2D(BaseGrid2D, BaseTraveltime):
    def __init__(self, grid, gridsize, origin, source, gradient, vzero):
        """
        2D traveltime grid class.

        Parameters
        ----------
        grid : array_like
            Traveltime grid array.
        gridsize : array_like
            Grid size (dz, dx).
        origin : array_like
            Grid origin coordinates.
        source : array_like
            Source coordinates.
        gradient : array_like
            Gradient grid.
        vzero : scalar
            Slowness at the source.

        """
        super().__init__(
            grid=grid,
            gridsize=gridsize,
            origin=numpy.asarray(origin, dtype=numpy.float64),
            source=numpy.asarray(source, dtype=numpy.float64),
            gradient=(
                numpy.asarray(gradient, dtype=numpy.float64)
                if gradient is not None
                else None
            ),
            vzero=vzero,
        )

    def __call__(self, points, fill_value=numpy.nan):
        """
        Bilinear apparent velocity interpolation.

        Parameters
        ----------
        points : array_like
            Query point coordinates or list of point coordinates.
        fill_value : scalar, optional, default nan
            Returned value for out-of-bound query points.

        Returns
        -------
        scalar or :class:`numpy.ndarray`
            Interpolated traveltime(s).

        """
        return vinterp2d(
            self.zaxis,
            self.xaxis,
            self._grid,
            numpy.asarray(points, dtype=numpy.float64),
            self._source,
            self._vzero,
            fill_value,
        )

    def raytrace(self, points, stepsize=None, max_step=None, honor_grid=False):
        """
        2D a posteriori ray-tracing.

        Parameters
        ----------
        points : array_like
            Query point coordinates or list of point coordinates.
        stepsize : scalar or None, optional, default None
            Unit length of ray. `stepsize` is ignored if `honor_grid` is `True`.
        max_step : scalar or None, optional, default None
            Maximum number of steps.
        honor_grid : bool, optional, default False
            If `True`, coordinates of raypaths are calculated with respect to traveltime grid discretization.

        Returns
        -------
        :class:`numpy.ndarray` or list of :class:`numpy.ndarray`
            Raypath(s).

        """
        gradient = self.gradient

        if honor_grid or not stepsize:
            stepsize = numpy.min(self._gridsize)

        if not max_step:
            nz, nx = self.shape
            dz, dx = self._gridsize
            max_dist = 2.0 * ((nz * dz) ** 2 + (nx * dx) ** 2) ** 0.5
            max_step = int(max_dist / stepsize)

        return ray2d(
            self.zaxis,
            self.xaxis,
            gradient[0].grid,
            gradient[1].grid,
            numpy.asarray(points, dtype=numpy.float64),
            self._source,
            stepsize,
            max_step,
            honor_grid,
        )

    @property
    def gradient(self):
        """Return Z and X gradient grids as a list of :class:`fteikpy.Grid2D`."""
        if self._gradient is None:
            raise ValueError(
                "no gradient grid, use option `return_gradient` to return gradient grids"
            )

        return [
            Grid2D(self._gradient[:, :, i], self._gridsize, self._origin)
            for i in range(2)
        ]


class TraveltimeGrid3D(BaseGrid3D, BaseTraveltime):
    def __init__(self, grid, gridsize, origin, source, gradient, vzero):
        """
        3D traveltime grid class.

        Parameters
        ----------
        grid : array_like
            Traveltime grid array.
        gridsize : array_like
            Grid size (dz, dx, dy).
        origin : array_like
            Grid origin coordinates.
        source : array_like
            Source coordinates.
        gradient : array_like
            Gradient grid.
        vzero : scalar
            Slowness at the source.

        """
        super().__init__(
            grid=grid,
            gridsize=gridsize,
            origin=numpy.asarray(origin, dtype=numpy.float64),
            source=numpy.asarray(source, dtype=numpy.float64),
            gradient=(
                numpy.asarray(gradient, dtype=numpy.float64)
                if gradient is not None
                else None
            ),
            vzero=vzero,
        )

    def __call__(self, points, fill_value=numpy.nan):
        """
        Trilinear apparent velocity interpolation.

        Parameters
        ----------
        points : array_like
            Query point coordinates or list of point coordinates.
        fill_value : scalar, optional, default nan
            Returned value for out-of-bound query points.

        Returns
        -------
        scalar or :class:`numpy.ndarray`
            Interpolated traveltime(s).

        """
        return vinterp3d(
            self.zaxis,
            self.xaxis,
            self.yaxis,
            self._grid,
            numpy.asarray(points, dtype=numpy.float64),
            self._source,
            self._vzero,
            fill_value,
        )

    def raytrace(self, points, stepsize=None, max_step=None, honor_grid=False):
        """
        3D a posteriori ray-tracing.

        Parameters
        ----------
        points : array_like
            Query point coordinates or list of point coordinates.
        stepsize : scalar or None, optional, default None
            Unit length of ray. `stepsize` is ignored if `honor_grid` is `True`.
        max_step : scalar or None, optional, default None
            Maximum number of steps.
        honor_grid : bool, optional, default False
            If `True`, coordinates of raypaths are calculated with respect to traveltime grid discretization.

        Returns
        -------
        :class:`numpy.ndarray` or list of :class:`numpy.ndarray`
            Raypath(s).

        """
        gradient = self.gradient

        if honor_grid or not stepsize:
            stepsize = numpy.min(self._gridsize)

        if not max_step:
            nz, nx, ny = self.shape
            dz, dx, dy = self._gridsize
            max_dist = 2.0 * ((nz * dz) ** 2 + (nx * dx) ** 2 + (ny * dy) ** 2) ** 0.5
            max_step = int(max_dist / stepsize)

        return ray3d(
            self.zaxis,
            self.xaxis,
            self.yaxis,
            gradient[0].grid,
            gradient[1].grid,
            gradient[2].grid,
            numpy.asarray(points, dtype=numpy.float64),
            self._source,
            stepsize,
            max_step,
            honor_grid,
        )

    @property
    def gradient(self):
        """Return Z, X and Y gradient grids as a list of :class:`fteikpy.Grid3D`."""
        if self._gradient is None:
            raise ValueError(
                "no gradient grid, use option `return_gradient` to return gradient grids"
            )

        return [
            Grid3D(self._gradient[:, :, :, i], self._gridsize, self._origin)
            for i in range(3)
        ]
