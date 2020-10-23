import numpy

from ._base import BaseGrid2D, BaseGrid3D, BaseTraveltime
from ._fteik import ray2d, ray3d
from ._interp import vinterp2d, vinterp3d


class Grid2D(BaseGrid2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Grid3D(BaseGrid3D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TraveltimeGrid2D(BaseGrid2D, BaseTraveltime):
    def __init__(self, grid, gridsize, origin, source, gradient, vzero):
        super().__init__(
            grid=grid,
            gridsize=gridsize,
            origin=origin,
            source=source,
            gradient=gradient,
            vzero=vzero,
        )

    def __call__(self, points, fill_value=numpy.nan):
        return vinterp2d(
            self.zaxis,
            self.xaxis,
            self._grid,
            numpy.asarray(points, dtype=numpy.float64),
            self._source,
            self._vzero,
            fill_value,
        )

    def raytrace(self, points, stepsize=None):
        if self._gradient is None:
            raise ValueError(
                "no gradient array to perform ray tracing, use option 'return_gradient' to return gradient array"
            )

        stepsize = stepsize if stepsize else numpy.min(self._gridsize)

        return ray2d(
            self.zaxis,
            self.xaxis,
            self._gradient,
            numpy.asarray(points, dtype=numpy.float64),
            self._source,
            stepsize,
        )

    @property
    def gradient_z(self):
        return (
            Grid2D(self._gradient[:, :, 0], self._gridsize, self._origin)
            if self._gradient is not None
            else None
        )

    @property
    def gradient_x(self):
        return (
            Grid2D(self._gradient[:, :, 1], self._gridsize, self._origin)
            if self._gradient is not None
            else None
        )


class TraveltimeGrid3D(BaseGrid3D, BaseTraveltime):
    def __init__(self, grid, gridsize, origin, source, gradient, vzero):
        super().__init__(
            grid=grid,
            gridsize=gridsize,
            origin=origin,
            source=source,
            gradient=gradient,
            vzero=vzero,
        )

    def __call__(self, points, fill_value=numpy.nan):
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

    def raytrace(self, points, stepsize=None):
        if self._gradient is None:
            raise ValueError(
                "no gradient array to perform ray tracing, use option 'return_gradient' to return gradient array"
            )

        stepsize = stepsize if stepsize else numpy.min(self._gridsize)

        return ray3d(
            self.zaxis,
            self.xaxis,
            self.yaxis,
            self._gradient,
            numpy.asarray(points, dtype=numpy.float64),
            self._source,
            stepsize,
        )

    @property
    def gradient_z(self):
        return (
            Grid3D(self._gradient[:, :, :, 0], self._gridsize, self._origin)
            if self._gradient is not None
            else None
        )

    @property
    def gradient_x(self):
        return (
            Grid3D(self._gradient[:, :, :, 1], self._gridsize, self._origin)
            if self._gradient is not None
            else None
        )

    @property
    def gradient_y(self):
        return (
            Grid3D(self._gradient[:, :, :, 2], self._gridsize, self._origin)
            if self._gradient is not None
            else None
        )
