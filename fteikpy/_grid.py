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
        stepsize = stepsize if stepsize else numpy.min(self._gridsize)
        gradient = self.gradient

        return ray2d(
            self.zaxis,
            self.xaxis,
            gradient[0].grid,
            gradient[1].grid,
            numpy.asarray(points, dtype=numpy.float64),
            self._source,
            stepsize,
        )

    @property
    def gradient(self):
        return (
            [
                Grid2D(self._gradient[:, :, i], self._gridsize, self._origin)
                for i in range(2)
            ]
            if self._gradient is not None
            else [
                Grid2D(grad, self._gridsize, self._origin)
                for grad in numpy.gradient(self._grid, *self._gridsize)
            ]
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
        stepsize = stepsize if stepsize else numpy.min(self._gridsize)
        gradient = self.gradient

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
        )

    @property
    def gradient(self):
        return (
            [
                Grid3D(self._gradient[:, :, :, i], self._gridsize, self._origin)
                for i in range(3)
            ]
            if self._gradient is not None
            else [
                Grid3D(grad, self._gridsize, self._origin)
                for grad in numpy.gradient(self._grid, *self._gridsize)
            ]
        )
