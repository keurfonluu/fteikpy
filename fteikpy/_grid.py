import numpy

from ._base import BaseGrid2D, BaseGrid3D, BaseTraveltime
from ._interp import vinterp2d, vinterp3d


class TraveltimeGrid2D(BaseGrid2D, BaseTraveltime):
    def __init__(self, grid, gridsize, origin, source, grad, vzero):
        super().__init__(
            grid=grid,
            gridsize=gridsize,
            origin=origin,
            source=source,
            grad=grad,
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

    @property
    def grad(self):
        return (
            BaseGrid2D(self._grad, self._gridsize, self._origin)
            if self._grad is not None
            else None
        )


class TraveltimeGrid3D(BaseGrid3D, BaseTraveltime):
    def __init__(self, grid, gridsize, origin, source, grad, vzero):
        super().__init__(
            grid=grid,
            gridsize=gridsize,
            origin=origin,
            source=source,
            grad=grad,
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

    @property
    def grad(self):
        return (
            BaseGrid3D(self._grad, self._gridsize, self._origin)
            if self._grad is not None
            else None
        )
