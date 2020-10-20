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

    def __call__(self, points):
        return vinterp2d(
            self.zaxis,
            self.xaxis,
            self._grid,
            numpy.asarray(points, dtype=numpy.float64),
            self._source,
            self._vzero,
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

    def __call__(self, points):
        return vinterp3d(
            self.zaxis,
            self.xaxis,
            self.yaxis,
            self._grid,
            numpy.asarray(points, dtype=numpy.float64),
            self._source,
            self._vzero,
        )
