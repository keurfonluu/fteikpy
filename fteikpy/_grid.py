import numpy

from ._base import BaseTraveltimeGrid
from ._fteik import interp2d


class TraveltimeGrid2D(BaseTraveltimeGrid):
    def __init__(self, grid, gridsize, origin, source, vzero):
        super().__init__(grid, gridsize, origin, source, vzero)

    def __call__(self, points):
        t = interp2d(
            self.zaxis,
            self.xaxis,
            self._grid,
            *points,
            *self._source,
            self._vzero,
        )
        
        return t
