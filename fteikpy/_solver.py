import numpy

from ._base import BaseEikonalSolver
from ._fteik import fteik2d, fteik3d
from ._grid import TraveltimeGrid2D, TraveltimeGrid3D


class EikonalSolver2D(BaseEikonalSolver):
    def __init__(self, velocity_model, gridsize, origin=None):
        origin = origin if origin else numpy.zeros(2)
        super().__init__(velocity_model, gridsize, origin)

    def solve(self, source, max_sweep=2):
        tt, vzero = fteik2d(
            1.0 / self._velocity_model,
            *self._gridsize,
            *(source - self._origin),
            max_sweep=max_sweep,
        )

        return TraveltimeGrid2D(
            grid=tt,
            gridsize=self._gridsize,
            origin=self._origin,
            source=source,
            vzero=vzero,
        )


class EikonalSolver3D(BaseEikonalSolver):
    def __init__(self, velocity_model, gridsize, origin=None):
        origin = origin if origin else numpy.zeros(3)
        super().__init__(velocity_model, gridsize, origin)

    def solve(self, source, max_sweep=2):
        tt, vzero = fteik3d(
            1.0 / self._velocity_model,
            *self._gridsize,
            *(source - self._origin),
            max_sweep=max_sweep,
        )

        return TraveltimeGrid3D(
            grid=tt,
            gridsize=self._gridsize,
            origin=self._origin,
            source=source,
            vzero=vzero,
        )
