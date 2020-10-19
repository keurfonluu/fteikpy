import numpy

from ._base import BaseEikonalSolver
from ._fteik import solve2d, solve3d
from ._grid import TraveltimeGrid2D, TraveltimeGrid3D


class EikonalSolver2D(BaseEikonalSolver):
    def __init__(self, velocity_model, gridsize, origin=None):
        origin = origin if origin else numpy.zeros(2)
        super().__init__(velocity_model, gridsize, origin)

    def solve(self, sources, max_sweep=2):
        tt, vzero = solve2d(
            1.0 / self._velocity_model,
            *self._gridsize,
            (sources - self._origin),
            max_sweep=max_sweep,
        )

        if isinstance(tt, numpy.ndarray):
            return [
                TraveltimeGrid2D(
                    grid=t,
                    gridsize=self._gridsize,
                    origin=self._origin,
                    source=source,
                    vzero=v,
                )
                for source, t, v in zip(sources, tt, vzero)
            ]

        else:
            return TraveltimeGrid2D(
                grid=tt,
                gridsize=self._gridsize,
                origin=self._origin,
                source=sources,
                vzero=vzero,
            )


class EikonalSolver3D(BaseEikonalSolver):
    def __init__(self, velocity_model, gridsize, origin=None):
        origin = origin if origin else numpy.zeros(3)
        super().__init__(velocity_model, gridsize, origin)

    def solve(self, sources, max_sweep=2):
        tt, vzero = solve3d(
            1.0 / self._velocity_model,
            *self._gridsize,
            (sources - self._origin),
            max_sweep=max_sweep,
        )

        if isinstance(tt, numpy.ndarray):
            return [
                TraveltimeGrid3D(
                    grid=t,
                    gridsize=self._gridsize,
                    origin=self._origin,
                    source=source,
                    vzero=v,
                )
                for source, t, v in zip(sources, tt, vzero)
            ]

        else:
            return TraveltimeGrid3D(
                grid=tt,
                gridsize=self._gridsize,
                origin=self._origin,
                source=sources,
                vzero=vzero,
            )
