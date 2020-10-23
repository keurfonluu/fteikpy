import numpy

from ._base import BaseGrid2D, BaseGrid3D
from ._fteik import solve2d, solve3d
from ._grid import TraveltimeGrid2D, TraveltimeGrid3D


class EikonalSolver2D(BaseGrid2D):
    def __init__(self, grid, gridsize, origin=None):
        super().__init__(
            grid=grid,
            gridsize=gridsize,
            origin=origin if origin else numpy.zeros(2),
        )

    def solve(self, sources, nsweep=2, return_gradient=False):
        tt, ttgrad, vzero = solve2d(
            1.0 / self._grid,
            *self._gridsize,
            (sources - self._origin),
            nsweep,
            return_gradient,
        )

        if isinstance(vzero, numpy.ndarray):
            return [
                TraveltimeGrid2D(
                    grid=t,
                    gridsize=self._gridsize,
                    origin=self._origin,
                    source=source,
                    gradient=tg if return_gradient else None,
                    vzero=v,
                )
                for source, t, tg, v in zip(sources, tt, ttgrad, vzero)
            ]

        else:
            return TraveltimeGrid2D(
                grid=tt,
                gridsize=self._gridsize,
                origin=self._origin,
                source=sources,
                gradient=ttgrad if return_gradient else None,
                vzero=vzero,
            )


class EikonalSolver3D(BaseGrid3D):
    def __init__(self, grid, gridsize, origin=None):
        super().__init__(
            grid=grid,
            gridsize=gridsize,
            origin=origin if origin else numpy.zeros(3),
        )

    def solve(self, sources, nsweep=2, return_gradient=False):
        tt, ttgrad, vzero = solve3d(
            1.0 / self._grid,
            *self._gridsize,
            (sources - self._origin),
            nsweep,
            return_gradient,
        )

        if isinstance(vzero, numpy.ndarray):
            return [
                TraveltimeGrid3D(
                    grid=t,
                    gridsize=self._gridsize,
                    origin=self._origin,
                    source=source,
                    gradient=tg if return_gradient else None,
                    vzero=v,
                )
                for source, t, tg, v in zip(sources, tt, ttgrad, vzero)
            ]

        else:
            return TraveltimeGrid3D(
                grid=tt,
                gridsize=self._gridsize,
                origin=self._origin,
                source=sources,
                gradient=ttgrad if return_gradient else None,
                vzero=vzero,
            )
