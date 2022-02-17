import numpy as np

from ._base import BaseGrid2D, BaseGrid3D
from ._fteik import solve2d, solve3d
from ._grid import TraveltimeGrid2D, TraveltimeGrid3D


class Eikonal2D(BaseGrid2D):
    def __init__(self, grid, gridsize, origin=None):
        """
        2D Eikonal solver.

        Parameters
        ----------
        grid : array_like
            Velocity model array.
        gridsize : array_like
            Grid size (dz, dx).
        origin : array_like or None, optional, default None
            Grid origin coordinates.

        """
        super().__init__(
            grid=grid,
            gridsize=gridsize,
            origin=origin
            if origin is not None
            else np.zeros(2, dtype=np.float64),
        )

    def solve(self, sources, nsweep=2, return_gradient=False):
        """
        Solve Eikonal for given sources.

        Parameters
        ----------
        sources : array_like
            Source coordinates or list of source coordinates.
        nsweep : int, optional, default 2
            Number of sweeps.
        return_gradient : bool, optional, default False
            If `True`, directions of gradient are computed at runtime. However, this option uses more memory as the gradient grid is saved. Gradient grids are required for a posteriori ray-tracing.

        Returns
        -------
        :class:`fteikpy.TraveltimeGrid2D` or list of :class:`fteikpy.TraveltimeGrid2D`
            Traveltime grid or list of traveltime grids.

        """
        tt, ttgrad, vzero = solve2d(
            1.0 / self._grid,
            *self._gridsize,
            (sources - self._origin),
            nsweep,
            return_gradient,
        )

        if isinstance(vzero, np.ndarray):
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


class Eikonal3D(BaseGrid3D):
    def __init__(self, grid, gridsize, origin=None):
        """
        3D Eikonal solver.

        Parameters
        ----------
        grid : array_like
            Velocity model array.
        gridsize : array_like
            Grid size (dz, dx, dy).
        origin : array_like or None, optional, default None
            Grid origin coordinates.

        """
        super().__init__(
            grid=grid,
            gridsize=gridsize,
            origin=origin
            if origin is not None
            else np.zeros(3, dtype=np.float64),
        )

    def solve(self, sources, nsweep=2, return_gradient=False):
        """
        Solve Eikonal for given sources.

        Parameters
        ----------
        sources : array_like
            Source coordinates or list of source coordinates.
        nsweep : int, optional, default 2
            Number of sweeps.
        return_gradient : bool, optional, default False
            If `True`, directions of gradient are computed at runtime. However, this option uses more memory as the gradient grid is saved. Gradient grids are required for a posteriori ray-tracing.

        Returns
        -------
        :class:`fteikpy.TraveltimeGrid3D` or list of :class:`fteikpy.TraveltimeGrid3D`
            Traveltime grid or list of traveltime grids.

        """
        tt, ttgrad, vzero = solve3d(
            1.0 / self._grid,
            *self._gridsize,
            (sources - self._origin),
            nsweep,
            return_gradient,
        )

        if isinstance(vzero, np.ndarray):
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
