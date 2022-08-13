import numpy as np
import pytest
from helpers import allclose, eik3d


@pytest.mark.parametrize(
    "points, tref",
    (
        ([0.0, 0.0, 0.0], 0.0),
        ([0.5, 0.5, 0.5], 0.86602540),
        ([1.5, 1.5, 1.5], 2.60631816),
        ([-1.0, -1.0, -1.0], np.nan),
        (
            [
                [0.0, 0.0, 0.0],
                [15.0, 0.0, 0.0],
                [15.0, 15.0, 0.0],
                [0.0, 15.0, 0.0],
                [0.0, 0.0, 15.0],
                [15.0, 0.0, 15.0],
                [15.0, 15.0, 15.0],
                [0.0, 15.0, 15.0],
            ],
            [0.0, 15.0, 21.21320343, 15.0, 15.0, 21.21320343, 25.98076211, 21.21320343],
        ),
    ),
)
def test_call(points, tref):
    sources = 0.0, 0.0, 0.0
    tt = eik3d.solve(sources, nsweep=3)

    allclose(tref, points, lambda x: tt(x))


@pytest.mark.parametrize(
    "points, pref, honor_grid",
    (
        ([0.5, 0.5, 0.5], 1.5, False),
        ([0.5, 0.5, 0.5], 1.5, True),
        ([1.5, 1.5, 1.5], 8.30384757, False),
        ([1.5, 1.5, 1.5], 7.5, True),
        ([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]], [1.5, 8.30384757], False),
        ([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]], [1.5, 7.5], True),
    ),
)
def test_raytrace(points, pref, honor_grid):
    sources = 0.0, 0.0, 0.0
    tt = eik3d.solve(sources, nsweep=3, return_gradient=True)
    rays = tt.raytrace(points, stepsize=1.0, honor_grid=honor_grid)

    allclose(pref, rays, lambda x: x.sum())
