import numpy
import pytest
from helpers import allclose, eik3d


@pytest.mark.parametrize(
    "points, tref",
    (
        ([0.0, 0.0, 0.0], 0.0),
        ([0.5, 0.5, 0.5], 0.86602540),
        ([1.5, 1.5, 1.5], 2.60631816),
        ([-1.0, -1.0, -1.0], numpy.nan),
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

    allclose(tref, points, lambda points: tt(points))


@pytest.mark.parametrize(
    "points, pref",
    (
        ([0.5, 0.5, 0.5], 1.5),
        ([1.5, 1.5, 1.5], 8.30384757),
        ([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]], [1.5, 8.30384757]),
    ),
)
def test_raytrace(points, pref):
    sources = 0.0, 0.0, 0.0
    tt = eik3d.solve(sources, nsweep=3, return_gradient=True)
    rays = tt.raytrace(points, stepsize=1.0)

    allclose(pref, rays, lambda rays: rays.sum())
