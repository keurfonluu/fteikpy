import numpy
import pytest
from helpers import allclose, eik3d


@pytest.mark.parametrize(
    "sources, tref",
    (
        ([0.0, 0.0, 0.0], 76631.78464590),
        ([3.5, 3.5, 3.5], 54091.60683054),
        ([[0.0, 0.0, 0.0], [3.5, 3.5, 3.5]], [76631.78464590, 54091.60683054]),
    ),
)
def test_solve(sources, tref):
    tt = eik3d.solve(sources, nsweep=3)

    allclose(tref, tt, lambda x: x.grid.sum())


@pytest.mark.parametrize(
    "points, vref",
    (
        ([0.0, 0.0, 0.0], 1.0),
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
            numpy.ones(8),
        ),
    ),
)
def test_call(points, vref):
    allclose(vref, points, lambda x: eik3d(x))
