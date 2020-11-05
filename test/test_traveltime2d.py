import numpy
import pytest

from helpers import eik2d, allclose


@pytest.mark.parametrize(
    "points, tref",
    (
        ([0.0, 0.0], 0.0),
        ([0.5, 0.5], 0.70710678),
        ([1.5, 1.5], 2.12132034),
        ([-1.0, -1.0], numpy.nan),
        (
            [
                [0.0, 0.0],
                [15.0, 0.0],
                [15.0, 15.0],
                [0.0, 15.0],
            ],
            [0.0, 15.0, 21.21320343, 15.0],
        ),
    ),
)
def test_call(points, tref):
    sources = 0.0, 0.0
    tt = eik2d.solve(sources, nsweep=2)

    allclose(tref, points, lambda points: tt(points))
