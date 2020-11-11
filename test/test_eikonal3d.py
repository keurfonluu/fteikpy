import numpy
import pytest

from helpers import eik3d, allclose


@pytest.mark.parametrize(
    "sources, tref",
    (
        ([0.0, 0.0, 0.0], 59952.28855941),
        ([3.5, 3.5, 3.5], 41432.44786414),
        ([[0.0, 0.0, 0.0], [3.5, 3.5, 3.5]], [59952.28855941, 41432.44786414]),
    ),
)
def test_solve(sources, tref):
    tt = eik3d.solve(sources, nsweep=3)

    allclose(tref, tt, lambda tt: tt.grid.sum())


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
    allclose(vref, points, lambda points: eik3d(points))
