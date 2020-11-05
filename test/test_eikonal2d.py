import numpy
import pytest

from helpers import eik2d, allclose


@pytest.mark.parametrize(
    "sources, tref",
    (
        ([0.0, 0.0], 2969.40942920),
        ([0.5, 0.5], 2809.23711951),
        ([[0.0, 0.0], [0.5, 0.5]], [2969.40942920, 2809.23711951]),
    ),
)
def test_solve(sources, tref):
    tt = eik2d.solve(sources, nsweep=2)

    allclose(tref, tt, lambda tt: tt.grid.sum())


@pytest.mark.parametrize(
    "points, vref",
    (
        ([0.0, 0.0], 1.0),
        ([-1.0, -1.0], numpy.nan),
        (
            [
                [0.0, 0.0],
                [15.0, 0.0],
                [15.0, 15.0],
                [0.0, 15.0],
            ],
            numpy.ones(4),
        ),
    ),
)
def test_call(points, vref):
    allclose(vref, points, lambda points: eik2d(points))
