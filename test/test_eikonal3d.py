import numpy
import pytest

from helpers import eik3d, allclose


@pytest.mark.parametrize(
    "sources, tref",
    (
        ([0.0, 0.0, 0.0], 60015.56014379),
        ([0.5, 0.5, 0.5], 57442.51188274),
        ([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], [60015.56014379, 57442.51188274]),
    ),
)
def test_solve(sources, tref):
    tt = eik3d.solve(sources, nsweep=2)

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
