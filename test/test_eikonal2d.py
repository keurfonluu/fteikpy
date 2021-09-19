import numpy
import pytest
from helpers import allclose, eik2d


@pytest.mark.parametrize(
    "sources, tref",
    (
        ([0.0, 0.0], 2969.40942920),
        ([3.5, 3.5], 2002.26724555),
        ([[0.0, 0.0], [3.5, 3.5]], [2969.40942920, 2002.26724555]),
    ),
)
def test_solve(sources, tref):
    tt = eik2d.solve(sources, nsweep=2)

    allclose(tref, tt, lambda x: x.grid.sum())


@pytest.mark.parametrize(
    "points, vref",
    (
        ([0.0, 0.0], 1.0),
        ([-1.0, -1.0], numpy.nan),
        ([[0.0, 0.0], [15.0, 0.0], [15.0, 15.0], [0.0, 15.0],], numpy.ones(4),),
    ),
)
def test_call(points, vref):
    allclose(vref, points, lambda x: eik2d(x))
