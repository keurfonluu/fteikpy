from copy import deepcopy

import numpy
import pytest
from helpers import allclose, eik2d

eik2d_copy = deepcopy(eik2d)


@pytest.mark.parametrize(
    "sources, tref",
    (
        ([0.0, 0.0], 3573.41870888),
        ([3.5, 3.5], 2462.96106736),
        ([[0.0, 0.0], [3.5, 3.5]], [3573.41870888, 2462.96106736]),
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
    allclose(vref, points, lambda x: eik2d(x))


def test_resample():
    eik2d = deepcopy(eik2d_copy)
    nz, nx = eik2d.shape
    eik2d.resample((nz * 2, nx * 2))

    assert eik2d.grid.sum() == nz * nx * 4.0


def test_smooth():
    eik2d = deepcopy(eik2d_copy)
    eik2d.smooth(1.0)

    assert eik2d.grid.sum() == eik2d.grid.size
