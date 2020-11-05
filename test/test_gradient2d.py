import numpy
import pytest

from helpers import eik2d, allclose


@pytest.mark.parametrize(
    "points, gzref, gxref",
    (
        ([0.0, 0.0], 0.0, 0.0),
        ([-1.0, -1.0], numpy.nan, numpy.nan),
        (
            [
                [0.0, 0.0],
                [15.0, 0.0],
                [15.0, 15.0],
                [0.0, 15.0],
            ],
            [0.0, 1.0, 0.69457650, 0.0],
            [0.0, 0.0, 0.69457650, 1.0],
        ),
    ),
)
def test_call(points, gzref, gxref):
    sources = 0.0, 0.0
    tt = eik2d.solve(sources, nsweep=2, return_gradient=True)

    allclose(gzref, points, lambda points: tt.gradient_z(points))
    allclose(gxref, points, lambda points: tt.gradient_x(points))
