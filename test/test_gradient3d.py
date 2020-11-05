import numpy
import pytest

from helpers import eik3d, allclose


@pytest.mark.parametrize(
    "points, gzref, gxref, gyref",
    (
        ([0.0, 0.0, 0.0], 0.0, 0.0, 0.0),
        ([-1.0, -1.0, -1.0], numpy.nan, numpy.nan, numpy.nan),
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
            [0.0, 1.0, 0.68766786, 0.0, 0.0, 0.68766786, 0.55795306, 0.0],
            [0.0, 0.0, 0.68766786, 1.0, 0.0, 0.0, 0.55795306, 0.68766786],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.68766786, 0.55795306, 0.68766786],
        ),
    ),
)
def test_call(points, gzref, gxref, gyref):
    sources = 0.0, 0.0, 0.0
    tt = eik3d.solve(sources, nsweep=2, return_gradient=True)

    allclose(gzref, points, lambda points: tt.gradient_z(points))
    allclose(gxref, points, lambda points: tt.gradient_x(points))
    allclose(gyref, points, lambda points: tt.gradient_y(points))
