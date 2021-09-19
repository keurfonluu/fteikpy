import numpy
import pytest
from helpers import allclose, eik2d


@pytest.mark.parametrize(
    "sources, points, gzref, gxref",
    (
        ([0.0, 0.0], [0.0, 0.0], 0.0, 0.0),
        ([3.5, 3.5], [0.0, 0.0], -0.64858483, -0.64858483),
        ([0.0, 0.0], [-1.0, -1.0], numpy.nan, numpy.nan),
        (
            [0.0, 0.0],
            [[0.0, 0.0], [15.0, 0.0], [15.0, 15.0], [0.0, 15.0],],
            [0.0, 1.0, 0.69457650, 0.0],
            [0.0, 0.0, 0.69457650, 1.0],
        ),
    ),
)
def test_call(sources, points, gzref, gxref):
    tt = eik2d.solve(sources, nsweep=2, return_gradient=True)

    allclose(gzref, points, lambda x: tt.gradient[0](x))
    allclose(gxref, points, lambda x: tt.gradient[1](x))
