import numpy
import pytest

from helpers import eik2d, tana


@pytest.mark.parametrize("sources", ([0.0, 0.0], [0.005, 0.005]))
def test_solve(sources):
    tt = eik2d.solve(sources, nsweep=2)
    z, x = numpy.meshgrid(eik2d.zaxis, eik2d.xaxis)
    points = numpy.column_stack((z.ravel(), x.ravel()))
    tref = [tana(sources, point) for point in points]

    assert numpy.allclose(tt(points), tref, atol=1.0e-3)
