import numpy
import pytest

from helpers import eik3d, tana


@pytest.mark.parametrize("sources", ([0.0, 0.0, 0.0], [0.005, 0.005, 0.005]))
def test_solve(sources):
    tt = eik3d.solve(sources, nsweep=2)
    z, x, y = numpy.meshgrid(eik3d.zaxis, eik3d.xaxis, eik3d.yaxis)
    points = numpy.column_stack((z.ravel(), x.ravel(), y.ravel()))
    tref = [tana(sources, point) for point in points]

    assert numpy.allclose(tt(points), tref, atol=1.0e-2)
