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


@pytest.mark.parametrize(
    "i, j, k",
    [
        (0, 0, 0),
        (0, -1, 0),
        (-1, 0, 0),
        (-1, -1, 0),
        (0, 0, -1),
        (0, -1, -1),
        (-1, 0, -1),
        (-1, -1, -1),
        (1, 2, 3),
    ],
)
def test_interp(i, j, k):
    sources = numpy.random.uniform(
        [eik3d.zaxis[0], eik3d.xaxis[0], eik3d.yaxis[0]],
        [eik3d.zaxis[-1], eik3d.xaxis[-1], eik3d.yaxis[-1]],
    )
    tt = eik3d.solve(sources, nsweep=2)

    assert numpy.allclose(tt((eik3d.zaxis[i], eik3d.xaxis[j], eik3d.yaxis[k])), tt[i, j, k]) 
