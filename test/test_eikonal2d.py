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


@pytest.mark.parametrize(
    "i, j",
    [
        (0, 0),
        (0, -1),
        (-1, 0),
        (-1, -1),
        (1, 2),
    ],
)
def test_interp(i, j):
    sources = numpy.random.uniform(
        [eik2d.zaxis[0], eik2d.xaxis[0]],
        [eik2d.zaxis[-1], eik2d.xaxis[-1]],
    )
    tt = eik2d.solve(sources, nsweep=2)

    assert numpy.allclose(tt((eik2d.zaxis[i], eik2d.xaxis[j])), tt[i, j]) 
