import numpy

from fteikpy import EikonalSolver2D, EikonalSolver3D

numpy.random.seed(42)


eik2d = EikonalSolver2D(
    grid=numpy.ones((16, 16), dtype=numpy.float64),
    gridsize=(0.01, 0.01),
)


eik3d = EikonalSolver3D(
    grid=numpy.ones((16, 16, 16), dtype=numpy.float64),
    gridsize=(0.01, 0.01, 0.01),
)


def tana(x, y, v=1.0):
    return numpy.linalg.norm(x - y) / v
