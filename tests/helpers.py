import numpy as np

from fteikpy import Eikonal2D, Eikonal3D

np.random.seed(42)


eik2d = Eikonal2D(grid=np.ones((16, 16), dtype=np.float64), gridsize=(1.0, 1.0))


eik3d = Eikonal3D(
    grid=np.ones((16, 16, 16), dtype=np.float64), gridsize=(1.0, 1.0, 1.0),
)


def allclose(a, b, bfun, atol=1.0e-8):
    def allclose_nan(a, b):
        try:
            if np.isnan(a):
                assert np.isnan(b)

            else:
                assert np.allclose(a, b, atol=atol)

        except AssertionError:
            raise AssertionError(f"{a} != {b}")

    ndim = np.ndim(a)

    if ndim == 0:
        allclose_nan(a, bfun(b))

    elif ndim == 1:
        for aa, bb in zip(a, b):
            allclose_nan(aa, bfun(bb))

    else:
        raise ValueError()
