import numpy

from numba import prange

from .._common import dist3d, jitted, norm3d
from .._interp import interp3d


@jitted("f8[:, :](f8[:], f8[:], f8[:], f8[:, :, :, :], f8, f8, f8, f8, f8, f8, f8)")
def _ray3d(z, x, y, ttgrad, zend, xend, yend, zsrc, xsrc, ysrc, stepsize):
    condz = z[0] <= zend <= z[-1]
    condx = x[0] <= xend <= x[-1]
    condy = y[0] <= yend <= y[-1]
    if not (condz and condx and condy):
        raise ValueError("end point out of bound")

    pcur = numpy.array([zend, xend, yend], dtype=numpy.float64)
    ray = [pcur.copy()]
    while dist3d(zsrc, xsrc, ysrc, pcur[0], pcur[1], pcur[2]) > stepsize:
        gz = interp3d(z, x, y, ttgrad[:, :, :, 0], pcur)
        gx = interp3d(z, x, y, ttgrad[:, :, :, 1], pcur)
        gy = interp3d(z, x, y, ttgrad[:, :, :, 2], pcur)
        gni = 1.0 / norm3d(gz, gx, gy)

        pcur[0] -= stepsize * gz * gni
        pcur[1] -= stepsize * gx * gni
        pcur[2] -= stepsize * gy * gni
        ray.append(pcur.copy())
    ray.append(numpy.array([zsrc, xsrc, ysrc], dtype=numpy.float64))

    out = numpy.empty((len(ray), 3), dtype=numpy.float64)
    for i, r in enumerate(ray):
        out[-i - 1] = r

    return out


@jitted(parallel=True)
def ray3d(z, x, y, ttgrad, p, src, stepsize):
    if p.ndim == 1:
        return _ray3d(z, x, y, ttgrad, p[0], p[1], p[2], src[0], src[1], src[2], stepsize)

    elif p.ndim == 2:
        out = []
        for i in prange(len(p)):
            out.append(_ray3d(z, x, y, ttgrad, p[i, 0], p[i, 1], p[0, 2], src[0], src[1], src[2], stepsize))

        return out

    else:
        raise ValueError()
