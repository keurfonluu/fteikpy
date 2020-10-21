import numpy

from numba import prange

from .._common import dist2d, jitted, norm2d
from .._interp import interp2d


@jitted("f8[:, :](f8[:], f8[:], f8[:, :, :], f8, f8, f8, f8, f8)")
def _ray2d(z, x, ttgrad, zend, xend, zsrc, xsrc, stepsize):
    condz = z[0] <= zend <= z[-1]
    condx = x[0] <= xend <= x[-1]
    if not (condz and condx):
        raise ValueError("end point out of bound")

    pcur = numpy.array([zend, xend], dtype=numpy.float64)
    ray = [pcur.copy()]
    while dist2d(zsrc, xsrc, pcur[0], pcur[1]) > stepsize:
        gz = interp2d(z, x, ttgrad[:, :, 0], pcur)
        gx = interp2d(z, x, ttgrad[:, :, 1], pcur)
        gni = 1.0 / norm2d(gz, gx)

        pcur[0] -= stepsize * gz * gni
        pcur[1] -= stepsize * gx * gni
        ray.append(pcur.copy())
    ray.append(numpy.array([zsrc, xsrc], dtype=numpy.float64))

    out = numpy.empty((len(ray), 2), dtype=numpy.float64)
    for i, r in enumerate(ray):
        out[-i - 1] = r

    return out


@jitted(parallel=True)
def ray2d(z, x, ttgrad, p, src, stepsize):
    if p.ndim == 1:
        return _ray2d(z, x, ttgrad, p[0], p[1], src[0], src[1], stepsize)

    elif p.ndim == 2:
        out = []
        for i in prange(len(p)):
            out.append(_ray2d(z, x, ttgrad, p[i, 0], p[i, 1], src[0], src[1], stepsize))

        return out

    else:
        raise ValueError()
