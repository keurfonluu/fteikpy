import numpy
from numba import prange

from .._common import dist3d, jitted, norm3d
from .._interp import interp3d


@jitted(
    "f8[:, :](f8[:], f8[:], f8[:], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8, f8, f8, f8, f8, f8, f8)"
)
def _ray3d(z, x, y, zgrad, xgrad, ygrad, zend, xend, yend, zsrc, xsrc, ysrc, stepsize):
    condz = z[0] <= zend <= z[-1]
    condx = x[0] <= xend <= x[-1]
    condy = y[0] <= yend <= y[-1]
    if not (condz and condx and condy):
        raise ValueError("end point out of bound")

    pcur = numpy.array([zend, xend, yend], dtype=numpy.float64)
    ray = [pcur.copy()]
    while dist3d(zsrc, xsrc, ysrc, pcur[0], pcur[1], pcur[2]) > stepsize:
        gz = interp3d(z, x, y, zgrad, pcur)
        gx = interp3d(z, x, y, xgrad, pcur)
        gy = interp3d(z, x, y, ygrad, pcur)
        gn = norm3d(gz, gx, gy)

        if gn > 0.0:
            gni = 1.0 / gn
        else:
            break

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
def _ray3d_vectorized(
    z, x, y, zgrad, xgrad, ygrad, zend, xend, yend, zsrc, xsrc, ysrc, stepsize
):
    out = []
    for i in prange(len(zend)):
        out.append(
            _ray3d(
                z,
                x,
                y,
                zgrad,
                xgrad,
                ygrad,
                zend[i],
                xend[i],
                yend[i],
                zsrc,
                xsrc,
                ysrc,
                stepsize,
            )
        )

    return out


@jitted
def ray3d(z, x, y, zgrad, xgrad, ygrad, p, src, stepsize):
    if p.ndim == 1:
        return _ray3d(
            z,
            x,
            y,
            zgrad,
            xgrad,
            ygrad,
            p[0],
            p[1],
            p[2],
            src[0],
            src[1],
            src[2],
            stepsize,
        )

    else:
        return _ray3d_vectorized(
            z,
            x,
            y,
            zgrad,
            xgrad,
            ygrad,
            p[:, 0],
            p[:, 1],
            p[:, 2],
            src[0],
            src[1],
            src[2],
            stepsize,
        )
