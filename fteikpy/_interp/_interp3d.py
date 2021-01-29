import numpy
from numba import prange

from .._common import jitted


@jitted("f8(f8[:], f8[:], f8[:], f8[:, :, :], f8, f8, f8, f8)")
def _interp3d(x, y, z, v, xq, yq, zq, fval):
    """Perform trilinear interpolation."""
    condx = x[0] <= xq <= x[-1]
    condy = y[0] <= yq <= y[-1]
    condz = z[0] <= zq <= z[-1]
    if not (condx and condy and condz):
        return fval

    nx, ny, nz = numpy.shape(v)
    nx -= 1
    ny -= 1
    nz -= 1

    i1 = numpy.searchsorted(x, xq, side="right") - 1
    j1 = numpy.searchsorted(y, yq, side="right") - 1
    k1 = numpy.searchsorted(z, zq, side="right") - 1
    i2 = i1 + 1
    j2 = j1 + 1
    k2 = k1 + 1

    if i1 == nx and j1 != ny and k1 != nz:
        x1 = x[i1]
        x2 = 2.0 * x1 - x[-2]
        y1 = y[j1]
        y2 = y[j2]
        z1 = z[k1]
        z2 = z[k2]

        v111 = v[i1, j1, k1]
        v211 = 1.0
        v121 = v[i1, j2, k1]
        v221 = 1.0
        v112 = v[i1, j1, k2]
        v212 = 1.0
        v122 = v[i1, j2, k2]
        v222 = 1.0

    elif i1 != nx and j1 == ny and k1 != nz:
        x1 = x[i1]
        x2 = x[i2]
        y1 = y[j1]
        y2 = 2.0 * y1 - y[-2]
        z1 = z[k1]
        z2 = z[k2]

        v111 = v[i1, j1, k1]
        v211 = v[i2, j1, k1]
        v121 = 1.0
        v221 = 1.0
        v112 = v[i1, j1, k2]
        v212 = v[i2, j1, k2]
        v122 = 1.0
        v222 = 1.0

    elif i1 != nx and j1 != ny and k1 == nz:
        x1 = x[i1]
        x2 = x[i2]
        y1 = y[j1]
        y2 = y[j2]
        z1 = z[k1]
        z2 = 2.0 * z1 - z[-2]

        v111 = v[i1, j1, k1]
        v211 = v[i2, j1, k1]
        v121 = v[i1, j2, k1]
        v221 = v[i2, j2, k1]
        v112 = 1.0
        v212 = 1.0
        v122 = 1.0
        v222 = 1.0

    elif i1 == nx and j1 == ny and k1 != nz:
        x1 = x[i1]
        x2 = 2.0 * x1 - x[-2]
        y1 = y[j1]
        y2 = 2.0 * y1 - y[-2]
        z1 = z[k1]
        z2 = z[k2]

        v111 = v[i1, j1, k1]
        v211 = 1.0
        v121 = 1.0
        v221 = 1.0
        v112 = v[i1, j1, k2]
        v212 = 1.0
        v122 = 1.0
        v222 = 1.0

    elif i1 == nx and j1 != ny and k1 == nz:
        x1 = x[i1]
        x2 = 2.0 * x1 - x[-2]
        y1 = y[j1]
        y2 = y[j2]
        z1 = z[k1]
        z2 = 2.0 * z1 - z[-2]

        v111 = v[i1, j1, k1]
        v211 = 1.0
        v121 = v[i1, j2, k1]
        v221 = 1.0
        v112 = 1.0
        v212 = 1.0
        v122 = 1.0
        v222 = 1.0

    elif i1 != nx and j1 == ny and k1 == nz:
        x1 = x[i1]
        x2 = x[i2]
        y1 = y[j1]
        y2 = 2.0 * y1 - y[-2]
        z1 = z[k1]
        z2 = 2.0 * z1 - z[-2]

        v111 = v[i1, j1, k1]
        v211 = v[i2, j1, k1]
        v121 = 1.0
        v221 = 1.0
        v112 = 1.0
        v212 = 1.0
        v122 = 1.0
        v222 = 1.0

    elif i1 == nx and j1 == ny and k1 == nz:
        x1 = x[i1]
        x2 = 2.0 * x1 - x[-2]
        y1 = y[j1]
        y2 = 2.0 * y1 - y[-2]
        z1 = z[k1]
        z2 = 2.0 * z1 - z[-2]

        v111 = v[i1, j1, k1]
        v211 = 1.0
        v121 = 1.0
        v221 = 1.0
        v112 = 1.0
        v212 = 1.0
        v122 = 1.0
        v222 = 1.0

    else:
        x1 = x[i1]
        x2 = x[i2]
        y1 = y[j1]
        y2 = y[j2]
        z1 = z[k1]
        z2 = z[k2]

        v111 = v[i1, j1, k1]
        v211 = v[i2, j1, k1]
        v121 = v[i1, j2, k1]
        v221 = v[i2, j2, k1]
        v112 = v[i1, j1, k2]
        v212 = v[i2, j1, k2]
        v122 = v[i1, j2, k2]
        v222 = v[i2, j2, k2]

    vq = v111 * numpy.abs((x2 - xq) * (y2 - yq) * (z2 - zq))
    vq += v211 * numpy.abs((x1 - xq) * (y2 - yq) * (z2 - zq))
    vq += v121 * numpy.abs((x2 - xq) * (y1 - yq) * (z2 - zq))
    vq += v221 * numpy.abs((x1 - xq) * (y1 - yq) * (z2 - zq))
    vq += v112 * numpy.abs((x2 - xq) * (y2 - yq) * (z1 - zq))
    vq += v212 * numpy.abs((x1 - xq) * (y2 - yq) * (z1 - zq))
    vq += v122 * numpy.abs((x2 - xq) * (y1 - yq) * (z1 - zq))
    vq += v222 * numpy.abs((x1 - xq) * (y1 - yq) * (z1 - zq))
    vq /= numpy.abs((x2 - x1) * (y2 - y1) * (z2 - z1))

    return vq


@jitted(parallel=True)
def _interp3d_vectorized(x, y, z, v, xq, yq, zq, fval):
    """Perform trilinear interpolation for different points."""
    nq = len(xq)
    out = numpy.empty(nq, dtype=numpy.float64)
    for i in prange(nq):
        out[i] = _interp3d(x, y, z, v, xq[i], yq[i], zq[i], fval)

    return out


@jitted
def interp3d(x, y, z, v, q, fval=numpy.nan):
    """Perform trilinear interpolation."""
    if q.ndim == 1:
        return _interp3d(x, y, z, v, q[0], q[1], q[2], fval)

    else:
        return _interp3d_vectorized(x, y, z, v, q[:, 0], q[:, 1], q[:, 2], fval)
