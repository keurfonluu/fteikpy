import numpy
from numba import prange

from .._common import jitted


@jitted("f8(f8[:], f8[:], f8[:, :], f8, f8, f8)")
def _interp2d(x, y, v, xq, yq, fval):
    condx = x[0] <= xq <= x[-1]
    condy = y[0] <= yq <= y[-1]
    if not (condx and condy):
        return fval

    nx, ny = numpy.shape(v)
    nx -= 1
    ny -= 1

    i1 = numpy.nonzero(x <= xq)[0][-1]
    j1 = numpy.nonzero(y <= yq)[0][-1]
    i2 = i1 + 1
    j2 = j1 + 1

    if i1 == nx and j1 != ny:
        x1 = x[i1]
        x2 = 2.0 * x1 - x[-2]
        y1 = y[j1]
        y2 = y[j2]

        v11 = v[i1, j1]
        v21 = 1.0
        v12 = v[i1, j2]
        v22 = 1.0

    elif i1 != nx and j1 == ny:
        x1 = x[i1]
        x2 = x[i2]
        y1 = y[j1]
        y2 = 2.0 * y1 - y[-2]

        v11 = v[i1, j1]
        v21 = v[i2, j1]
        v12 = 1.0
        v22 = 1.0

    elif i1 == nx and j1 == ny:
        x1 = x[i1]
        x2 = 2.0 * x1 - x[-2]
        y1 = y[j1]
        y2 = 2.0 * y1 - y[-2]

        v11 = v[i1, j1]
        v21 = 1.0
        v12 = 1.0
        v22 = 1.0

    else:
        x1 = x[i1]
        x2 = x[i2]
        y1 = y[j1]
        y2 = y[j2]

        v11 = v[i1, j1]
        v21 = v[i2, j1]
        v12 = v[i1, j2]
        v22 = v[i2, j2]

    ax = numpy.array([x2, x1, x2, x1])
    ay = numpy.array([y2, y2, y1, y1])
    av = numpy.array([v11, v21, v12, v22])
    N = numpy.abs((ax - xq) * (ay - yq)) / numpy.abs((x2 - x1) * (y2 - y1))
    vq = numpy.dot(av, N)

    return vq


@jitted(parallel=True)
def _interp2d_vectorized(x, y, v, xq, yq, fval):
    nq = len(xq)
    out = numpy.empty(nq, dtype=numpy.float64)
    for i in prange(nq):
        out[i] = _interp2d(x, y, v, xq[i], yq[i], fval)

    return out


@jitted
def interp2d(x, y, v, q, fval=numpy.nan):
    if q.ndim == 1:
        return _interp2d(x, y, v, q[0], q[1], fval)

    else:
        return _interp2d_vectorized(x, y, v, q[:, 0], q[:, 1], fval)
