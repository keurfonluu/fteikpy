import numpy as np
from numba import prange

from .._common import dist2d, jitted


@jitted("f8(f8[:], f8[:], f8[:, :], f8, f8, f8, f8, f8, f8)", boundscheck=True, debug=True)
def _vinterp2d(x, y, v, xq, yq, xsrc, ysrc, vzero, fval):
    """Perform bilinear apparent velocity interpolation."""
    condx = x[0] <= xq <= x[-1]
    condy = y[0] <= yq <= y[-1]
    if not (condx and condy):
        return fval

    xsi = np.searchsorted(x, xsrc, side="right") - 1
    ysi = np.searchsorted(y, ysrc, side="right") - 1
    i1 = np.searchsorted(x, xq, side="right") - 1
    j1 = np.searchsorted(y, yq, side="right") - 1

    if xsi == i1 and ysi == j1:
        vq = vzero * dist2d(xsrc, ysrc, xq, yq)
    else:
        nx, ny = np.shape(v)
        nx -= 1
        ny -= 1

        i2 = i1 + 1
        j2 = j1 + 1

        if i1 == nx and j1 != ny:
            x1 = x[i1]
            x2 = 2.0 * x1 - x[-2]
            y1 = y[j1]
            y2 = y[j2]

            d11 = dist2d(xsrc, ysrc, x1, y1)
            d21 = 0.0
            d12 = dist2d(xsrc, ysrc, x1, y2)
            d22 = 0.0

            v11 = v[i1, j1]
            v21 = 1.0
            v12 = v[i1, j2]
            v22 = 1.0

        elif i1 != nx and j1 == ny:
            x1 = x[i1]
            x2 = x[i2]
            y1 = y[j1]
            y2 = 2.0 * y1 - y[-2]

            d11 = dist2d(xsrc, ysrc, x1, y1)
            d21 = dist2d(xsrc, ysrc, x2, y1)
            d12 = 0.0
            d22 = 0.0

            v11 = v[i1, j1]
            v21 = v[i2, j1]
            v12 = 1.0
            v22 = 1.0

        elif i1 == nx and j1 == ny:
            x1 = x[i1]
            x2 = 2.0 * x1 - x[-2]
            y1 = y[j1]
            y2 = 2.0 * y1 - y[-2]

            d11 = dist2d(xsrc, ysrc, x1, y1)
            d21 = 0.0
            d12 = 0.0
            d22 = 0.0

            v11 = v[i1, j1]
            v21 = 1.0
            v12 = 1.0
            v22 = 1.0

        else:
            x1 = x[i1]
            x2 = x[i2]
            y1 = y[j1]
            y2 = y[j2]

            d11 = dist2d(xsrc, ysrc, x1, y1)
            d21 = dist2d(xsrc, ysrc, x2, y1)
            d12 = dist2d(xsrc, ysrc, x1, y2)
            d22 = dist2d(xsrc, ysrc, x2, y2)

            v11 = v[i1, j1]
            v21 = v[i2, j1]
            v12 = v[i1, j2]
            v22 = v[i2, j2]

        # If source is on a gridpoint, the first "if" may miss that query
        # point is close to the source
        # Only the source has zero traveltime
        if not (v11 and v21 and v12 and v22):
            vq = vzero * dist2d(xsrc, ysrc, xq, yq)

        else:
            vq = d11 / v11 * np.abs((x2 - xq) * (y2 - yq))
            vq += d21 / v21 * np.abs((x1 - xq) * (y2 - yq))
            vq += d12 / v12 * np.abs((x2 - xq) * (y1 - yq))
            vq += d22 / v22 * np.abs((x1 - xq) * (y1 - yq))
            vq /= np.abs((x2 - x1) * (y2 - y1))
            vq = dist2d(xsrc, ysrc, xq, yq) / vq

    return vq


@jitted(parallel=True)
def _vinterp2d_vectorized(x, y, v, xq, yq, xsrc, ysrc, vzero, fval):
    """Perform bilinear apparent velocity interpolation for different points."""
    nq = len(xq)
    out = np.empty(nq, dtype=np.float64)
    for i in prange(nq):
        out[i] = _vinterp2d(x, y, v, xq[i], yq[i], xsrc, ysrc, vzero, fval)

    return out


@jitted
def vinterp2d(x, y, v, q, src, vzero, fval=np.nan):
    """Perform bilinear apparent velocity interpolation."""
    if q.ndim == 1:
        return _vinterp2d(x, y, v, q[0], q[1], src[0], src[1], vzero, fval)

    else:
        return _vinterp2d_vectorized(
            x, y, v, q[:, 0], q[:, 1], src[0], src[1], vzero, fval
        )
