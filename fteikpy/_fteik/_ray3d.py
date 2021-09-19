import numpy
from numba import prange

from .._common import dist3d, jitted, norm3d
from .._interp import interp3d
from ._common import first_index, shrink


@jitted(
    "f8[:, :](f8[:], f8[:], f8[:], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8, f8, f8, f8, f8, f8, f8, b1)"
)
def _ray3d(
    z,
    x,
    y,
    zgrad,
    xgrad,
    ygrad,
    zend,
    xend,
    yend,
    zsrc,
    xsrc,
    ysrc,
    stepsize,
    honor_grid,
):
    """Perform a posteriori 3D ray-tracing."""
    condz = z[0] <= zend <= z[-1]
    condx = x[0] <= xend <= x[-1]
    condy = y[0] <= yend <= y[-1]
    if not (condz and condx and condy):
        raise ValueError("end point out of bound")

    if honor_grid:
        nz, nx, ny = len(z), len(x), len(y)

        i = numpy.searchsorted(z, zend, side="right") - 1
        j = numpy.searchsorted(x, xend, side="right") - 1
        k = numpy.searchsorted(y, yend, side="right") - 1
        zmin = z[max(i - 1, 0)] if zend == z[i] else z[i]
        xmin = x[max(j - 1, 0)] if xend == x[j] else x[j]
        ymin = y[max(k - 1, 0)] if yend == y[k] else y[k]
        lower = numpy.array([zmin, xmin, ymin])
        upper = numpy.array(
            [z[min(i + 1, nz - 1)], x[min(j + 1, nx - 1)], y[min(k + 1, ny - 1)]]
        )

        isrc = numpy.searchsorted(z, zsrc, side="right") - 1
        jsrc = numpy.searchsorted(x, xsrc, side="right") - 1
        ksrc = numpy.searchsorted(y, ysrc, side="right") - 1

    pcur = numpy.array([zend, xend, yend], dtype=numpy.float64)
    delta = numpy.empty(3, dtype=numpy.float64)
    ray = [pcur.copy()]
    while dist3d(zsrc, xsrc, ysrc, pcur[0], pcur[1], pcur[2]) >= stepsize:
        gz = interp3d(z, x, y, zgrad, pcur)
        gx = interp3d(z, x, y, xgrad, pcur)
        gy = interp3d(z, x, y, ygrad, pcur)
        gn = norm3d(gz, gx, gy)

        if gn > 0.0:
            gni = 1.0 / gn
        else:
            break

        delta[0] = stepsize * gz * gni
        delta[1] = stepsize * gx * gni
        delta[2] = stepsize * gy * gni

        if honor_grid:
            fac = shrink(pcur, delta, lower, upper)
            pcur -= fac * delta

            if fac < 1.0:
                i = numpy.searchsorted(z, pcur[0], side="right") - 1
                j = numpy.searchsorted(x, pcur[1], side="right") - 1
                k = numpy.searchsorted(y, pcur[2], side="right") - 1
                lower[0] = z[max(i - 1, 0)] if pcur[0] == z[i] else z[i]
                lower[1] = x[max(j - 1, 0)] if pcur[1] == x[j] else x[j]
                lower[2] = y[max(k - 1, 0)] if pcur[2] == y[k] else y[k]
                upper[0] = z[i + 1]
                upper[1] = x[j + 1]
                upper[2] = y[k + 1]

                # Handle precision issues due to fac
                pcur[0] = numpy.round(pcur[0], 8)
                pcur[1] = numpy.round(pcur[1], 8)
                pcur[2] = numpy.round(pcur[2], 8)

                if (pcur != ray[-1]).any():
                    ray.append(pcur.copy())

                else:
                    ray[-1] = pcur.copy()

                if i == isrc and j == jsrc and k == ksrc:
                    break

        else:
            pcur -= delta
            ray.append(pcur.copy())

    ray.append(numpy.array([zsrc, xsrc, ysrc], dtype=numpy.float64))
    out = numpy.empty((len(ray), 3), dtype=numpy.float64)
    for i, r in enumerate(ray):
        out[-i - 1] = r

    return out


@jitted(parallel=True)
def _ray3d_vectorized(
    z,
    x,
    y,
    zgrad,
    xgrad,
    ygrad,
    zend,
    xend,
    yend,
    zsrc,
    xsrc,
    ysrc,
    stepsize,
    honor_grid=False,
):
    """Perform ray-tracing in parallel for different points."""
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
                honor_grid,
            )
        )

    return out


@jitted
def ray3d(z, x, y, zgrad, xgrad, ygrad, p, src, stepsize, honor_grid=False):
    """Perform ray-tracing."""
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
            honor_grid,
        )

    else:
        rays = _ray3d_vectorized(
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
            honor_grid,
        )

        # Hack: append does not work in parallel, sort back list
        end_points = [ray[-1] for ray in rays]
        idx = [first_index(x, end_points) for x in p]

        return [rays[i] for i in idx]
