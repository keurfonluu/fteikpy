import numpy
from numba import prange

from .._common import dist2d, jitted, norm2d
from .._interp import interp2d
from ._common import first_index, shrink


@jitted("f8[:, :](f8[:], f8[:], f8[:, :], f8[:, :], f8, f8, f8, f8, f8, b1)")
def _ray2d(z, x, zgrad, xgrad, zend, xend, zsrc, xsrc, stepsize, honor_grid):
    """Perform a posteriori 2D ray-tracing."""
    condz = z[0] <= zend <= z[-1]
    condx = x[0] <= xend <= x[-1]
    if not (condz and condx):
        raise ValueError("end point out of bound")

    if honor_grid:
        nz, nx = len(z), len(x)

        i = numpy.searchsorted(z, zend, side="right") - 1
        j = numpy.searchsorted(x, xend, side="right") - 1
        zmin = z[max(i - 1, 0)] if zend == z[i] else z[i]
        xmin = x[max(j - 1, 0)] if xend == x[j] else x[j]
        lower = numpy.array([zmin, xmin])
        upper = numpy.array([z[min(i + 1, nz - 1)], x[min(j + 1, nx - 1)]])

    pcur = numpy.array([zend, xend], dtype=numpy.float64)
    delta = numpy.empty(2, dtype=numpy.float64)
    ray = [pcur.copy()]
    while dist2d(zsrc, xsrc, pcur[0], pcur[1]) >= stepsize:
        gz = interp2d(z, x, zgrad, pcur)
        gx = interp2d(z, x, xgrad, pcur)
        gn = norm2d(gz, gx)

        if gn > 0.0:
            gni = 1.0 / gn
        else:
            break

        delta[0] = stepsize * gz * gni
        delta[1] = stepsize * gx * gni

        if honor_grid:
            fac = shrink(pcur, delta, lower, upper)
            pcur -= fac * delta

            if fac < 1.0:
                i = numpy.searchsorted(z, pcur[0], side="right") - 1
                j = numpy.searchsorted(x, pcur[1], side="right") - 1
                lower[0] = z[max(i - 1, 0)] if pcur[0] == z[i] else z[i]
                lower[1] = x[max(j - 1, 0)] if pcur[1] == x[j] else x[j]
                upper[0] = z[i + 1]
                upper[1] = x[j + 1]

                # Handle precision issues due to fac
                for i in range(2):
                    pcur[i] = numpy.round(pcur[i], 8)

                if (pcur != ray[-1]).any():
                    ray.append(pcur.copy())

                else:
                    ray[-1] = pcur.copy()

        else:
            pcur -= delta
            ray.append(pcur.copy())

    ray.append(numpy.array([zsrc, xsrc], dtype=numpy.float64))
    out = numpy.empty((len(ray), 2), dtype=numpy.float64)
    for i, r in enumerate(ray):
        out[-i - 1] = r

    return out


@jitted(parallel=True)
def _ray2d_vectorized(
    z, x, zgrad, xgrad, zend, xend, zsrc, xsrc, stepsize, honor_grid=False
):
    """Perform ray-tracing in parallel for different points."""
    out = []
    for i in prange(len(zend)):
        out.append(
            _ray2d(
                z, x, zgrad, xgrad, zend[i], xend[i], zsrc, xsrc, stepsize, honor_grid
            )
        )

    return out


@jitted
def ray2d(z, x, zgrad, xgrad, p, src, stepsize, honor_grid=False):
    """Perform ray-tracing."""
    if p.ndim == 1:
        return _ray2d(
            z, x, zgrad, xgrad, p[0], p[1], src[0], src[1], stepsize, honor_grid
        )

    else:
        rays = _ray2d_vectorized(
            z, x, zgrad, xgrad, p[:, 0], p[:, 1], src[0], src[1], stepsize, honor_grid,
        )

        # Hack: append does not work in parallel, sort back list
        end_points = [ray[-1] for ray in rays]
        idx = [first_index(x, end_points) for x in p]

        return [rays[i] for i in idx]
