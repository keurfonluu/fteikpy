import numpy

from .._common import jitted


@jitted("f8(f8, f8, f8, f8)")
def dist2d(x1, y1, x2, y2):
    return ((x1 - x2) ** 2.0 + (y1 - y2) ** 2.0) ** 0.5


@jitted("f8(f8[:], f8[:], f8[:, :], f8, f8, f8, f8, f8)")
def interp2d(x, y, v, xq, yq, xsrc, ysrc, vzero):
    xsi = numpy.nonzero(x <= xsrc)[0][-1]
    ysi = numpy.nonzero(y <= ysrc)[0][-1]
    i1 = numpy.nonzero(x <= xq)[0][-1]
    j1 = numpy.nonzero(y <= yq)[0][-1]

    if xsi == i1 and ysi == j1:
        vq = vzero * dist2d(xsrc, ysrc, xq, yq)
    else:
        nx, ny = numpy.shape(v)
        i2 = i1 + 1
        j2 = j1 + 1

        if i1 == nx - 1 and j1 != ny - 1:
            x1 = x[i1]
            x2 = 2.0 * x1 - x[-2]
            y1 = y[j1]
            y2 = y[j2]

            d11 = dist2d(xsrc, ysrc, x1, y1)
            d21 = 0.0
            d22 = dist2d(xsrc, ysrc, x1, y2)
            d22 = 0.0

            v11 = v[i1, j1]
            v21 = 1.0
            v12 = v[i1, j2]
            v22 = 1.0

        elif i1 != nx - 1 and j1 == ny - 1:
            x1 = x[i1]
            x2 = x[i1]
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

        elif i1 == nx - 1 and j1 == ny - 1:
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

        ax = numpy.array([x2, x1, x2, x1])
        ay = numpy.array([y2, y2, y1, y1])
        av = numpy.array([v11, v21, v12, v22])
        ad = numpy.array([d11, d21, d12, d22])
        N = numpy.abs((ax - xq) * (ay - yq)) / numpy.abs((x2 - x1) * (y2 - y1))
        vq = dist2d(xsrc, ysrc, xq, yq) / numpy.dot(ad / av, N)

    return vq
