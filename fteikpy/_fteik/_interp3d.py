import numpy

from .._common import jitted


def dist3d(x1, y1, z1, x2, y2, z2):
    return ((x1 - x2) ** 2.0 + (y1 - y2) ** 2.0 + (z1 - z2) ** 2.0) ** 0.5


def interp3d(x, y, z, v, xq, yq, zq, xsrc, ysrc, zsrc, vzero):
    xsi = numpy.nonzero(x <= xsrc)[0][-1]
    ysi = numpy.nonzero(y <= ysrc)[0][-1]
    zsi = numpy.nonzero(z <= zsrc)[0][-1]
    i1 = numpy.nonzero(x <= xq)[0][-1]
    j1 = numpy.nonzero(y <= yq)[0][-1]
    k1 = numpy.nonzero(z <= zq)[0][-1]

    if xsi == i1 and ysi == j1 and zsi == k1:
        vq = vzero * dist3d(xsrc, ysrc, zsrc, xq, yq, zq)
    else:
        nx, ny, nz = numpy.shape(v)
        nx -= 1
        ny -= 1
        nz -= 1

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

            d111 = dist3d(xsrc, ysrc, zsrc, x1, y1, z1)
            d211 = 0.0
            d121 = dist3d(xsrc, ysrc, zsrc, x1, y2, z1)
            d221 = 0.0
            d112 = dist3d(xsrc, ysrc, zsrc, x1, y1, z2)
            d212 = 0.0
            d122 = dist3d(xsrc, ysrc, zsrc, x1, y2, z2)
            d222 = 0.0

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

            d111 = dist3d(xsrc, ysrc, zsrc, x1, y1, z1)
            d211 = dist3d(xsrc, ysrc, zsrc, x2, y1, z1)
            d121 = 0.0
            d221 = 0.0
            d112 = dist3d(xsrc, ysrc, zsrc, x1, y1, z2)
            d212 = dist3d(xsrc, ysrc, zsrc, x2, y1, z2)
            d122 = 0.0
            d222 = 0.0

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

            d111 = dist3d(xsrc, ysrc, zsrc, x1, y1, z1)
            d211 = dist3d(xsrc, ysrc, zsrc, x2, y1, z1)
            d121 = dist3d(xsrc, ysrc, zsrc, x1, y2, z1)
            d221 = dist3d(xsrc, ysrc, zsrc, x2, y2, z1)
            d112 = 0.0
            d212 = 0.0
            d122 = 0.0
            d222 = 0.0

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

            d111 = dist3d(xsrc, ysrc, zsrc, x1, y1, z1)
            d211 = 0.0
            d121 = 0.0
            d221 = 0.0
            d112 = dist3d(xsrc, ysrc, zsrc, x1, y1, z2)
            d212 = 0.0
            d122 = 0.0
            d222 = 0.0

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

            d111 = dist3d(xsrc, ysrc, zsrc, x1, y1, z1)
            d211 = 0.0
            d121 = dist3d(xsrc, ysrc, zsrc, x1, y2, z1)
            d221 = 0.0
            d112 = 0.0
            d212 = 0.0
            d122 = 0.0
            d222 = 0.0

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

            d111 = dist3d(xsrc, ysrc, zsrc, x1, y1, z1)
            d211 = dist3d(xsrc, ysrc, zsrc, x2, y1, z1)
            d121 = 0.0
            d221 = 0.0
            d112 = 0.0
            d212 = 0.0
            d122 = 0.0
            d222 = 0.0

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

            d111 = dist3d(xsrc, ysrc, zsrc, x1, y1, z1)
            d211 = 0.0
            d121 = 0.0
            d221 = 0.0
            d112 = 0.0
            d212 = 0.0
            d122 = 0.0
            d222 = 0.0

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

            d111 = dist3d(xsrc, ysrc, zsrc, x1, y1, z1)
            d211 = dist3d(xsrc, ysrc, zsrc, x2, y1, z1)
            d121 = dist3d(xsrc, ysrc, zsrc, x1, y2, z1)
            d221 = dist3d(xsrc, ysrc, zsrc, x2, y2, z1)
            d112 = dist3d(xsrc, ysrc, zsrc, x1, y1, z2)
            d212 = dist3d(xsrc, ysrc, zsrc, x2, y1, z2)
            d122 = dist3d(xsrc, ysrc, zsrc, x1, y2, z2)
            d222 = dist3d(xsrc, ysrc, zsrc, x2, y2, z2)

            v111 = v[i1, j1, k1]
            v211 = v[i2, j1, k1]
            v121 = v[i1, j2, k1]
            v221 = v[i2, j2, k1]
            v112 = v[i1, j1, k2]
            v212 = v[i2, j1, k2]
            v122 = v[i1, j2, k2]
            v222 = v[i2, j2, k2]

        ax = numpy.array([x2, x1, x2, x1, x2, x1, x2, x1])
        ay = numpy.array([y2, y2, y1, y1, y2, y2, y1, y1])
        az = numpy.array([z2, z2, z2, z2, z1, z1, z1, z1])
        av = numpy.array([v111, v211, v121, v221, v112, v212, v122, v222])
        ad = numpy.array([d111, d211, d121, d221, d112, d212, d122, d222])
        N = numpy.abs((ax - xq) * (ay - yq) * (az - zq)) / numpy.abs((x2 - x1) * (y2 - y1) * (z2 -z1))
        vq = dist3d(xsrc, ysrc, zsrc, xq, yq, zq) / numpy.dot(ad / av, N)

    return vq
