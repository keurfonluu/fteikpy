import numpy

from numba import prange

from .._common import jitted


Big = 1.0e5
eps = 1.0e-15
epsin = 5


@jitted("f8(i4, i4, f8, f8, f8, f8, f8)")
def t_ana(i, j, dz, dx, zsa, xsa, vzero):
    """Calculate analytical times in homogenous model."""
    return vzero * ((dz * (i - zsa)) ** 2.0 + (dx * (j - xsa)) ** 2.0) ** 0.5


@jitted("UniTuple(f8, 3)(i4, i4, f8, f8, f8, f8, f8)")
def t_anad(i, j, dz, dx, zsa, xsa, vzero):
    """Calculate analytical times in homogenous model and derivatives of times."""
    t = t_ana(i, j, dz, dx, zsa, xsa, vzero)

    if t > 0.0:
        tmp = vzero ** 2.0 / t
        tzc = (i - zsa) * dz * tmp
        txc = (j - xsa) * dx * tmp
    else:
        tzc = 0.0
        txc = 0.0

    return t, tzc, txc


@jitted("f8(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, i4, i4)")
def delta(t1, tauv, taue, tauev, t0c, tzc, txc, dzi, dxi, dz2i, dx2i, vzero, vref, sgntz, sgntx):
    """Solve quadratic equation."""
    ta = tauev + taue - tauv
    tb = tauev - taue + tauv

    apoly = dz2i + dx2i
    bpoly = 4.0 * (sgntx * txc * dxi + sgntz * tzc * dzi) - 2.0 * (ta * dx2i + tb * dz2i)
    cpoly = (ta * ta * dx2i) + (tb * tb * dz2i) - 4.0 * (sgntx * txc * dxi * ta + sgntz * tzc * dzi * tb) + 4.0 * (vzero * vzero - vref * vref)
    dpoly = bpoly * bpoly - 4.0 * apoly * cpoly

    return 0.5 * (dpoly ** 0.5 - bpoly) / apoly + t0c if dpoly >= 0.0 else t1


@jitted("void(f8[:, :], f8[:, :, :], f8[:, :], UniTuple(f8, 6), f8, f8, f8, f8, f8, i4, i4, i4, i4, i4, i4, i4, i4, b1)")
def sweep(tt, ttgrad, slow, dargs, zsi, xsi, zsa, xsa, vzero, i, j, sgnvz, sgnvx, sgntz, sgntx, nz, nx, grad):
    """Sweep in given direction."""
    dz, dx, dzi, dxi, dz2i, dx2i = dargs
    i1 = i - sgnvz
    j1 = j - sgnvx

    # Get local times of surrounding points
    tv = tt[i - sgntz, j]
    te = tt[i, j - sgntx]
    tev = tt[i - sgntz, j - sgntx]

    # 1D operators (refracted times)
    # First dimension (Z axis)
    vref = min(slow[i1, max(j - 1, 1)], slow[i1, min(j, nx - 1)])
    t1d1 = tv + dz * vref

    # Second dimension (X axis)
    vref = min(slow[max(i - 1, 1), j1], slow[min(i, nz - 1), j1])
    t1d2 = te + dx * vref

    t1d = min(t1d1, t1d2)

    # 2D operators
    t2d = Big
    vref = slow[i1, j1]

    # Choose plane wave or spherical
    # Test for plane wave
    if numpy.abs(i - zsi) > epsin or numpy.abs(j - xsi) > epsin:
        # 4 points operator if possible, otherwise do three points
        if tv <= te + dx * vref and te <= tv + dz * vref and te >= tev and tv >= tev:
            ta = tev + te - tv
            tb = tev - te + tv
            t2d = ((tb * dz2i + ta * dx2i) + (4.0 * vref * vref * (dz2i + dx2i) - dz2i * dx2i * (ta - tb) * (ta - tb)) ** 0.5) / (dz2i + dx2i)

        # Two 3 points operators
        elif te - tev <= dz * dz * vref / (dx * dx + dz * dz) ** 0.5 and te - tev > 0.0:
            t2d = te + dx * (vref * vref - ((te - tev) / dz) ** 2.0) ** 0.5

        elif tv - tev <= dx * dx * vref / (dx * dx + dz * dz) ** 0.5 and tv - tev > 0.0:
            t2d = tv + dz * (vref * vref - ((tv - tev) / dx) ** 2.0) ** 0.5 

    # Test for spherical
    else:
        # Do spherical operator if conditions ok
        if tv < te + dx * vref and te < tv + dz * vref and te >= tev and tv >= tev:
            t0c, tzc, txc = t_anad(i, j, dz, dx, zsa, xsa, vzero)
            tauv = tv - t_ana(i - sgntz, j, dz, dx, zsa, xsa, vzero)
            taue = te - t_ana(i, j - sgntx, dz, dx, zsa, xsa, vzero)
            tauev = tev - t_ana(i - sgntz, j - sgntx, dz, dx, zsa, xsa, vzero)

            t2d = delta(t2d, tauv, taue, tauev, t0c, tzc, txc, dzi, dxi, dz2i, dx2i, vzero, vref, sgntz, sgntx)
            if t2d < tv or t2d < te:
                t2d = Big

    # Select minimum time
    t0 = tt[i, j]
    tt[i, j] = min(t0, t1d, t2d)

    # Compute gradient according to minimum time direction
    if grad and tt[i, j] != t0:
        if tt[i, j] == t1d1:
            ttgrad[i, j, 0] = sgntz * (tt[i, j] - tv) / dz
            ttgrad[i, j, 1] = 0.0
        elif tt[i, j] == t1d2:
            ttgrad[i, j, 0] = 0.0
            ttgrad[i, j, 1] = sgntx * (tt[i, j] - te) / dx
        else:
            ttgrad[i, j, 0] = sgntz * (tt[i, j] - tv) / dz
            ttgrad[i, j, 1] = sgntx * (tt[i, j] - te) / dx


@jitted("void(f8[:, :], f8[:, :, :], f8[:, :], f8, f8, f8, f8, f8, f8, f8, i4, i4, b1)")
def sweep2d(tt, ttgrad, slow, dz, dx, zsi, xsi, zsa, xsa, vzero, nz, nx, grad):
    """Perform one sweeping."""
    dzi = 1.0 / dz
    dxi = 1.0 / dx
    dz2i = dzi / dz
    dx2i = dxi / dx
    dargs = (dz, dx, dzi, dxi, dz2i, dx2i)
    
    for j in range(1, nx):
        for i in range(1, nz):
            sweep(tt, ttgrad, slow, dargs, zsi, xsi, zsa, xsa, vzero, i, j, 1, 1, 1, 1, nz, nx, grad)

        for i in range(nz - 2, -1, -1):
            sweep(tt, ttgrad, slow, dargs, zsi, xsi, zsa, xsa, vzero, i, j, 0, 1, -1, 1, nz, nx, grad)

    for j in range(nx - 2, -1, -1):
        for i in range(1, nz):
            sweep(tt, ttgrad, slow, dargs, zsi, xsi, zsa, xsa, vzero, i, j, 1, 0, 1, -1, nz, nx, grad)

        for i in range(nz - 2, -1, -1):
            sweep(tt, ttgrad, slow, dargs, zsi, xsi, zsa, xsa, vzero, i, j, 0, 0, -1, -1, nz, nx, grad)


@jitted("Tuple((f8[:, :], f8[:, :, :], f8))(f8[:, :], f8, f8, f8, f8, i4, b1)")
def fteik2d(slow, dz, dx, zsrc, xsrc, max_sweep=2, grad=False):
    """Calculate traveltimes given a 2D velocity model."""
    # Parameters
    nz, nx = numpy.shape(slow)

    # Convert src to grid position and try and take into account machine precision
    zsa = zsrc / dz
    xsa = xsrc / dx

    # Try to handle edges simply for source due to precision
    zsa = zsa - eps if zsa > nz else zsa
    xsa = xsa - eps if xsa > nx else xsa

    # Grid points to initialize source
    zsi = int(zsa)
    xsi = int(xsa)
    vzero = slow[zsi, xsi]

    # Allocate work array
    tt = numpy.full((nz, nx), Big, dtype=numpy.float64)
    ttgrad = (
        numpy.zeros((nz, nx, 2), dtype=numpy.float64)
        if grad
        else numpy.empty((0, 0, 0), dtype=numpy.float64)
    )

    # Do our best to initialize source
    dzu = numpy.abs(zsa - float(zsi))
    dzd = numpy.abs(float(zsi) - zsa + 1.0)
    dxw = numpy.abs(xsa - float(xsi))
    dxe = numpy.abs(float(xsi) - xsa + 1.0)

    # Source seems close enough to a grid point in X and Y direction
    dzv_min = min(dzu, dzd)
    dzh_min = min(dxw, dxe)
    if dzv_min < eps and dzh_min < eps:
        zsa = numpy.round(zsa)
        xsa = numpy.round(xsa)
        iflag = 1

    # At least one of coordinates not close to any grid point in X and Y direction
    elif dzv_min > eps or dzh_min > eps:
        zsa = numpy.round(zsa) if dzv_min < eps else zsa
        xsa = numpy.round(xsa) if dzh_min < eps else xsa
        iflag = 2

    # Oops we are lost, not sure this happens - fix src to nearest grid point
    else:
        zsa = numpy.round(zsa)
        xsa = numpy.round(xsa)
        iflag = 3

    # We know where src is - start first propagation
    if iflag == 2:
        td = numpy.full(max(nz, nx), Big, dtype=numpy.float64)

        dzu = numpy.abs(zsa - float(zsi))
        dzd = numpy.abs(float(zsi) - zsa + 1.0)
        dxw = numpy.abs(xsa - float(xsi))
        dxe = numpy.abs(float(xsi) - xsa + 1.0)

        # First initialize 4 points around source
        tt[zsi, xsi] = t_ana(zsi, xsi, dz, dx, zsa, xsa, vzero)
        tt[zsi + 1, xsi] = t_ana(zsi + 1, xsi, dz, dx, zsa, xsa, vzero)
        tt[zsi, xsi + 1] = t_ana(zsi, xsi + 1, dz, dx, zsa, xsa, vzero)
        tt[zsi + 1, xsi + 1] = t_ana(zsi + 1, xsi + 1, dz, dx, zsa, xsa, vzero)

        dxi = 1.0 / dx
        dx2i = dxi / dx
        td[xsi + 1] = vzero * dxe * dx
        for j in range(xsi + 2, nx):
            vref = slow[zsi, j - 1]
            td[j] = td[j - 1] + dx * vref
            tauv = td[j] - vzero * numpy.abs(j - xsa) * dx
            tauev = td[j - 1] - vzero * numpy.abs(j - xsa - 1.0) * dx

            dzi = 1.0 / dzd
            dz2i = dz / dzd / dzd
            taue = tt[zsi + 1, j - 1] - t_ana(zsi + 1, j - 1, dz, dx, zsa, xsa, vzero)
            t0c, tzc, txc = t_anad(zsi + 1, j, dz, dx, zsa, xsa, vzero)
            tt[zsi + 1, j] = delta(tt[zsi + 1, j], tauv, taue, tauev, t0c, tzc, txc, 1.0 / dzd, dxi, dz / dzd / dzd, dx2i, vzero, vref, 1, 1)

            dzi = 1.0 / dzu
            dz2i = dz / dzu / dzu
            taue = tt[zsi, j - 1] - t_ana(zsi, j - 1, dz, dx, zsa, xsa, vzero)
            t0c, tzc, txc = t_anad(zsi, j, dz, dx, zsa, xsa, vzero)
            tt[zsi, j] = delta(tt[zsi, j], tauv, taue, tauev, t0c, tzc, txc, dzi, dxi, dz2i, dx2i, vzero, vref, -1, 1)

        td[xsi] = vzero * dxw * dx
        for j in range(xsi - 2, -1, -1):
            vref = slow[zsi, j]
            td[j] = td[j + 1] + dx * vref
            tauv = td[j] - vzero * numpy.abs(j - xsa) * dx
            tauev = td[j + 1] - vzero * numpy.abs(j - xsa + 1.0) * dx

            dzi = 1.0 / dzd
            dz2i = dz / dzd / dzd
            taue = tt[zsi + 1, j + 1] - t_ana(zsi + 1, j + 1, dz, dx, zsa, xsa, vzero)
            t0c, tzc, txc = t_anad(zsi + 1, j, dz, dx, zsa, xsa, vzero)
            tt[zsi + 1, j] = delta(tt[zsi + 1, j], tauv, taue, tauev, t0c, tzc, txc, dzi, dxi, dz2i, dx2i, vzero, vref, 1, -1)

            dzi = 1.0 / dzu
            dz2i = dz / dzu / dzu
            taue = tt[zsi + 1, j + 1] - t_ana(zsi + 1, j + 1, dz, dx, zsa, xsa, vzero)
            t0c, tzc, txc = t_anad(zsi, j, dz, dx, zsa, xsa, vzero)
            tt[zsi, j] = delta(tt[zsi, j], tauv, taue, tauev, t0c, tzc, txc, dzi, dxi, dz2i, dx2i, vzero, vref, -1, -1)

        dzi = 1.0 / dz
        dz2i = dzi / dz
        td[:] = Big
        td[zsi + 1] = vzero * dzd * dz
        for i in range(zsi + 2, nz):
            vref = slow[i - 1, xsi]
            td[i] = td[i - 1] + dz * vref
            taue = td[i] - vzero * numpy.abs(i - zsa) * dz
            tauev = td[i - 1] - vzero * numpy.abs(i - zsa - 1.0) * dz

            dxi = 1.0 / dxe
            dx2i = dx / dxe / dxe
            tauv = tt[i - 1, xsi + 1] - t_ana(i - 1, xsi + 1, dz, dx, zsa, xsa, vzero)
            t0c, tzc, txc = t_anad(i, xsi + 1, dz, dx, zsa, xsa, vzero)
            tt[i, xsi + 1] = delta(tt[i, xsi + 1], tauv, taue, tauev, t0c, tzc, txc, dzi, dxi, dz2i, dx2i, vzero, vref, 1, 1)

            dxi = 1.0 / dxw
            dx2i = dx / dxw / dxw
            tauv = tt[i - 1, xsi] - t_ana(i - 1, xsi, dz, dx, zsa, xsa, vzero)
            t0c, txc, tzc = t_anad(i, xsi, dz, dx, zsa, xsa, vzero)
            tt[i, xsi] = delta(tt[i, xsi], tauv, taue, tauev, t0c, tzc, txc, dzi, dxi, dz2i, dx2i, vzero, vref, 1, -1)

        td[zsi] = vzero * dzu * dz
        for i in range(zsi - 2, -1, -1):
            vref = slow[i, xsi]
            td[i] = td[i + 1] + dz * vref
            taue = td[i] - vzero * numpy.abs(i - zsa) * dz
            tauev = td[i + 1] - vzero * numpy.abs(i - zsa + 1.0) * dz

            dxi = 1.0 / dxe
            dx2i = dx / dxe / dxe
            tauv = tt[i + 1, xsi + 1] - t_ana(i + 1, xsi + 1, dz, dx, zsa, xsa, vzero)
            t0c, tzc, txc = t_anad(i, xsi + 1, dz, dx, zsa, xsa, vzero)
            tt[i, xsi + 1] = delta(tt[i, xsi + 1], tauv, taue, tauev, t0c, tzc, txc, dzi, dxi, dz2i, dx2i, vzero, vref, -1, 1)

            dxi = 1.0 / dxw
            dx2i = dx / dxw / dxw
            tauv = tt[i + 1, xsi] - t_ana(i + 1, xsi, dz, dx, zsa, xsa, vzero)
            t0c, tzc, txc = t_anad(i, xsi, dz, dx, zsa, xsa, vzero)
            tt[i, xsi] = delta(tt[i, xsi], tauv, taue, tauev, t0c, tzc, txc, dzi, dxi, dz2i, dx2i, vzero, vref, -1, -1)

    else:
        tt[int(zsa), int(xsa)] = 0.0

    for _ in range(max_sweep):
        sweep2d(tt, ttgrad, slow, dz, dx, zsi, xsi, zsa, xsa, vzero, nz, nx, grad)

    return tt, ttgrad, vzero


@jitted(parallel=True)
def solve2d(slow, dz, dx, src, max_sweep=2, grad=False):
    if src.ndim == 1:
        return fteik2d(slow, dz, dx, src[0], src[1], max_sweep, grad)

    elif src.ndim == 2:
        nsrc = len(src)
        nz, nx = slow.shape
        tt = numpy.empty((nsrc, nz, nx), dtype=numpy.float64)
        ttgrad = (
            numpy.empty((nsrc, nz, nx, 2), dtype=numpy.float64)
            if grad
            else numpy.empty((nsrc, 0, 0, 0), dtype=numpy.float64)
        )
        vzero = numpy.empty(nsrc, dtype=numpy.float64)
        for i in prange(nsrc):
            tt[i], ttgrad[i], vzero[i] = fteik2d(slow, dz, dx, src[i, 0], src[i, 1], max_sweep, grad)

        return tt, ttgrad, vzero

    else:
        raise ValueError()
