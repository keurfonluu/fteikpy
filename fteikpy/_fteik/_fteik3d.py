import numpy

from numba import prange

from .._common import jitted


Big = 1.0e5
eps = 1.0e-15


@jitted("f8(i4, i4, i4, f8, f8, f8, f8, f8, f8, f8)")
def t_ana(i, j, k, dz, dx, dy, zsa, xsa, ysa, vzero):
    """Calculate analytical times in homogenous model."""
    return vzero * ((dz * (i - zsa)) ** 2.0 + (dx * (j - xsa)) ** 2.0 + (dy * (k - ysa)) ** 2.0) ** 0.5


@jitted("void(f8[:, :, :], f8[:, :, :, :], f8[:, :, :], UniTuple(f8, 10), f8, f8, f8, f8, f8, f8, f8, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, b1)")
def sweep(tt, ttgrad, slow, dargs, zsi, xsi, ysi, zsa, xsa, ysa, vzero, i, j, k, sgnvz, sgnvx, sgnvy, sgntz, sgntx, sgnty, nz, nx, ny, grad):
    """Sweep in given direction."""
    dz, dx, dy, dz2i, dx2i, dy2i, dz2dx2, dz2dy2, dx2dy2, dsum = dargs

    i1 = i - sgnvz
    j1 = j - sgnvx
    k1 = k - sgnvy

    # Get local times of surrounding points
    tv = tt[i - sgntz, j, k]
    te = tt[i, j - sgntx, k]
    tn = tt[i, j, k - sgnty]
    tev = tt[i - sgntz, j - sgntx, k]
    ten = tt[i, j - sgntx, k - sgnty]
    tnv = tt[i - sgntz, j, k - sgnty]
    tnve = tt[i - sgntz, j - sgntx, k - sgnty]

    # 1D operators (refracted times)
    # First dimension (Z axis)
    vref = min(
        slow[i1, max(j - 1, 1), max(k - 1, 1)],
        slow[i1, max(j - 1, 1), min(k, ny - 1)],
        slow[i1, min(j, nx - 1), max(k - 1, 1)],
        slow[i1, min(j, nx - 1), min(k, ny - 1)],    
    )
    t1d1 = tv + dz * vref

    # Second dimension (X axis)
    vref = min(
        slow[max(i - 1, 1), j1, max(k - 1, 1)],
        slow[min(i, nz - 1), j1, max(k - 1, 1)],
        slow[max(i - 1, 1), j1, min(k, ny - 1)],
        slow[min(i, nz - 1), j1, min(k, ny - 1)],
    )
    t1d2 = te + dx * vref

    # Third dimension (Y axis)
    vref = min(
        slow[max(i - 1, 1), max(j - 1, 1), k1],
        slow[max(i - 1, 1), min(j, nx - 1), k1],
        slow[min(i, nz - 1), max(j - 1, 1), k1],
        slow[min(i, nz - 1), min(j, nx - 1), k1],
    )
    t1d3 = tn + dy * vref

    t1d = min(t1d1, t1d2, t1d3)

    # 2D operators
    # ZX plane
    t2d1 = Big
    vref = min(slow[i1, j1, max(k - 1, 1)], slow[i1, j1, min(k, ny - 1)])
    if tv < te + dx * vref and te < tv + dz * vref:
        ta = tev + te - tv
        tb = tev - te + tv
        t2d1 = ((tb * dz2i + ta * dx2i) + (4.0 * vref * vref * (dz2i + dx2i) - dz2i * dx2i * (ta - tb) * (ta - tb)) ** 0.5) / (dz2i + dx2i)

    # ZY plane
    t2d2 = Big
    vref = min(slow[i1, max(j - 1, 1), k1], slow[i1, min(j, nx - 1), k1])
    if tv < tn + dy * vref and tn < tv + dz * vref:
        ta = tv - tn + tnv
        tb = tn - tv + tnv
        t2d2 = ((ta * dz2i + tb * dy2i) + (4.0 * vref * vref * (dz2i + dy2i) - dz2i * dy2i * (ta - tb) * (ta - tb)) ** 0.5) / (dz2i + dy2i)

    # XY plane
    t2d3 = Big
    vref = min(slow[max(i - 1, 1), j1, k1], slow[min(i, nz - 1), j1, k1])
    if te < tn + dy * vref and tn < te + dx * vref:
        ta = te - tn + ten
        tb = tn - te + ten
        t2d3 = ((ta * dx2i + tb * dy2i) + (4.0 * vref * vref * (dx2i + dy2i) - dx2i * dy2i * (ta - tb) * (ta - tb)) ** 0.5) / (dx2i + dy2i)

    t2d = min(t2d1, t2d2, t2d3)

    # 3D operator
    t3d = Big
    vref = slow[i1, j1, k1]
    ta = te - 0.5 * tn + 0.5 * ten - 0.5 * tv + 0.5 * tev - tnv + tnve
    tb = tv - 0.5 * tn + 0.5 * tnv - 0.5 * te + 0.5 * tev - ten + tnve
    tc = tn - 0.5 * te + 0.5 * ten - 0.5 * tv + 0.5 * tnv - tev + tnve
    if min(t1d, t2d) > max(tv, te, tn):
        t2 = vref * vref * dsum * 9.0
        t3 = dz2dx2 * (ta - tb) * (ta - tb)
        t3 += dz2dy2 * (tb - tc) * (tb - tc)
        t3 += dx2dy2 * (ta - tc) * (ta - tc)
        if t2 >= t3:
            t1 = tb * dz2i + ta * dx2i + tc * dy2i
            t3d = (t1 + (t2 - t3) ** 0.5) / dsum

    # Select minimum time
    t0 = tt[i, j, k]
    tt[i, j, k] = min(t0, t1d, t2d, t3d)

    # Compute gradient according to minimum time direction
    if grad and tt[i, j, k] != t0:
        if tt[i, j, k] == t1d1:
            ttgrad[i, j, k, 0] = sgntz * (tt[i, j, k] - tv) / dz
            ttgrad[i, j, k, 1] = 0.0
            ttgrad[i, j, k, 2] = 0.0

        elif tt[i, j, k] == t1d2:
            ttgrad[i, j, k, 0] = 0.0
            ttgrad[i, j, k, 1] = sgntx * (tt[i, j, k] - te) / dx
            ttgrad[i, j, k, 2] = 0.0

        elif tt[i, j, k] == t1d3:
            ttgrad[i, j, k, 0] = 0.0
            ttgrad[i, j, k, 1] = 0.0
            ttgrad[i, j, k, 2] = sgnty * (tt[i, j, k] - tn) / dy

        elif tt[i, j, k] == t2d1:
            ttgrad[i, j, k, 0] = sgntz * (tt[i, j, k] - tv) / dz
            ttgrad[i, j, k, 1] = sgntx * (tt[i, j, k] - te) / dx
            ttgrad[i, j, k, 2] = 0.0

        elif tt[i, j, k] == t2d2:
            ttgrad[i, j, k, 0] = sgntz * (tt[i, j, k] - tv) / dz
            ttgrad[i, j, k, 1] = 0.0
            ttgrad[i, j, k, 2] = sgnty * (tt[i, j, k] - tn) / dy

        elif tt[i, j, k] == t2d3:
            ttgrad[i, j, k, 0] = 0.0
            ttgrad[i, j, k, 1] = sgntx * (tt[i, j, k] - te) / dx
            ttgrad[i, j, k, 2] = sgnty * (tt[i, j, k] - tn) / dy

        else:
            ttgrad[i, j, k, 0] = sgntz * (tt[i, j, k] - tv) / dz
            ttgrad[i, j, k, 1] = sgntx * (tt[i, j, k] - te) / dx
            ttgrad[i, j, k, 2] = sgnty * (tt[i, j, k] - tn) / dy


@jitted("void(f8[:, :, :], f8[:, :, :, :], f8[:, :, :], f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, i4, i4, i4, b1)")
def sweep3d(tt, ttgrad, slow, dz, dx, dy, zsi, xsi, ysi, zsa, xsa, ysa, vzero, nz, nx, ny, grad):
    """Perform one sweeping."""
    dz2i = 1.0 / dz / dz
    dx2i = 1.0 / dx / dx
    dy2i = 1.0 / dy / dy
    dz2dx2 = dz2i * dx2i
    dz2dy2 = dz2i * dy2i
    dx2dy2 = dx2i * dy2i
    dsum = dz2i + dx2i + dy2i
    dargs = (dz, dx, dy, dz2i, dx2i, dy2i, dz2dx2, dz2dy2, dx2dy2, dsum)

    # First sweeping: Top -> Bottom; West -> East; South -> North
    for k in range(1, ny):
        for j in range(1, nx):
            for i in range(1, nz):
                sweep(tt, ttgrad, slow, dargs, zsi, xsi, ysi, zsa, xsa, ysa, vzero, i, j, k, 1, 1, 1, 1, 1, 1, nz, nx, ny, grad)

    # Second sweeping: Top -> Bottom; East -> West; South -> North
    for k in range(1, ny):
        for j in range(nx - 2, -1, -1):
            for i in range(1, nz):
                sweep(tt, ttgrad, slow, dargs, zsi, xsi, ysi, zsa, xsa, ysa, vzero, i, j, k, 1, 0, 1, 1, -1, 1, nz, nx, ny, grad)

    # Third sweeping: Top -> Bottom; West -> East; North -> South
    for k in range(ny - 2, -1, -1):
        for j in range(1, nx):
            for i in range(1, nz):
                sweep(tt, ttgrad, slow, dargs, zsi, xsi, ysi, zsa, xsa, ysa, vzero, i, j, k, 1, 1, 0, 1, 1, -1, nz, nx, ny, grad)

    # Fouth sweeping: Top -> Bottom; East -> West; North -> South
    for k in range(ny - 2, -1, -1):
        for j in range(nx - 2, -1, -1):
            for i in range(1, nz):
                sweep(tt, ttgrad, slow, dargs, zsi, xsi, ysi, zsa, xsa, ysa, vzero, i, j, k, 1, 0, 0, 1, -1, -1, nz, nx, ny, grad)

    # Fifth sweeping: Bottom -> Top; West -> East; South -> North
    for k in range(1, ny):
        for j in range(1, nx):
            for i in range(nz - 2, -1, -1):
                sweep(tt, ttgrad, slow, dargs, zsi, xsi, ysi, zsa, xsa, ysa, vzero, i, j, k, 0, 1, 1, -1, 1, 1, nz, nx, ny, grad)

    # Sixth sweeping: Bottom -> Top; East -> West; South -> North
    for k in range(1, ny):
        for j in range(nx - 2, -1, -1):
            for i in range(nz - 2, -1, -1):
                sweep(tt, ttgrad, slow, dargs, zsi, xsi, ysi, zsa, xsa, ysa, vzero, i, j, k, 0, 0, 1, -1, -1, 1, nz, nx, ny, grad)

    # Seventh sweeping: Bottom -> Top; West -> East; North -> South
    for k in range(ny - 2, -1, -1):
        for j in range(1, nx):
            for i in range(nz - 2, -1, -1):
                sweep(tt, ttgrad, slow, dargs, zsi, xsi, ysi, zsa, xsa, ysa, vzero, i, j, k, 0, 1, 0, -1, 1, -1, nz, nx, ny, grad)

    # Eighth sweeping: Bottom -> Top; East -> West; North -> South
    for k in range(ny - 2, -1, -1):
        for j in range(nx - 2, -1, -1):
            for i in range(nz - 2, -1, -1):
                sweep(tt, ttgrad, slow, dargs, zsi, xsi, ysi, zsa, xsa, ysa, vzero, i, j, k, 0, 0, 0, -1, -1, -1, nz, nx, ny, grad)


@jitted("Tuple((f8[:, :, :], f8[:, :, :, :], f8))(f8[:, :, :], f8, f8, f8, f8, f8, f8, i4, b1)")
def fteik3d(slow, dz, dx, dy, zsrc, xsrc, ysrc, max_sweep=2, grad=False):
    """Calculate traveltimes given a 3D velocity model."""
    # Parameters
    nz, nx, ny = numpy.shape(slow)

    # Convert src to grid position and try and take into account machine precision
    zsa = zsrc / dz
    xsa = xsrc / dx
    ysa = ysrc / dy

    # Try to handle edges simply for source due to precision
    zsa = zsa - eps if zsa > nz else zsa
    xsa = xsa - eps if xsa > nx else xsa
    ysa = ysa - eps if ysa > ny else ysa

    # Grid points to initialize source
    zsi = int(zsa)
    xsi = int(xsa)
    ysi = int(ysa)
    vzero = slow[zsi, xsi, ysi]

    # Allocate work array
    tt = numpy.full((nz, nx, ny), Big, dtype=numpy.float64)
    ttgrad = (
        numpy.zeros((nz, nx, ny, 3), dtype=numpy.float64)
        if grad
        else numpy.empty((0, 0, 0, 0), dtype=numpy.float64)
    )

    # Initialize points around source
    tt[zsi, xsi, ysi] = t_ana(zsi, xsi, ysi, dz, dx, dy, zsa, xsa, ysa, vzero)
    tt[zsi + 1, xsi, ysi] = t_ana(zsi + 1, xsi, ysi, dz, dx, dy, zsa, xsa, ysa, vzero)
    tt[zsi, xsi + 1, ysi] = t_ana(zsi, xsi + 1, ysi, dz, dx, dy, zsa, xsa, ysa, vzero)
    tt[zsi, xsi, ysi + 1] = t_ana(zsi, xsi, ysi + 1, dz, dx, dy, zsa, xsa, ysa, vzero)
    tt[zsi + 1, xsi + 1, ysi] = t_ana(zsi + 1, xsi + 1, ysi, dz, dx, dy, zsa, xsa, ysa, vzero)
    tt[zsi + 1, xsi, ysi + 1] = t_ana(zsi + 1, xsi, ysi + 1, dz, dx, dy, zsa, xsa, ysa, vzero)
    tt[zsi, xsi + 1, ysi + 1] = t_ana(zsi, xsi + 1, ysi + 1, dz, dx, dy, zsa, xsa, ysa, vzero)
    tt[zsi + 1, xsi + 1, ysi + 1] = t_ana(zsi + 1, xsi + 1, ysi + 1, dz, dx, dy, zsa, xsa, ysa, vzero)

    for _ in range(max_sweep):
        sweep3d(tt, ttgrad, slow, dz, dx, dy, zsi, xsi, ysi, zsa, xsa, ysa, vzero, nz, nx, ny, grad)

    return tt, ttgrad, vzero


@jitted(parallel=True)
def solve3d(slow, dz, dx, dy, src, max_sweep=2, grad=False):
    if src.ndim == 1:
        return fteik3d(slow, dz, dx, dy, src[0], src[1], src[2], max_sweep, grad)

    elif src.ndim == 2:
        nsrc = len(src)
        nz, nx, ny = slow.shape
        tt = numpy.empty((nsrc, nz, nx, ny), dtype=numpy.float64)
        ttgrad = (
            numpy.empty((nsrc, nz, nx, ny, 3), dtype=numpy.float64)
            if grad
            else numpy.empty((nsrc, 0, 0, 0, 0), dtype=numpy.float64)
        )
        vzero = numpy.empty(nsrc, dtype=numpy.float64)
        for i in prange(nsrc):
            tt[i], ttgrad[i], vzero[i] = fteik3d(slow, dz, dx, dy, src[i, 0], src[i, 1], src[i, 2], max_sweep, grad)

        return tt, ttgrad, vzero

    else:
        raise ValueError()
