import numpy
from numba import prange

from .._common import jitted, norm3d

Big = 1.0e5
eps = 1.0e-15


@jitted("f8(i4, i4, i4, f8, f8, f8, f8, f8, f8, f8)")
def t_ana(i, j, k, dz, dx, dy, zsa, xsa, ysa, vzero):
    """Calculate analytical times in homogeneous model."""
    return (
        vzero
        * ((dz * (i - zsa)) ** 2.0 + (dx * (j - xsa)) ** 2.0 + (dy * (k - ysa)) ** 2.0)
        ** 0.5
    )


@jitted("UniTuple(f8, 4)(i4, i4, i4, f8, f8, f8, f8, f8, f8, f8)")
def t_anad(i, j, k, dz, dx, dy, zsa, xsa, ysa, vzero):
    """Calculate analytical times in homogeneous model and derivatives of times."""
    t = t_ana(i, j, k, dz, dx, dy, zsa, xsa, ysa, vzero)

    if t > 0.0:
        tmp = vzero ** 2.0 / t
        tzc = (i - zsa) * dz * tmp
        txc = (j - xsa) * dx * tmp
        tyc = (k - ysa) * dy * tmp
    else:
        tzc = 0.0
        txc = 0.0
        tyc = 0.0

    return t, tzc, txc, tyc


@jitted(
    "void(f8[:, :, :], i4[:, :, :, :], f8[:, :, :], UniTuple(f8, 10), i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, b1)"
)
def sweep(
    tt,
    ttsgn,
    slow,
    dargs,
    i,
    j,
    k,
    sgnvz,
    sgnvx,
    sgnvy,
    sgntz,
    sgntx,
    sgnty,
    nz,
    nx,
    ny,
    grad,
):
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
        slow[i1, max(j - 1, 0), max(k - 1, 0)],
        slow[i1, max(j - 1, 0), min(k, ny - 2)],
        slow[i1, min(j, nx - 2), max(k - 1, 0)],
        slow[i1, min(j, nx - 2), min(k, ny - 2)],
    )
    t1d1 = tv + dz * vref

    # Second dimension (X axis)
    vref = min(
        slow[max(i - 1, 0), j1, max(k - 1, 0)],
        slow[min(i, nz - 2), j1, max(k - 1, 0)],
        slow[max(i - 1, 0), j1, min(k, ny - 2)],
        slow[min(i, nz - 2), j1, min(k, ny - 2)],
    )
    t1d2 = te + dx * vref

    # Third dimension (Y axis)
    vref = min(
        slow[max(i - 1, 0), max(j - 1, 0), k1],
        slow[max(i - 1, 0), min(j, nx - 2), k1],
        slow[min(i, nz - 2), max(j - 1, 0), k1],
        slow[min(i, nz - 2), min(j, nx - 2), k1],
    )
    t1d3 = tn + dy * vref

    t1d = min(t1d1, t1d2, t1d3)

    # 2D operators
    # ZX plane
    t2d1 = Big
    vref = min(slow[i1, j1, max(k - 1, 0)], slow[i1, j1, min(k, ny - 2)])
    if tv < te + dx * vref and te < tv + dz * vref:
        ta = tev + te - tv
        tb = tev - te + tv
        t2d1 = (
            (tb * dz2i + ta * dx2i)
            + (4.0 * vref ** 2.0 * (dz2i + dx2i) - dz2i * dx2i * (ta - tb) ** 2.0)
            ** 0.5
        ) / (dz2i + dx2i)

    # ZY plane
    t2d2 = Big
    vref = min(slow[i1, max(j - 1, 0), k1], slow[i1, min(j, nx - 2), k1])
    if tv < tn + dy * vref and tn < tv + dz * vref:
        ta = tv - tn + tnv
        tb = tn - tv + tnv
        t2d2 = (
            (ta * dz2i + tb * dy2i)
            + (4.0 * vref ** 2.0 * (dz2i + dy2i) - dz2i * dy2i * (ta - tb) ** 2.0)
            ** 0.5
        ) / (dz2i + dy2i)

    # XY plane
    t2d3 = Big
    vref = min(slow[max(i - 1, 0), j1, k1], slow[min(i, nz - 2), j1, k1])
    if te < tn + dy * vref and tn < te + dx * vref:
        ta = te - tn + ten
        tb = tn - te + ten
        t2d3 = (
            (ta * dx2i + tb * dy2i)
            + (4.0 * vref ** 2.0 * (dx2i + dy2i) - dx2i * dy2i * (ta - tb) ** 2.0)
            ** 0.5
        ) / (dx2i + dy2i)

    t2d = min(t2d1, t2d2, t2d3)

    # 3D operator
    t3d = Big
    if min(t1d, t2d) > max(tv, te, tn):
        vref = slow[i1, j1, k1]
        ta = te - 0.5 * tn + 0.5 * ten - 0.5 * tv + 0.5 * tev - tnv + tnve
        tb = tv - 0.5 * tn + 0.5 * tnv - 0.5 * te + 0.5 * tev - ten + tnve
        tc = tn - 0.5 * te + 0.5 * ten - 0.5 * tv + 0.5 * tnv - tev + tnve

        t2 = vref ** 2.0 * dsum * 9.0
        t3 = dz2dx2 * (ta - tb) ** 2.0
        t3 += dz2dy2 * (tb - tc) ** 2.0
        t3 += dx2dy2 * (ta - tc) ** 2.0
        if t2 >= t3:
            t1 = tb * dz2i + ta * dx2i + tc * dy2i
            t3d = (t1 + (t2 - t3) ** 0.5) / dsum

    # Select minimum time
    t0 = tt[i, j, k]
    tt[i, j, k] = min(t0, t1d, t2d, t3d)

    # Compute gradient according to minimum time direction
    if grad and tt[i, j, k] != t0:
        if tt[i, j, k] == t1d1:
            ttsgn[i, j, k, 0] = sgntz
            ttsgn[i, j, k, 1] = 0
            ttsgn[i, j, k, 2] = 0

        elif tt[i, j, k] == t1d2:
            ttsgn[i, j, k, 0] = 0
            ttsgn[i, j, k, 1] = sgntx
            ttsgn[i, j, k, 2] = 0

        elif tt[i, j, k] == t1d3:
            ttsgn[i, j, k, 0] = 0
            ttsgn[i, j, k, 1] = 0
            ttsgn[i, j, k, 2] = sgnty

        elif tt[i, j, k] == t2d1:
            ttsgn[i, j, k, 0] = sgntz
            ttsgn[i, j, k, 1] = sgntx
            ttsgn[i, j, k, 2] = 0

        elif tt[i, j, k] == t2d2:
            ttsgn[i, j, k, 0] = sgntz
            ttsgn[i, j, k, 1] = 0
            ttsgn[i, j, k, 2] = sgnty

        elif tt[i, j, k] == t2d3:
            ttsgn[i, j, k, 0] = 0
            ttsgn[i, j, k, 1] = sgntx
            ttsgn[i, j, k, 2] = sgnty

        else:
            ttsgn[i, j, k, 0] = sgntz
            ttsgn[i, j, k, 1] = sgntx
            ttsgn[i, j, k, 2] = sgnty


@jitted(
    "void(f8[:, :, :], i4[:, :, :, :], f8[:, :, :], f8, f8, f8, f8, f8, f8, i4, i4, i4, b1, b1)"
)
def sweep3d(
    tt, ttsgn, slow, dz, dx, dy, zsi, xsi, ysi, nz, nx, ny, grad, init=False,
):
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
    if init:
        i0, j0, k0 = max(1, zsi - 1), max(1, xsi - 1), max(1, ysi - 1)
    else:
        i0, j0, k0 = 1, 1, 1

    for k in range(k0, ny):
        for j in range(j0, nx):
            for i in range(i0, nz):
                sweep(
                    tt, ttsgn, slow, dargs, i, j, k, 1, 1, 1, 1, 1, 1, nz, nx, ny, grad,
                )

    # Second sweeping: Top -> Bottom; East -> West; South -> North
    if init:
        i0, j0, k0 = max(1, zsi - 1), xsi + 1, max(1, ysi - 1)
    else:
        i0, j0, k0 = 1, nx - 2, 1

    for k in range(k0, ny):
        for j in range(j0, -1, -1):
            for i in range(i0, nz):
                sweep(
                    tt,
                    ttsgn,
                    slow,
                    dargs,
                    i,
                    j,
                    k,
                    1,
                    0,
                    1,
                    1,
                    -1,
                    1,
                    nz,
                    nx,
                    ny,
                    grad,
                )

    # Third sweeping: Top -> Bottom; West -> East; North -> South
    if init:
        i0, j0, k0 = max(1, zsi - 1), max(1, xsi - 1), ysi + 1
    else:
        i0, j0, k0 = 1, 1, ny - 2

    for k in range(k0, -1, -1):
        for j in range(j0, nx):
            for i in range(i0, nz):
                sweep(
                    tt,
                    ttsgn,
                    slow,
                    dargs,
                    i,
                    j,
                    k,
                    1,
                    1,
                    0,
                    1,
                    1,
                    -1,
                    nz,
                    nx,
                    ny,
                    grad,
                )

    # Fouth sweeping: Top -> Bottom; East -> West; North -> South
    if init:
        i0, j0, k0 = max(1, zsi - 1), xsi + 1, ysi + 1
    else:
        i0, j0, k0 = 1, nx - 2, ny - 2

    for k in range(k0, -1, -1):
        for j in range(j0, -1, -1):
            for i in range(i0, nz):
                sweep(
                    tt,
                    ttsgn,
                    slow,
                    dargs,
                    i,
                    j,
                    k,
                    1,
                    0,
                    0,
                    1,
                    -1,
                    -1,
                    nz,
                    nx,
                    ny,
                    grad,
                )

    # Fifth sweeping: Bottom -> Top; West -> East; South -> North
    if init:
        i0, j0, k0 = zsi + 1, max(1, xsi - 1), max(1, ysi - 1)
    else:
        i0, j0, k0 = nz - 2, 1, 1

    for k in range(k0, ny):
        for j in range(j0, nx):
            for i in range(i0, -1, -1):
                sweep(
                    tt,
                    ttsgn,
                    slow,
                    dargs,
                    i,
                    j,
                    k,
                    0,
                    1,
                    1,
                    -1,
                    1,
                    1,
                    nz,
                    nx,
                    ny,
                    grad,
                )

    # Sixth sweeping: Bottom -> Top; East -> West; South -> North
    if init:
        i0, j0, k0 = zsi + 1, xsi + 1, max(1, ysi - 1)
    else:
        i0, j0, k0 = nz - 2, nx - 2, 1
    for k in range(k0, ny):
        for j in range(j0, -1, -1):
            for i in range(i0, -1, -1):
                sweep(
                    tt,
                    ttsgn,
                    slow,
                    dargs,
                    i,
                    j,
                    k,
                    0,
                    0,
                    1,
                    -1,
                    -1,
                    1,
                    nz,
                    nx,
                    ny,
                    grad,
                )

    # Seventh sweeping: Bottom -> Top; West -> East; North -> South
    if init:
        i0, j0, k0 = zsi + 1, max(1, xsi - 1), ysi + 1
    else:
        i0, j0, k0 = nz - 2, 1, ny - 2

    for k in range(k0, -1, -1):
        for j in range(j0, nx):
            for i in range(i0, -1, -1):
                sweep(
                    tt,
                    ttsgn,
                    slow,
                    dargs,
                    i,
                    j,
                    k,
                    0,
                    1,
                    0,
                    -1,
                    1,
                    -1,
                    nz,
                    nx,
                    ny,
                    grad,
                )

    # Eighth sweeping: Bottom -> Top; East -> West; North -> South
    if init:
        i0, j0, k0 = zsi + 1, xsi + 1, ysi + 1
    else:
        i0, j0, k0 = nz - 2, nx - 2, ny - 2

    for k in range(k0, -1, -1):
        for j in range(j0, -1, -1):
            for i in range(i0, -1, -1):
                sweep(
                    tt,
                    ttsgn,
                    slow,
                    dargs,
                    i,
                    j,
                    k,
                    0,
                    0,
                    0,
                    -1,
                    -1,
                    -1,
                    nz,
                    nx,
                    ny,
                    grad,
                )


@jitted(
    "Tuple((f8[:, :, :], f8[:, :, :, :], f8))(f8[:, :, :], f8, f8, f8, f8, f8, f8, i4, b1)"
)
def fteik3d(slow, dz, dx, dy, zsrc, xsrc, ysrc, nsweep=2, grad=False):
    """Calculate traveltimes given a 3D velocity model."""
    # Parameters
    nz, nx, ny = numpy.shape(slow)

    # Check inputs
    condz = 0.0 <= zsrc <= dz * nz
    condx = 0.0 <= xsrc <= dx * nx
    condy = 0.0 <= ysrc <= dy * ny
    if not (condz and condx and condy):
        raise ValueError("source out of bound")

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
    nz += 1
    nx += 1
    ny += 1
    tt = numpy.full((nz, nx, ny), Big, dtype=numpy.float64)

    if grad:
        ttgrad = numpy.zeros((nz, nx, ny, 3), dtype=numpy.float64)
        ttsgn = numpy.zeros((nz, nx, ny, 3), dtype=numpy.int32)

    else:
        ttgrad = numpy.empty((0, 0, 0, 0), dtype=numpy.float64)
        ttsgn = numpy.empty((0, 0, 0, 0), dtype=numpy.int32)

    # Initialize points around source
    iterables = (
        (zsi, xsi, ysi),
        (zsi + 1, xsi, ysi),
        (zsi, xsi + 1, ysi),
        (zsi, xsi, ysi + 1),
        (zsi + 1, xsi + 1, ysi),
        (zsi + 1, xsi, ysi + 1),
        (zsi, xsi + 1, ysi + 1),
        (zsi + 1, xsi + 1, ysi + 1),
    )
    for i, j, k in iterables:
        tt[i, j, k], tzc, txc, tyc = t_anad(i, j, k, dz, dx, dy, zsa, xsa, ysa, vzero)

        if grad:
            ttgrad[i, j, k, 0] = tzc
            ttgrad[i, j, k, 1] = txc
            ttgrad[i, j, k, 2] = tyc

    # Start sweeping
    for i in range(nsweep):
        sweep3d(
            tt, ttsgn, slow, dz, dx, dy, zsi, xsi, ysi, nz, nx, ny, grad, i == 0,
        )

    if grad:
        for i in range(nz):
            for j in range(nx):
                for k in range(ny):
                    sgntz = ttsgn[i, j, k, 0]
                    if sgntz != 0:
                        t1 = tt[i - sgntz, j, k]
                        ttgrad[i, j, k, 0] = sgntz * (tt[i, j, k] - t1) / dz

                    sgntx = ttsgn[i, j, k, 1]
                    if sgntx != 0:
                        t1 = tt[i, j - sgntx, k]
                        ttgrad[i, j, k, 1] = sgntx * (tt[i, j, k] - t1) / dx

                    sgnty = ttsgn[i, j, k, 2]
                    if sgnty != 0:
                        t1 = tt[i, j, k - sgnty]
                        ttgrad[i, j, k, 2] = sgnty * (tt[i, j, k] - t1) / dy

                    # Normalize gradients
                    gn = norm3d(
                        ttgrad[i, j, k, 0], ttgrad[i, j, k, 1], ttgrad[i, j, k, 2]
                    )
                    if gn > 0.0:
                        ttgrad[i, j, k] /= gn

    return tt, ttgrad, vzero


@jitted(
    "Tuple((f8[:, :, :, :], f8[:, :, :, :, :], f8[:]))(f8[:, :, :], f8, f8, f8, f8[:], f8[:], f8[:], i4, b1)",
    parallel=True,
)
def fteik3d_vectorized(slow, dz, dx, dy, zsrc, xsrc, ysrc, nsweep=2, grad=False):
    """Calculate traveltimes in parallel for different sources."""
    nsrc = len(zsrc)
    nz, nx, ny = slow.shape
    tt = numpy.empty((nsrc, nz + 1, nx + 1, ny + 1), dtype=numpy.float64)
    ttgrad = (
        numpy.empty((nsrc, nz + 1, nx + 1, ny + 1, 3), dtype=numpy.float64)
        if grad
        else numpy.empty((nsrc, 0, 0, 0, 0), dtype=numpy.float64)
    )
    vzero = numpy.empty(nsrc, dtype=numpy.float64)
    for i in prange(nsrc):
        tt[i], ttgrad[i], vzero[i] = fteik3d(
            slow, dz, dx, dy, zsrc[i], xsrc[i], ysrc[i], nsweep, grad
        )

    return tt, ttgrad, vzero


@jitted
def solve3d(slow, dz, dx, dy, src, nsweep=2, grad=False):
    """Solve Eikonal."""
    if src.ndim == 1:
        return fteik3d(slow, dz, dx, dy, src[0], src[1], src[2], nsweep, grad)

    else:
        return fteik3d_vectorized(
            slow, dz, dx, dy, src[:, 0], src[:, 1], src[:, 2], nsweep, grad
        )
