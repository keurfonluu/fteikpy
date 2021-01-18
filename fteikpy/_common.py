from numba import jit


def jitted(*args, **kwargs):
    """Custom :func:`jit` with default options."""
    kwargs.update(
        {
            "nopython": True,
            "nogil": True,
            # Disable fast-math flag "nnan"
            # <https://llvm.org/docs/LangRef.html#fast-math-flags>
            "fastmath": {"ninf", "nsz", "arcp", "contract", "afn", "reassoc"},
            # "fastmath": True,
            # "boundscheck": False,
            "cache": True,
        }
    )
    return jit(*args, **kwargs)


@jitted
def norm2d(x, y):
    """Calculate norm of vector [x, y]."""
    return (x * x + y * y) ** 0.5


@jitted
def norm3d(x, y, z):
    """Calculate norm of vector [x, y, z]."""
    return (x * x + y * y + z * z) ** 0.5


@jitted
def dist2d(x1, y1, x2, y2):
    """Calculate Euclidean distance between [x1, y1] and [x2, y2]."""
    return norm2d(x1 - x2, y1 - y2)


@jitted
def dist3d(x1, y1, z1, x2, y2, z2):
    """Calculate Euclidean distance between [x1, y1, z1] and [x2, y2, z2]."""
    return norm3d(x1 - x2, y1 - y2, z1 - z2)
