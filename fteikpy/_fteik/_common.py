import numpy

from .._common import jitted


@jitted
def first_index(x, y):
    """Find index of first occurrence of x in array y."""
    for i, v in enumerate(y):
        if numpy.array_equal(v, x):
            return i

    return -1


@jitted("f8(f8[:], f8[:], f8[:], f8[:])")
def shrink(pcur, delta, lower, upper):
    """Stepsize shrinking factor if ray crosses interface."""
    tmp = pcur - delta
    maskl = tmp < lower
    masku = tmp > upper
    
    if maskl.any():
        return numpy.min((pcur[maskl] - lower[maskl]) / delta[maskl])

    elif masku.any():
        return numpy.min((pcur[masku] - upper[masku]) / delta[masku])

    else:
        return 1.0
