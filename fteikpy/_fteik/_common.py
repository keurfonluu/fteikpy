import numpy as np

from .._common import jitted


@jitted("f8(f8[:], f8[:], f8[:], f8[:])")
def shrink(pcur, delta, lower, upper):
    """Stepsize shrinking factor if ray crosses interface."""
    tmp = pcur - delta
    maskl = tmp < lower
    masku = tmp > upper

    if maskl.any():
        return np.min((pcur[maskl] - lower[maskl]) / delta[maskl])

    elif masku.any():
        return np.min((pcur[masku] - upper[masku]) / delta[masku])

    else:
        return 1.0
