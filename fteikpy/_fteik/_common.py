from .._common import jitted


@jitted("f8(f8[:], f8[:], f8[:], f8[:])")
def shrink(pcur, delta, lower, upper):
    """Stepsize shrinking factor if ray crosses interface."""
    tmp = pcur - delta
    maskl = tmp < lower
    masku = tmp > upper

    condl = maskl.any()
    condu = masku.any()

    if condl and condu:
        bl = (pcur[maskl] - lower[maskl]) / delta[maskl]
        bu = (pcur[masku] - upper[masku]) / delta[masku]
        return min(bl.min(), bu.min())

    elif condl and not condu:
        bl = (pcur[maskl] - lower[maskl]) / delta[maskl]
        return bl.min()

    elif not condl and condu:
        bu = (pcur[masku] - upper[masku]) / delta[masku]
        return bu.min()

    else:
        return 1.0
