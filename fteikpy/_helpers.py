import numba


def get_num_threads():
    """
    Get current number of threads.
    
    Returns
    -------
    int
        Number of threads.

    """
    return numba.get_num_threads()


def set_num_threads(n):
    """
    Set number of threads.

    Parameters
    ----------
    n : int
        Number of threads.

    """
    numba.set_num_threads(n)
