import numba


def get_num_threads():
    return numba.get_num_threads()


def set_num_threads(n):
    numba.set_num_threads(n)
