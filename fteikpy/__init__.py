from ._helpers import set_num_threads
from ._solver import EikonalSolver2D, EikonalSolver3D
from .__about__ import __version__

__all__ = [
    "EikonalSolver2D",
    "EikonalSolver3D",
    "set_num_threads",
    "__version__",
]
