from ._grid import Grid2D, Grid3D, TraveltimeGrid2D, TraveltimeGrid3D
from ._helpers import get_num_threads, set_num_threads
from ._solver import Eikonal2D, Eikonal3D
from .__about__ import __version__

__all__ = [
    "Eikonal2D",
    "Eikonal3D",
    "Grid2D",
    "Grid3D",
    "TraveltimeGrid2D",
    "TraveltimeGrid3D",
    "get_num_threads",
    "set_num_threads",
    "__version__",
]
