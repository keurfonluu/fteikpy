import numpy

from ._solver import Eikonal2D, Eikonal3D
from ._grid import TraveltimeGrid2D, TraveltimeGrid3D


def grid_to_meshio(*args):
    """
    Return a :class:`meshio.Mesh` object from grids.

    Parameters
    ----------
    arg1, arg2, ..., argn : grid_like
        Grid objects to convert to mesh. Should be one of:

         - :class:`fteikpy.Eikonal2D`
         - :class:`fteikpy.Eikonal3D`
         - :class:`fteikpy.TraveltimeGrid2D`
         - :class:`fteikpy.TraveltimeGrid3D`

    Returns
    -------
    :class:`meshio.Mesh`
        Output mesh.
    
    """
    import meshio

    for i, arg in enumerate(args):
        if not isinstance(arg, (Eikonal2D, Eikonal3D, TraveltimeGrid2D, TraveltimeGrid3D)):
            raise ValueError(f"argument {i + 1} is not a supported grid")

        if i == 0:
            ndim = arg._ndim
            shape = arg.shape
            gridsize = arg.gridsize
            origin = arg.origin

    # Generate mesh
    if ndim == 2:
        nz, nx = shape
        dz, dx = gridsize
        z0, x0 = origin

        if isinstance(args[0], TraveltimeGrid2D):
            nz -= 1
            nx -= 1

        points, cells = _generate_mesh_2d(nx, nz, dx, dz, x0, z0)

        # Append third dimension and swap axis
        points = numpy.column_stack((points, numpy.zeros(len(points))))
        points = points[:, [0, 2, 1]]

    else:
        nz, nx, ny = shape
        dz, dx, dy = gridsize
        z0, x0, y0 = origin

        if isinstance(args[0], TraveltimeGrid3D):
            nz -= 1
            nx -= 1
            ny -= 1

        points, cells = _generate_mesh_3d(nx, ny, nz, dx, dy, dz, x0, y0, z0)

    # Invert z-axis (depth -> elevation)
    points[:, 2] *= -1.0

    # Generate data arrays
    point_data = {}
    cell_data = {}
    vel_count = 0
    tt_count = 0

    for arg in args:
        # Velocity model
        if isinstance(arg, (Eikonal2D, Eikonal3D)):
            vel_count += 1

            name = f"Velocity {vel_count}" if vel_count > 1 else "Velocity"
            cell_data[name] = [_ravel_grid(arg.grid, ndim)]

        # Traveltime grid
        elif isinstance(arg, (TraveltimeGrid2D, TraveltimeGrid3D)):
            tt_count += 1

            name = f"Traveltime {tt_count}" if tt_count > 1 else "Traveltime"
            point_data[name] = _ravel_grid(arg.grid, ndim)

            # Gradient grid
            if arg._gradient is not None:
                name = f"Gradient {tt_count}" if tt_count > 1 else "Gradient"
                gradient = numpy.column_stack([
                    _ravel_grid(grad.grid, ndim) for grad in arg.gradient
                ])

                if ndim == 2:
                    gradient = numpy.column_stack((gradient, numpy.zeros(len(points))))
                gradient = gradient[:, [1, 2, 0]]    
                gradient[:, 2] *= -1.0
                point_data[name] = gradient

    return meshio.Mesh(points, cells, point_data, cell_data)


def _generate_mesh_2d(nx, ny, dx, dy, x0, y0, order="F"):
    """Generate 2D structured grid."""
    # Internal functions
    def meshgrid(x, y, indexing="ij", order=order):
        """Generate mesh grid."""
        X, Y = numpy.meshgrid(x, y, indexing=indexing)
        return X.ravel(order), Y.ravel(order)

    def mesh_vertices(i, j):
        """Generate vertices for each quad."""
        return [
            [i, j],
            [i + 1, j],
            [i + 1, j + 1],
            [i, j + 1],
        ]

    # Grid
    dx = numpy.arange(nx + 1) * dx + x0
    dy = numpy.arange(ny + 1) * dy + y0
    xy_shape = [nx + 1, ny + 1]
    ij_shape = [nx, ny]
    X, Y = meshgrid(dx, dy)
    I, J = meshgrid(*[numpy.arange(n) for n in ij_shape])

    # Points and cells
    points = [[x, y] for x, y in zip(X, Y)]
    cells = [
        [
            numpy.ravel_multi_index(vertex, xy_shape, order=order)
            for vertex in mesh_vertices(i, j)
        ]
        for i, j in zip(I, J)
    ]

    return numpy.array(points, dtype=float), [("quad", numpy.array(cells))]


def _generate_mesh_3d(nx, ny, nz, dx, dy, dz, x0, y0, z0):
    """Generate 3D structured grid."""
    # Internal functions
    def meshgrid(x, y, z, indexing="ij", order="C"):
        """Generate mesh grid."""
        X, Y, Z = numpy.meshgrid(x, y, z, indexing=indexing)
        return X.ravel(order), Y.ravel(order), Z.ravel(order)

    def mesh_vertices(i, j, k):
        """Generate vertices for each hexahedron."""
        return [
            [i, j, k],
            [i + 1, j, k],
            [i + 1, j + 1, k],
            [i, j + 1, k],
            [i, j, k + 1],
            [i + 1, j, k + 1],
            [i + 1, j + 1, k + 1],
            [i, j + 1, k + 1],
        ]

    # Grid
    dx = numpy.arange(nx + 1) * dx + x0
    dy = numpy.arange(ny + 1) * dy + y0
    dz = numpy.arange(nz + 1) * dz + z0
    xyz_shape = [nx + 1, ny + 1, nz + 1]
    ijk_shape = [nx, ny, nz]
    X, Y, Z = meshgrid(dx, dy, dz)
    I, J, K = meshgrid(*[numpy.arange(n) for n in ijk_shape])

    # Points and cells
    points = [[x, y, z] for x, y, z in zip(X, Y, Z)]
    cells = [
        [
            numpy.ravel_multi_index(vertex, xyz_shape, order="C")
            for vertex in mesh_vertices(i, j, k)
        ]
        for i, j, k in zip(I, J, K)
    ]

    return (
        numpy.array(points, dtype=float),
        [("hexahedron", numpy.array(cells))],
    )


def _ravel_grid(grid, ndim):
    """Ravel grid."""
    return (
        grid.ravel()
        if ndim == 2
        else numpy.transpose(grid, axes=[1, 2, 0]).ravel()
    )
