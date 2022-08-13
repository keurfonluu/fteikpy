import numpy as np

from fteikpy import Eikonal2D, Eikonal3D, grid_to_meshio, ray_to_meshio


def test_meshio_2d():
    nz, nx = 8, 10
    eik = Eikonal2D(np.ones((nz, nx)), (1.0, 1.0))
    tt = eik.solve((float(nz // 2), float(nx // 2)), return_gradient=True)
    ray = tt.raytrace((0.0, 0.0), honor_grid=True)

    mesh = grid_to_meshio(eik, tt)
    mesh_ray = ray_to_meshio(ray)

    npts = (nz + 1) * (nx + 1)
    assert len(mesh.points) == npts
    assert sum(len(cell) for cell in mesh.cells) == nz * nx

    assert mesh.point_data["Traveltime"][npts // 2] == 0.0
    assert np.allclose(mesh.point_data["Traveltime"].sum(), 378.23469225)

    for grad in mesh.point_data["Gradient"].T:
        assert grad[npts // 2] == 0.0
        assert np.allclose(grad.sum(), 0.0)

    assert mesh.cell_data["Velocity"][0].sum() == nz * nx

    assert len(mesh_ray.points) == len(ray)
    assert len(mesh_ray.cells[0].data) == len(ray) - 1
    assert mesh_ray.cells[0].data.sum() == 64.0


def test_meshio_3d():
    nz, nx, ny = 8, 10, 12
    eik = Eikonal3D(np.ones((nz, nx, ny)), (1.0, 1.0, 1.0))
    tt = eik.solve(
        (float(nz // 2), float(nx // 2), float(ny // 2)), return_gradient=True
    )
    ray = tt.raytrace((0.0, 0.0, 0.0), honor_grid=True)

    mesh = grid_to_meshio(eik, tt)
    mesh_ray = ray_to_meshio(ray)

    npts = (nz + 1) * (nx + 1) * (ny + 1)
    assert len(mesh.points) == npts
    assert sum(len(cell) for cell in mesh.cells) == nz * nx * ny

    assert mesh.point_data["Traveltime"][npts // 2] == 0.0
    assert np.allclose(mesh.point_data["Traveltime"].sum(), 6909.90160991)

    for grad in mesh.point_data["Gradient"].T:
        assert grad[npts // 2] == 0.0
        assert np.allclose(grad.sum(), 0.0)

    assert mesh.cell_data["Velocity"][0].sum() == nz * nx * ny

    assert len(mesh_ray.points) == len(ray)
    assert len(mesh_ray.cells[0].data) == len(ray) - 1
    assert mesh_ray.cells[0].data.sum() == 169.0
