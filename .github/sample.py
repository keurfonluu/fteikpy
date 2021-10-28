import numpy
import pyvista

from fteikpy import Eikonal2D, grid_to_meshio

pyvista.set_plot_theme("document")


def ray2line(ray):
    """Convert a ray array to PolyData."""
    nr = len(ray)
    points = numpy.column_stack((ray[:, 1], numpy.zeros(nr), -ray[:, 0]))
    lines = numpy.column_stack((numpy.full(nr - 1, 2), numpy.arange(nr - 1), numpy.arange(1, nr)))

    return pyvista.PolyData(points, lines=lines)


# Import Marmousi velocity model
vel = numpy.load("marmousi.npy")

# Calculate traveltime grid for one source point
eik = Eikonal2D(vel * 1.0e-3, gridsize=(0.01, 0.01))
eik.smooth(0.05)
tt = eik.solve((0.0, 0.0), nsweep=3, return_gradient=True)
ttgrid = tt.grid.ravel()

# Trace rays for 100 locations
nrays = 100
end_points = numpy.zeros((nrays, 2))
end_points[:, 1] = numpy.linspace(4.4, eik.xaxis[-1], nrays)
rays = tt.raytrace(end_points)
trays = [tt(ray) for ray in rays]

# Create mesh
mesh = pyvista.from_meshio(grid_to_meshio(eik, tt))
mesh2 = mesh.copy()

# Create contour and lines
contour = mesh.contour(isosurfaces=100, scalars="Traveltime")
lines = [ray2line(ray) for ray in rays]

# Initialize plotter
p = pyvista.Plotter(window_size=(1500, 600), notebook=False)
p.add_mesh(
    mesh,
    scalars="Velocity",
    scalar_bar_args={
        "title": "Velocity [km/s]",
        "height": 0.7,
        "width": 0.05,
        "position_x": 0.92,
        "position_y": 0.15,
        "vertical": True,
        "fmt": "%.1f",
        "title_font_size": 20,
        "label_font_size": 20,
        "font_family": "arial",
    },
)
p.add_mesh(
    contour,
    color="black",
    line_width=1,
    opacity=0.5,
)
for line in lines:
    p.add_mesh(
        line,
        color="black",
        line_width=1,
    )
time = p.add_text(
    "Time: {:.2f} seconds".format(ttgrid.max()),
    position="upper_right",
    font_size=12,
    shadow=True,
)
p.show_grid(
    grid=False,
    ticks="outside",
    show_yaxis=False,
    show_ylabels=False,
    xlabel="Distance [km]",
    zlabel="Elevation [km]",
    font_size=20,
    font_family="arial",
)
p.show(
    cpos=[
        (5.113352562818404, -9.397825056260835, -1.4194558109669757),
        (5.113352562818404, 0.0, -1.4194558109669757),
        (0.0, 0.0, 1.0),
    ],
    auto_close=False,
)

# Update isochrones and rays
nframes = 48
nisos = numpy.linspace(1, 100, nframes)
times = numpy.linspace(0.0, ttgrid.max(), nframes)

p.open_movie("sample.mp4", framerate=12)
for n, t in zip(nisos, times):
    time.SetText(3, "Time: {:.2f} seconds".format(t))

    # Update isochrones
    mesh2["Traveltime"] = ttgrid.copy()
    mesh2["Traveltime"][ttgrid > t] = t
    c = mesh2.contour(isosurfaces=int(n), scalars="Traveltime")
    contour.points = c.points
    contour.lines = c.lines

    # Update rays
    for ray, tray, line in zip(rays, trays, lines):
        idx = tray <= t
        if idx.sum() > 1:
            r = ray[idx]
            l = ray2line(r)
            line.points = l.points
            line.lines = l.lines
    
    p.write_frame()

p.close()

# Convert MP4 to GIF online (e.g., https://convertio.co/)
# GIF produced using imageio are low quality since each frame is limited to 256 colors
