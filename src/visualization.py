"""
visualization.py
================
Visualization utilities for the modalSolver package.

Provides:
  - XDMF export for ParaView (open the .xdmf files directly in ParaView on Windows)
  - Automated offscreen rendering via PyVista (no display required)
    * Side-by-side geometry/mesh comparison
    * Mode-shape heatmaps (displacement magnitude colored on the warped surface)
    * Frequency bar-chart comparison between two variants
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend, safe inside WSL
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    import pyvista as pv
    # Try to start a virtual framebuffer (requires: sudo apt-get install xvfb libgl1-mesa-glx)
    try:
        pv.start_xvfb()
        print("[visualization] PyVista xvfb virtual display started.")
    except Exception as e_xvfb:
        # xvfb not available — fall back to pure off-screen software rendering
        print(f"[visualization] xvfb not available ({e_xvfb}). Falling back to off-screen mode.")
        print("[visualization] To enable xvfb: sudo apt-get install -y xvfb libgl1-mesa-glx")
        pv.global_theme.allow_empty_mesh = True
    pv.OFF_SCREEN = True
    PYVISTA_OK = True
except ImportError:
    PYVISTA_OK = False
    print("[visualization] PyVista not installed – skipping 3D renders.")
    print("[visualization] Fix: conda install -c conda-forge pyvista vtk")

try:
    import dolfinx.io
    DOLFINX_IO_OK = True
except ImportError:
    DOLFINX_IO_OK = False


# ---------------------------------------------------------------------------
# XDMF export (ParaView)
# ---------------------------------------------------------------------------

def export_xdmf(mesh, functions: list, filepath: str):
    """
    Export mesh and a list of dolfinx Functions to an XDMF file.

    The resulting <filepath>.xdmf and <filepath>.h5 can be opened directly
    in ParaView on Windows by navigating to the G: drive path.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
    functions : list of dolfinx.fem.Function
        Each function will be written as a separate named field.
    filepath : str
        Path without extension (e.g. "results/modes_variantA/mode_01").
    """
    if not DOLFINX_IO_OK:
        return
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with dolfinx.io.XDMFFile(mesh.comm, filepath + ".xdmf", "w") as xf:
        xf.write_mesh(mesh)
        for fn in functions:
            xf.write_function(fn)


# ---------------------------------------------------------------------------
# PyVista helpers
# ---------------------------------------------------------------------------

def _dolfinx_mesh_to_pyvista(mesh):
    """Convert a dolfinx mesh to a PyVista UnstructuredGrid."""
    import pyvista as pv
    from dolfinx.plot import vtk_mesh as vtk_mesh_fn

    topology, cell_types, geometry = vtk_mesh_fn(mesh)
    return pv.UnstructuredGrid(topology, cell_types, geometry)


def _function_to_numpy(fn):
    """Return the nodal values of a dolfinx vector Function as a (n, 3) array."""
    vals = fn.x.array.reshape(-1, fn.function_space.dofmap.index_map_bs)
    if vals.shape[1] < 3:
        pad = np.zeros((vals.shape[0], 3 - vals.shape[1]))
        vals = np.hstack([vals, pad])
    return vals


# ---------------------------------------------------------------------------
# Public rendering functions
# ---------------------------------------------------------------------------

def plot_geometry_comparison(mesh_A, mesh_B, output_path: str, label_A="Variant A (Uniform)",
                              label_B="Variant B (Tapered)"):
    """
    Render both meshes side-by-side as wireframes and save a PNG.

    Parameters
    ----------
    mesh_A, mesh_B : dolfinx.mesh.Mesh
    output_path : str
        Full path to the output PNG file.
    """
    if not PYVISTA_OK:
        return
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    grid_A = _dolfinx_mesh_to_pyvista(mesh_A)
    grid_B = _dolfinx_mesh_to_pyvista(mesh_B)

    pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1600, 600))

    for col, (grid, label) in enumerate([(grid_A, label_A), (grid_B, label_B)]):
        pl.subplot(0, col)
        pl.add_mesh(grid, style="wireframe", color="steelblue", line_width=1.2)
        pl.add_mesh(grid, style="surface", color="lightcyan", opacity=0.35)
        pl.add_title(label, font_size=14)
        pl.camera_position = "iso"
        pl.camera.zoom(1.2)

    pl.screenshot(output_path)
    pl.close()
    print(f"[visualization] Saved geometry comparison → {output_path}")


def plot_mode_shapes(mesh, modes: list, variant_name: str, output_path: str,
                     n_modes: int = 3):
    """
    Render the first n_modes mode shapes as a horizontal panel of heatmaps.

    Each panel shows the displacement magnitude ||phi|| colored on the
    warp-by-vector deformed surface of the mesh.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
    modes : list of (freq_hz, dolfinx.fem.Function)
    variant_name : str
        Used as the figure title.
    output_path : str
        Full path to the output PNG file.
    n_modes : int
        Number of modes to render (default 3).
    """
    if not PYVISTA_OK:
        return
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    n = min(n_modes, len(modes))
    pl = pv.Plotter(shape=(1, n), off_screen=True, window_size=(560 * n, 620))

    grid_base = _dolfinx_mesh_to_pyvista(mesh)

    for col in range(n):
        freq_hz, phi = modes[col]
        vals = _function_to_numpy(phi)
        mag = np.linalg.norm(vals, axis=1)

        # Normalise displacements for warping (scale warp to ~10 % of beam length)
        coords = mesh.geometry.x
        L_approx = coords[:, 0].max() - coords[:, 0].min()
        warp_scale = 0.10 * L_approx / (mag.max() + 1e-12)

        grid = grid_base.copy()
        grid.point_data["displacement"] = vals
        grid.point_data["magnitude"] = mag
        warped = grid.warp_by_vector("displacement", factor=warp_scale)

        pl.subplot(0, col)
        pl.add_mesh(
            warped,
            scalars="magnitude",
            cmap="plasma",
            show_scalar_bar=(col == n - 1),
            scalar_bar_args={"title": "||φ|| (norm.)", "vertical": True},
        )
        pl.add_title(f"Mode {col + 1}  f = {freq_hz:.3f} Hz", font_size=11)
        pl.camera_position = "iso"
        pl.camera.zoom(1.15)

    pl.screenshot(output_path)
    pl.close()
    print(f"[visualization] Saved mode shapes for {variant_name} → {output_path}")


def plot_frequency_comparison(results_A: list, results_B: list, output_path: str,
                               label_A="Variant A (Uniform)", label_B="Variant B (Tapered)"):
    """
    Bar chart comparing eigenfrequencies of Variant A and Variant B.

    Parameters
    ----------
    results_A, results_B : list of (freq_hz, Function)
    output_path : str
        Full path to the output PNG file.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    n = min(len(results_A), len(results_B))
    freqs_A = [results_A[i][0] for i in range(n)]
    freqs_B = [results_B[i][0] for i in range(n)]

    x = np.arange(1, n + 1)
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_A = ax.bar(x - width / 2, freqs_A, width, label=label_A, color="#4C9BE8", edgecolor="white")
    bars_B = ax.bar(x + width / 2, freqs_B, width, label=label_B, color="#E8754C", edgecolor="white")

    # Annotate bars with percentage difference
    for i in range(n):
        diff_pct = (freqs_B[i] - freqs_A[i]) / freqs_A[i] * 100.0
        sign = "+" if diff_pct >= 0 else ""
        mid_x = x[i] + width / 2
        ax.text(mid_x, freqs_B[i] * 1.015, f"{sign}{diff_pct:.1f}%",
                ha="center", va="bottom", fontsize=8, color="#B84C20")

    ax.set_xlabel("Mode Number", fontsize=12)
    ax.set_ylabel("Eigenfrequency [Hz]", fontsize=12)
    ax.set_title("Eigenfrequency Comparison: Uniform vs. Tapered Cantilever", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Mode {i}" for i in x])
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", which="major", ls="--", alpha=0.4)
    ax.grid(axis="y", which="minor", ls=":", alpha=0.2)
    ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[visualization] Saved frequency comparison chart → {output_path}")
