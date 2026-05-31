import numpy as np
from mpi4py import MPI
import dolfinx
import dolfinx.mesh


def create_cantilever_box(L: float, B: float, H: float, Nx: int, Ny: int, Nz: int,
                          cell_type_str: str = "hexahedron"):
    """
    Generates a structured 3D box mesh for a uniform cantilever beam.

    The beam is aligned with the x-axis, spanning [0, L] x [0, B] x [0, H].

    Parameters
    ----------
    L, B, H : float
        Length (x), width (y), height (z) of the beam.
    Nx, Ny, Nz : int
        Number of elements along each axis.
    cell_type_str : str
        "hexahedron" (default) or "tetrahedron".

    Returns
    -------
    dolfinx.mesh.Mesh
    """
    cell_type = _parse_cell_type(cell_type_str)
    mesh = dolfinx.mesh.create_box(
        comm=MPI.COMM_WORLD,
        points=[np.array([0.0, 0.0, 0.0]), np.array([L, B, H])],
        n=[Nx, Ny, Nz],
        cell_type=cell_type,
    )
    return mesh


def create_tapered_cantilever(L: float, B: float, H: float, Nx: int, Ny: int, Nz: int,
                               tip_ratio: float = 0.2, cell_type_str: str = "hexahedron"):
    """
    Generates a linearly tapered cantilever beam mesh with the same total volume
    as a uniform beam of dimensions L x B x H.

    The cross-section is full (B_root x H_root) at x=0 and scales linearly to
    (tip_ratio * B_root) x (tip_ratio * H_root) at x=L.

    Root dimensions are enlarged to conserve volume:
        Volume_tapered = L * B_root * H_root * (1 + tip_ratio + tip_ratio^2) / 3
        => B_root = B * sqrt(3 / (1 + tip + tip^2))
        => H_root = H * sqrt(3 / (1 + tip + tip^2))

    Parameters
    ----------
    L, B, H : float
        Target length and nominal cross-section of the equivalent uniform beam.
    Nx, Ny, Nz : int
        Number of elements along each axis.
    tip_ratio : float
        Ratio of tip cross-section to root cross-section (0 < tip_ratio < 1).
    cell_type_str : str
        "hexahedron" (default) or "tetrahedron".

    Returns
    -------
    dolfinx.mesh.Mesh
        Tapered mesh with same volume as the uniform beam.
    float
        Actual root width B_root.
    float
        Actual root height H_root.
    """
    if not (0.0 < tip_ratio < 1.0):
        raise ValueError("tip_ratio must be strictly between 0 and 1.")

    alpha = tip_ratio  # shorthand

    # Scale root dimensions to conserve volume
    volume_factor = (1.0 + alpha + alpha**2) / 3.0
    scale = np.sqrt(1.0 / volume_factor)  # symmetric: apply same scale to B and H
    B_root = B * scale
    H_root = H * scale

    # Create a box mesh using root dimensions
    cell_type = _parse_cell_type(cell_type_str)
    mesh = dolfinx.mesh.create_box(
        comm=MPI.COMM_WORLD,
        points=[np.array([0.0, 0.0, 0.0]), np.array([L, B_root, H_root])],
        n=[Nx, Ny, Nz],
        cell_type=cell_type,
    )

    # Warp mesh coordinates to produce the linear taper.
    # Each node at position x gets its y and z scaled by (1 - (1-alpha)*x/L).
    coords = mesh.geometry.x  # shape (n_nodes, 3), in-place modification
    x = coords[:, 0]
    taper = 1.0 - (1.0 - alpha) * (x / L)
    coords[:, 1] *= taper   # y scaled
    coords[:, 2] *= taper   # z scaled

    return mesh, B_root, H_root


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_cell_type(cell_type_str: str):
    """Map a string cell type name to a dolfinx CellType enum value."""
    mapping = {
        "hexahedron": dolfinx.mesh.CellType.hexahedron,
        "tetrahedron": dolfinx.mesh.CellType.tetrahedron,
    }
    key = cell_type_str.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported cell type: '{cell_type_str}'. Choose from {list(mapping)}.")
    return mapping[key]
