# modalSolver package
from .geometry import create_cantilever_box, create_tapered_cantilever
from .solver import solve_modal, compute_modal_participation
from .visualization import (
    export_xdmf,
    plot_geometry_comparison,
    plot_mode_shapes,
    plot_frequency_comparison,
)
