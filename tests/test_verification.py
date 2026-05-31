import pytest
import numpy as np
from mpi4py import MPI
import dolfinx
from src.geometry import create_cantilever_box
from src.solver import solve_modal

def test_cantilever_first_eigenfrequency():
    """
    Verifies that the first eigenfrequency computed by the 3D elasticity model
    matches the analytical Euler-Bernoulli value for a slender beam.
    """
    # Material properties
    E = 1.0e5
    nu = 0.0
    rho = 1.0e-3
    
    # Beam geometry (slender beam: L / B = 40, L / H = 20)
    L = 20.0
    B = 0.5
    H = 1.0
    
    # Mesh resolution
    Nx = 100
    Ny = 2
    Nz = 4
    
    # Create box mesh (hexahedrons)
    mesh = create_cantilever_box(L, B, H, Nx, Ny, Nz, cell_type_str="hexahedron")
    
    # Solve eigenvalue problem for the first few modes
    num_eigenvalues = 2
    results, V = solve_modal(mesh, E, nu, rho, num_eigenvalues=num_eigenvalues)
    
    # Analytical first eigenfrequency for weak-axis bending (y-direction)
    # beta_1 * L = 1.875104
    beta_1_L = 1.875104068711961
    I_z = H * (B ** 3) / 12.0
    A = B * H
    
    # f_1 = (beta_1_L^2 / (2 * pi * L^2)) * sqrt(E * I_z / (rho * A))
    freq_analytical = ((beta_1_L ** 2) / (2.0 * np.pi * (L ** 2))) * np.sqrt((E * I_z) / (rho * A))
    
    # Find the numerical frequency corresponding to the first weak-axis bending mode
    # Since B = 0.5 and H = 1.0, the first mode is weak-axis bending (y-direction)
    # The second mode is strong-axis bending (z-direction) or torsion.
    # The first mode in results should be the weak-axis bending mode.
    assert len(results) >= 1, "No converged eigenvalues found."
    freq_numerical = results[0][0]
    
    # Calculate relative error
    rel_error = abs(freq_numerical - freq_analytical) / freq_analytical
    
    print(f"\nAnalytical First Frequency: {freq_analytical:.6f} Hz")
    print(f"Numerical First Frequency: {freq_numerical:.6f} Hz")
    print(f"Relative Error: {rel_error * 100:.4f}%")
    
    # The error should be small (within 2% for this discretization and aspect ratio)
    assert rel_error < 0.02, f"Relative error {rel_error*100:.2f}% exceeds the 2% threshold."

if __name__ == "__main__":
    test_cantilever_first_eigenfrequency()
