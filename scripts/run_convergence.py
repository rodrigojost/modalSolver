import os
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from src.geometry import create_cantilever_box
from src.solver import solve_modal

def run_convergence_study():
    """
    Runs a mesh convergence study for the first eigenfrequency of a cantilever beam.
    Computes the error relative to the Euler-Bernoulli analytical value and plots it.
    """
    # Material properties
    E = 1.0e5
    nu = 0.0
    rho = 1.0e-3
    
    # Beam geometry (slender beam: L = 20, B = 0.5, H = 1.0)
    L = 20.0
    B = 0.5
    H = 1.0
    
    # Analytical first eigenfrequency (weak-axis bending)
    beta_1_L = 1.875104068711961
    I_z = H * (B ** 3) / 12.0
    A = B * H
    freq_ref = ((beta_1_L ** 2) / (2.0 * np.pi * (L ** 2))) * np.sqrt((E * I_z) / (rho * A))
    
    # Mesh resolutions to study
    resolutions = [
        {"Nx": 16,  "Ny": 1, "Nz": 2},
        {"Nx": 32,  "Ny": 2, "Nz": 4},
        {"Nx": 64,  "Ny": 4, "Nz": 8},
        {"Nx": 128, "Ny": 8, "Nz": 12}
    ]
    
    h_values = []
    errors = []
    
    # Run the solver for each resolution
    for res in resolutions:
        Nx, Ny, Nz = res["Nx"], res["Ny"], res["Nz"]
        h = L / Nx  # Mesh size parameter
        
        # Only print from rank 0
        if MPI.COMM_WORLD.rank == 0:
            print(f"Solving on mesh: {Nx}x{Ny}x{Nz} (h = {h:.4f})...")
            
        mesh = create_cantilever_box(L, B, H, Nx, Ny, Nz, cell_type_str="hexahedron")
        results, V = solve_modal(mesh, E, nu, rho, num_eigenvalues=1)
        
        if len(results) > 0:
            freq_num = results[0][0]
            err = abs(freq_num - freq_ref) / freq_ref
            
            h_values.append(h)
            errors.append(err)
            
            if MPI.COMM_WORLD.rank == 0:
                print(f"  Numerical: {freq_num:.6f} Hz, Analytical: {freq_ref:.6f} Hz, Error: {err * 100:.4f}%")
        else:
            if MPI.COMM_WORLD.rank == 0:
                print(f"  Warning: No converged eigenvalues found for mesh {Nx}x{Ny}x{Nz}")

    if MPI.COMM_WORLD.rank == 0 and len(errors) > 1:
        h_values = np.array(h_values)
        errors = np.array(errors)
        
        # Calculate experimental convergence rate (slope of log-log line)
        slopes = np.diff(np.log(errors)) / np.diff(np.log(h_values))
        avg_rate = np.mean(slopes)
        
        print("\nMesh Convergence Results:")
        for idx in range(len(h_values)):
            rate_str = f", rate = {slopes[idx-1]:.2f}" if idx > 0 else ""
            print(f"h = {h_values[idx]:.4f}: relative error = {errors[idx]*100:.4f}%{rate_str}")
        print(f"Average Experimental Convergence Rate: {avg_rate:.2f}")
        
        # Plot and save
        os.makedirs("results/figures", exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        plt.loglog(h_values, errors, 'o-', linewidth=2, label="Numerical Error")
        
        # Reference slope lines
        plt.loglog(h_values, errors[-1] * (h_values / h_values[-1])**1, '--', color='gray', label="O(h) Reference")
        plt.loglog(h_values, errors[-1] * (h_values / h_values[-1])**2, ':', color='gray', label="O(h^2) Reference")
        
        plt.xlabel("Mesh Size $h = L/N_x$", fontsize=12)
        plt.ylabel("Relative Error in First Eigenfrequency", fontsize=12)
        plt.title("Mesh Convergence Study", fontsize=14)
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        plot_path = "results/figures/convergence_study.png"
        plt.savefig(plot_path, dpi=300)
        print(f"\nConvergence plot saved to: {plot_path}")

if __name__ == "__main__":
    run_convergence_study()
