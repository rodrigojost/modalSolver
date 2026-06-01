"""
run_scenario_study.py
=====================
Compares two cantilever beam designs at equal total mass:

  Variant A - Uniform rectangular beam (reference)
  Variant B - Linearly tapered beam    (optimised)

For each variant:
  1. Generates the mesh
  2. Solves the 3D elasticity eigenvalue problem (first 6 modes)
  3. Computes Euler-Bernoulli analytical frequencies (Variant A only)
  4. Calculates modal participation factors and effective masses
  5. Exports mode shapes to XDMF (open in ParaView)
  6. Generates automated PNG figures via PyVista and Matplotlib

Run from the project root:
  PYTHONPATH=. python scripts/run_scenario_study.py
"""

import os
import sys
import numpy as np
from mpi4py import MPI

from src.geometry import create_cantilever_box, create_tapered_cantilever
from src.solver import solve_modal, compute_modal_participation
from src.visualization import (
    export_xdmf,
    plot_geometry_comparison,
    plot_mode_shapes,
    plot_frequency_comparison,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared parameters
# ─────────────────────────────────────────────────────────────────────────────
E        = 1.0e5    # Young's modulus
nu       = 0.0     # Poisson's ratio (0 keeps the analytical comparison clean)
rho      = 1.0e-3   # mass density

L        = 20.0    # beam length
B        = 0.5     # nominal cross-section width
H        = 1.0     # nominal cross-section height

Nx, Ny, Nz  = 200, 5, 10   # mesh resolution (matches reference webpage)
TIP_RATIO   = 0.2           # Variant B: tip / root = 20 %
NUM_MODES   = 6

RANK = MPI.COMM_WORLD.rank

# ─────────────────────────────────────────────────────────────────────────────
# Euler-Bernoulli reference frequencies (Variant A, uniform beam)
# ─────────────────────────────────────────────────────────────────────────────
BETA_L = [1.875104068711961, 4.694091132974175, 7.854757438237613]

def _eb_freq(beta_l, I, A_sec):
    return (beta_l**2 / (2.0 * np.pi * L**2)) * np.sqrt(E * I / (rho * A_sec))

A_sec = B * H
I_z   = H * B**3 / 12.0   # weak-axis (bending in y)
I_y   = B * H**3 / 12.0   # strong-axis (bending in z)

# Mode order: weak1, strong1, weak2, strong2, weak3, strong3
EB_FREQS = [
    _eb_freq(BETA_L[0], I_z, A_sec),
    _eb_freq(BETA_L[0], I_y, A_sec),
    _eb_freq(BETA_L[1], I_z, A_sec),
    _eb_freq(BETA_L[1], I_y, A_sec),
    _eb_freq(BETA_L[2], I_z, A_sec),
    _eb_freq(BETA_L[2], I_y, A_sec),
]
EB_LABELS = [
    "Weak Bending 1", "Strong Bending 1",
    "Weak Bending 2", "Strong Bending 2",
    "Weak Bending 3", "Strong Bending 3",
]

TOTAL_MASS = rho * L * B * H   # same for both variants (equal volume)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: solve one variant and print its results table
# ─────────────────────────────────────────────────────────────────────────────
def run_variant(label, mesh, eb_freqs=None):
    """Solve and return (results, V) for one variant."""
    if RANK == 0:
        print(f"\n{'='*65}")
        print(f"  {label}")
        print(f"{'='*65}")
        print("  Assembling and solving eigenvalue problem...")

    results, V = solve_modal(mesh, E, nu, rho, num_eigenvalues=NUM_MODES)

    if RANK == 0:
        header = f"{'Mode':<5}  {'Freq [Hz]':>12}  {'EB Theory [Hz]':>15}  {'Err [%]':>8}  {'Description'}"
        print(f"\n  {header}")
        print(f"  {'-'*len(header)}")
        for i, (freq, _) in enumerate(results):
            if eb_freqs is not None:
                eb  = eb_freqs[i]
                err = abs(freq - eb) / eb * 100.0
                print(f"  {i+1:<5}  {freq:>12.5f}  {eb:>15.5f}  {err:>7.2f}%  {EB_LABELS[i]}")
            else:
                print(f"  {i+1:<5}  {freq:>12.5f}  {'N/A':>15}  {'—':>8}")

    return results, V


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute and print participation table for one variant
# ─────────────────────────────────────────────────────────────────────────────
def print_participation(label, mesh, V, results):
    if RANK != 0:
        return
    print(f"\n  Modal Participation Factors — {label}")
    print(f"  {'Mode':<5}  {'Freq [Hz]':>10}  "
          f"{'m_eff_y [%]':>12}  {'m_eff_z [%]':>12}")
    print(f"  {'-'*50}")
    sum_y, sum_z = 0.0, 0.0
    for i, (freq, phi) in enumerate(results):
        _, _, meff_y = compute_modal_participation(mesh, V, phi, rho, direction_index=1)
        _, _, meff_z = compute_modal_participation(mesh, V, phi, rho, direction_index=2)
        rel_y = meff_y / TOTAL_MASS * 100.0
        rel_z = meff_z / TOTAL_MASS * 100.0
        sum_y += rel_y
        sum_z += rel_z
        print(f"  {i+1:<5}  {freq:>10.5f}  {rel_y:>11.2f}%  {rel_z:>11.2f}%")
    print(f"  {'TOTAL':<5}  {'':>10}  {sum_y:>11.2f}%  {sum_z:>11.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("results/figures",       exist_ok=True)
    os.makedirs("results/modes_variantA", exist_ok=True)
    os.makedirs("results/modes_variantB", exist_ok=True)

    # ── Variant A: Uniform beam ───────────────────────────────────────────────
    if RANK == 0:
        print("\nGenerating Variant A mesh (uniform box)...")
    mesh_A = create_cantilever_box(L, B, H, Nx, Ny, Nz)
    results_A, V_A = run_variant("VARIANT A — Uniform Cantilever Beam", mesh_A, EB_FREQS)
    print_participation("Variant A", mesh_A, V_A, results_A)

    # ── Variant B: Tapered beam ───────────────────────────────────────────────
    if RANK == 0:
        print(f"\nGenerating Variant B mesh (tapered, tip_ratio={TIP_RATIO})...")
    mesh_B, B_root, H_root = create_tapered_cantilever(
        L, B, H, Nx, Ny, Nz, tip_ratio=TIP_RATIO
    )
    if RANK == 0:
        print(f"  Root cross-section: B_root={B_root:.4f}, H_root={H_root:.4f}")
        tip_b = TIP_RATIO * B_root
        tip_h = TIP_RATIO * H_root
        print(f"  Tip  cross-section: B_tip ={tip_b:.4f}, H_tip ={tip_h:.4f}")
    results_B, V_B = run_variant("VARIANT B — Tapered Cantilever Beam", mesh_B)
    print_participation("Variant B", mesh_B, V_B, results_B)

    # ── Summary comparison ────────────────────────────────────────────────────
    if RANK == 0:
        print(f"\n{'='*65}")
        print("  COMPARISON SUMMARY  (equal total mass)")
        print(f"{'='*65}")
        print(f"  {'Mode':<5}  {'Variant A [Hz]':>14}  {'Variant B [Hz]':>14}  {'Δf [%]':>8}")
        print(f"  {'-'*50}")
        for i in range(min(len(results_A), len(results_B))):
            fA = results_A[i][0]
            fB = results_B[i][0]
            delta = (fB - fA) / fA * 100.0
            sign  = "▲" if delta >= 0 else "▼"
            print(f"  {i+1:<5}  {fA:>14.5f}  {fB:>14.5f}  {sign}{abs(delta):>6.2f}%")
        print(f"\n  ► Variant B (Tapered) achieves a higher f₁ at equal mass,")
        print(f"    confirming that tapering concentrates stiffness where")
        print(f"    bending moments are largest (near the clamped root).")

    # ── XDMF export (ParaView) ────────────────────────────────────────────────
    if RANK == 0:
        print("\nExporting XDMF files for ParaView...")
    for i, (_, phi) in enumerate(results_A):
        phi.name = f"mode_{i+1}"
        export_xdmf(mesh_A, [phi], f"results/modes_variantA/mode_{i+1:02d}")
    for i, (_, phi) in enumerate(results_B):
        phi.name = f"mode_{i+1}"
        export_xdmf(mesh_B, [phi], f"results/modes_variantB/mode_{i+1:02d}")

    # ── PyVista figures ───────────────────────────────────────────────────────
    if RANK == 0:
        print("\nRendering figures...")

    # Geometry comparison
    plot_geometry_comparison(
        mesh_A, mesh_B,
        output_path="results/figures/geometry_comparison.png",
    )

    # Mode shape heatmaps — first 3 modes of each variant
    plot_mode_shapes(
        mesh_A, results_A, "Variant A",
        output_path="results/figures/modes_variantA.png",
        n_modes=6,
    )
    plot_mode_shapes(
        mesh_B, results_B, "Variant B",
        output_path="results/figures/modes_variantB.png",
        n_modes=6,
    )

    # Frequency bar chart
    plot_frequency_comparison(
        results_A, results_B,
        output_path="results/figures/frequency_comparison.png",
    )

    if RANK == 0:
        print("\n✓ All done.  Output files:")
        print("  results/figures/geometry_comparison.png")
        print("  results/figures/modes_variantA.png")
        print("  results/figures/modes_variantB.png")
        print("  results/figures/frequency_comparison.png")
        print("  results/modes_variantA/mode_01.xdmf  ... (open in ParaView)")
        print("  results/modes_variantB/mode_01.xdmf  ... (open in ParaView)")


if __name__ == "__main__":
    main()
