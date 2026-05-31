# modalSolver - FEniCSx Cantilever Beam Modal Analysis Template

This project implements a finite element model using FEniCSx and SLEPc to estimate the eigenfrequencies and mode shapes of a slender cantilever beam. The 3D elasticity results are validated against analytical solutions from Euler-Bernoulli beam theory, and we evaluate the modal participation factors and effective masses of the first 6 eigenmodes.

---

## Installation & Environment Setup

FEniCSx depends on Unix-specific libraries (PETSc, SLEPc, MPI) and is natively supported on Linux and macOS, but requires WSL2 or Docker on Windows.

### 1. Windows (via WSL2)
FEniCSx cannot be installed natively on Windows via conda. You must use WSL2 (Windows Subsystem for Linux):
1.  Open your WSL2 terminal (e.g., Ubuntu).
2.  Install Miniforge or Anaconda inside WSL2 if not already installed.
3.  Create and activate the environment using the provided config:
    ```bash
    conda env create -f environment.yml
    conda activate fenicsx-env
    ```

### 2. Linux (Native)
You can install the environment natively:
1.  Open your terminal.
2.  Create and activate the environment:
    ```bash
    conda env create -f environment.yml
    conda activate fenicsx-env
    ```

### 3. macOS (Native - Intel & Apple Silicon)
FEniCSx is natively supported via conda-forge on macOS:
1.  Open your terminal.
2.  Create and activate the environment:
    ```bash
    conda env create -f environment.yml
    conda activate fenicsx-env
    ```

### 4. Docker (Cross-Platform Alternative)
If you prefer not to use Conda, you can run the code inside the official FEniCSx Docker container:
```bash
docker run -it --rm -v $(pwd):/workspace -w /workspace dolfinx/dolfinx:stable
```


---

## Running the Verification and Studies

Ensure your environment is activated before running the commands.

### 1. Verify Environment Installation
Verify that the FEniCSx stack is successfully installed:
```bash
python -c "import dolfinx, petsc4py, mpi4py, slepc4py; print('FEniCSx stack ok')"
```

### 2. Run Automated Verification Tests
Run the unit tests to check the computed first eigenfrequency against the analytical Euler-Bernoulli solution:
```bash
pytest tests/test_verification.py
```

### 3. Run Mesh Convergence Study
Perform the mesh convergence study across four resolutions and plot the relative errors:
```bash
python scripts/run_convergence.py
```
This saves a convergence log-log plot to `results/figures/convergence_study.png`.

### 4. Run Scenario Study (Variant A vs. Variant B)
Solve for the first 6 eigenmodes of both a **uniform** and a **tapered** cantilever beam, compare their frequencies, compute modal participation factors, and generate all figures:
```bash
PYTHONPATH=. python scripts/run_scenario_study.py
```
Outputs:
- `results/figures/geometry_comparison.png` — side-by-side mesh render
- `results/figures/modes_variantA.png` — mode shape heatmaps, uniform beam
- `results/figures/modes_variantB.png` — mode shape heatmaps, tapered beam
- `results/figures/frequency_comparison.png` — bar chart comparing all 6 frequencies
- `results/modes_variantA/mode_*.xdmf` — open in **ParaView** for interactive 3D exploration
- `results/modes_variantB/mode_*.xdmf` — open in **ParaView** for interactive 3D exploration

> **ParaView (optional, Windows-native)**: Install ParaView from https://www.paraview.org/download/ directly on Windows. Open `.xdmf` files from your `G:` drive — no WSL needed for this step. Use *Warp By Vector* with the `mode_N` field to visualise animated mode shapes.

---

## Project Structure

```
modalSolver/
├── pyproject.toml              # Python project metadata and package requirements
├── environment.yml             # Conda environment (FEniCSx + pyvista + deps)
├── README.md                   # Environment setup and run instructions
├── run_in_wsl.ps1              # PowerShell helper to run scripts inside WSL
├── src/
│   ├── __init__.py             # Makes src a Python package
│   ├── solver.py               # Core FEniCSx + SLEPc eigenvalue solver
│   ├── geometry.py             # Uniform and tapered mesh generation
│   └── visualization.py        # PyVista rendering + XDMF export for ParaView
├── tests/
│   ├── __init__.py
│   └── test_verification.py    # Verifies eigenvalues against Euler-Bernoulli theory
├── scripts/
│   ├── run_convergence.py      # Mesh convergence study and log-log plot
│   └── run_scenario_study.py   # Variant A vs B: frequencies, participation, figures
├── results/
│   ├── figures/                # PNG figures (geometry, mode shapes, bar chart)
│   ├── modes_variantA/         # XDMF mode shape files for ParaView (Variant A)
│   └── modes_variantB/         # XDMF mode shape files for ParaView (Variant B)
└── report/
    └── report.qmd              # Quarto report template for compiling the minipaper
```

---

## Authors & Course Context
Developed as part of *Project 01 - Open FEM Project with FEniCSx* for the course *11.00153 Modern Simulation Software Development* (Dr. Lambert Theisen & Dr. Georgii Oblapenko), Summer Semester 2026.
