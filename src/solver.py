import numpy as np
import ufl
import dolfinx
import dolfinx.fem
import dolfinx.fem.petsc
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

def solve_modal(mesh, E: float, nu: float, rho: float, num_eigenvalues: int = 6):
    """
    Solves the 3D elasticity generalized eigenvalue problem for a clamped cantilever beam.
    Uses SLEPc to solve K * U = omega^2 * M * U.
    
    Parameters:
    -----------
    mesh : dolfinx.mesh.Mesh
        The FEniCSx mesh.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    rho : float
        Mass density.
    num_eigenvalues : int
        Number of eigenvalues to compute.
        
    Returns:
    --------
    results : list of tuples (freq_hz, phi)
        A list of tuples containing the eigenfrequency in Hz and the corresponding
        eigenmode as a dolfinx.fem.Function.
    V : dolfinx.fem.FunctionSpace
        The vector function space used for the discretization.
    """
    # Create the vector function space V (Lagrange, degree 1)
    if hasattr(dolfinx.fem, "functionspace"):
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
    elif hasattr(dolfinx.fem, "VectorFunctionSpace"):
        V = dolfinx.fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
    else:
        V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
        
    # Find boundary facets at x = 0 (clamped end)
    fdim = mesh.topology.dim - 1
    boundary_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[0], 0.0)
    )
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Dirichlet boundary condition: zero displacement at x = 0
    u_zero = np.zeros(mesh.geometry.dim, dtype=PETSc.ScalarType)
    bc = dolfinx.fem.dirichletbc(u_zero, boundary_dofs, V)
    
    # Lame parameters for isotropic linear elasticity
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    # Strain and stress operators
    def eps(u):
        return ufl.sym(ufl.grad(u))
        
    def sigma(u):
        return 2.0 * mu * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(len(u))
        
    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Bilinear forms for stiffness and mass matrices
    a_form = ufl.inner(sigma(u), eps(v)) * ufl.dx
    m_form = rho * ufl.inner(u, v) * ufl.dx
    
    # Compile forms
    a_compiled = dolfinx.fem.form(a_form)
    m_compiled = dolfinx.fem.form(m_form)
    
    # Assemble stiffness matrix K
    K = dolfinx.fem.petsc.assemble_matrix(a_compiled, bcs=[bc])
    K.assemble()
    
    # Assemble mass matrix M and zero out boundary DOFs
    try:
        M = dolfinx.fem.petsc.assemble_matrix(m_compiled, bcs=[bc], diagonal_value=0.0)
        M.assemble()
    except TypeError:
        # Fallback for environments where diagonal_value is not in assemble_matrix
        M = dolfinx.fem.petsc.assemble_matrix(m_compiled)
        M.assemble()
        if len(boundary_dofs) > 0:
            M.zeroRowsColumns(boundary_dofs, diag=0.0)
            
    # Setup SLEPc EPS solver
    eps_solver = SLEPc.EPS()
    eps_solver.create()
    eps_solver.setOperators(K, M)
    eps_solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    
    # Use shift-and-invert spectral transformation around target = 0.0
    # to find the smallest physical eigenvalues
    st = eps_solver.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    eps_solver.setTarget(0.0)
    eps_solver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    
    # Set tolerances and run options
    eps_solver.setDimensions(num_eigenvalues, SLEPc.DECIDE, SLEPc.DECIDE)
    eps_solver.setFromOptions()
    eps_solver.solve()
    
    # Extract results
    nconv = eps_solver.getConverged()
    results = []
    
    # Pre-create PETSc vectors for eigenvector extraction
    rx, cx = K.getVecs()
    
    for i in range(min(num_eigenvalues, nconv)):
        val_c = eps_solver.getEigenpair(i, rx, cx)
        # Real part is the eigenvalue omega^2
        eigval = val_c.real if isinstance(val_c, complex) else val_c
        
        # Check if the eigenvalue is physical
        if eigval > 1e-5:
            omega = np.sqrt(eigval)
            freq_hz = omega / (2.0 * np.pi)
            
            # Copy vector to dolfinx Function
            phi = dolfinx.fem.Function(V)
            phi.name = f"Mode_{len(results) + 1}"
            if hasattr(phi, "vector"):
                rx.copy(phi.vector)
            else:
                rx.copy(phi.x.petsc_vec)
            phi.x.scatter_forward()
            
            results.append((freq_hz, phi))
            
    # Sort results by frequency (just in case they aren't fully sorted)
    results.sort(key=lambda x: x[0])
    
    return results, V

def compute_modal_participation(mesh, V, phi, rho: float, direction_index: int):
    """
    Computes the modal participation factor and effective mass for a given mode
    shape along a specific coordinate direction.
    
    Parameters:
    -----------
    mesh : dolfinx.mesh.Mesh
        The FEniCSx mesh.
    V : dolfinx.fem.FunctionSpace
        The vector function space.
    phi : dolfinx.fem.Function
        The eigenmode.
    rho : float
        Mass density of the material.
    direction_index : int
        The coordinate direction index: 0 for x (longitudinal), 
        1 for y (transverse lateral), 2 for z (transverse vertical).
        
    Returns:
    --------
    q_i : float
        Modal participation factor.
    m_i : float
        Modal mass.
    m_eff : float
        Effective modal mass.
    """
    # Create the unit displacement vector field in the chosen direction
    u_unit_values = np.zeros(mesh.geometry.dim, dtype=PETSc.ScalarType)
    u_unit_values[direction_index] = 1.0
    u_unit = dolfinx.fem.Constant(mesh, u_unit_values)
    
    # Define integrals using UFL
    # m_i = \int \rho \phi \cdot \phi dx
    m_i_form = dolfinx.fem.form(rho * ufl.dot(phi, phi) * ufl.dx)
    # q_i = \int \rho \phi \cdot u_unit dx
    q_i_form = dolfinx.fem.form(rho * ufl.dot(phi, u_unit) * ufl.dx)
    
    # Assemble local integrals
    m_i_local = dolfinx.fem.assemble_scalar(m_i_form)
    q_i_local = dolfinx.fem.assemble_scalar(q_i_form)
    
    # Sum across MPI processes
    m_i = mesh.comm.allreduce(m_i_local, op=MPI.SUM)
    q_i = mesh.comm.allreduce(q_i_local, op=MPI.SUM)
    
    # Compute effective modal mass
    m_eff = (q_i ** 2) / m_i if m_i > 1e-12 else 0.0
    
    return q_i, m_i, m_eff
