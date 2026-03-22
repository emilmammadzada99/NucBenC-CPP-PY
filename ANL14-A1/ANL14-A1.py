"""
ANL14-A1 Benchmark: 2-Group Neutron Diffusion + Thermal Feedback
================================================================
Pure Python / NumPy / SciPy implementation using the Finite Difference Method (FDM)
on a structured Cartesian grid.
Only: numpy, scipy, matplotlib.

Physics overview
----------------
Two-group neutron diffusion k-eigenvalue problem (steady state):
  -∇·(D_g ∇φ_g) + (Σ_a,g + D_g·B²_z + Σ_s,g→g') φ_g
        = (1/k) χ_g Σ_{g'} ν Σ_f,g' φ_g'
        + Σ_s,g'→g φ_g'    (in-scatter from other groups)

Thermal diffusion (steady state):
  -∇·(k_th ∇T) = q''' = (E_f / k_eff) Σ_g Σ_f,g φ_g

Transient neutron diffusion (implicit Euler, with delayed neutrons):
  (1/v_g) ∂φ_g/∂t = ∇·(D_g ∇φ_g) - removal terms + fission source + precursor source

Transient thermal diffusion (implicit Euler):
  ρ c_p ∂T/∂t = ∇·(k_th ∇T) + q'''

Geometry (ANL14-A1 quarter-core, 165 cm × 165 cm, 2D):
  The domain is mapped to a 165×165 cm square.
  Regions are assigned analytically based on coordinate ranges extracted from the .geo file.
  Symmetry BC (zero-flux gradient) on the left and bottom edges.
  Vacuum (zero-flux Dirichlet) on the right and top edges.

Author: auto-generated from the original dolfinx code
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # non-interactive backend – change to 'TkAgg' for interactive use
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
import time
from scipy.sparse import lil_matrix, csr_matrix, block_diag
from scipy.sparse.linalg import spsolve, gmres, LinearOperator

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GRID DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

def build_grid(Lx=165., Ly=165., nx=165, ny=165):
    """
    Build a uniform Cartesian grid over [0, Lx] × [0, Ly].

    The ANL14-A1 benchmark domain is a 165 cm × 165 cm quarter-core.
    Key material boundaries are at x,y = 0, 15, 75, 105, 120, 135, 165 cm.
    We pick nx=ny so that grid lines fall on all those boundaries.

    Returns
    -------
    x, y   : 1D node coordinate arrays (length nx+1, ny+1 respectively)
    X, Y   : 2D meshgrid arrays  shape (ny+1, nx+1)
    dx, dy : uniform spacing
    """
    x = np.linspace(0., Lx, nx + 1)
    y = np.linspace(0., Ly, ny + 1)
    X, Y = np.meshgrid(x, y)            # shape (ny+1, nx+1)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return x, y, X, Y, dx, dy


# ─────────────────────────────────────────────────────────────────────────────
# 2.  REGION MAP  (analytical assignment from the .geo file)
# ─────────────────────────────────────────────────────────────────────────────

def build_region_map(X, Y):
    """
    Assign a region index (0-5) to every grid node based on (x, y) coordinates.

    ANL14-A1 quarter-core regions (from the .geo Physical Surface definitions):
        Index 0  →  reg-1  (marker 10) : fuel-1 – large central fuel zone
        Index 1  →  reg-2  (marker 20) : fuel-2 – corner/reflector fuel pieces
        Index 2  →  reg-3  (marker 30) : fuel-3 – reflector fuel strips
        Index 3  →  CR     (marker 35) : control rod
        Index 4  →  reg-4  (marker 40) : small corner region near CR
        Index 5  →  reg-5  (marker 50) : outer reflector / void region

    The mapping below reproduces the surfaces defined in ANL14-A1.geo as closely
    as possible on a regular Cartesian grid.
    """
    ny1, nx1 = X.shape
    region = np.full((ny1, nx1), 5, dtype=int)   # default: outer reflector (reg-5)

    x = X[0, :]          # 1D x-coords
    y = Y[:, 0]          # 1D y-coords

    # Convenience boolean 2D masks based on coordinate ranges from .geo
    # All dimensions in cm.

    # --- reg-5 (outer reflector): 
    #     Covers the L-shaped region outside the inner 135×135 core.
    #     Also the triangular corner cut: x+y > 165+15 (approx to match .geo shape)
    # Default = reg-5, so we overwrite inward regions below.

    # --- reg-1 (central fuel block):  15≤x≤105, 15≤y≤105  MINUS the small pieces
    m_reg1 = (X >= 15) & (X <= 105) & (Y >= 15) & (Y <= 105)
    # But sub-regions 1,2,3,4 are carved out of the big central area
    # Surface(1): 0≤x≤15,  0≤y≤15    → reg-2
    # Surface(2): 75≤x≤105,0≤y≤15    → reg-2
    # Surface(3): 0≤x≤15, 75≤y≤105   → reg-2
    # Surface(4): 75≤x≤105,75≤y≤105  → reg-2
    m_s1 = (X >= 0)  & (X <= 15)  & (Y >= 0)  & (Y <= 15)
    m_s2 = (X >= 75) & (X <= 105) & (Y >= 0)  & (Y <= 15)
    m_s3 = (X >= 0)  & (X <= 15)  & (Y >= 75) & (Y <= 105)
    m_s4 = (X >= 75) & (X <= 105) & (Y >= 75) & (Y <= 105)
    # Surface(5): 15≤x≤75, 15≤y≤75 + edges bordering the corner pieces → reg-1
    # Surface(6): 105≤x≤135, 0≤y≤75  → reg-3 (partially)
    m_s6 = (X >= 105) & (X <= 135) & (Y >= 0) & (Y <= 75)
    # Surface(7): 105≤x≤135, 75≤y≤105 → CR
    m_cr = (X >= 105) & (X <= 135) & (Y >= 75) & (Y <= 105)
    # Surface(8): 0≤x≤105, 105≤y≤135 → reg-3
    m_s8 = (X >= 0) & (X <= 105) & (Y >= 105) & (Y <= 135)
    # Surface(9): 105≤x≤120, 105≤y≤120 → reg-4 (small piece)
    m_reg4 = (X >= 105) & (X <= 120) & (Y >= 105) & (Y <= 120)
    # Outer reflector occupies everything from 135 to 165 (or 120 to 165 for the cut corner)
    # We keep region=5 as default and overwrite with inner materials.

    # Now assign in order (later assignments take priority)
    region[m_reg1]  = 0   # reg-1 (central fuel)
    region[m_s1]    = 1   # reg-2 (fuel corners)
    region[m_s2]    = 1
    region[m_s3]    = 1
    region[m_s4]    = 1
    region[m_s6]    = 2   # reg-3 (right fuel strip)
    region[m_s8]    = 2   # reg-3 (top fuel strip)
    region[m_cr]    = 3   # CR
    region[m_reg4]  = 4   # reg-4 (small corner piece)
    # Surface(10): outer L-shaped reflector – already 5 by default

    # Zero-flux (vacuum) mask: outside the L-shaped boundary of the physical domain.
    # The real boundary is a polygon: the outer wall makes a stair-step from (165,0)
    # up to (165,165) and (0,165), with a cut corner at (120,105)→(105,120) approximately.
    # For simplicity we treat the full 165×165 square as active domain with Dirichlet=0
    # on the outer boundary, which is the standard approximation for this benchmark.
    outside = (X > 165.) | (Y > 165.)
    region[outside] = -1   # inactive / outside

    return region


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MATERIAL CROSS SECTIONS  (log-coupling with temperature)
# ─────────────────────────────────────────────────────────────────────────────

# Reference values from test1.py  (6 regions, 2 energy groups)
# First index: energy group (0=fast, 1=thermal)
# Second index: [0]=base value, [1]=log-coupling coefficient
# Third index: region (0..5)

TREF = 600.   # K  – reference temperature

# Diffusion coefficient  D_g[g][coupling][region]
D_BASE = np.array([
    [1.255,  1.268,  1.259,  1.259,  1.259,  1.257],   # group 0 (fast)
    [0.211,  0.1902, 0.2091, 0.2091, 0.2091, 0.1592],  # group 1 (thermal)
])
D_LOG = np.array([
    [5e-4] * 6,    # fast
    [2.5e-3] * 6,  # thermal
])

# Absorption XS  xs_a
XSA_BASE = np.array([
    [0.008252, 0.007181, 0.008002, 0.008002, 0.008002, 0.0006034],
    [0.1003,   0.07047,  0.08344,  0.08344,  0.073324, 0.01911],
])
XSA_LOG = np.array([
    [7.5e-4] * 6,
    [1e-3] * 6,
])

# nu * fission XS
NUXSF_BASE = np.array([
    [0.004602, 0.004609, 0.004663, 0.004663, 0.004663, 0.],
    [0.1091,   0.08675,  0.1021,   0.1021,   0.1021,   0.],
])
NUXSF_LOG = np.array([
    [0.] * 6,
    [0.] * 6,
])

# Fission XS  xs_f
XSF_BASE = np.array([
    [0.001894, 0.001897, 0.001919, 0.001919, 0.001919, 0.],
    [0.044897, 0.035700, 0.042016, 0.042016, 0.042016, 0.],
])
XSF_LOG = np.array([
    [0.] * 6,
    [0.] * 6,
])

# Scattering XS xs_s[g_from][g_to][region]  (only fast→thermal is non-zero)
XSS_12_BASE = np.array([0.02533, 0.02767, 0.02617, 0.02617, 0.02617, 0.04754])  # fast→thermal
XSS_12_LOG  = np.array([0.] * 6)

# Axial buckling B²_z
B2Z = np.array([1e-4] * 6)   # same for both groups and all regions

# Fission spectrum  χ_g[g][region]  – chi[0]=1 for fast, chi[1]=0 for thermal
CHI = np.array([
    [1.] * 6,   # fast group receives all fission neutrons
    [0.] * 6,   # thermal group
])

# Kinetic parameters
V_NEUTRON = np.array([3e7, 3e5])    # cm/s  (fast, thermal)
BETA_L = np.array([0.0054, 0.001087])    # delayed neutron fractions  (2 precursor groups)
LAMBDA_L = np.array([0.0654, 1.35])      # decay constants  1/s

BETA_TOTAL = np.sum(BETA_L)   # total delayed fraction

# Thermal properties
TH_COND   = np.array([5.,  0.5,  2.,  0.5,  0.1,  10.])   # W/cm/K per region
RHO_CP    = np.array([1./1.1954] * 6) / 2500.              # J/cm³/K (Brega et al.)

# Physical constants
NU   = 2.43          # neutrons per fission
EF   = 200e6 * 1.6e-19   # J per fission
P0   = 6000.         # reactor power (arbitrary units / cm, consistent with test1.py)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CROSS-SECTION EVALUATION  (with log coupling)
# ─────────────────────────────────────────────────────────────────────────────

def get_xs(region_map, T_field, coupling='log'):
    """
    Return cross-section arrays over the whole grid using the 'log' temperature coupling.

    D_g  = D_base + D_log * ln(T/T_ref)
    etc.

    Parameters
    ----------
    region_map : 2D int array    – region index (0–5) at each node
    T_field    : 2D float array  – temperature [K] at each node
    coupling   : 'log' or None

    Returns
    -------
    Dictionary of 2D arrays for each cross section:
    D[g], xsa[g], nuxsf[g], xsf[g], xss12, chi[g], b2z
    """
    ny1, nx1 = region_map.shape
    G = 2

    D     = [np.zeros((ny1, nx1)) for _ in range(G)]
    xsa   = [np.zeros((ny1, nx1)) for _ in range(G)]
    nuxsf = [np.zeros((ny1, nx1)) for _ in range(G)]
    xsf   = [np.zeros((ny1, nx1)) for _ in range(G)]
    chi   = [np.zeros((ny1, nx1)) for _ in range(G)]
    xss12 = np.zeros((ny1, nx1))   # fast → thermal scattering
    b2z   = np.zeros((ny1, nx1))

    for r in range(6):
        mask = (region_map == r)
        if not np.any(mask):
            continue

        if coupling == 'log':
            # Avoid log(0) by clipping temperature
            T_clipped = np.clip(T_field[mask], 1., None)
            log_ratio = np.log(T_clipped / TREF)
        else:
            log_ratio = 0.

        for g in range(G):
            D[g][mask]     = D_BASE[g, r]     + D_LOG[g, r]     * log_ratio
            xsa[g][mask]   = XSA_BASE[g, r]   + XSA_LOG[g, r]   * log_ratio
            nuxsf[g][mask] = NUXSF_BASE[g, r] + NUXSF_LOG[g, r] * log_ratio
            xsf[g][mask]   = XSF_BASE[g, r]   + XSF_LOG[g, r]   * log_ratio
            chi[g][mask]   = CHI[g, r]

        xss12[mask] = XSS_12_BASE[r] + XSS_12_LOG[r] * log_ratio
        b2z[mask]   = B2Z[r]

    return dict(D=D, xsa=xsa, nuxsf=nuxsf, xsf=xsf, chi=chi, xss12=xss12, b2z=b2z)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  BOUNDARY CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_bc_mask(region_map):
    """
    Return boolean masks for:
      - dirichlet_bc : outer boundary nodes (vacuum BC, φ=0)
      - interior     : all other active nodes
    """
    ny1, nx1 = region_map.shape

    # Outer boundary: top row, right column, plus bottom and left for the Dirichlet edges.
    # The ANL14-A1 benchmark has:
    #   symmetry BC (zero flux gradient / Neumann zero) on left (x=0) and bottom (y=0)
    #   vacuum BC (Dirichlet φ=0) on right (x=165) and top (y=165)
    # In FDM: Neumann zero = natural BC (no extra treatment needed with central differences)
    # Dirichlet φ=0 applied on top row and right column.

    dirichlet = np.zeros((ny1, nx1), dtype=bool)
    dirichlet[-1, :] = True    # top edge  (y = 165)
    dirichlet[:, -1] = True    # right edge (x = 165)

    return dirichlet


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FDM MATRIX ASSEMBLY  (2-group coupled neutron diffusion)
# ─────────────────────────────────────────────────────────────────────────────

def assemble_neutron_matrix(xs, dx, dy, region_map, dirichlet_mask):
    """
    Assemble the 2-group coupled FDM matrix for the neutron diffusion eigenvalue problem.

    Equation for group g at interior node (i,j):
      -D_g * Δφ_g  +  (Σ_a,g + D_g·B²_z + Σ_s,g→other) φ_g
          = (1/k) χ_g Σ_{g'} ν·Σ_f,g' φ_g'
          + Σ_s,other→g φ_other

    Using standard 5-point Laplacian (central differences):
      -D_{i,j} * [ (φ_{i+1,j} - 2φ_{i,j} + φ_{i-1,j})/dx²
                 + (φ_{i,j+1} - 2φ_{i,j} + φ_{i,j-1})/dy² ]

    We use the harmonic mean of D at cell interfaces to handle discontinuities.

    The matrix is assembled in the block form:
      [A11  A12] [φ1]   1   [F11  F12] [φ1]
      [A21  A22] [φ2] = - * [F21  F22] [φ2]
                        k

    where A = loss matrix, F = fission matrix.

    Returns
    -------
    A     : (2N × 2N) sparse loss matrix (CSR)
    F     : (2N × 2N) sparse fission matrix (CSR)
    dofs  : 1D array mapping node index → linear DOF index (-1 if Dirichlet)
    N     : number of free DOFs per group
    """
    ny1, nx1 = region_map.shape
    n_total = ny1 * nx1

    # ---- Build DOF numbering ------------------------------------------------
    # Interior active nodes get a free DOF; Dirichlet nodes get -1.
    dofs = -np.ones(n_total, dtype=int)
    free = (~dirichlet_mask).ravel()
    # Also exclude nodes outside the domain (region == -1)
    active = free & (region_map.ravel() >= 0)
    dofs[active] = np.arange(np.sum(active))
    N = int(np.sum(active))   # free DOFs per group

    # ---- Harmonic mean diffusion at interfaces -------------------------------
    D0 = xs['D'][0]   # fast group  shape (ny1, nx1)
    D1 = xs['D'][1]   # thermal

    def harmonic_mean_x(D, i, j):
        """D at right interface of (i,j)"""
        if j + 1 < nx1:
            return 2. * D[i, j] * D[i, j+1] / (D[i, j] + D[i, j+1] + 1e-30)
        return D[i, j]

    def harmonic_mean_y(D, i, j):
        """D at top interface of (i,j)"""
        if i + 1 < ny1:
            return 2. * D[i, j] * D[i+1, j] / (D[i, j] + D[i+1, j] + 1e-30)
        return D[i, j]

    # ---- Fill sparse matrices -----------------------------------------------
    # Size: 2N × 2N  (block: group 0 lives in rows 0..N-1, group 1 in N..2N-1)
    A_sp = lil_matrix((2 * N, 2 * N))
    F_sp = lil_matrix((2 * N, 2 * N))

    def flat(i, j):
        return i * nx1 + j

    for i in range(ny1):
        for j in range(nx1):
            idx = flat(i, j)
            if dofs[idx] < 0:
                continue  # Dirichlet or outside

            row_g0 = dofs[idx]          # row for group 0
            row_g1 = dofs[idx] + N      # row for group 1

            for g, row in enumerate([row_g0, row_g1]):
                D_here = xs['D'][g][i, j]
                b2     = xs['b2z'][i, j]
                xsa_g  = xs['xsa'][g][i, j]

                # --- Diagonal: absorption + buckling + out-scatter + diffusion diagonal ---
                # Diffusion contributions from the 4 neighbours
                Dx_r = harmonic_mean_x(xs['D'][g], i, j)      if j < nx1-1 else 0.
                Dx_l = harmonic_mean_x(xs['D'][g], i, j-1)    if j > 0     else 0.
                Dy_u = harmonic_mean_y(xs['D'][g], i, j)      if i < ny1-1 else 0.
                Dy_d = harmonic_mean_y(xs['D'][g], i-1, j)    if i > 0     else 0.

                # Symmetry BC (Neumann zero) at left and bottom: ghost node = interior node
                # → the coefficient for the missing neighbour is effectively zero
                # (the ghost value equals the boundary value, so the difference is zero)
                # Actually with zero-gradient Neumann: φ_{-1} = φ_0 → finite-diff term vanishes
                # That means Dx_l = 0 at j=0, Dy_d = 0 at i=0 (already handled above)

                diag = (Dx_r + Dx_l) / dx**2 + (Dy_u + Dy_d) / dy**2
                diag += xsa_g + D_here * b2

                # Out-scatter (from group g to the other group)
                if g == 0:
                    diag += xs['xss12'][i, j]   # fast → thermal out-scatter

                A_sp[row, row] += diag

                # --- Off-diagonal diffusion terms (neighbours) ---
                if j < nx1 - 1:
                    col_right = dofs[flat(i, j+1)]
                    if col_right >= 0:
                        A_sp[row, col_right + (0 if g == 0 else N)] -= Dx_r / dx**2
                    # else: Dirichlet φ=0, no contribution to RHS

                if j > 0:
                    col_left = dofs[flat(i, j-1)]
                    if col_left >= 0:
                        A_sp[row, col_left + (0 if g == 0 else N)] -= Dx_l / dx**2

                if i < ny1 - 1:
                    col_up = dofs[flat(i+1, j)]
                    if col_up >= 0:
                        A_sp[row, col_up + (0 if g == 0 else N)] -= Dy_u / dy**2

                if i > 0:
                    col_down = dofs[flat(i-1, j)]
                    if col_down >= 0:
                        A_sp[row, col_down + (0 if g == 0 else N)] -= Dy_d / dy**2

                # --- In-scatter from the other group ---
                other = 1 - g
                other_row = row_g1 if g == 0 else row_g0
                other_col = dofs[idx] + (N if other == 1 else 0)
                if g == 1:
                    # thermal group receives scatter from fast group
                    A_sp[row, other_col] -= xs['xss12'][i, j]

            # --- Fission matrix (both groups contribute) ---
            for g in range(2):
                row_fiss = dofs[idx] + (0 if g == 0 else N)
                for gp in range(2):
                    col_fiss = dofs[idx] + (0 if gp == 0 else N)
                    F_sp[row_fiss, col_fiss] += xs['chi'][g][i, j] * xs['nuxsf'][gp][i, j]

    return A_sp.tocsr(), F_sp.tocsr(), dofs, N


# ─────────────────────────────────────────────────────────────────────────────
# 7.  POWER ITERATION (inverse power method for k-eigenvalue)
# ─────────────────────────────────────────────────────────────────────────────

def power_iteration(A, F, N, tol=1e-8, max_iter=500, verbose=True, print_every=50):
    """
    Solve the generalised eigenvalue problem  A φ = (1/k) F φ
    using the inverse power method (power iteration on A⁻¹ F).

    Parameters
    ----------
    A, F   : (2N × 2N) CSR sparse matrices
    N      : DOFs per group
    tol    : convergence tolerance on relative k_eff change
    max_iter : maximum iterations
    verbose  : print convergence info

    Returns
    -------
    phi_vec : (2N,) eigenvector
    k_eff   : eigenvalue (float)
    """
    # Initial guess: uniform non-zero flux
    phi = np.ones(2 * N)

    k_eff = 1.0
    k_eff_old = 0.0

    for iteration in range(max_iter):
        # Compute fission source: S = F φ / k
        S = F.dot(phi) / k_eff

        # Solve  A φ_new = S
        phi_new = spsolve(A, S)

        # Recompute k_eff using the Rayleigh quotient:
        #   k = <F φ_new, φ_new> / <A φ_new, φ_new>
        Fphi = F.dot(phi_new)
        Aphi = A.dot(phi_new)
        k_eff_new = np.dot(Fphi, phi_new) / np.dot(Aphi, phi_new)

        # Normalise eigenvector
        phi_new /= np.max(np.abs(phi_new))

        rel_err = abs(k_eff_new - k_eff) / abs(k_eff_new)

        if verbose and (iteration % print_every == 0):
            print(f"  Power iter {iteration:4d} | k_eff = {k_eff_new:.7f} | "
                  f"rel_err = {rel_err:.3e}")

        phi    = phi_new
        k_eff  = k_eff_new

        if rel_err < tol and iteration > 2:
            if verbose:
                print(f"  Converged at iter {iteration} | k_eff = {k_eff:.8f} | "
                      f"rel_err = {rel_err:.3e}")
            break

    return phi, k_eff


def normalise_flux(phi_vec, xs, dofs, N, ny1, nx1, power, k_eff):
    """
    Normalise flux so that the reactor power equals `power`.

    P = (Ef / (ν · k_eff)) Σ_g Σ_{i,j} ν·Σ_f,g(i,j) · φ_g(i,j) · dx·dy

    (The ν cancels: P = Ef/k_eff * integral of Σ_f,g φ_g)
    """
    flat_dofs = dofs.reshape(ny1, nx1)

    # Reconstruct 2D flux arrays
    phi0 = np.zeros((ny1, nx1))
    phi1 = np.zeros((ny1, nx1))

    for i in range(ny1):
        for j in range(nx1):
            d = flat_dofs[i, j]
            if d >= 0:
                phi0[i, j] = phi_vec[d]
                phi1[i, j] = phi_vec[d + N]

    # Compute unnormalised power integral (trapezoidal rule approximation = cell-centre sum)
    # P_raw = Ef/k_eff * sum over nodes of xs_f_g * phi_g * cell_area
    # Here we use a simple rectangular integration (node-centred)
    dy_g = (165. / ny1)  # approximate cell height
    dx_g = (165. / nx1)  # approximate cell width
    cell_area = dx_g * dy_g

    P_raw = 0.
    for g, phi_g in enumerate([phi0, phi1]):
        P_raw += np.sum(xs['xsf'][g] * phi_g) * cell_area

    P_raw *= EF / k_eff

    # Scale factor
    scale = power / (P_raw + 1e-30)

    phi0 *= scale
    phi1 *= scale

    return phi0, phi1


# ─────────────────────────────────────────────────────────────────────────────
# 8.  THERMAL DIFFUSION MATRIX ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def assemble_thermal_matrix(th_cond_map, dx, dy, dirichlet_mask, region_map, T_dirichlet=300.):
    """
    Assemble the FDM matrix for steady thermal diffusion:
      -∇·(k_th ∇T) = q'''

    Dirichlet BC: T = T_dirichlet on vacuum boundary.
    Neumann zero BC (symmetry) on left/bottom.

    Returns
    -------
    A_th   : (M × M) sparse CSR matrix (M = free thermal DOFs)
    dofs_T : 1D DOF map
    M      : number of free thermal DOFs
    rhs_bc : contribution from Dirichlet boundary to RHS
    """
    ny1, nx1 = region_map.shape
    n_total = ny1 * nx1

    dofs_T = -np.ones(n_total, dtype=int)
    free = (~dirichlet_mask).ravel() & (region_map.ravel() >= 0)
    dofs_T[free] = np.arange(np.sum(free))
    M = int(np.sum(free))

    A_th = lil_matrix((M, M))
    rhs_bc = np.zeros(M)   # Dirichlet contributions

    def flat(i, j):
        return i * nx1 + j

    def harmonic_k(K, i, j, direction):
        if direction == 'r' and j < nx1 - 1:
            k1, k2 = K[i, j], K[i, j+1]
        elif direction == 'l' and j > 0:
            k1, k2 = K[i, j-1], K[i, j]
        elif direction == 'u' and i < ny1 - 1:
            k1, k2 = K[i, j], K[i+1, j]
        elif direction == 'd' and i > 0:
            k1, k2 = K[i-1, j], K[i, j]
        else:
            return 0.
        return 2. * k1 * k2 / (k1 + k2 + 1e-30)

    for i in range(ny1):
        for j in range(nx1):
            idx = flat(i, j)
            d   = dofs_T[idx]
            if d < 0:
                continue

            kr = harmonic_k(th_cond_map, i, j, 'r')
            kl = harmonic_k(th_cond_map, i, j, 'l')
            ku = harmonic_k(th_cond_map, i, j, 'u')
            kd = harmonic_k(th_cond_map, i, j, 'd')

            diag = (kr + kl) / dx**2 + (ku + kd) / dy**2
            A_th[d, d] += diag

            for (di, dj, k_intf, h2) in [
                (0,  1, kr, dx**2),
                (0, -1, kl, dx**2),
                (1,  0, ku, dy**2),
                (-1, 0, kd, dy**2),
            ]:
                ni, nj = i + di, j + dj
                if 0 <= ni < ny1 and 0 <= nj < nx1:
                    nb_idx = flat(ni, nj)
                    nb_d   = dofs_T[nb_idx]
                    if nb_d >= 0:
                        A_th[d, nb_d] -= k_intf / h2
                    elif dirichlet_mask[ni, nj]:
                        # Dirichlet neighbour: contributes to RHS
                        rhs_bc[d] += k_intf / h2 * T_dirichlet

    return A_th.tocsr(), dofs_T, M, rhs_bc


def solve_thermal(phi0, phi1, xs, th_cond_map, dx, dy,
                  dirichlet_mask, region_map, k_eff, T_dirichlet=300.):
    """
    Solve the steady thermal diffusion equation given flux fields phi0, phi1.

    Returns T : 2D temperature array (K)
    """
    ny1, nx1 = region_map.shape

    # Build volumetric heat source: q''' = (Ef/k_eff) Σ_g Σ_f,g φ_g
    q3 = (EF / k_eff) * (xs['xsf'][0] * phi0 + xs['xsf'][1] * phi1)

    A_th, dofs_T, M, rhs_bc = assemble_thermal_matrix(
        th_cond_map, dx, dy, dirichlet_mask, region_map, T_dirichlet)

    # Assemble RHS = q''' + BC contributions
    rhs = np.zeros(M)
    for i in range(ny1):
        for j in range(nx1):
            d = dofs_T[i * nx1 + j]
            if d >= 0:
                rhs[d] += q3[i, j]

    rhs += rhs_bc

    # Solve linear system
    T_vec = spsolve(A_th, rhs)

    # Reconstruct 2D array
    T = np.full((ny1, nx1), T_dirichlet)
    for i in range(ny1):
        for j in range(nx1):
            d = dofs_T[i * nx1 + j]
            if d >= 0:
                T[i, j] = T_vec[d]

    return T


# ─────────────────────────────────────────────────────────────────────────────
# 9.  STEADY-STATE COUPLED SOLVER  (neutronics + thermal, outer iteration)
# ─────────────────────────────────────────────────────────────────────────────

def steady_coupled_solve(x, y, X, Y, region_map, dirichlet_mask,
                         tol_outer=1e-4, max_outer=20, verbose=True):
    """
    Outer fixed-point iteration to converge neutronics and thermal fields.

    1. Solve neutron diffusion with current temperature.
    2. Solve thermal with resulting flux.
    3. Repeat until temperature and heat source converge.

    Returns
    -------
    phi0, phi1 : fast and thermal flux arrays (normalised to P0)
    T          : temperature array [K]
    k_eff      : converged k_eff
    k_eff_uncoupled : k_eff from neutronics only (first iteration)
    """
    ny1, nx1 = X.shape
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Thermal conductivity map (constant, no T-coupling for k_th in this problem)
    th_cond_map = np.zeros((ny1, nx1))
    for r in range(6):
        th_cond_map[region_map == r] = TH_COND[r]

    # Initial temperature guess
    T_guess  = np.full((ny1, nx1), TREF)
    q3_guess = np.zeros((ny1, nx1))

    k_eff_uncoupled = None
    k_eff = 1.0

    print("=" * 60)
    print("Starting steady-state coupled solve")
    print("=" * 60)

    for outer in range(max_outer):

        # ---- Step A: neutron diffusion ----
        xs = get_xs(region_map, T_guess, coupling='log')

        A, F, dofs, N = assemble_neutron_matrix(xs, dx, dy, region_map, dirichlet_mask)

        phi_vec, k_eff = power_iteration(A, F, N,
                                         tol=1e-9, max_iter=600,
                                         verbose=(outer == 0),
                                         print_every=100)

        phi0, phi1 = normalise_flux(phi_vec, xs, dofs, N, ny1, nx1, P0, k_eff)

        if outer == 0:
            k_eff_uncoupled = k_eff

        # ---- Step B: thermal diffusion ----
        T_new = solve_thermal(phi0, phi1, xs, th_cond_map, dx, dy,
                              dirichlet_mask, region_map, k_eff, T_dirichlet=300.)

        # ---- Convergence check ----
        q3_new = (EF / k_eff) * (xs['xsf'][0] * phi0 + xs['xsf'][1] * phi1)

        dT  = np.linalg.norm(T_new - T_guess) / (np.linalg.norm(T_guess) + 1e-30)

        q3_norm = np.linalg.norm(q3_guess)
        dq3 = np.linalg.norm(q3_new - q3_guess) / (q3_norm + 1e-30) if q3_norm > 1e-10 else 0.

        err = dT + dq3

        print(f"Outer iter {outer+1:3d} | k_eff = {k_eff:.7f} | "
              f"err_T = {dT:.3e} | err_q3 = {dq3:.3e} | err = {err:.3e}")

        T_guess  = T_new.copy()
        q3_guess = q3_new.copy()

        if err < tol_outer and outer > 0:
            print(f"\n  Converged in {outer+1} iterations")
            break

    print(f"\nk_eff uncoupled : {k_eff_uncoupled:.6f}")
    print(f"k_eff   coupled : {k_eff:.6f}")
    drho = abs(k_eff - k_eff_uncoupled) / k_eff_uncoupled * 1e5
    print(f"Reactivity diff : {drho:.3f} pcm")

    return phi0, phi1, T_new, k_eff, k_eff_uncoupled
# ─────────────────────────────────────────────────────────────────────────────
# 11.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_steady_state(X, Y, phi0, phi1, T, region_map, filename='steady_state.png'):
    """
    Plot fast flux, thermal flux, and temperature side-by-side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Mask outside-domain nodes
    outside = (region_map < 0)
    phi0_plot = np.where(outside, np.nan, phi0)
    phi1_plot = np.where(outside, np.nan, phi1)
    T_plot    = np.where(outside, np.nan, T)

    data_list = [phi0_plot, phi1_plot, T_plot]
    titles = ['Fast Flux φ₁ (cm⁻²s⁻¹)', 'Thermal Flux φ₂ (cm⁻²s⁻¹)', 'Temperature T (K)']
    cmaps  = [cm.plasma, cm.plasma, cm.hot]

    for ax, data, title, cmap in zip(axes, data_list, titles, cmaps):
        img = ax.pcolormesh(X, Y, data, cmap=cmap, shading='auto')
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_aspect('equal')

    plt.suptitle('ANL14-A1 Steady-State Results', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_region_map(X, Y, region_map, filename='region_map.png'):
    """
    Plot the material region map for visual verification.
    """
    labels = ['reg-1 (fuel-1)', 'reg-2 (fuel-2)', 'reg-3 (fuel-3)',
              'CR', 'reg-4', 'reg-5 (reflector)']
    cmap_r = plt.cm.get_cmap('tab10', 7)

    rmap_plot = np.where(region_map < 0, np.nan, region_map.astype(float))

    fig, ax = plt.subplots(figsize=(7, 7))
    img = ax.pcolormesh(X, Y, rmap_plot, cmap=cmap_r, vmin=-0.5, vmax=5.5, shading='auto')
    cbar = fig.colorbar(img, ax=ax, ticks=range(6))
    cbar.ax.set_yticklabels(labels, fontsize=9)
    ax.set_title('ANL14-A1 Region Map', fontsize=13)
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")
# ─────────────────────────────────────────────────────────────────────────────
# 12.  MAIN DRIVER
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("ANL14-A1 — Pure Python FDM Solver")
    print("=" * 60)

    # ── Grid ────────────────────────────────────────────────────────────────
    # 66×66 gives grid lines at multiples of 2.5 cm, covering the main boundaries.
    # Increase for accuracy (82×82 aligns exactly with 15, 75, 105, 120, 135 cm).
    nx = ny = 165
    x, y, X, Y, dx, dy = build_grid(Lx=165., Ly=165., nx=nx, ny=ny)
    print(f"Grid: {nx+1}×{ny+1} nodes, dx = {dx:.3f} cm, dy = {dy:.3f} cm")

    # ── Region map ──────────────────────────────────────────────────────────
    region_map = build_region_map(X, Y)
    dirichlet_mask = get_bc_mask(region_map)
    print(f"Active nodes: {np.sum(region_map >= 0)}")

    # ── Plot region map ──────────────────────────────────────────────────────
    plot_region_map(X, Y, region_map, filename='region_map.png')

    # ── Steady-state coupled solve ───────────────────────────────────────────
    t0 = time.time()
    phi0_ss, phi1_ss, T_ss, k_eff, k_eff_uncoupled = steady_coupled_solve(
        x, y, X, Y, region_map, dirichlet_mask,
        tol_outer=1e-4, max_outer=15, verbose=True)

    print(f"\nSteady-state solve time: {time.time()-t0:.1f} s")

    # ── Plot steady state ────────────────────────────────────────────────────
    plot_steady_state(X, Y, phi0_ss, phi1_ss, T_ss, region_map,
                      filename='steady_state.png')
    print("\n" + "=" * 60)
    print("All done....")
    print("=" * 60)


if __name__ == '__main__':
    main()
 