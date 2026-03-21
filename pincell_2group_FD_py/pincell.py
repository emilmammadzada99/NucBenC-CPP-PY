import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# =============================================================================
# SHARED PARAMETERS
# =============================================================================
r_fuel = 0.39       # Fuel outer radius (cm)
r_clad_inner = 0.40 # Cladding inner radius (cm)
r_clad_outer = 0.46 # Cladding outer radius (cm)
pitch = 1.26        # Cell pitch (cm)
bc = 'reflective'
G = 2               # Number of energy groups

materials = {
    0: {  # Water (moderator)
        'name': 'Water',
        'SigmaT': np.array([6.40711e-1, 1.69131e-0]),
        'Chi':    np.array([1.0, 0.0]),  # Fission spectrum (not used for non-fissile)
        'SigmaS': np.array([
            [6.07382e-1, 0.0],
            [3.31316e-2, 1.68428e-0]
        ]),
        'NuSigF': np.zeros(2)
    },
    1: {  # Pin (fuel)
        'name': 'Pin',
        'SigmaT': np.array([3.62022e-1, 5.72155e-1]),
        'Chi':    np.array([1.0, 0.0]),  # All fission neutrons in fast group
        'SigmaS': np.array([
            [3.33748e-1, 0.0],
            [6.64881e-4, 3.80898e-1]
        ]),
        'NuSigF': np.array([1.86278e-2, 3.44137e-1])
    },
    2: {  # Cladding
        'name': 'Cladding',
        'SigmaT': np.array([2.74144e-1, 2.80890e-1]),
        'Chi':    np.array([1.0, 0.0]),  # Fission spectrum (not used for non-fissile)
        'SigmaS': np.array([
            [2.72377e-1, 0.0],
            [1.90838e-4, 2.77230e-1]
        ]),
        'NuSigF': np.zeros(2)
    }
}

# =============================================================================
# MESH SETUP
# =============================================================================
def build_mesh(N):
    """
    Build a uniform NxN mesh over [-pitch/2, pitch/2] x [-pitch/2, pitch/2].
    Assign material index to each cell based on distance from center:
    - Material 1 (Pin/Fuel): r <= r_fuel
    - Material 2 (Cladding): r_clad_inner < r <= r_clad_outer
    - Material 0 (Water): r > r_clad_outer
    
    Returns cell centers, cell width, and material map.
    """
    half = pitch / 2.0
    dx = pitch / N
    x = np.linspace(-half + dx/2, half - dx/2, N)
    y = np.linspace(-half + dx/2, half - dx/2, N)
    X, Y = np.meshgrid(x, y)
    
    # Calculate distance from center for each cell
    R = np.sqrt(X **2 + Y** 2)
    
    # Initialize with water (material 0)
    mat_map = np.zeros((N, N), dtype=int)
    
    # Assign fuel (material 1)
    mat_map[R <= r_fuel] = 1
    
    # Assign cladding (material 2)
    mat_map[(R > r_clad_inner) & (R <= r_clad_outer)] = 2
    
    return x, y, dx, mat_map

# =============================================================================
# GLOBAL MATRIX ASSEMBLY (DIFFUSION + SCATTERING) AND FISSION MATRIX
# =============================================================================
def build_global_matrices(mat_map, dx, N):
    """
    Assemble the global matrices M and F for the multi-group diffusion equation.
    M includes: diffusion, removal, and inter-group scattering (excluding self-scattering).
    F includes: fission production terms.
    The global ordering is: index = g * Nc + j * N + i, where Nc = N*N.
    """
    Nc = N * N
    size = G * Nc
    rows_M, cols_M, vals_M = [], [], []
    rows_F, cols_F, vals_F = [], [], []

    # Precompute material properties per cell
    D = np.zeros((G, N, N))
    SigR = np.zeros((G, N, N))  # Removal = SigmaT - SigmaS[g,g]
    NuSigF = np.zeros((G, N, N))
    Chi = np.zeros((G, N, N))

    for j in range(N):
        for i in range(N):
            m = mat_map[j, i]
            mat = materials[m]
            for g in range(G):
                # Transport correction
                sig_tr = mat['SigmaT'][g] - mat['SigmaS'][g, g]
                D[g, j, i] = 1.0 / (3.0 * sig_tr)
                SigR[g, j, i] = sig_tr
                NuSigF[g, j, i] = mat['NuSigF'][g]
                Chi[g, j, i] = mat['Chi'][g]

    # Build matrices
    for g in range(G):
        for j in range(N):
            for i in range(N):
                idx = g * Nc + j * N + i

                # Diagonal term: removal + diffusion neighbor contributions
                diag_val = SigR[g, j, i] * dx * dx

                # Diffusion in x-direction
                if i > 0:
                    D_w = 2.0 * D[g, j, i] * D[g, j, i-1] / (D[g, j, i] + D[g, j, i-1])
                    diag_val += D_w
                    rows_M.append(idx)
                    cols_M.append(g * Nc + j * N + (i-1))
                    vals_M.append(-D_w)
                # else reflective BC: no term

                if i < N-1:
                    D_e = 2.0 * D[g, j, i] * D[g, j, i+1] / (D[g, j, i] + D[g, j, i+1])
                    diag_val += D_e
                    rows_M.append(idx)
                    cols_M.append(g * Nc + j * N + (i+1))
                    vals_M.append(-D_e)

                # Diffusion in y-direction
                if j > 0:
                    D_s = 2.0 * D[g, j, i] * D[g, j-1, i] / (D[g, j, i] + D[g, j-1, i])
                    diag_val += D_s
                    rows_M.append(idx)
                    cols_M.append(g * Nc + (j-1) * N + i)
                    vals_M.append(-D_s)

                if j < N-1:
                    D_n = 2.0 * D[g, j, i] * D[g, j+1, i] / (D[g, j, i] + D[g, j+1, i])
                    diag_val += D_n
                    rows_M.append(idx)
                    cols_M.append(g * Nc + (j+1) * N + i)
                    vals_M.append(-D_n)

                # Add diagonal
                rows_M.append(idx)
                cols_M.append(idx)
                vals_M.append(diag_val)

                # Scattering from other groups (g' != g)
                m = mat_map[j, i]
                mat = materials[m]
                for gp in range(G):
                    if gp != g:
                        s_val = mat['SigmaS'][g, gp]  # scattering from gp to g
                        if s_val > 0:
                            rows_M.append(idx)
                            cols_M.append(gp * Nc + j * N + i)
                            vals_M.append(-s_val * dx * dx)  # multiplied by cell area

                # Fission matrix (F)
                for gp in range(G):
                    f_val = Chi[g, j, i] * NuSigF[gp, j, i]
                    if f_val > 0:
                        rows_F.append(idx)
                        cols_F.append(gp * Nc + j * N + i)
                        vals_F.append(f_val * dx * dx)  # multiplied by cell area

    M = sp.csr_matrix((vals_M, (rows_M, cols_M)), shape=(size, size))
    F = sp.csr_matrix((vals_F, (rows_F, cols_F)), shape=(size, size))

    return M, F

# =============================================================================
# POWER ITERATION
# =============================================================================
def power_iteration_global(N, mat_map, max_iter=500, tol=1e-7):
    dx = pitch / N
    Nc = N * N
    size = G * Nc

    # Build global matrices
    print("Building global matrices M and F...")
    M, F = build_global_matrices(mat_map, dx, N)

    # Factorize M (LU decomposition)
    print("Factorizing M...")
    solve_M = spla.factorized(M.tocsc())

    # Initial guess: uniform positive flux
    phi = np.ones(size)
    k_eff = 1.0

    print(f"\n{'Iter':>5}  {'k_eff':>12}  {'dk':>12}")
    print("-" * 35)

    for it in range(1, max_iter + 1):
        # Compute fission source from old flux
        f_source = F @ phi

        # Solve M * phi_new = f_source / k_old
        rhs = f_source / k_eff
        phi_new = solve_M(rhs)

        # Ensure positivity
        phi_new = np.maximum(phi_new, 0.0)

        # Compute new fission source
        f_source_new = F @ phi_new

        # Update k
        k_new = k_eff * (np.sum(f_source_new) / np.sum(f_source))

        # Convergence check
        dk = abs(k_new - k_eff)
        if it % 10 == 0:
            print(f"{it:>5}  {k_new:>12.6f}  {dk:>12.2e}")

        if dk < tol:
            print(f"Converged at iteration {it}. Final k_eff: {k_new:.6f}")
            k_eff = k_new
            phi = phi_new
            break

        # Update for next iteration
        k_eff = k_new
        phi = phi_new
    else:
        print(f"Maximum iterations reached. Final k_eff: {k_eff:.6f}")

    # Reshape flux back to (G, N, N)
    flux = phi.reshape(G, N, N)
    return k_eff, flux

# =============================================================================
# PLOTTING
# =============================================================================
def plot_results(flux, mat_map, k, N):
    x = np.linspace(-pitch/2, pitch/2, N)
    y = np.linspace(-pitch/2, pitch/2, N)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'FDM Pincell — 2-Group Flux  |  k_eff = {k:.6f}', fontsize=14)

    group_labels = ['Group 1 (Fast)', 'Group 2 (Thermal)']

    for g in range(G):
        ax = axes[g]
        cf = ax.contourf(X, Y, flux[g], levels=50, cmap='jet')
        plt.colorbar(cf, ax=ax)

        # Draw fuel circle
        circle_fuel = patches.Circle((0, 0), r_fuel, fill=False,
                                     edgecolor='white', linewidth=1.5, linestyle='--')
        ax.add_patch(circle_fuel)
        
        # Draw cladding circles
        circle_clad_inner = patches.Circle((0, 0), r_clad_inner, fill=False,
                                          edgecolor='yellow', linewidth=1.0, linestyle=':')
        circle_clad_outer = patches.Circle((0, 0), r_clad_outer, fill=False,
                                          edgecolor='white', linewidth=1.5, linestyle='--')
        ax.add_patch(circle_clad_inner)
        ax.add_patch(circle_clad_outer)

        ax.set_title(group_labels[g], fontsize=11)
        ax.set_xlabel('x (cm)', fontsize=9)
        ax.set_ylabel('y (cm)', fontsize=9)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('fdm_pincell_2g_flux.png', dpi=150, bbox_inches='tight')
    plt.show()

    # --- Material map ---
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    cmap_mat = plt.cm.colors.ListedColormap(['blue', 'red', 'gray'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap_mat.N)
    
    ax2.contourf(X, Y, mat_map, levels=bounds, cmap=cmap_mat, norm=norm)
    
    # Draw region boundaries
    circle_fuel = patches.Circle((0, 0), r_fuel, fill=False,
                                 edgecolor='black', linewidth=2)
    circle_clad_inner = patches.Circle((0, 0), r_clad_inner, fill=False,
                                      edgecolor='black', linewidth=1.5, linestyle=':')
    circle_clad_outer = patches.Circle((0, 0), r_clad_outer, fill=False,
                                      edgecolor='black', linewidth=2)
    ax2.add_patch(circle_fuel)
    ax2.add_patch(circle_clad_inner)
    ax2.add_patch(circle_clad_outer)
    
    ax2.set_title('Material Map\n(Blue=Water, Red=Fuel, Gray=Cladding)')
    ax2.set_xlabel('x (cm)')
    ax2.set_ylabel('y (cm)')
    ax2.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('fdm_pincell_2g_matmap.png', dpi=150, bbox_inches='tight')
    plt.show()

    # --- Flux along centerline ---
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    mid = N // 2
    colors = ['blue', 'red']
    for g in range(G):
        phi_line = flux[g, mid, :]
        phi_line = phi_line / np.max(phi_line)
        ax3.plot(x, phi_line, color=colors[g], linewidth=2, label=group_labels[g])
    
    # Draw material boundaries
    ax3.axvline(-r_fuel, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axvline( r_fuel, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Fuel edge')
    ax3.axvline(-r_clad_inner, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax3.axvline( r_clad_inner, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Clad inner')
    ax3.axvline(-r_clad_outer, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax3.axvline( r_clad_outer, color='k', linestyle='-', linewidth=1, alpha=0.5, label='Clad outer')
    
    ax3.set_xlabel('x (cm)')
    ax3.set_ylabel('Normalized Flux')
    ax3.set_title('FDM — Flux along y=0 Centerline')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fdm_pincell_2g_centerline.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    N = 80   # mesh cells per side (increase for accuracy)

    print("=" * 60)
    print("  FDM PINCELL SOLVER — 2-Group Diffusion (Global Matrix Approach)")
    print(f"  Mesh : {N} x {N}")
    print(f"  BC   : {bc}")
    print(f"  r_fuel = {r_fuel} cm")
    print(f"  r_clad_inner = {r_clad_inner} cm")
    print(f"  r_clad_outer = {r_clad_outer} cm")
    print(f"  pitch = {pitch} cm")
    print("=" * 60)

    x, y, dx, mat_map = build_mesh(N)

    print(f"\nFuel cells     : {np.sum(mat_map == 1)}")
    print(f"Cladding cells : {np.sum(mat_map == 2)}")
    print(f"Water cells    : {np.sum(mat_map == 0)}")
    print(f"Cell size dx   : {dx:.4f} cm")

    k_eff, flux = power_iteration_global(N, mat_map, max_iter=500, tol=1e-6)

    print(f"\n{'='*60}")
    print(f"  FINAL k_eff = {k_eff:.6f}")
    print(f"{'='*60}")

    plot_results(flux, mat_map, k_eff, N)