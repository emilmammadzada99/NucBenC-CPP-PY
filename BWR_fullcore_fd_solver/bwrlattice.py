"""
BWR 4x4 Assembly Benchmark - 2-Group Diffusion Solver
Geometry: 4x4 Pin Grid with Gd-poisoned locations.
Materials: 2-Group (Fast & Thermal) macro-cell data.
BCs: All Reflective (Infinite Lattice).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time

# ===================================================================================================
# MATERIAL DATA (2-Group)
# ===================================================================================================

materials = {
    # Mix 0: Water (Moderator)
    0: {
        'name': 'Water',
        'SigmaT': np.array([6.40711e-1, 1.69131e-0]),
        'Chi': np.array([1.0, 0.0]),
        'SigmaS': np.array([
            [6.07382e-1, 0.0],        # 1->1, 2->1
            [3.31316e-2, 1.68428e-0]   # 1->2, 2->2
        ]),
        'NuSigF': np.array([0.0, 0.0])
    },
    
    # Mix 1: Standard UO2 Fuel
    1: {
        'name': 'UO2',
        'SigmaT': np.array([3.62022e-1, 5.72155e-1]),
        'Chi': np.array([1.0, 0.0]),
        'SigmaS': np.array([
            [3.33748e-1, 0.0],
            [6.64881e-4, 3.80898e-1]
        ]),
        'NuSigF': np.array([1.86278e-2, 3.44137e-1])
    },
    
    # Mix 2: Gd-Doped Fuel
    2: {
        'name': 'UO2-Gd',
        'SigmaT': np.array([3.71785e-1, 1.75000e-0]),
        'Chi': np.array([1.0, 0.0]),
        'SigmaS': np.array([
            [3.38096e-1, 0.0],
            [6.92807e-4, 3.83204e-1]
        ]),
        'NuSigF': np.array([1.79336e-2, 1.57929e-1])
    },

    # Mix 6: Cladding
    6: {
        'name': 'Cladding',
        'SigmaT': np.array([2.74144e-1, 2.80890e-1]),
        'Chi': np.array([1.0, 0.0]),
        'SigmaS': np.array([
            [2.72377e-1, 0.0],
            [1.90838e-4, 2.77230e-1]
        ]),
        'NuSigF': np.array([0.0, 0.0])
    }
}

# ===================================================================================================
# GEOMETRY DATA
# ===================================================================================================

pitch = 1.6     # cm
ri = 0.5        # fuel radius
ro = 0.6        # cladding radius (ri + 0.1 thickness)

pins = {
    1: {'mats': [1, 6, 0], 'pitch': pitch, 'ri': ri, 'ro': ro}, # Std Pin
    2: {'mats': [2, 6, 0], 'pitch': pitch, 'ri': ri, 'ro': ro}, # Gd Pin
}

# 4x4 Grid (0-indexed: Gd pins at (1,2) and (2,1))
grid = np.array([
    [1, 1, 1, 1],
    [1, 1, 2, 1],
    [1, 2, 1, 1],
    [1, 1, 1, 1]
])

# ===================================================================================================
# DIFFUSION SOLVER
# ===================================================================================================

class BWRDiffusionSolver:
    def __init__(self, mesh_refinement=1):
        self.mesh_ref = mesh_refinement
        self.Nx = grid.shape[0] * mesh_refinement
        self.Ny = grid.shape[1] * mesh_refinement
        self.ngroups = 2  # Set to 2 to match new data
        self.Nc = self.Nx * self.Ny
        self.h = pitch / mesh_refinement
        
        print(f"Assembly: {grid.shape[0]}x{grid.shape[1]} Pins")
        print(f"Mesh: {self.Nx}x{self.Ny} (refinement={mesh_refinement})")
        print(f"Total unknowns: {self.ngroups * self.Nc}")
        
    def homogenize_cell(self, pin_id):
        pin = pins[pin_id]
        mat_fuel, mat_clad, mat_mod = [materials[m] for m in pin['mats']]
        
        v_tot = pin['pitch']**2
        v_f = np.pi * pin['ri']**2
        v_c = np.pi * pin['ro']**2 - v_f
        v_m = v_tot - v_f - v_c
        
        f = np.array([v_f, v_c, v_m]) / v_tot
        
        st = f[0]*mat_fuel['SigmaT'] + f[1]*mat_clad['SigmaT'] + f[2]*mat_mod['SigmaT']
        ss = f[0]*mat_fuel['SigmaS'] + f[1]*mat_clad['SigmaS'] + f[2]*mat_mod['SigmaS']
        nsf = f[0]*mat_fuel['NuSigF']
        chi = mat_fuel['Chi']
        
        return st, chi, nsf, ss

    def setup_problem(self):
        self.D = np.zeros((self.ngroups, self.Nx, self.Ny))
        self.Sigma_rem = np.zeros((self.ngroups, self.Nx, self.Ny))
        self.NuSigF = np.zeros((self.ngroups, self.Nx, self.Ny))
        self.Chi = np.zeros((self.ngroups, self.Nx, self.Ny))
        self.Sigma_s_mat = np.zeros((self.ngroups, self.ngroups, self.Nx, self.Ny))
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                pin_id = grid[i // self.mesh_ref, j // self.mesh_ref]
                st, chi, nsf, ss = self.homogenize_cell(pin_id)
                
                for g in range(self.ngroups):
                    self.D[g, i, j] = 1.0 / (3.0 * st[g])
                    self.Sigma_rem[g, i, j] = st[g] - ss[g, g]
                    self.NuSigF[g, i, j] = nsf[g]
                    self.Chi[g, i, j] = chi[g]
                    for gp in range(self.ngroups):
                        self.Sigma_s_mat[g, gp, i, j] = ss[g, gp]

    def build_matrices(self):
        rows_M, cols_M, vals_M = [], [], []
        rows_F, cols_F, vals_F = [], [], []
        
        for g in range(self.ngroups):
            for i in range(self.Nx):
                for j in range(self.Ny):
                    idx = g * self.Nc + i * self.Ny + j
                    m_diag = self.Sigma_rem[g, i, j]
                    
                    for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < self.Nx and 0 <= nj < self.Ny:
                            d_avg = 2 * self.D[g, i, j] * self.D[g, ni, nj] / (self.D[g, i, j] + self.D[g, ni, nj])
                            coeff = d_avg / self.h**2
                            m_diag += coeff
                            rows_M.append(idx); cols_M.append(g * self.Nc + ni * self.Ny + nj); vals_M.append(-coeff)
                    
                    rows_M.append(idx); cols_M.append(idx); vals_M.append(m_diag)
                    
                    for gp in range(self.ngroups):
                        if gp != g:
                            s_val = self.Sigma_s_mat[g, gp, i, j]
                            if s_val > 0:
                                rows_M.append(idx); cols_M.append(gp * self.Nc + i * self.Ny + j); vals_M.append(-s_val)
                        
                        f_val = self.Chi[g, i, j] * self.NuSigF[gp, i, j]
                        if f_val > 0:
                            rows_F.append(idx); cols_F.append(gp * self.Nc + i * self.Ny + j); vals_F.append(f_val)
                            
        self.M = sp.csr_matrix((vals_M, (rows_M, cols_M)), shape=(self.ngroups*self.Nc, self.ngroups*self.Nc))
        self.F = sp.csr_matrix((vals_F, (rows_F, cols_F)), shape=(self.ngroups*self.Nc, self.ngroups*self.Nc))

    def solve(self):
        solve = spla.factorized(self.M.tocsc())
        phi = np.ones(self.ngroups * self.Nc)
        k_eff = 1.0
        for it in range(200):
            fission_source = self.F @ phi
            phi_new = solve(fission_source / k_eff)
            k_new = k_eff * (np.sum(self.F @ phi_new) / np.sum(fission_source))
            err = abs(k_new - k_eff) / k_new
            phi = phi_new / np.linalg.norm(phi_new)
            k_eff = k_new
            if err < 1e-8: break
        self.phi = phi
        return k_eff, phi

# ===================================================================================================
# EXECUTION
# ===================================================================================================

if __name__ == "__main__":
    solver = BWRDiffusionSolver(mesh_refinement=16)
    solver.setup_problem()
    solver.build_matrices()
    k_eff, phi = solver.solve()
    
    print(f"\nConverged k_eff: {k_eff:.6f}")
    
    # Visualization update
    phi_res = phi.reshape(solver.ngroups, solver.Nx, solver.Ny)
    fission_power = sum(solver.NuSigF[g] * phi_res[g] for g in range(solver.ngroups))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    titles = ['Fast Flux (G1)', 'Thermal Flux (G2)', 'Total Flux', 'Fission Power']
    data = [phi_res[0], phi_res[1], np.sum(phi_res, axis=0), fission_power]
    cmaps = ['viridis', 'plasma', 'inferno', 'hot']
    
    for ax, d, t, c in zip(axes.flat, data, titles, cmaps):
        im = ax.imshow(d.T, cmap=c, origin='lower', extent=[0, 6.4, 0, 6.4])
        ax.set_title(t)
        plt.colorbar(im, ax=ax)
        
    plt.tight_layout()
    plt.show()