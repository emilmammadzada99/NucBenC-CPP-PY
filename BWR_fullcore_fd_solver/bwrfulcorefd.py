import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time

# ===================================================================================================
# 1. GEOMETRY AND 72x72 SUPER-GRID CONSTRUCTION
# ===================================================================================================

# 4x4 Lattice structure inside 'G' cell (1: UO2, 2: Gd-Fuel)
lattice_G = np.array([
    [1, 1, 1, 1],
    [1, 1, 2, 1],
    [1, 2, 1, 1],
    [1, 1, 1, 1]
])

# 18x18 Core Map
core_map_raw = """
W W W W W W W W W W W W W W W W W W
W W W W W G G G G G G G G W W W W W
W W W W G G G G G G G G G G W W W W
W W W G G G G G G G G G G G G W W W
W W G G G G G G G G G G G G G G W W
W G G G G G G G G G G G G G G G G W
W G G G G G G G G G G G G G G G G W
W G G G G G G G G G G G G G G G G W
W G G G G G G G G G G G G G G G G W
W G G G G G G G G G G G G G G G G W
W G G G G G G G G G G G G G G G G W
W G G G G G G G G G G G G G G G G W
W G G G G G G G G G G G G G G G G W
W W G G G G G G G G G G G G G G W W
W W W G G G G G G G G G G G G W W W
W W W W G G G G G G G G G G W W W W
W W W W W G G G G G G G G W W W W W
W W W W W W W W W W W W W W W W W W
"""

def build_super_grid(map_str):
    lines = map_str.strip().split('\n')
    rows = []
    for line in lines:
        row_cells = line.split()
        sub_rows = [[] for _ in range(4)]
        for cell in row_cells:
            if cell == 'G':
                for r in range(4): sub_rows[r].extend(lattice_G[r])
            else: # 'W' adds 4x4 water (0)
                for r in range(4): sub_rows[r].extend([0, 0, 0, 0])
        rows.extend(sub_rows)
    return np.array(rows)

grid = build_super_grid(core_map_raw) # 72x72 Pin Grid

# ===================================================================================================
# 2. MATERIAL DATA (2-Group)
# ===================================================================================================

# Homogenization Constants
v_f, v_c, v_m = (np.pi*0.5**2), (np.pi*0.6**2 - np.pi*0.5**2), (1.6**2 - np.pi*0.6**2)
f = np.array([v_f, v_c, v_m]) / 1.6**2

def mix(m_f, m_c, m_w):
    return f[0]*m_f + f[1]*m_c + f[2]*m_w

# Cross Section Data
water = {'st': np.array([6.407e-1, 1.691e0]), 'ss': np.array([[6.07e-1, 0.0], [3.31e-2, 1.68e0]]), 'nsf': np.array([0.0, 0.0])}
uo2 = {'st': np.array([3.620e-1, 5.721e-1]), 'ss': np.array([[3.33e-1, 0.0], [6.64e-4, 3.80e-1]]), 'nsf': np.array([1.86e-2, 3.44e-1])}
gd = {'st': np.array([3.717e-1, 1.750e0]), 'ss': np.array([[3.38e-1, 0.0], [6.92e-4, 3.83e-1]]), 'nsf': np.array([1.79e-2, 1.57e-1])}
clad = {'st': np.array([2.741e-1, 2.808e-1]), 'ss': np.array([[2.72e-1, 0.0], [1.90e-4, 2.77e-1]]), 'nsf': np.array([0.0, 0.0])}

# Cell Property Map (0: Water, 1: UO2 Pin, 2: Gd Pin)
prop_map = {
    0: (water['st'], np.array([1.0, 0.0]), water['nsf'], water['ss']),
    1: (mix(uo2['st'], clad['st'], water['st']), np.array([1.0, 0.0]), mix(uo2['nsf'], clad['nsf'], water['nsf']), mix(uo2['ss'], clad['ss'], water['ss'])),
    2: (mix(gd['st'], clad['st'], water['st']), np.array([1.0, 0.0]), mix(gd['nsf'], clad['nsf'], water['nsf']), mix(gd['ss'], clad['ss'], water['ss']))
}

# ===================================================================================================
# 3. SOLVER CLASS
# ===================================================================================================

class CoreLatticeSolver:
    def __init__(self, grid):
        self.grid = grid
        self.Nx, self.Ny = grid.shape
        self.ngroups = 2
        self.Nc = self.Nx * self.Ny
        self.h = 1.6 # cm

    def build(self):
        print(f"Building system: {self.Nx}x{self.Ny} grid...")
        rows_M, cols_M, vals_M = [], [], []
        rows_F, cols_F, vals_F = [], [], []
        self.NSF_map = np.zeros((self.ngroups, self.Nx, self.Ny))
        
        D_all = np.zeros((self.ngroups, self.Nx, self.Ny))
        Rem_all = np.zeros((self.ngroups, self.Nx, self.Ny))

        for i in range(self.Nx):
            for j in range(self.Ny):
                st, chi, nsf, ss = prop_map[self.grid[i, j]]
                for g in range(self.ngroups):
                    D_all[g, i, j] = 1.0 / (3.0 * st[g])
                    Rem_all[g, i, j] = st[g] - ss[g, g]
                    self.NSF_map[g, i, j] = nsf[g]
                    
                    idx = g * self.Nc + i * self.Ny + j
                    m_diag = Rem_all[g, i, j]
                    
                    # Diffusion (Leakage terms)
                    for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < self.Nx and 0 <= nj < self.Ny:
                            st_n, _, _, _ = prop_map[self.grid[ni, nj]]
                            dn = 1.0 / (3.0 * st_n[g])
                            d_avg = 2 * D_all[g, i, j] * dn / (D_all[g, i, j] + dn)
                            coeff = d_avg / self.h**2
                            m_diag += coeff
                            rows_M.append(idx); cols_M.append(g * self.Nc + ni * self.Ny + nj); vals_M.append(-coeff)
                    
                    rows_M.append(idx); cols_M.append(idx); vals_M.append(m_diag)
                    
                    # Scattering and Fission
                    if g == 1: # Thermal group gets downscatter from fast
                        rows_M.append(idx); cols_M.append(0 * self.Nc + i * self.Ny + j); vals_M.append(-ss[1,0])
                    
                    for gp in range(self.ngroups):
                        f_val = chi[g] * nsf[gp]
                        if f_val > 0:
                            rows_F.append(idx); cols_F.append(gp * self.Nc + i * self.Ny + j); vals_F.append(f_val)

        self.M = sp.csr_matrix((vals_M, (rows_M, cols_M)), shape=(self.ngroups*self.Nc, self.ngroups*self.Nc))
        self.F = sp.csr_matrix((vals_F, (rows_F, cols_F)), shape=(self.ngroups*self.Nc, self.ngroups*self.Nc))

    def solve(self):
        print("Running Power Iteration...")
        lu = spla.factorized(self.M.tocsc())
        phi = np.ones(self.ngroups * self.Nc)
        k = 1.0
        for it in range(300):
            src = self.F @ phi
            phi_new = lu(src / k)
            k_new = k * (np.sum(self.F @ phi_new) / np.sum(src))
            if abs(k_new - k) < 1e-8: break
            phi = phi_new / np.linalg.norm(phi_new)
            k = k_new
        return k, phi

# ===================================================================================================
# 4. EXECUTION
# ===================================================================================================
solver = CoreLatticeSolver(grid)
solver.build()
k_eff, phi = solver.solve()

phi_res = phi.reshape(2, 72, 72)
power = solver.NSF_map[0]*phi_res[0] + solver.NSF_map[1]*phi_res[1]

print(f"\nFinal k-eff: {k_eff:.6f}")

# ===================================================================================================
# 5. DETAILED ANALYSIS
# ===================================================================================================

group_info = [
    ("Group 0", "Fast", "> 0.625 eV"),
    ("Group 1", "Thermal", "< 0.625 eV")
]

phi_3d = phi_res

print("\n" + "="*80)
print("1. GROUP FLUX ANALYSIS")
print("="*80)
print("\n{:<8} {:<12} {:<25} {:<12} {:<12} {:<12}".format(
    "Group", "Category", "Energy Range", "Total", "Maximum", "Average"))
print("-"*80)

total_flux_all = np.sum(phi_3d)
for g in range(solver.ngroups):
    name, category, energy = group_info[g]
    total = np.sum(phi_3d[g])
    maximum = np.max(phi_3d[g])
    average = np.mean(phi_3d[g])
    print("{:<8} {:<12} {:<25} {:<12.4e} {:<12.4e} {:<12.4e}".format(
        name, category, energy, total, maximum, average))

print("-"*80)
print(f"{'TOTAL':<8} {'':<12} {'':<25} {total_flux_all:<12.4e}")

print("\nGROUP FRACTION ANALYSIS (%):")
for g in range(solver.ngroups):
    fraction = 100 * np.sum(phi_3d[g]) / total_flux_all
    print(f"{group_info[g][0]}: {fraction:6.2f}%")

print("\n" + "="*80)
print("2. REGION-WISE FLUX ANALYSIS")
print("="*80)

regions = {'UO2': [], 'Gd': [], 'Water': []}
for i in range(solver.Nx):
    for j in range(solver.Ny):
        pin_id = grid[i, j]
        if pin_id == 0: regions['Water'].append((i, j))
        elif pin_id == 1: regions['UO2'].append((i, j))
        elif pin_id == 2: regions['Gd'].append((i, j))

total_flux_map = np.sum(phi_3d, axis=0)
print("\nAverage flux by region (sum over all groups):")
print("{:<15} {:<12} {:<12} {:<12} {:<12}".format("Region", "Cell Count", "Avg Flux", "Max Flux", "Min Flux"))
print("-"*80)

for region_name, cells in regions.items():
    if cells:
        fluxes = [total_flux_map[i, j] for i, j in cells]
        print("{:<15} {:<12} {:<12.4e} {:<12.4e} {:<12.4e}".format(
            region_name, len(cells), np.mean(fluxes), np.max(fluxes), np.min(fluxes)))

print("\n" + "="*80)
print("3. SPECTRAL ANALYSIS (Thermal/Fast Ratio)")
print("="*80)
print("\n{:<15} {:<15} {:<15} {:<15}".format("Region", "Avg Thermal", "Avg Fast", "Thermal/Fast"))
print("-"*80)

for region_name, cells in regions.items():
    if cells:
        t_vals = [phi_3d[1, i, j] for i, j in cells]
        f_vals = [phi_3d[0, i, j] for i, j in cells]
        avg_t, avg_f = np.mean(t_vals), np.mean(f_vals)
        print("{:<15} {:<15.4e} {:<15.4e} {:<15.3f}".format(region_name, avg_t, avg_f, avg_t/avg_f))

print("\n" + "="*80)
print("4. FISSION POWER ANALYSIS")
print("="*80)
fuel_indices = power > 1e-10
avg_fuel_power = np.mean(power[fuel_indices])
max_power = np.max(power)

print(f"Total fission power   : {np.sum(power):.6e}")
print(f"Maximum (pin power)   : {max_power:.6e}")
print(f"Average (in fuel)     : {avg_fuel_power:.6e}")
print(f"Pin power peaking     : {max_power/avg_fuel_power:.4f}")

# ===================================================================================================
# 6. VISUALIZATION
# ===================================================================================================
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Fast flux
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(phi_3d[0].T, cmap='jet', origin='lower')
ax1.set_title('Fast Flux (Group 0)')
plt.colorbar(im1, ax=ax1)

# Thermal flux
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(phi_3d[1].T, cmap='jet', origin='lower')
ax2.set_title('Thermal Flux (Group 1)')
plt.colorbar(im2, ax=ax2)

# Total flux
ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.imshow(total_flux_map.T, cmap='viridis', origin='lower')
ax3.set_title('Total Flux')
plt.colorbar(im3, ax=ax3)

# Fission power
ax4 = fig.add_subplot(gs[1, 0])
im4 = ax4.imshow(power.T, cmap='hot', origin='lower')
ax4.set_title('Fission Power Distribution')
plt.colorbar(im4, ax=ax4)

# Thermal/Fast ratio
ax5 = fig.add_subplot(gs[1, 1])
ratio_map = phi_3d[1] / (phi_3d[0] + 1e-10)
im5 = ax5.imshow(ratio_map.T, cmap='RdYlBu_r', origin='lower', vmin=0, vmax=5)
ax5.set_title('Thermal/Fast Ratio')
plt.colorbar(im5, ax=ax5)

# Histogram
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(power[fuel_indices].flatten(), bins=50, color='orange', alpha=0.7)
ax6.set_title('Power Distribution Histogram')
ax6.set_yscale('log')

plt.tight_layout()
plt.show()