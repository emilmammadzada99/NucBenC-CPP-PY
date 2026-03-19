"""
ANL 11-A2 (2D IAEA) Benchmark — Animation 2D
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time, os

# --- 1. Grid and Geometry ---
h = 1.0; Lx = 170.0; Ly = 130.0
Nx, Ny = int(round(Lx/h)), int(round(Ly/h))
xc = (np.arange(Nx)+0.5)*h; yc = (np.arange(Ny)+0.5)*h
XX, YY = np.meshgrid(xc, yc, indexing='ij')

VOID=-1; FUEL1=0; FUEL2=1; FUELR=2; REFL=3

def pip(px, py, poly):
    inside = np.zeros(px.shape, dtype=bool); n = len(poly); j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        c = ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / (yj - yi + 1e-300) + xi)
        inside ^= c; j = i
    return inside

poly_s1 = np.array([[70,70],[10,10],[10,0],[70,0],[70,10],[90,10],[90,0],[130,0],[130,30],[110,30],[110,70],[90,70]], float)
poly_s2 = np.array([[90,90],[90,70],[110,70],[110,30],[130,30],[130,0],[150,0],[150,50],[130,50],[130,90],[110,90],[110,110]], float)
poly_s6 = np.array([[110,110],[130,130],[130,110],[150,110],[150,70],[170,70],[170,0],[150,0],[150,50],[130,50],[130,90],[110,90]], float)

region_map = np.full((Nx, Ny), VOID, dtype=np.int8)
region_map[pip(XX, YY, poly_s6)] = REFL
region_map[pip(XX, YY, poly_s2)] = FUEL2
region_map[pip(XX, YY, poly_s1)] = FUEL1
region_map[((XX>=0)&(XX<=10)&(YY>=0)&(YY<=XX)) | ((XX>=70)&(XX<=90)&(YY>=0)&(YY<=10)) | ((XX>=70)&(XX<=90)&(YY>=70)&(YY<=XX))] = FUELR

active = region_map >= 0; N = int(active.sum())
dof = np.full((Nx, Ny), -1, dtype=np.int32); dof[active] = np.arange(N)
ii, jj = np.where(active)

# --- Matrix Settings ---
D_m = [[1.5, 1.5, 1.5, 2.0], [0.4, 0.4, 0.4, 0.3]]
xa_m = [[0.010, 0.010, 0.010, 0.000], [0.085, 0.080, 0.130, 0.010]]
nf_m = [[0.000, 0.000, 0.000, 0.000], [0.135, 0.135, 0.135, 0.000]]
chi_m = [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
xs12_m = [0.02, 0.02, 0.02, 0.04]
B2z = 0.8e-4

def get_prop(arr):
    out = np.zeros((Nx, Ny))
    for m in range(4): out[region_map == m] = arr[m]
    return out

def build_leakage(g):
    D = get_prop(D_m[g])
    r, c, v = [], [], []
    diag = np.zeros(N)
    # X leakage
    ix, jx = np.where(active[:-1,:] & active[1:,:])
    val = (2*D[ix,jx]*D[ix+1,jx]/(D[ix,jx]+D[ix+1,jx])) / h**2
    r.extend(dof[ix,jx]); c.extend(dof[ix+1,jx]); v.extend(-val)
    r.extend(dof[ix+1,jx]); c.extend(dof[ix,jx]); v.extend(-val)
    np.add.at(diag, dof[ix,jx], val); np.add.at(diag, dof[ix+1,jx], val)
    # Y leakage
    iy, jy = np.where(active[:,:-1] & active[:,1:])
    valy = (2*D[iy,jy]*D[iy,jy+1]/(D[iy,jy]+D[iy,jy+1])) / h**2
    r.extend(dof[iy,jy]); c.extend(dof[iy,jy+1]); v.extend(-valy)
    r.extend(dof[iy,jy+1]); c.extend(dof[iy,jy]); v.extend(-valy)
    np.add.at(diag, dof[iy,jy], valy); np.add.at(diag, dof[iy,jy+1], valy)
    return sp.csr_matrix((v, (r, c)), shape=(N,N)) + sp.diags(diag)

L1 = build_leakage(0); L2 = build_leakage(1)
A11 = L1 + sp.diags(get_prop(xa_m[0])[active] + get_prop(D_m[0])[active]*B2z + get_prop(xs12_m)[active])
A22 = L2 + sp.diags(get_prop(xa_m[1])[active] + get_prop(D_m[1])[active]*B2z)
S12 = sp.diags(-get_prop(xs12_m)[active])
ZERO = sp.csr_matrix((N, N))

Mop = sp.bmat([[A11, ZERO], [S12, A22]], format='csc')

F12 = sp.diags(get_prop(chi_m[0])[active] * get_prop(nf_m[1])[active])
F22 = sp.diags(get_prop(chi_m[1])[active] * get_prop(nf_m[1])[active])
Fop = sp.bmat([[ZERO, F12], [ZERO, F22]], format='csr')

# --- FPS settings ---
fps_val = 20
video_len = 23
total_frames = video_len * fps_val  # 460 Frames
save_step = 3                      # Take a frame every 3 steps.
max_iter = total_frames * save_step # 1380 Iter

phi = np.random.rand(2*N); phi /= np.linalg.norm(phi)
k_eff = 1.0; lu = spla.splu(Mop)
p1_frames, p2_frames, k_history, iter_history = [], [], [], []

print(f"Hesaplanıyor: {max_iter} iter")
for it in range(1, max_iter + 1):
    rhs = (1.0/k_eff) * (Fop @ phi)
    phi_new = lu.solve(rhs)
    k_new = k_eff * np.linalg.norm(Fop @ phi_new) / np.linalg.norm(Fop @ phi)
    phi = phi_new / np.linalg.norm(phi_new)
    k_eff = k_new
    
    if it % save_step == 0:
        f1 = np.zeros((Nx, Ny)); f2 = np.zeros((Nx, Ny))
        f1[ii, jj] = phi[:N]; f2[ii, jj] = phi[N:]
        p1_frames.append(f1.T / (f1.max() + 1e-15))
        p2_frames.append(f2.T / (f2.max() + 1e-15))
        k_history.append(k_eff)
        iter_history.append(it)
        if it % 300 == 0: print(f"İlerleme: %{100*it/max_iter:.1f}")

# --- Animation ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
ext = [0, Lx, 0, Ly]
im1 = ax1.imshow(p1_frames[0], origin='lower', extent=ext, cmap='viridis', vmin=0, vmax=1)
im2 = ax2.imshow(p2_frames[0], origin='lower', extent=ext, cmap='magma', vmin=0, vmax=1)
ax1.set_title("Fast Flux ($\phi_1$)"); ax2.set_title("Thermal Flux ($\phi_2$)")
plt.colorbar(im1, ax=ax1, shrink=0.7); plt.colorbar(im2, ax=ax2, shrink=0.7)

def update(f):
    im1.set_array(p1_frames[f])
    im2.set_array(p2_frames[f])
    fig.suptitle(f"2D IAEA Benchmark\n Iter: {iter_history[f]} | k_eff: {k_history[f]:.6f}",
                 fontsize=14, fontweight='bold')
    return im1, im2

ani = FuncAnimation(fig, update, frames=len(p1_frames), blit=True)

print("Video Saving...")
ani.save('ANL11A2_23s_Final.gif', writer='pillow', fps=fps_val)
print("Finished! 'ANL11A2_23s_Final.gif' done.")