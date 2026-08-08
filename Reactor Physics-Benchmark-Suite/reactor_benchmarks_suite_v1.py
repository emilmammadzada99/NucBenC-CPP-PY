"""
================================================================================
 REACTOR PHYSICS VOXEL SUITE — Unified Pygame Application
================================================================================
ALL of the calculations from the ANL benchmark files (ANL14-A1, ANL 11-A2),
the BWR Core Lattice, and C5G7 are solved with a single SHARED 3-DIMENSIONAL
VOXEL multi-group neutron diffusion solver, and can all be selected from a
menu and run inside one pygame application.

  - ANL14-A1   : 2 groups, neutron diffusion ONLY (NO thermal feedback)
  - ANL 11-A2  : 2 groups, IAEA 2D benchmark, extruded into a 3D voxel mesh
  - BWR Core   : 2 groups, homogenized pin-cell full-core voxel model
  - C5G7       : 7 groups, pin-level material map, 3D voxel

Pick a benchmark from the menu -> click RUN -> the solve runs in a
background thread (the UI never freezes) -> the result opens in a
rotatable / zoomable REAL 3D VOXEL point-cloud viewer (with slice / material
map modes included).

Dependencies: pygame, numpy, scipy
Run with    : python reactor_voxel_suite.py
NumPy/SciPy/Matplotlib.
Author: Emil Mammadzada
"""

import sys
import time
import threading
import traceback

import numpy as np
import pygame
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ============================================================================
# COLOR / VISUAL HELPERS
# ============================================================================

BG        = (16, 18, 24)
PANEL     = (28, 31, 40)
PANEL_HI  = (40, 44, 56)
ACCENT    = (66, 165, 245)
ACCENT_HI = (100, 190, 255)
TEXT      = (230, 232, 238)
TEXT_DIM  = (150, 155, 165)
GOOD      = (100, 220, 140)
BAD       = (240, 110, 110)


def _jet(t):
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(np.minimum(4 * t - 1.5, -4 * t + 4.5), 0, 1)
    g = np.clip(np.minimum(4 * t - 0.5, -4 * t + 3.5), 0, 1)
    b = np.clip(np.minimum(4 * t + 0.5, -4 * t + 2.5), 0, 1)
    return np.stack([r, g, b], axis=-1)


def _viridis(t):
    stops = np.array([
        [0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530],
        [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
        [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318],
        [0.741, 0.873, 0.150], [0.993, 0.906, 0.144],
    ])
    x = np.linspace(0, 1, len(stops))
    return np.stack([np.interp(t, x, stops[:, c]) for c in range(3)], axis=-1)


def _plasma(t):
    stops = np.array([
        [0.050, 0.030, 0.528], [0.294, 0.011, 0.631], [0.494, 0.012, 0.657],
        [0.664, 0.139, 0.585], [0.798, 0.280, 0.470], [0.902, 0.428, 0.361],
        [0.972, 0.601, 0.256], [0.994, 0.795, 0.156], [0.940, 0.975, 0.131],
    ])
    x = np.linspace(0, 1, len(stops))
    return np.stack([np.interp(t, x, stops[:, c]) for c in range(3)], axis=-1)


def _hot(t):
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(3 * t, 0, 1)
    g = np.clip(3 * t - 1, 0, 1)
    b = np.clip(3 * t - 2, 0, 1)
    return np.stack([r, g, b], axis=-1)


def _grayscale(t):
    t = np.clip(t, 0.0, 1.0)
    return np.stack([t, t, t], axis=-1)


def _cool(t):
    t = np.clip(t, 0.0, 1.0)
    r = t
    g = 1.0 - t
    b = np.ones_like(t)
    return np.stack([r, g, b], axis=-1)


# 16-bit color resolution: each palette is precomputed as a 65536 (2**16)
# step lookup table (LUT), so field values (0..1) are displayed with a
# much smoother color gradient than an 8-bit / 256-step palette would
# give.
LUT_SIZE = 1 << 16  # 65536 — 16-bit color palette resolution


def _make_lut(func):
    return (np.clip(func(np.linspace(0, 1, LUT_SIZE)), 0, 1) * 255).astype(np.uint8)


PALETTES = {
    "Jet": _make_lut(_jet),
    "Viridis": _make_lut(_viridis),
    "Plasma": _make_lut(_plasma),
    "Hot": _make_lut(_hot),
    "Cool": _make_lut(_cool),
    "Grayscale": _make_lut(_grayscale),
}
PALETTE_NAMES = list(PALETTES.keys())


def field_to_lut_indices(field):
    """0..1 normalized field -> 16-bit LUT indices."""
    return np.clip((field * (LUT_SIZE - 1)).astype(np.int64), 0, LUT_SIZE - 1)


def draw_button(screen, font, rect, text, hovered=False, active=False):
    color = ACCENT_HI if hovered else (ACCENT if active else PANEL_HI)
    pygame.draw.rect(screen, color, rect, border_radius=8)
    pygame.draw.rect(screen, (10, 10, 14), rect, width=1, border_radius=8)
    label = font.render(text, True, (10, 10, 14) if (hovered or active) else TEXT)
    screen.blit(label, label.get_rect(center=rect.center))


def draw_text(screen, font, text, pos, color=TEXT):
    screen.blit(font.render(text, True, color), pos)


# ============================================================================
# FULL-GEOMETRY MIRRORING HELPERS
# ============================================================================
# ANL14-A1, ANL 11-A2 and C5G7 are solved on a symmetric QUARTER-core domain
# (reflective boundary at the low-index edges). The physical flux/geometry
# is mirror-symmetric about those edges, so for DISPLAY purposes we mirror
# the quarter-domain arrays across the low-x and low-y edges to reconstruct
# the FULL reactor geometry / flux map (no re-solving needed — the solved
# quarter-domain values already satisfy the symmetry condition exactly).
#
#   shared_x/shared_y = True  -> the i=0 / j=0 row is itself ON the symmetry
#                                 line (vertex-centered grid) and must not
#                                 be duplicated when mirrored.
#   shared_x/shared_y = False -> the i=0 / j=0 row is the CENTER of the
#                                 first cell next to the symmetry line
#                                 (cell-centered grid) -> plain mirror.
# ============================================================================

def mirror_xy(arr, axis_x, axis_y, shared_x=False, shared_y=False):
    a = arr
    if shared_x:
        rest = np.take(a, range(1, a.shape[axis_x]), axis=axis_x)
        a = np.concatenate([np.flip(rest, axis=axis_x), a], axis=axis_x)
    else:
        a = np.concatenate([np.flip(a, axis=axis_x), a], axis=axis_x)
    if shared_y:
        rest = np.take(a, range(1, a.shape[axis_y]), axis=axis_y)
        a = np.concatenate([np.flip(rest, axis=axis_y), a], axis=axis_y)
    else:
        a = np.concatenate([np.flip(a, axis=axis_y), a], axis=axis_y)
    return a


# ============================================================================
# SHARED 3D VOXEL MULTI-GROUP DIFFUSION SOLVER
# ============================================================================
# Each benchmark produces a 2D "region_map" (Nx x Ny, -1 = void/vacuum, >=0
# material index) and a "mat_table" (material index -> cross sections). This
# class "extrudes" that map along a shared z-axis (Nz layers) to build a
# real 3D voxel mesh and solves the multi-group neutron diffusion
# k-eigenvalue problem.
#
# mat_table[r] = {
#     'D':    np.array(G)      diffusion coefficient
#     'rem':  np.array(G)      removal (= absorption + total out-scatter)
#     'nsf':  np.array(G)      nu * Sigma_fission
#     'chi':  np.array(G)      fission spectrum
#     'scat': np.array(G,G)    scat[g,gp] = scattering from group gp -> g (g!=gp)
# }
# ============================================================================

class VoxelDiffusion3D:
    def __init__(self, region_map, mat_table, h, Nz, hz, G, name,
                 vacuum_hi_xy=False, vacuum_lo_xy=False, vacuum_z=False, alpha=0.5):
        """
        vacuum_hi_xy / vacuum_lo_xy : apply a Marshak vacuum boundary
            condition at the Nx-1/Ny-1 (high) or 0 (low) index edges. If
            False, no leakage term is added at that edge (reflective /
            symmetry approximation).
        vacuum_z : Marshak vacuum boundary condition at both ends of the
            z-axis.
        alpha    : Marshak vacuum coefficient (typically 0.5).
        """
        self.region_map = region_map
        self.mat_table = mat_table
        self.Nx, self.Ny = region_map.shape
        self.Nz = Nz
        self.h = h
        self.hz = hz
        self.G = G
        self.name = name
        self.active = region_map >= 0
        self.vacuum_hi_xy = vacuum_hi_xy
        self.vacuum_lo_xy = vacuum_lo_xy
        self.vacuum_z = vacuum_z
        self.alpha = alpha

        # Compact DOF map for ACTIVE (i,j) cells ONLY. If region_map
        # contains void/vacuum (-1) cells (e.g. ANL 11-A2), these must NOT
        # be included in the global indexing; otherwise the rows/columns
        # corresponding to those cells would stay entirely zero and the
        # matrix would become SINGULAR -> the solve breaks down / gives a
        # wrong result.
        self.dof2d = np.full((self.Nx, self.Ny), -1, dtype=np.int64)
        ii, jj = np.where(self.active)
        self.Nact = len(ii)
        self.dof2d[ii, jj] = np.arange(self.Nact)
        self.Nc = self.Nact * self.Nz

    def _idx(self, g, k, i, j):
        return g * self.Nc + k * self.Nact + self.dof2d[i, j]

    def build(self, progress=lambda s: None):
        progress(f"{self.name}: preparing material fields...")
        Nx, Ny, Nz, G = self.Nx, self.Ny, self.Nz, self.G
        D = np.zeros((G, Nx, Ny))
        rem = np.zeros((G, Nx, Ny))
        nsf = np.zeros((G, Nx, Ny))
        chi = np.zeros((G, Nx, Ny))
        scat = np.zeros((G, G, Nx, Ny))
        for r, props in self.mat_table.items():
            mask = self.region_map == r
            if not np.any(mask):
                continue
            D[:, mask] = props['D'][:, None]
            rem[:, mask] = props['rem'][:, None]
            nsf[:, mask] = props['nsf'][:, None]
            chi[:, mask] = props['chi'][:, None]
            scat[:, :, mask] = props['scat'][:, :, None]
        self.D, self.rem, self.nsf, self.chi, self.scat = D, rem, nsf, chi, scat

        progress(f"{self.name}: building 3D voxel matrices "
                  f"({Nx}x{Ny}x{Nz}, {G} groups, {G*self.Nc:,} unknowns)...")
        t0 = time.time()
        rows_M, cols_M, vals_M = [], [], []
        rows_F, cols_F, vals_F = [], [], []
        h2, hz2 = self.h ** 2, self.hz ** 2
        active = self.active

        for i in range(Nx):
            for j in range(Ny):
                if not active[i, j]:
                    continue
                for g in range(G):
                    idx = self._idx(g, 0, i, j)  # k will be added below
                for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    pass
            if i % 10 == 0:
                progress(f"{self.name}: scanning row {i + 1}/{Nx}...")

        # (the empty loops above are only for progress reporting; the
        #  actual assembly, including the k loop, is done below.)
        rows_M, cols_M, vals_M = [], [], []
        rows_F, cols_F, vals_F = [], [], []
        for i in range(Nx):
            for j in range(Ny):
                if not active[i, j]:
                    continue
                for k in range(Nz):
                    for g in range(G):
                        idx = self._idx(g, k, i, j)
                        m_diag = rem[g, i, j]

                        # x, y leakage
                        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < Nx and 0 <= nj < Ny and active[ni, nj]:
                                Dn = D[g, ni, nj]
                                d_avg = 2 * D[g, i, j] * Dn / (D[g, i, j] + Dn + 1e-30)
                                c = d_avg / h2
                                m_diag += c
                                rows_M.append(idx); cols_M.append(self._idx(g, k, ni, nj)); vals_M.append(-c)
                            else:
                                is_hi = (di == 1 and i == Nx - 1) or (dj == 1 and j == Ny - 1)
                                is_lo = (di == -1 and i == 0) or (dj == -1 and j == 0)
                                if (is_hi and self.vacuum_hi_xy) or (is_lo and self.vacuum_lo_xy):
                                    d_b = D[g, i, j] / (self.h * (self.alpha + D[g, i, j] / self.h))
                                    m_diag += d_b
                                # otherwise (reflective/symmetry or void neighbor): no term added

                        # z leakage (between voxel layers)
                        for dk in (1, -1):
                            nk = k + dk
                            if 0 <= nk < Nz:
                                c = D[g, i, j] / hz2  # D is constant within the same material column
                                m_diag += c
                                rows_M.append(idx); cols_M.append(self._idx(g, nk, i, j)); vals_M.append(-c)
                            elif self.vacuum_z:
                                d_b = D[g, i, j] / (self.hz * (self.alpha + D[g, i, j] / self.hz))
                                m_diag += d_b

                        rows_M.append(idx); cols_M.append(idx); vals_M.append(m_diag)

                        # in-group scattering + fission source
                        for gp in range(G):
                            if gp != g:
                                sval = scat[g, gp, i, j]
                                if sval > 1e-30:
                                    rows_M.append(idx); cols_M.append(self._idx(gp, k, i, j)); vals_M.append(-sval)
                            fval = chi[g, i, j] * nsf[gp, i, j]
                            if fval > 1e-30:
                                rows_F.append(idx); cols_F.append(self._idx(gp, k, i, j)); vals_F.append(fval)
            if i % 10 == 0:
                progress(f"{self.name}: assembling matrix row {i + 1}/{Nx}...")

        size = G * self.Nc
        self.M = sp.csr_matrix((vals_M, (rows_M, cols_M)), shape=(size, size))
        self.F = sp.csr_matrix((vals_F, (rows_F, cols_F)), shape=(size, size))
        progress(f"{self.name}: matrices ready M:nnz={self.M.nnz:,} "
                 f"F:nnz={self.F.nnz:,} [{time.time()-t0:.1f}s]")

    def solve(self, progress=lambda s: None, max_iter=250, tol=1e-6):
        size = self.G * self.Nc
        t0 = time.time()
        if size <= 45000:
            progress(f"{self.name}: direct LU factorization ({size:,} unknowns)...")
            solve_lin = spla.factorized(self.M.tocsc())
        else:
            progress(f"{self.name}: preparing ILU preconditioner ({size:,} unknowns)...")
            ilu = spla.spilu(self.M.tocsc(), drop_tol=1e-5, fill_factor=12)
            Mop = spla.LinearOperator(self.M.shape, ilu.solve)

            def solve_lin(rhs):
                sol, _ = spla.bicgstab(self.M, rhs, M=Mop, rtol=1e-8, maxiter=800)
                return sol

        progress(f"{self.name}: starting power iteration...")
        phi = np.ones(size)
        phi /= np.linalg.norm(phi)
        k_eff = 1.0
        for it in range(1, max_iter + 1):
            src_old = self.F @ phi
            phi_new = solve_lin(src_old / k_eff)
            phi_new = np.maximum(phi_new, 0.0)
            src_new = self.F @ phi_new
            s_old, s_new = np.sum(src_old), np.sum(src_new)
            k_new = k_eff * (s_new / s_old) if s_old > 1e-30 else k_eff
            err = abs(k_new - k_eff) / max(abs(k_new), 1e-30)
            phi = phi_new / (np.linalg.norm(phi_new) + 1e-30)
            k_eff = k_new
            if it % 5 == 0 or err < tol:
                progress(f"{self.name}: iter {it}  k_eff={k_eff:.6f}  error={err:.2e}")
            if err < tol and it > 2:
                break

        # Scatter the compact DOF vector back into the full (G, Nz, Nx, Ny)
        # field; void/vacuum cells are left at 0.
        phi_full = np.zeros((self.G, self.Nz, self.Nx, self.Ny))
        ii, jj = np.where(self.active)
        d = self.dof2d[ii, jj]
        phi_gk = phi.reshape(self.G, self.Nz, self.Nact)
        phi_full[:, :, ii, jj] = phi_gk[:, :, d]

        self.phi = phi_full
        self.k_eff = k_eff
        self.solve_time = time.time() - t0
        progress(f"{self.name}: converged. k_eff={k_eff:.6f} [{self.solve_time:.1f}s]")
        return k_eff, self.phi


def mt(D, rem, nsf, chi, scat_pairs, G):
    """Shorthand for building a mat_table entry. scat_pairs: {(g,gp): value}  (gp -> g)."""
    S = np.zeros((G, G))
    for (g, gp), v in scat_pairs.items():
        S[g, gp] = v
    return dict(D=np.array(D, float), rem=np.array(rem, float),
                nsf=np.array(nsf, float), chi=np.array(chi, float), scat=S)


# ============================================================================
# BENCHMARK 1 — ANL14-A1  (2 groups, neutron diffusion ONLY, NO thermal feedback)
# ============================================================================

def build_anl14a1(nx=50, ny=50):
    Lx = Ly = 165.0
    D_BASE = np.array([
        [1.255, 1.268, 1.259, 1.259, 1.259, 1.257],
        [0.211, 0.1902, 0.2091, 0.2091, 0.2091, 0.1592],
    ])
    XSA_BASE = np.array([
        [0.008252, 0.007181, 0.008002, 0.008002, 0.008002, 0.0006034],
        [0.1003, 0.07047, 0.08344, 0.08344, 0.073324, 0.01911],
    ])
    NUXSF_BASE = np.array([
        [0.004602, 0.004609, 0.004663, 0.004663, 0.004663, 0.],
        [0.1091, 0.08675, 0.1021, 0.1021, 0.1021, 0.],
    ])
    XSS_12_BASE = np.array([0.02533, 0.02767, 0.02617, 0.02617, 0.02617, 0.04754])
    CHI = np.array([[1.] * 6, [0.] * 6])

    x = np.linspace(0., Lx, nx + 1)
    y = np.linspace(0., Ly, ny + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    region = np.full((nx + 1, ny + 1), 5, dtype=int)
    m_reg1 = (X >= 15) & (X <= 105) & (Y >= 15) & (Y <= 105)
    m_s1 = (X >= 0) & (X <= 15) & (Y >= 0) & (Y <= 15)
    m_s2 = (X >= 75) & (X <= 105) & (Y >= 0) & (Y <= 15)
    m_s3 = (X >= 0) & (X <= 15) & (Y >= 75) & (Y <= 105)
    m_s4 = (X >= 75) & (X <= 105) & (Y >= 75) & (Y <= 105)
    m_s6 = (X >= 105) & (X <= 135) & (Y >= 0) & (Y <= 75)
    m_cr = (X >= 105) & (X <= 135) & (Y >= 75) & (Y <= 105)
    m_s8 = (X >= 0) & (X <= 105) & (Y >= 105) & (Y <= 135)
    m_reg4 = (X >= 105) & (X <= 120) & (Y >= 105) & (Y <= 120)
    region[m_reg1] = 0
    region[m_s1] = 1; region[m_s2] = 1; region[m_s3] = 1; region[m_s4] = 1
    region[m_s6] = 2; region[m_s8] = 2
    region[m_cr] = 3
    region[m_reg4] = 4
    region[(X > Lx) | (Y > Ly)] = -1

    mat_table = {}
    for r in range(6):
        mat_table[r] = mt(
            D=D_BASE[:, r], rem=[XSA_BASE[0, r] + XSS_12_BASE[r], XSA_BASE[1, r]],
            nsf=NUXSF_BASE[:, r], chi=CHI[:, r], scat_pairs={(1, 0): XSS_12_BASE[r]}, G=2)

    MAT_COLORS = {0: (66, 165, 245), 1: (240, 180, 60), 2: (140, 90, 220),
                  3: (230, 70, 70), 4: (90, 210, 130), 5: (110, 115, 125)}
    h = x[1] - x[0]
    return dict(region_map=region, mat_table=mat_table, h=h, G=2,
                mat_colors=MAT_COLORS, group_labels=["Fast Flux (Group 1)", "Thermal Flux (Group 2)"],
                display_groups=(0, 1), name="ANL14-A1 (2-group diffusion, no thermal feedback)")


# ============================================================================
# BENCHMARK 2 — ANL 11-A2  (2D IAEA benchmark, 2 groups)
# ============================================================================
# def build_anl11a2(h=2.5):
    # Lx, Ly = 170.0, 130.0
    # Nx = int(round(Lx / h)); Ny = int(round(Ly / h))
    # xc = (np.arange(Nx) + 0.5) * h
    # yc = (np.arange(Ny) + 0.5) * h
    # XX, YY = np.meshgrid(xc, yc, indexing='ij')

    # VOID, FUEL1, FUEL2, FUELR, REFL = -1, 0, 1, 2, 3

    # def pip(px, py, poly):
        # inside = np.zeros(px.shape, dtype=bool)
        # n = len(poly); j = n - 1
        # for i in range(n):
            # xi, yi = poly[i]; xj, yj = poly[j]
            # c = ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / (yj - yi + 1e-300) + xi)
            # inside ^= c
            # j = i
        # return inside
def build_anl11a2(h=2.5):
    Lx, Ly = 170.0, 130.0
    Nx = int(round(Lx / h)); Ny = int(round(Ly / h))
    xc = (np.arange(Nx) + 0.5) * h
    yc = (np.arange(Ny) + 0.5) * h
    XX, YY = np.meshgrid(xc, yc, indexing='ij')

    VOID, FUEL1, FUEL2, FUELR, REFL = -1, 0, 1, 2, 3

    def pip(px, py, poly):
        inside = np.zeros(px.shape, dtype=bool)
        n = len(poly); j = n - 1
        for i in range(n):
            xi, yi = poly[i]; xj, yj = poly[j]
            c = ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / (yj - yi + 1e-300) + xi)
            inside ^= c
            j = i
        return inside

    poly_s1 = np.array([[70, 70], [10, 10], [10, 0], [70, 0], [70, 10], [90, 10], [90, 0],
                         [130, 0], [130, 30], [110, 30], [110, 70], [90, 70]], float)
    poly_s2 = np.array([[90, 90], [90, 70], [110, 70], [110, 30], [130, 30], [130, 0], [150, 0],
                         [150, 50], [130, 50], [130, 90], [110, 90], [110, 110]], float)
    poly_s6 = np.array([[110, 110], [130, 130], [130, 110], [150, 110], [150, 70], [170, 70],
                         [170, 0], [150, 0], [150, 50], [130, 50], [130, 90], [110, 90]], float)

    in_s3 = (XX >= 0) & (XX <= 10) & (YY >= 0) & (YY <= XX)
    in_s4 = (XX >= 70) & (XX <= 90) & (YY >= 0) & (YY <= 10)
    in_s5 = (XX >= 70) & (XX <= 90) & (YY >= 70) & (YY <= XX)
    in_s1 = pip(XX, YY, poly_s1); in_s2 = pip(XX, YY, poly_s2); in_s6 = pip(XX, YY, poly_s6)

    region = np.full((Nx, Ny), VOID, dtype=np.int8)
    region[in_s6] = REFL; region[in_s2] = FUEL2; region[in_s1] = FUEL1
    region[in_s3 | in_s4 | in_s5] = FUELR

    D_m = [[1.5, 1.5, 1.5, 2.0], [0.4, 0.4, 0.4, 0.3]]
    xa_m = [[0.010, 0.010, 0.010, 0.000], [0.085, 0.080, 0.130, 0.010]]
    nf_m = [[0.000, 0.000, 0.000, 0.000], [0.135, 0.135, 0.135, 0.000]]
    ch_m = [[1., 1., 1., 1.], [0., 0., 0., 0.]]
    xs12 = [0.02, 0.02, 0.02, 0.04]

    mat_table = {}
    for r in range(4):
        mat_table[r] = mt(D=[D_m[0][r], D_m[1][r]], rem=[xa_m[0][r] + xs12[r], xa_m[1][r]],
                           nsf=[nf_m[0][r], nf_m[1][r]], chi=[ch_m[0][r], ch_m[1][r]],
                           scat_pairs={(1, 0): xs12[r]}, G=2)

    MAT_COLORS = {0: (230, 70, 70), 1: (240, 180, 60), 2: (90, 210, 130), 3: (140, 90, 220)}
    return dict(region_map=region, mat_table=mat_table, h=h, G=2,
                mat_colors=MAT_COLORS, group_labels=["Fast Flux (Group 1)", "Thermal Flux (Group 2)"],
                display_groups=(0, 1), name="ANL 11-A2 (2D IAEA benchmark, 2-group diffusion)")


# ============================================================================
# BENCHMARK 3 — BWR Core Lattice (homogenized pin-cell, full core)
# ============================================================================

def build_bwr_core():
    lattice_G = np.array([[1, 1, 1, 1], [1, 1, 2, 1], [1, 2, 1, 1], [1, 1, 1, 1]])
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
            sub_rows = [[] for _ in range(4)]
            for cell in line.split():
                if cell == 'G':
                    for r in range(4):
                        sub_rows[r].extend(lattice_G[r])
                else:
                    for r in range(4):
                        sub_rows[r].extend([0, 0, 0, 0])
            rows.extend(sub_rows)
        return np.array(rows)

    grid = build_super_grid(core_map_raw)  # 0=Water 1=UO2 2=Gd, all active

    v_f, v_c, v_m = (np.pi * 0.5 ** 2), (np.pi * 0.6 ** 2 - np.pi * 0.5 ** 2), (1.6 ** 2 - np.pi * 0.6 ** 2)
    f = np.array([v_f, v_c, v_m]) / 1.6 ** 2

    def mix(m_f, m_c, m_w):
        return f[0] * m_f + f[1] * m_c + f[2] * m_w

    water = {'st': np.array([6.407e-1, 1.691e0]), 'ss': np.array([[6.07e-1, 0.0], [3.31e-2, 1.68e0]]), 'nsf': np.array([0.0, 0.0])}
    uo2 = {'st': np.array([3.620e-1, 5.721e-1]), 'ss': np.array([[3.33e-1, 0.0], [6.64e-4, 3.80e-1]]), 'nsf': np.array([1.86e-2, 3.44e-1])}
    gd = {'st': np.array([3.717e-1, 1.750e0]), 'ss': np.array([[3.38e-1, 0.0], [6.92e-4, 3.83e-1]]), 'nsf': np.array([1.79e-2, 1.57e-1])}
    clad = {'st': np.array([2.741e-1, 2.808e-1]), 'ss': np.array([[2.72e-1, 0.0], [1.90e-4, 2.77e-1]]), 'nsf': np.array([0.0, 0.0])}

    props = {
        0: (water['st'], np.array([1.0, 0.0]), water['nsf'], water['ss']),
        1: (mix(uo2['st'], clad['st'], water['st']), np.array([1.0, 0.0]),
            mix(uo2['nsf'], clad['nsf'], water['nsf']), mix(uo2['ss'], clad['ss'], water['ss'])),
        2: (mix(gd['st'], clad['st'], water['st']), np.array([1.0, 0.0]),
            mix(gd['nsf'], clad['nsf'], water['nsf']), mix(gd['ss'], clad['ss'], water['ss'])),
    }

    mat_table = {}
    for r, (st, chi, nsf, ss) in props.items():
        D = 1.0 / (3.0 * st)
        rem = [st[0] - ss[0, 0], st[1] - ss[1, 1]]
        mat_table[r] = mt(D=D, rem=rem, nsf=nsf, chi=chi, scat_pairs={(1, 0): ss[1, 0]}, G=2)

    MAT_COLORS = {0: (60, 90, 200), 1: (210, 60, 60), 2: (60, 210, 60)}
    return dict(region_map=grid.astype(int), mat_table=mat_table, h=1.6, G=2,
                mat_colors=MAT_COLORS, group_labels=["Fast Flux (Group 1)", "Thermal Flux (Group 2)"],
                display_groups=(0, 1), name="BWR Core Lattice (homogenized pin-cell, full core)")


# ============================================================================
# BENCHMARK 4 — C5G7 (7 groups, pin-level material map)
# ============================================================================

def build_c5g7():
    materials = {
        0: dict(name='Moderator', SigmaT=[1.59206E-01, 4.12970E-01, 5.90310E-01, 5.84350E-01, 7.18000E-01, 1.25445E+00, 2.65038E+00],
                Chi=[5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, 0, 0, 0],
                SigmaS=[[4.44777E-02,0,0,0,0,0,0],[1.13400E-01,2.82334E-01,0,0,0,0,0],
                        [7.23470E-04,1.29940E-01,3.45256E-01,0,0,0,0],
                        [3.74990E-06,6.23400E-04,2.24570E-01,9.10284E-02,7.14370E-05,0,0],
                        [5.31840E-08,4.80020E-05,1.69990E-02,4.15510E-01,1.39138E-01,2.21570E-03,0],
                        [0,7.44860E-06,2.64430E-03,6.37320E-02,5.11820E-01,6.99913E-01,1.32440E-01],
                        [0,1.04550E-06,5.03440E-04,1.21390E-02,6.12290E-02,5.37320E-01,2.48070E+00]],
                NuSigF=[0]*7),
        1: dict(name='UO2', SigmaT=[1.77949E-01,3.29805E-01,4.80388E-01,5.54367E-01,3.11801E-01,3.95168E-01,5.64406E-01],
                Chi=[5.87910E-01,4.11760E-01,3.39060E-04,1.17610E-07,0,0,0],
                SigmaS=[[1.27537E-01,0,0,0,0,0,0],[4.23780E-02,3.24456E-01,0,0,0,0,0],
                        [9.43740E-06,1.63140E-03,4.50940E-01,0,0,0,0],
                        [5.51630E-09,3.14270E-09,2.67920E-03,4.52565E-01,1.25250E-04,0,0],
                        [0,0,0,5.56640E-03,2.71401E-01,1.29680E-03,0],
                        [0,0,0,0,1.02550E-02,2.65802E-01,8.54580E-03],
                        [0,0,0,0,1.00210E-08,1.68090E-02,2.73080E-01]],
                NuSigF=[2.005998E-02,2.027303E-03,1.570599E-02,4.518301E-02,4.334208E-02,2.020901E-01,5.257105E-01]),
        2: dict(name='MOX4.3', SigmaT=[1.78731E-01,3.30849E-01,4.83772E-01,5.66922E-01,4.26227E-01,6.78997E-01,6.82852E-01],
                Chi=[5.87910E-01,4.11760E-01,3.39060E-04,1.17610E-07,0,0,0],
                SigmaS=[[1.28876E-01,0,0,0,0,0,0],[4.14130E-02,3.25452E-01,0,0,0,0,0],
                        [8.22900E-06,1.63950E-03,4.53188E-01,0,0,0,0],
                        [5.04050E-09,1.59820E-09,2.61420E-03,4.57173E-01,1.60460E-04,0,0],
                        [0,0,0,5.53940E-03,2.76814E-01,2.00510E-03,0],
                        [0,0,0,0,9.31270E-03,2.52962E-01,8.49480E-03],
                        [0,0,0,0,9.16560E-09,1.48500E-02,2.65007E-01]],
                NuSigF=[0.0217530045,0.0025351033,0.0162679915,0.0654740997,0.0307240878,0.6666509616,0.7139904304]),
        3: dict(name='MOX7.0', SigmaT=[0.181323,0.334368,0.493785,0.591216,0.474198,0.833601,0.853603],
                Chi=[5.87910E-01,4.11760E-01,3.39060E-04,1.17610E-07,0,0,0],
                SigmaS=[[0.130457,0,0,0,0,0,0],[0.041792,0.328428,0,0,0,0,0],
                        [8.5105E-06,0.0016436,0.458371,0,0,0,0],
                        [5.1329E-09,2.2017E-09,0.0025331,0.463709,0.00017619,0,0],
                        [0,0,0,0.0054766,0.282313,0.002276,0],
                        [0,0,0,0,0.0087289,0.249751,0.0088645],
                        [0,0,0,0,9.0016E-09,0.013114,0.259529]],
                NuSigF=[0.023813952,0.0038586888,0.0241340014,0.09436622,0.0457698761,0.9281814045,1.0432001182]),
        4: dict(name='MOX8.7', SigmaT=[0.183045,0.336705,0.500507,0.606174,0.502754,0.921028,0.955231],
                Chi=[5.87910E-01,4.11760E-01,3.39060E-04,1.17610E-07,0,0,0],
                SigmaS=[[0.131504,0,0,0,0,0,0],[0.042046,0.330403,0,0,0,0,0],
                        [8.6972E-06,0.0016463,0.461792,0,0,0,0],
                        [5.1938E-09,2.6006E-09,0.0024749,0.468021,0.00018597,0,0],
                        [0,0,0,0.005433,0.285771,0.0023916,0],
                        [0,0,0,0,0.0083973,0.247614,0.0089681],
                        [0,0,0,0,8.928E-09,0.012322,0.256093]],
                NuSigF=[0.0251860041,0.0047395095,0.029478054,0.1122499985,0.0553030128,1.0749988378,1.23929836992]),
        5: dict(name='FissionChamber', SigmaT=[0.126032,0.29316,0.28425,0.28102,0.33446,0.56564,1.17214],
                Chi=[5.87910E-01,4.11760E-01,3.39060E-04,1.17610E-07,0,0,0],
                SigmaS=[[0.0661659,0,0,0,0,0,0],[0.059070,0.240377,0,0,0,0,0],
                        [0.00028334,0.052435,0.183425,0,0,0,0],
                        [1.4622E-06,0.0002499,0.092288,0.0790769,0.00003734,0,0],
                        [2.0642E-08,0.000019239,0.0069365,0.16999,0.099757,0.00091742,0],
                        [0,2.9875E-06,0.001079,0.02586,0.20679,0.316774,0.049793],
                        [0,4.214E-07,0.00020543,0.0049256,0.024478,0.23876,1.0991]],
                NuSigF=[1.3234E-08,1.4345E-08,1.1285993E-06,1.27629932E-05,3.538502E-07,1.7400989E-06,5.0633019E-06]),
        6: dict(name='GuideTube', SigmaT=[1.26032E-01,2.93160E-01,2.84240E-01,2.80960E-01,3.34440E-01,5.65640E-01,1.17215E+00],
                Chi=[5.87910E-01,4.11760E-01,3.39060E-04,1.17610E-07,0,0,0],
                SigmaS=[[6.61659E-02,0,0,0,0,0,0],[5.90700E-02,2.40377E-01,0,0,0,0,0],
                        [2.83340E-04,5.24350E-02,1.83297E-01,0,0,0,0],
                        [1.46220E-06,2.49900E-04,9.23970E-02,7.88511E-02,3.73330E-05,0,0],
                        [2.06420E-08,1.92390E-05,6.94460E-03,1.70140E-01,9.97372E-02,9.17260E-04,0],
                        [0,2.98750E-06,1.08030E-03,2.58810E-02,2.06790E-01,3.16765E-01,4.97920E-02],
                        [0,4.21400E-07,2.05670E-04,4.92970E-03,2.44780E-02,2.38770E-01,1.09912E+00]],
                NuSigF=[0]*7),
    }

    grid = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,6,1,1,6,1,1,6,1,1,1,1,1, 2,3,3,3,3,6,3,3,6,3,3,6,3,3,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,6,1,1,1,1,1,1,1,1,1,6,1,1,1, 2,3,3,6,3,4,4,4,4,4,4,4,3,6,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,3,3,3,4,4,4,4,4,4,4,4,4,3,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,6,1,1,6,1,1,6,1,1,6,1,1,6,1,1, 2,3,6,4,4,6,4,4,6,4,4,6,4,4,6,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,3,3,4,4,4,4,4,4,4,4,4,4,4,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,3,3,4,4,4,4,4,4,4,4,4,4,4,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,6,1,1,6,1,1,5,1,1,6,1,1,6,1,1, 2,3,6,4,4,6,4,4,5,4,4,6,4,4,6,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,3,3,4,4,4,4,4,4,4,4,4,4,4,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,3,3,4,4,4,4,4,4,4,4,4,4,4,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,6,1,1,6,1,1,6,1,1,6,1,1,6,1,1, 2,3,6,4,4,6,4,4,6,4,4,6,4,4,6,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,3,3,3,4,4,4,4,4,4,4,4,4,3,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,6,1,1,1,1,1,1,1,1,1,6,1,1,1, 2,3,3,6,3,4,4,4,4,4,4,4,3,6,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,6,1,1,6,1,1,6,1,1,1,1,1, 2,3,3,3,3,6,3,3,6,3,3,6,3,3,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [2,3,3,3,3,6,3,3,6,3,3,6,3,3,3,3,2, 1,1,1,1,1,6,1,1,6,1,1,6,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [2,3,3,6,3,4,4,4,4,4,4,4,3,6,3,3,2, 1,1,1,6,1,1,1,1,1,1,1,1,1,6,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [2,3,3,3,4,4,4,4,4,4,4,4,4,3,3,3,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [2,3,6,4,4,6,4,4,6,4,4,6,4,4,6,3,2, 1,1,6,1,1,6,1,1,6,1,1,6,1,1,6,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [2,3,3,4,4,4,4,4,4,4,4,4,4,4,3,3,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [2,3,3,4,4,4,4,4,4,4,4,4,4,4,3,3,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [2,3,6,4,4,6,4,4,5,4,4,6,4,4,6,3,2, 1,1,6,1,1,6,1,1,5,1,1,6,1,1,6,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [2,3,3,4,4,4,4,4,4,4,4,4,4,4,3,3,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [2,3,3,4,4,4,4,4,4,4,4,4,4,4,3,3,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [2,3,6,4,4,6,4,4,6,4,4,6,4,4,6,3,2, 1,1,6,1,1,6,1,1,6,1,1,6,1,1,6,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [2,3,3,3,4,4,4,4,4,4,4,4,4,3,3,3,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [2,3,3,6,3,4,4,4,4,4,4,4,3,6,3,3,2, 1,1,1,6,1,1,1,1,1,1,1,1,1,6,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [2,3,3,3,3,6,3,3,6,3,3,6,3,3,3,3,2, 1,1,1,1,1,6,1,1,6,1,1,6,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ] + [[0] * 51] * 17)

    # Pin geometry: each value in the grid is a "pin_id". pin_id=0 is a PURE
    # water cell; pin_id 1-6 is a fuel/material rod of RADIUS 0.54 cm,
    # volume-homogenized with WATER inside a cell with a 1.26 cm PITCH.
    # (This is exactly the same operation as homogenize_cell() in the
    #  original C5G7 code — skipping this step would use pure UO2/MOX
    #  material, which is PHYSICALLY WRONG and makes k_eff come out far
    #  too low; this has been corrected.)
    pitch, radius = 1.26, 0.54
    vol_fuel = (np.pi * radius ** 2) / (pitch ** 2)
    vol_mod = 1.0 - vol_fuel
    water = materials[0]

    G = 7
    mat_table = {}
    for pin_id in range(7):
        if pin_id == 0:
            st = np.array(water['SigmaT']); chi = np.array(water['Chi'])
            nsf = np.array(water['NuSigF']); ss = np.array(water['SigmaS'])
        else:
            fuel = materials[pin_id]
            st = vol_fuel * np.array(fuel['SigmaT']) + vol_mod * np.array(water['SigmaT'])
            chi = np.array(fuel['Chi'])
            nsf = vol_fuel * np.array(fuel['NuSigF'])
            ss = vol_fuel * np.array(fuel['SigmaS']) + vol_mod * np.array(water['SigmaS'])

        D = 1.0 / (3.0 * st)
        rem = st - np.diag(ss)
        scat_pairs = {}
        for gg in range(G):
            for gp in range(G):
                if gg != gp and ss[gg, gp] > 1e-30:
                    scat_pairs[(gg, gp)] = ss[gg, gp]
        mat_table[pin_id] = mt(D=D, rem=rem, nsf=nsf, chi=chi, scat_pairs=scat_pairs, G=G)

    MAT_COLORS = {0: (60, 90, 200), 1: (210, 60, 60), 2: (240, 150, 60), 3: (240, 100, 40),
                  4: (240, 60, 20), 5: (255, 255, 0), 6: (150, 150, 150)}
    return dict(region_map=grid.astype(int), mat_table=mat_table, h=1.26, G=7,
                mat_colors=MAT_COLORS, group_labels=[f"Group {g+1} Flux" for g in range(7)],
                display_groups=(0, 6), name="C5G7 (7-group, pin-level material map)")


# ============================================================================
# BENCHMARK 5 — PINCELL  (single fuel pin, 2 groups, fully-resolved circular
# fuel/cladding/water geometry, reflective on all sides -> k-infinity of an
# infinite lattice of identical pin cells)
# ============================================================================

PINCELL_N = 17          # voxels per axis (cubic mesh, matches original script)
PINCELL_PITCH = 1.26    # cm
PINCELL_H = PINCELL_PITCH / PINCELL_N


def build_pincell(N=PINCELL_N):
    r_fuel, r_clad_in, r_clad_out = 0.39, 0.40, 0.46
    pitch = PINCELL_PITCH
    h = pitch / N
    x = np.linspace(-pitch / 2 + h / 2, pitch / 2 - h / 2, N)
    y = np.linspace(-pitch / 2 + h / 2, pitch / 2 - h / 2, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Rr = np.sqrt(X ** 2 + Y ** 2)

    region = np.zeros((N, N), dtype=int)          # 0 = water (default)
    region[Rr <= r_fuel] = 1                       # 1 = fuel
    region[(Rr > r_clad_in) & (Rr <= r_clad_out)] = 2  # 2 = cladding

    materials = {
        0: dict(name='Water', SigmaT=[6.40711e-1, 1.69131e0], Chi=[1.0, 0.0],
                SigmaS=[[6.07382e-1, 0.0], [3.31316e-2, 1.68428e0]], NuSigF=[0.0, 0.0]),
        1: dict(name='Pin (Fuel)', SigmaT=[3.62022e-1, 5.72155e-1], Chi=[1.0, 0.0],
                SigmaS=[[3.33748e-1, 0.0], [6.64881e-4, 3.80898e-1]], NuSigF=[1.86278e-2, 3.44137e-1]),
        2: dict(name='Cladding', SigmaT=[2.74144e-1, 2.80890e-1], Chi=[1.0, 0.0],
                SigmaS=[[2.72377e-1, 0.0], [1.90838e-4, 2.77230e-1]], NuSigF=[0.0, 0.0]),
    }

    G = 2
    mat_table = {}
    for r, m in materials.items():
        st = np.array(m['SigmaT']); ss = np.array(m['SigmaS'])
        # Formula SPECIFIC to this benchmark (identical to the original
        # voxel_pincell_3d.py): D = 1 / (3 * sigma_transport), where
        # sigma_transport = SigmaT - SigmaS_own-group
        # (other benchmarks use D = 1/(3*SigmaT); here the original file's
        # definition was kept as-is.)
        sig_tr = st - np.diag(ss)
        D = 1.0 / (3.0 * sig_tr)
        rem = sig_tr
        scat_pairs = {}
        for gg in range(G):
            for gp in range(G):
                if gg != gp and ss[gg, gp] > 1e-30:
                    scat_pairs[(gg, gp)] = ss[gg, gp]
        mat_table[r] = mt(D=D, rem=rem, nsf=m['NuSigF'], chi=m['Chi'], scat_pairs=scat_pairs, G=G)

    MAT_COLORS = {0: (60, 90, 200), 1: (210, 60, 60), 2: (150, 150, 150)}
    return dict(region_map=region, mat_table=mat_table, h=h, G=2,
                mat_colors=MAT_COLORS, group_labels=["Fast Flux (Group 1)", "Thermal Flux (Group 2)"],
                display_groups=(0, 1), name="Pincell (single fuel pin, 2-group, k-infinity)")


BENCHMARKS = [
    ("ANL14-A1  (2-group diffusion, no thermal feedback)", build_anl14a1,
     dict(Nz=6, hz=3.75, vacuum_hi_xy=False, vacuum_lo_xy=False, vacuum_z=False,
          mirror=True, shared_x=True, shared_y=True)),
    ("ANL 11-A2  (2D IAEA benchmark, 2-group)", build_anl11a2,
     dict(Nz=6, hz=4.0, vacuum_hi_xy=False, vacuum_lo_xy=False, vacuum_z=False,
          mirror=True, shared_x=False, shared_y=False)),
    ("BWR Core Lattice  (voxel / pin-cell, full core)", build_bwr_core,
     dict(Nz=5, hz=15.0, vacuum_hi_xy=False, vacuum_lo_xy=False, vacuum_z=False,
          mirror=False, shared_x=False, shared_y=False)),
    ("C5G7  (7-group, pin-level, quarter core)", build_c5g7,
     dict(Nz=6, hz=12.85, vacuum_hi_xy=True, vacuum_lo_xy=False, vacuum_z=False,
          mirror=True, shared_x=False, shared_y=False)),
    ("Pincell  (single fuel pin, 2-group, k-infinity)", build_pincell,
     dict(Nz=PINCELL_N, hz=PINCELL_H, vacuum_hi_xy=False, vacuum_lo_xy=False, vacuum_z=False,
          mirror=False, shared_x=False, shared_y=False)),
]


def get_geometry(bench_idx):
    """Instant (no-solve) geometry preview: builds the material map and, for
    benchmarks solved on a symmetric quarter-domain, mirrors it into the
    FULL reactor geometry for display."""
    label, builder, extra = BENCHMARKS[bench_idx]
    spec = builder()
    region_map = spec['region_map']
    mat_color_grid = np.zeros((*region_map.shape, 3), dtype=np.uint8)
    for r, c in spec['mat_colors'].items():
        mat_color_grid[region_map == r] = c

    quarter_map = mat_color_grid
    quarter_region = region_map
    if extra['mirror']:
        full_map = mirror_xy(mat_color_grid, 0, 1, extra['shared_x'], extra['shared_y'])
        full_region = mirror_xy(region_map, 0, 1, extra['shared_x'], extra['shared_y'])
    else:
        full_map = mat_color_grid
        full_region = region_map

    return dict(name=spec['name'], h=spec['h'], mirror=extra['mirror'],
                quarter_map=quarter_map, full_map=full_map,
                quarter_region=quarter_region, full_region=full_region)


def run_benchmark(bench_idx, progress=lambda s: None):
    label, builder, extra = BENCHMARKS[bench_idx]
    progress(f"{label}: preparing geometry and cross sections...")
    spec = builder()
    solver = VoxelDiffusion3D(spec['region_map'], spec['mat_table'], spec['h'],
                               extra['Nz'], extra['hz'], spec['G'], spec['name'],
                               vacuum_hi_xy=extra['vacuum_hi_xy'], vacuum_lo_xy=extra['vacuum_lo_xy'],
                               vacuum_z=extra['vacuum_z'])
    solver.build(progress)
    k_eff, phi = solver.solve(progress)

    G, Nz, Nx, Ny = phi.shape
    nsf_field = np.zeros((G, Nx, Ny))
    for r, props in spec['mat_table'].items():
        mask = spec['region_map'] == r
        nsf_field[:, mask] = props['nsf'][:, None]
    power = np.einsum('gkij,gij->kij', phi, nsf_field)

    mat_color_grid = np.zeros((Nx, Ny, 3), dtype=np.uint8)
    for r, c in spec['mat_colors'].items():
        mat_color_grid[spec['region_map'] == r] = c
    region_map = spec['region_map']

    if extra['mirror']:
        progress(f"{label}: mirroring quarter-domain solution into full geometry...")
        sx, sy = extra['shared_x'], extra['shared_y']
        phi = mirror_xy(phi, 2, 3, sx, sy)
        power = mirror_xy(power, 1, 2, sx, sy)
        mat_color_grid = mirror_xy(mat_color_grid, 0, 1, sx, sy)
        region_map = mirror_xy(region_map, 0, 1, sx, sy)
        G, Nz, Nx, Ny = phi.shape

    return dict(
        name=spec['name'], k_eff=k_eff, phi=phi, power=power,
        Nx=Nx, Ny=Ny, Nz=Nz, G=G,
        group_labels=spec['group_labels'], display_groups=spec['display_groups'],
        mat_color_grid=mat_color_grid, region_map=region_map,
        solve_time=solver.solve_time, mirrored=extra['mirror'],
    )


# ============================================================================
# PYGAME APPLICATION
# ============================================================================

WIDTH, HEIGHT = 1300, 780
STATE_MENU, STATE_GEOMETRY, STATE_RUNNING, STATE_VIEWER = "menu", "geometry", "running", "viewer"


def normalize(field):
    mn, mx = float(field.min()), float(field.max())
    rng = mx - mn
    return np.ones_like(field) if rng <= 1e-30 else (field - mn) / rng


def rot_matrix(yaw, pitch_ang):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp_ = np.cos(pitch_ang), np.sin(pitch_ang)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp_], [0, sp_, cp]])
    return Rx @ Ry


class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Reactor Physics Voxel Suite — ANL / BWR / C5G7")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont("segoeui,arial", 30, bold=True)
        self.font = pygame.font.SysFont("segoeui,arial", 19)
        self.font_small = pygame.font.SysFont("consolas,arial", 15)

        self.state = STATE_MENU
        self.selected = 0
        self.thread = None
        self.progress_lock = threading.Lock()
        self.progress_msg = "Starting..."
        self.result = None
        self.error = None
        self.run_start_time = 0.0

        # geometry preview state
        self.geo = None
        self.geo_show_full = True
        self.solve_rect = None
        self.geo_cut_rect = None

        # color palette (16-bit resolution LUT, selectable)
        self.palette_name = "Jet"
        self.palette_rects = []

        # viewer state
        self.mode_i = 0
        self.modes = ['3d', 'slice', 'material']
        self.mode_labels = ["3D", "SLICE", "MATERIAL"]
        self.panel_i = 0  # 0=group A/B, 1=A/power, 2=B/power
        self.slice_k = 0
        self.yaw, self.pitch_ang, self.zoom = 0.7, 0.35, 1.0
        self.dragging = False
        self.last_mouse = (0, 0)
        self.density_threshold = 0.0
        self.cut_variant = 0  # 0 = full geometry, 1..4 = quarter cutaway variants
        self.mode_rects, self.field_rects = [], []
        self.cut_rect = self.reset_rect = self.back_rect = None
        self.slice_prev_rect = self.slice_next_rect = None

    def lut(self):
        return PALETTES[self.palette_name]

    # ---- progress ----
    def set_progress(self, msg):
        with self.progress_lock:
            self.progress_msg = msg

    def get_progress(self):
        with self.progress_lock:
            return self.progress_msg

    # ---- geometry preview (instant, no solve) ----
    def start_geometry_preview(self):
        self.geo = get_geometry(self.selected)
        self.geo_show_full = True
        self.state = STATE_GEOMETRY

    # ---- solve ----
    def start_solve(self):
        self.result = None
        self.error = None
        self.run_start_time = time.time()
        self.state = STATE_RUNNING
        idx = self.selected

        def worker():
            try:
                res = run_benchmark(idx, progress=self.set_progress)
                self.result = res
            except Exception as e:
                self.error = f"{e}\n{traceback.format_exc()}"

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()

    def enter_viewer(self):
        self.state = STATE_VIEWER
        self.mode_i = 0
        self.panel_i = 0
        r = self.result
        self.slice_k = r['Nz'] // 2
        self.yaw, self.pitch_ang, self.zoom = 0.7, 0.35, 1.0
        self.density_threshold, self.cut_variant = 0.0, 0
        gA, gB = r['display_groups']
        self.fields = {
            0: normalize(r['phi'][gA]),
            1: normalize(r['phi'][gB]),
            2: normalize(r['power']),
        }
        self.panel_titles = [r['group_labels'][gA], r['group_labels'][gB], "Fission Power Density"]

    # ---- draw: menu ----
    def draw_menu(self):
        self.screen.fill(BG)
        draw_text(self.screen, self.font_title, "Reactor Physics Benchmarks Suite", (60, 36))
        draw_text(self.screen, self.font_small,
                  "ANL14-A1 / ANL 11-A2 / BWR Core / C5G7 — shared 3D  multi-group neutron diffusion solver",
                  (60, 78), TEXT_DIM)

        list_y = 140
        item_h = 66
        mouse = pygame.mouse.get_pos()
        self.item_rects = []
        for i, (name, _, extra) in enumerate(BENCHMARKS):
            rect = pygame.Rect(60, list_y + i * (item_h + 14), WIDTH - 120, item_h)
            self.item_rects.append(rect)
            hovered = rect.collidepoint(mouse)
            active = (i == self.selected)
            color = ACCENT if active else (PANEL_HI if hovered else PANEL)
            pygame.draw.rect(self.screen, color, rect, border_radius=10)
            pygame.draw.rect(self.screen, (10, 10, 14), rect, width=1, border_radius=10)
            txt_color = (10, 10, 14) if active else TEXT
            draw_text(self.screen, self.font, name, (rect.x + 22, rect.y + 14), txt_color)
            sub_color = (30, 30, 34) if active else TEXT_DIM
            mirror_txt = "full geometry (mirrored)" if extra['mirror'] else "full geometry"
            draw_text(self.screen, self.font_small,
                      f"Nz={extra['Nz']}  hz={extra['hz']} cm  (3D  mesh, {mirror_txt})",
                      (rect.x + 22, rect.y + 38), sub_color)

        run_rect = pygame.Rect(60, list_y + len(BENCHMARKS) * (item_h + 14) + 20, 240, 54)
        self.run_rect = run_rect
        draw_button(self.screen, self.font, run_rect, "VIEW GEOMETRY >", run_rect.collidepoint(mouse))

        quit_rect = pygame.Rect(320, run_rect.y, 220, 54)
        self.quit_rect = quit_rect
        draw_button(self.screen, self.font, quit_rect, "QUIT", quit_rect.collidepoint(mouse))

    # ---- draw: geometry preview ----
    def draw_geometry(self):
        self.screen.fill(BG)
        g = self.geo
        draw_text(self.screen, self.font_title, g['name'], (24, 14))
        draw_text(self.screen, self.font,
                  "Step 1 of 2 — review the reactor geometry, then solve the flux",
                  (24, 54), TEXT_DIM)

        mouse = pygame.mouse.get_pos()
        grid = g['full_map'] if (self.geo_show_full or not g['mirror']) else g['quarter_map']
        Nx, Ny = grid.shape[0], grid.shape[1]
        area = pygame.Rect(24, 110, WIDTH - 300, HEIGHT - 190)
        cell_px = max(1, min(area.width - 20, area.height - 20) // max(Nx, Ny))
        ox = area.x + (area.width - cell_px * Nx) // 2
        oy = area.y + (area.height - cell_px * Ny) // 2
        surf = pygame.Surface((Nx, Ny))
        pygame.surfarray.blit_array(surf, grid)
        surf = pygame.transform.scale(surf, (cell_px * Nx, cell_px * Ny))
        self.screen.blit(surf, (ox, oy))
        pygame.draw.rect(self.screen, (10, 10, 14), (ox, oy, cell_px * Nx, cell_px * Ny), width=2)
        draw_text(self.screen, self.font_small, f"Material map  |  mesh {Nx} x {Ny}  |  h = {g['h']} cm",
                  (area.x, area.bottom + 8), TEXT_DIM)

        bx = WIDTH - 250
        solve_rect = pygame.Rect(bx, 130, 226, 56)
        self.solve_rect = solve_rect
        draw_button(self.screen, self.font, solve_rect, "SOLVE FLUX >", solve_rect.collidepoint(mouse))

        if g['mirror']:
            cut_rect = pygame.Rect(bx, 200, 226, 48)
            self.geo_cut_rect = cut_rect
            label = "View: FULL GEOMETRY" if self.geo_show_full else "View: QUARTER (CUTAWAY)"
            draw_button(self.screen, self.font_small, cut_rect, label, cut_rect.collidepoint(mouse))
        else:
            self.geo_cut_rect = None
            draw_text(self.screen, self.font_small, "Full-core model", (bx, 214), TEXT_DIM)

        back_rect = pygame.Rect(bx, 264, 226, 46)
        self.back_rect = back_rect
        draw_button(self.screen, self.font_small, back_rect, "< BACK TO MENU", back_rect.collidepoint(mouse))

        draw_text(self.screen, self.font_small,
                  "This preview builds instantly (no solve). Click SOLVE FLUX to run the",
                  (bx, 330), TEXT_DIM)
        draw_text(self.screen, self.font_small,
                  "3D voxel multi-group diffusion solve in the background.", (bx, 350), TEXT_DIM)

    # ---- draw: running ----
    def draw_running(self):
        self.screen.fill(BG)
        name = BENCHMARKS[self.selected][0]
        draw_text(self.screen, self.font_title, "Solving...", (60, 36))
        draw_text(self.screen, self.font, name, (60, 84), TEXT_DIM)

        elapsed = time.time() - self.run_start_time
        dots = "." * (int(elapsed * 2) % 4)
        draw_text(self.screen, self.font, f"{self.get_progress()}{dots}", (60, 150), GOOD)
        draw_text(self.screen, self.font_small, f"Elapsed: {elapsed:.1f} s", (60, 182), TEXT_DIM)

        cx, cy, r = WIDTH // 2, HEIGHT // 2 + 60, 42
        angle = (elapsed * 220) % 360
        for k in range(12):
            a = np.radians(angle + k * 30)
            x1 = cx + np.cos(a) * (r - 8); y1 = cy + np.sin(a) * (r - 8)
            x2 = cx + np.cos(a) * r; y2 = cy + np.sin(a) * r
            shade = 60 + int(195 * (k / 12))
            color = (min(shade, 255), min(shade + 40, 255), 255)
            pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), 4)

        if self.error:
            draw_text(self.screen, self.font, "An error occurred:", (60, 260), BAD)
            for i, line in enumerate(self.error.split("\n")[:14]):
                draw_text(self.screen, self.font_small, line[:130], (60, 296 + i * 20), (240, 140, 140))
            back_rect = pygame.Rect(60, HEIGHT - 80, 220, 48)
            self.back_rect = back_rect
            draw_button(self.screen, self.font, back_rect, "< BACK TO MENU", back_rect.collidepoint(pygame.mouse.get_pos()))
        else:
            self.back_rect = None

        if self.result is not None and self.error is None:
            self.enter_viewer()

    # ---- 3D voxel viewer helpers ----
    def cutaway_mask(self, Ig, Jg, Nx, Ny):
        if self.cut_variant == 0:
            return np.ones_like(Ig, dtype=bool)
        center_i, center_j = Nx / 2.0, Ny / 2.0
        pos_x = Ig >= center_i
        pos_y = Jg >= center_j
        v = self.cut_variant
        if v == 1: return ~(pos_x & pos_y)
        elif v == 2: return ~(~pos_x & pos_y)
        elif v == 3: return ~(~pos_x & ~pos_y)
        else: return ~(pos_x & ~pos_y)

    def draw_3d_panel(self, field, center, rect):
        r = self.result
        Nx, Ny, Nz = r['Nx'], r['Ny'], r['Nz']
        step_xy = max(1, Nx // 40)
        step_z = max(1, Nz // 10) if Nz > 1 else 1
        idx_i = np.arange(0, Nx, step_xy)
        idx_j = np.arange(0, Ny, step_xy)
        idx_k = np.arange(0, Nz, step_z)
        Kg, Ig, Jg = np.meshgrid(idx_k, idx_i, idx_j, indexing='ij')
        vals = field[Kg, Ig, Jg]
        keep = self.cutaway_mask(Ig, Jg, Nx, Ny) & (vals >= self.density_threshold)
        K, I, J, V = Kg[keep], Ig[keep], Jg[keep], vals[keep]
        if len(V) == 0:
            return
        max_dim = max(Nx, Ny, Nz)
        Xc = (I - Nx / 2.0) / (max_dim / 2.0)
        Yc = (J - Ny / 2.0) / (max_dim / 2.0)
        Zc = (K - Nz / 2.0) / (max_dim / 2.0)
        pts = np.stack([Xc, Yc, Zc], axis=1)
        R = rot_matrix(self.yaw, self.pitch_ang)
        proj = pts @ R.T
        scale = min(rect.width, rect.height) * 0.42 * self.zoom
        depth = proj[:, 2]
        order = np.argsort(-depth)
        sx = (proj[:, 0] * scale + center[0]).astype(int)
        sy = (proj[:, 1] * scale + center[1]).astype(int)
        colors = self.lut()[field_to_lut_indices(V)]
        voxel_px = max(2, int(scale / max_dim * 1.2 * step_xy) + 1)
        clip = self.screen.get_rect()
        for idx in order:
            x, y = sx[idx], sy[idx]
            if not clip.collidepoint(x, y):
                continue
            c = colors[idx]
            pygame.draw.rect(self.screen, (int(c[0]), int(c[1]), int(c[2])),
                              (x - voxel_px // 2, y - voxel_px // 2, voxel_px, voxel_px))

    def draw_slice_panel(self, field, center, rect):
        r = self.result
        Nx, Ny = r['Nx'], r['Ny']
        layer = field[self.slice_k]
        cell_px = max(1, min(rect.width - 20, rect.height - 20) // max(Nx, Ny))
        ox = center[0] - cell_px * Nx // 2
        oy = center[1] - cell_px * Ny // 2
        surf = pygame.Surface((Nx, Ny))
        arr = self.lut()[field_to_lut_indices(layer)]
        pygame.surfarray.blit_array(surf, arr)  # arr[i,j] -> width=Nx, height=Ny (no transpose needed)
        surf = pygame.transform.scale(surf, (cell_px * Nx, cell_px * Ny))
        self.screen.blit(surf, (ox, oy))

    def draw_material_panel(self, center, rect):
        r = self.result
        Nx, Ny = r['Nx'], r['Ny']
        cell_px = max(1, min(rect.width - 20, rect.height - 20) // max(Nx, Ny))
        ox = center[0] - cell_px * Nx // 2
        oy = center[1] - cell_px * Ny // 2
        surf = pygame.Surface((Nx, Ny))
        pygame.surfarray.blit_array(surf, r['mat_color_grid'])  # same convention, no transpose needed
        surf = pygame.transform.scale(surf, (cell_px * Nx, cell_px * Ny))
        self.screen.blit(surf, (ox, oy))

    def draw_viewer(self):
        self.screen.fill(BG)
        r = self.result
        mouse = pygame.mouse.get_pos()
        draw_text(self.screen, self.font_title, r['name'], (24, 8))
        draw_text(self.screen, self.font,
                  f"k_eff = {r['k_eff']:.6f}   |   mesh {r['Nx']}x{r['Ny']}x{r['Nz']}"
                  f"   |   solve time {r['solve_time']:.1f}s   |   full mirrored geometry"
                  if r['mirrored'] else
                  f"k_eff = {r['k_eff']:.6f}   |   mesh {r['Nx']}x{r['Ny']}x{r['Nz']}"
                  f"   |   solve time {r['solve_time']:.1f}s",
                  (24, 42), GOOD)

        # ---- control bar row 1: view mode / field pair / cutaway / reset ----
        bar1_y = 72
        x = 24
        self.mode_rects = []
        for i, lbl in enumerate(self.mode_labels):
            rect = pygame.Rect(x, bar1_y, 116, 30)
            self.mode_rects.append(rect)
            draw_button(self.screen, self.font_small, rect, lbl, rect.collidepoint(mouse), i == self.mode_i)
            x += 122
        x += 16
        self.field_rects = []
        gA, gB = r['display_groups']
        pair_labels = [f"G{gA+1}/G{gB+1}", f"G{gA+1}/Power", f"G{gB+1}/Power"]
        for i, lbl in enumerate(pair_labels):
            rect = pygame.Rect(x, bar1_y, 108, 30)
            self.field_rects.append(rect)
            draw_button(self.screen, self.font_small, rect, lbl, rect.collidepoint(mouse), i == self.panel_i)
            x += 114
        x += 16
        cut_rect = pygame.Rect(x, bar1_y, 160, 30)
        self.cut_rect = cut_rect
        cut_label = "CUTAWAY: OFF" if self.cut_variant == 0 else f"CUTAWAY: Q{self.cut_variant}/4"
        draw_button(self.screen, self.font_small, cut_rect, cut_label, cut_rect.collidepoint(mouse),
                    self.cut_variant != 0)
        x += 166
        reset_rect = pygame.Rect(x, bar1_y, 90, 30)
        self.reset_rect = reset_rect
        draw_button(self.screen, self.font_small, reset_rect, "RESET", reset_rect.collidepoint(mouse))

        # ---- control bar row 2: 16-bit color palette selector + slice nav ----
        bar2_y = bar1_y + 38
        x = 24
        draw_text(self.screen, self.font_small, "Palette (16-bit):", (x, bar2_y + 6), TEXT_DIM)
        x += 142
        self.palette_rects = []
        for name in PALETTE_NAMES:
            rect = pygame.Rect(x, bar2_y, 88, 28)
            self.palette_rects.append((rect, name))
            draw_button(self.screen, self.font_small, rect, name, rect.collidepoint(mouse), name == self.palette_name)
            x += 94

        mode = self.modes[self.mode_i]
        if mode == 'slice':
            x += 24
            prev_rect = pygame.Rect(x, bar2_y, 40, 28)
            self.slice_prev_rect = prev_rect
            draw_button(self.screen, self.font_small, prev_rect, "<", prev_rect.collidepoint(mouse))
            x += 46
            draw_text(self.screen, self.font_small, f"Z slice {self.slice_k+1}/{r['Nz']}", (x, bar2_y + 6), TEXT_DIM)
            x += 116
            next_rect = pygame.Rect(x, bar2_y, 40, 28)
            self.slice_next_rect = next_rect
            draw_button(self.screen, self.font_small, next_rect, ">", next_rect.collidepoint(mouse))
        else:
            self.slice_prev_rect = self.slice_next_rect = None

        header_h = bar2_y + 42
        panel_w = WIDTH // 2
        panel_top = header_h
        panel_h = HEIGHT - header_h - 46
        rects = [pygame.Rect(0, panel_top, panel_w, panel_h),
                 pygame.Rect(panel_w, panel_top, panel_w, panel_h)]
        centers = [(panel_w // 2, panel_top + panel_h // 2),
                   (panel_w // 2 + panel_w, panel_top + panel_h // 2)]

        pair = {0: (0, 1), 1: (0, 2), 2: (1, 2)}[self.panel_i]
        titles = [self.panel_titles[pair[0]], self.panel_titles[pair[1]]]

        for p in range(2):
            field = self.fields[pair[p]]
            if mode == '3d':
                self.draw_3d_panel(field, centers[p], rects[p])
            elif mode == 'slice':
                self.draw_slice_panel(field, centers[p], rects[p])
            else:
                self.draw_material_panel(centers[p], rects[p])
            draw_text(self.screen, self.font, titles[p], (rects[p].x + 12, panel_top + 6))

        pygame.draw.line(self.screen, (60, 64, 76), (panel_w, panel_top), (panel_w, panel_top + panel_h), 2)

        info_y = HEIGHT - 34
        if mode == '3d':
            info = (f"Drag to rotate  |  wheel to zoom  |  density threshold {self.density_threshold:.2f} "
                     "( [ / ] keys )  |  ESC: back to menu")
        elif mode == 'slice':
            info = "Use the < > buttons or Up/Down keys to change the Z slice  |  ESC: back to menu"
        else:
            info = "Material map mode  |  ESC: back to menu"
        draw_text(self.screen, self.font_small, info, (24, info_y), TEXT_DIM)

        back_rect = pygame.Rect(WIDTH - 210, 16, 190, 40)
        self.back_rect = back_rect
        draw_button(self.screen, self.font_small, back_rect, "< BACK TO MENU", back_rect.collidepoint(mouse))

    # ---- events ----
    def handle_click(self, pos):
        if self.state == STATE_MENU:
            for i, rect in enumerate(self.item_rects):
                if rect.collidepoint(pos):
                    self.selected = i
            if self.run_rect.collidepoint(pos):
                self.start_geometry_preview()
            elif self.quit_rect.collidepoint(pos):
                pygame.quit(); sys.exit(0)

        elif self.state == STATE_GEOMETRY:
            if self.back_rect and self.back_rect.collidepoint(pos):
                self.state = STATE_MENU
            elif self.solve_rect and self.solve_rect.collidepoint(pos):
                self.start_solve()
            elif self.geo_cut_rect and self.geo_cut_rect.collidepoint(pos):
                self.geo_show_full = not self.geo_show_full

        elif self.state == STATE_RUNNING:
            if self.back_rect and self.back_rect.collidepoint(pos):
                self.state = STATE_MENU

        elif self.state == STATE_VIEWER:
            if self.back_rect and self.back_rect.collidepoint(pos):
                self.state = STATE_MENU
                return
            for i, rect in enumerate(self.mode_rects):
                if rect.collidepoint(pos):
                    self.mode_i = i
            for i, rect in enumerate(self.field_rects):
                if rect.collidepoint(pos):
                    self.panel_i = i
            if self.cut_rect and self.cut_rect.collidepoint(pos):
                self.cut_variant = (self.cut_variant + 1) % 5
            if self.reset_rect and self.reset_rect.collidepoint(pos):
                self.yaw, self.pitch_ang, self.zoom = 0.7, 0.35, 1.0
                self.density_threshold, self.cut_variant = 0.0, 0
            for rect, name in self.palette_rects:
                if rect.collidepoint(pos):
                    self.palette_name = name
            if self.slice_prev_rect and self.slice_prev_rect.collidepoint(pos):
                self.slice_k = max(0, self.slice_k - 1)
            if self.slice_next_rect and self.slice_next_rect.collidepoint(pos):
                self.slice_k = min(self.result['Nz'] - 1, self.slice_k + 1)

    def handle_keydown(self, key):
        if key == pygame.K_ESCAPE:
            if self.state != STATE_MENU:
                self.state = STATE_MENU
            else:
                pygame.quit(); sys.exit(0)
            return
        if self.state == STATE_MENU:
            if key in (pygame.K_UP, pygame.K_w):
                self.selected = (self.selected - 1) % len(BENCHMARKS)
            elif key in (pygame.K_DOWN, pygame.K_s):
                self.selected = (self.selected + 1) % len(BENCHMARKS)
            elif key == pygame.K_RETURN:
                self.start_geometry_preview()
        elif self.state == STATE_GEOMETRY:
            if key == pygame.K_RETURN:
                self.start_solve()
            elif key == pygame.K_c:
                self.geo_show_full = not self.geo_show_full
        elif self.state == STATE_VIEWER:
            r = self.result
            if key == pygame.K_TAB:
                self.mode_i = (self.mode_i + 1) % len(self.modes)
            elif key == pygame.K_f:
                self.panel_i = (self.panel_i + 1) % 3
            elif key == pygame.K_UP:
                self.slice_k = min(r['Nz'] - 1, self.slice_k + 1)
            elif key == pygame.K_DOWN:
                self.slice_k = max(0, self.slice_k - 1)
            elif key == pygame.K_LEFTBRACKET:
                self.density_threshold = max(0.0, self.density_threshold - 0.05)
            elif key == pygame.K_RIGHTBRACKET:
                self.density_threshold = min(0.95, self.density_threshold + 0.05)
            elif key == pygame.K_c:
                self.cut_variant = (self.cut_variant + 1) % 5
            elif key == pygame.K_p:
                pi = PALETTE_NAMES.index(self.palette_name)
                self.palette_name = PALETTE_NAMES[(pi + 1) % len(PALETTE_NAMES)]
            elif key == pygame.K_r:
                self.yaw, self.pitch_ang, self.zoom = 0.7, 0.35, 1.0
                self.density_threshold, self.cut_variant = 0.0, 0

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit(0)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.dragging = (self.state == STATE_VIEWER and self.modes[self.mode_i] == '3d')
                        self.last_mouse = event.pos
                        self.handle_click(event.pos)
                    elif event.button == 4 and self.state == STATE_VIEWER:
                        self.zoom *= 1.1
                    elif event.button == 5 and self.state == STATE_VIEWER:
                        self.zoom /= 1.1
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        dx = event.pos[0] - self.last_mouse[0]
                        dy = event.pos[1] - self.last_mouse[1]
                        self.yaw += dx * 0.01
                        self.pitch_ang += dy * 0.01
                        self.last_mouse = event.pos
                elif event.type == pygame.KEYDOWN:
                    self.handle_keydown(event.key)

            if self.state == STATE_MENU:
                self.draw_menu()
            elif self.state == STATE_GEOMETRY:
                self.draw_geometry()
            elif self.state == STATE_RUNNING:
                self.draw_running()
            elif self.state == STATE_VIEWER:
                self.draw_viewer()

            pygame.display.flip()
            self.clock.tick(30)


def main():
    App().run()


if __name__ == "__main__":
    main()