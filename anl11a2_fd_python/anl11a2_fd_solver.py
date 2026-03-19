"""
ANL 11-A2 (2D IAEA) Benchmark — Pure Python Finite-Difference Solver
Two-group neutron diffusion k-eigenvalue problem.
NumPy/SciPy/Matplotlib.
Author: Emil Mammadzada
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time, os

# ---------------------------------------------------------------------------
# 1. Grid
# ---------------------------------------------------------------------------
h = 1.0; Lx = 170.0; Ly = 130.0
Nx = int(round(Lx/h)); Ny = int(round(Ly/h))
xc = (np.arange(Nx)+0.5)*h; yc = (np.arange(Ny)+0.5)*h
XX,YY = np.meshgrid(xc,yc,indexing='ij')
print(f"Grid: {Nx}×{Ny}, h={h} cm")

# ---------------------------------------------------------------------------
# 2. Material map
# ---------------------------------------------------------------------------
VOID=-1; FUEL1=0; FUEL2=1; FUELR=2; REFL=3

def pip(px,py,poly):
    inside=np.zeros(px.shape,dtype=bool); n=len(poly); j=n-1
    for i in range(n):
        xi,yi=poly[i]; xj,yj=poly[j]
        c=((yi>py)!=(yj>py))&(px<(xj-xi)*(py-yi)/(yj-yi+1e-300)+xi)
        inside^=c; j=i
    return inside

poly_s1=np.array([[70,70],[10,10],[10,0],[70,0],[70,10],[90,10],[90,0],[130,0],[130,30],[110,30],[110,70],[90,70]],float)
poly_s2=np.array([[90,90],[90,70],[110,70],[110,30],[130,30],[130,0],[150,0],[150,50],[130,50],[130,90],[110,90],[110,110]],float)
poly_s6=np.array([[110,110],[130,130],[130,110],[150,110],[150,70],[170,70],[170,0],[150,0],[150,50],[130,50],[130,90],[110,90]],float)

in_s3=(XX>=0)&(XX<=10)&(YY>=0)&(YY<=XX)
in_s4=(XX>=70)&(XX<=90)&(YY>=0)&(YY<=10)
in_s5=(XX>=70)&(XX<=90)&(YY>=70)&(YY<=XX)
in_s1=pip(XX,YY,poly_s1); in_s2=pip(XX,YY,poly_s2); in_s6=pip(XX,YY,poly_s6)

region_map=np.full((Nx,Ny),VOID,dtype=np.int8)
region_map[in_s6]=REFL; region_map[in_s2]=FUEL2; region_map[in_s1]=FUEL1
region_map[in_s3|in_s4|in_s5]=FUELR

active=region_map>=0; N=int(active.sum())
print(f"Active cells: {N}  (F1={int((region_map==FUEL1).sum())}, F2={int((region_map==FUEL2).sum())}, "
      f"FR={int((region_map==FUELR).sum())}, RF={int((region_map==REFL).sum())})")
dof=np.full((Nx,Ny),-1,dtype=np.int32); dof[active]=np.arange(N,dtype=np.int32)

# ---------------------------------------------------------------------------
# 3. Cross sections  [FUEL1, FUEL2, FUELR, REFL]
# ---------------------------------------------------------------------------
D_m  =[[1.5,1.5,1.5,2.0],[0.4,0.4,0.4,0.3]]
xa_m =[[0.010,0.010,0.010,0.000],[0.085,0.080,0.130,0.010]]
nf_m =[[0.000,0.000,0.000,0.000],[0.135,0.135,0.135,0.000]]
ch_m =[[1.,1.,1.,1.],[0.,0.,0.,0.]]
xs12 = np.array([0.02,0.02,0.02,0.04]); xs21=np.array([0.,0.,0.,0.])
#B2z  = 0.8e-4; albedo=np.array([0.4692,0.4692])
B2z  = 0.8e-4; albedo=np.array([0.0, 0.0])

def g2d(arr):
    out=np.zeros((Nx,Ny))
    for m in range(4): out[region_map==m]=arr[m]
    return out

Dg=[g2d(D_m[g]) for g in range(2)]; xa=[g2d(xa_m[g]) for g in range(2)]
nuf=[g2d(nf_m[g]) for g in range(2)]; chi=[g2d(ch_m[g]) for g in range(2)]
XS12=g2d(xs12); XS21=g2d(xs21)

# ---------------------------------------------------------------------------
# 4. Matrix assembly
# ---------------------------------------------------------------------------
def buildA(g):
    D=Dg[g]
    rows=[]; cols=[]; vals=[]; diag=np.zeros(N)

    # x-faces (vectorised)
    Dxh=2*D[:-1,:]*D[1:,:]/(D[:-1,:]+D[1:,:]+1e-300)
    ii,jj=np.where(active[:-1,:]&active[1:,:])
    c=Dxh[ii,jj]/h**2
    dL=dof[ii,jj]; dR=dof[ii+1,jj]
    rows.extend(dL.tolist()+dR.tolist()); cols.extend(dR.tolist()+dL.tolist())
    vals.extend((-c).tolist()+(-c).tolist())
    np.add.at(diag,dL,c); np.add.at(diag,dR,c)

    # y-faces (vectorised)
    Dyh=2*D[:,:-1]*D[:,1:]/(D[:,:-1]+D[:,1:]+1e-300)
    ii,jj=np.where(active[:,:-1]&active[:,1:])
    c=Dyh[ii,jj]/h**2
    dB=dof[ii,jj]; dT=dof[ii,jj+1]
    rows.extend(dB.tolist()+dT.tolist()); cols.extend(dT.tolist()+dB.tolist())
    vals.extend((-c).tolist()+(-c).tolist())
    np.add.at(diag,dB,c); np.add.at(diag,dT,c)

    # Removal xs (vectorised)
    rem=xa[g]+D*B2z+(XS12 if g==0 else XS21)
    ii,jj=np.where(active)
    np.add.at(diag,dof[ii,jj],rem[ii,jj])

    # Albedo Robin BC (only γ/h, NO D/h^2):
    # Symmetry at i=0 (left) and j=0 (bottom) → skip.
    # Void on any other boundary face → γ/h on diagonal.
    gam=albedo[g]
    ii,jj=np.where(active)
    for k in range(len(ii)):
        i,j=ii[k],jj[k]; d=dof[i,j]
        # left: symmetry if i==0, else void if not active[i-1,j]
        if i>0 and not active[i-1,j]: diag[d]+=gam/h
        # right: void
        if i==Nx-1 or not active[i+1,j]: diag[d]+=gam/h
        # bottom: symmetry if j==0, else void
        if j>0 and not active[i,j-1]: diag[d]+=gam/h
        # top: void
        if j==Ny-1 or not active[i,j+1]: diag[d]+=gam/h

    A=sp.csr_matrix((vals,(rows,cols)),shape=(N,N))+sp.diags(diag)
    return A

def diagM(arr2d):
    ii,jj=np.where(active); d=np.zeros(N)
    d[dof[ii,jj]]=arr2d[ii,jj]
    return sp.diags(d)

print("Assembling matrices …",end=' ',flush=True); t0=time.time()
A0=buildA(0); A1=buildA(1)
S12=diagM(-XS12); S21=diagM(-XS21)
F=[[diagM(chi[gt]*nuf[gs]) for gs in range(2)] for gt in range(2)]
Mop=sp.bmat([[A0,S21],[S12,A1]],format='csr')
Fop=sp.bmat([[F[0][0],F[0][1]],[F[1][0],F[1][1]]],format='csr')
print(f"done ({time.time()-t0:.2f}s), system {2*N}×{2*N}")

# ---------------------------------------------------------------------------
# 5. Inverse power iteration
# ---------------------------------------------------------------------------
print("LU factorisation …",end=' ',flush=True); t0=time.time()
lu=spla.splu(Mop.tocsc()); print(f"done ({time.time()-t0:.2f}s)")

phi=np.ones(2*N); phi[:N]*=1.0; phi[N:]*=0.5; phi/=np.linalg.norm(phi)
k_eff=1.0; tol=1e-8

print("Power iteration:")
t0=time.time()
for it in range(1,601):
    rhs=(1.0/k_eff)*(Fop@phi)
    pn=lu.solve(rhs)
    fn=Fop@pn; fo=Fop@phi
    k_new=k_eff*np.linalg.norm(fn)/np.linalg.norm(fo)
    pn/=np.linalg.norm(pn)
    dk=abs(k_new-k_eff)/abs(k_new)
    k_eff=k_new; phi=pn
    if it%25==0 or dk<tol:
        print(f"  {it:4d}:  k={k_eff:.7f}  |dk/k|={dk:.3e}")
    if dk<tol:
        print(f"Converged in {it} iters ({time.time()-t0:.1f}s)"); break
else:
    print("Max iters reached")

print(f"\nk_eff = {k_eff:.6f}  (ANL reference ≈ 1.02981)")

# ---------------------------------------------------------------------------
# 6. Reshape & plot
# ---------------------------------------------------------------------------
phi1=np.zeros((Nx,Ny)); phi2=np.zeros((Nx,Ny))
ii,jj=np.where(active)
phi1[ii,jj]=phi[dof[ii,jj]]; phi2[ii,jj]=phi[N+dof[ii,jj]]
phi1/=phi1.max(); phi2/=phi2.max()

os.makedirs('outputs', exist_ok=True)
ext=[0,Lx,0,Ly]

fig,axes=plt.subplots(1,2,figsize=(14,6))
i0=axes[0].imshow(phi1.T,origin='lower',extent=ext,cmap='viridis',vmin=0,vmax=1,aspect='equal')
axes[0].set_title('Fast Flux φ₁ (normalised)',fontsize=13); axes[0].set_xlabel('x [cm]'); axes[0].set_ylabel('y [cm]')
plt.colorbar(i0,ax=axes[0])
i1=axes[1].imshow(phi2.T,origin='lower',extent=ext,cmap='magma',vmin=0,vmax=1,aspect='equal')
axes[1].set_title('Thermal Flux φ₂ (normalised)',fontsize=13); axes[1].set_xlabel('x [cm]'); axes[1].set_ylabel('y [cm]')
plt.colorbar(i1,ax=axes[1])
plt.suptitle(f'ANL 11-A2  |  h={h} cm  |  k_eff={k_eff:.5f}  (ref≈1.02981)',fontsize=12)
plt.tight_layout(); plt.savefig('anl11a2_fluxes.png',dpi=150,bbox_inches='tight')
print("Saved anl11a2_fluxes.png")

fig2,ax2=plt.subplots(2,2,figsize=(14,10))
mx=active[:,0]
ax2[0,0].plot(xc[mx],phi1[mx,0],'r-',lw=2,label='FD solver'); ax2[0,0].set_ylabel(r'$\tilde\phi_1$ at $y=0$',fontsize=13); ax2[0,0].set_xlim(0,170); ax2[0,0].set_ylim(0,1.02); ax2[0,0].grid(True,ls='--',alpha=0.5); ax2[0,0].legend()
ax2[0,1].plot(xc[mx],phi2[mx,0],'b-',lw=2,label='FD solver'); ax2[0,1].set_ylabel(r'$\tilde\phi_2$ at $y=0$',fontsize=13); ax2[0,1].set_xlim(0,170); ax2[0,1].set_ylim(0,1.02); ax2[0,1].grid(True,ls='--',alpha=0.5); ax2[0,1].legend()
dv,p1d,p2d=[],[],[]
for i in range(min(Nx,Ny)):
    if active[i,i]: dv.append(xc[i]); p1d.append(phi1[i,i]); p2d.append(phi2[i,i])
dv=np.array(dv); p1d=np.array(p1d)/max(p1d); p2d=np.array(p2d)/max(p2d)
ax2[1,0].plot(dv,p1d,'r-',lw=2,label='FD solver'); ax2[1,0].set_xlabel('y [cm]',fontsize=12); ax2[1,0].set_ylabel(r'$\tilde\phi_1$ at $y=x$',fontsize=13); ax2[1,0].set_xlim(0,140); ax2[1,0].set_ylim(0,1.02); ax2[1,0].grid(True,ls='--',alpha=0.5); ax2[1,0].legend()
ax2[1,1].plot(dv,p2d,'b-',lw=2,label='FD solver'); ax2[1,1].set_xlabel('y [cm]',fontsize=12); ax2[1,1].set_ylabel(r'$\tilde\phi_2$ at $y=x$',fontsize=13); ax2[1,1].set_xlim(0,140); ax2[1,1].set_ylim(0,1.02); ax2[1,1].grid(True,ls='--',alpha=0.5); ax2[1,1].legend()
plt.suptitle(f'ANL 11-A2 Flux Profiles  (h={h} cm, k_eff={k_eff:.5f})',fontsize=12)
plt.tight_layout(); plt.savefig('anl11a2_profiles.png',dpi=150,bbox_inches='tight')
print("Saved anl11a2_profiles.png")

fig3,ax3=plt.subplots(figsize=(9,7))
cm4=matplotlib.colors.ListedColormap(['#EEEEEE','#2196F3','#4CAF50','#FF5722','#9C27B0'])
norm=matplotlib.colors.BoundaryNorm([-1.5,-0.5,0.5,1.5,2.5,3.5],cm4.N)
ax3.imshow(region_map.T.astype(float),origin='lower',extent=ext,cmap=cm4,norm=norm,aspect='equal')
patches=[mpatches.Patch(color='#EEEEEE',label='VOID'),mpatches.Patch(color='#2196F3',label='Fuel-1'),
         mpatches.Patch(color='#4CAF50',label='Fuel-2'),mpatches.Patch(color='#FF5722',label='Fuel-Rod'),
         mpatches.Patch(color='#9C27B0',label='Reflector')]
ax3.legend(handles=patches,loc='upper right',fontsize=10)
ax3.set_title('ANL 11-A2 — Material Map',fontsize=13); ax3.set_xlabel('x [cm]'); ax3.set_ylabel('y [cm]')
plt.tight_layout(); plt.savefig('anl11a2_material_map.png',dpi=150,bbox_inches='tight')
print("Saved anl11a2_material_map.png")
plt.close('all'); print("Done.")


import pandas as pd

# 1. Load and Prepare Excel Data
# Note: Make sure the values in Excel are in cm (check whether dividing by 1000 is necessary depending on your needs)
x_axis_data   = pd.read_excel('data.xlsx', sheet_name='x-axis').to_numpy() 
diagonal_data = pd.read_excel('data.xlsx', sheet_name='Diagonal').to_numpy()

# 2. Prepare Diagonal Data for the FD Solver
mx = active[:, 0]
dv, p1d, p2d = [], [], []
for i in range(min(Nx, Ny)):
    if active[i, i]:
        dv.append(xc[i])
        p1d.append(phi1[i, i])
        p2d.append(phi2[i, i])

dv = np.array(dv)
# Normalization (with respect to the peak value)
p1d_norm = p1d / max(p1d)
p2d_norm = p2d / max(p2d)

# visualization
fig, ax = plt.subplots(2, 2, figsize=(15, 12))
mark_size = 6
ls = 2

# --- SUBPLOT (0,0): Phi1 y=0 ---
# Argonne Data (Excel)
ax[0,0].plot(x_axis_data[:, 0]/ 1000, x_axis_data[:, 1] / max(x_axis_data[:, 1]), '^', 
             c='orange', label='Argonne (Benchmark)', markersize=mark_size)
# FD Solver Data
ax[0,0].plot(xc[mx], phi1[mx, 0] / max(phi1[mx, 0]), 'r-', lw=ls, label='FD Solver')
ax[0,0].set_ylabel(r'$\tilde\phi_1$ at $y=0$', fontsize=14)
ax[0,0].set_xlim(0, 170); ax[0,0].set_ylim(0, 1.05)

# --- SUBPLOT (0,1): Phi2 y=0 ---
ax[0,1].plot(x_axis_data[:, 0]/ 1000, x_axis_data[:, 2] / max(x_axis_data[:, 2]), '^', 
             c='g', label='Argonne (Benchmark)', markersize=mark_size)
ax[0,1].plot(xc[mx], phi2[mx, 0] / max(phi2[mx, 0]), 'b-', lw=ls, label='FD Solver')
ax[0,1].set_ylabel(r'$\tilde\phi_2$ at $y=0$', fontsize=14)
ax[0,1].set_xlim(0, 170); ax[0,1].set_ylim(0, 1.05)

# --- SUBPLOT (1,0): Phi1 y=x (Diagonal) ---
ax[1,0].plot(diagonal_data[:, 0]/ 1000, diagonal_data[:, 2] / max(diagonal_data[:, 2]), '^', 
             c='orange', label='Argonne (Benchmark)', markersize=mark_size)
ax[1,0].plot(dv, p1d_norm, 'r-', lw=ls, label='FD Solver')
ax[1,0].set_xlabel('Distance [cm]', fontsize=12)
ax[1,0].set_ylabel(r'$\tilde\phi_1$ at $y=x$', fontsize=14)
ax[1,0].set_xlim(0, 140); ax[1,0].set_ylim(0, 1.05)

# --- SUBPLOT (1,1): Phi2 y=x (Diagonal) ---
ax[1,1].plot(diagonal_data[:, 0]/ 1000, diagonal_data[:, 3] / max(diagonal_data[:, 3]), '^', 
             c='g', label='Argonne (Benchmark)', markersize=mark_size)
ax[1,1].plot(dv, p2d_norm, 'b-', lw=ls, label='FD Solver')
ax[1,1].set_xlabel('Distance [cm]', fontsize=12)
ax[1,1].set_ylabel(r'$\tilde\phi_2$ at $y=x$', fontsize=14)
ax[1,1].set_xlim(0, 140); ax[1,1].set_ylim(0, 1.05)

# Same settings
for a in ax.flat:
    a.grid(True, ls='--', alpha=0.6)
    a.legend(loc='best', fontsize=10)

plt.suptitle(f'Comparison: FD Solver vs Argonne Benchmark\n(k_eff={k_eff:.5f})', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#plt.show()
plt.savefig('comparison.png', dpi=300, bbox_inches='tight')
print("Done Saved!!!")