import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)

from domain import SquareDomain, CircleDomain
from fd    import FDSolver,  visualise as fd_vis
from wos   import WoSSolver, visualise as wos_vis

domain = CircleDomain(cx=0.5, cy=0.5, r=0.5)

# ---------------------------------------------------------------------------
# FD solve
# ---------------------------------------------------------------------------
fd_solver = FDSolver(domain=domain, N=256)
fd_solver.solve(max_iters=200_000, tol=1e-6, check_every=500)
fd_vis(fd_solver, title="FD — Circle domain", save_path="./img/fd_circle.png")

# ---------------------------------------------------------------------------
# WoS solve
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Walk-on-Spheres solver — Circle domain")
print("=" * 60)
GRID_RES  = 256
wos_solver = WoSSolver(domain=domain,
                       n_samples=GRID_RES * GRID_RES,
                       n_walks=20_000,
                       epsilon=1e-4,
                       max_steps=10000)
print(f"Running WoS  ({GRID_RES**2} samples, {wos_solver.n_walks} walks) …")
wos_solver.solve()
wos_vis(wos_solver, title="WoS — Circle domain", save_path="./img/wos_circle.png")

# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------
u_fd  = fd_solver.get_solution_numpy()
mask  = fd_solver.get_interior_mask_numpy() | \
        fd_solver.boundary_mask.to_numpy().astype(bool)
u_fd_masked = np.where(mask, u_fd, np.nan)

wos_vals, wos_origins = wos_solver.get_solution_numpy()
lo, hi = domain.bbox
grid_res = GRID_RES
wos_grid = np.full((grid_res, grid_res), np.nan)
for k in range(len(wos_vals)):
    ix = int((wos_origins[k, 0] - lo[0]) / (hi[0] - lo[0]) * grid_res)
    iy = int((wos_origins[k, 1] - lo[1]) / (hi[1] - lo[1]) * grid_res)
    ix = np.clip(ix, 0, grid_res - 1)
    iy = np.clip(iy, 0, grid_res - 1)
    wos_grid[iy, ix] = wos_vals[k]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ext = [lo[0], hi[0], lo[1], hi[1]]

im0 = axes[0].imshow(u_fd_masked.T, origin="lower", extent=ext,
                     cmap="hot", vmin=0, vmax=1)
plt.colorbar(im0, ax=axes[0])
axes[0].set_title("FD solution  (embedded grid)")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

im1 = axes[1].imshow(wos_grid, origin="lower", extent=ext,
                     cmap="hot", vmin=0, vmax=1)
plt.colorbar(im1, ax=axes[1])
axes[1].set_title("WoS solution  (Monte Carlo)")
axes[1].set_xlabel("x"); axes[1].set_ylabel("y")

plt.suptitle("Laplace equation on an elliptical domain", fontsize=13)
plt.tight_layout()
plt.savefig("./img/comparison_circle.png", dpi=150)
print("\nComparison figure saved to comparison_circle.png")
plt.show()
