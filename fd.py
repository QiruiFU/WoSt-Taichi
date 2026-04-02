"""
fd.py — Finite Difference (FD) solver for the Laplace equation.

Uses a red-black Gauss-Seidel scheme on a uniform grid covering the
bounding box of the domain.  Supports arbitrary domain shapes via the
embedded-domain (fictitious-domain) approach:

  • The grid always covers the domain's bounding box uniformly.
  • domain.grid_info(N) provides three arrays:
        interior_mask  — nodes where G-S updates are applied
        boundary_mask  — nodes where Dirichlet BCs are imposed
        bc_values      — prescribed values at boundary nodes
  • Exterior nodes (outside the actual domain) are frozen at 0 and
    never updated.  The 5-point stencil is unchanged; only the
    conditional update guard changes.

For a SquareDomain this is identical in behaviour to the original fd.py.
For an EllipseDomain the bounding-box grid is used and interior/exterior
nodes are distinguished automatically by the domain class.

Usage
-----
    from domain import SquareDomain, EllipseDomain
    import taichi as ti

    ti.init(arch=ti.gpu)

    domain = EllipseDomain(cx=0.5, cy=0.5, a=0.4, b=0.3)
    solver = FDSolver(domain, N=256)
    solver.solve()
"""

import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# FD Solver
# ---------------------------------------------------------------------------
@ti.data_oriented
class FDSolver:
    """
    Solves ∇²u = 0 on an arbitrary domain using red-black Gauss-Seidel
    on a uniform grid embedded in the domain's bounding box.

    Parameters
    ----------
    domain : BaseDomain subclass  (provides grid_info)
    N      : number of *interior* grid points per side
             Total grid is (N+2)×(N+2) including boundary / ghost layer.
    """

    def __init__(self, domain, N: int):
        self.domain = domain
        self.N = N
        M = N + 2

        # Build masks from the domain (pure Python / NumPy)
        print("  Building grid masks from domain …")
        int_mask, bnd_mask, bc_vals = domain.grid_info(N)

        # Solution field
        self.u = ti.field(dtype=float, shape=(M, M))

        # Masks and BC values as Taichi fields
        self.interior_mask = ti.field(dtype=int, shape=(M, M))
        self.boundary_mask = ti.field(dtype=int, shape=(M, M))
        self.bc_values     = ti.field(dtype=float, shape=(M, M))

        # Upload to GPU
        self.interior_mask.from_numpy(int_mask)
        self.boundary_mask.from_numpy(bnd_mask)
        self.bc_values.from_numpy(bc_vals)

        # Residual scalar
        self.residual = ti.field(dtype=float, shape=())

        lo, hi = domain.bbox
        self.h = float((hi[0] - lo[0]) / (N + 1))

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    @ti.kernel
    def _init(self):
        """Set boundary nodes to their prescribed values; interior to 0."""
        for i, j in self.u:
            if self.boundary_mask[i, j] == 1:
                self.u[i, j] = self.bc_values[i, j]
            else:
                self.u[i, j] = 0.0

    # ------------------------------------------------------------------
    # Red-black Gauss-Seidel sweep
    # ------------------------------------------------------------------
    @ti.kernel
    def _gs_sweep(self, color: int):
        """
        Update all *interior* nodes of the given color.
        Nodes outside the domain (interior_mask == 0) are skipped.
        The 5-point stencil is unmodified.
        """
        N = self.N
        for i, j in ti.ndrange((1, N + 1), (1, N + 1)):
            if (i + j) % 2 == color and self.interior_mask[i, j] == 1:
                self.u[i, j] = 0.25 * (
                    self.u[i - 1, j] +
                    self.u[i + 1, j] +
                    self.u[i, j - 1] +
                    self.u[i, j + 1]
                )

    # ------------------------------------------------------------------
    # Residual (max-norm of ∇²u at interior nodes)
    # ------------------------------------------------------------------
    @ti.kernel
    def _compute_residual(self):
        self.residual[None] = 0.0
        h = self.h
        N = self.N
        for i, j in ti.ndrange((1, N + 1), (1, N + 1)):
            if self.interior_mask[i, j] == 1:
                lap = (self.u[i - 1, j] + self.u[i + 1, j] +
                       self.u[i, j - 1] + self.u[i, j + 1] -
                       4.0 * self.u[i, j]) / (h * h)
                ti.atomic_max(self.residual[None], ti.abs(lap))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def solve(self, max_iters: int = 200_000,
              tol: float = 1e-6, check_every: int = 200):
        self._init()
        lo, hi = self.domain.bbox
        print(f"  Grid: {self.N}×{self.N} interior nodes  "
              f"(mesh spacing h = {self.h:.4f})")
        print(f"  Bounding box: [{lo[0]:.2f},{hi[0]:.2f}] × "
              f"[{lo[1]:.2f},{hi[1]:.2f}]")

        for it in range(1, max_iters + 1):
            self._gs_sweep(0)
            self._gs_sweep(1)

            if it % check_every == 0:
                self._compute_residual()
                res = self.residual[None]
                print(f"  iter {it:6d}  |residual|∞ = {res:.3e}")
                if res < tol:
                    print(f"  Converged at iteration {it}.")
                    break
        else:
            print(f"  Reached max iterations ({max_iters}).")

    def get_solution_numpy(self) -> np.ndarray:
        """Return the (N+2)×(N+2) solution array."""
        return self.u.to_numpy()

    def get_interior_mask_numpy(self) -> np.ndarray:
        return self.interior_mask.to_numpy().astype(bool)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def visualise(solver: FDSolver, title: str = "FD solution",
              save_path: str = "./img/fd_solution.png"):
    u_np   = solver.get_solution_numpy()       # (N+2, N+2)
    mask   = solver.get_interior_mask_numpy()  # (N+2, N+2) bool

    lo, hi = solver.domain.bbox

    # Mask exterior nodes so they show as white / transparent
    u_plot = np.where(mask | solver.boundary_mask.to_numpy().astype(bool),
                      u_np, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        u_plot.T, origin="lower", extent=[lo[0], hi[0], lo[1], hi[1]],
        cmap="hot", vmin=0, vmax=1,
    )
    plt.colorbar(im, ax=ax)
    ax.set_title(f"{title}  ({solver.N}×{solver.N} grid)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Figure saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point (square domain — identical to original fd.py behaviour)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import taichi as ti
    from domain import SquareDomain

    ti.init(arch=ti.gpu)

    GRID_RES    = 256
    MAX_ITERS   = 200_000
    TOL         = 1e-6
    CHECK_EVERY = 200

    domain = SquareDomain(lo=ti.Vector([0.0, 0.0]),
                          hi=ti.Vector([1.0, 1.0]))

    solver = FDSolver(domain=domain, N=GRID_RES)
    print("Running Finite Difference solver  (red-black Gauss-Seidel) …")
    solver.solve(max_iters=MAX_ITERS, tol=TOL, check_every=CHECK_EVERY)
    visualise(solver, title="FD — Square domain")
