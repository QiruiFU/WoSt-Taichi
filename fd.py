import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt


@ti.data_oriented
class FDSolver:
    def __init__(self, domain, N: int):
        self.domain = domain
        self.N = N
        M = N + 2

        print("  Building grid masks from domain …")
        int_mask, bnd_mask, bc_type, bc_vals, bc_norms, source_vals = domain.grid_info(N)

        self.u = ti.field(dtype=float, shape=(M, M))

        self.interior_mask = ti.field(dtype=int,   shape=(M, M))
        self.boundary_mask = ti.field(dtype=int,   shape=(M, M))
        self.bc_values     = ti.field(dtype=float, shape=(M, M))
        self.source        = ti.field(dtype=float, shape=(M, M))
        self.bc_type       = ti.field(dtype=int,   shape=(M, M))
        self.bc_normals    = ti.Vector.field(2, dtype=float, shape=(M, M))

        self.residual      = ti.field(dtype=float, shape=())

        self.interior_mask.from_numpy(int_mask)
        self.boundary_mask.from_numpy(bnd_mask)
        self.bc_values.from_numpy(bc_vals)
        self.source.from_numpy(source_vals)
        self.bc_type.from_numpy(bc_type)
        self.bc_normals.from_numpy(bc_norms)

        lo, hi = domain.bbox
        self.h = float((hi[0] - lo[0]) / (N + 1))

    @ti.kernel
    def _init(self):
        for i, j in self.u:
            if self.boundary_mask[i, j] == 1 and self.bc_type[i, j] == 0:
                self.u[i, j] = self.bc_values[i, j]
            else:
                self.u[i, j] = 0.0

    @ti.kernel
    def _gs_sweep(self, color: int):
        N = self.N
        h = self.h
        for i, j in ti.ndrange((0, N + 2), (0, N + 2)):
            if (i + j) % 2 == color:
                if self.interior_mask[i, j] == 1:
                    self.u[i, j] = 0.25 * (
                        self.u[i - 1, j] +
                        self.u[i + 1, j] +
                        self.u[i, j - 1] +
                        self.u[i, j + 1] -
                        h * h * self.source[i, j]
                    )
                elif self.boundary_mask[i, j] == 1 and self.bc_type[i, j] == 1:
                    # Neumann Boundary
                    nx = self.bc_normals[i, j][0]
                    ny = self.bc_normals[i, j][1]
                    g  = self.bc_values[i, j]
                    di = -ti.round(nx)
                    dj = -ti.round(ny)
                    ii = i + ti.cast(di, int)
                    jj = j + ti.cast(dj, int)
                    self.u[i, j] = self.u[ii, jj] - h * g

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
                r = lap - self.source[i, j]
                ti.atomic_max(self.residual[None], ti.abs(r))

    def solve(self, max_iters: int = 200_000,
              tol: float = 1e-6, check_every: int = 1000):
        self._init()
        lo, hi = self.domain.bbox
        print(f"  Grid: {self.N}*{self.N} interior nodes  "
              f"(mesh spacing h = {self.h:.4f})")
        print(f"  Bounding box: [{lo[0]:.2f},{hi[0]:.2f}] * "
              f"[{lo[1]:.2f},{hi[1]:.2f}]")

        for it in range(max_iters):
            self._gs_sweep(0)
            self._gs_sweep(1)

            if it % check_every == 0:
                self._compute_residual()
                res = self.residual[None]
                print(f"  iter {it:6d}  |residual| = {res:.3e}")
                if res < tol:
                    print(f"  Converged at iteration {it}.")
                    break

    def get_solution_numpy(self) -> np.ndarray:
        return self.u.to_numpy()

    def get_interior_mask_numpy(self) -> np.ndarray:
        return self.interior_mask.to_numpy().astype(bool)


def visualise(solver: FDSolver, title: str = "FD solution",
              save_path: str = "./img/fd_solution.png"):
    u_np   = solver.get_solution_numpy()
    mask   = solver.get_interior_mask_numpy()

    lo, hi = solver.domain.bbox

    valid = mask | solver.boundary_mask.to_numpy().astype(bool)

    nx, ny = u_np.shape
    xs = np.linspace(lo[0], hi[0], nx)
    ys = np.linspace(lo[1], hi[1], ny)
    X, Y = np.meshgrid(xs, ys, indexing='ij')

    X = X[valid]
    Y = Y[valid]
    values = u_np[valid]

    cmap = plt.cm.RdBu_r.copy()

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(X, Y,
                    c=values,
                    cmap=cmap,
                    s=4)

    plt.colorbar(sc, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='white')
    print(f"Figure saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    import taichi as ti
    from domain import SquareDomain, CircleDomain

    ti.init(arch=ti.gpu)

    GRID_RES    = 256
    MAX_ITERS   = 200_000
    TOL         = 1e-2
    CHECK_EVERY = 2000

    domain = SquareDomain(lo=ti.Vector([0.0, 0.0]),
                          hi=ti.Vector([1.0, 1.0]))
    # domain = CircleDomain()

    solver = FDSolver(domain=domain, N=GRID_RES)
    print("Running Finite Difference solver  (red-black Gauss-Seidel) …")
    solver.solve(max_iters=MAX_ITERS, tol=TOL, check_every=CHECK_EVERY)
    visualise(solver, title="FD — Square domain")