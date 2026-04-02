"""
wos.py — Walk on Spheres (WoS) solver for the Laplace equation.

Works with any domain that implements the BaseDomain interface from
domain.py (dist_to_boundary + boundary_value).  No changes to the
solver logic are needed when switching between domains.

Usage
-----
    from domain import SquareDomain, EllipseDomain
    import taichi as ti

    ti.init(arch=ti.gpu)

    domain = EllipseDomain(cx=0.5, cy=0.5, a=0.4, b=0.3)
    solver = WoSSolver(domain, n_samples=256*256, n_walks=128,
                       epsilon=1e-4, max_steps=10000)
    solver.solve()
    values, origins = solver.get_solution_numpy()
"""

import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Walk-state dataclass
# ---------------------------------------------------------------------------
@ti.dataclass
class WalkState:
    pos        : tm.vec2
    value      : float
    step       : int
    terminated : int


# ---------------------------------------------------------------------------
# WoS Solver
# ---------------------------------------------------------------------------
@ti.data_oriented
class WoSSolver:
    """
    Monte-Carlo Walk-on-Spheres estimator for  ∇²u = 0  in Ω.

    Parameters
    ----------
    domain    : BaseDomain subclass (must have dist_to_boundary / boundary_value
                decorated as @ti.func inside a @ti.data_oriented class)
    n_samples : number of estimation points
    n_walks   : independent walks per estimation point
    epsilon   : ε-shell thickness (termination threshold)
    max_steps : hard cap on steps per walk
    """

    def __init__(self, domain, n_samples: int, n_walks: int,
                 epsilon: float, max_steps: int):
        self.domain    = domain
        self.n_samples = n_samples
        self.n_walks   = n_walks
        self.epsilon   = epsilon
        self.max_steps = max_steps

        lo_np, hi_np = domain.bbox
        self.lo = ti.Vector(lo_np.tolist())
        self.hi = ti.Vector(hi_np.tolist())

        self.origins  = ti.Vector.field(2, dtype=float, shape=n_samples)
        self.accum    = ti.field(dtype=float, shape=n_samples)
        self.n_done   = ti.field(dtype=int,   shape=n_samples)
        self.walkers  = WalkState.field(shape=n_samples)
        self.solution = ti.field(dtype=float, shape=n_samples)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    @ti.kernel
    def _init_origins(self):
        """Place sample points on a regular grid inside the bounding box."""
        lo = self.lo
        hi = self.hi
        for i in self.origins:
            side = ti.cast(ti.sqrt(float(self.n_samples)), int)
            ix   = i % side
            iy   = i // side
            dx   = (hi[0] - lo[0]) / float(side)
            dy   = (hi[1] - lo[1]) / float(side)
            self.origins[i] = tm.vec2(
                lo[0] + (ix + 0.5) * dx,
                lo[1] + (iy + 0.5) * dy,
            )

    @ti.kernel
    def _reset_accumulators(self):
        for i in range(self.n_samples):
            self.accum[i]  = 0.0
            self.n_done[i] = 0

    @ti.kernel
    def _reset_walkers(self):
        for i in range(self.n_samples):
            self.walkers[i].pos        = self.origins[i]
            self.walkers[i].value      = 0.0
            self.walkers[i].step       = 0
            self.walkers[i].terminated = 0

    # ------------------------------------------------------------------
    # Walk-on-Spheres step
    # ------------------------------------------------------------------
    @ti.func
    def _sample_on_circle(self, center: tm.vec2, radius: float) -> tm.vec2:
        angle = 2.0 * tm.pi * ti.random()
        return center + radius * tm.vec2(tm.cos(angle), tm.sin(angle))

    @ti.kernel
    def _walk_step(self):
        """Advance every non-terminated walker by one WoS step."""
        for i in range(self.n_samples):
            if self.walkers[i].terminated == 0:
                x = self.walkers[i].pos

                # Only walk from points that are inside the domain.
                # If the origin was placed outside (e.g. outside an ellipse),
                # immediately mark as terminated with value 0.
                R = self.domain.dist_to_boundary(x)

                if R < self.epsilon:
                    bv = self.domain.boundary_value(x)
                    self.walkers[i].value      = bv
                    self.walkers[i].terminated = 1
                else:
                    new_pos = self._sample_on_circle(x, R)
                    self.walkers[i].pos   = new_pos
                    self.walkers[i].step += 1

                    if self.walkers[i].step >= self.max_steps:
                        bv = self.domain.boundary_value(new_pos)
                        self.walkers[i].value      = bv
                        self.walkers[i].terminated = 1

    @ti.kernel
    def _all_terminated(self) -> int:
        all_done = 1
        for i in range(self.n_samples):
            if self.walkers[i].terminated == 0:
                all_done = 0
        return all_done

    @ti.kernel
    def _accumulate(self):
        for i in range(self.n_samples):
            self.accum[i]  += self.walkers[i].value
            self.n_done[i] += 1

    # ------------------------------------------------------------------
    # Mark exterior points (for non-convex / non-rectangular domains)
    # ------------------------------------------------------------------
    @ti.kernel
    def _mark_exterior(self):
        """
        For each origin that starts outside the domain (dist < 0 in
        signed-distance sense, or equivalently the domain returns 0 for
        both dist AND we can't walk), immediately terminate those walkers
        with value 0 so they don't pollute the accumulator.

        Heuristic: if dist_to_boundary returns a very small value AND the
        walker has never moved, it was placed outside — mark terminated.

        For the embedded-domain approach: WoS is only meaningful for
        interior sample points.  Exterior samples are set to NaN so the
        visualiser can mask them out.
        """
        for i in range(self.n_samples):
            x = self.origins[i]
            R = self.domain.dist_to_boundary(x)
            # If the sample point is exactly on or outside the boundary
            # we can't start a walk; mark immediately.
            if R <= 0.0:
                self.walkers[i].terminated = 1
                self.walkers[i].value      = 0.0

    def _run_single_walk(self, walk_idx: int):
        self._reset_walkers()
        self._mark_exterior()
        for _ in range(self.max_steps):
            self._walk_step()
            if self._all_terminated():
                break
        self._accumulate()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    @ti.kernel
    def _compute_solution(self):
        for i in range(self.n_samples):
            n = self.n_done[i]
            self.solution[i] = self.accum[i] / float(n) if n > 0 else 0.0

    def solve(self):
        self._init_origins()
        self._reset_accumulators()
        for w in range(self.n_walks):
            if w % max(1, self.n_walks // 10) == 0:
                print(f"  Walk {w:4d}/{self.n_walks}")
            self._run_single_walk(w)
        self._compute_solution()
        print("  Done.")

    def get_solution_numpy(self):
        return self.solution.to_numpy(), self.origins.to_numpy()


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------
def visualise(solver: WoSSolver, title: str = "WoS solution",
              save_path: str = "./img/wos_solution.png"):
    values, origins = solver.get_solution_numpy()
    lo_np, hi_np = solver.domain.bbox
    lo, hi = lo_np, hi_np

    grid_res = int(np.sqrt(solver.n_samples))
    grid  = np.zeros((grid_res, grid_res))
    count = np.zeros((grid_res, grid_res), dtype=int)

    for k in range(len(values)):
        ix = int((origins[k, 0] - lo[0]) / (hi[0] - lo[0]) * grid_res)
        iy = int((origins[k, 1] - lo[1]) / (hi[1] - lo[1]) * grid_res)
        ix = np.clip(ix, 0, grid_res - 1)
        iy = np.clip(iy, 0, grid_res - 1)
        grid[iy, ix] += values[k]
        count[iy, ix] += 1

    mask = count > 0
    grid[mask] /= count[mask]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].imshow(grid, origin="lower",
                        extent=[lo[0], hi[0], lo[1], hi[1]],
                        cmap="hot", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title(title)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    sc = axes[1].scatter(origins[:, 0], origins[:, 1],
                         c=values, cmap="hot", s=4, vmin=0, vmax=1)
    plt.colorbar(sc, ax=axes[1])
    axes[1].set_title("Sample-point values")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Figure saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point (square domain — mirrors original main.py behaviour)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import taichi as ti
    from domain import SquareDomain

    ti.init(arch=ti.gpu)

    GRID_RES  = 256
    N_SAMPLES = GRID_RES * GRID_RES
    N_WALKS   = 128
    EPSILON   = 1e-4
    MAX_STEPS = 10000

    domain = SquareDomain(lo=ti.Vector([0.0, 0.0]),
                          hi=ti.Vector([1.0, 1.0]))

    solver = WoSSolver(domain=domain, n_samples=N_SAMPLES, n_walks=N_WALKS,
                       epsilon=EPSILON, max_steps=MAX_STEPS)

    print(f"Running WoS  ({N_SAMPLES} samples, {N_WALKS} walks) …")
    solver.solve()
    visualise(solver, title="WoS — Square domain")
