import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import PoissonDisk

@ti.dataclass
class WalkState:
    pos        : tm.vec2
    value      : float
    step       : int
    terminated : int
    source_val : float


@ti.data_oriented
class WoSSolver:
    def __init__(self, domain, dx: float, n_walks: int,
                 epsilon: float, max_steps: int):
        self.domain    = domain
        self.n_walks   = n_walks
        self.epsilon   = epsilon
        self.max_steps = max_steps

        lo, hi = domain.bbox
        # Use Poisson Disk to generate samples
        engine = PoissonDisk(d=2, radius=dx, seed=42)
        candidates = engine.fill_space()
        candidates[:, 0] = candidates[:, 0] * (hi[0] - lo[0]) + lo[0]
        candidates[:, 1] = candidates[:, 1] * (hi[1] - lo[1]) + lo[1]

        inside = np.array([self.domain._dist_numpy(p) > 0 for p in candidates])
        pts = candidates[inside].astype(np.float32)
        
        self.n_samples = len(pts)
        print(f"  Poisson disk sampling: {self.n_samples} points generated")

        self.lo = ti.Vector(lo.tolist())
        self.hi = ti.Vector(hi.tolist())

        self.origins  = ti.Vector.field(2, dtype=float, shape=self.n_samples)
        self.accum    = ti.field(dtype=float, shape=self.n_samples)
        self.n_done   = ti.field(dtype=int,   shape=self.n_samples)
        self.walkers  = WalkState.field(shape=self.n_samples)
        self.solution = ti.field(dtype=float, shape=self.n_samples)
        self.origins.from_numpy(pts)

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
            self.walkers[i].source_val = 0.0

    @ti.func
    def _sample_on_circle(self, center: tm.vec2, radius: float) -> tm.vec2:
        angle = 2.0 * tm.pi * ti.random()
        return center + radius * tm.vec2(tm.cos(angle), tm.sin(angle))

    @ti.kernel
    def _walk_step(self):
        # one step
        for i in range(self.n_samples):
            if self.walkers[i].terminated == 0:
                x = self.walkers[i].pos
                R = self.domain.dist_to_boundary(x)

                if R < self.epsilon:
                    bv = self.domain.boundary_value(x)
                    self.walkers[i].value      = bv
                    self.walkers[i].terminated = 1
                else:
                    # source
                    N_src = 100
                    src_val_sum = 0.0

                    for k in range(N_src):
                        r = R * ti.sqrt(ti.random())
                        theta = 2.0 * tm.pi * ti.random()
                        src_point = x + r * tm.vec2(tm.cos(theta), tm.sin(theta))
                        f_val = self.domain.source(src_point)
                        # r_safe = ti.max(r, 1e-3 * R)
                        G = tm.log(R / r) / (2.0 * tm.pi)
                        src_val_sum += f_val * G
                        if ti.math.isnan(G):
                            print("NaN detected:", R, r)

                    self.walkers[i].source_val -= (src_val_sum / N_src) * (tm.pi * R * R)

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
            self.accum[i]  += self.walkers[i].value + self.walkers[i].source_val
            self.n_done[i] += 1

    # Mark exterior points for non-convex domains
    @ti.kernel
    def _mark_exterior(self):
        for i in range(self.n_samples):
            x = self.origins[i]
            R = self.domain.dist_to_boundary(x)
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

    @ti.kernel
    def _compute_solution(self):
        for i in range(self.n_samples):
            n = self.n_done[i]
            self.solution[i] = self.accum[i] / float(n) if n > 0 else 0.0

    def solve(self, check_every: int = 1000):
        self._reset_accumulators()
        for it in range(self.n_walks):
            if it % check_every == 0:
                print(f"  Walk {it:4d}/{self.n_walks}")
            self._run_single_walk(it)
        self._compute_solution()
        print("  Done.")

    def get_solution_numpy(self):
        return self.solution.to_numpy(), self.origins.to_numpy()


def visualise(solver: WoSSolver, title: str = "WoS solution",
              save_path: str = "./img/wos_solution.png"):
    values, origins = solver.get_solution_numpy()
    cmap = plt.cm.RdBu_r.copy()

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(origins[:, 0], origins[:, 1],
                    c=values, cmap=cmap, s=4)
    plt.colorbar(sc, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("x");
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='white')
    print(f"Figure saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    import taichi as ti
    from domain import SquareDomain

    ti.init(arch=ti.gpu)

    radius = 1 / 256
    N_WALKS   = 2000
    EPSILON   = 1e-4
    MAX_STEPS = 10000

    domain = SquareDomain(lo=ti.Vector([0.0, 0.0]),
                          hi=ti.Vector([1.0, 1.0]))

    solver = WoSSolver(domain=domain, dx=radius, n_walks=N_WALKS,
                       epsilon=EPSILON, max_steps=MAX_STEPS)

    print(f"Running WoS  ({solver.n_samples} samples, {N_WALKS} walks) …")
    solver.solve()
    visualise(solver, title="WoS — Square domain")
