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
    on_neumann : int
    normal     : tm.vec2 # only valid when on_neumann


@ti.data_oriented
class WoStSolver:
    def __init__(self, domain, dx: float, n_walks: int,
                 epsilon: float, max_steps: int):
        self.domain    = domain
        self.n_walks   = n_walks
        self.epsilon   = epsilon
        self.max_steps = max_steps

        lo, hi = domain.bbox
        engine = PoissonDisk(d=2, radius=dx, seed=42)
        candidates = engine.fill_space()
        candidates[:, 0] = candidates[:, 0] * (hi[0] - lo[0]) + lo[0]
        candidates[:, 1] = candidates[:, 1] * (hi[1] - lo[1]) + lo[1]

        inside = np.array([self.domain.dist_numpy(p) > 0 for p in candidates])
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
    def reset_accum(self):
        for i in range(self.n_samples):
            self.accum[i]  = 0.0
            self.n_done[i] = 0

    @ti.kernel
    def reset_walkers(self):
        for i in range(self.n_samples):
            self.walkers[i].pos        = self.origins[i]
            self.walkers[i].value      = 0.0
            self.walkers[i].step       = 0
            self.walkers[i].terminated = 0
            self.walkers[i].source_val = 0.0
            self.walkers[i].on_neumann = 0
            self.walkers[i].normal     = tm.vec2(0.0, 0.0)

    @ti.func
    def sample_direction(self, on_neumann: int, normal: tm.vec2) -> tm.vec2:
        v = tm.vec2(0.0, 0.0)
        if on_neumann == 0:
            angle = 2.0 * tm.pi * ti.random()
            v = tm.vec2(tm.cos(angle), tm.sin(angle))
        else:
            angle = (ti.random() - 0.5) * tm.pi  # [-pi/2, pi/2]
            inward = -normal
            tangent = tm.vec2(-inward[1], inward[0]) # rotate 90 degrees anti-clockwise
            v = tm.cos(angle) * inward + tm.sin(angle) * tangent
        return v

    @ti.kernel
    def walk_step(self):
        for i in range(self.n_samples):
            if self.walkers[i].terminated == 0:
                x = self.walkers[i].pos
                on_neumann = self.walkers[i].on_neumann
                normal = self.walkers[i].normal

                r_D = self.domain.dist_to_dirichlet(x)
                r_S = self.domain.dist_to_silhouette(x)
                R = tm.min(r_S, r_D)

                if r_D < self.epsilon:
                    self.walkers[i].value = self.domain.boundary_value(x)
                    self.walkers[i].terminated = 1
                else:
                    # source 积分（与原版相同，保留不动）
                    # N_src       = 100
                    # src_val_sum = 0.0
                    # for k in range(N_src):
                    #     r         = R * ti.sqrt(ti.random())
                    #     theta     = 2.0 * tm.pi * ti.random()
                    #     src_point = x + r * tm.vec2(tm.cos(theta), tm.sin(theta))
                    #     f_val     = self.domain.source(src_point)
                    #     G         = tm.log(R / r) / (2.0 * tm.pi)
                    #     src_val_sum += f_val * G
                    # self.walkers[i].source_val -= (src_val_sum / N_src) * (tm.pi * R * R)

                    v = self.sample_direction(on_neumann, normal)
                    next_step, on_neumann, normal = self.domain.intersect_ray(x, v, R)
                    self.walkers[i].pos = next_step
                    self.walkers[i].step += 1

                    if self.walkers[i].step >= self.max_steps:
                        print("Step limit reached.")
                        self.walkers[i].value = self.domain.boundary_value(next_step)
                        self.walkers[i].terminated = 1

    @ti.kernel
    def all_terminated(self) -> int:
        all_done = 1
        for i in range(self.n_samples):
            if self.walkers[i].terminated == 0:
                all_done = 0
        return all_done

    @ti.kernel
    def accumulate(self):
        for i in range(self.n_samples):
            self.accum[i]  += self.walkers[i].value + self.walkers[i].source_val
            self.n_done[i] += 1

    def run_single_walk(self, walk_idx: int):
        self.reset_walkers()
        for _ in range(self.max_steps):
            self.walk_step()
            if self.all_terminated():
                break
        self.accumulate()

    @ti.kernel
    def compute_solution(self):
        for i in range(self.n_samples):
            n = self.n_done[i]
            self.solution[i] = self.accum[i] / float(n) if n > 0 else 0.0

    def solve(self, check_every: int = 1000):
        self.reset_accum()
        for it in range(self.n_walks):
            if it % check_every == 0:
                print(f"  Walk {it:4d}/{self.n_walks}")
            self.run_single_walk(it)
        self.compute_solution()
        print("  Done.")

    def get_solution_numpy(self):
        return self.solution.to_numpy(), self.origins.to_numpy()


def visualise(solver: WoStSolver, title: str = "WoSt solution",
              save_path: str = "./img/wost_solution.png"):
    values, origins = solver.get_solution_numpy()
    cmap = plt.cm.RdBu_r.copy()
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(origins[:, 0], origins[:, 1],
                    c=values, cmap=cmap, s=4)
    plt.colorbar(sc, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='white')
    print(f"Figure saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    from domain import SquareDomain, CircleDomain

    ti.init(arch=ti.gpu)

    domain = SquareDomain(lo=ti.Vector([0.0, 0.0]),
                          hi=ti.Vector([1.0, 1.0]))

    # domain = CircleDomain()

    solver = WoStSolver(domain=domain, dx=1/256, n_walks=2000,
                        epsilon=1e-5, max_steps=10000)

    print(f"Running WoSt ({solver.n_samples} samples, {solver.n_walks} walks) …")
    solver.solve()
    visualise(solver, title="WoSt — Square domain")