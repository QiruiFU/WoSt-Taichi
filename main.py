import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)

from domain import SquareDomain, CircleDomain
from fd    import FDSolver,  visualise as fd_vis
from WoSt   import WoSSolver, visualise as wos_vis

domain = CircleDomain(cx=0.5, cy=0.5, r=0.5)
# domain = SquareDomain(lo=(0.0, 0.0), hi=(1.0, 1.0))

# ---------------------------------------------------------------------------
# FD solve
# ---------------------------------------------------------------------------
fd_solver = FDSolver(domain=domain, N=256)
fd_solver.solve(max_iters=200_000, tol=1e-6, check_every=5000)
fd_vis(fd_solver, title="FD — Circle domain", save_path="./img/fd_circle.png")

# ---------------------------------------------------------------------------
# WoS solve
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Walk-on-Spheres solver — Circle domain")
print("=" * 60)
wos_solver = WoSSolver(domain=domain,
                       dx = 1 / 256,
                       n_walks=5000,
                       epsilon=1e-4,
                       max_steps=10000)
print(f"Running WoS  ({wos_solver.n_samples} samples, {wos_solver.n_walks} walks) …")
wos_solver.solve(check_every=400)
wos_vis(wos_solver, title="WoS — Circle domain", save_path="./img/wos_circle.png")