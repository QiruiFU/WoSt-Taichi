import numpy as np
import taichi as ti
import taichi.math as tm


class BaseDomain():
    """
    Abstract base class for all problem domains.

    Subclasses must implement:
      - the Taichi @ti.func methods dist_to_boundary / boundary_value
        (these are plain Python methods wrapping @ti.func logic, or
         directly decorated — see concrete classes below)
      - grid_info(N) → (interior_mask, boundary_mask, bc_values)
    """

    def dist_to_boundary(self, x: tm.vec2) -> float:
        """Distance from point x to the nearest point on ∂Ω."""
        raise NotImplementedError

    def boundary_value(self, x: tm.vec2) -> float:
        """Dirichlet value g(x) at (or near) the boundary."""
        raise NotImplementedError

    def grid_info(self, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return grid metadata for an (N+2)×(N+2) FD grid whose nodes
        span the bounding box of this domain.

        Returns
        -------
        interior_mask : bool ndarray (N+2, N+2)
            True  → node is inside Ω and should be updated by G-S.
            False → exterior or boundary node (frozen).
        boundary_mask : bool ndarray (N+2, N+2)
            True  → node carries a prescribed Dirichlet value.
        bc_values : float ndarray (N+2, N+2)
            Dirichlet value at each boundary node (0 elsewhere).
        """
        raise NotImplementedError

    @property
    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lo, hi) corners of the bounding box as numpy arrays."""
        raise NotImplementedError


@ti.data_oriented
class SquareDomain(BaseDomain):
    """
    Axis-aligned square domain [lo, hi]².

    Boundary condition:
      u = 1  on the top edge
      u = 0  on all other edges

    The bounding box *is* the domain, so interior_mask is 1 everywhere
    in the interior and the FD grid reduces to the original ghost-cell
    layout from fd.py.
    """

    def __init__(self,
                 lo: ti.template() = ti.Vector([0.0, 0.0]),
                 hi: ti.template() = ti.Vector([1.0, 1.0])):
        self.lo = lo
        self.hi = hi
        self._lo_np = lo.to_numpy() if hasattr(lo, "to_numpy") else np.array(lo)
        self._hi_np = hi.to_numpy() if hasattr(hi, "to_numpy") else np.array(hi)

    @property
    def bbox(self):
        return self._lo_np, self._hi_np

    @ti.func
    def dist_to_boundary(self, x: tm.vec2) -> float:
        d_left   = x[0] - self.lo[0]
        d_right  = self.hi[0] - x[0]
        d_bottom = x[1] - self.lo[1]
        d_top    = self.hi[1] - x[1]
        return ti.min(d_left, d_right, d_bottom, d_top)

    @ti.func
    def boundary_value(self, x: tm.vec2) -> float:
        d_left   = x[0] - self.lo[0]
        d_right  = self.hi[0] - x[0]
        d_bottom = x[1] - self.lo[1]
        d_top    = self.hi[1] - x[1]
        d_min = ti.min(d_left, d_right, d_bottom, d_top)
        val = 0.0
        if d_top == d_min:
            val = 1.0
        return val

    def grid_info(self, N: int):
        lo, hi = self._lo_np, self._hi_np
        M = N + 2

        xs = np.linspace(lo[0], hi[0], M)
        ys = np.linspace(lo[1], hi[1], M)

        interior_mask = np.zeros((M, M), dtype=bool)
        boundary_mask = np.zeros((M, M), dtype=bool)
        bc_values     = np.zeros((M, M), dtype=np.float32)

        for i in range(M):
            for j in range(M):
                on_edge = (i == 0) or (i == M-1) or (j == 0) or (j == M-1)
                if on_edge:
                    boundary_mask[i, j] = True
                    x = np.array([xs[i], ys[j]])
                    bc_values[i, j] = self._bc_numpy(x)
                else:
                    interior_mask[i, j] = True

        return interior_mask, boundary_mask, bc_values

    def _bc_numpy(self, x: np.ndarray) -> float:
        """Python version of boundary_value for grid_info construction."""
        lo, hi = self._lo_np, self._hi_np
        d_top = hi[1] - x[1]
        d_others = min(x[0] - lo[0], hi[0] - x[0], x[1] - lo[1])
        return 1.0 if d_top <= d_others else 0.0


@ti.data_oriented
class CircleDomain(BaseDomain):
    """
    Circular domain  { x : |x - c|² ≤ r² }.

    Boundary condition:
      u = 1  on the top half  (y ≥ cy)
      u = 0  on the bottom half (y < cy)

    SDF is exact: dist = r - |x - c|  (positive inside)
    """

    def __init__(self, cx: float = 0.5, cy: float = 0.5, r: float = 0.4):
        self.cx = cx
        self.cy = cy
        self.r  = r
        self._lo_np = np.array([cx - r, cy - r], dtype=np.float64)
        self._hi_np = np.array([cx + r, cy + r], dtype=np.float64)

    @property
    def bbox(self):
        return self._lo_np, self._hi_np

    def _dist_numpy(self, x: np.ndarray) -> float:
        """Signed distance: negative inside, positive outside."""
        d = np.hypot(x[0] - self.cx, x[1] - self.cy)
        return d - self.r

    def _bc_numpy(self, x: np.ndarray) -> float:
        px = x[0] - self.cx
        py = x[1] - self.cy
        norm = np.hypot(px, py)
        if norm < 1e-12:
            return 0.0
        ny = py / norm
        return 1.0 if ny >= 0.0 else 0.0

    @ti.func
    def dist_to_boundary(self, x):
        px = x[0] - ti.static(self.cx)
        py = x[1] - ti.static(self.cy)
        return ti.static(self.r) - tm.sqrt(px * px + py * py)

    @ti.func
    def boundary_value(self, x):
        px = x[0] - ti.static(self.cx)
        py = x[1] - ti.static(self.cy)
        norm = tm.sqrt(px * px + py * py)
        ny = py / ti.select(norm > 1e-12, norm, 1.0)
        return ti.select(ny >= 0.0, 1.0, 0.0)

    def grid_info(self, N: int):
        lo, hi = self._lo_np, self._hi_np
        M = N + 2
        h = (hi[0] - lo[0]) / (M - 1)

        xs = np.linspace(lo[0], hi[0], M)
        ys = np.linspace(lo[1], hi[1], M)

        interior_mask = np.zeros((M, M), dtype=bool)
        boundary_mask = np.zeros((M, M), dtype=bool)
        bc_values     = np.zeros((M, M), dtype=np.float32)

        for i in range(M):
            for j in range(M):
                pt = np.array([xs[i], ys[j]])
                d  = self._dist_numpy(pt)

                if d > 0:
                    pass
                elif d > -1.5 * h:
                    boundary_mask[i, j] = True
                    bc_values[i, j]     = self._bc_numpy(pt)
                else:
                    interior_mask[i, j] = True

        return interior_mask, boundary_mask, bc_values