import numpy as np
import taichi as ti
import taichi.math as tm
import math

class BaseDomain():
    # Finite Difference
    def bc_numpy(self, x: np.ndarray):
        """
            return bc_type, bc_value, bc_normal
        """
        raise NotImplementedError

    def source_numpy(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def grid_info(self, N: int):
        """
            return interior_mask, boundary_mask, bc_type, bc_values, bc_normals, source_values
        """
        raise NotImplementedError

    # Walk on Star
    def dist_numpy(self, x: np.ndarray) -> float:
        """
            dist to boundary, used to delete samples outside domain
        """
        raise NotImplementedError

    def dist_to_dirichlet(self, x: tm.vec2) -> float:
        raise NotImplementedError

    def dist_to_silhouette(self, x: tm.vec2) -> float:
        raise NotImplementedError

    def boundary_value(self, x: tm.vec2) -> float:
        """
            Dirichlet boundary value
        """
        raise NotImplementedError

    def source(self, x: tm.vec2) -> float:
        raise NotImplementedError

    def intersect_ray(self, x: tm.vec2, v: tm.vec2, R: float):
        """
            return pos_intersction, on_Neumann, normal_at_hit
        """
        raise NotImplementedError

    @property
    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


@ti.data_oriented
class SquareDomain(BaseDomain):
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

    # Finite difference
    def bc_numpy(self, x: np.ndarray):
        lo, hi = self._lo_np, self._hi_np
        d_left   = x[0] - lo[0]
        d_right  = hi[0] - x[0]
        d_bottom = x[1] - lo[1]
        d_top    = hi[1] - x[1]
        d_min = min(d_left, d_right, d_bottom, d_top)

        # bc_type, bc_value, bc_normal
        if d_top == d_min:
            return 0, 1.0, np.array([0.0,  1.0], dtype=np.float32)
        elif d_bottom == d_min:
            return 0, 0.0, np.array([0.0, -1.0], dtype=np.float32)
        elif d_left == d_min:
            return 1, 0.0, np.array([-1.0, 0.0], dtype=np.float32)
        else:
            return 0, 0.0, np.array([1.0,  0.0], dtype=np.float32)

    def source_numpy(self, x: np.ndarray) -> float:
        return 0.0

    def grid_info(self, N: int):
        lo, hi = self._lo_np, self._hi_np
        M = N + 2
        xs = np.linspace(lo[0], hi[0], M)
        ys = np.linspace(lo[1], hi[1], M)

        interior_mask = np.zeros((M, M), dtype=np.int32)
        boundary_mask = np.zeros((M, M), dtype=np.int32)
        bc_type       = np.zeros((M, M), dtype=np.int32)
        bc_values     = np.zeros((M, M), dtype=np.float32)
        bc_normals    = np.zeros((M, M, 2), dtype=np.float32)
        source_values = np.zeros((M, M), dtype=np.float32)

        for i in range(M):
            for j in range(M):
                on_edge = (i == 0) or (i == M-1) or (j == 0) or (j == M-1)
                x = np.array([xs[i], ys[j]])
                if on_edge:
                    interior_mask[i, j] = 0
                    boundary_mask[i, j] = 1
                    btype, bval, bnorm  = self.bc_numpy(x)
                    bc_type[i, j]       = btype
                    bc_values[i, j]     = bval
                    bc_normals[i, j]    = bnorm
                else:
                    boundary_mask[i, j] = 0
                    interior_mask[i, j] = 1
                source_values[i, j] = self.source_numpy(x)

        return interior_mask, boundary_mask, bc_type, bc_values, bc_normals, source_values

    # WoSt
    def dist_numpy(self, x: np.ndarray) -> float:
        d_left   = x[0] - self.lo[0]
        d_right  = self.hi[0] - x[0]
        d_bottom = x[1] - self.lo[1]
        d_top    = self.hi[1] - x[1]
        return min(d_left, d_right, d_bottom, d_top)

    @ti.func
    def dist_to_dirichlet(self, x: tm.vec2) -> float:
        d_top    = self.hi[1] - x[1]
        d_bottom = x[1] - self.lo[1]
        d_right  = self.hi[0] - x[0]
        return ti.min(d_top, ti.min(d_bottom, d_right))

    @ti.func
    def dist_to_silhouette(self, x: tm.vec2) -> float:
        return 1e8

    @ti.func
    def boundary_value(self, x: tm.vec2) -> float:
        d_right  = self.hi[0] - x[0]
        d_bottom = x[1] - self.lo[1]
        d_top    = self.hi[1] - x[1]
        d_min = ti.min(d_right, d_bottom, d_top)
        val = 0.0
        if d_top == d_min:
            val = 1.0
        return val

    @ti.func
    def source(self, x: tm.vec2) -> float:
        return 0.0

    @ti.func
    def intersect_ray(self, x: tm.vec2, v: tm.vec2, R: float):
        t_min = R
        on_Neumann = False
        n_hit = tm.vec2(0.0, 0.0)

        if v[0] < 0.0:
            t = (self.lo[0] - x[0]) / v[0]
            if t > 0.0 and t < t_min:
                t_min = t
                on_Neumann = True
                n_hit = tm.vec2(-1.0, 0.0)

        return x + t_min * v, on_Neumann, n_hit

@ti.data_oriented
class CircleDomain(BaseDomain):
    def __init__(self, cx: float = 0.5, cy: float = 0.5, r: float = 0.4):
        self.cx = cx
        self.cy = cy
        self.r  = r
        self._lo_np = np.array([cx - r, cy - r], dtype=np.float64)
        self._hi_np = np.array([cx + r, cy + r], dtype=np.float64)

    @property
    def bbox(self):
        return self._lo_np, self._hi_np

    # ------------------------------------------------------------------ #
    # Finite difference
    # ------------------------------------------------------------------ #
    def bc_numpy(self, x: np.ndarray):
        px = x[0] - self.cx
        py = x[1] - self.cy
        norm = np.hypot(px, py)
        if norm < 1e-12:
            nx, ny = 0.0, 1.0
        else:
            nx, ny = px / norm, py / norm
        normal = np.array([nx, ny], dtype=np.float32)
        if py >= 0:
            return 0, 1.0, normal
        elif px >= 0:
            return 0, -1.0, normal
        else:
            return 1, 0.0, normal

    def source_numpy(self, x: np.ndarray) -> float:
        return 0.0

    def grid_info(self, N: int):
        lo, hi = self._lo_np, self._hi_np
        M = N + 2
        h = (hi[0] - lo[0]) / (M - 1)
        xs = np.linspace(lo[0], hi[0], M)
        ys = np.linspace(lo[1], hi[1], M)

        interior_mask = np.zeros((M, M), dtype=np.int32)
        boundary_mask = np.zeros((M, M), dtype=np.int32)
        bc_type       = np.zeros((M, M), dtype=np.int32)
        bc_values     = np.zeros((M, M), dtype=np.float32)
        bc_normals    = np.zeros((M, M, 2), dtype=np.float32)
        source_values = np.zeros((M, M), dtype=np.float32)

        for i in range(M):
            for j in range(M):
                pt = np.array([xs[i], ys[j]])
                d  = self.r - np.hypot(pt[0] - self.cx, pt[1] - self.cy)
                if d < 0:
                    pass
                elif d < math.sqrt(2) * h:
                    interior_mask[i, j] = 0
                    boundary_mask[i, j] = 1
                    btype, bval, bnorm  = self.bc_numpy(pt)
                    bc_type[i, j]       = btype
                    bc_values[i, j]     = bval
                    bc_normals[i, j]    = bnorm
                elif d <= self.r:
                    boundary_mask[i, j] = 0
                    interior_mask[i, j] = 1
                source_values[i, j] = self.source_numpy(pt)

        return interior_mask, boundary_mask, bc_type, bc_values, bc_normals, source_values

    # ------------------------------------------------------------------ #
    # WoSt
    # ------------------------------------------------------------------ #
    def dist_numpy(self, x: np.ndarray) -> float:
        return self.r - np.hypot(x[0] - self.cx, x[1] - self.cy)

    @ti.func
    def boundary_value(self, x: tm.vec2) -> float:
        px = x[0] - ti.static(self.cx)
        py = x[1] - ti.static(self.cy)
        norm = tm.sqrt(px * px + py * py)
        ny   = py / ti.select(norm > 1e-12, norm, 1.0)
        return ti.select(ny >= 0.0, 1.0, -1.0)

    @ti.func
    def source(self, x: tm.vec2) -> float:
        return 0.0

    @ti.func
    def intersect_ray(self, x: tm.vec2, v: tm.vec2, R: float):
        t_min  = R
        on_Neumann = False
        n_hit  = tm.vec2(0.0, 0.0)

        cx = ti.static(self.cx)
        cy = ti.static(self.cy)
        cr = ti.static(self.r)

        oc   = x - tm.vec2(cx, cy)
        b    = 2.0 * tm.dot(oc, v)
        c    = tm.dot(oc, oc) - cr * cr
        disc = b * b - 4.0 * c
        t = (-b + tm.sqrt(disc)) * 0.5 # larger one in 2 solutions

        if t < t_min and v[0] < 0.0 and v[1] < 0.0:
            t_min  = t
            on_Neumann = True
            y_hit = x + t * v
            n_hit  = tm.normalize(y_hit - tm.vec2(cx, cy))

        return x + t_min * v, on_Neumann, n_hit

    @ti.func
    def dist_to_dirichlet(self, x: tm.vec2) -> float:
        cx = ti.static(self.cx)
        cy = ti.static(self.cy)
        cr = ti.static(self.r)
        px = x[0] - cx
        py = x[1] - cy

        dist = cr - tm.sqrt(px * px + py * py)
        if py >= 0.0 or px >= 0.0:
            dist = cr - tm.sqrt(px * px + py * py)
        else:
            node1 = tm.vec2(cx - cr, cy)
            node2 = tm.vec2(cx, cy - cr)
            d1 = tm.sqrt((x[0] - node1[0]) * (x[0] - node1[0]) + (x[1] - node1[1]) * (x[1] - node1[1]))
            d2 = tm.sqrt((x[0] - node2[0]) * (x[0] - node2[0]) + (x[1] - node2[1]) * (x[1] - node2[1]))
            dist = ti.min(d1, d2)
        return dist

    @ti.func
    def dist_to_silhouette(self, x: tm.vec2) -> float:
        return 1e8