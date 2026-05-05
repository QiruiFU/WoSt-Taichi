# Final Report

**Qirui Fu**

## A Grid-Free Monte Carlo Solver for the Poisson Equation: From Walk on Spheres to Walk on Stars

## Abstract

This project studies a Monte Carlo solver for the Poisson equation, with emphasis on the Walk on Spheres (WoS) method for Dirichlet problems and the Walk on Stars (WoSt) method for mixed Dirichlet-Neumann problems. Unlike finite difference or finite element methods, these approaches do not require a mesh covering the full domain. Instead, they estimate the solution by sampling random trajectories whose statistics are derived from the underlying partial differential equation. This grid-free viewpoint is attractive for domains with curved or irregular boundaries, where mesh generation and spatial discretization may become expensive or inaccurate.

The implementation in this project includes three main components. First, a finite difference solver is used as a traditional baseline. Second, a Monte Carlo solver based on WoS/WoSt is implemented in Taichi. Third, several test domains are evaluated, including square and circle domains, with Dirichlet boundary conditions, mixed Dirichlet-Neumann boundary conditions, and nonzero source terms. The final results show that the Monte Carlo method reproduces the qualitative behavior of the finite difference solution, while retaining the main advantage of being grid-free. In particular, for curved boundaries the WoSt solver avoids the boundary discretization artifacts that appear in the finite difference baseline.

## 1. Introduction

The Poisson equation is one of the most fundamental elliptic partial differential equations. A standard formulation is

$$
\Delta u(x) = f(x), \quad x \in \Omega,
$$

where $\Omega \subset \mathbb{R}^2$ is a domain, $u$ is the unknown scalar field, and $f$ is the source term. To determine a unique solution, the equation must be supplemented by boundary conditions. In this project, both Dirichlet and Neumann boundary conditions are considered:

$$
u(x) = g(x), \quad x \in \Gamma_D,
$$

$$
\frac{\partial u}{\partial n}(x) = h(x), \quad x \in \Gamma_N,
$$

where the boundary is decomposed as $\partial \Omega = \Gamma_D \cup \Gamma_N$.

Traditional numerical methods such as finite difference and finite element methods convert the PDE into a large linear system on a mesh or grid. These methods are powerful and widely used, but they also have limitations:

1. They introduce spatial discretization error.
2. They require preprocessing of the domain geometry.
3. Complicated boundaries can be expensive to represent faithfully.

Monte Carlo PDE solvers provide a different perspective. Instead of solving the equation everywhere at once on a grid, they estimate the value of the solution at sample points by averaging random paths. The computational cost is therefore concentrated on sampling rather than global linear algebra.

The goal of this project is to implement a practical Poisson solver along this line. The solver begins from the classical Walk on Spheres method, then extends to Walk on Stars in order to handle Neumann boundaries. A GPU-oriented implementation is built with Taichi, and the results are compared against a finite difference baseline.

## 2. Mathematical Background

### 2.1 The Poisson Equation

The Laplacian of a scalar function $u(x, y)$ in two dimensions is

$$
\Delta u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}.
$$

Therefore, the Poisson equation

$$
\Delta u = f
$$

describes a balance between the local curvature of the unknown field and the forcing term $f$. When $f = 0$, the equation reduces to Laplace's equation, whose solutions are harmonic functions.

### 2.2 Why an Integral or Probabilistic Formulation Helps

The key idea behind Monte Carlo PDE methods is that elliptic equations admit equivalent integral and probabilistic representations. Instead of approximating derivatives directly on a grid, one can represent the value $u(x)$ as an average of boundary values and source contributions collected along random paths. Once such a representation is available, Monte Carlo sampling becomes possible.

This is especially useful because:

1. Expectation values can be estimated by random sampling.
2. Random walks do not require a global mesh.
3. Curved domains can be handled through geometric queries such as distance-to-boundary and ray-boundary intersection.

## 3. From the PDE to a Local Integral Representation

### 3.1 Local Integral Representation on a Ball

Instead of simulating Brownian motion with very small time steps, one can use a local analytic solution on a ball. Suppose a ball $B(x, R)$ lies entirely inside the domain. Then the solution at the center satisfies a representation of the form

$$
u(x)
= \int_{\partial B(x,R)} P^B(x,z)\, u(z)\, dz + \int_{B(x,R)} G^B(x,y)\, f(y)\, dy,
$$

where $P^B$ is the Poisson kernel of the ball and $G^B$ is the Green's function of the ball. This formula replaces continuous Brownian evolution by a jump from one ball boundary to another.

The meaning of the two terms is intuitive:

1. The boundary integral transports information from the current point to a new point on the sphere.
2. The volume integral accumulates the effect of the source term inside that sphere.

In two dimensions, the Green's function for a disk contains a logarithm. This is why the implementation uses a term proportional to

$$
\log(R/t),
$$

when estimating source contributions inside a step.

### 3.2 Why This Leads Naturally to Monte Carlo

The integral formula above is already an expectation. If one samples:

1. A point on the sphere according to the Poisson kernel, and
2. A point in the ball for the source term,

then each sample produces an unbiased estimator of the local contribution. Repeating this process recursively yields a random walk that jumps between analytically valid local neighborhoods until it reaches a Dirichlet boundary.

In the special case $f = 0$, the solution can be interpreted as the expected boundary value obtained when the random walk eventually reaches the boundary. This is the basic probabilistic intuition behind Walk on Spheres, but for this project the local integral representation is the more important starting point.

This is the conceptual bridge from the PDE to Walk on Spheres.

## 4. Walk on Spheres for Dirichlet Problems

### 4.1 Core Idea

Walk on Spheres replaces the full Brownian path by a sequence of large jumps. At the current position $x_k$, one inscribes the largest ball centered at $x_k$ that remains inside the domain. For pure Dirichlet problems, this radius is simply the distance to the boundary:

$$
R_k = \operatorname{dist}(x_k, \partial \Omega).
$$

The next point is sampled on the sphere of radius $R_k$. The process repeats until the walk enters a small $\varepsilon$-neighborhood of the boundary, at which point the boundary value is evaluated directly.

The algorithm is efficient because each step moves as far as possible while staying inside the domain. Near the center of a large domain, this can replace many tiny Brownian increments by a single jump.

### 4.2 Dirichlet Case in This Project

For the first experiment, the domain is the unit square. The top side has prescribed value $1$, while the other three sides have value $-1$. Since the source term is zero, the estimator only needs boundary sampling.

In the current implementation, the square domain provides the following geometric queries:

1. Distance to the nearest Dirichlet boundary.
2. Boundary value at a termination point.
3. A trivial silhouette distance, since the square Dirichlet case does not require clipping by a Neumann silhouette.

This case is particularly useful as a sanity check because the finite difference baseline on an axis-aligned square is very reliable.

### 4.3 Relation to the Code

The implementation in `WoSt.py` uses the same recursive idea. At each step:

1. The walker state stores its current position, accumulated value, step count, and termination flag.
2. The radius $R$ is computed from the domain query.
3. A new direction is sampled uniformly when the current state is not constrained by a Neumann boundary.
4. The walk terminates once the distance to the Dirichlet boundary is below a small tolerance `epsilon`.

Although the file is named `WoSt.py`, the same implementation degenerates to classical WoS when the domain is purely Dirichlet.

## 5. From Walk on Spheres to Walk on Stars

### 5.1 Why Neumann Boundaries Are Harder

The classical WoS algorithm relies on the fact that the next sphere can be chosen to lie fully inside the domain and that the process terminates on a Dirichlet boundary where the value of $u$ is known. Neumann boundaries are different: they prescribe the normal derivative rather than the value itself. A random walk cannot simply stop there and read off the answer.

Moreover, near a Neumann boundary, the largest inscribed sphere may cross geometric regions where the local harmonic argument is no longer valid. This is where the Walk on Stars construction becomes important.

### 5.2 Boundary Integral Equation and Star-Shaped Local Region

The theoretical foundation of Walk on Stars is a local boundary integral equation. Walk on Stars replaces a full sphere by a star-shaped region that respects the local visibility of the boundary. In the notation used in the README, this boundary integral equation can be written as

$$
\alpha(x) u(x)
= \int_{\partial \mathrm{St}(x,r)} P^B(x,z)u(z)\,dz - \int_{\partial \mathrm{St}_N(x,r)} G^B(x,z)h(z)\,dz + \int_{\mathrm{St}(x,r)} G^B(x,y)f(y)\,dy.
$$

Here:

1. $\mathrm{St}(x,r)$ is the star-shaped region associated with point $x$.
2. $\partial \mathrm{St}_N(x,r)$ is the portion lying on the Neumann boundary.
3. $\alpha(x)$ is a normalization factor accounting for the clipped geometry.

This formula is the key extension from WoS to WoSt. The local domain is no longer a complete ball; it is a ball truncated by visible boundary constraints. The corresponding estimator therefore combines:

1. A boundary term over the star boundary.
2. A Neumann contribution over the Neumann portion of that boundary.
3. A volume contribution from the source term.

Because of this boundary integral equation, the algorithm can handle mixed boundary conditions while remaining mesh-free.

### 5.3 Intuition for the Neumann Contribution

For a Dirichlet boundary, the unknown value is known on the boundary, so the walk can terminate there. For a Neumann boundary, one instead knows the flux

$$
\frac{\partial u}{\partial n} = h.
$$

This flux appears as an additional boundary integral term. In the present project, the Neumann data is always zero, which simplifies the implementation because the explicit Neumann integral vanishes. Even so, the geometry of the walk must still respect the Neumann portion of the boundary. That is the role of the star construction and the ray-intersection logic.

### 5.4 How the Project Approximates WoSt

The implementation follows the central geometric idea of WoSt in a simplified form suitable for a numerical methods course project:

1. For each walker, compute the distance to the nearest Dirichlet boundary, denoted `r_D`.
2. Compute a silhouette-related radius `r_S`.
3. Use

$$
R = \min(r_D, r_S)
$$

as the admissible local radius.

4. Sample a direction.
5. Intersect the ray with the local boundary geometry.
6. If a Neumann boundary is hit first, continue the walk with a hemisphere-like directional restriction based on the boundary normal.

This logic appears in `sample_direction()` and `intersect_ray()` in `WoSt.py`.

## 6. Source Terms and Volume Sampling

### 6.1 Why the Source Term Introduces Noise

When $f \neq 0$, the solution is no longer determined only by the exit boundary value. The walk must also estimate the volume integral involving the Green's function. This adds variance because each random path now carries both:

1. A boundary contribution, and
2. A source contribution accumulated over steps.

In the current implementation, a random radius

$$
t = R \sqrt{\xi}, \qquad \xi \sim U[0,1],
$$

is used to sample a point along the chosen direction inside the disk. The corresponding Green's-function-based weight is

$$
G = \frac{\log(R/t)}{2\pi}.
$$

This contribution is then scaled by the disk area factor $\pi R^2$ and accumulated into the walker's estimate.

### 6.2 Connection to the Code

The relevant logic appears in `walk_step()` in `WoSt.py`. The code:

1. Draws a sample radius `t_sample`.
2. Evaluates the source at the sampled interior point.
3. Computes a Green's-function weight.
4. Accumulates the source contribution into `source_val`.

This design matches the integral representation discussed earlier. It also explains an important empirical observation from the README: once a source term is added, the Monte Carlo solution becomes noticeably noisier. This is expected, since the estimator now has higher variance.

## 7. GPU Parallelization with Taichi

### 7.1 Why the Method Is Naturally Parallel

Monte Carlo solvers are well suited to GPU execution because different random walks are statistically independent. In this project, two levels of parallelism are available:

1. Different sample points in the domain can be evaluated independently.
2. Different Monte Carlo trials for the same estimator can also be accumulated independently.

This is an example of embarrassingly parallel computation. Each thread updates its own walker state, and the final solution is obtained by averaging accumulated contributions.

### 7.2 Data Layout in the Implementation

The Taichi implementation defines a `WalkState` data class storing:

1. Current position.
2. Current boundary value contribution.
3. Step counter.
4. Termination status.
5. Source contribution.
6. Whether the walk is currently associated with a Neumann hit.
7. The local boundary normal.

The solver then allocates Taichi fields for:

1. Sample origins.
2. Running accumulators.
3. Number of finished walks.
4. The walker array itself.
5. Final averaged solution values.

Because these fields are parallel arrays managed by Taichi, the kernel `walk_step()` can update all active walkers simultaneously on the GPU.

### 7.3 Sampling the Domain

Before solving, the code uses Poisson disk sampling to generate interior query points. This is a good practical choice because it avoids strong clustering and distributes samples relatively uniformly throughout the domain. The sampling step:

1. Generates candidate points in the domain bounding box.
2. Filters them using a signed-distance-like inside test.
3. Stores the surviving points as the origins where the solution will be estimated.

This means the final Monte Carlo result is not represented on a structured grid. Instead, it is represented as values attached to scattered interior samples, which are later visualized as a point cloud.

### 7.4 Control Flow of the Solver

The high-level solver structure is:

1. Reset all accumulators.
2. For each Monte Carlo trial, reset all walkers to their original sample positions.
3. Repeatedly apply `walk_step()` until every walker terminates or a step cap is reached.
4. Accumulate the completed estimates.
5. Average the results after all trials are finished.

This organization is straightforward and maps well to Taichi kernels. It is not the only possible GPU design, but it is simple, readable, and effective for the scale of this project.

## 8. Finite Difference Baseline

To compare the Monte Carlo solver against a traditional approach, the project also implements a finite difference method in `fd.py`. The baseline uses:

1. A structured Cartesian grid.
2. Red-black Gauss-Seidel iteration.
3. Explicit handling of Dirichlet and Neumann boundary nodes.

For interior nodes, the standard five-point Laplacian stencil is used:

$$
u_{i,j} \leftarrow \frac{1}{4}\left(
u_{i-1,j}+u_{i+1,j}+u_{i,j-1}+u_{i,j+1}-h^2 f_{i,j}
\right).
$$

For Neumann boundaries, the implementation updates boundary-adjacent values using a one-sided relation derived from the normal derivative condition.

This baseline is useful for two reasons:

1. It provides a familiar reference method.
2. It highlights the geometric advantage of the grid-free Monte Carlo approach.

On simple rectangular domains, finite differences are very effective. On curved domains, however, the boundary must be approximated on a Cartesian grid, which introduces visible geometric discretization error.

## 9. Experiments and Results

### 9.1 Pure Dirichlet Problem on a Square

The first experiment solves Laplace's equation on the unit square with boundary values $1$ on the top side and $-1$ on the other three sides. This is the cleanest validation case.

The project README already notes that, because the domain is a square aligned with the Cartesian grid, the finite difference result can be treated as a strong reference. The WoS solution matches the same overall pattern, showing that the Monte Carlo estimator correctly transports boundary information through the domain.

<p>
  <img src="../img/fd_dirichlet.png" width="45%" />
  <img src="../img/WoSt_dirichlet.png" width="45%" />
</p>

### 9.2 Mixed Dirichlet-Neumann Problem on a Square

The second experiment changes the left boundary from Dirichlet to zero-Neumann, while keeping the other sides fixed. This is the first case where WoSt is required rather than pure WoS.

The README summarizes the outcome concisely: the result "looks good." That conclusion is supported by the visual agreement between the finite difference and Monte Carlo solutions. The main value of this test is that it confirms the ray-intersection and normal-handling logic for a simple domain where the Neumann side is planar and easy to verify.

<p>
  <img src="../img/fd_neumann_square.png" width="45%" />
  <img src="../img/WoSt_neumann_square.png" width="45%" />
</p>

### 9.3 Mixed Dirichlet-Neumann Problem on a Circle

The circular experiment is more interesting geometrically. The top semicircle uses a fixed Dirichlet value of $1$, while the bottom semicircle uses a zero-Neumann condition. According to the README, the analytical solution is the constant function

$$
u(x) = 1.
$$

This case demonstrates a central motivation for grid-free methods. The finite difference method must embed the curved boundary in a Cartesian grid and therefore suffers from boundary discretization artifacts near the circle. By contrast, the WoSt solver uses the circle geometry directly through distance and ray-intersection queries, so it can recover the expected constant solution much more faithfully.

<p>
  <img src="../img/fd_neumann_circle.png" width="45%" />
  <img src="../img/WoSt_neumann_circle.png" width="45%" />
</p>

### 9.4 Problems with Source Terms

The project also studies nonzero source terms in both square and circular domains.

For the square test:

$$
f(x) =
\begin{cases}
40, & |x-0.25| + |y-0.75| < 0.2, \\
0, & \text{otherwise}.
\end{cases}
$$

For the circular test:

$$
f(x) =
\begin{cases}
-80, & \|(x,y)-(0.5,0.5)\| < 0.1, \\
0, & \text{otherwise}.
\end{cases}
$$

The README remarks that these tests show the WoSt solution can gradually converge to the correct answer, but the source term makes the result visibly noisy. This is fully consistent with the estimator design: sampling the Green's-function volume integral adds significant variance.

<p>
  <img src="../img/fd_source_square.png" width="45%" />
  <img src="../img/WoSt_source_square.png" width="45%" />
</p>

<p>
  <img src="../img/fd_source_circle.png" width="45%" />
  <img src="../img/WoSt_source_circle.png" width="45%" />
</p>

## 10. Comparison with Traditional Methods

This section directly reflects the main motivation stated in the README: the advantage of WoSt is that it is grid-free, so there is no spatial discretization of the solution domain and no need to preprocess complicated geometry into a mesh.

### 10.1 Advantages of the Monte Carlo WoS/WoSt Approach

1. **Grid-free geometry handling.** The method only needs local geometric queries such as distance to the boundary and ray intersection. This is particularly attractive for smooth or complicated boundaries.
2. **No global sparse linear solve.** The algorithm estimates solution values pointwise by sampling, which can be simpler to parallelize.
3. **Natural compatibility with GPUs.** Independent random walks are well matched to data-parallel hardware.
4. **Potentially higher geometric fidelity.** In the circular Neumann example, the Monte Carlo approach avoids the jagged boundary approximation inherent in the Cartesian finite difference grid.

### 10.2 Advantages of Finite Difference

1. **Lower variance.** The finite difference solution is deterministic once the grid is fixed.
2. **Strong performance on simple domains.** For rectangles and uniform grids, the method is straightforward and accurate.
3. **Well-understood convergence behavior.** Classical iterative solvers and error analyses are readily available.

### 10.3 Main Weaknesses Observed in This Project

For the Monte Carlo solver:

1. The solution contains sampling noise.
2. Source terms increase variance significantly.
3. High accuracy requires many walks.

For the finite difference solver:

1. Curved boundaries are only approximated on the grid.
2. Boundary treatment for mixed conditions is more delicate.
3. A global discretization of the entire domain is required even if only local values are needed.

In summary, the comparison in this project does not suggest that Monte Carlo universally replaces traditional methods. Rather, it shows that WoS/WoSt is a compelling alternative when geometry handling is difficult, pointwise estimates are sufficient, or GPU-parallel sampling is attractive.

## 11. Limitations and Possible Improvements

Although the current implementation is successful as a course project, several limitations remain.

### 11.1 Variance Reduction

The most important practical improvement would be variance reduction. The noisy source-term results indicate that naive uniform sampling inside each local disk is not yet sufficient for high-quality solutions. Possible improvements include:

1. Better importance sampling for the source integral.
2. Control variates.
3. Antithetic sampling.
4. Stratified sampling.

### 11.2 More Accurate WoSt Boundary Treatment

The present solver captures the central idea of WoSt, but a complete production-quality implementation would require a more careful treatment of the exact star-shaped boundary integral equation, especially for general nonzero Neumann data and more complicated silhouettes.

### 11.3 Quantitative Error Analysis

This report mainly presents qualitative image comparisons. A stronger evaluation would measure:

1. Pointwise errors against analytical solutions when available.
2. Convergence with respect to the number of walks.
3. Runtime scaling on CPU versus GPU.
4. Sensitivity to the termination threshold `epsilon`.

## 12. Conclusion

This project implemented a Poisson solver based on Monte Carlo random walks, beginning with Walk on Spheres for Dirichlet problems and extending to a simplified Walk on Stars treatment for mixed Dirichlet-Neumann boundaries. The main mathematical idea is that the Poisson equation admits a local integral representation involving the Poisson kernel and Green's function, and that this representation leads naturally to random-walk-based estimators.

The implementation shows that the method works well on both square and circular domains, handles zero-Neumann boundaries, and can be accelerated naturally on the GPU through Taichi. The comparison with the finite difference baseline highlights the major tradeoff: Monte Carlo methods avoid global meshing and can better respect curved geometry, but they introduce sampling noise, especially when source terms are present.

Overall, the project demonstrates that WoS/WoSt is not only mathematically elegant, but also practically meaningful as a grid-free alternative to traditional Poisson solvers.

## References

1. Muller, Mervin E. "Some Continuous Monte Carlo Methods for the Dirichlet Problem." *The Annals of Mathematical Statistics* 27, no. 3 (1956): 569-589.
2. Sawhney, Rohan, and Keenan Crane. "Monte Carlo Geometry Processing: A Grid-Free Approach to PDE-Based Methods on Volumetric Domains." *ACM Transactions on Graphics* 39, no. 4 (2020).
3. Sawhney, Rohan, Dario Seyb, Wojciech Jarosz, and Keenan Crane. "Grid-Free Monte Carlo for PDEs with Spatially Varying Coefficients." *ACM Transactions on Graphics* 41, no. 4 (2022).
4. Sawhney, Rohan, Bailey Miller, Ioannis Gkioulekas, and Keenan Crane. "Walk on Stars: A Grid-Free Monte Carlo Method for PDEs with Neumann Boundary Conditions." *ACM Transactions on Graphics* 42, no. 4 (2023).
5. Sugimoto, Ryusuke. *Toward General-Purpose Monte Carlo PDE Solvers for Graphics Applications*. University of Waterloo.
