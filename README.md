# Poisson Equation Solver Based on Walk on Stars (WoSt) Method

This project implemented a solver for Poisson equation based on WoSt, which is a kind of Monte Carlo Method. The advantage of this method is it's grid-free so there is no spatial discretization error and no need to preprocess extremely complicated geometry.

The implementation is based on:
[Walk on Stars: A Grid-Free Monte Carlo Method for PDEs with Neumann Boundary Conditions](https://dl.acm.org/doi/10.1145/3592398)
[wost-simple repo](https://github.com/GeometryCollective/wost-simple)

We focus on zero-Neumann Poisson equation in this project, which can be written as:
$$
\begin{align*}
    \Delta u(x) &= f(x) \quad\text{on}\quad \Omega, \\
    u &= g(x) \quad\text{on}\quad \Omega_{Dirichlet}, \\
    \frac{\partial u(x)}{\partial n_x} &= h(x) \quad\text{on}\quad \Omega_{Neumann},
\end{align*}
$$
where $h(x)$ is always $0$.

Generally speaking, the core of WoSt is Boundary Integration Equation (BIE) for Poisson equation:
$$
\alpha(x) u(x)
= \int_{\partial \mathrm{St}(x,r)} P^B(x,z)\,u(z)
\;-\;
\int_{\partial \mathrm{St}_N(x,r)} G^B(x,z)\,h(z)\,dz
\;+\;
\int_{\mathrm{St}(x,r)} G^B(x,y)\,f(y)\,dy\,,
$$
where $P^B$ is Poisson Kernel of a ball and $G^B$ is Green Function of a ball. The radius of ball $R$ is $\min{(R_{Dirichlet}, R_{Sihouette})}$. We can employ Monte Carlo Method to estimate this BIE, which leads to:
$$
\hat{u}(x_k) =
\frac{P^B(x_k, x_{k+1})\,\hat{u}(x_{k+1})}
{\alpha(x_k)\,p^{\partial \mathrm{St}(x_k,r)}(x_{k+1})}
\;-\;
\frac{G^B(x_k, z_{k+1})\,h(z_{k+1})}
{\alpha(x_k)\,p^{\partial \mathrm{St}_N(x_k,r)}(z_{k+1})}
\;+\;
\frac{G^B(x_k, y_{k+1})\,f(y_{k+1})}
{\alpha(x_k)\,p^{\mathrm{St}(x_k,r)}(y_{k+1})}
$$

## Dirichlet Boundary
Firstly, we implement the pure Dirichlet version. In this case, WoSt degenerates to Walk on Sphere (WOS). We are solving in a square domain, the fixed value is $1$ on top side and $-1$ on other thress sides:
<p>
  <img src="img/fd_dirichlet.png" width="45%" />
  <img src="img/WoSt_dirichlet.png" width="45%" />
</p>

Because this is a square domain, Finite Difference doesn't have discretization error and can be used as groud truth.

## Neumann Boundary
Then, we change left side to zero-Neumann boundary:
<p>
  <img src="img/fd_neumann_square.png" width="45%" />
  <img src="img/WoSt_neumann_square.png" width="45%" />
</p>

The result looks good. We also test on a circle domain - the bounary at top half has fixed value $1$, and bottom half has Neumann boundary:
<p>
  <img src="img/fd_neumann_circle.png" width="45%" />
  <img src="img/WoSt_neumann_circle.png" width="45%" />
</p>

The analytical solution is $u(x) = 1$. However, due to discretization error around boundary, the solution of finite difference is not consistant. But our WoSt sovler is able to achieve proper answer. 

## Source Term
Sometimes rhs of Poisson equation is called "source term". In WoSt solver, it corresponds to the last term in BIE. In this project we use uniform distribution sampling to estimate this term, this may cause some noise in solution.

First problem is in square domain, boundary condition is same with last one, and:
$$
f(x) = 
\begin{cases}
    40,\quad |x - 0.25| + |y - 0.5| < 0.2 \\
    0,\quad \text{otherwise}
\end{cases}
$$

<p>
  <img src="img/fd_source_square.png" width="45%" />
  <img src="img/WoSt_source_square.png" width="45%" />
</p>

Second problem is in circle domain. On the boudnary of top semi-circle, $u = 1$; on the boundary of right bottom part, $u = -1$; left bottom boundary is Nuemann boundary. Source term $f(x)$ is $-80$ in the area where the radius is less than $0.1$.

<p>
  <img src="img/fd_source_circle.png" width="45%" />
  <img src="img/WoSt_source_circle.png" width="45%" />
</p>

These two problems prove that WoSt can gradually converge to correct solution. But adding source term makes result really noisy, some methods to reduce noise are needed.