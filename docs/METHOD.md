# Numerical method

FluxCell solves the two-dimensional shallow-water equations, with an optional passive tracer, by
a finite-volume MUSCL-Hancock scheme with hydrostatic reconstruction and an HLLC Riemann solver.
The method is implemented in the f64 CPU references (`src/reference.mjs` for the first-order
scheme, `src/muscl-reference.mjs` for the second-order scheme) and in the WGSL kernels in
`src/shaders/`. Constants, closed-form local updates, timestep bounds, and hydrograph ghost states
are defined once in `src/model.mjs`. Options are listed in [API.md](API.md), and test cases and
measured accuracy are reported in [VERIFICATION.md](VERIFICATION.md).

SI units are used. In this document $`g`$ is gravitational acceleration, $`h`$ the depth, $`z`$ the
bed elevation, $`\eta = h + z`$ the free-surface elevation, $`(u, v)`$ the depth-averaged velocity,
and $`\mathbf{q} = (q_x, q_y) = (hu, hv)`$ the unit discharge.

## Governing equations

```math
\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{U})}{\partial x} + \frac{\partial \mathbf{G}(\mathbf{U})}{\partial y}
  = \mathbf{S}_b + \mathbf{S}_f + \mathbf{S}_r + \mathbf{S}_w + \mathbf{S}_c + \mathbf{S}_\nu
```

```math
\mathbf{U} = \begin{pmatrix} h \\ q_x \\ q_y \end{pmatrix}, \quad
\mathbf{F} = \begin{pmatrix} q_x \\ q_x u + \tfrac{1}{2} g h^2 \\ q_x v \end{pmatrix}, \quad
\mathbf{G} = \begin{pmatrix} q_y \\ q_y u \\ q_y v + \tfrac{1}{2} g h^2 \end{pmatrix}
```

```math
\mathbf{S}_b = \begin{pmatrix} 0 \\ -g h \, \partial_x z \\ -g h \, \partial_y z \end{pmatrix}, \quad
\mathbf{S}_f = \begin{pmatrix} 0 \\ -\dfrac{g n^2 |\mathbf{q}| q_x}{h^{7/3}} \\ -\dfrac{g n^2 |\mathbf{q}| q_y}{h^{7/3}} \end{pmatrix}, \quad
\mathbf{S}_r = \begin{pmatrix} R + s \\ 0 \\ 0 \end{pmatrix}
```

```math
\mathbf{S}_w = \begin{pmatrix} 0 \\ \tau_x / \rho_w \\ \tau_y / \rho_w \end{pmatrix}, \quad
\mathbf{S}_c = \begin{pmatrix} 0 \\ f q_y \\ -f q_x \end{pmatrix}, \quad
\mathbf{S}_\nu = \begin{pmatrix} 0 \\ \nabla \cdot (h \nu \nabla u) \\ \nabla \cdot (h \nu \nabla v) \end{pmatrix}
```

$`\mathbf{S}_f`$ is Manning friction with a per-cell coefficient $`n \ge 0`$ (s m$`^{-1/3}`$).
$`\mathbf{S}_r`$ adds depth at the uniform rain rate $`R`$ and the per-cell rate $`s`$ (m/s), with no
horizontal momentum; a discharge $`Q`$ distributed with weights $`w_i`$ enters as
$`s_i = Q w_i / A_i`$. The wind stress of a uniform wind $`\mathbf{W}`$ at 10 m
($`|\mathbf{W}| \le 100`$ m/s) is $`\boldsymbol{\tau} / \rho_w = (1.225 / 1000) \, C_d |\mathbf{W}| \mathbf{W}`$,
with the drag law of Wu (1982), $`C_d = (0.8 + 0.065 |\mathbf{W}|) \times 10^{-3}`$, or a constant
$`C_d \in (0, 0.01]`$. $`\mathbf{S}_c`$ is the Coriolis acceleration
$`\mathrm{d}\mathbf{q}/\mathrm{d}t = -f \, \mathbf{k} \times \mathbf{q}`$, with $`f > 0`$ in the northern
hemisphere and $`|f| \le 10^{-3}`$ s$`^{-1}`$. The turbulent stresses use the Laplacian form; the
transpose part $`\nabla \cdot \big(h \nu (\nabla \mathbf{u})^\mathsf{T}\big)`$ of the full stress is
omitted, which is exact for parallel shear flow and small where $`\nabla \cdot \mathbf{u}`$ is close
to zero.

A passive tracer of concentration $`c`$ is carried as the conserved amount $`hc`$:

```math
\frac{\partial (hc)}{\partial t} + \nabla \cdot (h c \mathbf{u}) = \nabla \cdot (D_t h \nabla c) + s c_s - w_s c,
\qquad D_t = D + \frac{\nu}{Sc_t}, \qquad w_s = w_0 c^m
```

$`D`$ is the molecular diffusivity, $`Sc_t`$ the turbulent Schmidt number (default 0.7), and
$`c_s`$ the concentration of source water; rain carries no tracer. The settling velocity $`w_s`$ is
constant for $`m = 0`$. For $`0 < m \le 3`$ it is the power law of the flocculation range of
cohesive sediment (Mehta, 1986), with site-specific coefficients and $`c`$ in kg/m$`^3`$.

## Finite-volume discretization

Cells of size $`\Delta x \times \Delta y`$ hold the averages $`\mathbf{U}_{ij}`$, which are updated
conservatively:

```math
\mathbf{U}_{ij}^{n+1} = \mathbf{U}_{ij}^{n}
  - \frac{\Delta t}{\Delta x} \left( \mathbf{H}_{i+1/2,j} - \mathbf{H}_{i-1/2,j} \right)
  - \frac{\Delta t}{\Delta y} \left( \mathbf{K}_{i,j+1/2} - \mathbf{K}_{i,j-1/2} \right)
  + \Delta t \, \mathbf{S}_{ij}
```

Each face flux is computed once and applied with opposite signs to both adjacent cells, with one
global timestep, so interior mass fluxes telescope and the volume changes only through boundary
faces and sources.

The first-order scheme serves as a control. It uses the cell averages as face states, the
hydrostatic reconstruction $`h_K^* = \max\big(0, h_K + z_K - \max(z_L, z_R)\big)`$ with the
cell-side pressure correction of Audusse et al. (2004), and the Rusanov flux

```math
\mathbf{H} = \tfrac{1}{2} \left( \mathbf{F}_n(\mathbf{U}_L^*) + \mathbf{F}_n(\mathbf{U}_R^*) \right) - \tfrac{1}{2} a \left( \mathbf{U}_R^* - \mathbf{U}_L^* \right),
\qquad a = \max_K \left( |u_{n,K}| + k \sqrt{g h_K^*} \right)
```

with $`k = 2`$ when either side is dry and $`k = 1`$ otherwise. Sources, boundaries, and forcing
are the same as in the second-order scheme, which is described in the remainder of this document.

## MUSCL reconstruction

Each cell reconstructs $`\mathbf{V} = (\eta, u, v, z)`$ along each axis. Stencil values are taken
relative to the cell's own surface and bed,
$`\mathbf{V}_k = \big((h_k - h_c) + (z_k - z_c), \; u_k, \; v_k, \; z_k - z_c\big)`$ (see the
floating-point formulation). Slopes are increments over one cell, limited with the
monotonized-central limiter (van Leer, 1979),

```math
\mathrm{mc}(a, b, c) = \mathrm{minmod}\!\left( 2(b - a), \; \tfrac{1}{2}(c - a), \; 2(c - b) \right),
```

where minmod returns the argument of smallest magnitude if all three share a sign and zero
otherwise. All slope components of both axes are then scaled by the depth-positivity factor

```math
\theta = \min\!\left( 1, \; \frac{2 h_c}{\max\left( |\Delta\eta_x - \Delta z_x|, \; |\Delta\eta_y - \Delta z_y| \right)} \right)
```

($`\theta = 1`$ when the denominator is zero), which keeps the unpredicted trace depths
$`h_c \pm \tfrac{1}{2} \theta (\Delta\eta - \Delta z)`$ nonnegative. Neighbours outside the domain map
to the nearest edge cell (or wrap on a periodic axis), with the normal velocity reversed across a
wall boundary. A solid neighbour is the mirror image of the centre cell with reversed normal
velocity.

A cell is reconstructed at first order (all slopes zero) in three cases:

- **Thin layer.** Any depth of the five-point stencil is at most $`H_\mathrm{thin} = 10^{-4}`$ m.
- **Bed step.** The bed of a neighbour differs from the cell's bed by more than the cell's depth.
  The predictor would then accelerate thin water by $`g \, \partial_x \eta`$ across a step much
  larger than the depth, and hydrostatic reconstruction is inaccurate in this regime (Delestre et
  al., 2012).
- **Rejected prediction.** A predicted trace has a negative depth or a nonfinite component.

No water is removed by these reductions. They are counted in `reconstructionFallbacks`.

## Hancock predictor

The traces are advanced by half a step with the primitive quasilinear equations

```math
\begin{aligned}
\eta_t &= -u (\eta_x - z_x) - v (\eta_y - z_y) - h (u_x + v_y) \\
u_t &= -u u_x - v u_y - g \eta_x \\
v_t &= -u v_x - v v_y - g \eta_y \\
z_t &= 0
\end{aligned}
```

in which derivatives are replaced by limited slopes divided by $`\Delta x`$ or $`\Delta y`$ and
coefficients are taken at the cell centre. The traces are

```math
\mathbf{V}_{W,E} = \mathbf{V}_c \mp \tfrac{1}{2} \Delta \mathbf{V}_x + \tfrac{1}{2} \Delta t \, \mathbf{V}_t, \qquad
\mathbf{V}_{S,N} = \mathbf{V}_c \mp \tfrac{1}{2} \Delta \mathbf{V}_y + \tfrac{1}{2} \Delta t \, \mathbf{V}_t .
```

The predictor is unsplit, so both directions and the bed slope (through $`\eta`$) enter every
trace. It is a local Taylor step; conservation is enforced by the update. A trace depth is
$`h_c + (\Delta\eta - \Delta z)`$. If any of the four traces of a cell has a negative depth or a
nonfinite component, all four are reset to the constant cell state. All traces are predicted
before any face flux is evaluated.

## Hydrostatic reconstruction and well-balanced update

At each face the east (north) trace of the left cell and the west (south) trace of the right cell
are cut to the face bed $`z^* = \max(z_L, z_R)`$, $`h_K^* = \max(0, \eta_K - z^*)`$ (Audusse et al.,
2004). The cut is evaluated from differences only, with $`h_K`$ the predicted trace depths:

```math
\begin{aligned}
\delta z &= (z_{c,L} - z_{c,R}) + (\Delta z_L - \Delta z_R) \\
\sigma &= \big( (h_{c,L} - h_{c,R}) + (z_{c,L} - z_{c,R}) \big) + (\Delta\eta_L - \Delta\eta_R)
\end{aligned}
```

```math
(h_L^*, h_R^*) =
\begin{cases}
\big(h_L, \; \min(h_R, \max(0, h_L - \sigma))\big) & \delta z \ge 0 \\
\big(\min(h_L, \max(0, h_R + \sigma)), \; h_R\big) & \delta z < 0
\end{cases}
```

This equals the absolute form in exact arithmetic. Trace velocities are retained, and one
homogeneous flux $`\mathbf{H}`$ is computed per face. The $`x`$-momentum of a cell changes by
$`-\Delta t / \Delta x`$ times

```math
(H_E - p_E^*) - (H_W - p_W^*) + g \bar{h}_x (\eta_E - \eta_W), \qquad \bar{h}_x = \tfrac{1}{2}(h_W + h_E)
```

where $`H_E`$ and $`H_W`$ are the normal-momentum fluxes, $`p_E^*`$ and $`p_W^*`$ the pressures
$`\tfrac{1}{2} g (h^*)^2`$ of the cell's own side at its east and west faces, and $`h_W`$, $`h_E`$,
$`\eta_W`$, and $`\eta_E`$ belong to the cell's predicted, uncut traces. The expression combines the
cell-side correction $`p_K - p_K^*`$ with the within-cell bed source
$`-g \bar{h}_x (z_E - z_W) / \Delta x`$ through $`p_E - p_W = g \bar{h}_x (h_E - h_W)`$; the cell's own
pressure $`\tfrac{1}{2} g h^2`$ cancels and is never formed. The $`y`$-momentum is analogous, and mass
and tangential momentum use the shared flux unchanged.

For a lake at rest the surface slopes and predictor increments vanish, both cut depths at a face
are equal ($`\sigma = 0`$), the HLLC flux has zero mass component and a normal momentum component
equal to $`p^*`$, and the bed term is zero. In f32 this balance is exact whenever the neighbour
differences $`h_k - h_c`$ and $`z_k - z_c`$ are exact, which the Sterbenz lemma guarantees for
neighbouring values within a factor of two. At a resting shoreline the first-order fallback and the
cut preserve it.

## Riemann solver

The HLLC solver (Toro et al., 1994; Toro, 2009) acts on the cut states in the face-normal frame,
$`\mathbf{U} = (h, h u_n, h u_t)`$ and $`\mathbf{F} = (h u_n, \; h u_n^2 + \tfrac{1}{2} g h^2, \; h u_n u_t)`$.
In this section $`h_K`$ denotes the cut depth, $`c_K = \sqrt{g h_K}`$, and $`p_K = \tfrac{1}{2} g h_K^2`$:

```math
S_L = \min(u_{n,L} - c_L, \; u_{n,R} - c_R), \qquad
S_R = \max(u_{n,L} + c_L, \; u_{n,R} + c_R), \qquad
d_K = h_K (S_K - u_{n,K})
```

```math
S^* = \frac{d_R u_{n,R} - d_L u_{n,L} + p_L - p_R}{d_R - d_L}, \qquad
h_K^* = \frac{d_K}{S_K - S^*}
```

```math
\mathbf{F}_K^* = \left( S^* h_K^*, \;\; F_{K,n} + S_K \left( h_K^* S^* - h_K u_{n,K} \right), \;\; S^* h_K^* u_{t,K} \right)
```

The left flux is returned if $`S_L \ge 0`$ and the right flux if $`S_R \le 0`$. Otherwise
$`\mathbf{F}_K^*`$ is taken from the side $`K`$ that contains the face (left if $`S^* \ge 0`$). Its
mass and tangential components use the identities $`F_h^* = S^* h_K^*`$ and
$`F_t^* = S^* h_K^* u_{t,K}`$, which equal $`\mathbf{F}_K + S_K (\mathbf{U}_K^* - \mathbf{U}_K)`$
algebraically and give exactly zero mass flux when $`S^* = 0`$. Two dry states give zero flux. With
one dry side the rarefaction bounds $`S_L = u_{n,R} - 2 c_R`$, $`S_R = u_{n,R} + c_R`$ (dry left) or
$`S_L = u_{n,L} - c_L`$, $`S_R = u_{n,L} + 2 c_L`$ (dry right) are used. The HLL flux

```math
\mathbf{F}_\mathrm{HLL} = \frac{S_R \mathbf{F}_L - S_L \mathbf{F}_R + S_L S_R (\mathbf{U}_R - \mathbf{U}_L)}{S_R - S_L}
```

replaces the star flux when one side is dry, when the states separate
($`u_{n,R} - u_{n,L} \ge 2 (c_L + c_R)`$), when
$`|d_R - d_L| \le 10^{-6} \max(|d_L|, |d_R|, 10^{-30})`$, when $`S^*`$ is nonfinite or within
$`10^{-6} \max(|S_L|, |S_R|)`$ of $`S_L`$ or $`S_R`$, and when a star depth is negative or
nonfinite. Its wave-speed bounds are the acoustic estimates above; Roe-averaged (Einfeldt)
estimates are not used. Fallbacks are counted in `hlleFallbacks`.

## Low-Froude correction

Upwind schemes damp nearly incompressible motion with a dissipation that scales with the fastest
wave speed (Guillard and Viozat, 1999). With `lowFroude` enabled, the correction of Thornber et
al. (2008), with the Froude number in place of the Mach number, is applied to the normal trace
velocities at interior and periodic faces after hydrostatic reconstruction:

```math
m = \tfrac{1}{2} (u_{n,L} + u_{n,R}), \qquad
\delta = \tfrac{1}{2} (u_{n,L} - u_{n,R}), \qquad
\zeta = \min\!\left( 1, \; \max_K \frac{|\mathbf{u}_K|}{\sqrt{g h_K^*}} \right)
```

```math
u_{n,L} \leftarrow m + \zeta \delta, \qquad u_{n,R} \leftarrow m - \zeta \delta
```

$`|\mathbf{u}_K|`$ is the full trace speed. The velocities are unchanged when either cut depth is
at most $`H_\mathrm{thin}`$. Depths, tangential velocities, and the mean normal velocity are not
modified, $`\zeta = 1`$ on supercritical faces, and a lake at rest is unaffected.

## Turbulence closure

```math
\nu = \nu_0 + (C_s \Delta)^2 |S| + a \, u_* h, \qquad
|S| = \sqrt{2 u_x^2 + 2 v_y^2 + (u_y + v_x)^2}, \qquad
\Delta = \min(\Delta x, \Delta y), \qquad
u_* = \frac{\sqrt{g} \, n |\mathbf{u}|}{h^{1/6}}
```

The eddy viscosity combines a constant background $`\nu_0`$, the Smagorinsky (1963) term with
$`C_s \in [0, 0.3]`$, and a bed-generated term with coefficient $`a \in [0, 1]`$ ($`0.067 = \kappa/6`$ is
the classical value) and the Manning friction velocity $`u_*`$. The strain rate is formed by
central differences of cell velocities, with neighbours mapped and reflected as in the
reconstruction. The viscosity is evaluated from the state at the start of the step and is zero in
solid cells and in cells with $`h \le H_\mathrm{thin}`$. The viscous momentum flux across an
interior or periodic face is

```math
F_\nu = -h_f \nu_f \frac{u_R - u_L}{\Delta_n}, \qquad h_f = \min(h_L, h_R), \qquad \nu_f = \tfrac{1}{2} (\nu_L + \nu_R),
```

for both velocity components, with $`\Delta_n = \Delta x`$ or $`\Delta y`$. It is zero when either
cell is thinner than $`H_\mathrm{thin}`$ and is not applied at domain or solid boundaries (free slip
at walls, zero gradient at open, outflow, and hydrograph faces). Being part of the shared face
flux, it conserves momentum, and it reproduces parallel shear flow exactly. With a tracer,
$`\nu_f / Sc_t`$ is added to the diffusivity.

The closure is explicit. The Smagorinsky part needs no separate restriction. Central differences
give $`|S| \le 2\sqrt{2} \, u_\mathrm{max} / \Delta`$ and the CFL bound gives
$`\Delta t \le 0.4 \Delta / u_\mathrm{max}`$, so the diffusion number
$`2 \nu \Delta t (1/\Delta x^2 + 1/\Delta y^2) \le 4 \nu \Delta t / \Delta^2`$ is at most
$`4.53 \, C_s^2`$ (0.41 at $`C_s = 0.3`$) on any aspect ratio, and with the advective fraction of at
most 0.4 the update remains a convex combination. This sets the upper limit of $`C_s`$. The
cell-local parts are added to both directional wave speeds of each cell as the viscous speed

```math
a_\nu = \frac{4 \, \mathrm{CFL} \, \nu_\mathrm{eff} \left( 1/\Delta x^2 + 1/\Delta y^2 \right)}{1/\Delta x + 1/\Delta y},
\qquad \nu_\mathrm{eff} = \nu_0 + a \, u_* h,
```

multiplied by $`\max(1, 1/Sc_t)`$ when a tracer is carried. The timestep then satisfies
$`\nu_\mathrm{eff} \Delta t (1/\Delta x^2 + 1/\Delta y^2) \le 1/4`$.

## Source terms and splitting

Source terms are applied by first-order operator splitting, cell by cell, in the order below, so
the formal second-order accuracy applies to the homogeneous equations with bed topography.

1. **Update and depth sources.** The flux update is applied, and $`(R + s_i) \Delta t`$ is added to
   the depth of every non-solid cell. Momentum is set to zero where the new depth is exactly zero.
2. **Wind.** $`\mathbf{q} \leftarrow \mathbf{q} + \Delta t \, \boldsymbol{\tau} / \rho_w`$ where
   $`h > H_\mathrm{thin}`$.
3. **Coriolis.** Where $`h > H_\mathrm{thin}`$, $`\mathbf{q}`$ is rotated by the angle $`-f \Delta t`$,
   the exact solution of $`\mathrm{d}\mathbf{q}/\mathrm{d}t = -f \, \mathbf{k} \times \mathbf{q}`$ at
   fixed depth, which conserves $`|\mathbf{q}|`$.
4. **Manning friction.** $`\mathbf{q} \leftarrow \mathbf{q} \, r / (r + \Delta t \, g n^2 |\mathbf{q}|)`$
   with $`r = h^{7/3}`$ at the new depth, the exact local solution over one step; it is
   unconditionally stable and never reverses the flow.
5. **Thin-layer regularization.** Where $`h < H_\mathrm{thin}`$,
   $`\mathbf{q} \leftarrow \mathbf{q} \, \sqrt{2} \, h^2 / \sqrt{h^4 + H_\mathrm{thin}^4}`$ (Kurganov and
   Petrova, 2007). The factor is 1 at $`h = H_\mathrm{thin}`$ and tends to 0 with $`h`$, which bounds
   thin-film velocities and prevents overflow of $`\mathbf{q}/h`$.
6. **Tracer.** The amount $`hc`$ is updated, and settling is applied at the new depth.

## Wet and dry treatment

A cell is dry when $`h = 0`$, and its velocity is then zero; otherwise $`\mathbf{u} = \mathbf{q}/h`$.
No threshold removes stored water. The thin-layer depth $`H_\mathrm{thin} = 10^{-4}`$ m selects
first-order reconstruction, limits wind and Coriolis to deeper cells, triggers the momentum
regularization, and disables the low-Froude correction and the eddy viscosity.

Nonnegative depth is promoted by the $`\theta`$ limiter, the trace rejection, the cut, the HLL
fallback at dry interfaces, and the timestep bound $`|\mathbf{u}| + 2c`$. No positivity theorem is
claimed for the complete second-order scheme, so every updated cell is checked for a nonnegative,
finite depth and finite momentum and tracer amount. A failing step is rejected without changing the
state, clock, or boundary accounts. The second-order schemes then repeat the whole step with half
the trial timestep, up to eight times, and fail explicitly if no trial succeeds (counter
`retries`); the first-order scheme does not retry. No cell clips its depth, caps its velocity, or
uses its own timestep.

## Boundary conditions

Each boundary is a ghost trace that shares the interior trace's bed and passes through the same
hydrostatic reconstruction and Riemann solver as an interior face.

- **Wall.** The interior trace with reversed normal velocity; the mass flux is set to exactly zero.
- **Periodic.** Opposite edges are paired and evaluated as interior faces.
- **Open.** A copy of the interior trace (zero gradient). This transmissive condition is not
  non-reflecting and admits water where terrain slopes into the domain.
- **Outflow.** A copy when the interior normal velocity points outward or is zero, and a wall when
  it points inward, so water leaves but never enters.
- **Solid cells** are dry, impermeable, and never updated. A solid-fluid face is a wall, and a
  solid-solid face carries no flux.

**Hydrograph segments** prescribe a stage, a discharge, or an incident wave on a range of cells of
a non-periodic edge. Series are interpolated linearly and held constant outside their range.
Levels are evaluated at the step midpoint $`t_n + \Delta t / 2`$, so a constant level stays exact; a
discharge is averaged over the step (from the end values within one linear piece, and from a
cumulative-integral table across breakpoints), so the admitted volume equals the integral of
$`Q(t)`$. Below, $`w`$ is the inward normal velocity of the interior trace, $`c = \sqrt{g h}`$, and
$`z_f`$ the face bed.

- **Stage** $`\eta_b`$. Against supercritical outflow ($`w < -c`$) the interior trace is copied. For
  inflow ($`w > 0`$) the water enters from rest at the stage, with ghost depth
  $`\max\big(0, \eta_b - z_f - w^2 / (2g)\big)`$, the interior normal velocity, and no tangential
  velocity. Otherwise the ghost has depth $`\max(0, \eta_b - z_f)`$ and the interior velocity. A stage
  equal to the level of a resting lake reproduces the lake exactly.
- **Discharge** $`Q`$ over the open length $`L`$ of the segment, $`q = Q/L`$. For $`q > 0`$ the ghost
  has depth $`h_g = \max\big(h, (q^2/g)^{1/3}\big)`$, inward normal velocity $`q / h_g`$, and no
  tangential velocity, and the face mass flux is replaced by $`q`$, so a step admits exactly
  $`q L \Delta t`$. For $`q = 0`$ the face is a wall. The discharge is distributed uniformly per unit
  width.
- **Wave** (incident level $`\eta_i`$ on still water at level $`s_0`$). This characteristic boundary
  combines the incoming Riemann invariant of a simple wave entering still water of depth
  $`d_0 = s_0 - z_f`$ with the outgoing invariant of the interior:

  ```math
  R^+ = 4 \sqrt{g (d_0 + \eta_i - s_0)} - 2 \sqrt{g d_0}, \qquad R^- = w - 2c
  ```

  ```math
  c_b = \tfrac{1}{4} (R^+ - R^-), \qquad w_b = \tfrac{1}{2} (R^+ + R^-), \qquad
  h_g = h + \frac{\delta (2c + \delta)}{g}, \qquad \delta = c_b - c
  ```

  The ghost carries the inward velocity $`w_b`$, and no tangential velocity when $`w_b > 0`$. Its
  depth equals $`c_b^2 / g`$, written as a difference from $`h`$ so that a lake at the still level
  reproduces the interior trace exactly. Against supercritical outflow the interior trace is
  copied, and for $`c_b \le 0`$ the ghost is dry. With $`\eta_i = s_0`$ the boundary is
  non-reflecting for small-amplitude long waves.

The volume through every non-periodic boundary face is accumulated, so the budget
$`V(t) + \text{outflow} = V(0) + \text{sourceVolume}`$ holds with inflow counted as negative outflow.

## Timestep

```math
\Delta t = \min\!\left( \Delta t_\mathrm{max}, \; t_\mathrm{target} - t, \; \Delta t_\mathrm{source}, \; \Delta t_\mathrm{hydro}, \; \Delta t_\mathrm{diff}, \; \frac{\mathrm{CFL}}{a_x / \Delta x + a_y / \Delta y} \right)
```

```math
a_x = \max\!\left( \max_\mathrm{cells} \left( |u| + 2\sqrt{g h} + a_\nu \right), \; \max_\mathrm{faces} \left( |u_g| + 2 \sqrt{g h_g} \right) \right)
```

$`a_y`$ is formed likewise, with $`\mathrm{CFL} \le 0.4`$ (default 0.4); $`\Delta t_\mathrm{max}`$ is the
`maxDt` option. The maxima are global and directional; $`2\sqrt{g h}`$ bounds the front speed of a
dry-bed rarefaction, $`a_\nu`$ is the viscous speed, and the second term covers the hydrograph ghost
states at the start of the step, so a high stage next to a dry bed limits the step. The further
limits are as follows.

- **Sources.** On dry ground the depth added in one step must not itself violate the CFL bound,
  $`\Delta t \cdot 2 \sqrt{g s_\mathrm{max} \Delta t} \, (1/\Delta x + 1/\Delta y) \le \mathrm{CFL}`$
  with $`s_\mathrm{max} = R + \max_i s_i`$, which gives
  $`\Delta t_\mathrm{source} = \left( \mathrm{CFL} \big/ \big( 2 \sqrt{g s_\mathrm{max}} \, (1/\Delta x + 1/\Delta y) \big) \right)^{2/3}`$.
- **Hydrographs.** $`\Delta t_\mathrm{hydro} = \mathrm{CFL} / (S_x / \Delta x + S_y / \Delta y)`$, with
  $`S_x`$ and $`S_y`$ the largest ghost-speed bounds over the whole series of the $`x`$- and
  $`y`$-normal segments: $`3 (g q_\mathrm{max})^{1/3}`$ for a discharge,
  $`2\sqrt{g (\eta_\mathrm{max} - z_\mathrm{min})}`$ for a stage, and
  $`3\sqrt{g (\eta_\mathrm{max} - z_\mathrm{min})}`$ for a wave, where $`z_\mathrm{min}`$ is the lowest
  open bed of the segment. A step cannot span the start of an inflow onto a dry bed.
- **Diffusion.** $`\Delta t_\mathrm{diff} = 1 / \big( 4 D (1/\Delta x^2 + 1/\Delta y^2) \big)`$ for the
  molecular tracer diffusivity.

## Tracer transport

- **Advective flux.** The flux through a face is $`F_h c^*`$, where $`F_h`$ is the mass flux used in
  the depth update (HLLC or HLL, a prescribed discharge, or exactly zero on reflective faces) and
  $`c^*`$ the upwind concentration trace (left if $`F_h > 0`$). The tracer thus moves with exactly the
  water that moves, and a uniform concentration stays uniform, also through wetting and drying. A
  ghost side copies the interior concentration; inflow through a hydrograph face carries the
  segment's concentration.
- **Reconstruction.** Slopes $`s_x`$, $`s_y`$ are MC-limited from cell concentrations and predicted
  with $`c_t + u c_x + v c_y = 0`$, for example
  $`c_E = c + \tfrac{1}{2} s_x - \tfrac{1}{2} \Delta t \, (u s_x / \Delta x + v s_y / \Delta y)`$. Each
  trace is clipped to the range of the cell and its four neighbours (a solid neighbour takes the
  cell's value), and the tracer is first order wherever the flow is.
- **Diffusion.** The flux $`-D_f \min(h_L, h_R) (c_R - c_L) / \Delta_n`$, with
  $`D_f = D + \nu_f / Sc_t`$, acts across interior and periodic faces only; the shallower depth makes
  it vanish at a wet/dry face.
- **Update.** $`(hc)^{n+1} = hc - \Delta t \, \nabla \cdot \mathbf{F}_c + s c_s \Delta t`$. A cell whose
  new depth is exactly zero keeps no tracer.
- **Settling.** The exact local solution of $`\mathrm{d}c/\mathrm{d}t = -w_s(c) \, c / h`$ at the new
  depth is applied:

  ```math
  c \leftarrow c \, e^{-w_0 \Delta t / h} \quad (m = 0), \qquad
  c \leftarrow c \left( 1 + \frac{m w_0 \Delta t \, c^m}{h} \right)^{-1/m} \quad (m > 0).
  ```

  It is positive and monotone, and the settled amount leaves the water column. The CPU references
  accumulate the deposited amount; the GPU kernel does not.

**Bounds and positivity.** Fluxes are evaluated with
$`c = \mathrm{clamp}(hc / h, \; c_\mathrm{min}, \; c_\mathrm{max})`$, and $`c = 0`$ where $`h = 0`$. The
range spans the initial concentrations of wet cells, the source and inflow concentrations, and 0
under rain or settling, and it widens when the forcing changes. Advection and bounded diffusion
obey a discrete maximum principle within this range in exact arithmetic, so the clip acts only in
thin cells where $`hc / h`$ loses precision. The stored amount is never clipped, so conservation is
unaffected.

## Floating-point formulation

The GPU kernels compute in f32. The following measures keep equilibria exact and the results
consistent with the f64 references.

- **Bed datum.** A datum is subtracted from the bed in f64 before rounding. The default is the
  lowest non-solid bed; stage and still-water levels are shifted by the same datum.
- **Difference-form traces.** With an absolute $`\eta = h + z`$ in f32 the depth would be resolved
  only to one unit in the last place of $`z`$, about $`1.2 \times 10^{-4}`$ m at elevations between
  1,024 and 2,048 m. Stencils, traces, and the hydrostatic cut are therefore formed from
  differences relative to each cell's own surface and bed, which keeps full depth precision at any
  elevation.
- **Exact zero fluxes.** WGSL division is not correctly rounded, so $`h_K^*`$ can differ from
  $`h_K`$ by one unit in the last place for two identical states; the star-region identities still
  give zero mass flux at rest. Wall, solid, and inflow-blocking outflow faces have their mass flux
  set to exactly zero.
- **Compensated sums.** The fourth lane of the f32 state record holds a Kahan (1965) compensation
  term for the depth, so small increments such as rain on deep water are not rounded away. The
  tracer amount and the per-face boundary volumes carry compensation terms as well.
- **Coriolis rotation.** WGSL guarantees sin and cos only to $`2^{-11}`$ absolute. The angle is
  halved below 0.1 rad, evaluated by Taylor series through seventh order, and doubled back.
- **Hydrograph clock.** Time is a two-float f32 sum $`(t_\mathrm{hi}, t_\mathrm{lo})`$ advanced by
  TwoSum (Knuth, 1997), and the host repeats the same f32 operations, so hydrographs are evaluated
  at the same time however steps are batched.

## GPU implementation

The `rusanov` kernel runs the first-order scheme in four dispatches per step (wave speeds,
timestep, face fluxes, and update), and the multi-pass `muscl-hancock` kernel runs the
second-order scheme in five (with reconstruction added), storing four trace records per cell in
global memory. The `muscl-fused` kernel runs the same discretization in one dispatch per step.
Each 16 × 16 workgroup loads a 20 × 20 tile (two-cell halo) into workgroup memory, reconstructs
its cells and the 64 halo cells whose facing traces are needed, evaluates the faces, and updates
the cells. A seam face is evaluated by both workgroups from identical inputs with identical code,
so the fluxes are bitwise equal and interior mass still telescopes.

In the fused kernel the timestep stays on the GPU. The update reduces the wave speeds of the new
state into one atomic maximum per workgroup, and a single-thread kernel forms $`\Delta t`$ from
these maxima, the host cap (`maxDt` with the source, hydrograph, and diffusion limits, rounded down
to f32), and the remaining time, and advances the hydrograph clock. Up to 256 steps are recorded
per submission and read back once. State, tracer, and boundary accumulators are checkpointed at
the start of each batch. If a step reports an invalid update, the host restores the checkpoint,
replays the valid prefix as one batch (steps are deterministic), and applies the halving retry to
the failing step only. Results are bitwise independent of the batch size and equal to those of the
step-by-step multi-pass kernel for configurations both support. Scalar transport, the turbulence
closure, and settling run in the fused kernel and the CPU references, the low-Froude correction in
the MUSCL kernels and the MUSCL reference, and wind and Coriolis everywhere.

## References

- Audusse, E., Bouchut, F., Bristeau, M.-O., Klein, R., and Perthame, B. (2004). A fast and stable well-balanced scheme with hydrostatic reconstruction for shallow water flows. *SIAM Journal on Scientific Computing, 25*(6), 2050-2065. https://doi.org/10.1137/S1064827503431090
- Delestre, O., Cordier, S., Darboux, F., and James, F. (2012). A limitation of the hydrostatic reconstruction technique for shallow water equations. *Comptes Rendus Mathématique, 350*(13-14), 677-681. https://doi.org/10.1016/j.crma.2012.08.004
- Guillard, H., and Viozat, C. (1999). On the behaviour of upwind schemes in the low Mach number limit. *Computers and Fluids, 28*(1), 63-86. https://doi.org/10.1016/S0045-7930(98)00017-6
- Kahan, W. (1965). Pracniques: further remarks on reducing truncation errors. *Communications of the ACM, 8*(1), 40. https://doi.org/10.1145/363707.363723
- Knuth, D. E. (1997). *The art of computer programming, Volume 2: Seminumerical algorithms* (3rd ed.). Addison-Wesley.
- Kurganov, A., and Petrova, G. (2007). A second-order well-balanced positivity preserving central-upwind scheme for the Saint-Venant system. *Communications in Mathematical Sciences, 5*(1), 133-160. https://doi.org/10.4310/CMS.2007.v5.n1.a6
- Mehta, A. J. (1986). Characterization of cohesive sediment properties and transport processes in estuaries. In A. J. Mehta (Ed.), *Estuarine cohesive sediment dynamics* (Lecture Notes on Coastal and Estuarine Studies, Vol. 14, pp. 290-325). Springer New York. https://doi.org/10.1007/978-1-4612-4936-8_15
- Smagorinsky, J. (1963). General circulation experiments with the primitive equations. *Monthly Weather Review, 91*(3), 99-164. https://doi.org/10.1175/1520-0493(1963)091<0099:GCEWTP>2.3.CO;2
- Thornber, B., Mosedale, A., Drikakis, D., Youngs, D., and Williams, R. J. R. (2008). An improved reconstruction method for compressible flows with low Mach number features. *Journal of Computational Physics, 227*(10), 4873-4894. https://doi.org/10.1016/j.jcp.2008.01.036
- Toro, E. F. (2009). *Riemann solvers and numerical methods for fluid dynamics: A practical introduction* (3rd ed.). Springer. https://doi.org/10.1007/b79761
- Toro, E. F., Spruce, M., and Speares, W. (1994). Restoration of the contact surface in the HLL-Riemann solver. *Shock Waves, 4*(1), 25-34. https://doi.org/10.1007/BF01414629
- van Leer, B. (1979). Towards the ultimate conservative difference scheme. V. A second-order sequel to Godunov's method. *Journal of Computational Physics, 32*(1), 101-136. https://doi.org/10.1016/0021-9991(79)90145-1
- Wu, J. (1982). Wind-stress coefficients over sea surface from breeze to hurricane. *Journal of Geophysical Research: Oceans, 87*(C12), 9704-9706. https://doi.org/10.1029/JC087iC12p09704
