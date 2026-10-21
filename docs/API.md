# Solver API and GPU contract

Three GPU schemes share one state, forcing, boundary and diagnostic contract:

| Scheme | Shader | Dispatches per step | Host sync |
|---|---|---|---|
| `rusanov` | `src/shaders/rusanov.wgsl` | 4 (waveSpeeds, timestep, faceFluxes, update) | scalar readback per step |
| `muscl-hancock` | `src/shaders/muscl-hancock.wgsl` | 5 (adds reconstruct) | scalar readback per step |
| `muscl-fused` | `src/shaders/muscl-fused.wgsl` | 2 (timestep, fusedStep) | one readback per batch of up to 256 steps |

`rusanov` is the first-order control; `muscl-hancock` is the auditable multi-pass
second-order path; `muscl-fused` implements the same discretization as
`muscl-hancock` in a single tiled kernel and is the production path. The f64 CPU
references are `ReferenceSolver` (`src/reference.mjs`) and `MusclReferenceSolver`
(`src/muscl-reference.mjs`). All modules are native ES modules; there are no
dependencies.

Entry point: `src/index.mjs` (types in `src/index.d.ts`); subpaths `./benchmarks`, `./scenarios`
and `./terrain` hold the SWASHES solutions, test fixtures and 3DEP loaders.

## Run

```sh
npm test        # CPU suite (node --test)
npm run serve   # http://localhost:8080/
```

Then open `tools/validate/` (GPU correctness suites), `tools/benchmark/`
(throughput) or `examples/viewer/` (live demo). WebGPU requires localhost or HTTPS.

## Example

```js
import { compileKernels, createSolver } from './src/webgpu.mjs';
import { scenario } from './src/scenarios.mjs';

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const kernels = await compileKernels(device, 'muscl-fused');
const { config, initial } = scenario('obstacle', 256, 128);
const solver = await createSolver(device, config, initial, kernels);
try {
    const { steps, dts } = await solver.advance(60, { maxDt: 1, batch: 64 }); // 60 s simulated
    solver.setForcing({ source: newRates });          // e.g. next hydrograph interval
    const state = await solver.readState();            // explicit diagnostic readback
    const outflow = await solver.readBoundaryOutflow(); // net m³ out through the boundary (inflow < 0)
} finally {
    solver.destroy(); // caller owns the device
}
```

### Methods (all GPU solvers)

- `setParameters(changes)`: change `wind`, `windDrag`, `coriolis` or `settling` between steps (validated as at creation).
- `setHydrographs(hydrographs)`: replace the hydrograph series between steps; segments keep their edge, cells and kind, and the series may not have more points in total than at creation.
- `setSources(indices, rates, concentrations)`: sparse source update between steps. Only the listed cells' records are uploaded, and the inflow rate and the source timestep bound are updated from the changes, so frequent small updates (moving storms, point inflows) cost O(changed cells). The result equals `setForcing` with the full array.
- `setState(state, { concentration })`: replace the state between steps (hotstarts, restored scenarios, batch runs), validated as at creation; tracer runs also take the concentration per cell. The previous state is read back, and the volume change is kept in `stateEditVolume`, outside the source and boundary budgets. Returns a promise on the GPU solvers.
- `setBed(indices, elevations)`: edit the bed of the given cells (m, in the datum of the initial bed) for interactive terrain; depths are kept, so the water surface moves with the bed. Solid cells and nonfinite values are rejected; bounds derived at creation (hydrograph ghost speeds) are not recomputed.

- `step(maxDt)` – one accepted step; returns its dt. MUSCL schemes retry a step
  rejected for negative depth at half the trial dt, at most eight times.
- `advance(duration, { maxDt, maxSteps, batch })` – integrates to `time + duration`
  (or `maxSteps` steps) and returns `{ steps, dts, time }`. The fused runner records
  `batch` (default 64, max 256) steps per submission; the GPU chooses each dt and
  stops at the target time. If any step in a batch is invalid, the batch is
  rolled back to its checkpoint and replayed one step at a time with retries.
- `encode(encoder, { duration, maxDt, steps })` and `resolve()` (fused only) – non-blocking
  stepping for render loops: `encode` records up to `steps` steps toward `time + duration` into the
  caller's command encoder and returns the state buffer they will write (also `currentState`), so
  later passes in the same encoder can read it. After submitting, `await resolve()` accepts the
  steps, or restores the checkpoint and redoes the interval through the batched retry path
  (`recovered: true`). One encode may be pending; other operations wait for `resolve()`. Unlike
  `advance`, `encode` runs exactly one batch and does not top up an f32 remainder of the target.
- `setForcing({ manning, source, sourceConcentration })` – replace per-cell Manning n,
  source rates or the concentration of source water between steps. The solid mask is fixed
  at creation.
- `readState()`, `readBoundaryOutflow()`, `destroy()`. `time`, `sourceVolume` (m³
  added by rain and sources), `counters` (`hlleFallbacks`, `reconstructionFallbacks`,
  `retries`) and `allocatedBytes` are host fields.
- With `tracer: true` (fused runner and CPU references): `readTracer()` returns the conserved
  amount $`hc`$ per cell (m; concentration = amount / $`h`$), `readBoundaryOutflow({ tracer: true })` the
  net tracer amount through the boundary (m³ times concentration, inflow negative), and
  `tracerSourceAmount` the amount added by sources. `currentTracer` names the tracer buffer
  (`vec2<f32>` amount and compensation per cell) that encoded steps write.

### External state edits

Between operations (no `encode` pending), other GPU passes may write `currentState`
(`vec4<f32>` [h, hu, hv, compensation] per cell) directly, for example an infiltration pass, a 1D
channel coupling or a region fill in a host application. The edit must keep the state valid: finite,
$`h \ge 0`$, zero momentum where $`h = 0`$, solid cells dry, and the compensation lane of edited cells set to
0. The next step reads the edited state; its checkpoint is taken from it, so a rollback never undoes
the edit. Such edits are outside the solver's budgets (`sourceVolume`, boundary outflow,
`stateEditVolume`); the application accounts for them. In tracer runs the conserved amounts in
`currentTracer` must be edited consistently.

Await operations sequentially; overlapping operations fail explicitly. Invalid
updates never change the accepted state, boundary accounting or clock.

## Configuration and inputs

Configuration: integer `width`, `height`; positive `dx`, `dy` (m), `gravity`
(m/s²), `cfl` in $`(0, 0.4]`$; nonnegative uniform `rain` (m/s); `boundaryX` and
`boundaryY` each `wall`, `periodic`, `open` or `outflow`; `datum` (`'auto'` or an
elevation subtracted from the bed in f64); optional `hydrographs`; `tracer` (default false)
to transport a passive scalar, and `diffusivity` (m²/s, default 0, requires the tracer).
Transport runs on the `muscl-fused` kernels and on both CPU references; the multi-pass GPU
kernels reject it (they already use the eight storage bindings a stage may have).
`lowFroude` (default false; MUSCL kernels and the MUSCL reference) scales the normal velocity
jump at interior faces by $`\min(1, \mathrm{Fr})`$ (Thornber et al. 2008). The turbulence closure (fused kernel
and both references) is off unless a coefficient is set: `eddyViscosity` (constant, m²/s),
`smagorinsky` ($`C_s \in [0, 0.3]`$, filter width $`\min(\Delta x, \Delta y)`$), `bedTurbulence` ($`a`$ in $`\nu_b = a u_* h`$,
$`u_*`$ from Manning; 0.067 is $`\kappa/6`$), and `schmidt` (turbulent Schmidt number of the tracer, 0.7).
`wind` (`[x, y]`, m/s at 10 m, default `[0, 0]`, speed at most 100 m/s; all kernels and both
references) drives the surface with the stress of `windDrag`: `'wu1982'` (default,
$`C_d = (0.8 + 0.065 |\mathbf{W}|) \times 10^{-3}`$, Wu 1982) or a constant drag coefficient in (0, 0.01]. `settling` (`{ velocity,
exponent }`, requires the tracer; fused kernel and both references) makes the tracer settle out of
the water column at $`w_s = w_0 c^m`$, with $`w_0`$ the `velocity` (m/s) and $`m`$ the `exponent` (0 for a
constant settling velocity, up to 3; $`m > 0`$ is the power law of the flocculation range,
Mehta 1986, with site-specific coefficients and $`c`$ in kg/m³).
`coriolis` ($`f`$, 1/s, $`|f| \le 10^{-3}`$, default 0; all kernels and both references) rotates the momentum of wet
cells by the exact angle $`-f \Delta t`$ each step ($`f > 0`$ in the northern hemisphere).

`hydrographs` is a list of edge segments `{ edge, start, end, kind, series }`:
`edge` is `west`, `east`, `south` or `north` (not on a periodic axis); `[start, end)`
are cells along the edge (default: the whole edge; segments may not overlap);
`kind` is `stage` (water-surface elevation, m, in the bed's vertical datum),
`discharge` (inflow, m³/s over the segment, >= 0), or `wave` (incident water-surface
elevation, m, on still water at the segment's required `still` level, m); `series` is `[[t, value], ...]`
with strictly increasing $`t`$ (s), interpolated linearly and held constant outside its
range. A segment overrides the edge's boundary type on its cells. With a tracer, water
entering through a segment carries its `concentration` (default 0).

```js
hydrographs: [
    { edge: 'west', kind: 'discharge', series: [[0, 0], [600, 250], [3600, 40]] },  // m³/s
    { edge: 'east', start: 10, end: 90, kind: 'stage', series: [[0, 2.1], [43200, 3.4]] }, // m
]
```

`initial` holds `state` (`[h, hu, hv, 0]` per cell, row-major; m and m²/s; on the GPU the fourth lane
holds a depth-compensation term and is not physical state), `bed`
(m), and optional per-cell `manning` (s m$`^{-1/3}`$), `source` (m/s; a discharge $`Q`$
distributed over cells with weights $`w_i`$ is $`Q w_i / A_i`$) and `solid` (0/1). Dry
cells must have zero momentum; solid cells must be dry. Negative or nonfinite
inputs are rejected. Depth is never discarded below a wetness threshold. With a tracer,
`concentration` gives $`c`$ per cell (default 0) and `sourceConcentration` the concentration of
source water per cell (default 0); rain carries no tracer.

## Physics

- **Manning friction**, applied after the conservative update at the new depth:
  $`\mathbf{q} \leftarrow \mathbf{q} \, r / (r + \Delta t \, g n^2 |\mathbf{q}|)`$, $`r = h^{7/3}`$. This is exact for the local
  friction ODE over one step, unconditionally stable, and never reverses flow.
  The overall source treatment is first-order operator splitting.
- **Wind stress**, added to the momentum of wet cells ($`h > 10^{-4}`$ m) after the flux update and before
  friction: $`\mathbf{q} \leftarrow \mathbf{q} + \Delta t \, \boldsymbol{\tau}/\rho_w`$, $`\boldsymbol{\tau}/\rho_w = (1.225/1000) \, C_d |\mathbf{W}| \mathbf{W}`$. Friction then
  balances it: uniform wind against Manning friction tends to $`|\mathbf{q}| = \sqrt{(\tau/\rho_w) \, h^{7/3} / (g n^2)}`$.
- **Settling** of the tracer, after its transport, at the new depth: the exact local solution of
  $`\mathrm{d}c/\mathrm{d}t = -w_s(c) \, c / h`$, $`c \, e^{-w_0 \Delta t / h}`$ ($`m = 0`$) or
  $`c \left(1 + m w_0 \Delta t \, c^m / h\right)^{-1/m}`$ ($`m > 0`$). It is positive and monotone; the settled
  amount leaves the water column. The CPU references accumulate it (`tracerDeposited`); the GPU
  kernel does not, so its tracer budget closes only with that term from a reference run.
  Settling adds 0 to the concentration range used for flux evaluation.
- **Sources**: depth increases by $`(R + s_i) \, \Delta t`$ (`rain` $`R`$, `source` $`s_i`$) with no added momentum.
  Solid cells receive none.
- **Solid cells**: faces between a solid and a wet cell are reflective walls (mirror
  ghost with reversed normal momentum); faces between two solid cells carry no flux.
  MUSCL reconstruction uses the mirror state as the solid neighbour.
- **Reflective faces** (walls, solid cells, inflow-blocking outflow faces) carry exactly
  zero mass flux.
- **Open boundaries**: zero-gradient (transmissive) ghost, a named approximation and
  not a non-reflecting boundary. **Outflow boundaries**: zero-gradient where the flow
  leaves, reflective where it would enter (free outfall of a cut tile).
- **Hydrograph boundaries** (ghost states; stages at the step midpoint, discharges as their mean
  over the step; the whole series also bounds the timestep, so a step cannot span the start of an
  inflow onto a dry bed):
  - *Stage*: ghost depth $`\eta_b - z_f`$ for the stage $`\eta_b`$ and face bed $`z_f`$, ghost velocity =
    interior velocity. Water flowing in comes from rest at the stage: depth
    $`\eta_b - z_f - u^2/(2g)`$, no tangential velocity. The stage is not imposed against supercritical outflow (the interior state is copied).
  - *Discharge*: unit inflow $`q = Q/L`$ over the segment's open length $`L`$, ghost depth
    $`h_g = \max\big(h, (q^2/g)^{1/3}\big)`$, normal velocity $`q/h_g`$, no tangential velocity. The face
    mass flux is set to $`q`$ exactly, so the volume added equals the integral of $`Q(t)`$.
  - *Wave*: a characteristic boundary. The incoming Riemann invariant $`u + 2c`$ of the incident
    wave, $`4\sqrt{g (d_0 + \eta)} - 2\sqrt{g d_0}`$ with $`d_0 = s_0 - z_f`$ and $`\eta = \eta_i - s_0`$ for the
    `still` level $`s_0`$ and the incident `value` $`\eta_i`$, and the outgoing invariant $`u - 2c`$ of the interior give the face state. It generates the incident
    wave and lets outgoing waves leave; with $`\eta_i = s_0`$ it is a non-reflecting boundary.
    Water flowing in carries no tangential velocity; against supercritical outflow the interior
    state is copied. A lake at the still level stays exactly at rest.
  - Ghost wave speeds $`|u| + 2\sqrt{g h}`$ enter the CFL bound, so a high stage next to a dry
    bed limits the step.
- **Scalar transport** (passive tracer, conserved amount $`hc`$):
  $`(hc)_t + \nabla \cdot (h c \mathbf{u}) = \nabla \cdot (D h \nabla c) + s c_s`$. The advective flux through a face is the face mass flux times the
  upwind concentration trace, so tracer moves with exactly the water that moves. Traces are
  MC-limited with the Hancock predictor of $`c_t + u c_x + v c_y = 0`$ and clipped to the range of
  the cell and its four neighbours; they are first order where the flow reconstruction is.
  Ghost sides copy the interior concentration; inflow through a hydrograph face carries the
  segment's concentration. Diffusion uses the shallower of the two cell depths across interior
  and periodic faces only, and adds the bound $`D \Delta t (1/\Delta x^2 + 1/\Delta y^2) \le 1/4`$ to the step. For flux
  evaluation the concentration $`hc/h`$ is clipped to the range of the data (initial, source and
  inflow concentrations, and 0 under rain), which only matters in thin cells where the quotient
  loses precision; the stored amount is never altered, so the scheme stays conservative. A cell
  that dries completely keeps no tracer (its residual is rounding). Consequences: a uniform
  concentration stays uniform, a lake at rest leaves any tracer field unchanged, and the f64
  references obey the maximum principle.
- **Turbulence closure**: depth-averaged eddy viscosity
  $`\nu = \nu_0 + \big(C_s \min(\Delta x, \Delta y)\big)^2 |S| + a u_* h`$, with $`|S|`$ from central differences of the cell velocities (mapped and reflected as for the
  reconstruction) and zero in solid and thin cells. Viscous momentum fluxes $`-h_f \nu_f \nabla u`$
  (Laplacian form; $`h_f`$ the shallower cell depth, $`\nu_f`$ the mean of the two cells) act across
  interior and periodic faces only, and the tracer diffusivity becomes $`D + \nu_f / Sc_t`$. The
  Smagorinsky part stays within the CFL step for $`C_s \le 0.3`$ on any aspect ratio (docs/METHOD.md);
  the cell-local parts add
  $`4 \, \mathrm{CFL} \, \nu_\mathrm{eff} (1/\Delta x^2 + 1/\Delta y^2) / (1/\Delta x + 1/\Delta y)`$ to the wave speeds.
- **Boundary accounting**: the net volume through every non-periodic boundary face is
  accumulated on the GPU with compensated (Kahan) f32 sums and summed in f64 by the
  host. Budget: $`V(t) + \text{outflow} = V(0) + \text{sourceVolume}`$, with hydrograph inflow negative.

## Bindings

Binding 2 is the cell record `vec4<f32>(z, n, source, solid)`. The uniform
is u32 width, height, boundaryX, boundaryY (wall 0, periodic 1, open 2, outflow 3);
f32 dx, dy, gravity, cfl, maxDt, rain; then, for the multi-pass kernels, the hydrograph clock
(f32 lo, hi), and for the fused kernel u32 parity and padding; then (fused) u32 tracer flag and
f32 diffusivity and concentration range (cmin, cmax); then (fused, verification only) f32 faultDt
and faultTime, u32 lowFroude (word 18, read by both MUSCL kernels) and padding; then (fused) f32
eddyViscosity, smagorinsky, bedTurbulence and 1/schmidt; then f32 wind stress $`\boldsymbol{\tau}/\rho_w`$ ($`x`$, $`y`$; words
24 and 25, read by all kernels), (fused) the settling velocity and exponent, and f32 coriolis (word 28,
all kernels). The uniform is 128
bytes; the multi-pass kernels read the first 48 bytes, word 18 (MUSCL) and words 24 and 25.
`faultInjection: { dt, time }` in the options makes the fused kernel reject every step longer
than `dt` that starts at or after `time`, which exercises rollback, prefix replay and retry
deterministically; it is meant for tests only. The hydrograph clock is a two-float
f32 sum of accepted timesteps advanced with TwoSum; the host (`clockAdvance` in `src/model.mjs`)
and the kernels perform the same f32 operations, so every step evaluates its hydrographs at the
same time however steps are batched.

The boundary buffer holds $`B = 2(w + h)`$ compensated water accumulators `vec2<f32>(sum,
compensation)`, one per boundary face (west $`y`$, east $`h + y`$, south $`2h + x`$, north
$`2h + w + x`$), then B tracer accumulators, followed by the read-only hydrograph table
(`hydrographTable` in `src/model.mjs`): a per-face segment index, three header entries per
segment (kind and scale; series start and count, as exact f32 integers; inflow concentration),
and the `(t, value)` and `(cumulative, 0)` pairs. Checkpoints and readbacks cover only the
accumulators.

| Binding | `rusanov` / `muscl-hancock` | `muscl-fused` |
|---|---|---|
| 0 | Uniform parameters | Uniform parameters (one buffer per parity) |
| 1 | Old state (read) | Old state (read) |
| 2 | Cells (read) | Cells (read) |
| 3, 4 | X/Y face records | New state |
| 5 | New state | Control and dt log |
| 6 | Control (atomics) | Boundary accumulators and hydrograph table |
| 7 | Boundary buffer (`rusanov`) / traces (`muscl-hancock`) | Tracer in / out: 6, 7 (`vec2<f32>` amount, compensation) |
| 8 | Boundary buffer (`muscl-hancock`) | Source concentration (`f32`) |

Fused control words: 0 dt; 1–2 and 3–4 wave-speed maxima for the two parities;
5 error bits; 6 HLL fallbacks; 7 order reductions; 8 step index; 9 remaining time;
10 first failing step; 11–12 hydrograph clock (hi, lo) at the start of the current step;
13–14 running clock (written by the host at batch start); 16+ per-step dt log. Results are
bitwise independent of the batch size, and equal to the step-by-step multi-pass kernel. When a batch fails, the host restores the
checkpoint, replays the valid prefix as one batch (steps are deterministic), and applies the
halve-dt retry to the failing step only.

## Fused kernel design

Each 16×16 workgroup loads a 20×20 tile (two-cell halo) of `[h, u, v, z]` plus the
solid mask into workgroup memory, with the same boundary mapping and wall
reflection as the multi-pass `primitive()`. Each thread reconstructs and predicts
its own cell in registers, and 64 threads also reconstruct one halo cell whose
facing trace is needed. Each thread computes its west and south faces, threads on
the tile's east and north edges compute one more face, and the conservative update
reads neighbouring faces from workgroup memory. Faces on workgroup seams are
evaluated by both neighbours from identical inputs with identical code, so the
fluxes are bitwise equal and interior mass still telescopes. Fallback counters
count each face once.

The update also reduces the new state's wave speeds within the workgroup and
writes one atomic per workgroup into the next step's slot. The `timestep` kernel
reads that slot, clears the other one, and caps dt by `maxDt` and the remaining
time. Once the target is reached, later steps in the batch only copy the state.
Workgroup memory is 16,368 bytes of the 16,384 that WebGPU guarantees (state tile, traces,
per-cell bed, depth and solid flags, and the tracer tile, which holds cell concentrations, then
concentration traces, then tracer face fluxes). Global traffic is about 50 B/cell/step,
against about 600 B for the multi-pass path. Memory use is 80 B/cell (192 B
multi-pass), including the checkpoint used for rollback; the tracer adds 28 B/cell (two
amount buffers, their checkpoint, and the source concentration). With the tracer off its
buffers are 16-byte placeholders the kernel never touches.

## Present limits

- f32 only. On thin rain films over uneven terrain, the MUSCL dry-stencil switch
  is sensitive to f32 cancellation in $`\eta - z`$. GPU and f64 results then differ more
  than on wet problems, while budgets still close.
- Source and friction splitting is first order. Wet/dry fallbacks reduce order on purpose.
- Stage boundaries take the interior velocity for outflow (inflow enters from rest at the stage)
  and reflect outgoing waves; the `wave` kind is the non-reflecting option for long waves at normal
  incidence. The discharge along a segment is uniform per unit width (not conveyance weighted).
- Timestep selection uses global atomic maxima; one global dt per step (no local
  time stepping).
- The GPU results reported in docs/VERIFICATION.md come from a single adapter; other devices
  run the same suites (`tools/validate/`) to confirm agreement.
