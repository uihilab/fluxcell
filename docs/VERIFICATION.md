# Verification

FluxCell is verified at three levels. Each GPU kernel is compared with a double-precision (f64) CPU
implementation of the same method, both are compared with exact solutions, and the fused kernel is
compared with published benchmark data. The method is described in [METHOD.md](METHOD.md) and the
interface in [API.md](API.md). The GPU results were obtained on an Intel integrated GPU (xe-2lpg) in
Chromium and Microsoft Edge on Windows 11. Depth errors are in metres and unit-discharge errors in
m²/s unless stated otherwise; $`L_1`$ is the mean absolute error of cell averages.

## 1. How verification is organised

**CPU references.** `src/reference.mjs` (first-order Rusanov) and `src/muscl-reference.mjs`
(MUSCL-Hancock/HLLC) implement the full model in f64. `npm test` runs 150 tests in Node 20 or later
(about ten minutes, mostly refinement studies and seeded stress cases). The tests cover exact
equilibria, closed budgets, CFL and clock rules, HLLC branches (dry, vacuum, contact, supersonic),
convergence, exact Riemann problems, friction, sources, solid cells, all boundary types, SWASHES
regressions, jumps and dipoles (`flows`), transport, the turbulence closure, settling, wind,
Coriolis, run-time changes, sparse sources and state replacement, the datum shift and DEM loader, and 2 × 60 seeded stress cases.

**GPU suites.** `npm run serve` serves the repository at `http://localhost:8080/`. Each suite is a
page under `tools/` that runs in a WebGPU browser and downloads a JSON report.

| Page | Checks |
|---|---|
| `tools/validate/` | Per kernel (Rusanov, multi-pass MUSCL, fused MUSCL): 1,000-step cases against f64 and exact equilibria, budgets, grids from 1×1 to 65×65 with workgroup seams inside obstacles and boundaries, repeatability, convergence, dam breaks, fused against multi-pass equivalence, batch invariance, run-time changes, `setBed`, sparse sources, and state replacement |
| `tools/swashes/` | Thacker paraboloid and Goutal–Maurel bump flows, GPU and both f64 references |
| `tools/transport/` | Tracer equivalence with f64, batch invariance, forced rollback, exact solutions, stress; part `closure` for the turbulence closure |
| `tools/ea/` | UK Environment Agency tests 1 to 6B (data downloaded separately, see `tools/ea/README.md`) |
| `tools/nthmp/` | NTHMP BP6 and BP7; laboratory comparison by `tools/nthmp/analysis.py` |
| `tools/stress/` | 500 seeded random cases, every tenth compared bitwise between fused and multi-pass kernels |
| `tools/precision/` | Elevation offsets from 0 to 8,000 m with and without the datum shift |
| `tools/dynamics/` | Kelvin–Helmholtz growth, island wake, vortex leaving through a stage edge, decaying 2D turbulence |
| `tools/perf/` | Kernel time from timestamp queries, batch-size sweep, GPU time to reach a given error |

`?lowFroude` (validate, swashes, stress, dynamics) and `?smagorinsky=C` (dynamics) repeat a suite
with the low-Froude correction or the Smagorinsky closure.

## 2. Exactness and conservation

Each kernel runs 77 validation cases (9 with wind and 9 with Coriolis), all of which pass. The core cases (17 × 33 cells, 1,000 steps) are
listed below; the fused and multi-pass MUSCL kernels produced identical values. The gates are 1e-5
for the relative budget, 2e-4 for f64 agreement, and 1e-5 for equilibrium error.

| Case | MUSCL max difference from f64 | MUSCL relative budget | Rusanov max difference from f64 | Exact solution error |
|---|---:|---:|---:|---:|
| constant, lake, lake-rounded, shoreline, lake-obstacle, open-uniform, stage-lake, wave-lake | 0 | 0 | 0 | 0 |
| rain on dry ground | 4.7e-12 | 9.3e-9 | 4.7e-12 | 4.7e-12 |
| friction (Manning decay) | 1.1e-6 | 0 | 1.1e-6 | 1.1e-6 |
| dam, wet and dry bed | 1.2e-6, 2.1e-6 | 1.3e-8, 3.7e-9 | 1.6e-7, 2.3e-7 | |
| pulse | 2.0e-7 | 1.2e-9 | 4.5e-8 | |
| sources (rain and point sources, dry start) | 1.9e-8 | 1.3e-8 | 1.4e-8 | |
| obstacle (dam break around buildings) | 4.8e-6 | 6.2e-10 | 2.3e-7 | |
| open-dam (186.0 m³ leaves), outflow-drain (117.4 m³ leaves) | 3.3e-7, 1.7e-6 | 2.6e-9, 4.3e-9 | 5.5e-8, 3.0e-7 | |
| hydrograph-channel (80.0000 m³ enters), wave-channel | 1.5e-7, 1.2e-6 | 4.8e-9, 1.6e-9 | 3.2e-8, 3.0e-6 | |

**Exact rest in f32.** `lake-rounded` holds $`h + z = 2`$ exactly on 561 non-dyadic beds. The HLLC
star-region mass flux is evaluated as S* h*, so identical states at rest give exactly zero mass flux
in single precision, and reflective faces carry exactly zero mass flux. All lakes, including those
held by stage and wave boundaries, stay exactly at rest in all three kernels.

**Determinism.** Five repeated runs per suite are bitwise identical. Fused and multi-pass MUSCL
kernels give bitwise-identical states, timesteps, outflow volumes, and fallback counters on 11
equivalence cases of 200 steps (1×1 to 65², including 129×3) and on every tenth stress seed.
Batches of 1, 16, 64, and 256 steps give bitwise-identical states, outflow volumes, and clocks over
300 steps, including hydrograph and wave-boundary cases. Bitwise reproducibility across devices is
not claimed, because WGSL division and transcendental functions are not correctly rounded.

**Datum and precision sweep.** Five problems were lifted by 0 to 8,000 m (bed and stage shifted in
f64), run for 400 steps on the fused kernel, and compared with f64 at the same offset. With
`datum: 'auto'` every result is bitwise identical at every offset (lake-rounded 0, Thacker 2.4e-8,
pulse 1.8e-7, thin-film sources 2.3e-9, stage-lake 0). With raw elevations, errors appear only
where the bed itself is not representable in f32:

| Case, raw elevations | 0 m | 10 m | 100 m | 1,000 m | 3,000 m | 8,000 m |
|---|---:|---:|---:|---:|---:|---:|
| lake-rounded | 0 | 6.0e-7 | 4.1e-6 | 3.5e-5 | 1.4e-4 | 2.8e-4 |
| Thacker paraboloid, 64² | 2.8e-8 | 2.1e-6 | 1.7e-4 | 2.0e-4 | 5.7e-4 | 6.7e-4 |

The pulse, sources, and stage-lake beds are exactly representable, and their raw results equal the
shifted ones bitwise; the difference-form reconstruction never forms h + z. Budgets close to 1e-9
or better in every run.

## 3. Accuracy and convergence

| Smooth problem (fused GPU, f32) | Cells | $`L_1`$ error | Observed orders |
|---|---|---|---|
| Nonlinear simple wave, periodic, $`t = 0.1`$ | 32 to 256 | 1.624e-4, 3.825e-5, 8.408e-6, 1.886e-6 | 2.086, 2.186, 2.156 |
| Diagonal 2D wave | 16² to 64² | 4.691e-4, 1.158e-4, 2.629e-5 | 2.018, 2.140 |
| Moving equilibrium over a sinusoidal bed | 16 to 64 | 1.497e-4, 2.801e-5, 4.426e-6 | 2.418, 2.662 |

The first-order control has $`L_1`$ = 2.409e-4 on the simple wave at 256 cells. The results support
second-order behaviour on smooth problems, not a higher general order.

**Dam breaks** (flat 100 m domain, 256 cells, left depth 2 m, $`t = 2`$ s). Against the exact Riemann
solution, MUSCL has $`L_1`$ = 0.0017622 on a wet bed (right depth 0.25 m) and 0.0022598 on a dry bed,
0.130 and 0.151 times the Rusanov errors of 0.013505 and 0.015008.

**SWASHES, Thacker planar surface in a paraboloid** (Delestre et al., 2013; Thacker, 1981). The
shoreline moves; $`L = 4`$ m, $`a = 1`$ m, $`h_0 = 0.1`$ m, $`\eta = 0.5`$, and the period is $`T = 4.4857`$ s.

| Cells | GPU $`L_1`$ at T/2, T, 2T | GPU $`L_\infty`$ at T | Relative volume change | f64 MUSCL $`L_1`$ at T | f64 Rusanov $`L_1`$ at T |
|---|---|---:|---:|---:|---:|
| 50² | 9.43e-4, 1.68e-3, 2.74e-3 | 2.07e-2 | 6.8e-10 | 1.68e-3 | 4.21e-3 |
| 100² | 3.99e-4, 6.52e-4, 1.02e-3 | 1.50e-2 | 7.7e-10 | 6.52e-4 | 2.39e-3 |
| 200² | 1.57e-4, 2.39e-4, 3.45e-4 | 7.84e-3 | 2.9e-10 | | |
| 400² | 6.69e-5, 1.05e-4, 1.39e-4 | 5.43e-3 | 6.6e-10 | | |

$`L_1`$ orders at T are 1.36, 1.45, and 1.19 (Rusanov 0.81), limited by the wet/dry front. The f32 GPU
errors equal the f64 MUSCL errors to three significant digits.

**SWASHES, Goutal–Maurel flows over a bump** (Goutal and Maurel, 1997; 25 m channel, discharge inflow, stage outflow, 200 s
from still water; exact states from Bernoulli's equation and, for the jump, the Rankine–Hugoniot
conjugate depth at $`x = 11.6656`$ m):

| Case (q, downstream stage) | GPU $`L_1`$ at 100, 200, 400, 800 cells | Orders | f64 Rusanov $`L_1`$ at 100, 200 |
|---|---|---|---|
| Subcritical (4.42, 2.0) | 3.17e-4, 6.53e-5, 1.54e-5, 3.88e-6 | 2.28, 2.09, 1.98 | 7.53e-3, 4.15e-3 |
| Transcritical (1.53, 0.66) | 1.04e-3, 2.85e-4, 7.75e-5, 2.10e-5 | 1.87, 1.88, 1.89 | 4.65e-2, 2.40e-2 |
| With jump (0.18, 0.33) | 1.28e-3, 4.16e-4, 1.82e-4, 1.06e-4 | 1.62, 1.20, 0.78 | 7.35e-3, 4.51e-3 |

At 100 cells MUSCL is 24 (subcritical) and 44 (transcritical) times more accurate than Rusanov. On
the transcritical case at 50 cells, Rusanov settles on a different steady state (upstream depth
0.909 m against 1.014 m exact), and MUSCL reproduces the exact profile. The stage is correctly not
imposed on the supercritical outflow.

## 4. Benchmarks against data

**UK Environment Agency 2D benchmarks** (Néelz and Pender, 2010, 2013). The tests compare software
packages and, apart from 6A, have no measured truth. They were checked through exact volumes, the
pond level of Test 1, agreement between the specified resolution and half the cell size, and the
behaviour each test targets. No run needed a retry.

| Test | Grids | Budget residual | Inflow (m³), exact |
|---|---|---:|---|
| 1 Disconnected pond | 70 × 10 (10 m); 140 × 20 (5 m) | 3.1e-8; 2.6e-8 | stage boundary |
| 2 Floodplain depressions | 100² (20 m); 200² (10 m) | 1.1e-8; 1.1e-8 | 97,200.00; 97,199.99 (97,200) |
| 3 Momentum over an obstruction | 60 × 20 (5 m); 120 × 40 (2.5 m) | 1.7e-8; 9.3e-9 | 1,310.00; 1,310.00 (1,310) |
| 4 Flood propagation | 200 × 400 (5 m); 400 × 800 (2.5 m) | 1.8e-8; 1.8e-8 | 285,000.00; 285,000.00 (285,000) |
| 5 Valley flooding | 256 × 225 (50 m); 511 × 450 (25 m) | 2.8e-10; 8.9e-10 | 9,450,000.09; 9,449,999.96 (9,450,000) |
| 6A Dam break, laboratory | 1,010 × 56 (0.1 m); 2,021 × 112 (0.05 m) | 8.9e-11; 6.1e-11 | |
| 6B Dam break, field | 1,010 × 56 (2 m); 2,021 × 112 (1 m) | 1.8e-10; 5.2e-12 | |

In Test 1 the pond filled to 10.352 m (boundary 10.35 m) and drained to 10.252 m and 10.256 m
after 20 h, against the sill level of 10.25 m. In Test 2 final levels agreed between grids within
1 cm at most points and 4 cm at worst. In Test 3 water overtopped the obstruction and settled in the
second depression (9.827 m and 9.830 m), the inertial behaviour the test targets. In Tests 4 and 5
arrival times agreed between grids within one output interval, peak depths within 3 mm (Test 4),
and peak levels within 9 cm (Test 5). Gauge peaks in Tests 6A and 6B differ by up to 1.5 cm and
10 cm between grids.

**NTHMP BP6, solitary waves on a conical island** (Briggs et al., 1995). The acceptance criterion
(Synolakis et al., 2007) is runup on the back of the island within 20 % for cases B and C (A
optional). Setup choices were fixed before the comparison: shallow-water solitary wave, paddle face
a wall, absorbing edges open, no friction ($`n = 0.015`$ as a sensitivity), and one time shift per run
from gauge 1. The table gives errors in maximum water level at gauges and in runup.

| Run | g1 | g9 | g16 | g22 | Back runup (90°) | Mean runup error |
|---|---:|---:|---:|---:|---:|---:|
| C, 2.5 cm | +1.0 % | +5.3 % | +24.3 % | +11.1 % | −15.7 % | 15.7 % |
| C, 1.25 cm | +1.3 % | +2.2 % | +28.6 % | −18.3 % | −5.8 % | 18.6 % |
| C, 2.5 cm, $`n = 0.015`$ | +0.7 % | +5.1 % | +9.2 % | −6.8 % | −15.7 % | 12.0 % |
| B, 2.5 cm | +10.4 % | +9.8 % | +13.5 % | +54.8 % | +14.0 % | 40.3 % |
| B, 1.25 cm | +10.5 % | +9.2 % | +18.2 % | +78.9 % | +15.8 % | 37.6 % |
| B, recorded height, 2.5 cm | +0.4 % | +3.0 % | +0.7 % | +38.9 % | +6.7 % | 28.5 % |
| B, recorded height, 1.25 cm | +0.4 % | +2.3 % | +3.7 % | +46.3 % | +26.7 % | 35.5 % |
| A (optional), 2.5 cm | +3.6 % | +13.9 % | −0.5 % | +27.5 % | +30.5 % | 28.7 % |

Case C meets the criterion on both grids, and so does case B with the stated height. Case B with
the recorded height meets it on 2.5 cm cells but not on 1.25 cm cells; case A does not. The problem
description gives H/d = 0.096 for case B, whereas the mean incident peak of gauges 1 to 4 is 0.0872
(A: 0.0444 against 0.045; C: 0.1788 against 0.181), so case B was run with both. Peak times agree
within 0.55 s at all gauges.

**NTHMP BP7, runup onto the Monai Valley beach** (Liu et al., 2008). The criterion is maximum runup
in the valley within 10 %. The incident record is imposed by the `wave` boundary, which becomes
non-reflecting after the record ends; no friction ($`n = 0.01`$ as a sensitivity), no time shift.

| Run | g5 | g7 | g9 | Peak times | Runup at the valley head |
|---|---:|---:|---:|---|---:|
| $`n = 0`$ | +3.8 % | +5.3 % | +3.1 % | within 0.39 s | 8.17 cm (−8.8 %) |
| $`n = 0.01`$ | 0.0 % | +1.4 % | −0.3 % | within 0.16 s | 8.17 cm (−8.8 %) |

The observed runup at the valley head is 8.96 cm on average and 8.0 to 10.0 cm over six experiments
(Matsuyama and Tanaka, 2001). At the two other runup points the errors are +14.8 % and +0.3 %
($`n = 0`$).

## 5. Source terms, boundaries, and transport

**Hydrographs.** Discharge faces use the mean of the hydrograph over each step, so a discharge
boundary admits the exact integral of $`Q(t)`$, and hydrographs bound the timestep through their
largest ghost wave speed. CPU tests show that a stage equal to the lake level keeps a lake exactly
at rest, that constant and varying discharges add exactly the prescribed volume, that a delayed
inflow onto a dry bed is admitted exactly, and that discharge in with stage out reaches the steady
channel flow. On the GPU, `hydrograph-channel` agrees with f64 on seven grid shapes and admits
80.0000 m³, the exact integral of its ramp.

**Stage backflow.** Water entering through a stage face comes from rest outside the domain (ghost
depth $`\eta_b - z_f - u_n^2/(2g)`$ for the stage $`\eta_b`$ and face bed $`z_f`$, no tangential velocity). A Gaussian vortex ($`R = 3`$ m, peak swirl
1.2 m/s) carried out through a stage edge was run in nine configurations on 0.5, 0.25, and 0.125 m
cells. Speed never exceeded its initial maximum by more than 0.8 %, swirl energy never exceeded its
initial value, at most 0.09 % of it remained after 200 s, and no step was retried.

**Incident-wave boundary.** The `wave` face state combines the incoming invariant of a simple wave
at the prescribed level with the outgoing invariant of the interior. On the CPU a lake at the still
level stays at rest to 1e-12, a leaving Gaussian wave reflects less than 2 % of its amplitude, an
imposed wave enters within 5 % of its height, and continuous waves against a wall build a standing
wave of $`1.98 A`$ ($`2 A`$ expected, $`A`$ the incident amplitude) that stays unchanged from 60 to 120 s. On the GPU `wave-lake` is exact
and `wave-channel` agrees with f64 within 1.2e-6 on eight grid shapes.

**Wind stress** (Wu, 1982, or constant drag). Uniform wind on a periodic domain matches $`q = \tau t`$ to
1e-12 on the CPU and to 8.2e-9 m²/s after 1,000 GPU steps. On the CPU, wind against Manning
friction stays within $`2 \times 10^{-3} q_\infty`$ of $`q_\infty \tanh(\tau t / q_\infty)`$ at 50, 200, and 2,000 s. A closed basin started from its exact
6.2 cm setup deviates by less than 1e-3 (MUSCL) and 5e-3 (Rusanov) of the setup after 300 s at 100
cells. On `wind-basin` the GPU kernels agree with f64 within 1.3e-6 and close the budget to 3e-9.

**Settling** (constant velocity or flocculation power law; Mehta, 1986). The local update equals the
exact solution to 1e-14 and composes over 600 steps to 1e-13. On the CPU a lake at rest follows the
exact $`c(t)`$ in every cell to 1e-12, and moving water keeps concentrations in $`[0, c_\mathrm{max}]`$ with the
budget closed to 1e-11. On the GPU a lake at rest follows $`c(t)`$ to 2.5e-5 (constant) and 2.1e-5
(flocculating) relative after 400 steps, and seven fixtures with both laws agree with f64 within
1.6e-5 of the range without bound violations.

**Coriolis** (`tests/coriolis.test.mjs`, both references). The rotation step conserves $`|\mathbf{q}|`$ to
1e-13 over 1,000 steps, and a uniform current follows the exact inertial circle to 1e-12 over
3,000 s. A geostrophic jet ($`h = h_0 + a \tanh\big((y - L/2)/w\big)`$, $`f = 10^{-4}`$ s$`^{-1}`$, $`4 \times 10^4`$ s) stays in balance
with MUSCL. The depth error is below 1 % of $`a`$ at 128 cells and less than half the 32-cell error,
and volume is conserved to 1e-11. A rotating basin with an obstacle keeps depths nonnegative and
volume to 1e-12. On the GPU (all three kernels) the inertial circle is exact to 1.05e-7 after 1,000
steps, and the rotating basin agrees with f64 within 9.2e-7 (MUSCL) and 1.6e-7 (Rusanov) on eight
grid shapes. The Coriolis rotation is evaluated with a Taylor series and angle halving, because
the WGSL sin and cos built-ins guarantee only an absolute accuracy of $`2^{-11}`$.

**Flow cases** (`tests/flows.test.mjs`, f64 MUSCL):

| Case | Criterion met |
|---|---|
| Stationary hydraulic jump | Depths within 1 % of the Bélanger conjugate depths; jump within 6 cells of its position after 60 s |
| Oblique jump at a wall deflection | Angle within 5 % of Ippen's relation and depth ratio within 2 % at 0.5 m cells; angle error not larger than at 1 m |
| Lamb–Chaplygin dipole (96², low-Froude correction) | Translation speed within 10 % of $`U`$ over 100 s; lateral drift below 1 m |

**Run-time changes** (`tests/runtime.test.mjs`, both references). A wind change mid-run gives
$`q = \tau_1 t_1 + \tau_2 (t - t_1)`$ to 1e-12, Coriolis switched on mid-run turns the current from its value at
that time to 1e-12, and `setHydrographs` mid-run equals one run with the combined series (state to
1e-12, outflow to 1e-9). On the GPU, the same changes applied at the same step to a kernel and its f64 reference keep them
within 8.9e-8 (a raised inflow series) and 1.2e-6 (new wind and Coriolis on the wind basin) with
MUSCL, and 3.2e-8 and 1.7e-7 with Rusanov. After `setBed` the depths are unchanged exactly, and the
flow the raised bed then drives closes the water budget to 3.0e-10 over 200 steps.

**Sparse sources and state replacement** (`tests/state-sources.test.mjs`, both references, with and
without a tracer). Moving point inflows updated with `setSources` give states, tracer amounts and
source budgets bitwise equal to `setForcing` with the full array; after `setState`, a run equals one
started from that state, and the budget closes to 1e-11 with the edit volume included. On the GPU
(37 × 29 cells, 8 frames of moving inflows, 200 steps), the kernels stay within 9.6e-8 (Rusanov) and
4.6e-7 (MUSCL) of their f64 reference with `setSources`, and within 9.1e-8 and 2.4e-7 after
`setState` and after the same edit written directly into `currentState`; the inflow rate kept from
the sparse changes equals a full recomputation exactly.

**Scalar transport.** Fifteen runs of eleven fixtures (37 × 29 cells, 300 steps, source water at
$`c = 1.5`$, inflow at $`c = 2`$, four runs with $`D = 0.05`$ m²/s) were compared with f64. The worst flow
state difference is 3.8e-6, the tracer amount difference 1.1e-6 of the largest concentration, the
GPU tracer budget 4.1e-8, and the bound violation 7.1e-8 of the range; a uniform $`c = 0.3`$ stays
uniform to 7.0e-7 through dry fronts, obstacles, and every boundary type.

| Exact solution (GPU) | Grids | Finest $`L_1`$ | Observed orders |
|---|---|---|---|
| Gaussian advected in uniform flow, 2 s | 64 to 512 | 6.2e-5 | 1.80, 1.93, 1.96 |
| Square pulse, one period | 256 | 1.6e-2 | bounded in [0, 1] |
| Gaussian diffusing in still water, $`D = 0.05`$, 10 s | 32 to 256 | 1.1e-5 | 2.04, 2.10, 2.22 |
| Steady cyclostrophic vortex, depth, 60 s | 64 to 256 | 6.6e-6 m | 2.75, 2.57 |
| Same vortex, tracer front wound by rotation | 64 to 256 | 2.0e-3 | 1.28, 1.34 |
| Steady mixing layer against erf, $`D = 0.02`$ | 50 to 400 | 3.4e-4 | 1.65, 1.92, 1.97 |

On the CPU, Rusanov advects the Gaussian at orders 0.84 and 0.92 against 1.80 and 1.93 for MUSCL.

**Turbulence closure.** A viscous shear layer with constant $`\nu = 0.05`$ m²/s (20 s) tests the viscous
terms against the exact erf solution. From 32 to 256 cells the f64 MUSCL $`L_1`$ falls from 1.87e-4 to
2.65e-6 m/s (orders 2.03, 2.04, 2.08) and the GPU $`L_1`$ from 1.87e-4 to 8.26e-6 m/s (orders 2.02,
1.84, 0.64). The f64 Rusanov errors remain between 7.58e-2 and 1.72e-2 m/s, because the scheme
dissipates the shear by itself.

With the Smagorinsky (1963) closure ($`C_s = 0.2`$), seven fixtures agree with f64 within 4.1e-6 in
state and 1.1e-6 in tracer, with tracer budgets to 4.7e-9. A lake at rest and uniform flow stay
unchanged to 1e-12 with the closure on.

**Vortical flows and the low-Froude correction** (fused GPU; correction after Thornber et al.,
2008):

| Measure | Default | `lowFroude` | Smagorinsky 0.15 |
|---|---|---|---|
| Kelvin–Helmholtz $`\sigma \delta / U`$, 128² / 256² / 512² (theory 0.1897) | 0.164 / 0.185 / 0.187 | 0.183 / 0.188 / 0.187 | 0.132 / 0.174 / 0.184 |
| Decaying turbulence, energy lost in 300 s, 256² / 512² | 21 % / 6.2 % | 7.5 % / 2.3 % | 24 % / 7 % |
| Island wake Strouhal number, 1 / 0.5 / 0.25 m cells | 0.196 / 0.213 / 0.199 | | 0.195 / 0.220 / 0.210 |

The Kelvin–Helmholtz growth rate converges to about 0.188 at 1024², within 1 % of linear theory
(Michalke, 1964). The Strouhal numbers lie in or just above the 0.19 to 0.21 range of circular
cylinders (Williamson, 1996). Decaying turbulence moves its energy peak from $`k = 9`$ to $`k = 2`$ and
conserves mass and tracer to 4.6e-10 or better. With `lowFroude` the subcritical bump $`L_1`$ at 800
cells falls from 3.88e-6 to 2.32e-6, the transcritical rises from 2.10e-5 to 2.14e-5, Thacker stays
at 1.05e-4, and the validation and stress suites pass.

## 6. Robustness

Four rules are applied in all implementations: first-order reconstruction where a stencil depth is
at most 1e-4 m, first-order reconstruction where a bed step exceeds the cell depth, thin-layer
momentum regularization, and a compensated GPU depth update with retry of any invalid step.

| Stress suite | Cases | Result |
|---|---:|---|
| CPU, Rusanov and MUSCL (up to 20², 200 steps) | 2 × 60 | 0 retries; budget at most 1e-10 |
| GPU fused (up to 96², 400 steps) | 500 | 0 failures, 0 retries; budget at most 2.6e-7; 50 seeds bitwise equal to multi-pass |
| GPU with tracers | 200 | 0 failures, 0 retries; tracer budget at most 1.6e-7 |
| GPU with the closure | 200 | 0 failures, 0 retries; tracer budget 2.5e-7; bounds held to 1.3e-5 of the range |

Cases combine anisotropic grids, relief up to 500 m with vertical cliffs, elevation offsets up to
8,000 m, films down to 1e-9 m, deep water, initial speeds up to 40 m/s, every boundary type (78
with random wave segments), rain, point sources, friction, and buildings.

**Rollback and retry.** A verification-only fault injection rejects every step longer than 0.6 of
the first CFL step, so batches fail part-way, the valid prefix is replayed, and the failing step is
retried at half dt. In five tracer cases (57 to 131 retries), two closure cases (91 and 65 retries),
and the settling cases, the accepted run equals a plain step-by-step run with the same timesteps
bit for bit, including batches recorded with `encode()` and recovered by `resolve()`.

**Real terrain.** Six cases on two 256² USGS 3DEP 10 m tiles (Yosemite with El Capitan, and New
Orleans below sea level) passed all gates without retries: budget at most 1.3e-8, lakes at rest,
fused and multi-pass bitwise, and depth within 2.0e-6 m of f64 over 40 steps. Runs on 2 m lidar
(a 600 m³/s breach of the 17th Street Canal, 1024²; a 353,300 m³ release on El Capitan, 1536²)
closed their budgets to 3.0e-10 or better without retries.

## 7. Performance

- Measured with timestamp queries, the fused kernel takes 1.655 ms per step at 1024² (634
  Mcell-steps/s) on a wet pulse and runs at 610 to 700 Mcell-steps/s from 256² to 1024², falling to
  406 to 474 at 2048² (4.2 M cells), where the multi-pass kernels exceed the dispatch limit.
- The fused kernel is 3.4 to 7.4 times faster than multi-pass MUSCL and 2.2 to 4.9 times faster
  than first-order Rusanov, using 80 B per cell (192 B multi-pass, 128 B Rusanov).
- One host round trip costs 3.5 to 4 ms; 256 steps per submission raise wall-clock throughput from
  17 to 464 Mcell-steps/s at 256² and from 106 to 611 at 1024².
- On Thacker's paraboloid MUSCL reaches $`L_1`$ = 6.5e-4 on 100² cells with 31 ms of GPU time; Rusanov
  needs 400² and 5.9 s, about 190 times more.

## 8. Known limitations revealed by verification

- **No frequency dispersion.** In BP6 case B the overshoot at gauge 22, where the split fronts
  collide, grows with refinement, and case C overpredicts runup on the face toward the wave by about
  35 % (R/d 0.74 against 0.55).
- **First-order control.** Rusanov's mass diffusion flattens the geostrophic jet at the tested
  resolutions, so the jet test runs on MUSCL only.
- **Fronts and shocks.** The order falls to 1.2 to 1.5 on Thacker's moving shoreline and toward one
  with a captured hydraulic jump. Run to 12,800 s, the transcritical bump keeps a bounded MC-limiter
  cycle (face mass-flux deviation 3e-4) at the bed-slope discontinuity, and the subcritical bump an
  undamped seiche of about 5e-4.
- **Low-Froude dissipation.** Without the correction, upwind dissipation damps vortical flow on
  coarse grids (Kelvin–Helmholtz 13 % below theory at 128², 4 % with `lowFroude`).
- **Single precision.** The GPU viscous shear layer stalls at about 8e-6 m/s because the momentum
  update is not compensated. In one stress seed ($`\Delta x = 17`$ m, $`\Delta y = 0.55`$ m, 40 m/s, 8,000 m offset) a
  tracer bound is exceeded by 9.5e-5 of the range through f32 cancellation; f64 shows none.
- **Splitting and settling.** Source and friction terms are split at first order, and the GPU
  kernel does not accumulate deposited mass.
- **Boundaries.** Open boundaries are zero-gradient, and the `wave` boundary is non-reflecting only
  for long waves at normal incidence. The partial reflection of the BP6 absorbing edges (about
  12 %) is not modelled.
- **Steep terrain.** The shallow-water equations accelerate flow by g tan θ on near-vertical faces
  (53.2 m/s in thin sheets on El Capitan), so velocities on cliff faces should not be interpreted.
- **One adapter.** Absolute rates and last-bit reproducibility on other GPUs are not established.

## 9. Data sources and references

Benchmark data come from the UK Environment Agency 2D benchmark datasets (2010, Open Government
Licence v3.0; https://assets.publishing.service.gov.uk/media/6033a9bbd3bf7f72182e99ca/Benchmarking_Model_Data.zip),
the NTHMP benchmark repository (https://github.com/rjleveque/nthmp-benchmark-problems, BP06 and
BP07), and the USGS 3D Elevation Program (Sugarbaker et al., 2014).

- Briggs, M. J., Synolakis, C. E., Harkins, G. S., & Green, D. R. (1995). Laboratory experiments of tsunami runup on a circular island. *Pure and Applied Geophysics, 144*, 569–593.
- Delestre, O., Lucas, C., Ksinant, P.-A., Darboux, F., Laguerre, C., Vo, T.-N.-T., James, F., & Cordier, S. (2013). SWASHES: A compilation of shallow water analytic solutions for hydraulic and environmental studies. *International Journal for Numerical Methods in Fluids, 72*(3), 269–300. https://doi.org/10.1002/fld.3741
- Goutal, N., & Maurel, F. (1997). *Proceedings of the 2nd Workshop on Dam-Break Wave Simulation* (Report HE-43/97/016/B). Electricité de France, Direction des Etudes et Recherches.
- Liu, P. L.-F., Yeh, H., & Synolakis, C. (2008). Benchmark problems. In *Advanced numerical models for simulating tsunami waves and runup* (Advances in Coastal and Ocean Engineering, Vol. 10, pp. 223–230). World Scientific.
- Matsuyama, M., & Tanaka, H. (2001). An experimental study of the highest run-up height in the 1993 Hokkaido Nansei-Oki earthquake tsunami. In *Proceedings of the International Tsunami Symposium 2001* (pp. 879–889).
- Mehta, A. J. (1986). Characterization of cohesive sediment properties and transport processes in estuaries. In A. J. Mehta (Ed.), *Estuarine cohesive sediment dynamics* (Lecture Notes on Coastal and Estuarine Studies, Vol. 14, pp. 290–325). Springer New York. https://doi.org/10.1007/978-1-4612-4936-8_15
- Michalke, A. (1964). On the inviscid instability of the hyperbolic-tangent velocity profile. *Journal of Fluid Mechanics, 19*(4), 543–556. https://doi.org/10.1017/S0022112064000908
- Néelz, S., & Pender, G. (2010). *Benchmarking of 2D hydraulic modelling packages* (Science Report SC080035). Environment Agency.
- Néelz, S., & Pender, G. (2013). *Benchmarking the latest generation of 2D hydraulic modelling packages* (Report SC120002). Environment Agency.
- Smagorinsky, J. (1963). General circulation experiments with the primitive equations. *Monthly Weather Review, 91*(3), 99–164. https://doi.org/10.1175/1520-0493(1963)091<0099:GCEWTP>2.3.CO;2
- Sugarbaker, L. J., Constance, E. W., Heidemann, H. K., Jason, A. L., Lukas, V., Saghy, D. L., & Stoker, J. M. (2014). *The 3D Elevation Program initiative: A call for action* (U.S. Geological Survey Circular 1399). https://doi.org/10.3133/cir1399
- Synolakis, C. E., Bernard, E. N., Titov, V. V., Kânoğlu, U., & González, F. I. (2007). *Standards, criteria, and procedures for NOAA evaluation of tsunami numerical models* (NOAA Technical Memorandum OAR PMEL-135). NOAA Pacific Marine Environmental Laboratory.
- Thacker, W. C. (1981). Some exact solutions to the nonlinear shallow-water wave equations. *Journal of Fluid Mechanics, 107*, 499–508. https://doi.org/10.1017/S0022112081001882
- Thornber, B., Mosedale, A., Drikakis, D., Youngs, D., & Williams, R. J. R. (2008). An improved reconstruction method for compressible flows with low Mach number features. *Journal of Computational Physics, 227*(10), 4873–4894. https://doi.org/10.1016/j.jcp.2008.01.036
- Williamson, C. H. K. (1996). Vortex dynamics in the cylinder wake. *Annual Review of Fluid Mechanics, 28*, 477–539. https://doi.org/10.1146/annurev.fl.28.010196.002401
- Wu, J. (1982). Wind-stress coefficients over sea surface from breeze to hurricane. *Journal of Geophysical Research: Oceans, 87*(C12), 9704–9706. https://doi.org/10.1029/JC087iC12p09704
