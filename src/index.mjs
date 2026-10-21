// Public entry point. GPU solvers (WebGPU), the f64 CPU references, and the configuration and
// diagnostic helpers they share. Benchmarks, test fixtures and DEM tiles are separate subpaths
// (see package.json "exports").
export { SCHEMES, compileKernels, createSolver, WebGpuSolver, FusedMusclSolver } from './webgpu.mjs';
export { ReferenceSolver } from './reference.mjs';
export { MusclReferenceSolver } from './muscl-reference.mjs';
export { configuration, diagnostics, hydrographValue, BOUNDARIES, EDGES, HYDROGRAPH_KINDS, H_THIN } from './model.mjs';
