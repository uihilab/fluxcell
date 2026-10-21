// Type declarations for the public entry point (src/index.mjs). SI units throughout.

export type Scheme = 'rusanov' | 'muscl-hancock' | 'muscl-fused';
export type Boundary = 'wall' | 'periodic' | 'open' | 'outflow';
export type Edge = 'west' | 'east' | 'south' | 'north';
export type HydrographKind = 'stage' | 'discharge' | 'wave';

export const SCHEMES: readonly Scheme[];
export const BOUNDARIES: readonly Boundary[];
export const EDGES: readonly Edge[];
export const HYDROGRAPH_KINDS: readonly HydrographKind[];
/** Thin-layer depth (m) below which reconstruction is first order and momentum is regularized. */
export const H_THIN: number;

/** Edge segment with a prescribed stage (m, bed datum), inflow discharge (m³/s, >= 0), or incident
 *  wave ('wave': incident water-surface elevation, m, on still water at `still`; a characteristic
 *  boundary that generates the incident wave and lets outgoing waves leave). */
export interface Hydrograph {
    edge: Edge;
    /** First cell along the edge (default 0). */
    start?: number;
    /** One past the last cell along the edge (default: edge length). */
    end?: number;
    kind: HydrographKind;
    /** [t (s), value] pairs with strictly increasing t; linear in between, constant outside. */
    series: Array<[number, number]>;
    /** Tracer concentration of water entering through the segment (default 0). */
    concentration?: number;
    /** Still-water level (m, bed datum) of a 'wave' segment; required for that kind. */
    still?: number;
}

export interface Options {
    width: number;
    height: number;
    /** Cell sizes (m); default 1. */
    dx?: number;
    dy?: number;
    /** m/s²; default 9.81. */
    gravity?: number;
    /** Courant number, 0 < cfl <= 0.4; default 0.4. */
    cfl?: number;
    /** Uniform rain (m/s); default 0. */
    rain?: number;
    /** Elevation subtracted from the bed in f64 before f32 rounding; 'auto' (default) = minimum open-cell bed. */
    datum?: 'auto' | number;
    boundaryX?: Boundary;
    boundaryY?: Boundary;
    hydrographs?: Hydrograph[];
    /** Transport a passive scalar (muscl-fused kernels and CPU references); default false. */
    tracer?: boolean;
    /** Tracer diffusivity (m²/s); default 0; requires tracer. */
    diffusivity?: number;
    /** Low-Froude correction of the normal velocity jump (MUSCL kernels); default false. */
    lowFroude?: boolean;
    /** Turbulence closure (muscl-fused and CPU references): constant eddy viscosity (m²/s), Smagorinsky constant (<= 0.3),
     *  bed-generated coefficient (nu_b = a u* h; kappa/6 = 0.067), and turbulent Schmidt number for the tracer (default 0.7). */
    eddyViscosity?: number;
    smagorinsky?: number;
    bedTurbulence?: number;
    schmidt?: number;
    /** Wind at 10 m (m/s, [x, y]); default [0, 0]; speed at most 100 m/s. */
    wind?: [number, number];
    /** Wind drag: 'wu1982' (default, Wu 1982) or a constant coefficient in (0, 0.01]. */
    windDrag?: 'wu1982' | number;
    /** Coriolis parameter f (1/s), |f| <= 1e-3; default 0 (f > 0 in the northern hemisphere). */
    coriolis?: number;
    /** Tracer settling w_s = velocity c^exponent (m/s; exponent 0 to 3); requires tracer (muscl-fused kernels and CPU references). */
    settling?: { velocity: number; exponent?: number };
    /** Optional; must match the kernels passed to createSolver. */
    scheme?: Scheme;
}

/** Physics that may change between steps (setParameters). */
export type RuntimeParameters = Pick<Options, 'wind' | 'windDrag' | 'coriolis' | 'settling'>;

export type Configuration = Readonly<Required<Omit<Options, 'scheme' | 'hydrographs'>>> & {
    readonly hydrographs: ReadonlyArray<Readonly<{ edge: Edge; start: number; end: number; kind: HydrographKind; times: Float64Array; values: Float64Array; concentration: number }>>;
};

type Numbers = ArrayLike<number>;

export interface Initial {
    /** [h, hu, hv, 0] per cell, row-major, row 0 at the south edge (m, m²/s). Dry cells need zero momentum. */
    state: Numbers;
    /** Bed elevation per cell (m). */
    bed: Numbers;
    /** Manning n per cell (s/m^(1/3)); default 0. */
    manning?: Numbers;
    /** Depth source per cell (m/s); a discharge Q over cells with weights w_i is Q w_i / A_i. */
    source?: Numbers;
    /** Impermeable cells (0/1); must be dry. Fixed at creation. */
    solid?: Numbers;
    /** Tracer concentration per cell (>= 0, default 0); requires tracer. */
    concentration?: Numbers;
    /** Concentration of the source water per cell (>= 0, default 0); rain carries none. */
    sourceConcentration?: Numbers;
}

export interface Counters { hlleFallbacks: number; reconstructionFallbacks: number; retries: number }
export interface AdvanceResult { steps: number; dts: Float64Array; time: number }

export interface Kernels {
    readonly device: GPUDevice;
    readonly scheme: Scheme;
    readonly sha256: string;
    readonly messages: ReadonlyArray<{ type: string; line: number; message: string }>;
}

export function compileKernels(device: GPUDevice, scheme?: Scheme): Promise<Kernels>;
export function createSolver(device: GPUDevice, options: Options, initial: Initial, kernels: Kernels): Promise<WebGpuSolver | FusedMusclSolver>;

declare abstract class GpuSolver {
    readonly config: Configuration;
    /** Accepted simulated time (s, f64 sum of accepted f32 timesteps). */
    readonly time: number;
    /** Datum subtracted from the bed (m). */
    readonly datum: number;
    /** Volume added by rain and sources (m³). */
    readonly sourceVolume: number;
    readonly counters: Counters;
    readonly allocatedBytes: number;
    readonly busy: boolean;
    readonly destroyed: boolean;
    /** Replace Manning n, source rates and/or source concentrations between steps. */
    setForcing(forcing: { manning?: Numbers; source?: Numbers; sourceConcentration?: Numbers }): void;
    /** Change wind, windDrag, coriolis or settling between steps (validated as at creation). */
    setParameters(changes: RuntimeParameters): void;
    /** Replace the hydrograph series between steps; segments keep their edge, cells and kind, with no more points in total than at creation. */
    setHydrographs(hydrographs: Hydrograph[]): void;
    /** Edit the bed of the given cells (m, datum of the initial bed); depths are kept. Solid cells and nonfinite values are rejected. */
    setBed(indices: Numbers, elevations: Numbers): void;
    /** Sparse source update: new rates (m/s, >= 0) and, in tracer runs, optional source concentrations for the listed cells; uploads only those cells. */
    setSources(indices: Numbers, rates: Numbers, concentrations?: Numbers): void;
    /** Replace the state [h, hu, hv, 0] per cell (validated as at creation); tracer runs also take the concentration per cell. */
    setState(state: Numbers, options?: { concentration?: Numbers }): Promise<void>;
    /** Volume changed by setState (m³), outside the source and boundary budgets. */
    readonly stateEditVolume: number;
    /** State buffer the solver reads next: vec4<f32> [h, hu, hv, compensation] per cell; may be edited between operations (docs/API.md). */
    readonly currentState: GPUBuffer;
    /** Integrate to time + duration (or maxSteps steps), each step capped by maxDt. */
    advance(duration: number, options?: { maxDt?: number; maxSteps?: number }): Promise<AdvanceResult>;
    /** [h, hu, hv, c] per cell; the fourth lane is GPU depth compensation, not physical state. */
    readState(): Promise<Float32Array>;
    /** Net volume (or, with tracer: true, tracer amount) through the boundary since creation (outward positive). */
    readBoundaryOutflow(options?: { tracer?: boolean }): Promise<number>;
    destroy(): void;
}

export class WebGpuSolver extends GpuSolver {
    static create(device: GPUDevice, options: Options, initial: Initial, kernels?: Kernels): Promise<WebGpuSolver>;
    step(maxDt: number, options?: { timestamps?: GPUComputePassTimestampWrites }): Promise<number>;
}

export class FusedMusclSolver extends GpuSolver {
    static create(device: GPUDevice, options: Options, initial: Initial, kernels?: Kernels): Promise<FusedMusclSolver>;
    step(maxDt: number): Promise<number>;
    advance(duration: number, options?: { maxDt?: number; maxSteps?: number; batch?: number; timestamps?: GPUComputePassTimestampWrites }): Promise<AdvanceResult>;
    /** Records up to `steps` steps into the caller's encoder; returns the buffer they will write. Call resolve() next. */
    encode(encoder: GPUCommandEncoder, options: { duration: number; maxDt?: number; steps?: number }): GPUBuffer;
    /** Accepts the encoded steps after submission, or rolls back and redoes them with the retry policy. */
    resolve(): Promise<(AdvanceResult & { recovered: boolean }) | null>;
    /** Tracer buffer matching currentState: vec2<f32> [h c, compensation] per cell. */
    readonly currentTracer: GPUBuffer;
    /** Conserved tracer amount h c per cell (m). */
    readTracer(): Promise<Float32Array>;
    /** Tracer amount added by sources (m³ times concentration). */
    readonly tracerSourceAmount: number;
}

export class ReferenceSolver {
    constructor(options: Options, initial: Initial);
    readonly config: Configuration;
    readonly state: Float64Array;
    readonly bed: Float64Array;
    readonly time: number;
    readonly boundaryOutflow: number;
    readonly sourceVolume: number;
    step(maxDt: number): number;
    setForcing(forcing: { manning?: Numbers; source?: Numbers; sourceConcentration?: Numbers }): void;
    setParameters(changes: RuntimeParameters): void;
    setHydrographs(hydrographs: Hydrograph[]): void;
    setSources(indices: Numbers, rates: Numbers, concentrations?: Numbers): void;
    setState(state: Numbers, options?: { concentration?: Numbers }): void;
    readonly stateEditVolume: number;
    readState(): Float64Array;
    /** Tracer runs: conserved amount h c per cell, boundary outflow and source amount. */
    readTracer(): Float64Array;
    readonly tracerOutflow: number;
    readonly tracerSourceAmount: number;
}
export class MusclReferenceSolver extends ReferenceSolver {
    readonly counters: Counters;
}

export function configuration(options: Options): Configuration;
export function diagnostics(state: Numbers, config: { dx: number; dy: number }): { volume: number; minDepth: number; maxDepth: number; maxSpeed: number };
export function hydrographValue(times: Numbers, values: Numbers, t: number): number;
