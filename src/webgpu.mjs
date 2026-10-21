import { configuration, windStress, updatedConfiguration, initialData, forcingData, stateData, sourceUpdates, packCells, sourceDischarge,
    maxSource, sourceLimitFor, validateMaxDt, boundaryCode,
    hydrographSegments, hydrographTable, hydrographTimestepLimit, clockAdvance,
    tracerBounds, diffusionTimestepLimit, tracerSourceRate } from './model.mjs';

// rusanov / muscl-hancock: auditable multi-pass paths with a scalar readback per step.
// muscl-fused: one tiled dispatch per step, GPU-resident timestep, batched submission.
export const SCHEMES = Object.freeze(['rusanov', 'muscl-hancock', 'muscl-fused']);
const LAYOUTS = {
    'rusanov': { bindings: 8, stages: ['waveSpeeds', 'timestep', 'faceFluxes', 'update'] },
    'muscl-hancock': { bindings: 9, stages: ['waveSpeeds', 'timestep', 'reconstruct', 'faceFluxes', 'update'] },
    'muscl-fused': { bindings: 9, stages: ['waveSpeeds', 'timestep', 'fusedStep'] },
};
const MAX_BATCH = 256; // Must match MAX_LOG in muscl-fused.wgsl.
const LOG_OFFSET = 16;

// Shader source next to this module: fetched over HTTP(S), read from disk when the module is loaded
// from a file in Node (whose fetch does not read file: URLs).
async function readShader(url) {
    if (url.protocol === 'file:' && typeof process !== 'undefined' && process.versions?.node) {
        const { readFile } = await import('node:fs/promises');
        return readFile(url, 'utf8');
    }
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Shader fetch failed: HTTP ${response.status}`);
    return response.text();
}

export async function compileKernels(device, scheme = 'rusanov') {
    if (!SCHEMES.includes(scheme)) throw new RangeError('Unknown numerical scheme');
    const code = await readShader(new URL(`./shaders/${scheme}.wgsl`, import.meta.url));
    const module = device.createShaderModule({ label: scheme, code });
    const info = await module.getCompilationInfo();
    const messages = Array.from(info.messages, m => ({ type: m.type, line: m.lineNum, message: m.message }));
    if (messages.some(m => m.type === 'error')) throw new Error(JSON.stringify(messages));
    const { bindings, stages } = LAYOUTS[scheme];
    const layout = device.createBindGroupLayout({ entries: Array.from({ length: bindings }, (_, binding) => ({
        binding, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: binding === 0 ? 'uniform' : binding <= 2 ? 'read-only-storage' : 'storage' },
    })) });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipelines = {};
    for (const entryPoint of stages) {
        pipelines[entryPoint] = await device.createComputePipelineAsync({
            label: entryPoint, layout: pipelineLayout, compute: { module, entryPoint },
        });
    }
    const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(code));
    return { device, scheme, stages, layout, pipelines, messages,
        sha256: Array.from(new Uint8Array(hash), b => b.toString(16).padStart(2, '0')).join('') };
}

export async function createSolver(device, options, initial, kernels) {
    const scheme = kernels?.scheme ?? options.scheme ?? 'rusanov';
    return (scheme === 'muscl-fused' ? FusedMusclSolver : WebGpuSolver).create(device, options, initial, kernels);
}

// Largest f32 not above x, so float conversion never exceeds a caller's time cap.
function f32Below(x) {
    const f = new Float32Array([x]), u = new Uint32Array(f.buffer);
    if (f[0] > x) u[0]--;
    return f[0];
}

function f32Bits(x) { return new Uint32Array(new Float32Array([x]).buffer)[0]; }

// Shared buffer ownership, forcing, readback and lifecycle.
class GpuSolverBase {
    async init(device, options, initial, compiled) {
        if (compiled.device !== device) throw new Error('Kernels belong to a different GPU device');
        if (options.scheme && options.scheme !== compiled.scheme) throw new Error('Kernel scheme does not match options.scheme');
        if (options.tracer && compiled.scheme !== 'muscl-fused') throw new RangeError('Scalar transport (tracer) requires the muscl-fused kernels');
        if (options.lowFroude && compiled.scheme === 'rusanov') throw new RangeError('lowFroude applies to the MUSCL-Hancock/HLLC kernels');
        if (compiled.scheme !== 'muscl-fused' && (options.eddyViscosity > 0 || options.smagorinsky > 0 || options.bedTurbulence > 0)) {
            throw new RangeError('The turbulence closure requires the muscl-fused kernels');
        }
        this.device = device;
        this.kernels = compiled;
        this.config = configuration(options);
        this.buffers = [];
        this.time = 0;
        this.clock = [0, 0]; // hydrograph clock (f32 hi, lo), advanced exactly as in the kernels
        this.ping = 0;
        this.busy = false;
        this.destroyed = false;
        this.sourceVolume = 0;
        this.stateEditVolume = 0; // volume changed by setState (m³), outside the source and boundary budgets
        this.counters = { hlleFallbacks: 0, reconstructionFallbacks: 0, retries: 0 };
        const data = initialData(this.config, initial, Float32Array);
        const p = this.config;
        this.bedValues = data.bed;
        this.datum = data.datum; // bed on the GPU is z - datum (m)
        this.forcing = { manning: data.manning, source: data.source, sourceConcentration: data.sourceConcentration, solid: data.solid };
        this.discharge = sourceDischarge(p, this.forcing);
        this.maxSource = maxSource(this.forcing);
        this.sourceLimit = sourceLimitFor(p, this.maxSource);
        this.diffusionLimit = diffusionTimestepLimit(p);
        if (p.tracer) {
            this.bounds = tracerBounds(p, data);
            this.tracerSourceAmount = 0;
            this.tracerRate = tracerSourceRate(p, this.forcing);
        }
        for (const key of ['dx', 'dy', 'gravity', 'cfl', 'rain', 'diffusivity']) {
            const value = Math.fround(p[key]);
            if (!Number.isFinite(value) || (p[key] > 0 && value <= 0)) throw new RangeError(`${key} is not representable as positive f32`);
        }
        this.boundarySlots = 2 * (p.width + p.height);
        // Hydrograph table, stored after the outflow accumulators in the boundary buffer.
        this.hydrographs = hydrographSegments(p, data.datum, data.solid, data.bed);
        this.hydroLimit = hydrographTimestepLimit(p, this.hydrographs);
        this.hydrographTable = hydrographTable(this.hydrographs);
        return data;
    }

    make(label, size, usage) {
        const limits = this.device.limits;
        if (size > limits.maxBufferSize || ((usage & GPUBufferUsage.STORAGE) && size > limits.maxStorageBufferBindingSize)) {
            throw new RangeError(`${label} exceeds adapter buffer limits`);
        }
        const buffer = this.device.createBuffer({ label, size, usage });
        this.buffers.push(buffer);
        this.allocatedBytes = (this.allocatedBytes ?? 0) + size;
        return buffer;
    }

    async allocate(build) {
        this.device.pushErrorScope('validation');
        try { build(); }
        catch (error) { this.destroy(); throw error; }
        finally {
            const error = await this.device.popErrorScope();
            if (error && !this.destroyed) { this.destroy(); throw new Error(error.message); }
        }
    }

    commonBuffers(state) {
        const p = this.config, n = p.width * p.height;
        const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
        this.states = [this.make('State A', n * 16, storage), this.make('State B', n * 16, storage)];
        this.cells = this.make('Cells [z,n,source,solid]', n * 16, storage);
        // Water and tracer outflow accumulators (2 x boundarySlots vec2), then the hydrograph table.
        const accumulators = 2 * this.boundarySlots * 8;
        this.boundary = this.make('Boundary outflow and hydrographs', accumulators + this.hydrographTable.byteLength, storage);
        this.boundaryCheckpoint = this.make('Outflow checkpoint', accumulators, storage);
        this.boundaryReadback = this.make('Outflow readback', accumulators, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        this.stateReadback = this.make('State readback', n * 16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        this.device.queue.writeBuffer(this.states[0], 0, state);
        this.device.queue.writeBuffer(this.cells, 0, packCells(this.bedValues, this.forcing));
        this.device.queue.writeBuffer(this.boundary, accumulators, this.hydrographTable);
    }

    // parity: fused runner; otherwise words 10 and 11 carry the hydrograph clock (lo, hi). Words 12-15:
    // tracer flag, diffusivity and the concentration range (fused runner only).
    uniform(maxDt, parity = null) {
        const p = this.config;
        const uniform = new ArrayBuffer(128), u = new Uint32Array(uniform), f = new Float32Array(uniform);
        u.set([p.width, p.height, boundaryCode(p.boundaryX), boundaryCode(p.boundaryY)]);
        f.set([p.dx, p.dy, p.gravity, p.cfl, f32Below(Math.min(maxDt, this.sourceLimit, this.hydroLimit, this.diffusionLimit)), p.rain], 4);
        if (parity === null) { f[10] = this.clock[1]; f[11] = this.clock[0]; } else u[10] = parity;
        if (p.tracer) { u[12] = 1; f.set([p.diffusivity, this.bounds[0], this.bounds[1]], 13); }
        // Verification only: options.faultInjection = { dt, time } (see muscl-fused.wgsl).
        if (p.faultInjection) f.set([p.faultInjection.dt, p.faultInjection.time ?? 0], 16);
        if (p.lowFroude) u[18] = 1;
        f.set([p.eddyViscosity, p.smagorinsky, p.bedTurbulence, 1 / p.schmidt], 20); // turbulence closure (fused)
        f.set(windStress(p), 24); // kinematic wind stress (all kernels)
        if (p.settling) f.set([p.settling.velocity, p.settling.exponent], 26); // tracer settling (fused)
        f[28] = p.coriolis; // Coriolis parameter (all kernels)
        if (!(f[8] > 0) || !Number.isFinite(f[8])) throw new RangeError('maxDt is outside the supported f32 range');
        return uniform;
    }

    assertAvailable() {
        if (this.destroyed) throw new Error('Solver has been destroyed');
        if (this.busy) throw new Error('Await the previous solver operation before starting another');
    }

    async exclusive(operation) {
        this.assertAvailable();
        this.busy = true;
        try { return await operation(); } finally { this.busy = false; }
    }

    // Change wind, wind drag, Coriolis or settling between steps; they enter the next step's uniform
    // (model.mjs: updatedConfiguration). Turning settling on in a tracer run widens the concentration
    // range to 0, as at creation.
    setParameters(changes) {
        this.assertAvailable();
        if ('settling' in changes && changes.settling && this.kernels.scheme !== 'muscl-fused') throw new RangeError('settling requires the muscl-fused kernels');
        this.config = updatedConfiguration(this.config, changes);
        if (this.config.tracer && this.config.settling) this.bounds = [Math.min(0, this.bounds[0]), this.bounds[1]];
    }

    // Replace the hydrograph series between steps: the same segments (edge, cells, kind), new times and
    // values on the solver clock. The table must fit the space allocated at creation.
    setHydrographs(hydrographs) {
        this.assertAvailable();
        const config = updatedConfiguration(this.config, { hydrographs });
        const data = hydrographSegments(config, this.datum, this.forcing.solid, this.bedValues);
        const table = hydrographTable(data);
        if (table.byteLength > this.hydrographTable.byteLength) {
            throw new RangeError(`hydrograph table grows from ${this.hydrographTable.byteLength} to ${table.byteLength} bytes; use at most as many points as at creation`);
        }
        this.config = config; this.hydrographs = data; this.hydroLimit = hydrographTimestepLimit(config, data);
        if (config.tracer) for (const s of config.hydrographs) this.bounds = [Math.min(this.bounds[0], s.concentration), Math.max(this.bounds[1], s.concentration)];
        this.device.queue.writeBuffer(this.boundary, 2 * this.boundarySlots * 8, table);
    }

    // Replace Manning n and/or per-cell source rates (m/s) between steps.
    setForcing(forcing) {
        this.assertAvailable();
        const current = { manning: forcing.manning ?? this.forcing.manning, source: forcing.source ?? this.forcing.source,
            ...(this.config.tracer ? { sourceConcentration: forcing.sourceConcentration ?? this.forcing.sourceConcentration } : {}) };
        this.forcing = forcingData(this.config, current, Float32Array, this.forcing.solid);
        this.discharge = sourceDischarge(this.config, this.forcing);
        this.maxSource = maxSource(this.forcing);
        this.sourceLimit = sourceLimitFor(this.config, this.maxSource);
        this.device.queue.writeBuffer(this.cells, 0, packCells(this.bedValues, this.forcing));
        if (this.config.tracer) {
            // Source water may bring new concentrations: widen the range (never narrow it).
            const { source, sourceConcentration, solid } = this.forcing;
            for (let i = 0; i < solid.length; i++) if (!solid[i] && source[i] > 0) {
                this.bounds = [Math.min(this.bounds[0], sourceConcentration[i]), Math.max(this.bounds[1], sourceConcentration[i])];
            }
            this.tracerRate = tracerSourceRate(this.config, this.forcing);
            this.device.queue.writeBuffer(this.tracerSource, 0, this.forcing.sourceConcentration);
        }
    }

    // Sparse source update between steps: new rates (m/s) and, in tracer runs, optionally new source
    // concentrations for the listed cells (model.mjs: sourceUpdates). Only the records of those cells
    // are uploaded, and the inflow rate and the source timestep bound are updated from the changes,
    // so frequent small updates (moving storms, point inflows) cost O(changed cells).
    setSources(indices, rates, concentrations = undefined) {
        this.assertAvailable();
        const p = this.config, area = p.dx * p.dy;
        sourceUpdates(p, indices, rates, concentrations);
        const { source, sourceConcentration, solid } = this.forcing;
        let rescan = false;
        for (let k = 0; k < indices.length; k++) {
            const i = indices[k], old = source[i], next = Math.fround(rates[k]);
            const oldC = sourceConcentration[i], nextC = concentrations === undefined ? oldC : Math.fround(concentrations[k]);
            if (!solid[i]) {
                this.discharge += (next - old) * area;
                if (p.tracer) this.tracerRate += (next * nextC - old * oldC) * area;
                if (next < old && old === this.maxSource) rescan = true;
                else if (next > this.maxSource) this.maxSource = next;
                if (p.tracer && next > 0) this.bounds = [Math.min(this.bounds[0], nextC), Math.max(this.bounds[1], nextC)];
            }
            source[i] = next; sourceConcentration[i] = nextC;
        }
        if (rescan) this.maxSource = maxSource(this.forcing);
        this.sourceLimit = sourceLimitFor(p, this.maxSource);
        // Upload the touched cells in contiguous runs.
        const cells = [...new Set(indices)].sort((a, b) => a - b);
        for (let start = 0; start < cells.length;) {
            let end = start + 1;
            while (end < cells.length && cells[end] === cells[end - 1] + 1) end++;
            const first = cells[start], count = cells[end - 1] - first + 1, records = new Float32Array(4 * count);
            for (let j = 0; j < count; j++) {
                const i = first + j;
                records.set([this.bedValues[i], this.forcing.manning[i], source[i], solid[i]], 4 * j);
            }
            this.device.queue.writeBuffer(this.cells, 16 * first, records);
            if (p.tracer) this.device.queue.writeBuffer(this.tracerSource, 4 * first, sourceConcentration.subarray(first, first + count));
            start = end;
        }
    }

    // Replace the state [h, hu, hv, 0] per cell between steps (model.mjs: stateData), with the
    // concentration per cell in tracer runs (hotstarts, restored scenarios, batch runs). The previous
    // state is read back so the volume change (m³) can be kept in stateEditVolume.
    setState(state, { concentration } = {}) {
        this.assertAvailable();
        const p = this.config;
        if (p.tracer && concentration === undefined) throw new RangeError('setState needs concentration in tracer runs');
        const next = new Float32Array(state);
        const { tracer } = stateData(p, next, this.forcing.solid, concentration);
        return this.exclusive(async () => {
            const old = new Float32Array(await this.readBuffer(this.states[this.ping], this.stateReadback, this.stateReadback.size));
            let change = 0;
            for (let k = 0; k < next.length; k += 4) change += next[k] - old[k];
            this.stateEditVolume += change * p.dx * p.dy;
            this.device.queue.writeBuffer(this.states[this.ping], 0, next);
            if (p.tracer) {
                const packed = new Float32Array(2 * tracer.length);
                for (let i = 0; i < tracer.length; i++) {
                    packed[2 * i] = tracer[i];
                    if (next[4 * i] > 0) this.bounds = [Math.min(this.bounds[0], concentration[i]), Math.max(this.bounds[1], concentration[i])];
                }
                this.device.queue.writeBuffer(this.tracers[this.ping], 0, packed);
            }
        });
    }

    // State buffer the solver reads next (vec4<f32> [h, hu, hv, compensation] per cell). Between
    // operations a caller may also edit it on the GPU (see docs/API.md, "External state edits").
    get currentState() { return this.states[this.ping]; }

    // Edit the bed (interactive terrain): `indices` are cells, `elevations` the new bed (m, in the
    // datum of the initial bed). Depths are kept, so the water surface moves with the bed. Solid
    // cells and nonfinite values are rejected; bounds derived at creation (hydrograph ghost speeds)
    // are not recomputed.
    setBed(indices, elevations) {
        this.assertAvailable();
        if (indices.length !== elevations.length) throw new RangeError('setBed needs one elevation per index');
        const n = this.config.width * this.config.height, record = new Float32Array(4);
        for (let k = 0; k < indices.length; k++) {
            const i = indices[k];
            if (!Number.isInteger(i) || i < 0 || i >= n) throw new RangeError(`setBed: invalid cell ${i}`);
            if (!Number.isFinite(elevations[k])) throw new RangeError(`setBed: nonfinite elevation at cell ${i}`);
            if (this.forcing.solid[i]) throw new RangeError(`setBed: cell ${i} is solid`);
        }
        for (let k = 0; k < indices.length; k++) {
            const i = indices[k];
            this.bedValues[i] = elevations[k] - this.datum;
            record.set([this.bedValues[i], this.forcing.manning[i], this.forcing.source[i], 0]);
            this.device.queue.writeBuffer(this.cells, 16 * i, record);
        }
    }

    async readBuffer(source, readback, size) {
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(source, 0, readback, 0, size);
        this.device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ, 0, size);
        const copy = readback.getMappedRange(0, size).slice(0);
        readback.unmap();
        return copy;
    }

    readState() {
        return this.exclusive(async () => new Float32Array(
            await this.readBuffer(this.states[this.ping], this.stateReadback, this.stateReadback.size)));
    }

    // Cumulative boundary outflow (positive outward) summed in f64: water volume (m³), or with
    // tracer = true the tracer amount (m³ times concentration).
    readBoundaryOutflow({ tracer = false } = {}) {
        return this.exclusive(async () => {
            const values = new Float32Array(await this.readBuffer(this.boundary, this.boundaryReadback, this.boundaryReadback.size));
            let total = 0;
            const start = tracer ? 2 * this.boundarySlots : 0;
            for (let k = start; k < start + 2 * this.boundarySlots; k += 2) total += values[k] - values[k + 1];
            return total;
        });
    }

    checkpoint(encoder) { encoder.copyBufferToBuffer(this.boundary, 0, this.boundaryCheckpoint, 0, this.boundaryCheckpoint.size); }
    restore(encoder) { encoder.copyBufferToBuffer(this.boundaryCheckpoint, 0, this.boundary, 0, this.boundaryCheckpoint.size); }

    accept(dt) {
        this.time += dt;
        this.clock = clockAdvance(this.clock, dt);
        this.sourceVolume += this.discharge * dt;
        if (this.config.tracer) this.tracerSourceAmount += this.tracerRate * dt;
    }

    // Integrate for `duration` seconds (or until maxSteps) with per-step cap maxDt.
    async advance(duration, { maxDt = duration, maxSteps = Infinity } = {}) {
        validateMaxDt(duration); validateMaxDt(maxDt);
        const target = this.time + duration, dts = [];
        while (dts.length < maxSteps && target - this.time > 1e-12 * Math.max(1, Math.abs(target))) {
            dts.push(await this.step(Math.min(maxDt, target - this.time)));
        }
        return { steps: dts.length, dts: Float64Array.from(dts), time: this.time };
    }

    destroy() {
        if (this.busy) throw new Error('Cannot destroy a solver during an operation');
        for (const buffer of this.buffers) buffer.destroy();
        this.destroyed = true;
    }
}

// Multi-pass runner (Rusanov or MUSCL reference kernels). One readback per step.
export class WebGpuSolver extends GpuSolverBase {
    static async create(device, options, initial, kernels) {
        const compiled = kernels ?? await compileKernels(device, options.scheme ?? 'rusanov');
        if (compiled.scheme === 'muscl-fused') throw new Error('Use FusedMusclSolver for muscl-fused kernels');
        const solver = new WebGpuSolver();
        const { state } = await solver.init(device, options, initial, compiled);
        const p = solver.config, n = p.width * p.height;
        const nx = (p.width + 1) * p.height, ny = p.width * (p.height + 1);
        solver.cellGroups = Math.ceil(n / 64);
        solver.faceGroups = Math.ceil(Math.max(nx, ny) / 64);
        if (solver.faceGroups > device.limits.maxComputeWorkgroupsPerDimension) {
            throw new RangeError('Grid exceeds this runner\'s one-dimensional dispatch limit');
        }
        await solver.allocate(() => {
            const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
            solver.params = solver.make('Numerics parameters', 128, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
            solver.commonBuffers(state);
            solver.fx = solver.make('X face fluxes', nx * 32, storage);
            solver.fy = solver.make('Y face fluxes', ny * 32, storage);
            solver.controlBytes = compiled.scheme === 'rusanov' ? 16 : 32;
            solver.control = solver.make('Timestep and errors', solver.controlBytes, storage);
            solver.controlReadback = solver.make('Scalar readback', solver.controlBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
            if (compiled.scheme !== 'rusanov') solver.traces = solver.make('Predicted W/E/S/N traces', n * 64, storage);
            solver.groups = [0, 1].map(ping => device.createBindGroup({ layout: compiled.layout,
                entries: [solver.params, solver.states[ping], solver.cells, solver.fx, solver.fy,
                    solver.states[1 - ping], solver.control, ...(solver.traces ? [solver.traces] : []), solver.boundary]
                    .map((buffer, binding) => ({ binding, resource: { buffer } })),
            }));
        });
        return solver;
    }

    // timestamps: optional { querySet, beginningOfPassWriteIndex, endOfPassWriteIndex } written at
    // the start of the first pass and the end of the last pass of the step (GPU kernel time).
    step(maxDt, { timestamps } = {}) {
        validateMaxDt(maxDt);
        return this.exclusive(async () => {
            let cap = maxDt;
            for (let retry = 0; retry <= 8; retry++) {
                try { return await this._stepOnce(cap, timestamps); }
                catch (error) {
                    if (this.kernels.scheme === 'rusanov' || !error.retryable || retry === 8) throw error;
                    cap = error.attemptDt / 2;
                    this.counters.retries++;
                }
            }
        });
    }

    async _stepOnce(maxDt, timestamps) {
        const uniform = this.uniform(maxDt);
        this.device.pushErrorScope('validation');
        let scopeOpen = true;
        try {
            this.device.queue.writeBuffer(this.params, 0, uniform);
            const encoder = this.device.createCommandEncoder();
            encoder.clearBuffer(this.control);
            this.checkpoint(encoder);
            const stages = this.kernels.stages;
            for (const [index, name] of stages.entries()) {
                const groups = name === 'timestep' ? 1 : name === 'faceFluxes' ? this.faceGroups : this.cellGroups;
                const timestampWrites = !timestamps ? undefined : index === 0 && index === stages.length - 1 ? timestamps
                    : index === 0 ? { querySet: timestamps.querySet, beginningOfPassWriteIndex: timestamps.beginningOfPassWriteIndex }
                    : index === stages.length - 1 ? { querySet: timestamps.querySet, endOfPassWriteIndex: timestamps.endOfPassWriteIndex }
                    : undefined;
                const pass = encoder.beginComputePass({ label: name, timestampWrites });
                pass.setPipeline(this.kernels.pipelines[name]);
                pass.setBindGroup(0, this.groups[this.ping]);
                pass.dispatchWorkgroups(groups);
                pass.end();
            }
            encoder.copyBufferToBuffer(this.control, 0, this.controlReadback, 0, this.controlBytes);
            this.device.queue.submit([encoder.finish()]);
            await this.controlReadback.mapAsync(GPUMapMode.READ);
            const scalars = this.controlReadback.getMappedRange().slice(0);
            this.controlReadback.unmap();
            const dt = new Float32Array(scalars)[0], errorBits = new Uint32Array(scalars)[3];
            if (errorBits || !(dt > 0) || !Number.isFinite(dt)) {
                const restore = this.device.createCommandEncoder();
                this.restore(restore);
                this.device.queue.submit([restore.finish()]);
                const error = new Error(`GPU rejected step (error bits ${errorBits}); state/time unchanged`);
                error.retryable = (errorBits & ~5) === 0 && dt > 0 && Number.isFinite(dt); // invalid state only
                error.attemptDt = dt;
                throw error;
            }
            const validation = await this.device.popErrorScope();
            scopeOpen = false;
            if (validation) throw new Error(validation.message);
            this.ping = 1 - this.ping;
            this.accept(dt);
            if (this.traces) {
                this.counters.hlleFallbacks += new Uint32Array(scalars)[4];
                this.counters.reconstructionFallbacks += new Uint32Array(scalars)[5];
            }
            return dt;
        } finally {
            if (scopeOpen) await this.device.popErrorScope();
        }
    }
}

// Fused MUSCL-Hancock runner. advance() records up to `batch` steps per submission;
// the GPU selects each dt and stops at the target time. A failed batch is rolled back
// to its checkpoint and replayed step by step with the retry policy.
export class FusedMusclSolver extends GpuSolverBase {
    static async create(device, options, initial, kernels) {
        const compiled = kernels ?? await compileKernels(device, 'muscl-fused');
        if (compiled.scheme !== 'muscl-fused') throw new Error('FusedMusclSolver requires muscl-fused kernels');
        if (device.limits.maxComputeWorkgroupStorageSize < 16384 || device.limits.maxComputeInvocationsPerWorkgroup < 256) {
            throw new RangeError('Adapter workgroup limits are below the fused kernel requirements');
        }
        const solver = new FusedMusclSolver();
        const { state, tracer } = await solver.init(device, options, initial, compiled);
        const p = solver.config, n = p.width * p.height;
        const limit = device.limits.maxComputeWorkgroupsPerDimension;
        solver.tileGroups = [Math.ceil(p.width / 16), Math.ceil(p.height / 16)];
        const speedGroups = Math.ceil(n / 256);
        solver.speedGroups = [Math.min(speedGroups, limit), Math.ceil(speedGroups / limit)];
        if (Math.max(...solver.tileGroups, solver.speedGroups[1]) > limit) throw new RangeError('Grid exceeds dispatch limits');
        solver.lastDt = 0;
        await solver.allocate(() => {
            const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
            solver.params = [0, 1].map(k => solver.make(`Parameters (parity ${k})`, 128, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST));
            solver.commonBuffers(state);
            solver.stateCheckpoint = solver.make('State checkpoint', n * 16, storage);
            // Tracer [h c, compensation] ping-pong pair, checkpoint and source concentrations; minimal
            // placeholders when the tracer is off (the kernel never touches them).
            const tracerBytes = p.tracer ? n * 8 : 16;
            solver.tracers = [0, 1].map(k => solver.make(`Tracer ${'AB'[k]}`, tracerBytes, storage));
            solver.tracerCheckpoint = solver.make('Tracer checkpoint', tracerBytes, storage);
            solver.tracerSource = solver.make('Source concentration', p.tracer ? Math.max(16, n * 4) : 16, storage);
            if (p.tracer) {
                const packed = new Float32Array(2 * n);
                for (let i = 0; i < n; i++) packed[2 * i] = tracer[i];
                device.queue.writeBuffer(solver.tracers[0], 0, packed);
                device.queue.writeBuffer(solver.tracerSource, 0, solver.forcing.sourceConcentration);
                solver.tracerReadback = solver.make('Tracer readback', n * 8, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
            }
            solver.controlBytes = (LOG_OFFSET + MAX_BATCH) * 4;
            solver.control = solver.make('Timestep control and dt log', solver.controlBytes, storage);
            solver.controlReadback = solver.make('Control readback', solver.controlBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
            solver.groups = [0, 1].map(ping => device.createBindGroup({ layout: compiled.layout,
                entries: [solver.params[ping], solver.states[ping], solver.cells, solver.states[1 - ping], solver.control, solver.boundary,
                    solver.tracers[ping], solver.tracers[1 - ping], solver.tracerSource]
                    .map((buffer, binding) => ({ binding, resource: { buffer } })),
            }));
        });
        return solver;
    }

    // Records `count` steps (count <= MAX_BATCH) into `encoder`: checkpoint, one wave-speed pass,
    // count x (timestep, fusedStep), and a copy of the control words to the readback buffer.
    _record(encoder, remaining, maxDt, count, timestamps) {
        const start = this.ping, queue = this.device.queue;
        for (const parity of [0, 1]) queue.writeBuffer(this.params[parity], 0, this.uniform(maxDt, parity));
        const header = new Uint32Array(LOG_OFFSET);
        header[9] = f32Bits(f32Below(remaining));
        header[10] = 0xffffffff; // index of the first failing step
        header[13] = f32Bits(this.clock[0]); header[14] = f32Bits(this.clock[1]); // hydrograph clock
        queue.writeBuffer(this.control, 0, header);
        encoder.copyBufferToBuffer(this.states[start], 0, this.stateCheckpoint, 0, this.stateCheckpoint.size);
        if (this.config.tracer) encoder.copyBufferToBuffer(this.tracers[start], 0, this.tracerCheckpoint, 0, this.tracerCheckpoint.size);
        this.checkpoint(encoder);
        const pass = encoder.beginComputePass({ label: `fused batch x${count}`, timestampWrites: timestamps });
        const { waveSpeeds, timestep, fusedStep } = this.kernels.pipelines;
        pass.setPipeline(waveSpeeds);
        pass.setBindGroup(0, this.groups[start]);
        pass.dispatchWorkgroups(...this.speedGroups);
        let ping = start;
        for (let k = 0; k < count; k++) {
            pass.setBindGroup(0, this.groups[ping]);
            pass.setPipeline(timestep);
            pass.dispatchWorkgroups(1);
            pass.setPipeline(fusedStep);
            pass.dispatchWorkgroups(...this.tileGroups);
            ping = 1 - ping;
        }
        pass.end();
        const bytes = (LOG_OFFSET + count) * 4;
        encoder.copyBufferToBuffer(this.control, 0, this.controlReadback, 0, bytes);
        return { start, end: ping, count, bytes };
    }

    // Reads the control words of a submitted record. Accepts its dts, or restores the checkpoint.
    async _finish({ start, end, count, bytes }) {
        await this.controlReadback.mapAsync(GPUMapMode.READ, 0, bytes);
        const words = new Uint32Array(this.controlReadback.getMappedRange(0, bytes).slice(0));
        this.controlReadback.unmap();
        const dts = new Float32Array(words.buffer, LOG_OFFSET * 4, count);
        const errorBits = words[5];
        if (errorBits) {
            const restore = this.device.createCommandEncoder();
            restore.copyBufferToBuffer(this.stateCheckpoint, 0, this.states[start], 0, this.stateCheckpoint.size);
            if (this.config.tracer) restore.copyBufferToBuffer(this.tracerCheckpoint, 0, this.tracers[start], 0, this.tracerCheckpoint.size);
            this.restore(restore);
            this.device.queue.submit([restore.finish()]);
            this.ping = start;
            const firstBad = words[10] < count ? words[10] : 0;
            return { ok: false, errorBits, firstBad, firstDt: dts[firstBad] };
        }
        this.ping = end;
        const accepted = [];
        for (const dt of dts) if (dt > 0) { this.accept(dt); accepted.push(dt); }
        this.counters.hlleFallbacks += words[6];
        this.counters.reconstructionFallbacks += words[7];
        if (accepted.length) this.lastDt = accepted.at(-1);
        return { ok: true, dts: accepted, finished: words[9] === 0 };
    }

    // One submitted batch. Returns accepted dts or a rejection.
    async _batch(remaining, maxDt, count, { timestamps } = {}) {
        this.device.pushErrorScope('validation');
        let scopeOpen = true;
        try {
            const encoder = this.device.createCommandEncoder();
            const record = this._record(encoder, remaining, maxDt, count, timestamps);
            this.device.queue.submit([encoder.finish()]);
            const result = await this._finish(record);
            const validation = await this.device.popErrorScope();
            scopeOpen = false;
            if (validation) throw new Error(validation.message);
            return result;
        } finally {
            if (scopeOpen) await this.device.popErrorScope();
        }
    }

    // Non-blocking stepping for render loops (for example an interactive viewer): records up to `steps` steps
    // toward time + duration into the caller's command encoder, which the caller submits with its
    // own work. `currentState` already names the buffer the steps will write, so later passes in
    // the same encoder can read it. Call resolve() after the submission and before any other
    // solver operation; it accepts the steps, or rolls back and redoes them with the retry policy.
    encode(encoder, { duration, maxDt = duration, steps = 64 } = {}) {
        validateMaxDt(duration); validateMaxDt(maxDt);
        if (!Number.isSafeInteger(steps) || steps < 1 || steps > MAX_BATCH) throw new RangeError(`steps must be in [1, ${MAX_BATCH}]`);
        this.assertAvailable();
        this.busy = true;
        this.pending = { ...this._record(encoder, duration, maxDt, steps), duration, maxDt };
        this.ping = this.pending.end;
        return this.states[this.ping];
    }

    async resolve() {
        const pending = this.pending;
        if (!pending) return null;
        this.pending = null;
        let result;
        try {
            this.ping = pending.start;
            result = await this._finish(pending);
        } finally { this.busy = false; }
        if (result.ok) return { steps: result.dts.length, dts: Float64Array.from(result.dts), time: this.time, recovered: false };
        // Deterministic replay of the same interval through the batched retry path.
        const replay = await this.advance(pending.duration, { maxDt: pending.maxDt, batch: pending.count, maxSteps: pending.count });
        return { ...replay, recovered: true };
    }

    // Buffer of [h c, compensation] pairs that the recorded steps write (tracer runs).
    get currentTracer() { return this.tracers[this.ping]; }

    // Conserved tracer amount h c per cell (m), with the compensation folded in.
    readTracer() {
        if (!this.config.tracer) throw new Error('The solver was created without tracer: true');
        return this.exclusive(async () => {
            const pairs = new Float32Array(await this.readBuffer(this.tracers[this.ping], this.tracerReadback, this.tracerReadback.size));
            const amount = new Float32Array(pairs.length / 2);
            for (let i = 0; i < amount.length; i++) amount[i] = pairs[2 * i] - pairs[2 * i + 1];
            return amount;
        });
    }

    async _stepWithRetry(maxDt) {
        let cap = maxDt;
        for (let retry = 0; retry <= 8; retry++) {
            const result = await this._batch(cap, cap, 1);
            if (result.ok) {
                if (!result.dts.length) throw new Error('GPU accepted a zero timestep');
                return result.dts[0];
            }
            const retryable = (result.errorBits & ~5) === 0 && result.firstDt > 0 && Number.isFinite(result.firstDt);
            if (!retryable || retry === 8) {
                throw new Error(`GPU rejected step (error bits ${result.errorBits}); state/time unchanged`);
            }
            cap = result.firstDt / 2;
            this.counters.retries++;
        }
    }

    // Single accepted step, identical semantics to WebGpuSolver.step.
    step(maxDt) {
        validateMaxDt(maxDt);
        return this.exclusive(() => this._stepWithRetry(maxDt));
    }

    advance(duration, { maxDt = duration, batch = 64, maxSteps = Infinity, timestamps } = {}) {
        validateMaxDt(duration); validateMaxDt(maxDt);
        if (!Number.isSafeInteger(batch) || batch < 1 || batch > MAX_BATCH) throw new RangeError(`batch must be in [1, ${MAX_BATCH}]`);
        return this.exclusive(async () => {
            const target = this.time + duration, dts = [];
            let size = Math.min(batch, 16);
            while (dts.length < maxSteps && target - this.time > 1e-12 * Math.max(1, Math.abs(target))) {
                const remaining = target - this.time;
                if (this.lastDt > 0) size = Math.min(batch, Math.max(1, Math.ceil(remaining / Math.min(maxDt, this.lastDt) * 1.05) + 1));
                size = Math.min(size, maxSteps - dts.length);
                const result = await this._batch(remaining, maxDt, size, { timestamps });
                if (result.ok) {
                    dts.push(...result.dts);
                    // f32 remaining-time rounding can leave the host clock a few ulps short; top up.
                    if (result.finished && Math.abs(target - this.time) <= 1e-7 * Math.max(1, Math.abs(target))) break;
                    continue;
                }
                // Steps are deterministic: replay the valid prefix as one batch, then apply the
                // single-step retry policy to the failing step only.
                if (result.firstBad > 0) {
                    const prefix = await this._batch(remaining, maxDt, result.firstBad, { timestamps });
                    if (!prefix.ok) throw new Error('Replay of a previously valid batch prefix failed');
                    dts.push(...prefix.dts);
                    if (prefix.finished) continue;
                }
                if (dts.length < maxSteps && target - this.time > 1e-12 * Math.max(1, Math.abs(target))) {
                    dts.push(await this._stepWithRetry(Math.min(maxDt, target - this.time)));
                }
            }
            return { steps: dts.length, dts: Float64Array.from(dts), time: this.time };
        });
    }
}
