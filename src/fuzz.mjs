// Seeded random stress scenarios for robustness testing (CPU references and GPU kernels).
// Every case is reproducible from its seed. Cases deliberately combine the conditions that
// break shallow-water codes: very thin and very deep water, steep and vertical bed steps,
// buildings, all boundary types, sources on dry ground, friction and large elevation offsets.
import { BOUNDARIES } from './model.mjs';

// Mulberry32: small, fast, deterministic.
export function rng(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const pick = (r, values) => values[Math.floor(r() * values.length)];
const logUniform = (r, lo, hi) => Math.exp(Math.log(lo) + r() * (Math.log(hi) - Math.log(lo)));

// Smooth random field: a sum of random Fourier modes, scaled to [0, 1].
function field(r, w, h, modes = 6) {
    const terms = Array.from({ length: modes }, () => ({
        kx: (1 + Math.floor(r() * 6)) * (r() < 0.5 ? -1 : 1), ky: (1 + Math.floor(r() * 6)) * (r() < 0.5 ? -1 : 1),
        phase: r() * 2 * Math.PI, amp: r() }));
    const out = new Float64Array(w * h);
    let lo = Infinity, hi = -Infinity;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        let v = 0;
        for (const t of terms) v += t.amp * Math.sin(2 * Math.PI * (t.kx * x / w + t.ky * y / h) + t.phase);
        out[y * w + x] = v; lo = Math.min(lo, v); hi = Math.max(hi, v);
    }
    for (let i = 0; i < out.length; i++) out[i] = hi > lo ? (out[i] - lo) / (hi - lo) : 0;
    return out;
}

// Returns { seed, description, config, initial, steps } with Float32 inputs (GPU-representable).
// tracer: also a random concentration field, source and inflow concentrations and, in a third of
// the cases, a diffusivity, drawn from a second stream so the flow part of each seed is unchanged.
export function fuzzScenario(seed, { maxCells = 96 * 96, steps = 400, tracer = false } = {}) {
    const r = rng(seed);
    const width = Math.max(1, Math.floor(logUniform(r, 1, Math.sqrt(maxCells) * 1.6)));
    const height = Math.max(1, Math.min(Math.floor(maxCells / width), Math.floor(logUniform(r, 1, Math.sqrt(maxCells) * 1.6))));
    const n = width * height;
    const dx = logUniform(r, 0.5, 50), dy = r() < 0.5 ? dx : logUniform(r, 0.5, 50);
    const relief = pick(r, [0, 0.1, 2, 50, 500]);           // m across the domain
    const offset = pick(r, [0, 0, -5, 300, 2500, 8000]);     // absolute elevation, tests f32 handling
    const cliffs = r() < 0.4;                                 // add vertical steps of up to relief/2
    const depthKind = pick(r, ['dry', 'film', 'shallow', 'deep', 'dam', 'mixed']);
    const config = { width, height, dx, dy, gravity: 9.81, cfl: 0.4,
        boundaryX: pick(r, BOUNDARIES), boundaryY: pick(r, BOUNDARIES),
        rain: r() < 0.3 ? logUniform(r, 1e-7, 1e-4) : 0 };
    const base = field(r, width, height), mask = field(r, width, height), patch = field(r, width, height);
    const bed = new Float32Array(n), state = new Float32Array(4 * n);
    const manning = new Float32Array(n), source = new Float32Array(n), solid = new Uint8Array(n);
    const n0 = r() < 0.3 ? 0 : logUniform(r, 0.01, 0.1);
    const buildings = r() < 0.35 ? logUniform(r, 0.02, 0.3) : 0;
    const level = offset + relief * r();
    const speed = pick(r, [0, 0, 1, 10, 40]);
    for (let i = 0; i < n; i++) {
        let z = offset + relief * base[i];
        if (cliffs && mask[i] > 0.6) z += relief / 2;
        bed[i] = z;
        manning[i] = n0 * (0.5 + r());
        solid[i] = buildings && patch[i] < buildings ? 1 : 0;
        if (solid[i]) continue;
        let h = 0;
        if (depthKind === 'film') h = patch[i] > 0.5 ? logUniform(r, 1e-9, 1e-3) : 0;
        if (depthKind === 'shallow') h = 0.01 + 0.5 * patch[i];
        if (depthKind === 'deep') h = 5 + 95 * patch[i];
        if (depthKind === 'dam') h = (i % width) < width / 2 ? 2 + 8 * patch[i] : 0;
        if (depthKind === 'mixed') h = Math.max(0, level - z);
        h = Math.fround(h);
        state[4 * i] = h;
        if (h > 0 && speed) {
            state[4 * i + 1] = h * speed * (r() - 0.5) * 2;
            state[4 * i + 2] = h * speed * (r() - 0.5) * 2;
        }
        if (r() < 0.02) source[i] = logUniform(r, 1e-6, 1e-2);
    }
    // Hydrograph segments on non-periodic edges: stages from below the bed to well above it,
    // discharges from a trickle to a flood, constant or varying in time. Half of the stage segments
    // become incident-wave segments with a random still level; that choice draws from a separate
    // stream so the other seeded cases are unchanged.
    const hydrographs = [], waves = rng(seed ^ 0x5bd1e995);
    if (r() < 0.5) {
        for (const edge of ['west', 'east', 'south', 'north']) {
            const periodic = (edge === 'west' || edge === 'east' ? config.boundaryX : config.boundaryY) === 'periodic';
            if (periodic || r() < 0.5) continue;
            const length = edge === 'west' || edge === 'east' ? height : width;
            const start = Math.floor(r() * length), end = start + 1 + Math.floor(r() * (length - start));
            const kind = r() < 0.5 ? 'stage' : 'discharge';
            const value = () => kind === 'stage' ? offset + relief * r() + 3 * r() - 1 : logUniform(r, 1e-3, 50);
            const series = r() < 0.5 ? [[0, value()]] : [[0, value()], [steps * (0.01 + r()), value()], [steps * (1.01 + r()), value()]];
            let open = 0;
            for (let k = start; k < end; k++) {
                const x = edge === 'west' ? 0 : edge === 'east' ? width - 1 : k, y = edge === 'south' ? 0 : edge === 'north' ? height - 1 : k;
                if (!solid[y * width + x]) open++;
            }
            if (!open) continue;
            if (kind === 'stage' && waves() < 0.5) hydrographs.push({ edge, start, end, kind: 'wave', series, still: offset + relief * waves() + 3 * waves() - 1 });
            else hydrographs.push({ edge, start, end, kind, series });
        }
    }
    config.hydrographs = hydrographs;
    let tracerData = {}, tracerText = '';
    if (tracer) {
        const t = rng(seed ^ 0x9e3779b9), scale = pick(t, [1, 1, 1e-3, 1e3]);
        const smooth = field(t, width, height), sharp = t() < 0.5;
        const concentration = new Float32Array(n), sourceConcentration = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            concentration[i] = scale * (sharp ? (smooth[i] > 0.5 ? 1 : 0) : smooth[i]);
            sourceConcentration[i] = scale * t();
        }
        config.tracer = true;
        config.hydrographs = hydrographs.map(h => ({ ...h, concentration: scale * t() }));
        // Diffusivity up to a cell Peclet-like ratio that keeps the diffusion limit near the CFL step.
        config.diffusivity = t() < 1 / 3 ? logUniform(t, 1e-4, 1) * Math.min(dx, dy) : 0;
        tracerData = { concentration, sourceConcentration };
        tracerText = ` tracer=${sharp ? 'step' : 'smooth'}x${scale} D=${config.diffusivity.toPrecision(2)}`;
    }
    const description = `${width}x${height} dx=${dx.toPrecision(3)} dy=${dy.toPrecision(3)} relief=${relief} offset=${offset}`
        + `${cliffs ? ' cliffs' : ''} depth=${depthKind} speed=${speed} n=${n0.toPrecision(2)} buildings=${buildings ? 'yes' : 'no'}`
        + ` bc=${config.boundaryX}/${config.boundaryY} rain=${config.rain.toPrecision(2)}`
        + (hydrographs.length ? ` hydrographs=${hydrographs.map(h => `${h.edge}:${h.kind}[${h.start},${h.end})`).join(',')}` : '')
        + tracerText;
    return { seed, description, config, initial: { state, bed, manning, source, solid, ...tracerData }, steps };
}
