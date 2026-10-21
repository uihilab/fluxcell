// Deterministic, local-only fixtures. Dyadic beds allow exact f32 lake equilibria.
export const SCENARIOS = Object.freeze(['constant', 'lake', 'lake-rounded', 'shoreline', 'rain', 'dam', 'dry-dam', 'pulse',
    'friction', 'sources', 'obstacle', 'lake-obstacle', 'open-uniform', 'open-dam', 'outflow-drain',
    'stage-lake', 'hydrograph-channel', 'wave-lake', 'wave-channel', 'wind-uniform', 'wind-basin', 'coriolis-uniform', 'coriolis-basin']);
// Manning decay of a uniform state: |q| obeys d|q|/dt = -k|q|², k = g n² / h^(7/3).
export function frictionExact(state0, n, gravity, time) {
    const [h, qx, qy] = state0, q = Math.hypot(qx, qy), k = gravity * n * n / h ** (7 / 3);
    const factor = 1 / (1 + k * q * time);
    return [h, qx * factor, qy * factor];
}

export function scenario(name, width = 17, height = 33) {
    const config = { width, height, dx: 1, dy: 1.5, gravity: 9.81, cfl: 0.4,
        boundaryX: 'wall', boundaryY: 'wall', rain: 0 };
    const state = new Float32Array(4 * width * height), bed = new Float32Array(width * height);
    if (name === 'constant') { config.boundaryX = 'periodic'; config.boundaryY = 'periodic'; }
    if (name === 'rain') config.rain = 1e-5;
    if (!SCENARIOS.includes(name)) throw new Error(`Unknown scenario: ${name}`);
    const n = width * height, manning = new Float32Array(n), source = new Float32Array(n), solid = new Uint8Array(n);
    if (name === 'friction') { config.boundaryX = 'periodic'; config.boundaryY = 'periodic'; manning.fill(0.03); }
    if (name === 'hydrograph-channel') manning.fill(0.05);
    // Wind: uniform acceleration on a periodic domain (exact: q = tau t), and a closed basin with
    // a bumpy bed, a shoreline, an island block and Manning friction.
    if (name === 'wind-uniform') Object.assign(config, { boundaryX: 'periodic', boundaryY: 'periodic', wind: [20, -10] });
    if (name === 'wind-basin') { config.wind = [18, 6]; manning.fill(0.03); }
    // Coriolis: a uniform current on a periodic domain turns in an inertial circle (exact), and a
    // closed basin with a bumpy bed, a raised dome of water and an obstacle adjusts under rotation.
    if (name === 'coriolis-uniform') Object.assign(config, { boundaryX: 'periodic', boundaryY: 'periodic', coriolis: 1e-3 });
    if (name === 'coriolis-basin') config.coriolis = 5e-4;
    if (name === 'open-uniform' || name === 'open-dam') { config.boundaryX = 'open'; config.boundaryY = 'open'; }
    if (name === 'outflow-drain') { config.boundaryX = 'outflow'; config.boundaryY = 'outflow'; }
    if (name === 'sources') config.rain = 2e-6;
    if (name === 'stage-lake') {
        // Stage = lake level on a full, a partial and a central segment; the rest is wall/outflow.
        config.boundaryY = 'outflow';
        config.hydrographs = [{ edge: 'west', kind: 'stage', series: [[0, 2]] },
            { edge: 'east', start: Math.min(1, height - 1), end: Math.max(1, height - 1), kind: 'stage', series: [[0, 2], [5, 2]] },
            { edge: 'south', start: Math.floor(width / 4), end: Math.max(Math.floor(width / 4) + 1, Math.floor(3 * width / 4)), kind: 'stage', series: [[0, 2]] }];
    }
    if (name === 'hydrograph-channel') {
        // Flat frictional channel: 2 m³/s (ramped over 20 s) enters on the west edge; the east
        // edge holds a 0.5 m stage. Steady state: uniform unit discharge, surface at the stage.
        Object.assign(config, { dx: 10, dy: 5 });
        config.hydrographs = [{ edge: 'west', kind: 'discharge', series: [[0, 0], [20, 2]] },
            { edge: 'east', kind: 'stage', series: [[0, 0.5]] }];
    }
    if (name === 'wave-lake') {
        // Incident-wave boundaries at the still level (a full, a partial and a central segment).
        config.boundaryY = 'outflow';
        config.hydrographs = [{ edge: 'west', kind: 'wave', series: [[0, 2]], still: 2 },
            { edge: 'east', start: Math.min(1, height - 1), end: Math.max(1, height - 1), kind: 'wave', series: [[0, 2], [5, 2]], still: 2 },
            { edge: 'north', start: Math.floor(width / 4), end: Math.max(Math.floor(width / 4) + 1, Math.floor(3 * width / 4)), kind: 'wave', series: [[0, 2]], still: 2 }];
    }
    if (name === 'wave-channel') {
        // A wave enters on the west over a bumpy bed and leaves through the east wave boundary.
        config.hydrographs = [{ edge: 'west', kind: 'wave', series: [[0, 1.5], [1, 1.75], [2, 1.5], [3, 1.5]], still: 1.5 },
            { edge: 'east', kind: 'wave', series: [[0, 1.5]], still: 1.5 }];
    }
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const i = y * width + x, k = 4 * i;
        if (name === 'wave-lake') { bed[i] = ((x * 3 + y * 5) % 13) / 8; state[k] = 2 - bed[i]; }
        if (name === 'wind-uniform') state[k] = 1.25;
        if (name === 'coriolis-uniform') { state[k] = 1.25; state[k + 1] = 0.375; state[k + 2] = -0.125; }
        if (name === 'coriolis-basin') {
            bed[i] = ((x * 3 + y * 5) % 7) / 32;
            solid[i] = x === Math.floor(width / 2) && y > 2 && y < 6 ? 1 : 0;
            const r2 = ((x + 0.5) / width - 0.4) ** 2 + ((y + 0.5) / height - 0.5) ** 2;
            state[k] = solid[i] ? 0 : 1.5 + 0.5 * Math.exp(-30 * r2) - bed[i];
        }
        if (name === 'wind-basin') {
            bed[i] = ((x * 3 + y * 5) % 13) / 16 + 1.5 * x / width;
            solid[i] = x === Math.floor(width / 3) && y > 1 && y < 4 ? 1 : 0;
            state[k] = solid[i] ? 0 : Math.max(0, 1.5 - bed[i]);
        }
        if (name === 'wave-channel') { bed[i] = ((x * 3 + y * 5) % 7) / 32; state[k] = 1.5 - bed[i]; }
        if (name === 'stage-lake') { bed[i] = ((x * 3 + y * 5) % 13) / 8; state[k] = 2 - bed[i]; }
        if (name === 'hydrograph-channel') state[k] = 0.5;
        if (name === 'lake' || name === 'shoreline') {
            bed[i] = ((x * 3 + y * 5) % 13) / 8;
            state[k] = Math.max(0, (name === 'lake' ? 2 : 1) - bed[i]);
        }
        if (name === 'lake-rounded') {
            // Non-dyadic bed, adjusted so the f32 data are an exact discrete lake: h + z == 2.
            let h = Math.fround(2 - Math.fround(0.6 + 0.5 * Math.sin(0.3 * x) * Math.cos(0.2 * y)));
            let z = Math.fround(2 - h);
            if (h + z !== 2) { h = Math.fround(2 - z); z = Math.fround(2 - h); }
            if (h + z !== 2) throw new Error('lake-rounded fixture is not an exact f32 lake');
            bed[i] = z; state[k] = h;
        }
        if (name === 'dam' || name === 'dry-dam') {
            state[k] = x < Math.floor(width / 2) ? 2 : name === 'dam' ? 0.25 : 0;
        }
        if (name === 'constant' || name === 'open-uniform' || name === 'outflow-drain') { state[k] = 1.25; state[k + 1] = 0.375; state[k + 2] = -0.125; }
        if (name === 'friction') { state[k] = 0.5; state[k + 1] = 0.4; state[k + 2] = 0.3; }
        if (name === 'open-dam') {
            const r = ((x + 0.5) / width - 0.3) ** 2 + ((y + 0.5) / height - 0.6) ** 2;
            state[k] = r < 0.04 ? 2 : 0.5;
        }
        if (name === 'sources') {
            bed[i] = ((x * 3 + y * 5) % 7) / 32;
            // A distributed discharge over a disk plus a point-like line source; dry start.
            if (((x + 0.5) / width - 0.5) ** 2 + ((y + 0.5) / height - 0.5) ** 2 < 0.03) source[i] = 1e-3;
            if (x === 1) source[i] += 5e-4;
        }
        if (name === 'obstacle' || name === 'lake-obstacle') {
            // An L-shaped building and a free-standing block; solid cells must be dry.
            const block = (x >= Math.floor(width / 2) + 1 && x <= Math.floor(width / 2) + 3 && y >= 3 && y < height - 4)
                || (y === 3 && x > Math.floor(width / 2) + 3 && x < width - 2) || (x === 2 && y === 2);
            solid[i] = block ? 1 : 0;
            if (name === 'obstacle') state[k] = block ? 0 : x < Math.floor(width / 2) ? 2 : 0.25;
            else { bed[i] = ((x * 3 + y * 5) % 13) / 8; state[k] = block ? 0 : Math.max(0, 1.5 - bed[i]); }
        }
        if (name === 'pulse') {
            bed[i] = ((x * 3 + y * 5) % 7) / 32;
            const radius = ((x + 0.5) / width - 0.5) ** 2 + ((y + 0.5) / height - 0.5) ** 2;
            state[k] = 0.25 + Math.exp(-40 * radius);
            state[k + 1] = state[k] * 0.2 * Math.sin(x + 2 * y);
            state[k + 2] = state[k] * 0.15 * Math.cos(2 * x - y);
        }
    }
    return { config, initial: { state, bed, manning, source, solid } };
}
