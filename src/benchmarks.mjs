// SWASHES analytic benchmarks (Delestre et al. 2013) with exact cell averages from 4x4 Gauss
// quadrature. Thacker (1981) planar surface in a paraboloid: 2D, periodic in time, moving
// shoreline. Goutal and Maurel (1997) steady flows over a bump: subcritical, transcritical, and
// transcritical with a hydraulic jump, driven by a discharge inflow and a stage outflow.
const points = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526];
const weights = [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538];

// ---------------------------------------------------------------- Thacker planar paraboloid
export const THACKER = Object.freeze({ L: 4, a: 1, h0: 0.1, eta: 0.5, gravity: 9.81 });
const thackerOmega = ({ a, h0, gravity }) => Math.sqrt(2 * gravity * h0) / a;
export const thackerPeriod = (p = THACKER) => 2 * Math.PI / thackerOmega(p);

export function thackerBed(x, y, { L, a, h0 } = THACKER) {
    const r2 = (x - L / 2) ** 2 + (y - L / 2) ** 2;
    return -h0 * (1 - r2 / (a * a));
}

// [h, hu, hv] at a point and time.
export function thackerExact(x, y, t, p = THACKER) {
    const { L, a, h0, eta } = p, w = thackerOmega(p);
    const h = Math.max(0, eta * h0 / (a * a) * (2 * (x - L / 2) * Math.cos(w * t) + 2 * (y - L / 2) * Math.sin(w * t) - eta)
        - thackerBed(x, y, p));
    return [h, -h * eta * w * Math.sin(w * t), h * eta * w * Math.cos(w * t)];
}

// Cell averages of [h, hu, hv, 0] and of the bed on an n x n grid at time t.
export function thackerAverages(n, t, p = THACKER) {
    const d = p.L / n, state = new Float64Array(4 * n * n), bed = new Float64Array(n * n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const c = j * n + i;
        for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) {
            const x = (i + 0.5 + points[a] / 2) * d, y = (j + 0.5 + points[b] / 2) * d, w = weights[a] * weights[b] / 4;
            const v = thackerExact(x, y, t, p);
            for (let k = 0; k < 3; k++) state[4 * c + k] += w * v[k];
            bed[c] += w * thackerBed(x, y, p);
        }
        // Zero momentum where the average depth is zero (input contract for dry cells).
        if (state[4 * c] === 0) { state[4 * c + 1] = 0; state[4 * c + 2] = 0; }
    }
    return { state, bed };
}

export function thackerScenario(n, p = THACKER) {
    const { state, bed } = thackerAverages(n, 0, p);
    return { config: { width: n, height: n, dx: p.L / n, dy: p.L / n, gravity: p.gravity, cfl: 0.4,
        boundaryX: 'wall', boundaryY: 'wall' }, initial: { state, bed }, period: thackerPeriod(p) };
}

// ---------------------------------------------------------------- Goutal-Maurel bump flows
export const BUMP_CASES = Object.freeze({
    subcritical: { q: 4.42, hOut: 2 },
    transcritical: { q: 1.53, hOut: 0.66 },
    shock: { q: 0.18, hOut: 0.33 },
});
const BUMP_LENGTH = 25, BUMP_TOP = 0.2, BUMP_CREST = 10;
export const bumpBed = x => (x > 8 && x < 12 ? BUMP_TOP - 0.05 * (x - BUMP_CREST) ** 2 : 0);

// Depth with specific energy head - z on the subcritical (h >= hc) or supercritical branch.
function bernoulliDepth(head, z, q, g, subcritical) {
    const hc = Math.cbrt(q * q / g), f = h => h + q * q / (2 * g * h * h) + z - head;
    if (f(hc) > 1e-12) throw new Error('Specific energy below the critical minimum');
    let lo = subcritical ? hc : hc * 1e-3, hi = subcritical ? Math.max(2 * hc, head - z + 1) : hc;
    for (let i = 0; i < 200; i++) {
        const mid = (lo + hi) / 2;
        // f decreases with h on the supercritical branch and increases on the subcritical one.
        if ((f(mid) > 0) === subcritical) hi = mid; else lo = mid;
    }
    return (lo + hi) / 2;
}

// Exact steady depth h(x) and the jump position (shock case).
export function bumpSolution(kind, g = 9.81) {
    const { q, hOut } = BUMP_CASES[kind], hc = Math.cbrt(q * q / g);
    const downstreamHead = hOut + q * q / (2 * g * hOut * hOut), criticalHead = 1.5 * hc + BUMP_TOP;
    if (kind === 'subcritical') return { q, depth: x => bernoulliDepth(downstreamHead, bumpBed(x), q, g, true) };
    const upstream = x => bernoulliDepth(criticalHead, bumpBed(x), q, g, x <= BUMP_CREST);
    if (kind === 'transcritical') return { q, depth: upstream };
    // Hydraulic jump: the conjugate depth of the supercritical branch meets the downstream branch.
    const conjugate = h1 => h1 / 2 * (Math.sqrt(1 + 8 * q * q / (g * h1 ** 3)) - 1);
    const mismatch = x => conjugate(upstream(x)) - bernoulliDepth(downstreamHead, bumpBed(x), q, g, true);
    // The downstream branch exists only where z <= downstreamHead - 1.5 hc.
    const zMax = downstreamHead - 1.5 * hc;
    let lo = zMax >= BUMP_TOP ? BUMP_CREST + 1e-9 : BUMP_CREST + Math.sqrt((BUMP_TOP - zMax) / 0.05) + 1e-9, hi = BUMP_LENGTH;
    if (mismatch(lo) > 0 === mismatch(hi) > 0) throw new Error('No hydraulic jump in the domain');
    for (let i = 0; i < 200; i++) { const mid = (lo + hi) / 2; if (mismatch(mid) > 0 === mismatch(lo) > 0) lo = mid; else hi = mid; }
    const jump = (lo + hi) / 2;
    return { q, jump, depth: x => x < jump ? upstream(x) : bernoulliDepth(downstreamHead, bumpBed(x), q, g, true) };
}

// 1D channel (height 1): discharge q per unit width on the west edge, stage hOut on the east.
// Initial state: still water at the outlet stage (or a 0.5 m minimum), as in SWASHES.
export function bumpScenario(kind, n, g = 9.81) {
    const { q, hOut } = BUMP_CASES[kind], d = BUMP_LENGTH / n, dy = 1;
    const bed = new Float64Array(n), state = new Float64Array(4 * n);
    for (let i = 0; i < n; i++) {
        for (let a = 0; a < 4; a++) bed[i] += weights[a] / 2 * bumpBed((i + 0.5 + points[a] / 2) * d);
        state[4 * i] = Math.max(Math.max(hOut, 0.5) - bed[i], 0);
    }
    const config = { width: n, height: 1, dx: d, dy, gravity: g, cfl: 0.4, boundaryX: 'wall', boundaryY: 'wall',
        hydrographs: [{ edge: 'west', kind: 'discharge', series: [[0, q * dy]] },
            { edge: 'east', kind: 'stage', series: [[0, hOut]] }] };
    return { config, initial: { state, bed }, finalTime: 200 };
}

// NTHMP benchmark problem 6, solitary waves on a conical island (Briggs et al., 1995; laboratory of
// the USACE Coastal and Hydraulics Laboratory). Geometry and gauges from the NTHMP problem description
// (data/nthmp/BP06/Description.pdf): basin water depth 0.32 m; truncated cone of 7.2 m toe and 2.2 m
// crest diameter, 0.625 m high (1:4 face), centred at (12.96, 13.80) m; gauge positions of Table 1,
// with the incident gauges 1 to 4 at a case-dependent x. Measured incident heights H/d: A 0.045,
// B 0.096, C 0.181.
// Modelling choices (not part of the description): the wave region x in [0, 23] m (paddle face to
// the absorber), y in [0, 28.2] m; the paddle face is a wall, the absorbing edges are open
// boundaries (their ~12 % reflection is not modelled); the incident wave is a solitary-wave
// profile eta = H sech²(k (x - x0)), k = sqrt(3H / (4 d³)), x0 = 3/k by default, with the velocity of a right-going
// simple wave of the shallow-water equations, u = 2 (sqrt(g (d + eta)) - sqrt(g d)), so the
// initial state carries no left-going wave.
export const CONICAL_ISLAND = Object.freeze({
    depth: 0.32, centre: [12.96, 13.80], toeRadius: 3.6, crestRadius: 1.1, height: 0.625, Lx: 23, Ly: 28.2,
    cases: { A: { H: 0.045, gauge1x: 5.76 }, B: { H: 0.096, gauge1x: 6.82 }, C: { H: 0.181, gauge1x: 7.56 } },
    // Incident H/d recorded by the laboratory: mean peak of gauges 1 to 4 in ts2a/ts2b/ts2cnew1.txt
    // (baseline: median of the first 2 s). Case B's record (0.0872) is 9 % below the stated 0.096.
    measured: { A: 0.0444, B: 0.0872, C: 0.1788 },
    gauges: { 6: [9.36, 13.80], 9: [10.36, 13.80], 16: [12.96, 11.22], 22: [15.56, 13.80] },
    incidentY: { 1: 16.05, 2: 14.55, 3: 13.05, 4: 11.55 },
});
export const conicalIslandBed = (x, y, p = CONICAL_ISLAND) => {
    const r = Math.hypot(x - p.centre[0], y - p.centre[1]);
    return Math.min(p.height, Math.max(0, (p.toeRadius - r) / 4));
};
export function conicalIslandCase(dx, kind = 'C', { x0, manning = 0, gravity = 9.81, measured = false } = {}) {
    const p = CONICAL_ISLAND, d = p.depth, H = (measured ? p.measured[kind] : p.cases[kind].H) * d, k = Math.sqrt(3 * H / (4 * d ** 3));
    x0 ??= 3 / k; // crest position: the profile is below 1 % of H at the paddle face
    const W = Math.round(p.Lx / dx), Hc = Math.round(p.Ly / dx), N = W * Hc;
    const bed = new Float64Array(N), state = new Float64Array(4 * N);
    for (let j = 0; j < Hc; j++) for (let i = 0; i < W; i++) {
        const x = (i + 0.5) * dx, y = (j + 0.5) * dx, n = j * W + i, z = conicalIslandBed(x, y);
        bed[n] = z;
        const eta = H / Math.cosh(k * (x - x0)) ** 2, h = Math.max(0, d + eta - z);
        const u = h > 0 && z === 0 ? 2 * (Math.sqrt(gravity * (d + eta)) - Math.sqrt(gravity * d)) : 0;
        state.set([h, h * u, 0, 0], 4 * n);
    }
    // Open edges; the west edge (paddle face) is closed by a zero-discharge segment, which reflects.
    const config = { width: W, height: Hc, dx, dy: dx, gravity, cfl: 0.4, boundaryX: 'open', boundaryY: 'open',
        hydrographs: [{ edge: 'west', kind: 'discharge', series: [[0, 0]] }] };
    const cell = (x, y) => Math.min(Hc - 1, Math.floor(y / dx)) * W + Math.min(W - 1, Math.floor(x / dx));
    const gauges = { 1: cell(p.cases[kind].gauge1x, p.incidentY[1]), 2: cell(p.cases[kind].gauge1x, p.incidentY[2]),
        3: cell(p.cases[kind].gauge1x, p.incidentY[3]), 4: cell(p.cases[kind].gauge1x, p.incidentY[4]),
        ...Object.fromEntries(Object.entries(p.gauges).map(([g, [x, y]]) => [g, cell(x, y)])) };
    const initial = { state, bed, ...(manning > 0 ? { manning: new Float64Array(N).fill(manning) } : {}) };
    return { config, initial, gauges, H, k, x0, celerity: Math.sqrt(gravity * (d + H)) };
}

// NTHMP benchmark problem 7, runup onto the Monai Valley beach (1:400 laboratory model; Liu et al.,
// 2008; problem description in data/nthmp/BP07/description.pdf). The bathymetry is given as still-water
// depth (m, positive below still water) at 393 x 244 nodes 0.014 m apart; each node becomes a cell
// centre. The incident wave (water level, m, for 0 to 22.5 s) is imposed at x = 0 by an incident-wave
// boundary, which becomes non-reflecting after 22.5 s (the recommended treatment); the other three
// sides are reflective walls. Gauges 5, 7 and 9 of the description's Table 1.
export const MONAI = Object.freeze({ dx: 0.014, width: 393, height: 244, incidentEnd: 22.5,
    gauges: { 5: [4.521, 1.196], 7: [4.521, 1.696], 9: [4.521, 2.196] } });
export function parseColumns(text) {
    const rows = [];
    for (const line of text.split(/\r?\n/)) {
        const v = line.trim().split(/[\s,]+/).map(Number);
        if (v.length >= 2 && v.every(Number.isFinite)) rows.push(v);
    }
    return rows;
}
export function monaiCase(bathymetryText, inputText, { manning = 0 } = {}) {
    const p = MONAI, rows = parseColumns(bathymetryText), W = p.width, H = p.height, N = W * H;
    if (rows.length !== N) throw new Error(`Monai bathymetry has ${rows.length} nodes, expected ${N}`);
    const bed = new Float64Array(N);
    for (const [x, y, depth] of rows) {
        const i = Math.round(x / p.dx), j = Math.round(y / p.dx);
        if (Math.abs(i * p.dx - x) > 1e-6 || Math.abs(j * p.dx - y) > 1e-6) throw new Error(`node (${x}, ${y}) is off the 0.014 m grid`);
        bed[j * W + i] = -depth;
    }
    const incident = parseColumns(inputText).map(([t, eta]) => [t, eta]);
    if (Math.abs(incident.at(-1)[0] - p.incidentEnd) > 1e-9) throw new Error('incident record does not end at 22.5 s');
    // After the record the incident level returns to still water, so the boundary lets waves leave.
    return monaiCaseFromBed(bed, [...incident, [p.incidentEnd + 0.05, 0]], { manning });
}
// The same case from a bed elevation array (rows south to north) and the incident-wave series.
export function monaiCaseFromBed(bed, series, { manning = 0 } = {}) {
    const p = MONAI, W = p.width, H = p.height, N = W * H, state = new Float64Array(4 * N);
    if (bed.length !== N) throw new Error(`Monai bed has ${bed.length} cells, expected ${N}`);
    for (let n = 0; n < N; n++) state[4 * n] = Math.max(0, -bed[n]);
    const config = { width: W, height: H, dx: p.dx, dy: p.dx, gravity: 9.81, cfl: 0.4, boundaryX: 'wall', boundaryY: 'wall',
        hydrographs: [{ edge: 'west', kind: 'wave', series, still: 0 }] };
    const gauges = Object.fromEntries(Object.entries(p.gauges).map(([g, [x, y]]) => [g, Math.round(y / p.dx) * W + Math.round(x / p.dx)]));
    return { config, initial: { state, bed, ...(manning > 0 ? { manning: new Float64Array(N).fill(manning) } : {}) }, gauges };
}

// Exact cell averages of h for the bump cases.
export function bumpAverages(kind, n, g = 9.81) {
    const { depth, q } = bumpSolution(kind, g), d = BUMP_LENGTH / n, h = new Float64Array(n);
    for (let i = 0; i < n; i++) for (let a = 0; a < 4; a++) h[i] += weights[a] / 2 * depth((i + 0.5 + points[a] / 2) * d);
    return { h, q };
}
