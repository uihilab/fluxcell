import test from 'node:test';
import assert from 'node:assert/strict';
import { ReferenceSolver } from '../src/reference.mjs';
import { MusclReferenceSolver } from '../src/muscl-reference.mjs';
import { configuration, diagnostics, hydrographGhost } from '../src/model.mjs';

// Incident-wave ('wave') boundaries: a characteristic boundary that generates a prescribed incident
// water level on still water and lets outgoing waves leave (model.mjs: hydrographGhost, kind 3).
const methods = [['Rusanov', ReferenceSolver], ['MUSCL', MusclReferenceSolver]];
const g = 9.81, depth = 1;
const volume = solver => diagnostics(solver.state, solver.config).volume;

// Channel of n cells (dx = 0.1 m, one row) with a wave segment on the east and/or west edge.
function channel(n, hydrographs, eta = () => 0) {
    const state = new Float64Array(4 * n);
    for (let i = 0; i < n; i++) {
        const e = eta((i + 0.5) * 0.1), h = depth + e;
        state.set([h, h * 2 * (Math.sqrt(g * h) - Math.sqrt(g * depth)), 0, 0], 4 * i); // right-going simple wave
    }
    return { config: { width: n, height: 1, dx: 0.1, dy: 0.1, gravity: g, boundaryX: 'wall', boundaryY: 'wall', hydrographs },
        initial: { state, bed: new Float64Array(n) } };
}

test('wave ghost: still water gives the interior state exactly; inflow carries no tangential velocity', () => {
    for (const inward of [1, -1]) {
        assert.deepEqual(hydrographGhost(3, 2, 1.5, 0, 0.4, 0.5, g, inward, 2).map(v => v + 0), [1.5, 0, 0.4]); // + 0: -0 to 0
        const [h, un, vt] = hydrographGhost(3, 2.1, 1.5, 0, 0.4, 0.5, g, inward, 2); // incident crest
        assert.ok(h > 1.5 && inward * un > 0 && vt === 0, `${h} ${un} ${vt}`);
    }
    // Supercritical outflow: the interior state is kept.
    assert.deepEqual(hydrographGhost(3, 2, 1, -5, 0.2, 1, g, 1, 2), [1, -5, 0.2]);
});

for (const [label, Solver] of methods) {
    test(`${label}: a lake at the still level with wave boundaries on every edge stays exactly at rest`, () => {
        const w = 13, h = 9, n = w * h, bed = new Float64Array(n), state = new Float64Array(4 * n);
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
            const i = y * w + x;
            bed[i] = 0.3 * Math.exp(-((x - 6) ** 2 + (y - 4) ** 2) / 6) + 0.05 * Math.sin(x + 2 * y);
            state[4 * i] = 2 - bed[i];
        }
        const wave = edge => ({ edge, kind: 'wave', series: [[0, 2], [5, 2]], still: 2 });
        const config = { width: w, height: h, dx: 1, dy: 1, hydrographs: ['west', 'east', 'south', 'north'].map(wave) };
        const solver = new Solver(config, { state, bed });
        for (let i = 0; i < 300; i++) solver.step(0.05);
        const error = Math.max(...solver.state.map((v, i) => i % 4 === 3 ? 0 : Math.abs(v - state[i])));
        assert.ok(error < 1e-12, `max state error ${error}`);
    });

    test(`${label}: a wave leaves through a wave boundary with little reflection; the budget closes`, () => {
        const A = 0.05, sigma = 0.5;
        const { config, initial } = channel(300, [{ edge: 'east', kind: 'wave', series: [[0, depth]], still: depth }],
            x => A * Math.exp(-(((x - 8) / sigma) ** 2)));
        const solver = new Solver(config, initial), v0 = volume(solver);
        // The crest travels at about sqrt(g (d + A)) = 3.2 m/s and reaches the east edge (30 m) in 7 s.
        while (solver.time < 14) solver.step(0.05);
        let remaining = 0;
        for (let i = 0; i < 300; i++) remaining = Math.max(remaining, Math.abs(solver.state[4 * i] - depth));
        assert.ok(remaining < 0.02 * A, `largest remaining surface displacement ${remaining} (amplitude ${A})`);
        const residual = volume(solver) + solver.boundaryOutflow - v0;
        assert.ok(Math.abs(residual) < 1e-10 * v0, `budget residual ${residual}`);
        assert.ok(solver.boundaryOutflow > 0.9 * A * sigma * Math.sqrt(Math.PI) * 0.1, `outflow ${solver.boundaryOutflow}`);
    });

    test(`${label}: an incident wave enters through a wave boundary at its prescribed height`, () => {
        const A = 0.02, T = 2, series = [];
        for (let t = 0; t <= 6; t += 0.05) series.push([t, depth + A * Math.exp(-(((t - T) / 0.6) ** 2))]);
        series.push([60, depth]);
        const { config, initial } = channel(400, [{ edge: 'west', kind: 'wave', series, still: depth },
            { edge: 'east', kind: 'wave', series: [[0, depth]], still: depth }]);
        const solver = new Solver(config, initial);
        let peak = 0;
        while (solver.time < 8) { solver.step(0.02); peak = Math.max(peak, solver.state[4 * 20] - depth); } // x = 2.05 m
        assert.ok(Math.abs(peak - A) < 0.05 * A, `peak ${peak} against incident ${A}`);
    });
}

// Continuous waves against a wall: the boundary keeps generating the incident wave and absorbs the
// reflected one, so a standing wave forms with twice the incident amplitude at the wall and does
// not grow (a reflecting generator would trap the reflected energy). 5 mm waves on 1 m of water,
// period 4 s (wavelength 12.5 m, 50 cells), in a 50 m channel; amplitudes after the transient.
for (const [label, Solver] of methods) {
    test(`${label}: continuous waves reflected by a wall form a steady standing wave`, () => {
        const A = 0.005, T = 4, n = 200, dx = 0.25, series = [];
        for (let t = 0; t <= 200; t += 0.05) series.push([t, depth + A * Math.min(1, t / (2 * T)) * Math.sin(2 * Math.PI * t / T)]);
        const state = new Float64Array(4 * n);
        for (let i = 0; i < n; i++) state[4 * i] = depth;
        const solver = new Solver({ width: n, height: 1, dx, dy: dx, gravity: g, boundaryX: 'wall', boundaryY: 'wall',
            hydrographs: [{ edge: 'west', kind: 'wave', series, still: depth }] }, { state, bed: new Float64Array(n) });
        const wall = n - 1, node = Math.floor((n * dx - T * Math.sqrt(g * depth) / 4) / dx); // the cell a quarter wavelength from the wall
        let first = 0, second = 0, nodeAmplitude = 0;
        while (solver.time < 120) {
            solver.step(0.02);
            const e = Math.abs(solver.state[4 * wall] - depth);
            if (solver.time > 60 && solver.time <= 90) first = Math.max(first, e);
            if (solver.time > 90) second = Math.max(second, e);
            if (solver.time > 80) nodeAmplitude = Math.max(nodeAmplitude, Math.abs(solver.state[4 * node] - depth));
        }
        assert.ok(Math.abs(second / first - 1) < 0.01, `wall amplitude ${first / A} A then ${second / A} A: no build-up`);
        if (label === 'MUSCL') { // the first-order scheme damps the 50-cell wave by about a quarter over 10 m
            assert.ok(Math.abs(second / (2 * A) - 1) < 0.03, `wall amplitude ${second / A} A against 2 A`);
            assert.ok(nodeAmplitude < 0.25 * A, `node amplitude ${nodeAmplitude / A} A`);
        }
    });
}

test('wave segments need a finite still-water level', () => {
    const base = { width: 8, height: 5 };
    assert.throws(() => configuration({ ...base, hydrographs: [{ edge: 'west', kind: 'wave', series: [[0, 1]] }] }), RangeError);
    assert.throws(() => configuration({ ...base, hydrographs: [{ edge: 'west', kind: 'wave', series: [[0, 1]], still: NaN }] }), RangeError);
    assert.equal(configuration({ ...base, hydrographs: [{ edge: 'west', kind: 'wave', series: [[0, 1]], still: 1 }] }).hydrographs[0].still, 1);
});
