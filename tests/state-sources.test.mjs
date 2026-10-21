import test from 'node:test';
import assert from 'node:assert/strict';
import { ReferenceSolver } from '../src/reference.mjs';
import { MusclReferenceSolver } from '../src/muscl-reference.mjs';
import { diagnostics } from '../src/model.mjs';

const methods = [['Rusanov', ReferenceSolver], ['MUSCL', MusclReferenceSolver]];

// A sloping 14 x 9 basin, partly wet, open to the east, with a solid block and a tracer.
function basin({ tracer = false } = {}) {
    const width = 14, height = 9, n = width * height;
    const bed = new Float64Array(n), state = new Float64Array(4 * n), solid = new Uint8Array(n), concentration = new Float64Array(n);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const i = y * width + x;
        bed[i] = 0.05 * x + 0.02 * Math.sin(y);
        if (x >= 4 && x <= 5 && y >= 3 && y <= 4) solid[i] = 1;
        else if (x < 6) { state[4 * i] = Math.max(0, 0.4 - bed[i]); concentration[i] = 0.5; }
    }
    const config = { width, height, dx: 2, dy: 1.5, boundaryX: 'outflow', boundaryY: 'wall', tracer };
    return { config, initial: { state, bed, solid, manning: new Float64Array(n).fill(0.03), ...(tracer ? { concentration } : {}) } };
}

// Moving point inflows: frame f puts rate r at three cells that shift east each frame.
const frameSources = (f, width) => [1 + f % 9, width + 2 + f % 9, 3 * width + 7].map((i, k) => [i, (k + 1) * 1e-3 * (1 + 0.1 * f)]);

for (const [label, Solver] of methods) {
    for (const tracer of [false, true]) {
        test(`${label}${tracer ? ' (tracer)' : ''}: setSources equals setForcing with the full array`, () => {
            const { config, initial } = basin({ tracer }), n = config.width * config.height;
            const a = new Solver(config, initial), b = new Solver(config, initial);
            let previous = [];
            const full = new Float64Array(n), fullC = new Float64Array(n);
            for (let f = 0; f < 12; f++) {
                const now = frameSources(f, config.width);
                // Sparse: clear last frame's cells, set this frame's (a cell may be in both lists; the last write wins).
                const indices = [...previous.map(([i]) => i), ...now.map(([i]) => i)];
                const rates = [...previous.map(() => 0), ...now.map(([, r]) => r)];
                const concentrations = tracer ? [...previous.map(() => 0), ...now.map(() => 0.8 + 0.01 * f)] : undefined;
                a.setSources(indices, rates, concentrations);
                full.fill(0); fullC.fill(0);
                for (const [i, r] of now) { full[i] = r; fullC[i] = 0.8 + 0.01 * f; }
                b.setForcing({ source: full, ...(tracer ? { sourceConcentration: fullC } : {}) });
                for (let k = 0; k < 10; k++) { a.step(0.05); b.step(0.05); }
                previous = now;
            }
            assert.deepEqual(a.readState(), b.readState());
            assert.equal(a.sourceVolume, b.sourceVolume);
            assert.equal(a.time, b.time);
            if (tracer) { assert.deepEqual(a.readTracer(), b.readTracer()); assert.equal(a.tracerSourceAmount, b.tracerSourceAmount); }
        });
    }

    test(`${label}: setState equals a solver started from that state, and the budget closes`, () => {
        const { config, initial } = basin(), n = config.width * config.height;
        const a = new Solver(config, initial);
        a.setSources([2, 3], [2e-3, 1e-3]);
        for (let k = 0; k < 40; k++) a.step(0.05);
        // Edit: raise the western cells by 0.1 m keeping their velocity, dry one cell.
        const edited = a.readState();
        for (let i = 0; i < n; i++) {
            if (initial.solid[i] || i % config.width > 2) continue;
            const h = edited[4 * i], u = h > 0 ? edited[4 * i + 1] / h : 0, v = h > 0 ? edited[4 * i + 2] / h : 0;
            edited[4 * i] = h + 0.1; edited[4 * i + 1] = u * (h + 0.1); edited[4 * i + 2] = v * (h + 0.1);
        }
        edited.fill(0, 4 * 30, 4 * 31);
        const before = diagnostics(a.readState(), a.config).volume;
        a.setState(edited);
        assert.ok(Math.abs(a.stateEditVolume - (diagnostics(edited, a.config).volume - before)) < 1e-12);
        const b = new Solver(config, { ...initial, state: edited });
        b.setSources([2, 3], [2e-3, 1e-3]);
        for (let k = 0; k < 60; k++) { a.step(0.05); b.step(0.05); }
        assert.deepEqual(a.readState(), b.readState());
        // Volume now = initial + sources - outflow + edits.
        const v0 = diagnostics(initial.state, a.config).volume, v = diagnostics(a.readState(), a.config).volume;
        const balance = v0 + a.sourceVolume - a.boundaryOutflow + a.stateEditVolume;
        assert.ok(Math.abs(v - balance) < 1e-11 * Math.max(1, v), `${v} vs ${balance}`);
    });

    test(`${label}: setSources and setState are validated`, () => {
        const { config, initial } = basin(), n = config.width * config.height, solver = new Solver(config, initial);
        assert.throws(() => solver.setSources([0], [-1e-3]), RangeError);
        assert.throws(() => solver.setSources([n], [1e-3]), RangeError);
        assert.throws(() => solver.setSources([0, 1], [1e-3]), RangeError);
        assert.throws(() => solver.setSources([0], [1e-3], [0.5]), RangeError, 'concentrations need a tracer');
        const state = solver.readState();
        assert.throws(() => solver.setState(state.subarray(4)), RangeError);
        const wetSolid = state.slice(); wetSolid[4 * (3 * config.width + 4)] = 0.1;
        assert.throws(() => solver.setState(wetSolid), RangeError, 'solid cells stay dry');
        const dryMoving = state.slice(); dryMoving[4 * 13] = 0; dryMoving[4 * 13 + 1] = 0.2;
        assert.throws(() => solver.setState(dryMoving), RangeError, 'dry cells have no momentum');
        const negative = state.slice(); negative[0] = -0.01;
        assert.throws(() => solver.setState(negative), RangeError);
        assert.throws(() => new Solver({ ...config, tracer: true }, { ...initial, concentration: new Float64Array(n) }).setState(state), RangeError, 'tracer runs need concentration');
        assert.deepEqual(solver.readState(), state, 'rejected changes leave the state untouched');
    });
}
