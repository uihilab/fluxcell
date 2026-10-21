import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { ReferenceSolver } from '../src/reference.mjs';
import { MusclReferenceSolver } from '../src/muscl-reference.mjs';
import { scenario } from '../src/scenarios.mjs';
import { diagnostics, initialData, configuration } from '../src/model.mjs';
import { loadDem, terrainScenario, DEMS } from '../src/terrain.mjs';

test('datum shift: auto datum is the minimum open-cell bed and does not change the f64 solution', () => {
    const { config, initial } = scenario('obstacle', 15, 11);
    const raised = { ...initial, bed: Float64Array.from(initial.bed, z => z + 1500.25) };
    assert.equal(initialData(configuration(config), raised).datum, 1500.25);
    assert.equal(initialData(configuration({ ...config, datum: 0 }), raised).datum, 0);
    assert.throws(() => configuration({ ...config, datum: 'sea' }));
    const a = new MusclReferenceSolver(config, initial), b = new MusclReferenceSolver(config, raised);
    for (let i = 0; i < 100; i++) b.step(a.step(0.05));
    assert.ok(Math.max(...a.state.map((v, i) => Math.abs(v - b.state[i]))) < 1e-12);
});

// The 2 m lidar tiles are fetched locally (tools/dem/fetch-3dep.mjs) and not versioned.
const available = async name => readFile(new URL(`../data/dem/${name}.f32`, import.meta.url)).then(() => true, () => false);
const range = values => { let lo = Infinity, hi = -Infinity; for (const v of values) { lo = Math.min(lo, v); hi = Math.max(hi, v); } return [lo, hi]; };

for (const name of Object.keys(DEMS)) {
    test(`${name}: loader flips rows to south-first, crops and matches metadata`, async t => {
        if (!(await available(name))) { t.skip('tile not fetched'); return; }
        const dem = await loadDem(name), file = await readFile(new URL(`../data/dem/${name}.f32`, import.meta.url));
        const served = new Float32Array(file.buffer, file.byteOffset, file.byteLength / 4), W = dem.meta.width, H = dem.meta.height;
        const [x0, y0] = dem.origin;
        assert.equal(dem.bed.length, dem.width * dem.height);
        assert.equal(dem.bed[0], served[(H - 1 - y0) * W + x0]);
        assert.equal(dem.bed.at(-1), served[(H - y0 - dem.height) * W + x0 + dem.width - 1]);
        const [lo, hi] = range(dem.bed);
        if (!DEMS[name].crop) { assert.equal(lo, dem.meta.min); assert.equal(hi, dem.meta.max); }
        else assert.ok(lo >= dem.meta.min && hi <= dem.meta.max);
        assert.ok(dem.dx <= 10, 'test tiles are 10 m or finer');
    });

    test(`${name}: 48x48 crop runs 150 steps with closed budget on the f64 references`, async t => {
        if (!(await available(name))) { t.skip('tile not fetched'); return; }
        const dem = await loadDem(name), size = 48, x0 = 100, y0 = 100;
        const bed = new Float32Array(size * size);
        for (let y = 0; y < size; y++) bed.set(dem.bed.subarray((y0 + y) * dem.width + x0, (y0 + y) * dem.width + x0 + size), y * size);
        const crop = { ...dem, width: size, height: size, bed, floodwall: false };
        for (const kind of ['lake', 'rain']) {
            const { config, initial } = terrainScenario(crop, kind);
            for (const Solver of [ReferenceSolver, MusclReferenceSolver]) {
                const solver = new Solver(config, initial), volume0 = diagnostics(initial.state, solver.config).volume;
                for (let i = 0; i < 150; i++) solver.step(5);
                const stats = diagnostics(solver.state, solver.config);
                const residual = stats.volume + solver.boundaryOutflow - volume0 - solver.sourceVolume;
                assert.ok(stats.minDepth >= 0);
                assert.ok(Math.abs(residual) <= 1e-10 * Math.max(volume0 + solver.sourceVolume, 1), `${kind} residual ${residual}`);
                if (kind === 'lake') {
                    const change = Math.max(...Array.from({ length: size * size }, (_, i) => Math.abs(solver.state[4 * i] - initial.state[4 * i])));
                    assert.ok(change < 1e-9, `lake depth change ${change}`);
                }
            }
        }
    });
}

test('17th Street Canal: the restored floodwall separates the canal from Lakeview', async t => {
    if (!(await available('neworleans-17thstreet-2m'))) { t.skip('tile not fetched'); return; }
    for (const crop of [undefined, [640, 1084, 640, 640]]) {
        const dem = await loadDem('neworleans-17thstreet-2m', { crop }), { config, initial } = terrainScenario(dem, 'breach', { tracer: true });
        const { width: w, height: h } = config, seen = new Uint8Array(w * h), queue = [];
        // Flood fill through open cells from the west edge (the canal side).
        for (let y = 0; y < h; y++) if (!initial.solid[y * w]) { seen[y * w] = 1; queue.push(y * w); }
        while (queue.length) {
            const i = queue.pop(), x = i % w, y = (i - x) / w;
            for (const j of [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1]) {
                if (j >= 0 && !seen[j] && !initial.solid[j]) { seen[j] = 1; queue.push(j); }
            }
        }
        let reached = 0, sources = 0;
        for (let i = 0; i < w * h; i++) if (initial.source[i] > 0) { sources++; if (seen[i]) reached++; }
        assert.ok(sources > 0 && reached === 0, `${reached} of ${sources} breach cells reachable from the canal`);
        assert.ok(Math.abs(initial.source.reduce((a, v) => a + v, 0) * config.dx * config.dy - 600) < 1e-3, 'breach discharge');
    }
});
