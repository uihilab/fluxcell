import test from 'node:test';
import assert from 'node:assert/strict';
import { ReferenceSolver } from '../src/reference.mjs';
import { MusclReferenceSolver } from '../src/muscl-reference.mjs';
import { fuzzScenario } from '../src/fuzz.mjs';

// Seeded stress cases (thin films, cliffs, buildings, all boundaries, sources, offsets) run
// through both f64 references. Gates: no failure, finite nonnegative state, closed budget.
const SEEDS = Array.from({ length: 60 }, (_, i) => i + 1);

for (const [label, Solver] of [['Rusanov', ReferenceSolver], ['MUSCL', MusclReferenceSolver]]) {
    test(`${label}: ${SEEDS.length} seeded stress cases stay valid and conserve water`, () => {
        let retries = 0;
        for (const seed of SEEDS) {
            const c = fuzzScenario(seed, { maxCells: 20 * 20, steps: 200 });
            const solver = new Solver(c.config, c.initial), area = solver.config.dx * solver.config.dy;
            let volume0 = 0;
            for (let k = 0; k < c.initial.state.length; k += 4) volume0 += c.initial.state[k] * area;
            assert.doesNotThrow(() => { for (let i = 0; i < c.steps; i++) solver.step(10); }, `seed ${seed}: ${c.description}`);
            let volume = 0;
            for (let k = 0; k < solver.state.length; k += 4) {
                assert.ok(solver.state[k] >= 0 && Number.isFinite(solver.state[k + 1] + solver.state[k + 2]), `seed ${seed}`);
                volume += solver.state[k] * area;
            }
            const residual = volume + solver.boundaryOutflow - volume0 - solver.sourceVolume;
            const scale = Math.max(volume0 + solver.sourceVolume, volume, Math.abs(solver.boundaryOutflow), 1e-30);
            assert.ok(Math.abs(residual) <= 1e-10 * scale, `seed ${seed}: residual ${residual} of ${scale}; ${c.description}`);
            retries += solver.counters?.retries ?? 0;
        }
        console.log(`${label}: ${SEEDS.length} cases, ${retries} retries`);
    });
}
