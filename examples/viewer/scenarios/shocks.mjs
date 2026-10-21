// Shocks and jumps: a spillway with a stilling basin, an oblique jump, scour at bridge piles and a
// four-quadrant Riemann problem.
import { obliqueShockCase } from '../../../src/flows.mjs';

// Spillway: reservoir energy level E and toe state for a head over the crest, and the tailwater stage
// h2 as a percentage of the Belanger conjugate depth of the flow at the toe.
export function spillwayLevels({ head, tail }) {
    const g = 9.81, crest = 8, q = Math.sqrt(g) * (2 / 3 * head) ** 1.5, E = crest + head;
    let h1 = 0.1; for (let k = 0; k < 100; k++) h1 = q / Math.sqrt(2 * g * (E - h1)); // energy equation, fixed point
    const fr1 = q / h1 / Math.sqrt(g * h1), conjugate = h1 * (Math.sqrt(1 + 8 * fr1 * fr1) - 1) / 2, h2 = conjugate * tail / 100;
    return { q, E, h1, fr1, conjugate, h2 };
}

// Each entry returns { config, initial, ... } (fields in scenarios/index.mjs).
export const scenarios = {
    // Spillway and stilling basin: a reservoir 2 m over the crest spills down a 1:2.5 chute into a
    // flat basin whose tailwater is set at the Belanger conjugate depth of the flow at the toe (the
    // relation is verified in tests/flows.test.mjs). Toe state from the energy equation (no loss on
    // the chute): E = h1 + q²/(2 g h1²) with critical flow over the crest, q = sqrt(g) (2/3 H)^1.5.
    'hydraulic-jump'({ head, tail }) {
        const d = 0.25, W = 640, Hc = 160, n = W * Hc, crest = 8;
        const { q, E, h1, fr1, conjugate, h2 } = spillwayLevels({ head, tail });
        const bed = new Float32Array(n), state = new Float32Array(4 * n), concentration = new Float32Array(n);
        for (let y = 0; y < Hc; y++) for (let x = 0; x < W; x++) {
            const X = (x + 0.5) * d, i = y * W + x;
            bed[i] = X < 30 ? crest : X < 50 ? crest * (50 - X) / 20 : 0;
            state[4 * i] = Math.max(0, (X < 30 ? E : h2) - bed[i]);
            concentration[i] = X < 30 ? 1 : 0;
        }
        return { config: { width: W, height: Hc, dx: d, dy: d, boundaryX: 'wall', boundaryY: 'wall', tracer: true,
                hydrographs: [{ edge: 'west', kind: 'stage', series: [[0, E]], concentration: 1 }, { edge: 'east', kind: 'stage', series: [[0, h2]] }] },
            initial: { state, bed, manning: new Float32Array(n).fill(0.012), concentration },
            depthScale: 4, wetDepth: 0.01, rate: 2, colour: 'speed', speedScale: 14, concentration: 'linear', view3d: true, exaggeration: 1.5,
            camera: { distance: 150, yaw: -2.15, pitch: 0.42, target: [-10, 0, 0] },
            note: `A reservoir ${head} m above the crest spills down a steep chute into a stilling basin. At the toe the water is ${h1.toFixed(2)} m deep and moves at ${(q / h1).toFixed(1)} m/s (Froude number ${fr1.toFixed(1)}); the Belanger conjugate depth is ${conjugate.toFixed(2)} m and the basin's tailwater is held at ${h2.toFixed(2)} m (${tail} %). At 100 % the jump stands at the foot of the chute; lower tailwater sweeps it out across the basin, higher tailwater drowns it up the chute (40 m wide, 0.25 m cells; colour: speed).` };
    },
    // Oblique hydraulic jump at a wall (tests/flows.test.mjs: Ippen's angle to 0.7 %, depth ratio exact).
    'oblique-shock'({ fr, theta }) {
        let c;
        try { c = obliqueShockCase(0.25, { fr1: fr, theta: theta * Math.PI / 180 }); }
        catch { throw new Error(`At Froude number ${fr} a ${theta}° wall exceeds the attached-shock limit; lower the angle or raise the Froude number.`); }
        return { ...c, depthScale: 1, wetDepth: 0.01, rate: 2, colour: 'depth',
            note: `Supercritical flow (Froude number ${fr}) arrives turned ${theta}° toward a wall that starts 20 m in; the wall turns it back, and an oblique jump leaves its leading corner at ${((c.beta - c.theta) * 180 / Math.PI).toFixed(1)}° (Ippen) with the depth rising ${c.ratio.toFixed(2)} times. The shock starts at 25° and settles to Ippen's angle as the flow sweeps the initial error away. Periodic in y, 0.25 m cells.` };
    },
    // Bridge with four round piles in a gravel-bed river. The approach flow is below the threshold of
    // motion; past the piles the flow speeds up and the bed erodes beside and between them, depositing
    // downstream (bed evolution: MORPH, accelerated by a morphological factor).
    'bridge-scour'({ U }) {
        const d = 0.5, W = 600, H = 160, n = W * H, h0 = 2, D = 3;
        const bed = new Float32Array(n), state = new Float32Array(4 * n), solid = new Uint8Array(n), shape = new Float32Array(n).fill(99);
        const piles = [16, 32, 48, 64].map(y => [100, y]);
        const theta = 0.025 ** 2 * U * U / (h0 ** (1 / 3) * 1.65 * 0.005); // Shields number of the approach flow
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const i = y * W + x, X = (x + 0.5) * d, Y = (y + 0.5) * d;
            const r = Math.min(...piles.map(([px, py]) => Math.hypot(X - px, Y - py))) - D / 2;
            shape[i] = r / d; solid[i] = r < 0 ? 1 : 0;
            if (!solid[i]) state.set([h0, h0 * U, 0, 0], 4 * i);
        }
        return { config: { width: W, height: H, dx: d, dy: d, boundaryX: 'wall', boundaryY: 'wall',
                hydrographs: [{ edge: 'west', kind: 'discharge', series: [[0, U * h0 * H * d]] }, { edge: 'east', kind: 'stage', series: [[0, h0]] }] },
            initial: { state, bed, solid, manning: new Float32Array(n).fill(0.025) }, shape, inflow: U,
            morph: { d: 0.005, s: 2.65, porosity: 0.4, thetaC: 0.047, beta: 1.5, repose: 0.6, factor: 200, maxStep: 4 },
            cylinders: piles.map(([x, y]) => ({ x, y, r: D / 2 })), cylinderTop: h0 + 1.5, buildingHeight: 0,
            depthScale: 2.5, wetDepth: 0.01, rate: 2, colour: 'bed', elevationScale: 0.3, speedScale: 2.5 * U, contour: 0.1,
            view3d: true, exaggeration: 3, buildingHeight: 3, camera: { target: [-50, 0, 0], distance: 130, yaw: -2.3, pitch: 0.7 },
            note: `A river 80 m wide and 2 m deep flows at ${U} m/s over 5 mm gravel past a bridge on four 3 m piles. Upstream the Shields number is ${theta.toFixed(3)}, ${theta < 0.047 ? 'below' : 'just above'} the threshold of motion (0.047); the flow speeds up past the piles and the gravel moves much faster there (bedload of Meyer-Peter and Mueller), and holes open beside the piles while the sediment settles downstream (contours every 10 cm). Bed time runs 200 times faster than the flow. Depth-averaged: the horseshoe vortex that digs the deepest local scour at a pile front is three-dimensional and not represented.` };
    },
    // Four-quadrant Riemann problem: four dam breaks at once; shocks, rarefactions and their
    // interaction form a symmetric 2D pattern.
    riemann({ ratio }) {
        const d = 0.25, W = 400, n = W * W, state = new Float32Array(4 * n);
        for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) state[4 * (y * W + x)] = (x < W / 2) === (y < W / 2) ? 2 : 2 * ratio;
        return { config: { width: W, height: W, dx: d, dy: d, boundaryX: 'open', boundaryY: 'open' }, initial: { state, bed: new Float32Array(n) },
            depthScale: 2, wetDepth: 0.01, rate: 0.5, colour: 'depth', view3d: true, exaggeration: 4, hideGround: true, speedScale: 3,
            note: `Four quadrants of still water, 2 m and ${(2 * ratio).toFixed(1)} m deep in a checkerboard, are released at once: four dam breaks whose bores and rarefactions meet in the middle and form a symmetric 2D pattern, a standard test of shock capturing. Half speed; depth shown ×4.` };
    },
};
