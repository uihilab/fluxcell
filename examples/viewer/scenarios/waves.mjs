// Wave demos: Grand Isle (NOAA CUDEM bed, OpenStreetMap buildings), the NTHMP laboratory benchmarks,
// and classic wave phenomena driven through incident-wave boundaries.
import { conicalIslandCase, monaiCaseFromBed } from '../../../src/benchmarks.mjs';

// Incident sine wave about the still level: WAVE_POINTS points, 16 per period, from time t0 with phase
// phase0 (ramped in over two periods when starting from rest). A fixed point count keeps the
// hydrograph table the same size, so a live change of period can replace it (setHydrographs).
export const WAVE_POINTS = 4096;
export function sineSeries(still, A, T, t0 = 0, phase0 = 0, ramp = true) {
    const series = [];
    for (let k = 0; k < WAVE_POINTS; k++) {
        const t = t0 + k * T / 16, r = ramp ? Math.min(1, (t - t0) / (2 * T)) : 1;
        series.push([t, still + A * r * Math.sin(phase0 + 2 * Math.PI * (t - t0) / T)]);
    }
    return series;
}

async function grandIsle({ still, A, T, note, view3d = false }) {
    const [meta, bedBytes, solidBytes, shapeBytes] = await Promise.all([fetch('./data/grandisle.json').then(r => r.json()),
        ...['bed.f32', 'solid.u8', 'shape.i8'].map(f => fetch(`./data/grandisle-${f}`).then(r => r.arrayBuffer()))]);
    const { width: W, height: H, dx } = meta, n = W * H, bed = new Float32Array(bedBytes), solid = new Uint8Array(solidBytes);
    const state = new Float32Array(4 * n), manning = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        manning[i] = bed[i] > still ? 0.03 : 0.02;
        if (!solid[i]) state[4 * i] = Math.max(0, still - bed[i]);
    }
    const series = sineSeries(still, A, T);
    return { wave: { index: 0, still, A, T, t0: 0, phase0: 0 }, config: { width: W, height: H, dx, dy: dx, boundaryX: 'wall', boundaryY: 'wall',
        hydrographs: [{ edge: 'south', kind: 'wave', series, still }, { edge: 'north', kind: 'wave', series: [[0, still]], still }] },
        initial: { state, bed, manning, solid }, depthScale: 5, wetDepth: 0.02, rate: 4, colour: 'elevation', still,
        elevationScale: A, relief: 12, speedScale: 2.5, note, view3d, exaggeration: 8, buildingHeight: 3, elevationDatum: 'm NAVD88',
        shape: Float32Array.from(new Int8Array(shapeBytes), d => d / 16) }; // outline distance in cells
}


// Each entry returns { config, initial, ... } (fields in scenarios/index.mjs).
export const scenarios = {
    // Grand Isle, Louisiana: NOAA CUDEM topobathymetry on a coast-aligned 3 m grid with the
    // OpenStreetMap buildings and breakwaters as solid cells (tools/dem/fetch_grandisle.py,
    // grandisle_grid.py). Waves enter through the Gulf (south) edge, an incident-wave boundary
    // that also lets reflected waves leave; the bay (north) edge lets waves leave; the cross-shore
    // edges are walls. Levels in m NAVD88.
    async 'grandisle-ida'({ H, T }) { return grandIsle({ still: 1.638, A: H / 2, T, view3d: true,
        note: `Grand Isle, Louisiana, at the peak still-water level recorded during Hurricane Ida (+1.64 m NAVD88, NOAA station 8761724, 29 August 2021), with ${H} m, ${T} s waves from the Gulf: an illustration, not a reconstruction of the storm. The dune crest stands near +4 m, about 2.4 m above the surge: the larger waves break over the breakwaters and run up the beach, and the biggest overtop the dune into the town. Lower the wave height to see the dune hold. Bed: NOAA CUDEM topobathymetry; buildings and breakwaters: © OpenStreetMap contributors.` }); },
    // NTHMP laboratory benchmarks (src/benchmarks.mjs; comparison with the measurements in,
    // docs/VERIFICATION.md).
    'conical-island'() {
        return { ...conicalIslandCase(0.05, 'C'), depthScale: 0.4, wetDepth: 0.001, rate: 0.5, colour: 'elevation', view3d: true, exaggeration: 4,
            still: 0.32, elevationScale: 0.08, relief: 10, speedScale: 0.8,
            note: 'NTHMP benchmark problem 6 (Briggs et al., 1995): a solitary wave (H/d = 0.181) in a 0.32 m deep basin splits around a conical island and the two fronts collide behind it. Laboratory scale (5 cm cells; the benchmark runs use 2.5 and 1.25 cm), shown at half speed.' };
    },
    async monai() {
        const [meta, bytes] = await Promise.all([fetch('./data/monai.json').then(r => r.json()), fetch('./data/monai-bed.f32').then(r => r.arrayBuffer())]);
        return { ...monaiCaseFromBed(new Float32Array(bytes), meta.incident), depthScale: 0.15, wetDepth: 0.0005, rate: 0.5,
            colour: 'elevation', still: 0, elevationScale: 0.03, relief: 6, speedScale: 0.6,
            note: 'NTHMP benchmark problem 7: the measured incident wave runs up a 1:400 laboratory model of the Monai Valley coast (Liu et al., 2008); the highest runup is in the narrow valley at the right. Shown at half speed.' };
    },
    // Waves through two gaps in a wall: incident waves enter on the west (wave boundary), the other
    // edges let waves leave; beyond the wall the two diffracted waves interfere.
    'double-slit'({ T, sep }) {
        const d = 0.5, W = 400, H = 300, n = W * H, h0 = 1, A = 0.03, c = Math.sqrt(9.81 * h0), L = c * T;
        const bed = new Float32Array(n), state = new Float32Array(4 * n), solid = new Uint8Array(n), wall = 120, gap = 6;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const i = y * W + x, Y = (y + 0.5) * d - 75;
            const open = Math.abs(Math.abs(Y) - sep / 2) < gap / 2;
            solid[i] = (x === wall || x === wall + 1) && !open ? 1 : 0;
            state[4 * i] = solid[i] ? 0 : h0;
        }
        return { config: { width: W, height: H, dx: d, dy: d, hydrographs: [
                { edge: 'west', kind: 'wave', series: sineSeries(h0, A, T), still: h0 }, { edge: 'east', kind: 'wave', series: [[0, h0]], still: h0 },
                { edge: 'south', kind: 'wave', series: [[0, h0]], still: h0 }, { edge: 'north', kind: 'wave', series: [[0, h0]], still: h0 }] },
            initial: { state, bed, solid }, wave: { index: 0, still: h0, A, T, t0: 0, phase0: 0 }, depthScale: 2, wetDepth: 0.01, rate: 4, colour: 'elevation', still: h0, elevationScale: A,
            particles: false, note: `Waves ${L.toFixed(1)} m long (period ${T} s, 1 m deep) pass two ${gap} m gaps ${sep} m apart. Beyond the wall the two diffracted waves add where their paths differ by whole wavelengths and cancel in between: the bright fringes fan out at angles whose sines are multiples of ${(L / sep).toFixed(2)} (wavelength / spacing). Smaller spacing or longer waves spread them.` };
    },
    // A submerged shoal as a lens: waves slow over the shallow crest (c = sqrt(g h)) and their crests
    // bend toward it, focusing energy behind the shoal.
    shoal({ T, height }) {
        const d = 0.5, W = 400, H = 300, n = W * H, h0 = 2, A = 0.04, L = Math.sqrt(9.81 * h0) * T;
        const bed = new Float32Array(n), state = new Float32Array(4 * n);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const i = y * W + x, r = Math.hypot((x + 0.5) * d - 60, (y + 0.5) * d - 75);
            bed[i] = r < 30 ? height * Math.cos(Math.PI * r / 60) ** 2 : 0;
            state[4 * i] = h0 - bed[i];
        }
        return { config: { width: W, height: H, dx: d, dy: d, hydrographs: [
                { edge: 'west', kind: 'wave', series: sineSeries(h0, A, T), still: h0 }, { edge: 'east', kind: 'wave', series: [[0, h0]], still: h0 },
                { edge: 'south', kind: 'wave', series: [[0, h0]], still: h0 }, { edge: 'north', kind: 'wave', series: [[0, h0]], still: h0 }] },
            initial: { state, bed }, wave: { index: 0, still: h0, A, T, t0: 0, phase0: 0 }, depthScale: 2.5, wetDepth: 0.01, rate: 4, colour: 'elevation', still: h0, elevationScale: 2 * A,
            particles: false, view3d: true, exaggeration: 1, surfaceGain: 30,
            note: `Long waves (${L.toFixed(0)} m, period ${T} s) in water 2 m deep cross a round shoal rising ${height} m. Over the shoal they travel slower (c = sqrt(g h)), so the crests bend inward like light through a lens and the waves grow where the rays meet, behind the shoal. Shallow-water waves only (no dispersion); surface shown ×30.` };
    },
    // Harbour resonance: a narrow harbour (50 m x 500 m, closed at its head) opens onto a wide sea
    // forced by waves. The sudden widening at the mouth reflects waves back into the harbour, so it
    // rings at its quarter-wave periods, T_n = 4 L / ((2n - 1) c) (plus a mouth correction).
    seiche({ T }) {
        const d = 10, W = 110, H = 150, n = W * H, h0 = 5, A = 0.05, Lh = 500, c = Math.sqrt(9.81 * h0), T1 = 4 * Lh / c;
        // Land 2 m above the still water (bed +2 m), sea and harbour 5 m deep (bed -5 m); still level 0.
        const bed = new Float32Array(n), state = new Float32Array(4 * n);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const i = y * W + x, land = x >= 60 && (y < 72 || y >= 77);
            bed[i] = land ? 2 : -h0; state[4 * i] = land ? 0 : h0;
        }
        const still = { kind: 'wave', series: [[0, 0]], still: 0 };
        return { config: { width: W, height: H, dx: d, dy: d, hydrographs: [{ edge: 'west', kind: 'wave', series: sineSeries(0, A, T), still: 0 },
                { ...still, edge: 'south', start: 0, end: 60 }, { ...still, edge: 'north', start: 0, end: 60 }] },
            initial: { state, bed }, wave: { index: 0, still: 0, A, T, t0: 0, phase0: 0 },
            gaugeCell: 74 * W + W - 1, depthScale: 6, wetDepth: 0.01, rate: 60, colour: 'elevation', still: 0, elevationScale: 0.2,
            view3d: true, exaggeration: 4, surfaceGain: 100, camera: { distance: 1500, yaw: -2.4, pitch: 0.6, target: [150, 0, 0] },
            note: `A harbour 50 m wide and 500 m long, 5 m deep and closed at its head, opens onto a wide sea where 5 cm waves of period ${T} s arrive. The narrow mouth reflects waves back in, so the harbour rings: near its natural periods (about ${T1.toFixed(0)} s and ${(T1 / 3).toFixed(0)} s for a quarter-wave basin, a little longer with the mouth) the water at the head swings far higher than the waves outside; away from them it barely moves. Slide the period through resonance. One minute per second; surface shown ×400.` };
    },
    // A coastal Kelvin wave: a bump of water against the coast of a rotating square basin travels
    // around it with the coast on its right (northern hemisphere), trapped within a Rossby radius.
    'kelvin-wave'({ lat }) {
        const d = 5000, W = 200, n = W * W, h0 = 100, f = 2 * 7.2921e-5 * Math.sin(lat * Math.PI / 180);
        const state = new Float32Array(4 * n);
        for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
            const i = y * W + x, r2 = (((x + 0.5) * d) ** 2 + ((y + 0.5 - W / 2) * d) ** 2) / (120e3) ** 2;
            state[4 * i] = h0 + Math.exp(-r2);
        }
        const R = f > 0 ? Math.sqrt(9.81 * h0) / f : Infinity;
        return { config: { width: W, height: W, dx: d, dy: d, coriolis: f }, initial: { state, bed: new Float32Array(n) },
            depthScale: 100, wetDepth: 0.01, rate: 3600, colour: 'elevation', still: h0, elevationScale: 0.4, hideGround: true,
            view3d: true, exaggeration: 1, surfaceGain: 50000, speedScale: 0.05,
            note: `A 1 m bump of water against the west coast of a 1,000 km basin, 100 m deep, at ${lat}° N${f > 0 ? ` (Rossby radius ${(R / 1000).toFixed(0)} km)` : ''}. With rotation it runs around the basin as a Kelvin wave, coast on its right, at sqrt(g h) = ${Math.sqrt(9.81 * h0).toFixed(0)} m/s, once round in about ${(4e6 / Math.sqrt(9.81 * h0) / 3600).toFixed(0)} h; at the equator it spreads as a ring instead. One hour per second.` };
    },
};
