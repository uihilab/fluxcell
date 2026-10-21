// Wind-driven circulation and a settling sediment plume.

// Each entry returns { config, initial, ... } (fields in scenarios/index.mjs).
export const scenarios = {
    // Wind-driven circulation: an elliptical lake (2 km x 1 km, 6 m deep, parabolic sides) under a
    // steady 15 m/s west wind (Wu 1982 drag). Downwind flow over the shallow sides and an upwind
    // return current over the deep axis form two gyres; the surface tilts up toward the east shore.
    'wind-lake'({ speed, from }) {
        const to = (from + 180) * Math.PI / 180, wind = [speed * Math.sin(to), speed * Math.cos(to)];
        const W = 200, H = 100, d = 10, n = W * H, depth = 6, still = 6;
        const bed = new Float32Array(n), state = new Float32Array(4 * n), concentration = new Float32Array(n);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const i = y * W + x, ex = ((x + 0.5) * d - 1000) / 950, ey = ((y + 0.5) * d - 500) / 450, r2 = ex * ex + ey * ey;
            bed[i] = depth * Math.min(1.2, r2) + 0.3 * Math.sin(x / 7) * Math.sin(y / 5) * r2;
            state[4 * i] = Math.max(0, still - bed[i]);
            if (((x + 0.5) * d - 450) ** 2 + ((y + 0.5) * d - 300) ** 2 < 140 ** 2 && state[4 * i] > 0) concentration[i] = 1;
        }
        return { config: { width: W, height: H, dx: d, dy: d, wind, tracer: true },
            initial: { state, bed, manning: new Float32Array(n).fill(0.025), concentration },
            depthScale: 6, wetDepth: 0.01, rate: 120, colour: 'concentration', concentration: 'log', speedScale: 0.3,
            view3d: true, exaggeration: 20, surfaceGain: 400, still, elevationScale: 0.02, camera: { distance: 2400 },
            note: `A 2 km lake, 6 m deep, under a steady ${speed} m/s wind (direction on the right; drag of Wu, 1982). Water runs downwind over the shallow sides and returns upwind along the deep axis, forming two gyres; the surface sets up toward the east shore (shown ×400). A dye patch marks the circulation. The darker gust patches drifting across the water show the wind; they are drawn, not simulated.` };
    },
    'mud-plume'({ k, Q }) {
        const W = 240, H = 150, d = 10, n = W * H, still = 4;
        const bed = new Float32Array(n), state = new Float32Array(4 * n), solid = new Uint8Array(n);
        const lo = 67, hi = 82; // channel rows (150 m)
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const i = y * W + x, inChannel = y >= lo && y < hi;
            const shore = Math.min(y, H - 1 - y) * d;                      // distance to the north/south shores
            let z = Math.max(0, 3.2 - shore / 60) + 0.15 * Math.sin(x / 9) * Math.cos(y / 7);
            if (x < 40) z = inChannel ? -1 + 0.002 * (40 - x) * d / 10 : 5;   // banks of the river channel
            bed[i] = z; state[4 * i] = Math.max(0, still - z);
        }
        const series = [[0, 0], [600, Q]];
        return { config: { width: W, height: H, dx: d, dy: d, tracer: true, boundaryX: 'outflow',
                settling: k > 0 ? { velocity: k * 1e-3, exponent: 4 / 3 } : null,
                hydrographs: [{ edge: 'west', kind: 'discharge', start: lo, end: hi, series, concentration: 1 },
                    { edge: 'east', kind: 'stage', series: [[0, still]] }] },
            initial: { state, bed, manning: new Float32Array(n).fill(0.025), solid, concentration: new Float32Array(n) },
            depthScale: 5, wetDepth: 0.01, rate: 90, colour: 'concentration', concentration: 'log', speedScale: 0.6,
            view3d: true, exaggeration: 25, still, elevationScale: 0.05,
            note: 'A river carrying 1 kg/m³ of suspended mud (400 m³/s, ramped over 10 min) enters a 4 m deep basin. The mud flocculates and settles at w_s = k c^(4/3) (Mehta, 1986; k illustrative, 1 mm/s at 1 kg/m³), so the plume thins downstream as sediment drops out; settled mass leaves the water column.' };
    },
};
