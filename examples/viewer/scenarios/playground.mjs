// Playgrounds: terrain the user sculpts and floods (brush tools in brush.mjs).
import { loadDem, terrainScenario } from '../../../src/terrain.mjs';

// Each entry returns { config, initial, ... } (fields in scenarios/index.mjs).
export const scenarios = {
    // Playground: sculpt the ground (raise, dig) and pour clean or dyed water, live, in 2D or 3D.
    // A sand table with contour lines (0.5 m cells), or the Lakeview, New Orleans 3DEP 10 m tile.
    'playground-sand'() {
        const W = 320, H = 240, d = 0.5, n = W * H;
        const bed = new Float32Array(n), state = new Float32Array(4 * n);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const i = y * W + x, X = x / W, Y = y / H;
            const rim = Math.max(0, Math.max(Math.abs(X - 0.5), Math.abs(Y - 0.5)) - 0.42) * 30;
            bed[i] = 1.2 * Math.exp(-((X - 0.3) ** 2 + (Y - 0.65) ** 2) / 0.02) + 0.8 * Math.exp(-((X - 0.72) ** 2 + (Y - 0.3) ** 2) / 0.03)
                - 0.9 * Math.exp(-((X - 0.62) ** 2 + (Y - 0.7) ** 2) / 0.015) + 0.6 * (1 - Y) + 0.12 * Math.sin(x / 9) * Math.sin(y / 11) + rim;
        }
        return { config: { width: W, height: H, dx: d, dy: d, boundaryX: 'outflow', boundaryY: 'outflow', tracer: true },
            initial: { state, bed, manning: new Float32Array(n).fill(0.03) }, depthScale: 0.6, wetDepth: 0.005, rate: 2,
            sandbox: true, concentration: 'log', contour: 0.1, view3d: true, exaggeration: 6, flow: Math.log10(2),
            note: 'A 160 m x 120 m sand table (0.5 m cells, contours every 10 cm). Choose a tool on the right, then drag on the map: raise or dig the ground, pour clean or dyed water, or let it rain. In 3D, the left button uses the tool and the right button turns the view.' };
    },
    async 'playground-neworleans'() {
        const s = terrainScenario(await loadDem('neworleans-lakeview'), 'rain');
        return { config: { ...s.config, rain: 0, tracer: true }, initial: s.initial, depthScale: 1.5, wetDepth: 0.01, rate: 60,
            sandbox: true, concentration: 'log', contour: 1, view3d: true, exaggeration: 25, elevationDatum: 'm NAVD88',
            note: 'USGS 3DEP 10 m, Lakeview, New Orleans, much of it below sea level (contours every 1 m). Raise levees, dig channels, pour water and see where it collects.' };
    },
};
