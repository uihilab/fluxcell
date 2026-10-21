// Playground tools. While the left button is down, the cells within the brush radius receive water
// (a source of the chosen discharge, clean or dyed with concentration 1) or have their bed raised or
// dug (solver.setBed). Rain adds 50 mm/h everywhere and carries no tracer. Applied once per frame.
import { app } from './app.mjs';
import { el } from './dom.mjs';
import { pick, toDomain } from './camera.mjs';

export const playground = { tool: 'water', size: 1 }; // set by the Tool and Size rows (ui/controls.mjs)
const brush = { active: false, x: 0, y: 0 };          // centre in cell units
let lastBrush = performance.now();

function moveBrush(event) {
    const { config } = app.renderer.viewData;
    const [x, y] = app.renderer.mode3d ? pick(event) ?? [brush.x / config.width, brush.y / config.height] : toDomain(event);
    brush.x = x * config.width; brush.y = y * config.height;
}
el.canvas.addEventListener('pointerdown', event => {
    if (!app.renderer?.viewData?.sandbox || event.button !== 0 || !event.isPrimary) return; // a second finger navigates
    if (app.renderer.mode3d && !pick(event)) return; // in 3D the tool acts only on the terrain
    brush.active = true; el.canvas.setPointerCapture(event.pointerId); moveBrush(event);
});
el.canvas.addEventListener('pointermove', event => { if (brush.active) moveBrush(event); });
for (const type of ['pointerup', 'pointercancel']) el.canvas.addEventListener(type, () => { brush.active = false; });
export function stopBrush() { brush.active = false; } // a two-finger gesture began
const showFlow = () => { el.flowValue.textContent = `${(10 ** Number(el.flow.value)).toPrecision(2)} m³/s`; };
el.flow.addEventListener('input', showFlow); showFlow();

// Brush cells within radius r (cells) of the brush centre, with a smooth weight (1 at the centre).
function brushCells(config, r) {
    const w = config.width, h = config.height, cx = Math.round(brush.x - 0.5), cy = Math.round(brush.y - 0.5), out = [];
    for (let y = Math.max(0, cy - r); y <= Math.min(h - 1, cy + r); y++) for (let x = Math.max(0, cx - r); x <= Math.min(w - 1, cx + r); x++) {
        const d2 = ((x - cx) ** 2 + (y - cy) ** 2) / (r * r);
        if (d2 <= 1 && !app.solver.forcing.solid[y * w + x]) out.push([y * w + x, 0.5 + 0.5 * Math.cos(Math.PI * Math.sqrt(d2))]);
    }
    return out;
}

export function applyBrush() {
    const { solver } = app, view = app.renderer.viewData;
    if (!view?.sandbox || solver.busy) return;
    const { config } = view, w = config.width, h = config.height;
    const r = Math.max(2, Math.round(0.04 * Math.min(w, h) * playground.size)); // brush radius in cells
    const now = performance.now(), wall = Math.min(0.1, (now - lastBrush) / 1000); lastBrush = now;
    const sculpt = playground.tool === 'raise' || playground.tool === 'dig';
    if (brush.active && sculpt) {
        // Raise or dig at 30 % of the starting relief per second at the centre (0.3 to 3 m/s), within
        // 10 m of that relief; the water above keeps its depth.
        const sign = playground.tool === 'raise' ? 1 : -1, datum = solver.datum ?? 0;
        const rate = Math.min(3, Math.max(0.3, 0.3 * (view.zMax - view.zMin)));
        const cells = brushCells(config, r), lo = view.zMin + datum - 10, hi = view.zMax + datum + 10;
        solver.setBed(cells.map(([i]) => i), cells.map(([i, weight]) => Math.min(hi, Math.max(lo, solver.bedValues[i] + datum + sign * rate * wall * weight))));
    }
    const rain = el.rain.checked ? 50e-3 / 3600 : 0, dyed = playground.tool === 'dye', q = 10 ** Number(el.flow.value);
    const pouring = brush.active && !sculpt;
    // Sources change only when the brush moves or the settings change.
    const key = pouring ? [Math.round(brush.x), Math.round(brush.y), r, q, dyed, rain].join() : `off ${rain}`;
    if (key === view.brushKey) return;
    view.brushKey = key;
    const source = new Float32Array(w * h).fill(rain), sourceConcentration = new Float32Array(w * h), cells = [];
    if (pouring) {
        const cx = Math.round(brush.x - 0.5), cy = Math.round(brush.y - 0.5);
        for (let y = Math.max(0, cy - r); y <= Math.min(h - 1, cy + r); y++) for (let x = Math.max(0, cx - r); x <= Math.min(w - 1, cx + r); x++) {
            const i = y * w + x;
            if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r && !solver.forcing.solid[i]) cells.push(i);
        }
        for (const i of cells) {
            source[i] += q / (cells.length * config.dx * config.dy);
            sourceConcentration[i] = dyed ? 1 : 0;
        }
    }
    solver.setForcing({ source, sourceConcentration });
}
