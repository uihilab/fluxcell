// Canvas size and the View uniform (shaders/view.wgsl) shared by the 2D and 3D shaders.
import { app } from '../app.mjs';
import { el } from '../dom.mjs';
import { zoom } from '../camera.mjs';

export const MODES = { depth: 0, speed: 1, vorticity: 2, concentration: 3, elevation: 4, bed: 5 };

// Fits the canvas to its grid cell and renders at the displayed size in device pixels (capped), not
// at the cell resolution; then writes the View uniform.
export function resize() {
    const { device, renderer } = app;
    if (!renderer?.viewData) return;
    const { config, zMin, zMax, depthScale, wetDepth, tracer, logScale, speedScale, vorticityScale, still, elevationScale, relief, datum } = renderer.viewData;
    // The canvas fits the map's cell. On narrow screens the cell follows the map, so only the width
    // counts. A hidden page can report a zero width; at least one pixel keeps the swap chain valid.
    // On phones (the page scrolls) the map is as wide as the screen, square in 3D, and at most 72 % of
    // the screen's height.
    const work = el.work, fixed = innerWidth > 860, room = fixed ? work.clientHeight : 0.72 * innerHeight;
    // In 3D the canvas fills the cell (no taller than 1 : 1.1); in 2D it keeps the domain's aspect.
    const aspect = renderer.mode3d ? (fixed ? Math.max(work.clientWidth / Math.max(room, 1), 1.1) : 1) : config.width / config.height;
    const cssWidth = Math.max(1, Math.min(work.clientWidth, Math.max(200, room * aspect)));
    const canvas = el.canvas;
    canvas.style.width = `${cssWidth}px`;
    const width = Math.max(1, Math.min(2400, Math.round(cssWidth * devicePixelRatio)));
    canvas.width = width; canvas.height = Math.max(1, Math.round(width / aspect));
    if (renderer.depth?.width !== canvas.width || renderer.depth?.height !== canvas.height) {
        renderer.depth?.destroy();
        renderer.depth = device.createTexture({ size: [canvas.width, canvas.height], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
    const view = new ArrayBuffer(112), u = new Uint32Array(view), f = new Float32Array(view);
    u.set([config.width, config.height]);
    f.set([canvas.width, canvas.height, zMin, zMax, depthScale, wetDepth, config.dx ?? 1, config.dy ?? config.dx ?? 1, tracer ? 1 : 0], 2);
    u[11] = MODES[el.colour.value]; u[12] = logScale ? 1 : 0; f[13] = speedScale; f[14] = vorticityScale;
    f[15] = (still ?? 0) - datum; f[16] = elevationScale; f[17] = relief; // the bed on the GPU is z - datum
    f.set([zoom.k, zoom.cx, zoom.cy, performance.now() / 1000 % 3600, ...renderer.viewData.wind, renderer.viewData.contour], 18);
    device.queue.writeBuffer(renderer.view, 0, view);
}

// The zoom window alone (2D pan and zoom), without resizing.
export function writeZoom() {
    const { device, renderer } = app;
    if (renderer?.viewData) device.queue.writeBuffer(renderer.view, 72, new Float32Array([zoom.k, zoom.cx, zoom.cy]));
}
