// Viewpoints: the 2D zoom window (zoom.k and its centre, as fractions of the domain) and the 3D orbit
// camera, with the view-projection written to the scene uniform and pointer picking on the ground.
import { app } from './app.mjs';
import { el } from './dom.mjs';

export const zoom = { k: 1, cx: 0.5, cy: 0.5, max: 32 };
export const camera = { yaw: -2.2, pitch: 0.62, distance: 1, target: [0, 0, 0], home: 1 };

// Clamps the zoom window to the domain (the caller writes it to the GPU: render/view.mjs writeZoom).
export function setZoomWindow(k, cx = zoom.cx, cy = zoom.cy) {
    zoom.k = Math.min(zoom.max, Math.max(1, k));
    const half = 0.5 / zoom.k;
    zoom.cx = Math.min(1 - half, Math.max(half, cx)); zoom.cy = Math.min(1 - half, Math.max(half, cy));
}

// Domain fractions (x east, y north) of a pointer position in the 2D map.
export function toDomain(event) {
    const rect = el.canvas.getBoundingClientRect();
    return [zoom.cx + ((event.clientX - rect.left) / rect.width - 0.5) / zoom.k,
        zoom.cy + (0.5 - (event.clientY - rect.top) / rect.height) / zoom.k];
}

// The whole domain, with its relief, in view; long domains are seen from their long side, the others
// from the south-west. A demo can aim the camera (scenario.camera, target in metres from the centre).
export function resetCamera() {
    const sc = app.renderer.scene;
    const home = 0.95 * Math.hypot(sc.Lx, sc.Ly, sc.relief * sc.ex);
    const long = sc.Lx > 1.8 * sc.Ly || sc.Ly > 1.8 * sc.Lx;
    const yaw = sc.Lx > 1.8 * sc.Ly ? -1.75 : sc.Ly > 1.8 * sc.Lx ? -0.2 : -2.2;
    Object.assign(camera, { yaw, pitch: 0.62, distance: long ? 0.75 * home : home, target: [0, 0, 0], home }, sc.camera ?? {});
}

const eyeOf = c => [c.target[0] + c.distance * Math.cos(c.pitch) * Math.cos(c.yaw), c.target[1] + c.distance * Math.cos(c.pitch) * Math.sin(c.yaw),
    c.target[2] + c.distance * Math.sin(c.pitch)];

// Scene uniform (shaders/scene3d.wgsl Scene): column-major view-projection, eye, display parameters
// and the mesh layout; and the cylinders, whose heights are in scene units.
export function writeScene() {
    const { device, solver, renderer } = app, sc = renderer.scene, aspect = el.canvas.width / el.canvas.height, c = camera;
    const eye = eyeOf(c);
    const f = norm(sub(c.target, eye)), side = norm(cross(f, [0, 0, 1])), up = cross(side, f);
    const near = c.distance * 0.01, far = c.distance * 6, t = 1 / Math.tan(0.35);
    const V = [side[0], side[1], side[2], -dot(side, eye), up[0], up[1], up[2], -dot(up, eye), -f[0], -f[1], -f[2], dot(f, eye), 0, 0, 0, 1];
    const P = [t / aspect, 0, 0, 0, 0, t, 0, 0, 0, 0, far / (near - far), near * far / (near - far), 0, 0, -1, 0];
    const M = new Float32Array(28);
    for (let col = 0; col < 4; col++) for (let row = 0; row < 4; row++) {
        let m = 0; for (let k = 0; k < 4; k++) m += P[row * 4 + k] * V[k * 4 + col];
        M[col * 4 + row] = m;
    }
    M.set([...eye, 1, sc.ex, sc.buildingHeight, sc.zCentre, sc.surfaceGain], 16);
    new Uint32Array(M.buffer).set([sc.stride, sc.nxv, sc.nyv, sc.hideGround ? 1 : 0], 24);
    device.queue.writeBuffer(renderer.sceneUniform, 0, M);
    const cy = renderer.cylinders, datum = solver.datum ?? 0;
    if (cy.count) device.queue.writeBuffer(cy.buffer, 0, Float32Array.from(cy.spec.flatMap(([x, y, r, bottom, top]) =>
        [x, y, r, (bottom - datum - sc.zCentre) * sc.ex, (top - datum - sc.zCentre) * sc.ex])));
}

// 3D picking: the domain fractions (x east, y north) where the ray under the pointer meets the
// displayed ground, by marching along the ray in quarter-cell steps; null when it misses.
export function pick(event) {
    const { solver, renderer } = app, sc = renderer.scene, c = camera, rect = el.canvas.getBoundingClientRect(), { config } = renderer.viewData;
    const nx = 2 * (event.clientX - rect.left) / rect.width - 1, ny = 1 - 2 * (event.clientY - rect.top) / rect.height;
    const eye = eyeOf(c);
    const f = norm(sub(c.target, eye)), side = norm(cross(f, [0, 0, 1])), up = cross(side, f), t = Math.tan(0.35), aspect = rect.width / rect.height;
    const dir = norm([0, 1, 2].map(k => f[k] + side[k] * nx * t * aspect + up[k] * ny * t));
    const dx = config.dx, dy = config.dy ?? dx, step = 0.25 * Math.min(dx, dy), bed = solver.bedValues;
    for (let s = 0; s < 6 * c.distance; s += step) {
        const p = [eye[0] + dir[0] * s, eye[1] + dir[1] * s, eye[2] + dir[2] * s];
        const gx = Math.floor(p[0] / dx + config.width / 2), gy = Math.floor(p[1] / dy + config.height / 2);
        if (gx < 0 || gy < 0 || gx >= config.width || gy >= config.height) continue;
        if (p[2] <= (bed[gy * config.width + gx] - sc.zCentre) * sc.ex) return [(gx + 0.5) / config.width, (gy + 0.5) / config.height];
    }
    return null;
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = a => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
