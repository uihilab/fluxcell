// Pointer navigation. 2D: the wheel zooms about the cursor, a drag pans, a double click fits the
// domain. 3D: a drag orbits, a right-drag (or shift-drag) pans, the wheel moves closer, a double click
// resets the camera. On the playgrounds the left button belongs to the tool (brush.mjs), so the right
// button pans in 2D and orbits in 3D. Touch: one finger acts as the left button; two fingers pinch to
// zoom, move together to pan, and in 3D twist to turn the view.
import { app } from './app.mjs';
import { el } from './dom.mjs';
import { zoom, camera, resetCamera, toDomain } from './camera.mjs';
import { setZoom } from './ui/controls.mjs';
import { stopBrush } from './brush.mjs';

const canvas = el.canvas, pan = { active: false, orbit: false, x: 0, y: 0 };
const is3d = () => !!app.renderer?.mode3d;

canvas.addEventListener('wheel', event => {
    event.preventDefault();
    if (is3d()) { camera.distance = Math.min(3 * camera.home, Math.max(0.03 * camera.home, camera.distance * Math.exp(event.deltaY * 0.0015))); return; }
    const [x, y] = toDomain(event), k = Math.min(zoom.max, Math.max(1, zoom.k * Math.exp(-event.deltaY * 0.0015)));
    const rect = canvas.getBoundingClientRect();
    // Keep the point under the cursor fixed.
    setZoom(k, x - ((event.clientX - rect.left) / rect.width - 0.5) / k, y - (0.5 - (event.clientY - rect.top) / rect.height) / k);
}, { passive: false });

// Fingers on the map: their positions, and the two-finger gesture (distance, centre and angle at the
// last move).
const touches = new Map();
let gesture = null;
const twoFingers = () => {
    const [[x1, y1], [x2, y2]] = [...touches.values()];
    return { d: Math.hypot(x2 - x1, y2 - y1), x: (x1 + x2) / 2, y: (y1 + y2) / 2, a: Math.atan2(y2 - y1, x2 - x1) };
};
canvas.addEventListener('pointerdown', event => {
    if (event.pointerType !== 'touch') return;
    touches.set(event.pointerId, [event.clientX, event.clientY]);
    canvas.setPointerCapture(event.pointerId);
    if (touches.size === 2) { pan.active = false; stopBrush(); gesture = twoFingers(); }
});
canvas.addEventListener('pointermove', event => {
    if (!touches.has(event.pointerId)) return;
    touches.set(event.pointerId, [event.clientX, event.clientY]);
    if (touches.size !== 2 || !gesture) return;
    const g = twoFingers(), rect = canvas.getBoundingClientRect(), ratio = g.d / Math.max(gesture.d, 1);
    if (is3d()) {
        camera.distance = Math.min(3 * camera.home, Math.max(0.03 * camera.home, camera.distance / ratio));
        camera.yaw -= g.a - gesture.a;
        const k = camera.distance / rect.height, dx = g.x - gesture.x, dy = g.y - gesture.y, cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
        camera.target[0] += k * (dx * sy - dy * cy); camera.target[1] += k * (-dx * cy - dy * sy);
    } else {
        // The domain point under the fingers' centre stays under it.
        const u0 = (gesture.x - rect.left) / rect.width, v0 = (gesture.y - rect.top) / rect.height;
        const u1 = (g.x - rect.left) / rect.width, v1 = (g.y - rect.top) / rect.height;
        const px = zoom.cx + (u0 - 0.5) / zoom.k, py = zoom.cy + (0.5 - v0) / zoom.k;
        const k = Math.min(zoom.max, Math.max(1, zoom.k * ratio));
        setZoom(k, px - (u1 - 0.5) / k, py - (0.5 - v1) / k);
    }
    gesture = g;
});
for (const type of ['pointerup', 'pointercancel']) canvas.addEventListener(type, event => {
    touches.delete(event.pointerId);
    if (touches.size < 2) gesture = null;
});

canvas.addEventListener('pointerdown', event => {
    if (!event.isPrimary) return; // a second finger: the gesture above
    const sandbox = app.renderer?.viewData?.sandbox;
    if (is3d()) {
        if (sandbox && event.button === 0) return; // the tool
        pan.orbit = sandbox ? event.button === 2 && !event.shiftKey : event.button === 0 && !event.shiftKey;
    } else if (sandbox ? event.button !== 2 : event.button !== 0) return;
    pan.active = true; pan.x = event.clientX; pan.y = event.clientY;
    canvas.setPointerCapture(event.pointerId); canvas.style.cursor = 'grabbing';
});
canvas.addEventListener('pointermove', event => {
    if (!pan.active || !event.isPrimary) return;
    const rect = canvas.getBoundingClientRect(), dx = event.clientX - pan.x, dy = event.clientY - pan.y;
    pan.x = event.clientX; pan.y = event.clientY;
    if (!is3d()) { setZoom(zoom.k, zoom.cx - dx / rect.width / zoom.k, zoom.cy + dy / rect.height / zoom.k); return; }
    if (pan.orbit) { camera.yaw -= dx * 0.006; camera.pitch = Math.min(1.5, Math.max(0.08, camera.pitch + dy * 0.004)); return; }
    const k = camera.distance / rect.height, cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
    camera.target[0] += k * (dx * sy - dy * cy); camera.target[1] += k * (-dx * cy - dy * sy);
});
for (const type of ['pointerup', 'pointercancel']) canvas.addEventListener(type, () => {
    pan.active = false; canvas.style.cursor = el.tools.hidden ? '' : 'crosshair';
});
canvas.addEventListener('contextmenu', event => event.preventDefault());
canvas.addEventListener('dblclick', () => { if (is3d()) resetCamera(); else setZoom(1, 0.5, 0.5); });
