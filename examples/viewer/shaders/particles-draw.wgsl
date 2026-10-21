//#include particles-common.wgsl
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> state: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cells: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> particles: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> pp: Particles;
@group(0) @binding(5) var<uniform> scene: Scene;
struct Out { @builtin(position) pos: vec4<f32>, @location(0) t: vec2<f32> }
// A streak along the local velocity: tail at the particle minus a length that grows with speed.
fn streak(k: u32) -> array<vec2<f32>, 2> {
    let p = particles[k].xy; let v = velocity(p) * vec2<f32>(1.0 / view.dx, 1.0 / view.dy); let speed = length(velocity(p));
    let len = 0.6 + 4.0 * clamp(speed / max(view.speedScale, 1e-6), 0.0, 1.0);
    let dir = select(vec2<f32>(1.0, 0.0), normalize(v), length(v) > 1e-9);
    return array<vec2<f32>, 2>(p - dir * len * min(1.0, 2.0 * f32(view.width + view.height) / 2000.0 + 0.5), p);
}
fn quad(a: vec4<f32>, b: vec4<f32>, vi: u32, k: u32) -> Out {
    // Clip-space endpoints expanded across the segment by pp.width pixels.
    let na = a.xy / a.w; let nb = b.xy / b.w; let px = vec2<f32>(view.canvasW, view.canvasH);
    var d = (nb - na) * px; if length(d) < 1e-6 { d = vec2<f32>(1.0, 0.0); }
    let n = normalize(vec2<f32>(-d.y, d.x)) * pp.width / px;
    var corner = array<vec2<f32>, 6>(vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0));
    let c = corner[vi]; let e = select(a, b, c.x > 0.5);
    var o: Out;
    o.pos = vec4<f32>(e.xy + n * c.y * e.w, e.z, e.w);
    let q = particles[k];
    o.t = vec2<f32>(c.x, select(sin(3.14159 * q.z / q.w), 0.0, q.z >= q.w));
    return o;
}
@vertex fn vs2d(@builtin(vertex_index) vi: u32, @builtin(instance_index) k: u32) -> Out {
    let ends = streak(k); let size = vec2<f32>(f32(view.width), f32(view.height));
    let ndc = (((ends[0] + 0.5) / size - vec2<f32>(view.centreX, view.centreY)) * view.zoom) * 2.0;
    let ndc2 = (((ends[1] + 0.5) / size - vec2<f32>(view.centreX, view.centreY)) * view.zoom) * 2.0;
    return quad(vec4<f32>(ndc, 0.0, 1.0), vec4<f32>(ndc2, 0.0, 1.0), vi, k);
}
fn world(g: vec2<f32>) -> vec4<f32> {
    let i = cellAt(i32(round(g.x)), i32(round(g.y))); let l = cells[i].x + state[i].x;
    let anomaly = clamp(l - view.still, -4.0 * view.elevationScale, 4.0 * view.elevationScale);
    let lift = 0.004 * min(f32(view.width) * view.dx, f32(view.height) * view.dy);
    let z = (l - scene.params.z) * scene.params.x + anomaly * scene.params.w + lift;
    let xy = (g + 0.5 - 0.5 * vec2<f32>(f32(view.width), f32(view.height))) * vec2<f32>(view.dx, view.dy);
    return scene.viewProj * vec4<f32>(xy, z, 1.0);
}
@vertex fn vs3d(@builtin(vertex_index) vi: u32, @builtin(instance_index) k: u32) -> Out {
    let ends = streak(k);
    return quad(world(ends[0]), world(ends[1]), vi, k);
}
@fragment fn fs(o: Out) -> @location(0) vec4<f32> {
    // Fades in and out over the particle's life; the tail is transparent, the head opaque.
    return vec4<f32>(pp.colour.rgb, pp.colour.a * o.t.y * (0.15 + 0.85 * o.t.x));
}
