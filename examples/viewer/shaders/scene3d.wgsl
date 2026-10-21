// 3D view: the terrain as a mesh through the cell centres (solid cells raised as buildings) and the
// water surface as a second, partly transparent mesh, both read from the solver buffers each frame.
// Dry vertices next to water carry the level of their wet neighbours (just under their own bed), so
// the surface meets the ground smoothly; the drawn edge is the zero contour of the waterline field.
//#include view.wgsl
struct Scene { viewProj: mat4x4<f32>, eye: vec4<f32>, params: vec4<f32>, mesh: vec4<u32> }
// params: vertical exaggeration, building height (m), bed level at the scene centre, and the gain on
// the water level's departure from still water (flat-bed flows, where the surface would look flat).
// mesh: vertex stride in cells, vertices per row, rows, 1 to hide the bed (flat-bed flows).
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> state: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cells: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> amount: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> shape: array<f32>;
@group(0) @binding(5) var<storage, read> waterline: array<f32>;
@group(0) @binding(6) var<uniform> scene: Scene;
@group(0) @binding(7) var<storage, read> bed0: array<f32>;
//#include wind.wgsl
struct Out { @builtin(position) pos: vec4<f32>, @location(0) world: vec3<f32>, @location(1) normal: vec3<f32>,
             @location(2) data: vec4<f32>, @location(3) misc: vec4<f32> }
//#include grid.wgsl
fn node(vi: u32) -> vec2<i32> {
    let s = i32(scene.mesh.x);
    return vec2<i32>(min(i32(vi % scene.mesh.y) * s, i32(view.width) - 1), min(i32(vi / scene.mesh.y) * s, i32(view.height) - 1));
}
fn planar(p: vec2<i32>) -> vec2<f32> {
    return vec2<f32>((f32(p.x) + 0.5 - 0.5 * f32(view.width)) * view.dx, (f32(p.y) + 0.5 - 0.5 * f32(view.height)) * view.dy);
}
fn ground(x: i32, y: i32) -> f32 {
    let i = cellAt(x, y);
    return (cells[i].x - scene.params.z + select(0.0, scene.params.y, cells[i].w > 0.5)) * scene.params.x;
}
// Water level (m) at a cell: its surface if wet; next to water, the mean level of the wet neighbours
// (under the own bed by one wet depth for an open dry cell); otherwise well under the ground.
fn level(x: i32, y: i32) -> f32 {
    let i = cellAt(x, y); let solid = cells[i].w > 0.5; let wetMin = 0.25 * view.wetDepth;
    if !solid && state[i].x > wetMin { return cells[i].x + state[i].x; }
    var n = 0.0; var sum = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let j = cellAt(x + dx, y + dy);
            if cells[j].w < 0.5 && state[j].x > wetMin { n += 1.0; sum += cells[j].x + state[j].x; }
        }
    }
    if n == 0.0 { return cells[i].x - 1.0 - 10.0 * view.wetDepth; }
    if solid { return sum / n; }
    return min(sum / n, cells[i].x - view.wetDepth);
}
fn vel(x: i32, y: i32) -> vec2<f32> {
    let s = state[cellAt(x, y)];
    if s.x > view.wetDepth { return s.yz / s.x; }
    return vec2<f32>(0.0);
}
//#include colormaps.wgsl
fn normalOf(zl: f32, zr: f32, zd: f32, zu: f32) -> vec3<f32> {
    let s = f32(scene.mesh.x);
    return normalize(vec3<f32>(-(zr - zl) / (2.0 * s * view.dx), -(zu - zd) / (2.0 * s * view.dy), 1.0));
}
@vertex fn vsGround(@builtin(vertex_index) vi: u32) -> Out {
    let p = node(vi); let s = i32(scene.mesh.x); let i = cellAt(p.x, p.y);
    var o: Out;
    o.world = vec3<f32>(planar(p), ground(p.x, p.y));
    o.pos = scene.viewProj * vec4<f32>(o.world, 1.0);
    o.normal = normalOf(ground(p.x - s, p.y), ground(p.x + s, p.y), ground(p.x, p.y - s), ground(p.x, p.y + s));
    o.data = vec4<f32>(0.0);
    o.misc = vec4<f32>(cells[i].x, 0.0, shape[i], 0.0);
    return o;
}
fn surfaceZ(l: f32) -> f32 {
    let anomaly = clamp(l - view.still, -4.0 * view.elevationScale, 4.0 * view.elevationScale); // dry cells are not raised
    return (l - scene.params.z) * scene.params.x + anomaly * scene.params.w;
}
@vertex fn vsWater(@builtin(vertex_index) vi: u32) -> Out {
    let p = node(vi); let s = i32(scene.mesh.x); let i = cellAt(p.x, p.y);
    let ex = scene.params.x; let zc = scene.params.z;
    var o: Out;
    let surface = level(p.x, p.y);
    o.world = vec3<f32>(planar(p), surfaceZ(surface));
    o.pos = scene.viewProj * vec4<f32>(o.world, 1.0);
    o.normal = normalOf(surfaceZ(level(p.x - s, p.y)), surfaceZ(level(p.x + s, p.y)),
                        surfaceZ(level(p.x, p.y - s)), surfaceZ(level(p.x, p.y + s)));
    let h = select(state[i].x, 0.0, cells[i].w > 0.5);
    let vort = (vel(p.x + 1, p.y).y - vel(p.x - 1, p.y).y) / (2.0 * view.dx) - (vel(p.x, p.y + 1).x - vel(p.x, p.y - 1).x) / (2.0 * view.dy);
    var c = 0.0;
    if view.tracer > 0.5 && h > 0.0 { c = clamp(amount[i].x / h, 0.0, 1.0); }
    o.data = vec4<f32>(h, length(vel(p.x, p.y)), vort, c);
    o.misc = vec4<f32>(cells[i].x, waterline[i], cells[i].x - bed0[i], surface - view.still);
    return o;
}
const SUN = vec3<f32>(-0.45, 0.55, 0.70);
@fragment fn fsGround(o: Out) -> @location(0) vec4<f32> {
    let n = normalize(o.normal); let l = normalize(SUN);
    let t = clamp((o.misc.x - view.zMin) / max(view.zMax - view.zMin, 1e-6), 0.0, 1.0);
    var colour = land(t);
    if view.relief > 0.0 { colour = mix(vec3<f32>(0.87, 0.80, 0.62), colour, smoothstep(0.3, 2.5, o.misc.x - view.still)); }
    // Buildings and walls: the outline field, so their footprint keeps its shape between vertices.
    let solid = clamp(0.5 - o.misc.z / max(fwidth(o.misc.z), 1e-6), 0.0, 1.0);
    // Flat-bed flows (mesh.w = 1) show the water alone: the bed is not drawn, solid obstacles are.
    if scene.mesh.w == 1u && solid < 0.5 { discard; }
    colour = mix(colour, vec3<f32>(0.62, 0.62, 0.65), solid);
    var shade = 0.38 + 0.62 * max(dot(n, l), 0.0);
    if view.contour > 0.0 {
        let cz = o.misc.x / view.contour; let w = max(fwidth(cz), 1e-6);
        shade *= 1.0 - 0.4 * (1.0 - smoothstep(0.0, 1.2, abs(fract(cz + 0.5) - 0.5) / w));
    }
    return vec4<f32>(colour * shade, 1.0);
}
@fragment fn fsWater(o: Out) -> @location(0) vec4<f32> {
    let edge = clamp(o.misc.y / max(fwidth(o.misc.y), 1e-6) + 0.5, 0.0, 1.0);
    if edge <= 0.0 { discard; }
    let n = normalize(o.normal); let l = normalize(SUN); let v = normalize(scene.eye.xyz - o.world);
    let depth = o.data.x;
    var water = mix(vec3<f32>(0.30, 0.70, 0.76), vec3<f32>(0.02, 0.15, 0.37), 1.0 - exp(-2.0 * depth / view.depthScale));
    var alpha = 0.55 + 0.4 * (1.0 - exp(-3.0 * depth / view.depthScale));
    if view.mode == 1u { water = viridis(clamp(o.data.y / view.speedScale, 0.0, 1.0)); alpha = 0.95; }
    if view.mode == 2u {
        water = diverging(clamp(0.5 + 0.5 * o.data.z / view.vorticityScale, 0.0, 1.0), vec3<f32>(0.13, 0.30, 0.65), vec3<f32>(0.70, 0.10, 0.12));
        alpha = 0.95;
    }
    if view.mode == 5u {
        water = diverging(clamp(0.5 + 0.5 * o.misc.z / view.elevationScale, 0.0, 1.0), vec3<f32>(0.10, 0.30, 0.70), vec3<f32>(0.55, 0.33, 0.12));
        alpha = 0.97;
    }
    if view.mode == 4u {
        let e = clamp(o.misc.w / view.elevationScale, -1.0, 1.0);
        water = select(water * (1.0 + 0.45 * e), mix(water, vec3<f32>(0.80, 0.93, 1.0), 0.5 * e), e > 0.0);
    }
    if view.mode == 3u {
        let c = o.data.w;
        if view.logScale == 1u {
            if c > 0.01 {
                let a = clamp(log2(c * 100.0) / log2(100.0), 0.0, 1.0);
                water = mix(water, mix(vec3<f32>(0.98, 0.85, 0.25), vec3<f32>(0.75, 0.08, 0.1), a), 0.35 + 0.6 * a);
                alpha = max(alpha, 0.6 + 0.35 * a);
            }
        } else { water = diverging(c, vec3<f32>(0.10, 0.25, 0.55), vec3<f32>(0.80, 0.35, 0.05)); alpha = 0.95; }
    }
    let windScale = min(f32(view.width) * view.dx, f32(view.height) * view.dy);
    let gusts = windVisual(o.world.xy, vec2<f32>(view.windX, view.windY), windScale, view.time);
    water *= 1.0 - 0.25 * min(1.0, length(vec2<f32>(view.windX, view.windY)) / 12.0) * gusts.y;
    // Light: diffuse, sky reflection at grazing angles, a sun highlight, and foam on fast shallow water.
    let fresnel = 0.03 + 0.97 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    var colour = water * (0.62 + 0.38 * max(dot(n, l), 0.0));
    colour = mix(colour, vec3<f32>(0.86, 0.92, 0.97), 0.55 * fresnel);
    colour += vec3<f32>(0.6) * pow(max(dot(reflect(-l, n), v), 0.0), 90.0);
    if view.relief > 0.0 && (view.mode == 0u || view.mode == 4u) {
        let foam = smoothstep(0.3, 0.8, o.data.y / view.speedScale) * (1.0 - smoothstep(0.03, 0.12, depth / view.depthScale));
        colour = mix(colour, vec3<f32>(0.96, 0.98, 1.0), 0.8 * foam); alpha = max(alpha, foam);
    }
    // Films thinner than a few wet depths (sheets on steep rock, rain films) fade out.
    return vec4<f32>(colour, alpha * edge * smoothstep(0.5 * view.wetDepth, 4.0 * view.wetDepth, depth));
}
