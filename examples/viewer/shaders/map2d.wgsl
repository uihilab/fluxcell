// 2D map: one fragment per pixel, sampling the solver buffers at the pixel's position in the zoom
// window. Display only; the solution stays a set of cell averages.
// - Ground: bilinear bed, shaded from interpolated central-difference slopes, hypsometric tint,
//   optional contour lines; solid cells (buildings, walls) drawn from the outline field `shape`.
// - Water: the waterline is the zero contour of the per-cell waterline field (waterline.wgsl),
//   interpolated between cell centres, so it does not follow the cell staircase.
// - Colour modes (view.mode): 0 depth, 1 speed, 2 vorticity, 3 concentration (log or linear),
//   4 water level relative to still water, 5 bed change since the start of the run.
//#include view.wgsl
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> state: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cells: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> amount: array<vec2<f32>>;
// Outline of the solid cells (buildings, walls): a signed distance in cells at the cell centres,
// negative inside. Its zero contour, interpolated between centres, draws the outline smoothly.
@group(0) @binding(4) var<storage, read> shape: array<f32>;
// Waterline field at the cell centres, written each frame by the WATERLINE passes.
@group(0) @binding(5) var<storage, read> waterline: array<f32>;
// Water level at every cell (dry cells next to water carry their wet neighbours' level) and its
// slope, from the same passes; interpolated, they shade the water without cell-sized patches.
@group(0) @binding(6) var<storage, read> levels: array<f32>;
@group(0) @binding(7) var<storage, read> slopes: array<vec2<f32>>;
// Bed at the start of the run (bed-change colouring of demos whose bed evolves).
@group(0) @binding(8) var<storage, read> bed0: array<f32>;
//#include wind.wgsl
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}
//#include grid.wgsl
fn z(x: i32, y: i32) -> f32 { return cells[cellAt(x, y)].x; }
fn slope(x: i32, y: i32) -> vec2<f32> {
    return vec2<f32>((z(x + 1, y) - z(x - 1, y)) / (2.0 * view.dx), (z(x, y + 1) - z(x, y - 1)) / (2.0 * view.dy));
}
fn vel(x: i32, y: i32) -> vec2<f32> {
    let s = state[cellAt(x, y)];
    if s.x > view.wetDepth { return s.yz / s.x; }
    return vec2<f32>(0.0);
}
//#include colormaps.wgsl
fn vorticity(x: i32, y: i32) -> f32 {
    return (vel(x + 1, y).y - vel(x - 1, y).y) / (2.0 * view.dx) - (vel(x, y + 1).x - vel(x, y - 1).x) / (2.0 * view.dy);
}
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    // Continuous cell coordinates with cell centres at integers; row 0 is the south edge. The canvas
    // shows the window of size 1/zoom (as a fraction of the domain) centred on (centreX, centreY).
    let screen = vec2<f32>(pos.x / view.canvasW, 1.0 - pos.y / view.canvasH) - 0.5;
    let g = (vec2<f32>(view.centreX, view.centreY) + screen / view.zoom) * vec2<f32>(f32(view.width), f32(view.height)) - 0.5;
    let b = floor(g); let f = g - b; let x0 = i32(b.x); let y0 = i32(b.y);
    var corners = array<vec2<i32>, 4>(vec2<i32>(x0, y0), vec2<i32>(x0 + 1, y0), vec2<i32>(x0, y0 + 1), vec2<i32>(x0 + 1, y0 + 1));
    var weights = array<f32, 4>((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    var bed = 0.0; var grad = vec2<f32>(0.0); var depth = 0.0; var c = 0.0; var open = 0.0;
    var spd = 0.0; var vort = 0.0; var surface = 0.0; var sgrad = vec2<f32>(0.0);
    var solidShape = 0.0; var gap = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let j = cellAt(corners[k].x, corners[k].y);
        gap += weights[k] * waterline[j]; surface += weights[k] * levels[j]; sgrad += weights[k] * slopes[j];
        solidShape += weights[k] * shape[cellAt(corners[k].x, corners[k].y)];
    }
    for (var k = 0u; k < 4u; k++) {
        let p = corners[k]; let i = cellAt(p.x, p.y); let w = weights[k];
        bed += w * cells[i].x;
        grad += w * slope(p.x, p.y);
        if cells[i].w < 0.5 { // solid corners do not pull the water field down
            let h = state[i].x;
            depth += w * h; open += w;
            if view.tracer > 0.5 && h > 0.0 { c += w * clamp(amount[i].x / h, 0.0, 1.0); }
            if view.mode == 1u || view.relief > 0.0 { spd += w * length(vel(p.x, p.y)); }
            if view.mode == 2u { vort += w * vorticity(p.x, p.y); }

        }
    }
    if open > 1e-6 { depth /= open; c /= open; spd /= open; vort /= open; }
    // Anti-aliased over about one screen pixel; films thinner than wetDepth fade out.
    let coverage = clamp(gap / max(fwidth(gap), 1e-6) + 0.5, 0.0, 1.0);
    let solidCover = clamp(0.5 - solidShape / max(fwidth(solidShape), 1e-6), 0.0, 1.0);
    let normal = normalize(vec3<f32>(-grad.x, -grad.y, 1.0));
    let light = normalize(vec3<f32>(-0.6, 0.6, 1.0));
    let shade = 0.45 + 0.55 * max(dot(normal, light), 0.0);
    let t = clamp((bed - view.zMin) / max(view.zMax - view.zMin, 1e-6), 0.0, 1.0);
    var ground = land(t);
    // Coastal scenes (relief > 0): a sand band from the waterline to about 2 m above still water.
    if view.relief > 0.0 { ground = mix(vec3<f32>(0.87, 0.80, 0.62), ground, smoothstep(0.3, 2.5, bed - view.still)); }
    ground *= shade;
    if view.contour > 0.0 {
        // Contour lines every view.contour metres (the playground sand-table look).
        let cz = bed / view.contour; let w = max(fwidth(cz), 1e-6);
        ground *= 1.0 - 0.4 * (1.0 - smoothstep(0.0, 1.2, abs(fract(cz + 0.5) - 0.5) / w));
    }
    let metres = (g + 0.5) * vec2<f32>(view.dx, view.dy);
    let windScale = min(f32(view.width) * view.dx, f32(view.height) * view.dy);
    let gusts = windVisual(metres, vec2<f32>(view.windX, view.windY), windScale, view.time);
    let windStrength = min(1.0, length(vec2<f32>(view.windX, view.windY)) / 12.0);
    // Water: turquoise shallows to navy deep water.
    var water = mix(vec3<f32>(0.32, 0.72, 0.78), vec3<f32>(0.02, 0.15, 0.37), 1.0 - exp(-2.0 * depth / view.depthScale));
    if view.mode == 1u { water = viridis(clamp(spd / view.speedScale, 0.0, 1.0)); }
    if view.mode == 2u {
        water = diverging(clamp(0.5 + 0.5 * vort / view.vorticityScale, 0.0, 1.0), vec3<f32>(0.13, 0.30, 0.65), vec3<f32>(0.70, 0.10, 0.12));
    }
    if view.mode == 4u {
        // Water level: the depth colour, lighter on crests and darker in troughs.
        let e = clamp((surface - view.still) / view.elevationScale, -1.0, 1.0);
        water = select(water * (1.0 + 0.55 * e), mix(water, vec3<f32>(0.80, 0.93, 1.0), 0.55 * e), e > 0.0);
    }
    if view.relief > 0.0 && (view.mode == 0u || view.mode == 4u) {
        // Waves in relief: light on the water surface (exaggerated slope), a soft highlight, and foam
        // where fast water runs over the shallows.
        let sl = sgrad * clamp(gap, 0.0, 1.0); // no relief in the last few wet depths at the waterline
        let n = normalize(vec3<f32>(-sl.x * view.relief, -sl.y * view.relief, 1.0));
        water *= 0.72 + 0.4 * max(dot(n, light), 0.0);
        water += vec3<f32>(0.22) * pow(max(dot(n, normalize(light + vec3<f32>(0.0, 0.0, 1.0))), 0.0), 60.0);
        let foam = smoothstep(0.3, 0.8, spd / view.speedScale) * (1.0 - smoothstep(0.03, 0.12, depth / view.depthScale));
        water = mix(water, vec3<f32>(0.95, 0.97, 1.0), 0.75 * foam);
    }
    // Wind on the water: darker moving gust patches and fine ripples (visual only).
    water *= 1.0 - 0.25 * windStrength * gusts.y;
    if view.mode == 3u {
        if view.logScale == 1u {
            if c > 0.01 {
                let a = clamp(log2(c * 100.0) / log2(100.0), 0.0, 1.0);
                water = mix(water, mix(vec3<f32>(0.98, 0.85, 0.25), vec3<f32>(0.75, 0.08, 0.1), a), 0.35 + 0.6 * a);
            }
        } else {
            water = diverging(c, vec3<f32>(0.10, 0.25, 0.55), vec3<f32>(0.80, 0.35, 0.05));
        }
    }
    if view.mode == 5u {
        // Bed change: erosion blue, deposition brown, over view.elevationScale metres.
        var dz = 0.0;
        for (var k = 0u; k < 4u; k++) { let j = cellAt(corners[k].x, corners[k].y); dz += weights[k] * (cells[j].x - bed0[j]); }
        water = diverging(clamp(0.5 + 0.5 * dz / view.elevationScale, 0.0, 1.0), vec3<f32>(0.10, 0.30, 0.70), vec3<f32>(0.55, 0.33, 0.12));
    }
    // Terrain shading shows through depth and contaminant colouring, not through data maps.
    let shaded = (view.mode == 0u && view.relief == 0.0) || (view.mode == 3u && view.logScale == 1u);
    let color = mix(ground, water * select(1.0, 0.75 + 0.25 * shade, shaded), coverage * 0.94);
    return vec4<f32>(mix(color, vec3<f32>(0.22, 0.22, 0.25), solidCover), 1.0);
}
