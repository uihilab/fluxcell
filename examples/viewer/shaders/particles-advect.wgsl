//#include particles-common.wgsl
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> state: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cells: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> particles: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> pp: Particles;
@compute @workgroup_size(64) fn advect(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    if k >= pp.count { return; }
    var p = particles[k];
    let size = vec2<f32>(f32(view.width), f32(view.height));
    let toCells = 1.0 / vec2<f32>(view.dx, view.dy);
    // Midpoint (RK2) steps over the simulated time of this frame, in four substeps.
    let h = pp.dt / 4.0;
    for (var j = 0; j < 4; j++) {
        let mid = p.xy + 0.5 * h * velocity(p.xy) * toCells;
        p = vec4<f32>(p.xy + h * velocity(mid) * toCells, p.zw);
    }
    p.z += 1.0;
    let i = cellAt(i32(round(p.x)), i32(round(p.y)));
    let gone = p.z >= p.w || any(p.xy < vec2<f32>(-0.5)) || any(p.xy > size - 0.5) || state[i].x <= view.wetDepth || cells[i].w > 0.5;
    if gone {
        // Respawn at a random wet cell (a dry pick stays dead and retries next frame).
        let seed = hash(k * 9781u + pp.frame * 6271u);
        let q = (pp.seed.xy + vec2<f32>(rand(seed), rand(seed ^ 0x9e3779b9u)) * (pp.seed.zw - pp.seed.xy)) * size - 0.5;
        let j = cellAt(i32(round(q.x)), i32(round(q.y)));
        let life = pp.life.x + pp.life.y * rand(seed ^ 0x85ebca6bu);
        p = select(vec4<f32>(q, life, life), vec4<f32>(q, 0.0, life), state[j].x > view.wetDepth && cells[j].w < 0.5);
    }
    particles[k] = p;
}
