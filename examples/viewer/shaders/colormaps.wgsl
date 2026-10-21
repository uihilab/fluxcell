// Hypsometric land tint: green lowland, olive, brown, pale rock.
fn land(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.45, 0.58, 0.36); let c1 = vec3<f32>(0.62, 0.64, 0.42);
    let c2 = vec3<f32>(0.64, 0.54, 0.41); let c3 = vec3<f32>(0.84, 0.82, 0.78);
    if t < 0.4 { return mix(c0, c1, t / 0.4); }
    if t < 0.75 { return mix(c1, c2, (t - 0.4) / 0.35); }
    return mix(c2, c3, (t - 0.75) / 0.25);
}
// Perceptually ordered maps: viridis-like for speed, blue-white-red for vorticity, and for linear
// concentration deep blue (0) through white (0.5, mixed) to dark orange (1).
fn viridis(t: f32) -> vec3<f32> {
    let a = vec3<f32>(0.267, 0.005, 0.329); let b = vec3<f32>(0.128, 0.567, 0.551); let c = vec3<f32>(0.993, 0.906, 0.144);
    return select(mix(b, c, (t - 0.5) * 2.0), mix(a, b, t * 2.0), t < 0.5);
}
fn diverging(t: f32, lo: vec3<f32>, hi: vec3<f32>) -> vec3<f32> {
    let w = vec3<f32>(0.97, 0.97, 0.95);
    return select(mix(w, hi, (t - 0.5) * 2.0), mix(lo, w, t * 2.0), t < 0.5);
}
