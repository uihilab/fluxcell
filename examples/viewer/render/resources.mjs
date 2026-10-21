// Per-demo GPU resources and display settings: the view description (renderer.viewData), the outline
// and waterline fields, bind groups over the solver's buffers, the 3D mesh, flow tracers, cylinders
// and bed evolution. Called once per demo, after the solver is created.
import { app } from '../app.mjs';

const STORAGE = GPUBufferUsage.STORAGE, COPY = GPUBufferUsage.COPY_DST;

export function prepareResources(scenario) {
    const { device, solver, renderer } = app;
    const { config, depthScale, wetDepth, sandbox } = scenario;
    const n = config.width * config.height;
    let zMin = Infinity, zMax = -Infinity;
    for (const z of solver.bedValues) { zMin = Math.min(zMin, z); zMax = Math.max(zMax, z); }
    const tracer = !!config.tracer, wind = config.wind ?? [0, 0];
    // Gauge: volume of a region (the reservoir) and the outflow from it, -dV/dt, read back twice a second.
    const gauge = scenario.gauge ? { ...scenario.gauge, cells: Uint32Array.from(scenario.gauge.mask.reduce((a, v, i) => (v && a.push(i), a), [])),
        area: config.dx * (config.dy ?? config.dx), previous: null, peak: 0, lastWall: 0 } : null;
    renderer.viewData = { gauge, wind, windSpeed: Math.hypot(...wind), contour: scenario.contour ?? 0, config, zMin, zMax, depthScale, wetDepth,
        tracer, sandbox: !!sandbox, brushKey: '', logScale: scenario.concentration !== 'linear',
        speedScale: scenario.speedScale ?? 2 * Math.sqrt(9.81 * depthScale), vorticityScale: scenario.vorticityScale ?? 0.2,
        release: scenario.release, still: scenario.still, elevationScale: scenario.elevationScale ?? 0.1, relief: scenario.relief ?? 0,
        datum: solver.datum ?? 0 };

    // Outline field of the solid cells: from the scenario if it has one (Grand Isle: distances to the
    // OpenStreetMap outlines), otherwise from the cell mask smoothed over 3 x 3 cells (straight walls
    // stay at the cell faces; staircases of a curved or oblique outline become a smooth curve).
    const shapeField = scenario.shape ?? maskShape(config.width, config.height, scenario.initial.solid);
    renderer.shape?.destroy();
    renderer.shape = device.createBuffer({ size: shapeField.byteLength, usage: STORAGE | COPY });
    device.queue.writeBuffer(renderer.shape, 0, shapeField);
    renderer.bed0?.destroy(); // the bed at the start, for the bed-change colouring
    renderer.bed0 = device.createBuffer({ size: 4 * n, usage: STORAGE | COPY });
    device.queue.writeBuffer(renderer.bed0, 0, Float32Array.from(solver.bedValues));
    for (const b of renderer.waterline ?? []) b.destroy(); // raw and smoothed waterline, level, slope
    renderer.waterline = [4, 4, 4, 8].map(bytes => device.createBuffer({ size: bytes * n, usage: STORAGE }));

    const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const wl = renderer.waterline;
    renderer.waterlineGroups = [0, 1].map(sp => group(renderer.waterlineLayout, [renderer.view, solver.states[sp], solver.cells, ...wl]));
    renderer.groups = [0, 1].map(sp => group(renderer.layout,
        [renderer.view, solver.states[sp], solver.cells, solver.tracers[sp], renderer.shape, wl[1], wl[2], wl[3], renderer.bed0]));

    // 3D mesh: vertices at the cell centres (every stride-th cell on large grids), two triangles per quad.
    const stride = Math.max(1, Math.ceil(Math.sqrt(n / 700000)));
    const nxv = Math.floor((config.width - 1) / stride) + 1, nyv = Math.floor((config.height - 1) / stride) + 1;
    const indices = new Uint32Array(6 * (nxv - 1) * (nyv - 1));
    for (let y = 0, k = 0; y < nyv - 1; y++) for (let x = 0; x < nxv - 1; x++) {
        const a = y * nxv + x, b = a + 1, c = a + nxv, d = c + 1;
        indices.set([a, b, d, a, d, c], k); k += 6;
    }
    renderer.indices?.destroy();
    renderer.indices = device.createBuffer({ size: indices.byteLength, usage: GPUBufferUsage.INDEX | COPY });
    device.queue.writeBuffer(renderer.indices, 0, indices);
    renderer.indexCount = indices.length;
    renderer.sceneGroups = [0, 1].map(sp => group(renderer.sceneLayout,
        [renderer.view, solver.states[sp], solver.cells, solver.tracers[sp], renderer.shape, wl[1], renderer.sceneUniform, renderer.bed0]));

    prepareTracers(scenario, group);
    prepareCylinders(scenario, zMin, zMax);
    prepareMorph(scenario, group);

    // Vertical exaggeration: the relief (with one depth scale of water) shown at about a sixth of the
    // domain size, between 1 and 15, unless the demo sets it.
    const Lx = config.width * (config.dx ?? 1), Ly = config.height * (config.dy ?? config.dx ?? 1);
    const ex = scenario.exaggeration ?? Math.min(15, Math.max(1, Math.min(Lx, Ly) / 6 / (zMax - zMin + depthScale)));
    renderer.scene = { stride, nxv, nyv, ex, Lx, Ly, cylinders: !!scenario.cylinders, surfaceGain: scenario.surfaceGain ?? 0,
        hideGround: !!scenario.hideGround, camera: scenario.camera, relief: zMax - zMin, zCentre: (zMin + zMax) / 2,
        buildingHeight: scenario.buildingHeight ?? 1.5 * depthScale };
    renderer.mode3d = !!scenario.view3d;
    renderer.rate = scenario.rate;
    renderer.stats = { steps: 0, wall: 0 };
}

// Flow tracers: about one per 24 cells (1,500 to 12,000), released at random places; a demo may release
// them from a region (particles: { seed: [x0, y0, x1, y1], life: [min, range], count }). Off by default.
function prepareTracers(scenario, group) {
    const { device, solver, renderer } = app, t = renderer.tracers, { config } = scenario;
    const opt = typeof scenario.particles === 'object' ? scenario.particles : {};
    const count = opt.count ?? Math.max(1500, Math.min(12000, Math.floor(config.width * config.height / 24)));
    t.seed = opt.seed ?? [0, 0, 1, 1]; t.life = opt.life ?? [90, 210];
    const init = new Float32Array(4 * count);
    for (let k = 0; k < count; k++) {
        // Staggered: all start dead with spread-out ages, so they are released over the first lifetime.
        const life = t.life[0] + t.life[1] * Math.random();
        init.set([0, 0, life * (1 - Math.random() * 0.999), life], 4 * k);
    }
    t.buffer?.destroy();
    t.buffer = device.createBuffer({ size: init.byteLength, usage: STORAGE | COPY });
    device.queue.writeBuffer(t.buffer, 0, init);
    t.count = count; t.lastTime = 0; t.on = false;
    t.computeGroups = [0, 1].map(sp => group(t.computeLayout, [renderer.view, solver.states[sp], solver.cells, t.buffer, t.params]));
    t.renderGroups = [0, 1].map(sp => group(t.renderLayout, [renderer.view, solver.states[sp], solver.cells, t.buffer, t.params, renderer.sceneUniform]));
}

// Round structures drawn as cylinders (scenario.cylinders = [{ x, y, r }], metres from the south-west
// corner, up to scenario.cylinderTop); their solid cells are then not raised in the ground mesh.
function prepareCylinders(scenario, zMin, zMax) {
    const { device, renderer } = app, cy = renderer.cylinders, list = scenario.cylinders ?? [], { config } = scenario;
    cy.buffer?.destroy(); cy.count = list.length;
    if (!list.length) return;
    const Lx = config.width * config.dx, Ly = config.height * (config.dy ?? config.dx), top = scenario.cylinderTop ?? zMax + scenario.depthScale;
    cy.spec = list.map(c => [c.x - Lx / 2, c.y - Ly / 2, c.r, zMin - 1, top]);
    cy.buffer = device.createBuffer({ size: 20 * list.length, usage: STORAGE | COPY });
    cy.group = device.createBindGroup({ layout: cy.layout, entries: [{ binding: 0, resource: { buffer: renderer.sceneUniform } }, { binding: 1, resource: { buffer: cy.buffer } }] });
}

// Bed evolution (scenario.morph = { d, s, porosity, thetaC, beta, repose, factor, maxStep }).
function prepareMorph(scenario, group) {
    const { device, solver, renderer } = app, m = renderer.morph, { config } = scenario;
    m.faces?.destroy(); m.on = !!scenario.morph;
    if (!m.on) return;
    m.config = scenario.morph; m.lastTime = 0;
    m.faces = device.createBuffer({ size: 8 * config.width * config.height, usage: STORAGE });
    m.groups = [0, 1].map(sp => group(m.layout, [renderer.view, solver.states[sp], solver.cells, m.faces, m.params]));
}

// Outline field of a solid mask: 0.5 minus the mask blurred by a Gaussian of 0.7 cell (separable,
// five taps), negative inside. Its zero contour is a smooth curve through the staircase of the
// cells; straight walls stay at the cell faces and walls one cell thick are kept.
export function maskShape(W, H, solid) {
    const field = new Float32Array(W * H).fill(1);
    if (!solid || !solid.some(v => v)) return field;
    const k = [0.0097, 0.2054, 0.5698, 0.2054, 0.0097], row = new Float32Array(W * H);
    const clampX = x => Math.min(W - 1, Math.max(0, x)), clampY = y => Math.min(H - 1, Math.max(0, y));
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        let m = 0;
        for (let i = -2; i <= 2; i++) m += k[i + 2] * (solid[y * W + clampX(x + i)] ? 1 : 0);
        row[y * W + x] = m;
    }
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        let m = 0;
        for (let j = -2; j <= 2; j++) m += k[j + 2] * row[clampY(y + j) * W + x];
        field[y * W + x] = 0.5 - m;
    }
    return field;
}
