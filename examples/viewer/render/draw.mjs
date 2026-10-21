// One frame: compute passes (waterline fields, bed evolution, tracer advection), then the 2D map or
// the 3D scene, all reading the solver's current state buffer.
import { app } from '../app.mjs';
import { el } from '../dom.mjs';
import { writeScene } from '../camera.mjs';

export function draw() {
    const { device, solver, renderer } = app, ping = solver.ping;
    device.queue.writeBuffer(renderer.view, 84, new Float32Array([performance.now() / 1000 % 3600])); // wind animation clock
    const encoder = device.createCommandEncoder();
    const { width, height } = renderer.viewData.config, gx = Math.ceil(width / 16), gy = Math.ceil(height / 16);
    const compute = encoder.beginComputePass();
    for (const pipeline of renderer.waterlinePipelines) {
        compute.setPipeline(pipeline); compute.setBindGroup(0, renderer.waterlineGroups[ping]);
        compute.dispatchWorkgroups(gx, gy);
    }
    encodeMorph(compute, gx, gy);
    encodeTracers(compute);
    compute.end();
    const t = renderer.tracers, target = renderer.context.getCurrentTexture().createView();
    if (renderer.mode3d) {
        writeScene();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0.95, g: 0.96, b: 0.97, a: 1 } }],
            depthStencilAttachment: { view: renderer.depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } });
        pass.setIndexBuffer(renderer.indices, 'uint32');
        pass.setBindGroup(0, renderer.sceneGroups[ping]);
        pass.setPipeline(renderer.groundPipeline); pass.drawIndexed(renderer.indexCount);
        const cy = renderer.cylinders;
        if (cy.count) {
            pass.setPipeline(cy.pipeline); pass.setBindGroup(0, cy.group); pass.draw(48 * 9, cy.count);
            pass.setBindGroup(0, renderer.sceneGroups[ping]);
        }
        pass.setPipeline(renderer.waterPipeline); pass.drawIndexed(renderer.indexCount);
        if (t.on) { pass.setPipeline(t.draw3d); pass.setBindGroup(0, t.renderGroups[ping]); pass.draw(6, t.count); }
        pass.end();
    } else {
        const pass = encoder.beginRenderPass({ colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
        pass.setPipeline(renderer.pipeline);
        pass.setBindGroup(0, renderer.groups[ping]);
        pass.draw(3);
        if (t.on) { pass.setPipeline(t.draw2d); pass.setBindGroup(0, t.renderGroups[ping]); pass.draw(6, t.count); }
        pass.end();
    }
    device.queue.submit([encoder.finish()]);
}

// Bed evolution over the simulated time since the last frame, times the morphological factor, in up
// to eight substeps of at most maxStep seconds, then the repose slides.
function encodeMorph(compute, gx, gy) {
    const { device, solver, renderer } = app, m = renderer.morph;
    if (!m.on) return;
    const c = m.config, elapsed = Math.max(0, solver.time - m.lastTime) * c.factor; m.lastTime = solver.time;
    const substeps = Math.min(8, Math.max(1, Math.ceil(elapsed / c.maxStep)));
    device.queue.writeBuffer(m.params, 0, new Float32Array([elapsed / substeps, c.d, c.s, c.porosity, c.thetaC, c.beta, c.repose, 0]));
    compute.setBindGroup(0, m.groups[solver.ping]);
    for (let k = 0; k < substeps; k++) {
        compute.setPipeline(m.fluxes); compute.dispatchWorkgroups(gx, gy);
        compute.setPipeline(m.exner); compute.dispatchWorkgroups(gx, gy);
    }
    compute.setPipeline(m.slides); compute.dispatchWorkgroups(gx, gy);
    compute.setPipeline(m.settle); compute.dispatchWorkgroups(gx, gy);
}

// Tracers move over the simulated time since the last frame; streaks are light on dark colourings and
// dark on the light diverging maps.
function encodeTracers(compute) {
    const { device, solver, renderer } = app, t = renderer.tracers;
    if (!t.on) return;
    const colour = el.colour.value, dark = colour === 'vorticity' || (colour === 'concentration' && !renderer.viewData.logScale);
    const params = new ArrayBuffer(64), pu = new Uint32Array(params), pf = new Float32Array(params);
    pf[0] = Math.max(0, solver.time - t.lastTime); pu[1] = ++t.frame; pu[2] = t.count; pf[3] = devicePixelRatio;
    pf.set(dark ? [0.08, 0.08, 0.1, 0.6] : [1, 1, 1, 0.65], 4);
    pf.set(t.seed, 8); pf.set(t.life, 12);
    t.lastTime = solver.time;
    device.queue.writeBuffer(t.params, 0, params);
    compute.setPipeline(t.advect); compute.setBindGroup(0, t.computeGroups[solver.ping]);
    compute.dispatchWorkgroups(Math.ceil(t.count / 64));
}
