// FluxCell demos: entry point. Creates the device, solver kernels and renderer, starts a demo when one
// is chosen (setup), and runs the loop that advances the solver and draws each frame.
//   scenarios/   what each demo simulates (data only)
//   params.mjs   the sliders and their live changes
//   render/      GPU pipelines, per-demo resources, view uniform, frame passes
//   shaders/     WGSL for the display (the solver's own WGSL is in src/shaders/)
//   ui/, input.mjs, brush.mjs, camera.mjs   page controls, pointer navigation, playground tools, viewpoints
import { compileKernels, FusedMusclSolver } from '../../src/webgpu.mjs';
import { app, DEBUG, TIMER } from './app.mjs';
import { el } from './dom.mjs';
import { SCENARIOS } from './scenarios/index.mjs';
import { paramValues, renewWave } from './params.mjs';
import { createRenderer } from './render/pipelines.mjs';
import { prepareResources } from './render/resources.mjs';
import { resize } from './render/view.mjs';
import { draw } from './render/draw.mjs';
import { zoom, camera, resetCamera } from './camera.mjs';
import { applyBrush } from './brush.mjs';
import { showLegend, showElevationLegend, showWind } from './ui/legend.mjs';
import { syncControls, showParams, markCurrent, showSpeed } from './ui/controls.mjs';
import './input.mjs';

// ?demo=<name> opens a demo.
const requested = new URLSearchParams(location.search).get('demo');
el.select.value = requested && el.select.querySelector(`option[value="${CSS.escape(requested)}"]`) ? requested : 'grandisle-ida';

// The loading box over the map: a message while starting up or switching examples, hidden when ready.
const loading = text => { el.loading.hidden = !text; if (text) el.loadingText.textContent = text; };

async function main() {
    loading('Starting WebGPU…');
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is unavailable in this browser.');
    const device = app.device = await adapter.requestDevice({ requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
    window.gpuErrors = []; // the first error explains later invalid command buffers
    device.addEventListener('uncapturederror', event => {
        window.gpuErrors.push(event.error.message);
        if (window.gpuErrors.length === 1) el.error.textContent = event.error.message;
    });
    loading('Compiling the solver…');
    app.kernels = await compileKernels(device, 'muscl-fused');
    const format = navigator.gpu.getPreferredCanvasFormat(), context = el.canvas.getContext('webgpu');
    context.configure({ device, format, alphaMode: 'opaque' });
    app.renderer = await createRenderer(device, context, format);
    await setup(el.select.value);
    loop();
}

// Starts the named demo: builds its scenario with the current parameters, creates the solver, and
// prepares the display. A newer call supersedes an older one still in progress (app.generation).
async function setup(name) {
    const token = ++app.generation;
    app.switching = true; // the loop starts no new steps until the new solver is ready
    while (app.solver?.busy) await new Promise(r => setTimeout(r, 5));
    app.solver?.destroy();
    let scenario;
    loading(`Loading ${el.select.selectedOptions[0]?.textContent?.trim() || 'the example'}…`);
    try {
        scenario = await SCENARIOS[name](paramValues(name));
        if (token !== app.generation) return;
        // Optional numerics, applied to any demo on reset: the low-Froude correction and the turbulence
        // closure (Smagorinsky 0.15 and bed-generated viscosity, kappa/6).
        if (el.lowFroude.checked) scenario.config = { ...scenario.config, lowFroude: true };
        if (el.closure.checked) scenario.config = { ...scenario.config, smagorinsky: 0.15, bedTurbulence: 0.067 };
        app.solver = await FusedMusclSolver.create(app.device, scenario.config, scenario.initial, app.kernels);
    } finally { if (token === app.generation) app.switching = false; }
    if (token !== app.generation) return;
    app.scenario = scenario;
    loading(null);
    if (DEBUG) window.__demo = { solver: app.solver, scenario, camera, zoom }; // for automated checks and captures
    zoom.k = 1; zoom.cx = zoom.cy = 0.5; // each demo opens fitted
    prepareResources(scenario);
    // Variables shown: concentration only with a tracer, water level only with a still level, bed change
    // only where the bed evolves.
    const tracer = !!scenario.config.tracer;
    el.colour.querySelector('option[value="concentration"]').disabled = !tracer;
    el.colour.querySelector('option[value="elevation"]').disabled = scenario.still === undefined;
    el.colour.querySelector('option[value="bed"]').disabled = !scenario.morph;
    el.colour.value = scenario.colour ?? (tracer ? 'concentration' : 'depth');
    resetCamera();
    el.about.textContent = scenario.note ?? '';
    el.tools.hidden = !scenario.sandbox;
    showParams(name);
    if (scenario.sandbox) { el.flow.value = String(scenario.flow ?? 1.5); el.flow.dispatchEvent(new Event('input')); }
    showWind(app.renderer.viewData.wind);
    markCurrent();
    showLegend(); showSpeed(); showElevationLegend(scenario);
}

// Simulated time follows wall-clock time at a fixed playback rate, so every frame advances by a
// small, steady amount (a few solver steps) and motion stays smooth whatever the frame timing.
const nextFrame = TIMER ? () => new Promise(r => setTimeout(() => r(performance.now()), 16)) : () => new Promise(requestAnimationFrame);
async function loop() {
    let last = performance.now();
    for (;;) {
        const now = await nextFrame();
        const wall = Math.min((now - last) / 1000, 1 / 30); // no catch-up jump after a stall
        last = now;
        const { solver, renderer, scenario } = app;
        if (!solver || solver.destroyed || app.switching) continue;
        const token = app.generation, view = renderer.viewData;
        applyBrush();
        if (scenario.wave && !solver.busy) renewWave(scenario.wave);
        if (view.release && solver.time >= view.release.until && !solver.busy) {
            solver.setForcing({ source: view.release.source }); // end of a timed release
            view.release = null;
        }
        if (app.playing && wall > 0) {
            let simulated = renderer.rate * 10 ** Number(el.speed.value) * wall;
            if (view.release) simulated = Math.min(simulated, Math.max(1e-3, view.release.until - solver.time));
            const t0 = performance.now();
            let steps = 0;
            try { ({ steps } = await solver.advance(simulated, { maxDt: simulated, batch: 64 })); }
            catch (error) { el.error.textContent = `Solver stopped: ${error.message}`; setPlaying(false); }
            if (token !== app.generation) continue;
            renderer.stats.steps += steps; renderer.stats.wall += performance.now() - t0;
        }
        if (view.gauge && performance.now() - view.gauge.lastWall > 500 && !solver.busy) {
            await updateGauge(view.gauge);
            if (token !== app.generation) continue;
        }
        if (!view.gauge) el.readout.gaugeRow.hidden = el.readout.outflowRow.hidden = true;
        draw();
        showReadout(view);
    }
}

// Gauge: volume of a region (the reservoir) and the outflow from it, -dV/dt.
async function updateGauge(gauge) {
    const { solver } = app, r = el.readout;
    gauge.lastWall = performance.now();
    const state = await solver.readState();
    let volume = 0;
    for (const i of gauge.cells) volume += state[4 * i];
    volume *= gauge.area;
    const t = solver.time, before = gauge.previous;
    if (before && t > before.t) gauge.outflow = (gauge.sign ?? 1) * -(volume - before.volume) / (t - before.t);
    if (gauge.outflow > gauge.peak) gauge.peak = gauge.outflow;
    gauge.previous = { t, volume };
    r.gaugeRow.hidden = r.outflowRow.hidden = false;
    r.gaugeLabel.textContent = gauge.label;
    r.gauge.textContent = `${(volume / 1e6).toFixed(1)} million m³`;
    r.outflow.textContent = gauge.outflow === undefined ? '—' : `${Math.max(0, gauge.outflow).toFixed(0)} m³/s (peak ${gauge.peak.toFixed(0)})`;
    r.outflow.title = gauge.reference;
}

function showReadout(view) {
    const { solver, renderer } = app, r = el.readout, s = renderer.stats, c = solver.counters;
    r.time.textContent = `${solver.time.toFixed(1)} s`;
    r.grid.textContent = `${solver.config.width} × ${solver.config.height}`;
    r.addedRow.hidden = !view.sandbox;
    if (view.sandbox) r.added.textContent = `${solver.sourceVolume.toFixed(0)} m³`;
    r.steps.textContent = s.wall > 0 ? (1000 * s.steps / s.wall).toFixed(0) : '—';
    if (DEBUG) el.hud.textContent = `HLL fallbacks ${c.hlleFallbacks} · order reductions ${c.reconstructionFallbacks} · retries ${c.retries}`;
}

function setPlaying(on) { app.playing = on; el.play.textContent = on ? 'Pause' : 'Play'; syncControls(); }

const started = main().catch(error => { loading(null); el.error.textContent = error.message; el.hud.textContent = ''; });
const restart = () => started.then(() => setup(el.select.value)).catch(e => { el.error.textContent = e.message; });
for (const input of [el.select, el.lowFroude, el.closure]) input.addEventListener('change', restart);
el.reset.addEventListener('click', restart);
el.play.addEventListener('click', () => setPlaying(!app.playing));
el.speed.addEventListener('input', showSpeed);
el.colour.addEventListener('change', () => { resize(); showLegend(); });
addEventListener('resize', resize);
