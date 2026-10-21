// Demo parameters: the sliders under Parameters. A slider with `live` changes the running simulation
// (solver.setParameters, setHydrographs or setForcing); the others change the starting geometry and
// apply when the demo is reset.
import { app } from './app.mjs';
import { sineSeries, WAVE_POINTS } from './scenarios/waves.mjs';
import { spillwayLevels } from './scenarios/shocks.mjs';
import { showWind } from './ui/legend.mjs';
import { resize } from './render/view.mjs';

const coriolisAt = lat => 2 * 7.2921e-5 * Math.sin(lat * Math.PI / 180);
const liveWave = v => retuneWave(app.scenario.wave, { T: v });
// Discharge boundaries scaled by the ratio of the new to the previous inflow speed.
const liveInflow = v => {
    const { solver, scenario } = app, r = v / scenario.inflow;
    scenario.inflow = v;
    solver.setHydrographs(solver.config.hydrographs.map(h => h.kind === 'discharge' ? { ...h, series: [[solver.time, h.values.at(-1) * r]] } : h));
};
const liveCoriolis = v => app.solver.setParameters({ coriolis: coriolisAt(v) });

export const PARAMS = {
    'grandisle-ida': [{ key: 'H', label: 'Wave height', min: 1, max: 6, step: 0.25, value: 5, unit: 'm', live: v => retuneWave(app.scenario.wave, { A: v / 2 }) },
        { key: 'T', label: 'Wave period', min: 8, max: 16, step: 1, value: 12, unit: 's', live: liveWave }],
    'kelvin-helmholtz': [{ key: 'U', label: 'Stream speed', min: 0.25, max: 2, step: 0.05, value: 1, unit: 'm/s' }],
    'vortex-street': [{ key: 'u', label: 'Inflow speed', min: 0.1, max: 1, step: 0.05, value: 0.5, unit: 'm/s', live: liveInflow }],
    dipoles: [{ key: 'U', label: 'Dipole speed', min: 0.1, max: 0.5, step: 0.05, value: 0.25, unit: 'm/s' },
        { key: 'offset', label: 'Offset', min: 0, max: 12, step: 1, value: 4, unit: 'm' }],
    'hydraulic-jump': [{ key: 'head', label: 'Head over crest', min: 0.5, max: 3, step: 0.1, value: 2, unit: 'm', live: (v, all) => setSpillway(all) },
        { key: 'tail', label: 'Tailwater', min: 70, max: 130, step: 5, value: 100, unit: '% of conjugate', live: (v, all) => setSpillway(all) }],
    'oblique-shock': [{ key: 'fr', label: 'Froude number', min: 2, max: 5, step: 0.1, value: 3, unit: '' },
        { key: 'theta', label: 'Wall angle', min: 4, max: 14, step: 1, value: 10, unit: '°' }],
    'rotating-dam-break': [{ key: 'lat', label: 'Latitude', min: 0, max: 90, step: 5, value: 45, unit: '°', live: liveCoriolis },
        { key: 'dome', label: 'Dome height', min: 0.2, max: 3, step: 0.1, value: 1, unit: 'm' }],
    'wind-lake': [{ key: 'speed', label: 'Wind speed', min: 0, max: 30, step: 1, value: 15, unit: 'm/s', live: (v, all) => setWind(all) },
        { key: 'from', label: 'Wind from', min: 0, max: 359, step: 15, value: 270, unit: '° (N = 0)', live: (v, all) => setWind(all) }],
    'mud-plume': [{ key: 'k', label: 'Settling at 1 kg/m³', min: 0, max: 3, step: 0.1, value: 1, unit: 'mm/s',
            live: v => app.solver.setParameters({ settling: v > 0 ? { velocity: v * 1e-3, exponent: 4 / 3 } : null }) },
        { key: 'Q', label: 'River discharge', min: 100, max: 1000, step: 50, value: 400, unit: 'm³/s',
            live: v => { const { solver } = app; solver.setHydrographs(solver.config.hydrographs.map(h => h.kind === 'discharge' ? { ...h, series: [[solver.time, v]] } : h)); } }],
    'double-slit': [{ key: 'T', label: 'Wave period', min: 2, max: 8, step: 0.5, value: 4, unit: 's', live: liveWave },
        { key: 'sep', label: 'Slit spacing', min: 10, max: 50, step: 2, value: 30, unit: 'm' }],
    shoal: [{ key: 'T', label: 'Wave period', min: 3, max: 12, step: 0.5, value: 6, unit: 's', live: liveWave },
        { key: 'height', label: 'Shoal height', min: 0, max: 1.8, step: 0.1, value: 1.2, unit: 'm' }],
    seiche: [{ key: 'T', label: 'Forcing period', min: 60, max: 600, step: 10, value: 290, unit: 's', live: liveWave }],
    'kelvin-wave': [{ key: 'lat', label: 'Latitude', min: 0, max: 90, step: 5, value: 45, unit: '°', live: liveCoriolis }],
    'bridge-scour': [{ key: 'U', label: 'River speed', min: 0.4, max: 1.4, step: 0.05, value: 0.9, unit: 'm/s', live: liveInflow }],
    riemann: [{ key: 'ratio', label: 'Depth ratio', min: 0.1, max: 0.9, step: 0.05, value: 0.4, unit: '' }],
    oroville: [{ key: 'release', label: 'Main spillway release', min: 0, max: 150, step: 5, value: 55, unit: '× 1,000 ft³/s',
        live: v => app.solver.setForcing({ source: app.scenario.releaseSource(v) }) }],
};

// Current values of a demo's parameters (the defaults until a slider moves).
export const paramValues = name => Object.fromEntries((PARAMS[name] ?? []).map(p => [p.key, p.current ?? p.value]));

function setWind({ speed, from }) {
    const to = (from + 180) * Math.PI / 180, wind = [speed * Math.sin(to), speed * Math.cos(to)], view = app.renderer.viewData;
    app.solver.setParameters({ wind });
    view.wind = wind; view.windSpeed = speed;
    showWind(wind); resize(); // the gust animation follows the wind
}

// Spillway: the reservoir energy level on the west edge and the tailwater on the east edge.
function setSpillway(values) {
    const { solver } = app, { E, h2 } = spillwayLevels(values);
    solver.setHydrographs(solver.config.hydrographs.map(h => ({ ...h, series: [[solver.time, h.edge === 'west' ? E : h2]] })));
}

// A demo's driven wave (scenario.wave = { index, still, A, T, t0, phase0 }): a new period or amplitude
// from the present without a jump in phase, and the series renewed before it runs out (renewWave,
// every frame). A wave still ramping in keeps ramping.
const wavePhase = (w, t) => w.phase0 + 2 * Math.PI * (t - w.t0) / w.T;
function retuneWave(w, changes) {
    const { solver } = app, t = solver.time, phase = wavePhase(w, t), started = t - w.t0 > 2 * w.T;
    Object.assign(w, changes, { t0: t, phase0: phase });
    solver.setHydrographs(solver.config.hydrographs.map((h, k) => k === w.index ? { ...h, series: sineSeries(w.still, w.A, w.T, t, phase, !started) } : h));
}
export function renewWave(w) {
    if (app.solver.time > w.t0 + (WAVE_POINTS - 64) * w.T / 16) retuneWave(w, {});
}
