// Scales shown with the map: the colour scale in the readout, the ground elevation scale drawn on the
// map, and the wind readout.
import { app } from '../app.mjs';
import { el } from '../dom.mjs';

// Scale of each variable: gradient, low end, unit, high end (the variable's name is in its selector).
const LEGENDS = {
    depth: ['linear-gradient(90deg, #52b8c7, #05265e)', () => '0', 'm', v => `${v.depthScale}`],
    bed: ['linear-gradient(90deg, #1a4db3, #f7f7f2, #8c5420)', v => `−${v.elevationScale}`, 'm', v => `+${v.elevationScale}`],
    elevation: ['linear-gradient(90deg, #04163a, #12569e, #d3ecff)', v => `−${v.elevationScale}`, 'm', v => `+${v.elevationScale}`],
    speed: ['linear-gradient(90deg, #440154, #21918c, #fde725)', () => '0', 'm/s', v => `${v.speedScale.toPrecision(2)}`],
    vorticity: ['linear-gradient(90deg, #2150a6, #f7f7f2, #b31a1f)', v => `−${v.vorticityScale}`, '1/s', v => `+${v.vorticityScale}`],
    log: ['linear-gradient(90deg, #fad940, #bf1419)', () => '1 %', '', () => '100 %'],
    linear: ['linear-gradient(90deg, #1a408c, #f7f7f2, #cc5a0d)', () => '0', '', () => '1'],
};

export function showLegend() {
    const v = app.renderer.viewData, mode = el.colour.value;
    const [gradient, lo, label, hi] = LEGENDS[mode === 'concentration' ? (v.logScale ? 'log' : 'linear') : mode];
    el.legendBar.style.background = gradient; el.legendLo.textContent = lo(v); el.legend.textContent = label; el.legendHi.textContent = hi(v);
}

// Ground elevation scale on the map: the land colours from the lowest to the highest bed (m, in the
// data's vertical datum), shown where the ground has relief and is drawn.
export function showElevationLegend(scenario) {
    const v = app.renderer.viewData, box = el.mapLegend, datum = app.solver.datum ?? 0;
    box.hidden = !(v.zMax - v.zMin > 0.05) || !!scenario.hideGround;
    if (box.hidden) return;
    const digits = v.zMax - v.zMin < 5 ? 2 : 0;
    el.mapLegendLabel.textContent = `Ground elevation (${scenario.elevationDatum ?? 'm'})`;
    el.mapLegendLo.textContent = (v.zMin + datum).toFixed(digits);
    el.mapLegendHi.textContent = (v.zMax + datum).toFixed(digits);
    // Coastal scenes draw sand up to about 2.5 m above still water before the land colours.
    const sandTop = v.relief > 0 ? Math.min(100, Math.max(0, 100 * ((v.still ?? 0) - datum + 2.5 - v.zMin) / (v.zMax - v.zMin))) : 0;
    el.mapLegendBar.style.background = sandTop > 0
        ? `linear-gradient(90deg, rgb(222, 204, 158), rgb(222, 204, 158) ${sandTop * 0.4}%, rgb(150, 160, 105) ${sandTop}%, rgb(163, 138, 105) ${Math.min(100, sandTop + 30)}%, rgb(214, 209, 199))`
        : '';
}

// Wind readout: an arrow pointing downwind (north up) and the speed with the direction it blows from.
export function showWind(wind) {
    const speed = Math.hypot(...wind);
    el.windRow.hidden = !(speed > 0.1);
    if (el.windRow.hidden) return;
    const to = Math.atan2(wind[1], wind[0]), from = (to + 3 * Math.PI) % (2 * Math.PI);
    const names = ['east', 'north-east', 'north', 'north-west', 'west', 'south-west', 'south', 'south-east'];
    el.windArrow.style.transform = `rotate(${-to}rad)`;
    el.wind.textContent = `${speed.toFixed(0)} m/s from the ${names[Math.round(from / (Math.PI / 4)) % 8]}`;
}
