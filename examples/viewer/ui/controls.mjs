// Page controls: the examples window (numbered demos by group) and the previous and next buttons, the
// text-button rows in the right column (run, speed, view, variable, numerics, playground tools) and the
// parameter sliders. The buttons drive the native inputs in
// .hidden-controls, whose change events main.mjs listens to.
import { app } from '../app.mjs';
import { el } from '../dom.mjs';
import { PARAMS, paramValues } from '../params.mjs';
import { setZoomWindow } from '../camera.mjs';
import { resize, writeZoom } from '../render/view.mjs';
import { playground } from '../brush.mjs';

const fire = (input, type) => input.dispatchEvent(new Event(type));

// Examples window: one group per option group, the demos numbered in order.
const demos = [...el.select.options];
let number = 0;
for (const group of el.select.querySelectorAll('optgroup')) {
    const column = document.createElement('section'); column.className = 'group';
    const label = document.createElement('h2'); label.textContent = group.label;
    const note = document.createElement('p'); note.textContent = group.dataset.note ?? '';
    column.append(label, note);
    for (const option of group.querySelectorAll('option')) {
        const b = document.createElement('button'); b.dataset.value = option.value;
        b.innerHTML = `<span>${String(++number).padStart(2, '0')}</span><span>${option.textContent}</span>`;
        b.addEventListener('click', () => { el.examples.close(); choose(option.value); });
        column.append(b);
    }
    el.catalog.append(column);
}
el.examplesCount.textContent = `${demos.length} demos`;
const choose = value => { if (el.select.value !== value) { el.select.value = value; fire(el.select, 'change'); } };
const step = k => choose(demos[(el.select.selectedIndex + k + demos.length) % demos.length].value);
el.prevDemo.addEventListener('click', () => step(-1));
el.nextDemo.addEventListener('click', () => step(1));
el.openExamples.addEventListener('click', () => {
    el.examples.showModal();
    el.catalog.querySelector('button[aria-current="true"]')?.focus();
});
el.closeExamples.addEventListener('click', () => el.examples.close());
el.examples.addEventListener('click', event => { if (event.target === el.examples) el.examples.close(); }); // the backdrop

// Keys: E opens the examples, the left and right arrows step through them, space plays and pauses.
addEventListener('keydown', event => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.target.closest?.('input, select, textarea')) return;
    const key = event.key.toLowerCase();
    if (key === 'e' && !el.examples.open) { event.preventDefault(); el.openExamples.click(); }
    if (el.examples.open) return;
    if (key === 'arrowleft' || key === 'arrowright') { event.preventDefault(); step(key === 'arrowleft' ? -1 : 1); }
    if (key === ' ' && !event.target.closest?.('button, a')) { event.preventDefault(); el.play.click(); }
});

// A labelled row of text buttons; items are { text, pressed(), disabled(), live(), click }.
function row(label, items, parent = el.controls) {
    const r = document.createElement('div'); r.className = 'row';
    const l = document.createElement('span'); l.className = 'label'; l.textContent = label;
    const list = document.createElement('span'); list.className = 'items'; r.append(l, list);
    for (const item of items) {
        const b = document.createElement('button'); b.textContent = item.text;
        // A mouse click leaves no focus behind, so the space bar keeps pausing (keys, below).
        b.addEventListener('click', event => { item.click(); syncControls(); if (event.detail > 0) b.blur(); });
        item.button = b; list.append(b);
    }
    parent.append(r);
    return { items, element: r };
}

export function setZoom(k, cx, cy) { setZoomWindow(k, cx, cy); writeZoom(); syncControls(); }
export function set3d(on) { app.renderer.mode3d = on; resize(); syncControls(); }
const is3d = () => !!app.renderer?.mode3d;
// The map zooms with the wheel and fits on a double click (input.mjs), so there are no zoom buttons.

const SPEEDS = [0.25, 0.5, 1, 2, 4, 8];
const rows = [
    row('Run', [
        { text: 'Pause', pressed: () => false, click: () => el.play.click(), live: () => el.play.textContent },
        { text: 'Reset', pressed: () => false, click: () => el.reset.click() }]),
    row('Speed', SPEEDS.map(k => ({ text: `×${k}`, pressed: () => Math.abs(Number(el.speed.value) - Math.log10(k)) < 0.03,
        click: () => { el.speed.value = String(Math.log10(k)); fire(el.speed, 'input'); } }))),
    row('View', [
        { text: '2D', pressed: () => !is3d(), click: () => set3d(false) },
        { text: '3D', pressed: is3d, click: () => set3d(true) },
        { text: 'Particles', pressed: () => !!app.renderer?.tracers?.on, click: () => { app.renderer.tracers.on = !app.renderer.tracers.on; } }]),
    row('Variable', [...el.colour.options].map(o => ({ text: o.textContent, pressed: () => el.colour.value === o.value, disabled: () => o.disabled,
        click: () => { el.colour.value = o.value; fire(el.colour, 'change'); } })), el.colours),
    row('Numerics', [['Low-Froude', el.lowFroude], ['Closure', el.closure]].map(([text, box]) => ({ text, pressed: () => box.checked,
        click: () => { box.checked = !box.checked; fire(box, 'change'); } })), el.settings),
];
// What the numerics options do, shown on a short hover (index.html [data-tip]); docs/METHOD.md has more.
const NUMERICS = [
    'Less numerical damping in slow flow, so eddies and weak currents last longer. Restarts the example.',
    'Adds turbulent eddy viscosity (Smagorinsky and bed friction), which damps the smallest eddies. Restarts the example.',
];
rows.at(-1).items.forEach((item, k) => { item.button.dataset.tip = NUMERICS[k]; });
// The variable's colour scale under its choices.
el.colours.append(el.colours.querySelector('.scale'));
// Playground tools, shown on the playground demos only.
const toolRows = [
    row('Tool', [
        ...[['water', 'Water'], ['dye', 'Dye'], ['raise', 'Raise'], ['dig', 'Dig']].map(([tool, text]) => ({ text,
            pressed: () => playground.tool === tool, click: () => { playground.tool = tool; } })),
        { text: 'Rain', pressed: () => el.rain.checked, click: () => { el.rain.checked = !el.rain.checked; } }], el.toolbox),
    row('Size', [['S', 0.5], ['M', 1], ['L', 2]].map(([text, k]) => ({ text, pressed: () => playground.size === k,
        click: () => { playground.size = k; } })), el.toolbox),
    row('Flow', [
        { text: '−', pressed: () => false, click: () => { el.flow.value = String(Math.max(0, Number(el.flow.value) - 0.25)); fire(el.flow, 'input'); } },
        { text: '', pressed: () => true, live: () => `${(10 ** Number(el.flow.value)).toPrecision(2)} m³/s`, click: () => {} },
        { text: '+', pressed: () => false, click: () => { el.flow.value = String(Math.min(3, Number(el.flow.value) + 0.25)); fire(el.flow, 'input'); } }], el.toolbox),
];

export function syncControls() {
    for (const { items } of [...rows, ...toolRows]) for (const t of items) {
        t.button.setAttribute('aria-pressed', String(!!t.pressed()));
        if (t.disabled) t.button.toggleAttribute('disabled', !!t.disabled());
        if (t.live) t.button.textContent = t.live();
    }
    el.toolbox.hidden = el.tools.hidden;
    // On the playgrounds the left button uses the tool (crosshair) and the right button pans.
    el.canvas.style.cursor = el.tools.hidden ? '' : 'crosshair';
    for (const b of el.catalog.querySelectorAll('button')) b.setAttribute('aria-current', String(b.dataset.value === el.select.value));
}

// Sliders of the current demo. Live sliders act at once; the others are marked to apply on reset.
export function showParams(name) {
    const box = el.params;
    box.replaceChildren();
    const list = PARAMS[name] ?? [];
    box.hidden = !list.length;
    if (!list.length) return;
    const head = document.createElement('div'); head.className = 'label'; head.textContent = 'Parameters'; box.append(head);
    for (const p of list) {
        const r = document.createElement('label'); r.className = 'param';
        const label = document.createElement('span'), value = document.createElement('span'), input = document.createElement('input');
        const note = document.createElement('span'); note.className = 'pending'; note.textContent = 'Applies on reset'; note.hidden = true;
        label.textContent = p.label; value.className = 'value';
        const show = v => { value.textContent = `${Number(v).toLocaleString('en-US', { maximumFractionDigits: 2 })}${p.unit ? ` ${p.unit}` : ''}`; };
        Object.assign(input, { type: 'range', min: p.min, max: p.max, step: p.step, value: p.current ?? p.value });
        show(input.value);
        input.addEventListener('input', () => show(input.value));
        if (p.live) input.addEventListener('input', () => {
            p.current = Number(input.value);
            try { if (app.solver && !app.solver.busy && !app.switching) p.live(p.current, paramValues(el.select.value)); }
            catch (e) { el.error.textContent = e.message; }
        });
        else {
            input.addEventListener('change', () => { p.current = Number(input.value); note.hidden = false; });
            label.title = 'Applies on reset';
        }
        r.append(label, value, input, note); box.append(r);
    }
}

// On phones the description shows three lines; More opens the rest (shown only when there is more).
export function fitAbout() {
    el.about.classList.remove('open'); el.aboutMore.textContent = 'More';
    el.aboutMore.hidden = !(el.about.scrollHeight > el.about.clientHeight + 1);
}
el.aboutMore.addEventListener('click', () => {
    const open = el.about.classList.toggle('open');
    el.aboutMore.textContent = open ? 'Less' : 'More';
});
addEventListener('resize', fitAbout);

// The current demo's number and title, the controls and the address bar (?demo=<name>, so a demo can
// be linked).
export function markCurrent() {
    fitAbout();
    el.number.textContent = `${String(el.select.selectedIndex + 1).padStart(2, '0')} / ${demos.length}`;
    el.title.textContent = el.select.selectedOptions[0]?.textContent ?? '';
    syncControls(); resize();
    const url = new URL(location.href); url.searchParams.set('demo', el.select.value); history.replaceState(null, '', url);
}

// Playback rate: simulated seconds per second, as a multiple of real time, two significant digits
// (720×, not 7.2e+2).
export function showSpeed() {
    const rate = app.renderer.rate * 10 ** Number(el.speed.value);
    el.speedValue.textContent = `${Number(rate.toPrecision(2)).toLocaleString('en-US')}×`;
}

// ./#examples (the Examples link of the other pages) opens the examples; ./#references, an older
// link, goes to the references page.
if (location.hash === '#references') location.replace('./references.html');
if (location.hash === '#examples') { history.replaceState(null, '', location.pathname + location.search); el.examples.showModal(); }
