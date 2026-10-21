// Elements of index.html used by the viewer. The native inputs in .hidden-controls hold the state that
// the text controls (ui/controls.mjs) display and change.
const $ = selector => document.querySelector(selector);

export const el = {
    canvas: $('#canvas'), work: $('.work'), hud: $('#hud'), error: $('#error'),
    select: $('#scenario'), play: $('#play'), reset: $('#reset'), speed: $('#speed'), speedValue: $('#speedValue'),
    colour: $('#colour'), lowFroude: $('#lowFroude'), closure: $('#closure'),
    tools: $('#tools'), flow: $('#flow'), flowValue: $('#flowValue'), rain: $('#rain'),
    title: $('#title'), about: $('#about'), aboutMore: $('#aboutMore'), number: $('#number'), prevDemo: $('#prevDemo'), nextDemo: $('#nextDemo'),
    examples: $('#examples'), catalog: $('#catalog'), examplesCount: $('#examplesCount'), openExamples: $('#openExamples'), closeExamples: $('#closeExamples'), controls: $('#controls'), settings: $('#settings'), toolbox: $('#toolbox'), colours: $('#colours'), params: $('#params'),
    legend: $('#legend'), legendBar: $('#legendBar'), legendLo: $('#legendLo'), legendHi: $('#legendHi'),
    mapLegend: $('#mapLegend'), mapLegendLabel: $('#mlLabel'), mapLegendBar: $('#mlBar'), mapLegendLo: $('#mlLo'), mapLegendHi: $('#mlHi'),
    windRow: $('#windRow'), windArrow: $('#windArrow'), wind: $('#rWind'),
    readout: {
        time: $('#rTime'), grid: $('#rGrid'), steps: $('#rSteps'), added: $('#rAdded'), addedRow: $('#addedRow'),
        gauge: $('#rGauge'), gaugeRow: $('#gaugeRow'), gaugeLabel: $('#gaugeLabel'), outflow: $('#rOutflow'), outflowRow: $('#outflowRow'),
    },
};
