// State shared by the viewer's modules: the GPU device and solver kernels, the running solver and
// its scenario, the renderer's GPU objects, and the play state. `generation` changes whenever a demo
// is (re)started, so work begun for an older demo can tell that it is stale; `switching` is true
// while a new solver is being created.
export const app = {
    device: null, kernels: null, solver: null, scenario: null, renderer: null,
    playing: true, generation: 0, switching: false,
};

// ?debug shows solver counters; ?timer drives the loop from a timer (automated captures).
const query = new URLSearchParams(location.search);
export const DEBUG = query.has('debug');
export const TIMER = query.has('timer');
