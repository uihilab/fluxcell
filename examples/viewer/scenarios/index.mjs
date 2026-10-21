// All demos by name (the values of the page's #scenario options). Each builder takes the demo's
// parameter values (params.mjs) and returns a scenario:
// - config, initial: solver configuration and initial data (src/webgpu.mjs createSolver).
// - Display: depthScale (m, darkest water), wetDepth (m), rate (simulated seconds per second at x1),
//   colour (depth, speed, vorticity, concentration, elevation or bed), still and elevationScale (m,
//   water level about still water), concentration ('log' 1 % to 100 %, or 'linear' 0 to 1),
//   speedScale (m/s), vorticityScale (1/s), contour (m), note (the text above the map).
// - 3D: view3d, exaggeration, surfaceGain, camera, hideGround, buildingHeight, cylinders.
// - Behaviour: release ({ until, source }), sandbox, shape (outline field), particles, morph (bed
//   evolution), gauge (reservoir readout), wave (live-tunable incident wave), inflow.
// Units: metres and seconds; rain and sources in m/s.

import { scenarios as waves } from './waves.mjs';
import { scenarios as shocks } from './shocks.mjs';
import { scenarios as vortices } from './vortices.mjs';
import { scenarios as windsediment } from './wind-sediment.mjs';
import { scenarios as lidar } from './lidar.mjs';
import { scenarios as dams } from './dams.mjs';
import { scenarios as playground } from './playground.mjs';

export const SCENARIOS = { ...waves, ...shocks, ...vortices, ...windsediment, ...lidar, ...dams, ...playground };
