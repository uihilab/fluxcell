// Flooding on 2 m lidar (USGS 3DEP): Yosemite and the 17th Street Canal breach in New Orleans.
import { loadDem, terrainScenario } from '../../../src/terrain.mjs';

// Each entry returns { config, initial, ... } (fields in scenarios/index.mjs).
export const scenarios = {
    // 1.3 km (New Orleans: the breach and Lakeview) windows of the fetched tiles.
    async 'yosemite2m-release'() {
        return { ...terrainScenario(await loadDem('yosemite-elcapitan-2m', { crop: [404, 700, 768, 768] }), 'release', { tracer: true }),
            depthScale: 3, wetDepth: 0.02, rate: 20, colour: 'depth', concentration: 'log', view3d: true, exaggeration: 1, elevationDatum: 'm NAVD88',
            note: 'Lidar 2 m (CA Yosemite 2019 / Southern Sierra 2020), 768 × 768 cells. A 5 m column above the face of El Capitan; its water is traced.' };
    },
    async 'yosemite2m-rain'() {
        return { ...terrainScenario(await loadDem('yosemite-elcapitan-2m', { crop: [404, 700, 768, 768] }), 'rain'),
            depthScale: 0.3, wetDepth: 0.003, rate: 60, elevationDatum: 'm NAVD88', note: 'Lidar 2 m, 768 × 768 cells. 100 mm/h of rain on a dry tile.' };
    },
    async 'neworleans2m-breach'() {
        return { ...terrainScenario(await loadDem('neworleans-17thstreet-2m', { crop: [640, 1084, 640, 640] }), 'breach', { tracer: true }),
            depthScale: 2, wetDepth: 0.02, rate: 60, colour: 'depth', concentration: 'log', elevationDatum: 'm NAVD88',
            note: 'Lidar 2 m (LA Greater New Orleans 2021), 640 × 640 cells. Illustrative 600 m³/s through a 140 m gap at the 2005 17th Street Canal breach; the floodwall, too thin for bare-earth lidar, is restored as solid cells; lake water is traced.' };
    },
};
