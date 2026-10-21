# UK Environment Agency 2D benchmarks (tests 1 to 6B)

The Environment Agency benchmark tests (Néelz and Pender 2010, Science Report SC080035; Néelz and
Pender 2013, SC120002) are distributed by the UK government under the Open Government Licence
v3.0. The data are not redistributed here; the scripts below download and convert them.

```sh
mkdir -p data/ea
curl -L -o data/ea/Benchmarking_Model_Data.zip https://assets.publishing.service.gov.uk/media/6033a9bbd3bf7f72182e99ca/Benchmarking_Model_Data.zip
unzip -q data/ea/Benchmarking_Model_Data.zip -d data/ea/archive
# each inner archive goes to a folder of its own name, for example data/ea/extracted/Test1_dataset_2010/
find data/ea/archive -name '*.zip' -exec sh -c 'unzip -q "$1" -d "data/ea/extracted/$(basename "$1" .zip)"' _ {} \;
node tools/ea/prepare.mjs data/ea/extracted data/ea/cases
npm run serve   # then open http://localhost:8080/tools/ea/
```

The archive (52,456,734 bytes) holds the 2010 datasets for tests 1 to 7; test 8 (urban rainfall
and sewer surcharge) is not included. Test 7 couples a 1D river model and is not run.

Set-up choices that deviate from, or interpret, the specifications:

- Beds are area-weighted averages of the supplied DEMs over each model cell.
- Tests 1 to 4 apply their inflows on domain edges with the solver's stage and discharge
  hydrograph boundaries. Test 5 applies its inflow along an interior diagonal line as a
  distributed source on the cells the line crosses; the source rate is updated every output
  interval (60 s) to the exact interval mean of the hydrograph, so the inflow volume is exact.
- Test 5 cells outside the active-area polygon, or without DEM data, are solid.
- Tests 6A and 6B treat the 10 m DEM cells (dam walls and building) as solid. The initial water
  is level at 0.4 m and 0.02 m (6A) and 8 m and 0.4 m (6B) either side of x = 0.
- Point output reports the water level h + z and the speed |q|/h of the cell that contains the
  point, or the bed level and zero speed where h <= 1 mm. Peaks are sampled at the output interval.
