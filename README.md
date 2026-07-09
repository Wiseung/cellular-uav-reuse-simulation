# Cellular + UAV Reuse Simulation

This workspace implements the course-design simulation for:

`蜂窝 + 低空融合网络频率复用与 SIR 信干比仿真`

The code follows the execution plan in `执行计划.md` and now uses a more engineering-oriented model:

- Analytic reuse baseline: `SIR = (3N)^(n/2) / 6`
- Explicit `SIR` and `SINR` computation with thermal noise, bandwidth, and receiver noise figure
- Perturbed hexagonal co-channel deployment instead of a perfectly regular lattice
- Monte Carlo user drops inside the serving hexagon
- Custom antenna-pattern file support (`.csv`, `.msi`, `.pln`), three-sector base-station panels, and optional beam codebook steering
- Lognormal shadowing and small-scale fading on every link
- External beam/load/handover profile JSON support plus a calibration script for deriving those settings from site-layout and dynamic-trace data
- A hybrid UAV LOS model:
  - 3GPP TR 38.901 UMa LOS probability for near-ground user heights
  - ITU-R P.1410 statistical building-blockage LOS probability for higher aerial links
  - LOS/NLOS pathloss exponents, shadowing, and fading sampled per link during the height experiment
  - Material-aware excess loss that adjusts the GIS penetration branch when facade or roof materials are available
- Effective ASE with traffic activity, scheduling efficiency, control overhead, and outage gating
- Dynamic trajectory experiment with external site-layout CSV input, L3-filtered handover events with time-to-trigger, correlated load, dominant-interferer coordinated scheduling, per-user power splitting, and optional GIS-driven LOS/loss overrides from building footprints
- Public-data preparation pipelines for OpenCelliD/CellMapper enhanced site layouts and Overture + USGS 3DEP building-height enrichment
- Site-specific static and dynamic overrides for sector azimuth, downtilt, transmit height, transmit power, and same-frequency (`radio`/`ARFCN`) interference filtering when those fields are present in the site-layout CSV
- One-command public-data loop that produces enhanced inputs, calibrated profiles, baseline/enhanced simulation outputs, and a before/after gain report

## How to run

From the workspace root:

```powershell
pip install -r requirements.txt
```

Then run:

```powershell
python -m cellular_uav_sir.main
```

Or override the runtime profile and inputs explicitly:

```powershell
python -m cellular_uav_sir.main `
  --profile-json cellular_uav_sir/data/default_parameter_profile.json `
  --site-layout-csv cellular_uav_sir/data/real_site_layout_knoxville_tn.csv `
  --building-geojson cellular_uav_sir/data/knoxville_site_layout_buildings.geojson
```

The script generates CSV tables and PNG figures under:

`cellular_uav_sir/results/`

## Output files

- `table_1_sir_vs_reuse.csv`
- `table_2_ase_vs_reuse.csv`
- `table_3_sir_vs_height.csv`
- `table_4_sir_cdf_samples.csv`
- `table_5_pathloss_sweep.csv`
- `table_6_dynamic_summary.csv`
- `table_7_dynamic_trace.csv`
- `table_8_dynamic_site_layout.csv`
- `figure_1_reuse_geometry.png`
- `figure_2_sir_vs_reuse.png`
- `figure_3_ase_vs_reuse.png`
- `figure_4_sir_vs_height.png`
- `figure_5_sir_cdf.png`
- `figure_6_pathloss_sweep.png`
- `figure_7_los_probability_vs_height.png`
- `figure_8_dynamic_sinr_timeline.png`
- `figure_9_dynamic_layout_map.png`

## Repository contents

- `cellular_uav_sir/`: simulation source code
- `cellular_uav_sir/data/`: antenna-pattern, site-layout, and building-footprint inputs, including a real Knoxville cell-tower cluster derived from a public ArcGIS Hub dataset
- `cellular_uav_sir/results/`: generated CSV tables and PNG figures
- `tools/`: public-data preparation and calibration scripts for ArcGIS/OpenCelliD/CellMapper/Osm/Overture/3DEP inputs
- `完整中文报告初稿.md`: full Chinese report
- `答辩讲稿.md`: oral defense script
- `PPT汇报提纲.md`: slide-ready page copy

## Main assumptions

- Downlink only
- Equal transmit power for all base stations
- Thermal noise is explicitly included through SINR
- Ground users use multiple co-channel tiers plus random site perturbation
- UAV users use a hybrid 3GPP/ITU LOS probability model above the ground-user regime
- Real site layouts can be injected via CSV for dynamic experiments
- Building-footprint GeoJSON can override dynamic LOS/NLOS states and add deterministic obstruction loss for links covered by the local GIS sample
- ASE is reported as effective ASE after applying activity factor, scheduler efficiency, control overhead, and a coverage threshold
- Dynamic experiments add UAV trajectory, filtered handover events, correlated cell load, dominant-interferer coordinated scheduling, load-aware power sharing, and GIS obstruction loss

## Real layout note

The default dynamic layout now uses `cellular_uav_sir/data/real_site_layout_knoxville_tn.csv`, which was derived from the public ArcGIS Hub dataset `Cellular Towers in the United States`. The processing note is recorded in `cellular_uav_sir/data/real_site_layout_knoxville_tn_source.md`.

The default dynamic GIS sample now also includes `cellular_uav_sir/data/knoxville_site_layout_buildings.geojson`, which was downloaded from OpenStreetMap building footprints through the Overpass API across the current Knoxville site-cluster bounding box using tiled requests. The processing note is recorded in `cellular_uav_sir/data/knoxville_site_layout_buildings_source.md`.

The default antenna-pattern file now uses `cellular_uav_sir/data/reference_sector_panel_pattern.msi`, a vendor-style MSI Planet text file derived from the legacy reference cuts. The source note is recorded in `cellular_uav_sir/data/reference_sector_panel_pattern_source.md`.

The default runtime profile now uses `cellular_uav_sir/data/default_parameter_profile.json`, which externalizes the beam, load, handover, coordination, and path-input defaults that were previously hard-coded in `SimulationConfig`.

The default material-aware GIS loss profile now uses `cellular_uav_sir/data/building_material_loss_profile.json`, which externalizes the facade/roof material to penetration-loss mapping instead of hard-coding it in `building_gis.py`.

For static and dynamic real-site experiments, the simulator now also reads optional per-site override columns from the site-layout CSV when present:

- `sector_azimuths_deg` or `antenna_azimuth_deg`
- `mechanical_downtilt_deg`
- `electrical_downtilt_deg`
- `tx_height_m` or `antenna_height_m`
- `tx_power_dbm`
- `antenna_peak_gain_db`
- `radio`
- `arfcn_list` or `arfcn`

If both the serving site and a neighbor expose `radio` and `ARFCN`, only overlapping same-frequency neighbors are counted as interferers in both the static site-layout path and the dynamic downlink SINR calculation.

When building features expose `facade_material` or `roof_material`, the GIS excess-loss model now scales the penetration component by material class and adds a small material entry loss. Diffraction remains geometry-driven.

## Public data preparation

Rebuild a local real-site cluster from the ArcGIS public tower dataset:

```powershell
python tools/extract_arcgis_site_cluster.py `
  --center-lat 35.9606388889 `
  --center-lon -83.9846388889 `
  --count 21 `
  --output cellular_uav_sir/data/real_site_layout_knoxville_tn.csv
```

Download a site-cluster building-footprint GeoJSON sample:

```powershell
python tools/download_osm_buildings.py `
  --site-layout-csv cellular_uav_sir/data/real_site_layout_knoxville_tn.csv `
  --margin-deg 0.003 `
  --tile-size-deg 0.04 `
  --tile-overlap-deg 0.001 `
  --output cellular_uav_sir/data/knoxville_site_layout_buildings.geojson
```

Normalize a public OpenCelliD or CellMapper export into the enhanced site-layout CSV used by the simulator:

```powershell
python tools/prepare_enhanced_site_layout.py `
  --source opencellid `
  --input-csv path\to\public_cells.csv `
  --center-lat 35.9606388889 `
  --center-lon -83.9846388889 `
  --radius-km 4 `
  --limit-sites 21 `
  --output cellular_uav_sir/data/enhanced_public_site_layout.csv
```

Normalize an Overture building export and enrich it with USGS 3DEP ground elevation:

```powershell
python tools/prepare_overture_3dep_buildings.py `
  --overture-geojson path\to\overture_buildings.geojson `
  --site-layout-csv cellular_uav_sir/data/enhanced_public_site_layout.csv `
  --output cellular_uav_sir/data/enhanced_public_buildings.geojson
```

Calibrate the external beam/load/handover profile from the enhanced site layout and a dynamic trace:

```powershell
python tools/calibrate_parameter_profile.py `
  --site-layout-csv cellular_uav_sir/data/enhanced_public_site_layout.csv `
  --dynamic-trace-csv cellular_uav_sir/results/table_7_dynamic_trace.csv `
  --output cellular_uav_sir/data/calibrated_parameter_profile.json
```

Run the full public-data loop and write a before/after gain report:

```powershell
python tools/run_public_data_pipeline.py `
  --source opencellid `
  --raw-source-csv path\to\public_cells.csv `
  --overture-geojson path\to\overture_buildings.geojson `
  --output-root deliverables\public_data_pipeline_run `
  --center-lat 35.9606388889 `
  --center-lon -83.9846388889 `
  --radius-km 4 `
  --limit-sites 21 `
  --skip-ground-elevation
```

This one command generates:

- `inputs/enhanced_site_layout.csv`
- `inputs/enhanced_buildings.geojson`
- `profiles/initial_site_profile.json`
- `profiles/calibrated_profile.json`
- `results/baseline/`
- `results/enhanced/`
- `report/public_data_gain_report.md`
- `report/comparison_metrics.csv`
